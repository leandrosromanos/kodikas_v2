"""
adaptive_pipeline_v2.py
──────────────────────
Self-guided adaptive detection pipeline v2.

Βελτιώσεις:
  - Crop score αντί για σκληρό AND rule
  - Perspective-aware weighting
  - Καλύτερο skip guard
  - Top-K weak detections για crop zones
  - Smart merge: crop boxes μπορούν να αντικαταστήσουν full-frame boxes
  - Σωστό evaluation πάνω στα filtered detections
  - SegFormer input tensors μεταφέρονται σωστά σε GPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO


# ── config ────────────────────────────────────────────────────────────────────

DATAROOT    = r"/Users/evry/A.LE.KOS/student_dataset"
OUT_DIR     = Path(__file__).parent / "adaptive_viz_v2"
OUTPUT_JSON = Path(__file__).parent / "submission_adaptive_v2.json"

IMG_W, IMG_H = 1600, 900

YOLO_TO_NS = {
    2: "vehicle.car",
    5: "vehicle.bus.rigid",
    7: "vehicle.truck",
}

VALID_CATS = {
    "vehicle.car",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.truck",
}


# ── Pass 1 thresholds ─────────────────────────────────────────────────────────

FULL_CONF  = 0.20
FULL_IMGSZ = 1280


# ── FP reduction filters ──────────────────────────────────────────────────────

MIN_BOX_W  = 30
MIN_ASPECT = 0.30


# ── Adaptive crop scoring ─────────────────────────────────────────────────────

SMALL_AREA_THR = 0.0008
LOW_CONF_THR   = 0.30
BOX_W_THR      = 35

CROP_SCORE_THR = 0.45

MAX_ZONES         = 2
MAX_WEAK_PER_ZONE = 4


# ── Crop / tile parameters ────────────────────────────────────────────────────

CROP_CONF   = 0.15
CROP_IMGSZ  = 640
TILE_W      = 384
TILE_OVR    = 64
ZONE_PAD    = 60

NMS_IOU_THR = 0.50
NEW_DET_THR = 0.35


# ── Segmentation fallback ─────────────────────────────────────────────────────

SEG_WIDTH    = 512
ROAD_LABEL   = 0
FAR_ROAD_PCT = 15


# ── Scene-level crop guard ─────────────────────────────────────────────────────

SKIP_STRONG_THR = 5


# ══ Step 1 — Full-frame YOLO pass ═════════════════════════════════════════════

def run_full_pass(yolo, image: Image.Image):
    t0 = time.perf_counter()

    results = yolo.predict(
        image,
        imgsz=FULL_IMGSZ,
        conf=FULL_CONF,
        iou=0.5,
        verbose=False,
    )

    dt = time.perf_counter() - t0
    dets = []

    for r in results:
        for d in r.boxes:
            cid = int(d.cls.item())

            if cid not in YOLO_TO_NS:
                continue

            x1, y1, x2, y2 = [float(v) for v in d.xyxy[0].tolist()]

            dets.append({
                "box": [x1, y1, x2, y2],
                "cid": cid,
                "conf": float(d.conf.item()),
                "source": "full",
            })

    return dets, dt


# ══ Step 2 — Adaptive prefiltering / analysis ═════════════════════════════════

def perspective_weight(box, fh=IMG_H):
    """
    Δίνει μεγαλύτερο βάρος σε boxes που βρίσκονται πιο ψηλά στην εικόνα.
    Σε road scenes, ψηλότερα συνήθως σημαίνει πιο μακριά.
    """
    _, y1, _, y2 = box
    cy = (y1 + y2) / 2.0

    return 1.0 + 0.7 * (1.0 - cy / fh)


def crop_need_score(det: dict, fw=IMG_W, fh=IMG_H) -> float:
    """
    Soft score για το αν ένα detection χρειάζεται crop.

    Αντί για:
        small AND low-confidence

    χρησιμοποιούμε weighted score:
        width-risk + area-risk + confidence-risk

    και το ενισχύουμε με perspective weight.
    """
    x1, y1, x2, y2 = det["box"]

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    rel_area = (w * h) / (fw * fh)
    conf = det["conf"]

    size_score = max(0.0, 1.0 - w / BOX_W_THR)
    area_score = max(0.0, 1.0 - rel_area / SMALL_AREA_THR)
    conf_score = max(0.0, 1.0 - conf / LOW_CONF_THR)

    base = (
        0.45 * size_score +
        0.30 * area_score +
        0.25 * conf_score
    )

    return base * perspective_weight(det["box"], fh)


def should_crop(det: dict, fw=IMG_W, fh=IMG_H) -> bool:
    return crop_need_score(det, fw, fh) > CROP_SCORE_THR


def analyse(dets: list[dict], fw=IMG_W, fh=IMG_H) -> dict:
    """
    Προσθέτει σε κάθε detection:
      - rel_area
      - crop_score
      - needs_crop
    """
    for d in dets:
        x1, y1, x2, y2 = d["box"]

        d["rel_area"] = ((x2 - x1) * (y2 - y1)) / (fw * fh)
        d["crop_score"] = crop_need_score(d, fw, fh)
        d["needs_crop"] = d["crop_score"] > CROP_SCORE_THR

    n_weak = sum(1 for d in dets if d["needs_crop"])
    n_strong = len(dets) - n_weak

    return {
        "n_weak": n_weak,
        "n_strong": n_strong,
        "total": len(dets),
    }


# ══ Step 3 — Select adaptive crop zones ═══════════════════════════════════════

def _cluster_zone(boxes, frame_w, frame_h, pad=ZONE_PAD):
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]

    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(frame_w, max(xs) + pad)
    y2 = min(frame_h, max(ys) + pad)

    return int(x1), int(y1), int(x2), int(y2)


def _box_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _road_far_zone(seg_model, image: Image.Image):
    """
    Fallback μόνο όταν YOLO δεν βρήκε τίποτα.

    Χρησιμοποιεί SegFormer για να βρει δρόμο και κρατά το πάνω/far κομμάτι.
    """
    ow, oh = image.size

    small_h = int(oh * SEG_WIDTH / ow)
    small = image.resize((SEG_WIDTH, small_h), Image.BILINEAR)

    inp = {
        k: v.to(seg_model["device"])
        for k, v in seg_model["proc"](
            images=small,
            return_tensors="pt"
        ).items()
    }

    with torch.no_grad():
        logits = seg_model["model"](**inp).logits

    up = F.interpolate(
        logits,
        size=(oh, ow),
        mode="bilinear",
        align_corners=False,
    )

    seg = up.argmax(1).squeeze(0).cpu().numpy()
    mask = (seg == ROAD_LABEL).astype(np.uint8)

    ys, xs = np.where(mask)

    if ys.size == 0:
        return None

    y_cut = int(np.percentile(ys, FAR_ROAD_PCT))
    sel = ys <= y_cut

    fy, fx = ys[sel], xs[sel]

    if fy.size < 20:
        return None

    pad = ZONE_PAD

    x1 = max(0, int(fx.min()) - pad)
    y1 = max(0, int(fy.min()) - pad)
    x2 = min(ow, int(fx.max()) + pad)
    y2 = min(oh, int(fy.max()) + pad)

    if (x2 - x1) <= 60 or (y2 - y1) <= 20:
        return None

    return x1, y1, x2, y2


def _split_weak_into_zones(weak: list[dict], fw: int, fh: int):
    """
    Παίρνει τα πιο σημαντικά weak detections και τα χωρίζει σε λίγες ζώνες.

    Απλή στρατηγική:
      - sort με crop_score
      - κρατά top-K
      - τα χωρίζει με βάση x-center σε MAX_ZONES groups
    """
    if not weak:
        return []

    weak = sorted(
        weak,
        key=lambda d: d.get("crop_score", 0.0),
        reverse=True,
    )

    weak = weak[:MAX_ZONES * MAX_WEAK_PER_ZONE]

    if len(weak) <= MAX_WEAK_PER_ZONE or MAX_ZONES == 1:
        roi = _cluster_zone([d["box"] for d in weak], fw, fh)

        return [{
            "roi": roi,
            "reason": f"{len(weak)} high-risk detections",
            "trigger_boxes": [d["box"] for d in weak],
        }]

    weak = sorted(weak, key=lambda d: _box_center(d["box"])[0])

    groups = []
    chunk_size = int(np.ceil(len(weak) / MAX_ZONES))

    for i in range(0, len(weak), chunk_size):
        group = weak[i:i + chunk_size]

        if not group:
            continue

        roi = _cluster_zone([d["box"] for d in group], fw, fh)

        groups.append({
            "roi": roi,
            "reason": f"{len(group)} high-risk detections",
            "trigger_boxes": [d["box"] for d in group],
        })

    return groups


def select_zones(
    dets: list[dict],
    image: Image.Image,
    seg_model=None,
) -> list[dict]:
    """
    Αποφασίζει πού θα γίνει δεύτερο YOLO pass.

    Κύρια λογική:
      - crop σε high-risk detections
      - segmentation fallback μόνο αν δεν υπάρχουν καθόλου detections
      - skip μόνο αν υπάρχουν αρκετά strong ΚΑΙ κανένα weak
    """
    fw, fh = image.size

    weak = [d for d in dets if d.get("needs_crop")]
    strong = [d for d in dets if not d.get("needs_crop")]

    # Καλύτερο skip guard:
    # όχι crop μόνο αν η Pass 1 εικόνα φαίνεται αρκετά πλήρης.
    if len(strong) >= SKIP_STRONG_THR and len(weak) == 0:
        return []

    zones = []

    if weak:
        zones.extend(_split_weak_into_zones(weak, fw, fh))

    # fallback μόνο όταν δεν βρέθηκε τίποτα
    if not dets and seg_model is not None:
        roi = _road_far_zone(seg_model, image)

        if roi:
            zones.append({
                "roi": roi,
                "reason": "no detections: segmentation far-road fallback",
                "trigger_boxes": [],
            })

    return _merge_zones(zones, fw, fh)


def _zone_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 1e-6 else 0.0


def _merge_zones(zones: list[dict], fw: int, fh: int):
    """
    Συγχωνεύει μόνο έντονα overlapping zones.
    Δεν τα κάνει πάντα όλα ένα μεγάλο zone.
    """
    if len(zones) <= 1:
        return zones

    merged = []

    for z in zones:
        placed = False

        for m in merged:
            if _zone_iou(z["roi"], m["roi"]) > 0.25:
                r1 = m["roi"]
                r2 = z["roi"]

                x1 = max(0, min(r1[0], r2[0]))
                y1 = max(0, min(r1[1], r2[1]))
                x2 = min(fw, max(r1[2], r2[2]))
                y2 = min(fh, max(r1[3], r2[3]))

                m["roi"] = (x1, y1, x2, y2)
                m["reason"] += " + " + z["reason"]
                m["trigger_boxes"].extend(z["trigger_boxes"])

                placed = True
                break

        if not placed:
            merged.append(z)

    return merged


# ══ Step 4 — Crop / tiled second pass ════════════════════════════════════════

def run_crop_pass(yolo, image: Image.Image, zones: list[dict]):
    """
    Τρέχει YOLO σε κάθε crop zone.
    Αν το zone είναι πολύ φαρδύ, το σπάει σε overlapping tiles.
    """
    all_dets = []
    n_tiles = 0

    for zone in zones:
        x1, y1, x2, y2 = zone["roi"]
        rw = x2 - x1

        if rw <= TILE_W:
            crop = image.crop((x1, y1, x2, y2))
            raw = _yolo_crop(yolo, crop)
            all_dets.extend(_shift(raw, x1, y1))
            n_tiles += 1

        else:
            step = TILE_W - TILE_OVR
            tx = x1

            while tx < x2:
                tx2 = min(tx + TILE_W, x2)

                crop = image.crop((tx, y1, tx2, y2))
                raw = _yolo_crop(yolo, crop)

                all_dets.extend(_shift(raw, tx, y1))
                n_tiles += 1

                if tx2 >= x2:
                    break

                tx += step

    return all_dets, n_tiles


def _yolo_crop(yolo, crop: Image.Image):
    results = yolo.predict(
        crop,
        imgsz=CROP_IMGSZ,
        conf=CROP_CONF,
        iou=0.5,
        verbose=False,
    )

    dets = []

    for r in results:
        for d in r.boxes:
            cid = int(d.cls.item())

            if cid not in YOLO_TO_NS:
                continue

            x1, y1, x2, y2 = [float(v) for v in d.xyxy[0].tolist()]

            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue

            dets.append({
                "box": [x1, y1, x2, y2],
                "cid": cid,
                "conf": float(d.conf.item()),
                "source": "crop",
            })

    return dets


def _shift(dets: list[dict], ox: float, oy: float):
    shifted = []

    for d in dets:
        shifted.append({
            "box": [
                d["box"][0] + ox,
                d["box"][1] + oy,
                d["box"][2] + ox,
                d["box"][3] + oy,
            ],
            "cid": d["cid"],
            "conf": d["conf"],
            "source": d.get("source", "crop"),
        })

    return shifted
# ══ Step 5 — Merge / NMS / filtering ═════════════════════════════════════════

def _iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union > 1e-6 else 0.0


def _nms(dets: list[dict], thr=NMS_IOU_THR):
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept = []

    for d in dets:
        if all(_iou(d["box"], k["box"]) < thr for k in kept):
            kept.append(d)

    return kept


def merge(full_dets: list[dict], crop_dets: list[dict]):
    """
    Smart merge.

    Παλιά:
      - full-frame boxes ήταν πάντα authoritative
      - crop boxes προστίθεντο μόνο αν ήταν νέα

    Τώρα:
      - αν crop box δεν επικαλύπτεται, προστίθεται
      - αν επικαλύπτεται αλλά έχει αρκετά καλύτερο confidence,
        αντικαθιστά το full-frame box
    """
    kept = list(full_dets)

    crop_dets = sorted(crop_dets, key=lambda d: d["conf"], reverse=True)

    for d in crop_dets:
        overlaps = [
            k for k in kept
            if _iou(d["box"], k["box"]) >= NEW_DET_THR
        ]

        if not overlaps:
            kept.append(d)
            continue

        best = max(overlaps, key=lambda x: x["conf"])

        if d["conf"] > best["conf"] + 0.05:
            kept.remove(best)
            kept.append(d)

    return _nms(kept, NMS_IOU_THR)


def _clip(box):
    x1, y1, x2, y2 = box

    return [
        max(0.0, min(x1, IMG_W)),
        max(0.0, min(y1, IMG_H)),
        max(0.0, min(x2, IMG_W)),
        max(0.0, min(y2, IMG_H)),
    ]


def filter_dets_for_eval(dets: list[dict]):
    """
    Τα ίδια φίλτρα που χρησιμοποιείς και στο submission.
    Έτσι το evaluation μετράει αυτό που πραγματικά βγάζεις.
    """
    out = []

    for d in dets:
        box = _clip(d["box"])

        bw = box[2] - box[0]
        bh = box[3] - box[1]

        if bw < MIN_BOX_W:
            continue

        if bh > 0 and (bw / bh) < MIN_ASPECT:
            continue

        out.append({
            "box": box,
            "cid": d["cid"],
            "conf": d["conf"],
            "source": d.get("source", "unknown"),
        })

    return out


def dets_to_submission(dets: list[dict], cam_tok: str):
    preds = []

    for d in filter_dets_for_eval(dets):
        cat = YOLO_TO_NS.get(d["cid"])

        if not cat:
            continue

        preds.append({
            "camera": "CAM_FRONT",
            "category": cat,
            "bbox_2d": d["box"],
            "sample_data_token": cam_tok,
        })

    return preds


# ══ Step 6 — Ground truth / evaluation ═══════════════════════════════════════

def load_gt(nusc):
    gt = defaultdict(list)

    for sample in nusc.sample:
        ctok = sample["data"]["CAM_FRONT"]
        sd = nusc.get("sample_data", ctok)

        K = np.array(
            nusc.get(
                "calibrated_sensor",
                sd["calibrated_sensor_token"],
            )["camera_intrinsic"]
        )

        img = Image.open(nusc.get_sample_data_path(ctok))
        W, H = img.size

        _, boxes3d, _ = nusc.get_sample_data(ctok)

        for box in boxes3d:
            if box.name not in VALID_CATS:
                continue

            ann = nusc.get("sample_annotation", box.token)

            if int(ann.get("visibility_token", "0")) <= 1:
                continue

            c2d = view_points(box.corners(), K, normalize=True)[:2, :]

            x1, y1 = np.clip(c2d.min(1), [0, 0], [W, H])
            x2, y2 = np.clip(c2d.max(1), [0, 0], [W, H])

            if x2 > x1 and y2 > y1:
                gt[ctok].append({
                    "box": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "cat": box.name,
                })

    return gt


def _match(preds, gts, thr=0.40):
    """
    Greedy IoU matching.
    Returns:
      tp_ious, fp, fn
    """
    if not gts:
        return [], len(preds), 0

    if not preds:
        return [], 0, len(gts)

    mat = np.array([
        [_iou(p["box"], g["box"]) for g in gts]
        for p in preds
    ])

    mp, mg = set(), set()
    tp_ious = []

    while True:
        if mat.size == 0:
            break

        idx = np.unravel_index(np.argmax(mat), mat.shape)

        if mat[idx] < thr:
            break

        tp_ious.append(float(mat[idx]))

        mp.add(idx[0])
        mg.add(idx[1])

        mat[list(mp), :] = -1
        mat[:, list(mg)] = -1

    fp = len(preds) - len(mp)
    fn = len(gts) - len(mg)

    return tp_ious, fp, fn


def evaluate_crop_benefit(full_dets, final_dets, gt_boxes, iou_thr=0.40):
    tp_f, fp_f, fn_f = _match(full_dets, gt_boxes, iou_thr)
    tp_c, fp_c, fn_c = _match(final_dets, gt_boxes, iou_thr)

    def recall(tp, fn):
        return len(tp) / (len(tp) + fn) if (len(tp) + fn) > 0 else 0.0

    def precision(tp, fp):
        return len(tp) / (len(tp) + fp) if (len(tp) + fp) > 0 else 0.0

    full_recall = recall(tp_f, fn_f)
    final_recall = recall(tp_c, fn_c)

    return {
        "full_recall": full_recall,
        "final_recall": final_recall,
        "recall_gain": final_recall - full_recall,

        "full_prec": precision(tp_f, fp_f),
        "final_prec": precision(tp_c, fp_c),

        "full_fp": fp_f,
        "final_fp": fp_c,
        "extra_fp": fp_c - fp_f,

        "full_tp": len(tp_f),
        "final_tp": len(tp_c),
        "full_fn": fn_f,
        "final_fn": fn_c,

        "mean_iou_full": float(np.mean(tp_f)) if tp_f else 0.0,
        "mean_iou_final": float(np.mean(tp_c)) if tp_c else 0.0,
    }


# ══ Visualisation ═════════════════════════════════════════════════════════════

COLORS = {
    True: "#FF4444",
    False: "#44FF88",
}

ZONE_C = "#FFD700"
PRED_C = "#FF6600"
GT_C = "#00FF44"

CAT_ABBR = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
}


def visualise(
    image,
    full_dets,
    zones,
    crop_dets,
    final_preds,
    ev: dict,
    gt_boxes: list,
    frame_idx: int,
    out_path,
):
    arr = np.array(image)

    fig = plt.figure(figsize=(22, 14), facecolor="#0a0a0a")

    recall = ev.get("final_recall", 0.0)
    prec = ev.get("final_prec", 0.0)
    f1 = 2 * recall * prec / (recall + prec) if (recall + prec) > 0 else 0.0

    fig.suptitle(
        f"Frame {frame_idx} | "
        f"GT={len(gt_boxes)} Pred={len(final_preds)} "
        f"TP={ev.get('final_tp', 0)} FP={ev.get('final_fp', 0)} "
        f"FN={ev.get('final_fn', 0)} | "
        f"Recall={recall:.3f} Precision={prec:.3f} "
        f"F1={f1:.3f} Mean-IoU={ev.get('mean_iou_final', 0):.3f}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    gs = fig.add_gridspec(
        3,
        3,
        hspace=0.07,
        wspace=0.04,
        left=0.01,
        right=0.99,
        top=0.95,
        bottom=0.04,
        height_ratios=[2, 1, 1],
    )

    # Main comparison
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(arr)

    for g in gt_boxes:
        x1, y1, x2, y2 = g["box"]

        ax_main.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=GT_C,
                lw=2.5,
                linestyle="--",
                alpha=0.9,
            )
        )

        ax_main.text(
            x2,
            y1,
            g.get("cat", "gt").split(".")[-1],
            color=GT_C,
            fontsize=7,
            ha="right",
            bbox=dict(
                facecolor="#003300",
                alpha=0.7,
                pad=1,
                edgecolor="none",
            ),
        )

    for p in final_preds:
        x1, y1, x2, y2 = p["bbox_2d"]
        label = CAT_ABBR.get(
            p["category"],
            p["category"].split(".")[-1],
        )

        ax_main.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=PRED_C,
                lw=2.5,
            )
        )

        ax_main.text(
            x1,
            max(0, y1 - 3),
            label,
            color=PRED_C,
            fontsize=7,
            bbox=dict(
                facecolor="#1a0800",
                alpha=0.7,
                pad=1,
                edgecolor="none",
            ),
        )

    ax_main.set_title(
        f"GT green dashed vs Pipeline orange | "
        f"Recall {recall:.3f} Precision {prec:.3f} F1 {f1:.3f}",
        color="white",
        fontsize=10,
        pad=4,
    )
    ax_main.axis("off")

    # Pass 1
    ax0 = fig.add_subplot(gs[1, :2])
    ax0.imshow(arr)

    for d in full_dets:
        x1, y1, x2, y2 = d["box"]
        is_weak = d.get("needs_crop", False)
        col = COLORS[is_weak]

        label = f"{d['conf']:.2f}"
        if "crop_score" in d:
            label += f" / s={d['crop_score']:.2f}"

        ax0.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=col,
                lw=1.7 if is_weak else 2.0,
            )
        )

        ax0.text(
            x1,
            max(0, y1 - 3),
            label,
            color=col,
            fontsize=5.5,
            bbox=dict(
                facecolor="black",
                alpha=0.5,
                pad=1,
                edgecolor="none",
            ),
        )

    ax0.set_title(
        "Pass 1 raw detections | green=strong, red=high-risk crop triggers",
        color="white",
        fontsize=8,
        pad=3,
    )
    ax0.axis("off")

    # Zones
    ax1 = fig.add_subplot(gs[1, 2])
    ax1.imshow(arr)

    for z in zones:
        x1, y1, x2, y2 = z["roi"]

        ax1.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=True,
                facecolor=ZONE_C,
                alpha=0.22,
                edgecolor=ZONE_C,
                lw=2,
            )
        )

        ax1.text(
            x1 + 4,
            y1 + 12,
            z["reason"],
            color=ZONE_C,
            fontsize=5.5,
            bbox=dict(
                facecolor="black",
                alpha=0.6,
                pad=1,
                edgecolor="none",
            ),
        )

    if not zones:
        ax1.text(
            0.5,
            0.5,
            "No crop zone\nPass 1 sufficient",
            color="lime",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=9,
        )

    ax1.set_title(
        f"Crop zones={len(zones)} | Pass 2 raw dets={len(crop_dets)}",
        color=ZONE_C,
        fontsize=8,
        pad=3,
    )
    ax1.axis("off")

    # Stats
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.set_facecolor("#111111")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    cols_data = [
        ("GROUND TRUTH", GT_C, [
            f"Boxes : {len(gt_boxes)}",
        ]),
        ("PASS 1", "#AAAAAA", [
            f"Detections : {len(full_dets)}",
            f"Weak       : {sum(1 for d in full_dets if d.get('needs_crop'))}",
            f"Recall     : {ev.get('full_recall', 0):.3f}",
            f"Precision  : {ev.get('full_prec', 0):.3f}",
            f"TP / FP / FN: {ev.get('full_tp', 0)} / {ev.get('full_fp', 0)} / {ev.get('full_fn', 0)}",
            f"Mean IoU   : {ev.get('mean_iou_full', 0):.3f}",
        ]),
        ("CROP PASS", ZONE_C, [
            f"Zones      : {len(zones)}",
            f"Raw dets   : {len(crop_dets)}",
            f"Recall gain: {ev.get('recall_gain', 0):+.3f}",
            f"Extra FP   : {ev.get('extra_fp', 0):+d}",
        ]),
        ("FINAL OUTPUT", PRED_C, [
            f"Predictions: {len(final_preds)}",
            f"Recall     : {ev.get('final_recall', 0):.3f}",
            f"Precision  : {ev.get('final_prec', 0):.3f}",
            f"F1         : {f1:.3f}",
            f"TP / FP / FN: {ev.get('final_tp', 0)} / {ev.get('final_fp', 0)} / {ev.get('final_fn', 0)}",
            f"Mean IoU   : {ev.get('mean_iou_final', 0):.3f}",
        ]),
    ]

    x_step = 1.0 / len(cols_data)

    for ci, (title, col, items) in enumerate(cols_data):
        xpos = ci * x_step + 0.01

        ax_stats.text(
            xpos,
            0.93,
            title,
            color=col,
            fontsize=9,
            fontweight="bold",
            transform=ax_stats.transAxes,
            family="monospace",
        )

        for li, item in enumerate(items):
            ax_stats.text(
                xpos,
                0.78 - li * 0.15,
                item,
                color="white",
                fontsize=8,
                transform=ax_stats.transAxes,
                family="monospace",
            )

    fig.legend(
        handles=[
            mpatches.Patch(
                facecolor="none",
                edgecolor=GT_C,
                ls="--",
                lw=2,
                label="Ground Truth",
            ),
            mpatches.Patch(
                facecolor="none",
                edgecolor=PRED_C,
                lw=2,
                label="Pipeline Prediction",
            ),
            mpatches.Patch(
                facecolor="none",
                edgecolor="#44FF88",
                label="Pass 1 strong",
            ),
            mpatches.Patch(
                facecolor="none",
                edgecolor="#FF4444",
                label="Pass 1 high-risk",
            ),
            mpatches.Patch(
                facecolor=ZONE_C,
                alpha=0.4,
                label="Crop zone",
            ),
        ],
        loc="lower center",
        ncol=5,
        facecolor="#1a1a1a",
        edgecolor="#444",
        labelcolor="white",
        fontsize=9,
    )

    fig.savefig(
        str(out_path),
        dpi=120,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)


# ══ Main ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument(
        "--no-seg",
        action="store_true",
        help="Disable SegFormer fallback",
    )
    ap.add_argument(
        "--random-sweeps",
        action="store_true",
        help="Run on random unannotated photos from the sweeps folder",
    )

    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    nusc = NuScenes(
        version="v1.0-eval",
        dataroot=DATAROOT,
        verbose=False,
    )

    yolo = YOLO("yolov8m.pt")

    seg_model = None

    if not args.no_seg:
        print("Loading segmentation fallback: SegFormer-b0 ...")

        proc = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
        )

        model = (
            SegformerForSemanticSegmentation
            .from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-640-1280")
            .to(device)
            .eval()
        )

        seg_model = {
            "proc": proc,
            "model": model,
            "device": device,
        }

        print("SegFormer ready.")

    if args.random_sweeps:
        import random, glob
        sweep_files = glob.glob(f"{DATAROOT}/sweeps/**/*.jpg", recursive=True)
        if not sweep_files:
            print("No sweeps found in", f"{DATAROOT}/sweeps")
            return
            
        target = args.n
        sample_paths = random.sample(sweep_files, min(target, len(sweep_files)))
        
        print(f"\nProcessing {len(sample_paths)} random sweep photos ...\n")
        hdr = f"{'F':3} {'P1':3} {'Wk':3} {'Zo':2} {'P2':3} {'Fin':3} | {'Tiles':5}"
        print(hdr)
        print("-" * len(hdr))

        for done, img_path in enumerate(sample_paths, 1):
            image = Image.open(img_path).convert("RGB")

            # Pass 1
            full_dets, full_time = run_full_pass(yolo, image)
            analyse(full_dets, *image.size)

            # Zone selection
            zones = select_zones(full_dets, image, seg_model)

            # Pass 2
            crop_dets, n_tiles = run_crop_pass(yolo, image, zones)

            # Merge
            final_dets = merge(full_dets, crop_dets)
            filtered_final = filter_dets_for_eval(final_dets)

            n_weak = sum(1 for d in full_dets if d.get("needs_crop"))
            print(
                f"{done:3d} {len(full_dets):3d} "
                f"{n_weak:3d} {len(zones):2d} {len(crop_dets):3d} "
                f"{len(filtered_final):3d} | {n_tiles:5d}"
            )

            if not args.no_viz:
                out_path = OUT_DIR / f"random_sweep_{done:03d}.png"
                # Dummy eval struct to prevent visualize from crashing
                ev = {"final_tp": 0, "final_fp": 0, "final_fn": 0, "full_recall": 0, 
                      "final_recall": 0, "recall_gain": 0, "final_prec": 0, "extra_fp": 0}
                # Create fake predictions in the expected format
                preds = []
                for d in filtered_final:
                    cat = YOLO_TO_NS.get(d["cid"])
                    if not cat: continue
                    preds.append({"camera": "UNKNOWN", "category": cat, "bbox_2d": d["box"]})
                
                visualise(image, full_dets, zones, crop_dets, preds, ev, [], done, out_path)

        if not args.no_viz:
            print(f"\nVisualisations  : {OUT_DIR}/")
        return

    print("Loading GT...")
    gt_by_tok = load_gt(nusc)

    target = (
        sum(s["nbr_samples"] for s in nusc.scene)
        if args.all
        else args.n
    )

    done = 0
    all_preds = []

    agg = defaultdict(float)
    agg_counts = defaultdict(float)

    print(f"\nProcessing {target} frame(s) ...\n")

    hdr = (
        f"{'F':3} {'GT':3} {'P1':3} {'Wk':3} {'Zo':2} "
        f"{'P2':3} {'Fin':3} | "
        f"{'R1':5} {'R2':5} {'dR':5} | "
        f"{'Pr2':5} {'dFP':4} {'Tiles':5}"
    )

    print(hdr)
    print("-" * len(hdr))

    for scene in nusc.scene:
        token = scene["first_sample_token"]

        while token:
            sample = nusc.get("sample", token)

            cam_tok = sample["data"]["CAM_FRONT"]
            img_path = nusc.get_sample_data_path(cam_tok)

            image = Image.open(img_path).convert("RGB")
            gt_boxes = gt_by_tok.get(cam_tok, [])

            # Pass 1
            full_dets, full_time = run_full_pass(yolo, image)
            analyse(full_dets, *image.size)

            # Zone selection
            zones = select_zones(full_dets, image, seg_model)

            # Pass 2
            crop_dets, n_tiles = run_crop_pass(yolo, image, zones)

            # Merge
            final_dets = merge(full_dets, crop_dets)

            # Filtered versions for fair evaluation
            filtered_full = filter_dets_for_eval(full_dets)
            filtered_final = filter_dets_for_eval(final_dets)

            # Submission predictions
            preds = dets_to_submission(final_dets, cam_tok)
            all_preds.extend(preds)

            # Evaluation
            ev = evaluate_crop_benefit(
                filtered_full,
                filtered_final,
                gt_boxes,
            )

            efficiency = ev["recall_gain"] / max(1, n_tiles)
            fp_cost = ev["extra_fp"] / max(1, n_tiles)

            ev["efficiency"] = efficiency
            ev["fp_cost"] = fp_cost
            ev["tiles"] = n_tiles

            for k, v in ev.items():
                agg[k] += v

            agg_counts["gt_total"] += len(gt_boxes)
            agg_counts["pred_total"] += len(preds)
            agg_counts["tiles_total"] += n_tiles

            done += 1

            n_weak = sum(1 for d in full_dets if d.get("needs_crop"))

            print(
                f"{done:3d} {len(gt_boxes):3d} {len(full_dets):3d} "
                f"{n_weak:3d} {len(zones):2d} {len(crop_dets):3d} "
                f"{len(preds):3d} | "
                f"{ev['full_recall']:5.2f} {ev['final_recall']:5.2f} "
                f"{ev['recall_gain']:+5.2f} | "
                f"{ev['final_prec']:5.2f} {ev['extra_fp']:+4d} "
                f"{n_tiles:5d}"
            )

            if not args.no_viz:
                out_path = OUT_DIR / f"adaptive_v2_{done:03d}.png"

                visualise(
                    image=image,
                    full_dets=full_dets,
                    zones=zones,
                    crop_dets=crop_dets,
                    final_preds=preds,
                    ev=ev,
                    gt_boxes=gt_boxes,
                    frame_idx=done,
                    out_path=out_path,
                )

            token = sample["next"]

            if done >= target:
                token = ""

        if done >= target:
            break

    n = max(1, done)

    sep = "=" * 54

    print(f"\n{sep}")
    print(f"AGGREGATE EVALUATION ({done} frames)")
    print(sep)

    print(f"Avg GT per frame        : {agg_counts['gt_total'] / n:.2f}")
    print(f"Avg Pred per frame      : {agg_counts['pred_total'] / n:.2f}")
    print(f"Avg tiles per frame     : {agg_counts['tiles_total'] / n:.2f}")

    print(f"Avg Pass-1 recall       : {agg['full_recall'] / n:.3f}")
    print(f"Avg Final recall        : {agg['final_recall'] / n:.3f}")
    print(f"Avg recall gain         : {agg['recall_gain'] / n:+.3f}")

    print(f"Avg Pass-1 precision    : {agg['full_prec'] / n:.3f}")
    print(f"Avg Final precision     : {agg['final_prec'] / n:.3f}")

    print(f"Avg extra FP / frame    : {agg['extra_fp'] / n:+.2f}")
    print(f"Avg TP IoU Pass 1       : {agg['mean_iou_full'] / n:.3f}")
    print(f"Avg TP IoU Final        : {agg['mean_iou_final'] / n:.3f}")

    print(f"Avg efficiency dR/tile  : {agg['efficiency'] / n:+.3f}")
    print(f"Avg FP cost FP/tile     : {agg['fp_cost'] / n:+.3f}")

    print(sep)

    OUTPUT_JSON.write_text(json.dumps(all_preds, indent=2))

    print(f"Submission JSON : {OUTPUT_JSON}")

    if not args.no_viz:
        print(f"Visualisations  : {OUT_DIR}/")


if __name__ == "__main__":
    main()