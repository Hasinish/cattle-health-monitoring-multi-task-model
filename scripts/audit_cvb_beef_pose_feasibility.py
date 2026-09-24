# -*- coding: utf-8 -*-
"""CVB + Kaggle Beef Behavior Pose Feasibility Audit (SuperAnimal-Quadruped ResNet-50).

Scientific Feasibility Assessment:
  Evaluates frozen DeepLabCut SuperAnimal-Quadruped ResNet-50 on the authentic
  CVB + Kaggle Beef Behavior representation (cattle-centered RGB crops + binary masks,
  T=8 frames per sequence) staged under /mtl-data/behavior/.

Sampling:
  Deterministic representative sample (seed 2026) stratified across all 9 available
  source x class cells:
    - CVB: Standing, Lying, Feeding, Drinking, Walking (5 cells)
    - Beef: Standing, Lying, Feeding, Drinking (4 cells; Walking does not exist in Beef)
  6 sequences per cell (4 train, 2 val) = 54 sequences total x 8 frames = 432 frames.

Strict Isolation:
  Evaluates ONLY staged train/val sequences from retained_train.csv and retained_val.csv.
  Held-out Behavior test set is STRICTLY UNTOUCHED.

Metrics & Integrity:
  - Frame-level detection success and keypoint confidence across all 39 landmarks.
  - Mask containment sanity: percentage of predicted keypoints inside the Run-5 mask.
    EXPLICIT LABEL: "predicted-mask geometric containment sanity check"
    (NOT pose accuracy; no ground-truth keypoint annotations exist for CVB/Beef).
  - Temporal consistency sanity: frame-to-frame keypoint displacement and jump detection.
  - Multi-frame visual contact sheets covering all 9 source x class cells.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical Behavior classes & available source x class cells
CANONICAL_CLASSES = ["Standing", "Lying", "Feeding", "Drinking", "Walking"]

AVAILABLE_CELLS: List[Tuple[str, str]] = [
    ("cvb", "Standing"),
    ("cvb", "Lying"),
    ("cvb", "Feeding"),
    ("cvb", "Drinking"),
    ("cvb", "Walking"),
    ("beef_cattle_behavior", "Standing"),
    ("beef_cattle_behavior", "Lying"),
    ("beef_cattle_behavior", "Feeding"),
    ("beef_cattle_behavior", "Drinking"),
]

# Anatomical quadruped skeleton fallback connections for the official 39-keypoint schema
STANDARD_QUADRUPED_SKELETON_PAIRS: List[Tuple[str, str]] = [
    # Head & Neck
    ("nose", "upper_jaw"),
    ("upper_jaw", "lower_jaw"),
    ("upper_jaw", "neck_base"),
    ("right_eye", "right_earbase"),
    ("left_eye", "left_earbase"),
    ("right_earbase", "neck_base"),
    ("left_earbase", "neck_base"),
    ("right_earbase", "right_earend"),
    ("left_earbase", "left_earend"),
    ("right_antler_base", "right_antler_end"),
    ("left_antler_base", "left_antler_end"),
    ("right_earbase", "right_antler_base"),
    ("left_earbase", "left_antler_base"),
    # Spine & Tail
    ("neck_base", "throat_base"),
    ("neck_base", "neck_end"),
    ("neck_end", "back_base"),
    ("back_base", "back_middle"),
    ("back_middle", "back_end"),
    ("back_end", "tail_base"),
    ("tail_base", "tail_end"),
    # Belly & Flank
    ("throat_base", "throat_end"),
    ("throat_end", "belly_bottom"),
    ("belly_bottom", "back_end"),
    ("back_middle", "body_middle_right"),
    ("back_middle", "body_middle_left"),
    # Forelimbs
    ("back_base", "front_left_thai"),
    ("front_left_thai", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),
    ("back_base", "front_right_thai"),
    ("front_right_thai", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),
    # Hindlimbs
    ("back_end", "back_left_thai"),
    ("back_left_thai", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),
    ("back_end", "back_right_thai"),
    ("back_right_thai", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
]


def check_keypoints_inside_mask(
    keypoints: List[Tuple[float, float, float]],
    mask_np: np.ndarray,
) -> Tuple[float, int, int, float, int]:
    """Calculate the percentage of keypoints landing on foreground mask pixels.

    Explicit Label: predicted-mask geometric containment sanity check
    NOT pose accuracy (no keypoint GT exists, masks are perception-derived).

    Returns:
        (inside_rate_all, inside_count_all, total_keypoints,
         inside_rate_conf02, inside_count_conf02)
    """
    h, w = mask_np.shape[:2]
    inside_all = 0
    total = len(keypoints)
    if total == 0:
        return 0.0, 0, 0, 0.0, 0

    inside_c02 = 0
    total_c02 = 0

    for kx, ky, conf in keypoints:
        ix = int(round(kx))
        iy = int(round(ky))
        is_inside = False
        if 0 <= ix < w and 0 <= iy < h:
            if mask_np[iy, ix] > 0:
                is_inside = True

        if is_inside:
            inside_all += 1
        if conf >= 0.2:
            total_c02 += 1
            if is_inside:
                inside_c02 += 1

    rate_all = round(inside_all / total, 4)
    rate_c02 = round(inside_c02 / max(total_c02, 1), 4) if total_c02 > 0 else 0.0
    return rate_all, inside_all, total, rate_c02, inside_c02


def compute_temporal_displacement(
    kpts_prev: List[Tuple[float, float, float]],
    kpts_curr: List[Tuple[float, float, float]],
    img_size: int = 224,
    conf_thresh: float = 0.2,
) -> Dict[str, Any]:
    """Compute frame-to-frame keypoint displacement for temporal consistency sanity."""
    if len(kpts_prev) != len(kpts_curr) or len(kpts_prev) == 0:
        return {
            "valid_keypoints_tracked": 0,
            "mean_displacement_px": 0.0,
            "mean_displacement_norm": 0.0,
            "max_displacement_px": 0.0,
            "max_displacement_norm": 0.0,
            "jump_count_gt_025": 0,
        }

    displacements_px = []
    jumps_gt_025 = 0

    for (x1, y1, c1), (x2, y2, c2) in zip(kpts_prev, kpts_curr):
        if c1 >= conf_thresh and c2 >= conf_thresh:
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            displacements_px.append(dist)
            if (dist / float(img_size)) > 0.25:
                jumps_gt_025 += 1

    if not displacements_px:
        return {
            "valid_keypoints_tracked": 0,
            "mean_displacement_px": 0.0,
            "mean_displacement_norm": 0.0,
            "max_displacement_px": 0.0,
            "max_displacement_norm": 0.0,
            "jump_count_gt_025": 0,
        }

    mean_d = float(np.mean(displacements_px))
    max_d = float(np.max(displacements_px))
    return {
        "valid_keypoints_tracked": len(displacements_px),
        "mean_displacement_px": round(mean_d, 2),
        "mean_displacement_norm": round(mean_d / float(img_size), 4),
        "max_displacement_px": round(max_d, 2),
        "max_displacement_norm": round(max_d / float(img_size), 4),
        "jump_count_gt_025": jumps_gt_025,
    }


def draw_pose_frame_tile(
    rgb_img: Image.Image,
    mask_np: np.ndarray,
    keypoints: List[Tuple[float, float, float]],
    skeleton_indices: List[Tuple[int, int]],
    header_text: str,
    tile_size: int = 224,
) -> Image.Image:
    """Renders a single frame tile with RGB crop, mask overlay, and pose skeleton."""
    tile = rgb_img.resize((tile_size, tile_size), Image.Resampling.BILINEAR).convert("RGB")
    tile_np = np.array(tile)

    # Blend cyan mask overlay if mask exists
    if mask_np is not None:
        mask_resized = cv2.resize(
            (mask_np > 0).astype(np.uint8),
            (tile_size, tile_size),
            interpolation=cv2.INTER_NEAREST,
        )
        # Highlight mask contour or semi-transparent tint
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tile_np, contours, -1, (0, 230, 230), 1)

    tile = Image.fromarray(tile_np)
    draw = ImageDraw.Draw(tile)

    scale = tile_size / 224.0

    # Draw skeleton connections
    for i1, i2 in skeleton_indices:
        if i1 < len(keypoints) and i2 < len(keypoints):
            x1, y1, c1 = keypoints[i1]
            x2, y2, c2 = keypoints[i2]
            if c1 >= 0.2 and c2 >= 0.2:
                sx1, sy1 = int(round(x1 * scale)), int(round(y1 * scale))
                sx2, sy2 = int(round(x2 * scale)), int(round(y2 * scale))
                draw.line([(sx1, sy1), (sx2, sy2)], fill=(0, 220, 255), width=2)

    # Draw keypoints
    for kx, ky, conf in keypoints:
        px = int(round(kx * scale))
        py = int(round(ky * scale))
        if conf >= 0.2:
            color = (0, 255, 0)
            rad = 3
        else:
            color = (255, 60, 60)
            rad = 2
        draw.ellipse([(px - rad, py - rad), (px + rad, py + rad)], fill=color)

    # Create tile with top header bar
    header_h = 28
    tile_with_header = Image.new("RGB", (tile_size, tile_size + header_h), (25, 25, 30))
    h_draw = ImageDraw.Draw(tile_with_header)
    h_draw.text((4, 6), header_text, fill=(220, 220, 240))
    tile_with_header.paste(tile, (0, header_h))

    return tile_with_header


def generate_9cell_contact_sheet(
    rep_sequences: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: Path,
    tile_size: int = 224,
) -> None:
    """Generates a comprehensive 9-row contact sheet covering all available source x class cells.

    Each row represents one representative sequence with all T=8 frames displayed chronologically.
    """
    num_rows = len(AVAILABLE_CELLS)
    num_cols = 8  # T=8 frames

    header_h = 28
    row_tile_h = tile_size + header_h
    row_label_w = 200
    margin = 8
    top_banner_h = 60

    total_w = row_label_w + num_cols * (tile_size + margin) + margin
    total_h = top_banner_h + num_rows * (row_tile_h + margin) + margin

    sheet = Image.new("RGB", (total_w, total_h), (15, 15, 20))
    draw = ImageDraw.Draw(sheet)

    # Main Title Banner
    title = "Phase 3 Behavior Pose Feasibility: SuperAnimal-Quadruped ResNet-50 on CVB + Kaggle Beef (T=8)"
    subtitle = "Geometric Sanity Overlays: Predicted Mask Contour (Cyan) | Skeleton Connections (Conf >= 0.2) | Keypoints (Green>=0.2, Red<0.2)"
    draw.text((margin + 10, 12), title, fill=(255, 255, 255))
    draw.text((margin + 10, 36), subtitle, fill=(180, 190, 210))

    for row_idx, cell in enumerate(AVAILABLE_CELLS):
        src, cls = cell
        y_pos = top_banner_h + row_idx * (row_tile_h + margin)

        seq_data = rep_sequences.get(cell, None)
        if seq_data is None:
            continue

        sample_id = seq_data["sample_id"]
        split = seq_data["split"]
        frames = seq_data["frames"]  # list of 8 frame dicts

        # Row Label
        src_label = "CVB" if src == "cvb" else "Beef"
        row_text = f"{src_label} - {cls}\n{split.upper()}\n{sample_id[:16]}..."
        draw.text((margin + 10, y_pos + 20), row_text, fill=(240, 240, 240))

        # 8 Frame Tiles
        for col_idx in range(min(num_cols, len(frames))):
            f_data = frames[col_idx]
            x_pos = row_label_w + col_idx * (tile_size + margin)

            header_text = (
                f"t={f_data['t']} | Conf: {f_data['mean_conf']:.2f} | "
                f"InMask: {f_data['inside_rate']*100:.0f}%"
            )
            tile_img = draw_pose_frame_tile(
                rgb_img=f_data["rgb_pil"],
                mask_np=f_data["mask_np"],
                keypoints=f_data["keypoints"],
                skeleton_indices=seq_data["skeleton_indices"],
                header_text=header_text,
                tile_size=tile_size,
            )
            sheet.paste(tile_img, (x_pos, y_pos))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(f"[OK] Saved comprehensive 9-cell contact sheet ({total_w}x{total_h}): {output_path}")


def audit_cvb_beef_pose(
    behavior_data_dir: Path,
    output_dir: Path,
    sample_seed: int = 2026,
    seqs_per_cell: int = 6,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Runs frozen SuperAnimal-Quadruped ResNet-50 feasibility audit on CVB + Beef Behavior."""
    import torch
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
        create_superanimal_inference_runners,
    )

    t0 = time.time()
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  PHASE 3: CVB + KAGGLE BEEF BEHAVIOR POSE FEASIBILITY AUDIT")
    print("  Model : DeepLabCut SuperAnimal-Quadruped ResNet-50 (Frozen Pretrained)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Data  : {behavior_data_dir}")
    print(f"  Seed  : {sample_seed}")
    print("=" * 80)

    # 1. Load train and val retained metadata
    train_csv_path = behavior_data_dir / "retained_train.csv"
    val_csv_path = behavior_data_dir / "retained_val.csv"

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Missing train manifest at {train_csv_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Missing val manifest at {val_csv_path}")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    print(f"[OK] Loaded metadata manifests:")
    print(f"     - Train sequences: {len(df_train)}")
    print(f"     - Val sequences  : {len(df_val)}")

    # 2. Deterministic stratified sampling across all 9 source x class cells
    # Target: 6 sequences per cell (4 train, 2 val where possible)
    sampled_records: List[Dict[str, Any]] = []
    print(f"\n[*] Sampling {seqs_per_cell} sequences per available source x class cell (seed {sample_seed}):")

    for src, cls in AVAILABLE_CELLS:
        train_pool = df_train[
            (df_train["dataset"] == src) & (df_train["behavior_canonical"] == cls)
        ].copy()
        val_pool = df_val[
            (df_val["dataset"] == src) & (df_val["behavior_canonical"] == cls)
        ].copy()

        n_train_target = 4
        n_val_target = 2

        # Safe fallback if pool is small
        n_train_actual = min(len(train_pool), n_train_target)
        n_val_actual = min(len(val_pool), n_val_target)
        # If val has fewer, take more from train
        if n_val_actual < n_val_target:
            n_train_actual = min(len(train_pool), seqs_per_cell - n_val_actual)

        train_sample = train_pool.sample(n=n_train_actual, random_state=sample_seed)
        val_sample = val_pool.sample(n=n_val_actual, random_state=sample_seed)

        cell_df = pd.concat([train_sample, val_sample], ignore_index=True)
        for _, r in cell_df.iterrows():
            rec = r.to_dict()
            rec["audit_cell"] = f"{src}_{cls}"
            sampled_records.append(rec)

        print(
            f"     - [{src:20s} x {cls:8s}]: {len(cell_df)} sequences "
            f"({n_train_actual} train, {n_val_actual} val)"
        )

    print(f"[OK] Total sampled sequences for audit: {len(sampled_records)} (All 8 frames = {len(sampled_records) * 8} frames)")

    # 3. Initialize official SuperAnimal runners
    print("\n[*] Initializing official SuperAnimal-Quadruped ResNet-50 pipeline...")
    init_t0 = time.time()
    pose_runner, det_runner, model_cfg = create_superanimal_inference_runners(
        superanimal_name="superanimal_quadruped",
        model_name="resnet_50",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        max_individuals=1,
        batch_size=1,
        detector_batch_size=1,
        device=str(device),
    )
    bodyparts: List[str] = model_cfg["metadata"]["bodyparts"]
    raw_skeleton = model_cfg.get("skeleton", [])
    skeleton_indices: List[Tuple[int, int]] = []

    if raw_skeleton:
        for b1, b2 in raw_skeleton:
            if b1 in bodyparts and b2 in bodyparts:
                skeleton_indices.append((bodyparts.index(b1), bodyparts.index(b2)))
    if not skeleton_indices:
        # Fallback to anatomical quadruped skeleton
        for b1, b2 in STANDARD_QUADRUPED_SKELETON_PAIRS:
            if b1 in bodyparts and b2 in bodyparts:
                skeleton_indices.append((bodyparts.index(b1), bodyparts.index(b2)))

    print(f"[OK] SuperAnimal initialized in {time.time() - init_t0:.2f}s ({len(bodyparts)} keypoints, {len(skeleton_indices)} skeleton edges)")

    # Save official 39-keypoint schema reference
    schema_path = output_dir / "superanimal_39_keypoints_schema.json"
    schema_payload = {
        "model_name": "resnet_50",
        "superanimal_name": "superanimal_quadruped",
        "detector_name": "fasterrcnn_resnet50_fpn_v2",
        "num_keypoints": len(bodyparts),
        "keypoint_names": bodyparts,
        "skeleton_connections": [
            [bodyparts[i], bodyparts[j]] for i, j in skeleton_indices
        ],
        "skeleton_indices": skeleton_indices,
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2)
    print(f"[OK] Saved official keypoint schema reference: {schema_path}")

    # 4. Evaluation Loop
    frame_results: List[Dict[str, Any]] = []
    seq_summaries: List[Dict[str, Any]] = []
    rep_visual_sequences: Dict[Tuple[str, str], Dict[str, Any]] = {}

    per_keypoint_conf: Dict[str, List[float]] = {bp: [] for bp in bodyparts}
    per_keypoint_inside: Dict[str, List[int]] = {bp: [] for bp in bodyparts}
    per_keypoint_disp: Dict[str, List[float]] = {bp: [] for bp in bodyparts}

    print("\n[*] Running SuperAnimal inference across all sequences and T=8 frames...")
    infer_start_time = time.time()

    for seq_idx, s_rec in enumerate(sampled_records):
        sample_id = str(s_rec["sample_id"])
        src = str(s_rec["dataset"])
        cls = str(s_rec["behavior_canonical"])
        split = str(s_rec["split"])
        cell_key = (src, cls)

        seq_dir = behavior_data_dir / sample_id
        if not seq_dir.exists():
            print(f"[WARN] Sequence folder missing at {seq_dir}, skipping")
            continue

        seq_frame_data: List[Dict[str, Any]] = []
        seq_kpts_by_frame: Dict[int, List[Tuple[float, float, float]]] = {}
        seq_confs: List[float] = []
        seq_inside_rates: List[float] = []
        seq_inside_c02_rates: List[float] = []
        detector_success_count = 0
        pose_returned_count = 0

        for t in range(8):
            f_img_p = seq_dir / f"frame_{t:02d}.jpg"
            f_msk_p = seq_dir / f"mask_{t:02d}.png"

            if not f_img_p.exists() or not f_msk_p.exists():
                print(f"[WARN] Missing frame/mask at {seq_dir} t={t}")
                continue

            with Image.open(f_img_p) as im:
                rgb_pil = im.convert("RGB")
            mask_gray = cv2.imread(str(f_msk_p), cv2.IMREAD_GRAYSCALE)
            mask_np = (mask_gray > 127).astype(np.uint8) if mask_gray is not None else np.zeros((224, 224), dtype=np.uint8)

            rgb_np = np.asarray(rgb_pil)
            img_h, img_w = rgb_np.shape[:2]

            t_infer = time.time()

            # Run Detector
            det_preds = det_runner.inference([rgb_np]) if det_runner is not None else None
            has_box = False
            if det_preds is not None and len(det_preds) > 0:
                bboxes = det_preds[0].get("bboxes", [])
                scores = det_preds[0].get("bbox_scores", [])
                if len(bboxes) > 0 and (len(scores) == 0 or np.max(scores) > 0.0):
                    has_box = True

            status = "pose_output_returned"
            keypoints_frame: List[Tuple[float, float, float]] = []
            frame_confs: List[float] = []

            if not has_box and det_runner is not None:
                status = "pose_detector_failure"
            else:
                detector_success_count += 1
                pose_inputs = [(rgb_np, det_preds[0])] if det_preds is not None else [rgb_np]
                pose_preds = pose_runner.inference(pose_inputs)
                if not pose_preds:
                    status = "pose_inference_error"
                else:
                    pred = pose_preds[0]
                    raw_bpts = pred.get("bodyparts", None)
                    if raw_bpts is not None:
                        bpts_arr = np.asarray(raw_bpts)
                        if bpts_arr.ndim == 3:
                            bpts_arr = bpts_arr[0]
                        for k_idx, bp_name in enumerate(bodyparts):
                            if k_idx < len(bpts_arr):
                                kx = float(bpts_arr[k_idx, 0])
                                ky = float(bpts_arr[k_idx, 1])
                                kc = float(bpts_arr[k_idx, 2])
                                keypoints_frame.append((kx, ky, kc))
                                frame_confs.append(kc)
                                per_keypoint_conf[bp_name].append(kc)
                    else:
                        coords = pred.get("coordinates", [])
                        conf_arr = pred.get("confidence", [])
                        if len(coords) > 0 and len(conf_arr) > 0:
                            c_xy = coords[0] if len(coords.shape) == 3 else coords
                            c_conf = conf_arr[0] if len(conf_arr.shape) == 2 else conf_arr
                            for k_idx, bp_name in enumerate(bodyparts):
                                kx = float(c_xy[k_idx][0])
                                ky = float(c_xy[k_idx][1])
                                kc = float(c_conf[k_idx])
                                keypoints_frame.append((kx, ky, kc))
                                frame_confs.append(kc)
                                per_keypoint_conf[bp_name].append(kc)

            infer_ms = (time.time() - t_infer) * 1000.0

            if status == "pose_output_returned" and len(keypoints_frame) == len(bodyparts):
                pose_returned_count += 1
                seq_kpts_by_frame[t] = keypoints_frame

            # Mask containment sanity check
            in_rate_all, in_cnt_all, tot_kpts, in_rate_c02, in_cnt_c02 = check_keypoints_inside_mask(
                keypoints_frame, mask_np
            )

            for k_idx, bp_name in enumerate(bodyparts):
                if k_idx < len(keypoints_frame):
                    kx, ky, _ = keypoints_frame[k_idx]
                    ix, iy = int(round(kx)), int(round(ky))
                    is_in = int(0 <= ix < img_w and 0 <= iy < img_h and mask_np[iy, ix] > 0)
                    per_keypoint_inside[bp_name].append(is_in)

            m_conf = float(np.mean(frame_confs)) if frame_confs else 0.0
            med_conf = float(np.median(frame_confs)) if frame_confs else 0.0
            seq_confs.extend(frame_confs)
            seq_inside_rates.append(in_rate_all)
            seq_inside_c02_rates.append(in_rate_c02)

            f_rec = {
                "dataset": src,
                "behavior_canonical": cls,
                "split": split,
                "sample_id": sample_id,
                "t": t,
                "detector_success": has_box,
                "pose_returned": (status == "pose_output_returned"),
                "status": status,
                "infer_ms": round(infer_ms, 2),
                "mean_conf": round(m_conf, 4),
                "median_conf": round(med_conf, 4),
                "points_conf_ge_02": sum(1 for c in frame_confs if c >= 0.2),
                "points_conf_ge_05": sum(1 for c in frame_confs if c >= 0.5),
                "mask_containment_sanity_rate_all": in_rate_all,
                "mask_containment_sanity_rate_conf02": in_rate_c02,
                "mask_inside_count": in_cnt_all,
            }

            # Append individual keypoint coords/conf/inside to frame record
            for k_idx, bp in enumerate(bodyparts):
                if k_idx < len(keypoints_frame):
                    kx, ky, kc = keypoints_frame[k_idx]
                    ix, iy = int(round(kx)), int(round(ky))
                    is_in = int(0 <= ix < img_w and 0 <= iy < img_h and mask_np[iy, ix] > 0)
                    f_rec[f"{bp}_x"] = round(kx, 2)
                    f_rec[f"{bp}_y"] = round(ky, 2)
                    f_rec[f"{bp}_conf"] = round(kc, 4)
                    f_rec[f"{bp}_in_mask"] = is_in
                else:
                    f_rec[f"{bp}_x"] = None
                    f_rec[f"{bp}_y"] = None
                    f_rec[f"{bp}_conf"] = 0.0
                    f_rec[f"{bp}_in_mask"] = 0

            frame_results.append(f_rec)

            seq_frame_data.append({
                "t": t,
                "rgb_pil": rgb_pil,
                "mask_np": mask_np,
                "keypoints": keypoints_frame,
                "mean_conf": m_conf,
                "inside_rate": in_rate_all,
            })

        # Calculate Sequence-Level Temporal Displacement
        transitions_evaluated = 0
        seq_displacements = []
        seq_jumps_gt_025 = 0

        for t in range(7):
            if t in seq_kpts_by_frame and (t + 1) in seq_kpts_by_frame:
                disp_info = compute_temporal_displacement(
                    seq_kpts_by_frame[t], seq_kpts_by_frame[t + 1]
                )
                if disp_info["valid_keypoints_tracked"] > 0:
                    transitions_evaluated += 1
                    seq_displacements.append(disp_info["mean_displacement_norm"])
                    seq_jumps_gt_025 += disp_info["jump_count_gt_025"]

                    # Track per-keypoint temporal displacement
                    for bp_idx, bp_name in enumerate(bodyparts):
                        (x1, y1, c1) = seq_kpts_by_frame[t][bp_idx]
                        (x2, y2, c2) = seq_kpts_by_frame[t + 1][bp_idx]
                        if c1 >= 0.2 and c2 >= 0.2:
                            d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 224.0
                            per_keypoint_disp[bp_name].append(d)

        mean_seq_disp = float(np.mean(seq_displacements)) if seq_displacements else 0.0
        max_seq_disp = float(np.max(seq_displacements)) if seq_displacements else 0.0

        # Determine temporal stability flag
        if pose_returned_count < 6:
            stability_flag = "PARTIAL_DETECTION_FAILURE"
        elif seq_jumps_gt_025 > 5:
            stability_flag = "SEVERE_INSTABILITY"
        elif seq_jumps_gt_025 > 0:
            stability_flag = "MODERATE_JUMPS"
        else:
            stability_flag = "STABLE"

        seq_summary = {
            "sample_id": sample_id,
            "dataset": src,
            "behavior_canonical": cls,
            "split": split,
            "pose_returned_frames": pose_returned_count,
            "detector_success_frames": detector_success_count,
            "all_8_frames_pose": (pose_returned_count == 8),
            "mean_keypoint_conf": round(float(np.mean(seq_confs)), 4) if seq_confs else 0.0,
            "mask_containment_sanity_rate_all": round(float(np.mean(seq_inside_rates)), 4) if seq_inside_rates else 0.0,
            "mask_containment_sanity_rate_conf02": round(float(np.mean(seq_inside_c02_rates)), 4) if seq_inside_c02_rates else 0.0,
            "transitions_evaluated": transitions_evaluated,
            "mean_frame_displacement_norm": round(mean_seq_disp, 4),
            "max_frame_displacement_norm": round(max_seq_disp, 4),
            "jumps_gt_025_count": seq_jumps_gt_025,
            "temporal_stability_flag": stability_flag,
        }
        seq_summaries.append(seq_summary)

        # Retain candidate representative sequence for visual contact sheet
        if cell_key not in rep_visual_sequences:
            if pose_returned_count >= 6 and len(seq_frame_data) == 8:
                rep_visual_sequences[cell_key] = {
                    "sample_id": sample_id,
                    "split": split,
                    "frames": seq_frame_data,
                    "skeleton_indices": skeleton_indices,
                }

        if (seq_idx + 1) % 9 == 0 or (seq_idx + 1) == len(sampled_records):
            print(f"     [Progress] {seq_idx + 1}/{len(sampled_records)} sequences processed ({(seq_idx + 1) * 8} frames)")

    total_infer_time = time.time() - infer_start_time
    print(f"\n[OK] Completed inference in {total_infer_time:.2f}s ({total_infer_time / len(frame_results):.3f}s/frame)")

    # 5. Fallback selection for representative visual sequences if any cell was missed
    for cell in AVAILABLE_CELLS:
        if cell not in rep_visual_sequences:
            # Pick first available sequence for this cell
            for s_rec in sampled_records:
                if (s_rec["dataset"], s_rec["behavior_canonical"]) == cell:
                    sid = str(s_rec["sample_id"])
                    # Gather frames
                    s_frames = [f for f in frame_results if f["sample_id"] == sid]
                    # Reload images
                    frames_loaded = []
                    s_dir = behavior_data_dir / sid
                    for t in range(8):
                        im_p = s_dir / f"frame_{t:02d}.jpg"
                        mk_p = s_dir / f"mask_{t:02d}.png"
                        if im_p.exists() and mk_p.exists():
                            with Image.open(im_p) as im:
                                r_pil = im.convert("RGB")
                            m_gr = cv2.imread(str(mk_p), cv2.IMREAD_GRAYSCALE)
                            m_np = (m_gr > 127).astype(np.uint8) if m_gr is not None else np.zeros((224, 224), dtype=np.uint8)
                            kpts = []
                            f_row = next((r for r in s_frames if r["t"] == t), None)
                            if f_row and f_row["pose_returned"]:
                                for bp in bodyparts:
                                    kpts.append((f_row[f"{bp}_x"], f_row[f"{bp}_y"], f_row[f"{bp}_conf"]))
                            frames_loaded.append({
                                "t": t,
                                "rgb_pil": r_pil,
                                "mask_np": m_np,
                                "keypoints": kpts,
                                "mean_conf": f_row["mean_conf"] if f_row else 0.0,
                                "inside_rate": f_row["mask_containment_sanity_rate_all"] if f_row else 0.0,
                            })
                    rep_visual_sequences[cell] = {
                        "sample_id": sid,
                        "split": str(s_rec["split"]),
                        "frames": frames_loaded,
                        "skeleton_indices": skeleton_indices,
                    }
                    break

    # 6. Save Contact Sheet
    contact_sheet_path = output_dir / "behavior_pose_all_classes_contact_sheet.jpg"
    generate_9cell_contact_sheet(rep_visual_sequences, contact_sheet_path)

    # 7. Convert to DataFrames and Aggregate Metrics
    df_frames = pd.DataFrame(frame_results)
    df_seqs = pd.DataFrame(seq_summaries)

    frame_csv_path = output_dir / "behavior_pose_frame_results.csv"
    seq_csv_path = output_dir / "behavior_pose_sequence_summary.csv"
    df_frames.to_csv(frame_csv_path, index=False)
    df_seqs.to_csv(seq_csv_path, index=False)
    print(f"[OK] Saved frame-level CSV ({len(df_frames)} rows): {frame_csv_path}")
    print(f"[OK] Saved sequence-level CSV ({len(df_seqs)} rows): {seq_csv_path}")

    total_frames = len(df_frames)
    returned_frames = int(df_frames["pose_returned"].sum())
    detector_failures = int((df_frames["status"] == "pose_detector_failure").sum())
    inference_errors = int((df_frames["status"] == "pose_inference_error").sum())
    pose_return_rate = round(returned_frames / max(total_frames, 1), 4)
    detector_failure_rate = round(detector_failures / max(total_frames, 1), 4)

    total_seqs = len(df_seqs)
    all_8_seqs = int(df_seqs["all_8_frames_pose"].sum())
    ge_1_seqs = int((df_seqs["pose_returned_frames"] > 0).sum())
    all_8_seq_rate = round(all_8_seqs / max(total_seqs, 1), 4)
    ge_1_seq_rate = round(ge_1_seqs / max(total_seqs, 1), 4)

    mean_conf_overall = round(float(df_frames[df_frames["pose_returned"]]["mean_conf"].mean()), 4)
    median_conf_overall = round(float(df_frames[df_frames["pose_returned"]]["median_conf"].median()), 4)

    mean_inside_all = round(float(df_frames[df_frames["pose_returned"]]["mask_containment_sanity_rate_all"].mean()), 4)
    mean_inside_c02 = round(float(df_frames[df_frames["pose_returned"]]["mask_containment_sanity_rate_conf02"].mean()), 4)

    mean_disp_overall = round(float(df_seqs["mean_frame_displacement_norm"].mean()), 4)
    total_jumps = int(df_seqs["jumps_gt_025_count"].sum())

    # Results by CVB vs Beef
    by_source: Dict[str, Any] = {}
    for src in ["cvb", "beef_cattle_behavior"]:
        f_sub = df_frames[df_frames["dataset"] == src]
        s_sub = df_seqs[df_seqs["dataset"] == src]
        by_source[src] = {
            "total_sequences": len(s_sub),
            "total_frames": len(f_sub),
            "pose_return_rate": round(float(f_sub["pose_returned"].mean()), 4),
            "detector_failure_rate": round(float((f_sub["status"] == "pose_detector_failure").mean()), 4),
            "all_8_frames_rate": round(float(s_sub["all_8_frames_pose"].mean()), 4),
            "mean_keypoint_confidence": round(float(f_sub[f_sub["pose_returned"]]["mean_conf"].mean()), 4),
            "predicted_mask_containment_sanity_rate_all": round(float(f_sub[f_sub["pose_returned"]]["mask_containment_sanity_rate_all"].mean()), 4),
            "predicted_mask_containment_sanity_rate_conf02": round(float(f_sub[f_sub["pose_returned"]]["mask_containment_sanity_rate_conf02"].mean()), 4),
            "mean_temporal_displacement_norm": round(float(s_sub["mean_frame_displacement_norm"].mean()), 4),
            "total_jumps_gt_025": int(s_sub["jumps_gt_025_count"].sum()),
        }

    # Results by Canonical Behavior Class
    by_class: Dict[str, Any] = {}
    for cls in CANONICAL_CLASSES:
        f_sub = df_frames[df_frames["behavior_canonical"] == cls]
        s_sub = df_seqs[df_seqs["behavior_canonical"] == cls]
        by_class[cls] = {
            "total_sequences": len(s_sub),
            "total_frames": len(f_sub),
            "pose_return_rate": round(float(f_sub["pose_returned"].mean()), 4),
            "detector_failure_rate": round(float((f_sub["status"] == "pose_detector_failure").mean()), 4),
            "all_8_frames_rate": round(float(s_sub["all_8_frames_pose"].mean()), 4),
            "mean_keypoint_confidence": round(float(f_sub[f_sub["pose_returned"]]["mean_conf"].mean()), 4),
            "predicted_mask_containment_sanity_rate_all": round(float(f_sub[f_sub["pose_returned"]]["mask_containment_sanity_rate_all"].mean()), 4),
            "predicted_mask_containment_sanity_rate_conf02": round(float(f_sub[f_sub["pose_returned"]]["mask_containment_sanity_rate_conf02"].mean()), 4),
            "mean_temporal_displacement_norm": round(float(s_sub["mean_frame_displacement_norm"].mean()), 4),
            "total_jumps_gt_025": int(s_sub["jumps_gt_025_count"].sum()),
        }

    # Results by Source x Class (all 9 cells)
    by_cell: Dict[str, Any] = {}
    for src, cls in AVAILABLE_CELLS:
        cell_key = f"{src}_{cls}"
        f_sub = df_frames[(df_frames["dataset"] == src) & (df_frames["behavior_canonical"] == cls)]
        s_sub = df_seqs[(df_seqs["dataset"] == src) & (df_seqs["behavior_canonical"] == cls)]
        by_cell[cell_key] = {
            "source": src,
            "class": cls,
            "total_sequences": len(s_sub),
            "total_frames": len(f_sub),
            "pose_return_rate": round(float(f_sub["pose_returned"].mean()), 4),
            "detector_failure_rate": round(float((f_sub["status"] == "pose_detector_failure").mean()), 4),
            "all_8_frames_rate": round(float(s_sub["all_8_frames_pose"].mean()), 4),
            "mean_keypoint_confidence": round(float(f_sub[f_sub["pose_returned"]]["mean_conf"].mean()), 4),
            "predicted_mask_containment_sanity_rate_all": round(float(f_sub[f_sub["pose_returned"]]["mask_containment_sanity_rate_all"].mean()), 4),
            "predicted_mask_containment_sanity_rate_conf02": round(float(f_sub[f_sub["pose_returned"]]["mask_containment_sanity_rate_conf02"].mean()), 4),
            "mean_temporal_displacement_norm": round(float(s_sub["mean_frame_displacement_norm"].mean()), 4),
            "total_jumps_gt_025": int(s_sub["jumps_gt_025_count"].sum()),
        }

    # Per-keypoint confidence and containment ranking
    keypoint_stats: Dict[str, Any] = {}
    for bp in bodyparts:
        c_vals = per_keypoint_conf[bp]
        in_vals = per_keypoint_inside[bp]
        d_vals = per_keypoint_disp[bp]
        keypoint_stats[bp] = {
            "mean_confidence": round(float(np.mean(c_vals)), 4) if c_vals else 0.0,
            "median_confidence": round(float(np.median(c_vals)), 4) if c_vals else 0.0,
            "pct_ge_02": round(float(np.mean([c >= 0.2 for c in c_vals])), 4) if c_vals else 0.0,
            "pct_ge_05": round(float(np.mean([c >= 0.5 for c in c_vals])), 4) if c_vals else 0.0,
            "predicted_mask_containment_rate": round(float(np.mean(in_vals)), 4) if in_vals else 0.0,
            "mean_temporal_displacement_norm": round(float(np.mean(d_vals)), 4) if d_vals else 0.0,
        }

    sorted_by_conf = sorted(keypoint_stats.items(), key=lambda x: x[1]["mean_confidence"], reverse=True)
    top_reliable_keypoints = [{"name": k, **v} for k, v in sorted_by_conf[:10]]
    consistently_weak_keypoints = [{"name": k, **v} for k, v in sorted_by_conf[-10:]]

    aggregate_metrics: Dict[str, Any] = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_name": "DeepLabCut SuperAnimal-Quadruped ResNet-50",
        "sample_seed": sample_seed,
        "sample_size_sequences": int(total_seqs),
        "sample_size_frames": int(total_frames),
        "overall": {
            "frame_level_pose_return_rate": pose_return_rate,
            "frame_level_detector_failure_rate": detector_failure_rate,
            "frame_level_inference_error_rate": round(inference_errors / max(total_frames, 1), 4),
            "sequence_level_all_8_frames_rate": all_8_seq_rate,
            "sequence_level_ge_1_frame_rate": ge_1_seq_rate,
            "mean_keypoint_confidence": mean_conf_overall,
            "median_keypoint_confidence": median_conf_overall,
            "predicted_mask_geometric_containment_sanity_rate_all": mean_inside_all,
            "predicted_mask_geometric_containment_sanity_rate_conf02": mean_inside_c02,
            "mean_temporal_displacement_norm": mean_disp_overall,
            "total_skeleton_jumps_gt_025": int(total_jumps),
            "temporal_stability_breakdown": {str(k): int(v) for k, v in df_seqs["temporal_stability_flag"].value_counts().items()},
        },
        "by_source": by_source,
        "by_canonical_class": by_class,
        "by_source_x_class_cells": by_cell,
        "keypoint_rankings": {
            "top_10_reliable_keypoints": top_reliable_keypoints,
            "bottom_10_weak_noisy_keypoints": consistently_weak_keypoints,
        },
        "keypoint_details": keypoint_stats,
    }

    def _json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_path = output_dir / "behavior_pose_aggregate_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2, default=_json_default)
    print(f"[OK] Saved aggregate metrics JSON: {json_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("  AUDIT SUMMARY: CVB + KAGGLE BEEF SUPERANIMAL-QUADRUPED POSE FEASIBILITY")
    print("=" * 80)
    print(f"Overall Frames Evaluated              : {total_frames} (54 sequences x 8 frames)")
    print(f"Frame-Level Pose Return Rate          : {pose_return_rate * 100:.2f}% ({returned_frames}/{total_frames})")
    print(f"Frame-Level Detector Failure Rate     : {detector_failure_rate * 100:.2f}% ({detector_failures}/{total_frames})")
    print(f"Sequence-Level All 8 Frames Return Rate: {all_8_seq_rate * 100:.2f}% ({all_8_seqs}/{total_seqs})")
    print(f"Mean Keypoint Confidence              : {mean_conf_overall:.4f} (median: {median_conf_overall:.4f})")
    print(f"Predicted-Mask Containment Sanity Rate : {mean_inside_all * 100:.2f}% (All keypoints)")
    print(f"Predicted-Mask Containment (Conf>=0.2) : {mean_inside_c02 * 100:.2f}%")
    print(f"Mean Frame-to-Frame Displacement      : {mean_disp_overall:.4f} (norm to 224px)")
    print(f"Skeleton Jumps (>0.25 norm disp)      : {total_jumps}")
    print("-" * 80)
    print("RESULTS BY SOURCE:")
    for src, s_data in by_source.items():
        print(
            f"  {src:22s} | Return: {s_data['pose_return_rate']*100:.1f}% | "
            f"All-8: {s_data['all_8_frames_rate']*100:.1f}% | Conf: {s_data['mean_keypoint_confidence']:.4f} | "
            f"MaskIn: {s_data['predicted_mask_containment_sanity_rate_all']*100:.1f}%"
        )
    print("-" * 80)
    print("RESULTS BY CANONICAL CLASS:")
    for cls, c_data in by_class.items():
        print(
            f"  {cls:12s} | Return: {c_data['pose_return_rate']*100:.1f}% | "
            f"All-8: {c_data['all_8_frames_rate']*100:.1f}% | Conf: {c_data['mean_keypoint_confidence']:.4f} | "
            f"MaskIn: {c_data['predicted_mask_containment_sanity_rate_all']*100:.1f}%"
        )
    print("=" * 80 + "\n")

    return aggregate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit frozen SuperAnimal-Quadruped on CVB+Beef Behavior")
    parser.add_argument(
        "--behavior-data-dir",
        type=Path,
        default=Path("/mtl-data/behavior"),
        help="Path to /mtl-data/behavior containing retained_train.csv and sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/behavior_pose_audit"),
        help="Directory to save audit outputs",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2026,
        help="Deterministic sampling seed",
    )
    parser.add_argument(
        "--seqs-per-cell",
        type=int,
        default=6,
        help="Number of sequences per available cell",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    audit_cvb_beef_pose(
        behavior_data_dir=args.behavior_data_dir,
        output_dir=args.output_dir,
        sample_seed=args.sample_seed,
        seqs_per_cell=args.seqs_per_cell,
        device_str=args.device,
    )
