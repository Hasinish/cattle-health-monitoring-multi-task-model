# -*- coding: utf-8 -*-
"""SideViewCows2026 Pose Feasibility Audit on Run 6 GT-Mask Crops.

Scientific Feasibility Assessment:
  Evaluates frozen DeepLabCut SuperAnimal-Quadruped ResNet-50 on the exact
  Run 6 GT-mask-derived cow crops (5% proportional margin) using a deterministic
  representative subset of Protocol D train/val images from the 41 training cows only.

Strict Evaluation Isolation:
  The 69 held-out Protocol A evaluation cows (gallery and barn/snapshots queries)
  are NEVER accessed, loaded, or inspected.

Caveat:
  SideViewCows2026 contains NO ground-truth keypoint annotations.
  Confidence scores and GT-mask containment rates are geometric sanity checks only,
  NOT pose accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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

try:
    from scripts.train_sideview_reid_perception import (
        _mask_bbox_with_margin,
        _validate_pair_paths,
        resolve_sideview_image_path,
    )
except ModuleNotFoundError:
    from train_sideview_reid_perception import (  # type: ignore
        _mask_bbox_with_margin,
        _validate_pair_paths,
        resolve_sideview_image_path,
    )


def check_keypoints_inside_mask(
    keypoints_crop: List[Tuple[float, float, float]],
    mask_np: np.ndarray,
) -> Tuple[float, int, int]:
    """Calculate the percentage of keypoints landing on foreground mask pixels.

    Geometric sanity check only; NOT ground-truth anatomical accuracy.
    """
    h, w = mask_np.shape[:2]
    inside = 0
    total = len(keypoints_crop)
    if total == 0:
        return 0.0, 0, 0

    for kx, ky, _ in keypoints_crop:
        ix = int(round(kx))
        iy = int(round(ky))
        if 0 <= ix < w and 0 <= iy < h:
            if mask_np[iy, ix] > 0:
                inside += 1
    return round(inside / total, 4), inside, total


def create_pose_contact_sheet(
    visual_records: List[Dict[str, Any]],
    output_path: Path,
    cols: int = 5,
) -> None:
    """Create a contact sheet showing Run 6 crops with SuperAnimal keypoints overlaid."""
    if not visual_records:
        return

    num_samples = len(visual_records)
    cols = min(cols, num_samples)
    rows = math.ceil(num_samples / cols)
    tile_w, tile_h = 240, 240
    header_h = 40
    margin = 8

    sheet_w = cols * (tile_w + margin) + margin
    sheet_h = rows * (tile_h + header_h + margin) + 50
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 25))
    draw = ImageDraw.Draw(sheet)

    # Title
    title = "SideViewCows2026 Pose Feasibility: SuperAnimal-Quadruped ResNet-50 on Run 6 GT-Mask Crops"
    draw.text((margin, 15), title, fill=(255, 255, 255))

    for idx, rec in enumerate(visual_records):
        r = idx // cols
        c = idx % cols
        x_pos = margin + c * (tile_w + margin)
        y_pos = 50 + r * (tile_h + header_h + margin)

        # Draw tile
        crop_img = rec["crop_rgb"].resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        crop_draw = ImageDraw.Draw(crop_img)

        # Draw keypoints on tile
        scale_x = tile_w / rec["crop_size"][0]
        scale_y = tile_h / rec["crop_size"][1]

        # Draw skeleton connections if present
        skeleton = rec.get("skeleton", [])
        kpts = rec["keypoints"]
        for i1, i2 in skeleton:
            if i1 < len(kpts) and i2 < len(kpts):
                x1, y1, c1 = kpts[i1]
                x2, y2, c2 = kpts[i2]
                if c1 >= 0.2 and c2 >= 0.2:
                    sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                    sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                    crop_draw.line([(sx1, sy1), (sx2, sy2)], fill=(0, 220, 255), width=2)

        # Draw points
        for kx, ky, conf in kpts:
            px = int(round(kx * scale_x))
            py = int(round(ky * scale_y))
            color = (0, 255, 0) if conf >= 0.2 else (255, 50, 50)
            rad = 3 if conf >= 0.2 else 2
            crop_draw.ellipse([(px - rad, py - rad), (px + rad, py + rad)], fill=color)

        sheet.paste(crop_img, (x_pos, y_pos + header_h))

        # Header info
        info = f"Cow {rec['cow_id']} | {rec['split'].upper()}\nConf: {rec['mean_conf']:.2f} | Inside: {rec['inside_rate']*100:.1f}%"
        draw.text((x_pos, y_pos + 4), info, fill=(200, 220, 240))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    print(f"[OK] Saved pose feasibility contact sheet: {output_path}")


def audit_pose_feasibility(
    data_root: Path,
    protocols_dir: Path,
    output_dir: Path,
    sample_size: int = 50,
    seed: int = 2026,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Run frozen SuperAnimal ResNet-50 feasibility audit on Run 6 crops."""
    import torch
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
        create_superanimal_inference_runners,
    )

    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("STEP B: SIDEVIEWCOWS2026 POSE FEASIBILITY AUDIT (SUPERANIMAL RESNET-50)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Sample Size: {sample_size} (Protocol D Train/Val strictly from 41 training cows)")
    print("=" * 76)

    # 1. Load protocols & enforce isolation
    proto_a_path = protocols_dir / "protocol_cross_setting.csv"
    proto_d_path = protocols_dir / "protocol_closed_set.csv"
    df_a = pd.read_csv(proto_a_path)
    df_d = pd.read_csv(proto_d_path)

    train_cows = sorted(df_a[df_a["setting_role"] == "train"]["individual_id"].astype(str).unique())
    eval_cows = sorted(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique())

    if len(train_cows) != 41 or len(eval_cows) != 69:
        raise AssertionError("Protocol cow counts mismatch")
    if set(train_cows).intersection(set(eval_cows)):
        raise AssertionError("Train and held-out evaluation cows overlap detected!")

    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train = df_d_41[df_d_41["closed_set_split"] == "train"]
    df_val = df_d_41[df_d_41["closed_set_split"] == "val"]

    # Sample deterministically across cows
    n_per_split = sample_size // 2
    sample_train = df_train.sample(n=n_per_split, random_state=seed).reset_index(drop=True)
    sample_val = df_val.sample(n=n_per_split, random_state=seed).reset_index(drop=True)
    sample_df = pd.concat([sample_train, sample_val], ignore_index=True)

    print(f"[OK] Deterministically sampled {len(sample_df)} crops:")
    print(f"     - Train crops: {len(sample_train)} (unique cows: {sample_train['individual_id'].nunique()})")
    print(f"     - Val crops:   {len(sample_val)} (unique cows: {sample_val['individual_id'].nunique()})")
    print(f"     - Overlap with 69 held-out cows: ZERO (0)")

    # 2. Initialize official SuperAnimal runners
    print("\n[*] Initializing SuperAnimal-Quadruped ResNet-50 inference pipeline...")
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
    bodyparts = model_cfg["metadata"]["bodyparts"]
    skeleton_raw = model_cfg.get("skeleton", [])
    skeleton_indices = []
    for b1, b2 in skeleton_raw:
        if b1 in bodyparts and b2 in bodyparts:
            skeleton_indices.append((bodyparts.index(b1), bodyparts.index(b2)))
    print(f"[OK] SuperAnimal initialized in {time.time() - init_t0:.2f}s ({len(bodyparts)} keypoints)")

    # 3. Feasibility execution loop
    results: List[Dict[str, Any]] = []
    visual_records: List[Dict[str, Any]] = []
    per_keypoint_conf: Dict[str, List[float]] = {bp: [] for bp in bodyparts}

    for idx, row in sample_df.iterrows():
        cow_id = str(row["individual_id"])
        split = str(row["closed_set_split"])
        raw_img = str(row["image_path"])
        raw_mask = str(row["mask_path"])

        img_path = resolve_sideview_image_path(raw_img, data_root)
        mask_path = resolve_sideview_image_path(raw_mask, data_root)
        if img_path is None or mask_path is None:
            continue

        _validate_pair_paths(img_path, mask_path, cow_id)
        with Image.open(img_path) as im:
            full_rgb = im.convert("RGB")
        with Image.open(mask_path) as mk:
            mask_gray = mk.convert("L")

        mask_np = np.asarray(mask_gray) > 0
        crop_box = _mask_bbox_with_margin(mask_np, margin_fraction=0.05)
        image_crop = full_rgb.crop(crop_box)
        mask_crop = mask_gray.crop(crop_box)
        mask_crop_np = np.asarray(mask_crop) > 0

        crop_w, crop_h = image_crop.size
        crop_rgb_np = np.asarray(image_crop)

        # Run SuperAnimal inference on crop
        t_infer_start = time.time()
        det_preds = det_runner.inference([crop_rgb_np]) if det_runner is not None else None
        has_box = False
        if det_preds is not None and len(det_preds) > 0:
            bboxes = det_preds[0].get("bboxes", [])
            scores = det_preds[0].get("bbox_scores", [])
            if len(bboxes) > 0 and (len(scores) == 0 or np.max(scores) > 0.0):
                has_box = True

        status = "pose_output_returned"
        keypoints_crop: List[Tuple[float, float, float]] = []
        confs: List[float] = []

        if not has_box and det_runner is not None:
            status = "pose_detector_failure"
        else:
            pose_inputs = [(crop_rgb_np, det_preds[0])] if det_preds is not None else [crop_rgb_np]
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
                            keypoints_crop.append((kx, ky, kc))
                            confs.append(kc)
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
                            keypoints_crop.append((kx, ky, kc))
                            confs.append(kc)
                            per_keypoint_conf[bp_name].append(kc)

        infer_ms = (time.time() - t_infer_start) * 1000

        # Geometric sanity check: points inside mask
        inside_rate, inside_count, total_pts = check_keypoints_inside_mask(keypoints_crop, mask_crop_np)
        mean_c = float(np.mean(confs)) if confs else 0.0
        med_c = float(np.median(confs)) if confs else 0.0
        pts_above_02 = sum(1 for c in confs if c >= 0.2)

        res = {
            "sample_index": idx,
            "cow_id": cow_id,
            "split": split,
            "status": status,
            "crop_size": [crop_w, crop_h],
            "inference_time_ms": round(infer_ms, 2),
            "num_keypoints": len(keypoints_crop),
            "mean_confidence": round(mean_c, 4),
            "median_confidence": round(med_c, 4),
            "points_above_02": pts_above_02,
            "keypoints_inside_mask_rate": inside_rate,
            "keypoints_inside_mask_count": inside_count,
        }
        results.append(res)

        if idx < 10 and status == "pose_output_returned":
            visual_records.append({
                "cow_id": cow_id,
                "split": split,
                "crop_rgb": image_crop,
                "crop_size": (crop_w, crop_h),
                "keypoints": keypoints_crop,
                "skeleton": skeleton_indices,
                "mean_conf": mean_c,
                "inside_rate": inside_rate,
            })

    # 4. Aggregate metrics
    df_res = pd.DataFrame(results)
    total_evaluated = len(df_res)
    returned_df = df_res[df_res["status"] == "pose_output_returned"]
    returned_count = len(returned_df)
    det_fail_count = len(df_res[df_res["status"] == "pose_detector_failure"])
    infer_err_count = len(df_res[df_res["status"] == "pose_inference_error"])

    mean_conf_all = float(returned_df["mean_confidence"].mean()) if returned_count > 0 else 0.0
    median_conf_all = float(returned_df["median_confidence"].median()) if returned_count > 0 else 0.0
    mean_inside_rate = float(returned_df["keypoints_inside_mask_rate"].mean()) if returned_count > 0 else 0.0

    # Per-keypoint stats
    kpt_stats: Dict[str, Dict[str, float]] = {}
    for bp, c_list in per_keypoint_conf.items():
        if c_list:
            kpt_stats[bp] = {
                "mean_confidence": round(float(np.mean(c_list)), 4),
                "pct_above_02": round(float(sum(1 for c in c_list if c >= 0.2) / len(c_list) * 100), 2),
            }

    # Save per-keypoint summary CSV
    kpt_df = pd.DataFrame([
        {"keypoint": bp, "mean_confidence": d["mean_confidence"], "pct_above_02": d["pct_above_02"]}
        for bp, d in kpt_stats.items()
    ])
    kpt_csv_path = output_dir / "pose_keypoints_summary.csv"
    kpt_df.to_csv(kpt_csv_path, index=False)

    # Save visual contact sheet
    contact_sheet_path = output_dir / "sideview_pose_contact_sheet.jpg"
    create_pose_contact_sheet(visual_records, contact_sheet_path)

    metrics_summary = {
        "dataset": "SideViewCows2026",
        "condition": "Run 6 GT-Mask Cow Crop (5% margin)",
        "pose_model": "DeepLabCut SuperAnimal-Quadruped ResNet-50",
        "detector_model": "fasterrcnn_resnet50_fpn_v2",
        "evaluation_subset": "Protocol D Train/Val strictly from 41 training cows",
        "total_evaluated": total_evaluated,
        "pose_output_returned": returned_count,
        "pose_output_returned_rate": round(returned_count / total_evaluated * 100, 2),
        "pose_detector_failures": det_fail_count,
        "pose_detector_failure_rate": round(det_fail_count / total_evaluated * 100, 2),
        "pose_inference_errors": infer_err_count,
        "mean_raw_confidence": round(mean_conf_all, 4),
        "median_raw_confidence": round(median_conf_all, 4),
        "mean_keypoints_inside_mask_rate": round(mean_inside_rate * 100, 2),
        "scientific_caveats": [
            "SideViewCows2026 provides NO keypoint ground truth.",
            "Confidence scores reflect model activation strength, NOT anatomical pose accuracy.",
            "Keypoints-inside-mask rate is a geometric sanity check only, confirming points land on cow silhouette.",
            "Canonical Protocol A held-out cows (69 identities) remained strictly untouched.",
        ],
        "top_5_confident_keypoints": sorted(kpt_stats.items(), key=lambda x: x[1]["mean_confidence"], reverse=True)[:5],
        "bottom_5_confident_keypoints": sorted(kpt_stats.items(), key=lambda x: x[1]["mean_confidence"])[:5],
        "contact_sheet_path": str(contact_sheet_path),
        "keypoints_csv_path": str(kpt_csv_path),
        "duration_seconds": round(time.time() - t0, 2),
    }

    metrics_json_path = output_dir / "pose_feasibility_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 76)
    print("STEP B AUDIT SUMMARY:")
    print(f"[*] Total Evaluated: {total_evaluated}")
    print(f"[*] Output Returned Rate: {metrics_summary['pose_output_returned_rate']}% ({returned_count}/{total_evaluated})")
    print(f"[*] Detector Failure Rate: {metrics_summary['pose_detector_failure_rate']}% ({det_fail_count}/{total_evaluated})")
    print(f"[*] Mean Raw Confidence: {metrics_summary['mean_raw_confidence']:.4f}")
    print(f"[*] Keypoints Inside GT Mask Rate (Sanity Check): {metrics_summary['mean_keypoints_inside_mask_rate']}%")
    print(f"[*] Saved Metrics JSON: {metrics_json_path}")
    print(f"[*] Saved Keypoints CSV: {kpt_csv_path}")
    print(f"[*] Saved Contact Sheet: {contact_sheet_path}")
    print("=" * 76 + "\n")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SideView Pose Feasibility Audit")
    parser.add_argument("--data-root", type=Path, default=Path("/data/sideviewcows2026"))
    parser.add_argument("--protocols-dir", type=Path, default=REPO_ROOT / "datasets" / "id" / "sideviewcows2026")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "reid_pose_ablation")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    audit_pose_feasibility(
        data_root=args.data_root,
        protocols_dir=args.protocols_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        device_str=args.device,
    )
