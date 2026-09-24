# -*- coding: utf-8 -*-
"""SideViewCows2026 Viewpoint Transfer Sanity Audit on Run 6 GT-Mask Crops.

Scientific Feasibility Assessment:
  Evaluates the frozen certified real-cattle viewpoint classifier
  (viewpoint_resnet18_real_best.pth) on the exact Run 6 GT-mask-derived cow crops
  (5% margin) using a deterministic representative subset of Protocol D train/val
  images from the 41 training cows only.

Strict Evaluation Isolation:
  The 69 held-out Protocol A evaluation cows (parlor gallery, barn queries,
  snapshot queries) are NEVER accessed, loaded, or inspected.

Important Scientific Caveat:
  SideViewCows2026 does NOT provide verified viewpoint ground truth.
  Camera IDs and subset names must NEVER be treated as viewpoint labels.
  This audit measures predicted viewpoint distributions, confidence, and entropy
  as a CROSS-DOMAIN TRANSFER SANITY CHECK, NOT an accuracy benchmark.
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

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF

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
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
except ModuleNotFoundError:
    from train_sideview_reid_perception import (  # type: ignore
        _mask_bbox_with_margin,
        _validate_pair_paths,
        resolve_sideview_image_path,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

CLASS_NAMES = ["front", "side", "rear"]
EXPECTED_CHECKPOINT_HASH = "a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a"


def load_frozen_viewpoint_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load and freeze certified 3-class viewpoint ResNet-18 model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Viewpoint checkpoint not found: {checkpoint_path}")

    model = models.resnet18()
    model.fc = nn.Linear(512, 3)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch! missing={missing}, unexpected={unexpected}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    print(f"[OK] Viewpoint model loaded from {checkpoint_path.name} on {device}")
    return model


def create_viewpoint_contact_sheet(
    records: List[Dict[str, Any]],
    output_path: Path,
    summary_stats: Dict[str, Any],
    cols: int = 5,
) -> None:
    """Create a contact sheet showing Run 6 crops with predicted viewpoint & confidence."""
    if not records:
        return

    num_samples = len(records)
    cols = min(cols, num_samples)
    rows = math.ceil(num_samples / cols)
    tile_w, tile_h = 240, 240
    header_h = 50
    margin = 8

    top_banner_h = 75
    sheet_w = cols * (tile_w + margin) + margin
    sheet_h = top_banner_h + rows * (tile_h + header_h + margin) + margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 22, 28))
    draw = ImageDraw.Draw(sheet)

    # Top Banner
    title = "SideViewCows2026 Viewpoint Transfer Sanity Audit (Frozen ResNet-18)"
    draw.text((margin + 4, 10), title, fill=(255, 255, 255))
    dist_str = (
        f"Distribution: Front={summary_stats['front_pct']:.1f}% | "
        f"Side={summary_stats['side_pct']:.1f}% | "
        f"Rear={summary_stats['rear_pct']:.1f}%  |  "
        f"Mean Conf={summary_stats['mean_confidence']*100:.1f}% | "
        f"Low-Conf (<50%)={summary_stats['low_confidence_pct_50']:.1f}%"
    )
    draw.text((margin + 4, 32), dist_str, fill=(180, 210, 240))
    note_str = "Cross-domain transfer sanity check on Run 6 GT-mask crops (Protocol D Train/Val, 41 cows). NOT ground truth accuracy."
    draw.text((margin + 4, 52), note_str, fill=(140, 160, 180))

    # Tiled crops
    for idx, rec in enumerate(records):
        r = idx // cols
        c = idx % cols
        x_pos = margin + c * (tile_w + margin)
        y_pos = top_banner_h + r * (tile_h + header_h + margin)

        crop_img = rec["crop_rgb"].resize((tile_w, tile_h), Image.Resampling.BILINEAR)

        # Border color based on confidence & prediction
        conf = rec["confidence"]
        pred = rec["predicted_class"]
        if conf >= 0.70:
            border_color = (40, 180, 80) if pred == "side" else (60, 140, 220)
        elif conf >= 0.50:
            border_color = (220, 180, 40)
        else:
            border_color = (220, 60, 60)

        # Draw border
        crop_draw = ImageDraw.Draw(crop_img)
        crop_draw.rectangle([(0, 0), (tile_w - 1, tile_h - 1)], outline=border_color, width=3)

        # Paste crop
        sheet.paste(crop_img, (x_pos, y_pos + header_h))

        # Header Info Card
        card_bg = (30, 34, 42)
        draw.rectangle([(x_pos, y_pos), (x_pos + tile_w, y_pos + header_h - 4)], fill=card_bg)
        
        info_line1 = f"Cow: {rec['cow_id']} | {rec['split'].upper()} | {rec['subset']}"
        info_line2 = f"Pred: {pred.upper()} ({conf*100:.1f}%) | H={rec['entropy']:.2f}"
        probs_line = f"F:{rec['p_front']*100:.0f}% S:{rec['p_side']*100:.0f}% R:{rec['p_rear']*100:.0f}%"

        draw.text((x_pos + 4, y_pos + 3), info_line1, fill=(200, 210, 220))
        pred_color = (100, 255, 120) if pred == "side" else (120, 200, 255) if pred == "front" else (255, 180, 100)
        draw.text((x_pos + 4, y_pos + 18), info_line2, fill=pred_color)
        draw.text((x_pos + 4, y_pos + 33), probs_line, fill=(150, 160, 170))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(f"[OK] Saved viewpoint transfer contact sheet: {output_path}")


def audit_viewpoint_transfer(
    data_root: Path,
    protocols_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    sample_size: int = 50,
    seed: int = 2026,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Run viewpoint cross-domain sanity audit on Run 6 GT-mask crops."""
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("STEP C: SIDEVIEWCOWS2026 VIEWPOINT CROSS-DOMAIN TRANSFER SANITY AUDIT")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Sample Size   : {sample_size} (Protocol D Train/Val strictly from 41 training cows)")
    print(f"Seed          : {seed}")
    print(f"Checkpoint    : {checkpoint_path}")
    print("=" * 78)

    # 1. Protocols & Isolation Check
    proto_a_path = protocols_dir / "protocol_cross_setting.csv"
    proto_d_path = protocols_dir / "protocol_closed_set.csv"
    if not proto_a_path.exists() or not proto_d_path.exists():
        raise FileNotFoundError(f"Missing protocol files in {protocols_dir}")

    df_a = pd.read_csv(proto_a_path)
    df_d = pd.read_csv(proto_d_path)

    train_cows = sorted(df_a[df_a["setting_role"] == "train"]["individual_id"].astype(str).unique())
    eval_cows = sorted(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique())

    if len(train_cows) != 41 or len(eval_cows) != 69:
        raise AssertionError(f"Expected 41 train cows and 69 eval cows, got {len(train_cows)} and {len(eval_cows)}")
    overlap = set(train_cows).intersection(set(eval_cows))
    if overlap:
        raise AssertionError(f"CRITICAL: Train and eval cows overlap! {overlap}")

    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train = df_d_41[df_d_41["closed_set_split"] == "train"]
    df_val = df_d_41[df_d_41["closed_set_split"] == "val"]

    n_per_split = sample_size // 2
    sample_train = df_train.sample(n=n_per_split, random_state=seed).reset_index(drop=True)
    sample_val = df_val.sample(n=n_per_split, random_state=seed).reset_index(drop=True)
    sample_df = pd.concat([sample_train, sample_val], ignore_index=True)

    print(f"[OK] Deterministically sampled {len(sample_df)} crops across 41 training cows:")
    print(f"     - Train crops: {len(sample_train)} (unique cows: {sample_train['individual_id'].nunique()})")
    print(f"     - Val crops  : {len(sample_val)} (unique cows: {sample_val['individual_id'].nunique()})")
    print(f"     - Overlap with 69 held-out evaluation cows: ZERO (0)")

    # 2. Load Frozen Viewpoint Classifier
    model = load_frozen_viewpoint_model(checkpoint_path, device)

    # 3. Inference Loop
    records: List[Dict[str, Any]] = []
    visual_records: List[Dict[str, Any]] = []

    pred_counts = {"front": 0, "side": 0, "rear": 0}
    confidences: List[float] = []
    entropies: List[float] = []
    low_conf_50_count = 0
    low_conf_60_count = 0

    print("\n[*] Running frozen viewpoint inference on Run 6 crops...")
    for idx, row in sample_df.iterrows():
        cow_id = str(row["individual_id"])
        split = str(row["closed_set_split"])
        subset = str(row.get("subset", ""))
        raw_img = str(row["image_path"])
        raw_mask = str(row["mask_path"])

        img_path = resolve_sideview_image_path(raw_img, data_root)
        mask_path = resolve_sideview_image_path(raw_mask, data_root)
        if img_path is None or mask_path is None or not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing pair: {raw_img} | {raw_mask}")

        _validate_pair_paths(img_path, mask_path, cow_id)

        with Image.open(img_path) as im:
            full_rgb = im.convert("RGB")
        with Image.open(mask_path) as im:
            full_mask = im.convert("L")

        mask_np = np.asarray(full_mask) > 0
        crop_box = _mask_bbox_with_margin(mask_np, margin_fraction=0.05)
        crop_rgb = full_rgb.crop(crop_box)

        # Preprocess exactly matching viewpoint model: resize to 224x224, ImageNet norm
        crop_resized = crop_rgb.resize((224, 224), Image.Resampling.BILINEAR)
        tensor_rgb = TF.to_tensor(crop_resized)
        tensor_norm = TF.normalize(tensor_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_norm)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        p_front, p_side, p_rear = float(probs[0]), float(probs[1]), float(probs[2])
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

        # Shannon Entropy in nats: H = - sum(p * log(p + 1e-12))
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        pred_counts[pred_class] += 1
        confidences.append(conf)
        entropies.append(entropy)

        is_low_50 = conf < 0.50
        is_low_60 = conf < 0.60
        if is_low_50:
            low_conf_50_count += 1
        if is_low_60:
            low_conf_60_count += 1

        rec = {
            "index": idx,
            "cow_id": cow_id,
            "split": split,
            "subset": subset,
            "image_filename": img_path.name,
            "crop_box_xyxy": list(crop_box),
            "predicted_class": pred_class,
            "confidence": round(conf, 4),
            "entropy": round(entropy, 4),
            "p_front": round(p_front, 4),
            "p_side": round(p_side, 4),
            "p_rear": round(p_rear, 4),
            "is_low_conf_50": is_low_50,
            "is_low_conf_60": is_low_60,
        }
        records.append(rec)

        visual_rec = dict(rec)
        visual_rec["crop_rgb"] = crop_rgb
        visual_records.append(visual_rec)

    # 4. Aggregate Summary
    total_samples = len(records)
    summary_stats = {
        "sample_size": total_samples,
        "seed": seed,
        "device": str(device),
        "source_checkpoint": checkpoint_path.name,
        "protocol_d_cows_audited": len(train_cows),
        "held_out_protocol_a_cows_touched": 0,
        "prediction_distribution": {
            "front_count": pred_counts["front"],
            "front_pct": round(pred_counts["front"] / total_samples * 100, 2),
            "side_count": pred_counts["side"],
            "side_pct": round(pred_counts["side"] / total_samples * 100, 2),
            "rear_count": pred_counts["rear"],
            "rear_pct": round(pred_counts["rear"] / total_samples * 100, 2),
        },
        "front_pct": round(pred_counts["front"] / total_samples * 100, 2),
        "side_pct": round(pred_counts["side"] / total_samples * 100, 2),
        "rear_pct": round(pred_counts["rear"] / total_samples * 100, 2),
        "confidence_statistics": {
            "mean": round(float(np.mean(confidences)), 4),
            "std": round(float(np.std(confidences)), 4),
            "median": round(float(np.median(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "max": round(float(np.max(confidences)), 4),
        },
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "entropy_statistics": {
            "mean": round(float(np.mean(entropies)), 4),
            "std": round(float(np.std(entropies)), 4),
            "median": round(float(np.median(entropies)), 4),
            "min": round(float(np.min(entropies)), 4),
            "max": round(float(np.max(entropies)), 4),
        },
        "low_confidence_pct_50": round(low_conf_50_count / total_samples * 100, 2),
        "low_confidence_pct_60": round(low_conf_60_count / total_samples * 100, 2),
        "low_confidence_count_50": low_conf_50_count,
        "low_confidence_count_60": low_conf_60_count,
        "subset_breakdown": {},
    }

    # Breakdown by subset (parlor / barn / snapshots)
    subsets = set(r["subset"] for r in records if r["subset"])
    for s in subsets:
        s_recs = [r for r in records if r["subset"] == s]
        if s_recs:
            s_confs = [r["confidence"] for r in s_recs]
            s_preds = [r["predicted_class"] for r in s_recs]
            summary_stats["subset_breakdown"][s] = {
                "count": len(s_recs),
                "mean_conf": round(float(np.mean(s_confs)), 4),
                "front_count": s_preds.count("front"),
                "side_count": s_preds.count("side"),
                "rear_count": s_preds.count("rear"),
            }

    # 5. Save Artifacts
    csv_path = output_dir / "viewpoint_transfer_samples.csv"
    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False)
    print(f"[OK] Saved per-sample CSV: {csv_path}")

    json_path = output_dir / "viewpoint_transfer_sanity_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"[OK] Saved sanity metrics JSON: {json_path}")

    sheet_path = output_dir / "viewpoint_transfer_contact_sheet.jpg"
    create_viewpoint_contact_sheet(visual_records, sheet_path, summary_stats, cols=5)

    print("\n" + "=" * 78)
    print("STEP C AUDIT SUMMARY (CROSS-DOMAIN TRANSFER SANITY CHECK)")
    print(f"Evaluated Samples  : {total_samples}")
    print(f"Predicted Front    : {summary_stats['front_pct']}% ({pred_counts['front']})")
    print(f"Predicted Side     : {summary_stats['side_pct']}% ({pred_counts['side']})")
    print(f"Predicted Rear     : {summary_stats['rear_pct']}% ({pred_counts['rear']})")
    print(f"Mean Confidence    : {summary_stats['confidence_statistics']['mean']:.4f} (median: {summary_stats['confidence_statistics']['median']:.4f})")
    print(f"Mean Entropy       : {summary_stats['entropy_statistics']['mean']:.4f}")
    print(f"Low-Conf Rate (<50%): {summary_stats['low_confidence_pct_50']}% ({low_conf_50_count}/{total_samples})")
    print(f"Elapsed Time       : {time.time() - t0:.2f}s")
    print("=" * 78)

    return summary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SideView Viewpoint Transfer Sanity Audit")
    parser.add_argument("--data-root", type=str, default="/data/sideviewcows2026")
    parser.add_argument("--protocols-dir", type=str, default="/root/datasets/id/sideviewcows2026")
    parser.add_argument("--checkpoint-path", type=str, default="/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth")
    parser.add_argument("--output-dir", type=str, default="/checkpoints/sideview_viewpoint_sanity")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    audit_viewpoint_transfer(
        data_root=Path(args.data_root),
        protocols_dir=Path(args.protocols_dir),
        checkpoint_path=Path(args.checkpoint_path),
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
        seed=args.seed,
        device_str=args.device,
    )
