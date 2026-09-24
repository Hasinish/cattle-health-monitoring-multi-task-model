# -*- coding: utf-8 -*-
"""CVB + Kaggle Beef Behavior Viewpoint Transfer Sanity Audit.

Scientific Assessment:
  Evaluates the frozen certified real-cattle 3-class viewpoint classifier
  (viewpoint_resnet18_real_best.pth, test accuracy 86.26% on source domain)
  across authentic CVB + Kaggle Beef Behavior T=8 sequences staged under /mtl-data/behavior/.

Transfer Sanity Scope:
  SideView, CVB, and Kaggle Beef do NOT provide viewpoint ground truth annotations.
  Metrics reported here are TRANSFER SANITY METRICS, NOT VIEWPOINT ACCURACY.
  Evaluates:
    - Predicted class distributions (front, side, rear)
    - Softmax confidence and prediction entropy distributions
    - Temporal consistency across 8 frames of each sequence
    - Degeneracy checks (class collapse vs healthy spread)
    - Source/camera shortcut risk
    - Visual multi-frame contact sheets across all 9 source x class cells.

Strict Evaluation Isolation:
  Evaluates ONLY staged train/val sequences from retained_train.csv and retained_val.csv.
  The held-out Behavior test set is STRICTLY UNTOUCHED.
"""

from __future__ import annotations

import argparse
import hashlib
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

VIEWPOINT_CLASSES = ["front", "side", "rear"]
EXPECTED_CHECKPOINT_HASH = "a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

VIEWPOINT_COLORS = {
    "front": (30, 144, 255),   # Dodger Blue
    "side": (46, 204, 113),    # Emerald Green
    "rear": (231, 76, 60),     # Alizarin Red
}


def compute_entropy_nats(probs: np.ndarray, eps: float = 1e-9) -> float:
    """Compute Shannon entropy in nats: -sum(p * ln(p))."""
    p_clamped = np.clip(probs, eps, 1.0)
    return float(-np.sum(p_clamped * np.log(p_clamped)))


def load_frozen_viewpoint_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load and freeze certified 3-class viewpoint ResNet-18 model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Viewpoint checkpoint not found: {checkpoint_path}")

    # Verify SHA-256
    with open(checkpoint_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    if file_hash != EXPECTED_CHECKPOINT_HASH:
        raise ValueError(
            f"Viewpoint checkpoint SHA-256 mismatch! Expected {EXPECTED_CHECKPOINT_HASH}, got {file_hash}"
        )
    print(f"[OK] Verified viewpoint checkpoint SHA-256 bit identity: {file_hash}")

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
    print(f"[OK] Frozen Viewpoint model loaded on {device}")
    return model


def draw_viewpoint_tile(
    rgb_img: Image.Image,
    pred_cls: str,
    probs: np.ndarray,
    entropy: float,
    header_text: str,
    tile_size: int = 224,
) -> Image.Image:
    """Renders a single frame tile with color-coded viewpoint badge and probability breakdown."""
    tile = rgb_img.resize((tile_size, tile_size), Image.Resampling.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(tile)

    # Viewpoint badge color bar at bottom
    badge_color = VIEWPOINT_COLORS.get(pred_cls, (200, 200, 200))
    bar_h = 24
    draw.rectangle([0, tile_size - bar_h, tile_size, tile_size], fill=(20, 20, 25))
    draw.rectangle([0, tile_size - 4, tile_size, tile_size], fill=badge_color)

    # Probabilities bar text: F:XX% | S:XX% | R:XX%
    prob_text = f"F:{probs[0]*100:.0f}% S:{probs[1]*100:.0f}% R:{probs[2]*100:.0f}%"
    draw.text((6, tile_size - bar_h + 4), prob_text, fill=(230, 230, 240))

    # Top header banner
    header_h = 28
    tile_with_header = Image.new("RGB", (tile_size, tile_size + header_h), (25, 25, 30))
    h_draw = ImageDraw.Draw(tile_with_header)
    h_draw.text((4, 6), header_text, fill=badge_color)
    tile_with_header.paste(tile, (0, header_h))

    return tile_with_header


def generate_viewpoint_contact_sheet(
    rep_sequences: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: Path,
    tile_size: int = 224,
) -> None:
    """Generates a 9-row visual contact sheet displaying representative sequences for all cells."""
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

    title = "Phase 3 Behavior Viewpoint Transfer Sanity Audit: Frozen Real Viewpoint ResNet-18 on CVB + Beef (T=8)"
    subtitle = "Transfer Sanity Overlays: Predicted Viewpoint [Front: Blue, Side: Green, Rear: Red] | Confidence & Shannon Entropy"
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
        frames = seq_data["frames"]

        src_label = "CVB" if src == "cvb" else "Beef"
        row_text = f"{src_label} - {cls}\n{split.upper()}\n{sample_id[:16]}..."
        draw.text((margin + 10, y_pos + 20), row_text, fill=(240, 240, 240))

        for col_idx in range(min(num_cols, len(frames))):
            f_data = frames[col_idx]
            x_pos = row_label_w + col_idx * (tile_size + margin)

            header_text = (
                f"t={f_data['t']} | {f_data['pred_cls'].upper()} ({f_data['conf']*100:.0f}%) | "
                f"H:{f_data['entropy']:.2f}"
            )
            tile_img = draw_viewpoint_tile(
                rgb_img=f_data["rgb_pil"],
                pred_cls=f_data["pred_cls"],
                probs=f_data["probs"],
                entropy=f_data["entropy"],
                header_text=header_text,
                tile_size=tile_size,
            )
            sheet.paste(tile_img, (x_pos, y_pos))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(f"[OK] Saved comprehensive 9-cell viewpoint contact sheet: {output_path}")


def audit_cvb_beef_viewpoint(
    behavior_data_dir: Path,
    viewpoint_checkpoint_path: Path,
    output_dir: Path,
    sample_seed: int = 2026,
    seqs_per_cell: int = 6,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Runs frozen real-cattle viewpoint transfer sanity audit on CVB + Beef Behavior."""
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  PHASE 3: CVB + KAGGLE BEEF BEHAVIOR VIEWPOINT TRANSFER SANITY AUDIT")
    print("  Model : Frozen Certified Real-Cattle Viewpoint ResNet-18 (3 Classes)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Data  : {behavior_data_dir}")
    print(f"  Checkpoint: {viewpoint_checkpoint_path}")
    print("=" * 80)

    # 1. Load Model
    vp_model = load_frozen_viewpoint_model(viewpoint_checkpoint_path, device)

    # 2. Load Manifests
    train_csv = behavior_data_dir / "retained_train.csv"
    val_csv = behavior_data_dir / "retained_val.csv"
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # 3. Deterministic Stratified Sampling
    sampled_records: List[Dict[str, Any]] = []
    print(f"\n[*] Sampling {seqs_per_cell} sequences per available source x class cell (seed {sample_seed}):")

    for src, cls in AVAILABLE_CELLS:
        train_pool = df_train[
            (df_train["dataset"] == src) & (df_train["behavior_canonical"] == cls)
        ].copy()
        val_pool = df_val[
            (df_val["dataset"] == src) & (df_val["behavior_canonical"] == cls)
        ].copy()

        n_train_actual = min(len(train_pool), 4)
        n_val_actual = min(len(val_pool), 2)
        if n_val_actual < 2:
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

    print(f"[OK] Total sampled sequences for audit: {len(sampled_records)} (Total frames = {len(sampled_records) * 8})")

    # 4. Inference Loop
    frame_results: List[Dict[str, Any]] = []
    seq_summaries: List[Dict[str, Any]] = []
    rep_visual_sequences: Dict[Tuple[str, str], Dict[str, Any]] = {}

    infer_start_time = time.time()

    for seq_idx, s_rec in enumerate(sampled_records):
        sample_id = str(s_rec["sample_id"])
        src = str(s_rec["dataset"])
        cls = str(s_rec["behavior_canonical"])
        split = str(s_rec["split"])
        cell_key = (src, cls)

        seq_dir = behavior_data_dir / sample_id
        if not seq_dir.exists():
            continue

        seq_frame_data: List[Dict[str, Any]] = []
        seq_preds: List[str] = []
        seq_confs: List[float] = []
        seq_entropies: List[float] = []
        seq_probs: List[np.ndarray] = []

        # Batch all 8 frames together for fast GPU evaluation
        batch_tensors = []
        pil_images = []
        for t in range(8):
            f_img_p = seq_dir / f"frame_{t:02d}.jpg"
            with Image.open(f_img_p) as im:
                rgb_pil = im.convert("RGB")
            pil_images.append(rgb_pil)
            t_tensor = TF.to_tensor(rgb_pil)
            t_norm = TF.normalize(t_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            batch_tensors.append(t_norm)

        inp_batch = torch.stack(batch_tensors, dim=0).to(device)  # [8, 3, 224, 224]

        with torch.no_grad():
            logits = vp_model(inp_batch)
            probs_batch = F.softmax(logits, dim=-1).cpu().numpy()  # [8, 3]

        for t in range(8):
            probs = probs_batch[t]
            pred_idx = int(np.argmax(probs))
            pred_cls = VIEWPOINT_CLASSES[pred_idx]
            conf = float(probs[pred_idx])
            entropy = compute_entropy_nats(probs)

            seq_preds.append(pred_cls)
            seq_confs.append(conf)
            seq_entropies.append(entropy)
            seq_probs.append(probs)

            f_rec = {
                "dataset": src,
                "behavior_canonical": cls,
                "split": split,
                "sample_id": sample_id,
                "t": t,
                "pred_viewpoint": pred_cls,
                "confidence": round(conf, 4),
                "entropy_nats": round(entropy, 4),
                "prob_front": round(float(probs[0]), 4),
                "prob_side": round(float(probs[1]), 4),
                "prob_rear": round(float(probs[2]), 4),
                "low_confidence_flag": (conf < 0.5),
            }
            frame_results.append(f_rec)

            seq_frame_data.append({
                "t": t,
                "rgb_pil": pil_images[t],
                "pred_cls": pred_cls,
                "probs": probs,
                "conf": conf,
                "entropy": entropy,
            })

        # Sequence-level Temporal Consistency Analysis
        transitions = 7
        stable_transitions = sum(1 for t in range(7) if seq_preds[t] == seq_preds[t + 1])
        transition_stability_rate = round(stable_transitions / float(transitions), 4)

        # Number of viewpoint class switches across sequence
        class_switches = sum(1 for t in range(7) if seq_preds[t] != seq_preds[t + 1])

        # Variance of probability across the 8 frames
        probs_arr = np.array(seq_probs)  # [8, 3]
        mean_p_var = float(np.mean(np.var(probs_arr, axis=0)))

        seq_summary = {
            "sample_id": sample_id,
            "dataset": src,
            "behavior_canonical": cls,
            "split": split,
            "majority_viewpoint": max(set(seq_preds), key=seq_preds.count),
            "pct_front": round(float(np.mean([p == "front" for p in seq_preds])), 4),
            "pct_side": round(float(np.mean([p == "side" for p in seq_preds])), 4),
            "pct_rear": round(float(np.mean([p == "rear" for p in seq_preds])), 4),
            "mean_confidence": round(float(np.mean(seq_confs)), 4),
            "mean_entropy_nats": round(float(np.mean(seq_entropies)), 4),
            "transition_stability_rate": transition_stability_rate,
            "class_switches_count": class_switches,
            "temporal_probability_variance": round(mean_p_var, 6),
            "is_temporally_constant": (class_switches == 0),
        }
        seq_summaries.append(seq_summary)

        if cell_key not in rep_visual_sequences:
            rep_visual_sequences[cell_key] = {
                "sample_id": sample_id,
                "split": split,
                "frames": seq_frame_data,
            }

    total_infer_time = time.time() - infer_start_time
    print(f"[OK] Completed viewpoint inference in {total_infer_time:.2f}s ({total_infer_time / len(frame_results):.4f}s/frame)")

    # 5. Save Visual Contact Sheet
    contact_sheet_path = output_dir / "behavior_viewpoint_all_classes_contact_sheet.jpg"
    generate_viewpoint_contact_sheet(rep_visual_sequences, contact_sheet_path)

    # 6. Save CSVs
    df_frames = pd.DataFrame(frame_results)
    df_seqs = pd.DataFrame(seq_summaries)

    frame_csv_path = output_dir / "behavior_viewpoint_frame_results.csv"
    seq_csv_path = output_dir / "behavior_viewpoint_sequence_summary.csv"
    df_frames.to_csv(frame_csv_path, index=False)
    df_seqs.to_csv(seq_csv_path, index=False)
    print(f"[OK] Saved frame-level CSV ({len(df_frames)} rows): {frame_csv_path}")
    print(f"[OK] Saved sequence-level CSV ({len(df_seqs)} rows): {seq_csv_path}")

    # 7. Aggregate Metrics & Non-Degeneracy Analysis
    total_frames = len(df_frames)
    total_seqs = len(df_seqs)

    overall_class_counts = df_frames["pred_viewpoint"].value_counts().to_dict()
    overall_dist = {
        cls: round(float(overall_class_counts.get(cls, 0)) / float(total_frames), 4)
        for cls in VIEWPOINT_CLASSES
    }

    mean_conf_overall = round(float(df_frames["confidence"].mean()), 4)
    median_conf_overall = round(float(df_frames["confidence"].median()), 4)
    mean_entropy_overall = round(float(df_frames["entropy_nats"].mean()), 4)
    low_conf_rate_overall = round(float(df_frames["low_confidence_flag"].mean()), 4)

    mean_stability_overall = round(float(df_seqs["transition_stability_rate"].mean()), 4)
    temporally_constant_seqs_rate = round(float(df_seqs["is_temporally_constant"].mean()), 4)
    mean_class_switches = round(float(df_seqs["class_switches_count"].mean()), 2)

    # Check Non-Degeneracy:
    # A degenerate distribution is where one class is >=95% or any class is 0.0%
    is_non_degenerate = bool(
        all(v > 0.02 for v in overall_dist.values()) and max(overall_dist.values()) < 0.90
    )

    # Breakdown by Source
    by_source: Dict[str, Any] = {}
    for src in ["cvb", "beef_cattle_behavior"]:
        f_sub = df_frames[df_frames["dataset"] == src]
        s_sub = df_seqs[df_seqs["dataset"] == src]
        c_counts = f_sub["pred_viewpoint"].value_counts().to_dict()
        by_source[src] = {
            "total_sequences": len(s_sub),
            "total_frames": len(f_sub),
            "distribution": {
                cls: round(float(c_counts.get(cls, 0)) / float(len(f_sub)), 4)
                for cls in VIEWPOINT_CLASSES
            },
            "mean_confidence": round(float(f_sub["confidence"].mean()), 4),
            "mean_entropy_nats": round(float(f_sub["entropy_nats"].mean()), 4),
            "low_confidence_rate": round(float(f_sub["low_confidence_flag"].mean()), 4),
            "mean_temporal_stability": round(float(s_sub["transition_stability_rate"].mean()), 4),
            "temporally_constant_rate": round(float(s_sub["is_temporally_constant"].mean()), 4),
            "mean_class_switches": round(float(s_sub["class_switches_count"].mean()), 2),
        }

    # Breakdown by Canonical Class
    by_class: Dict[str, Any] = {}
    for cls in CANONICAL_CLASSES:
        f_sub = df_frames[df_frames["behavior_canonical"] == cls]
        s_sub = df_seqs[df_seqs["behavior_canonical"] == cls]
        c_counts = f_sub["pred_viewpoint"].value_counts().to_dict()
        by_class[cls] = {
            "total_sequences": len(s_sub),
            "total_frames": len(f_sub),
            "distribution": {
                c: round(float(c_counts.get(c, 0)) / float(len(f_sub)), 4)
                for c in VIEWPOINT_CLASSES
            },
            "mean_confidence": round(float(f_sub["confidence"].mean()), 4),
            "mean_entropy_nats": round(float(f_sub["entropy_nats"].mean()), 4),
            "low_confidence_rate": round(float(f_sub["low_confidence_flag"].mean()), 4),
            "mean_temporal_stability": round(float(s_sub["transition_stability_rate"].mean()), 4),
            "temporally_constant_rate": round(float(s_sub["is_temporally_constant"].mean()), 4),
            "mean_class_switches": round(float(s_sub["class_switches_count"].mean()), 2),
        }

    # Breakdown by Source x Class Cells (9 cells)
    by_cell: Dict[str, Any] = {}
    for src, cls in AVAILABLE_CELLS:
        cell_key = f"{src}_{cls}"
        f_sub = df_frames[(df_frames["dataset"] == src) & (df_frames["behavior_canonical"] == cls)]
        s_sub = df_seqs[(df_seqs["dataset"] == src) & (df_seqs["behavior_canonical"] == cls)]
        c_counts = f_sub["pred_viewpoint"].value_counts().to_dict()
        by_cell[cell_key] = {
            "source": src,
            "class": cls,
            "total_sequences": len(s_sub),
            "total_frames": len(f_sub),
            "distribution": {
                c: round(float(c_counts.get(c, 0)) / float(len(f_sub)), 4)
                for c in VIEWPOINT_CLASSES
            },
            "mean_confidence": round(float(f_sub["confidence"].mean()), 4),
            "mean_entropy_nats": round(float(f_sub["entropy_nats"].mean()), 4),
            "low_confidence_rate": round(float(f_sub["low_confidence_flag"].mean()), 4),
            "mean_temporal_stability": round(float(s_sub["transition_stability_rate"].mean()), 4),
            "temporally_constant_rate": round(float(s_sub["is_temporally_constant"].mean()), 4),
            "mean_class_switches": round(float(s_sub["class_switches_count"].mean()), 2),
        }

    # Possible Source / Camera Shortcut Risk Assessment
    cvb_dist = by_source["cvb"]["distribution"]
    beef_dist = by_source["beef_cattle_behavior"]["distribution"]
    # Compute Total Variation Distance between CVB and Beef distributions
    tvd_source = 0.5 * sum(abs(cvb_dist[c] - beef_dist[c]) for c in VIEWPOINT_CLASSES)
    shortcut_risk_level = (
        "HIGH" if tvd_source > 0.40 else ("MODERATE" if tvd_source > 0.20 else "LOW")
    )

    aggregate_metrics: Dict[str, Any] = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_name": "Frozen Real-Cattle Viewpoint ResNet-18 (3 Classes: front, side, rear)",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_HASH,
        "sample_seed": sample_seed,
        "sample_size_sequences": int(total_seqs),
        "sample_size_frames": int(total_frames),
        "overall": {
            "predicted_distribution": overall_dist,
            "mean_confidence": mean_conf_overall,
            "median_confidence": median_conf_overall,
            "mean_entropy_nats": mean_entropy_overall,
            "low_confidence_rate_lt_05": low_conf_rate_overall,
            "mean_temporal_transition_stability": mean_stability_overall,
            "temporally_constant_sequence_rate": temporally_constant_seqs_rate,
            "mean_class_switches_per_sequence": mean_class_switches,
            "is_non_degenerate": is_non_degenerate,
            "source_shortcut_risk_level": shortcut_risk_level,
            "source_tvd": round(tvd_source, 4),
        },
        "by_source": by_source,
        "by_canonical_class": by_class,
        "by_source_x_class_cells": by_cell,
    }

    def _json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_path = output_dir / "behavior_viewpoint_transfer_sanity_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2, default=_json_default)
    print(f"[OK] Saved aggregate metrics JSON: {json_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("  AUDIT SUMMARY: CVB + KAGGLE BEEF VIEWPOINT TRANSFER SANITY")
    print("=" * 80)
    print(f"Overall Frames Evaluated         : {total_frames} (54 sequences x 8 frames)")
    print(f"Predicted Distribution           : Front: {overall_dist['front']*100:.1f}%, Side: {overall_dist['side']*100:.1f}%, Rear: {overall_dist['rear']*100:.1f}%")
    print(f"Mean Confidence / Median         : {mean_conf_overall:.4f} / {median_conf_overall:.4f}")
    print(f"Mean Entropy                     : {mean_entropy_overall:.4f} nats (max 1.0986)")
    print(f"Low-Confidence Rate (<50%)       : {low_conf_rate_overall*100:.1f}%")
    print(f"Temporal Transition Stability    : {mean_stability_overall*100:.1f}% (adjacent frame agreement)")
    print(f"Temporally Constant Sequences    : {temporally_constant_seqs_rate*100:.1f}% (zero class flips across 8 frames)")
    print(f"Non-Degenerate Check             : {'PASS (Healthy multi-class spread)' if is_non_degenerate else 'FAIL (Collapsed)'}")
    print(f"Source Shortcut Risk Level       : {shortcut_risk_level} (TVD = {tvd_source:.4f})")
    print("-" * 80)
    print("RESULTS BY SOURCE:")
    for src, s_data in by_source.items():
        d = s_data["distribution"]
        print(
            f"  {src:22s} | F:{d['front']*100:4.1f}% S:{d['side']*100:4.1f}% R:{d['rear']*100:4.1f}% | "
            f"Conf: {s_data['mean_confidence']:.4f} | Stab: {s_data['mean_temporal_stability']*100:.1f}%"
        )
    print("-" * 80)
    print("RESULTS BY CANONICAL CLASS:")
    for cls, c_data in by_class.items():
        d = c_data["distribution"]
        print(
            f"  {cls:12s} | F:{d['front']*100:4.1f}% S:{d['side']*100:4.1f}% R:{d['rear']*100:4.1f}% | "
            f"Conf: {c_data['mean_confidence']:.4f} | Stab: {c_data['mean_temporal_stability']*100:.1f}%"
        )
    print("=" * 80 + "\n")

    return aggregate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit frozen viewpoint on CVB+Beef Behavior")
    parser.add_argument(
        "--behavior-data-dir",
        type=Path,
        default=Path("/mtl-data/behavior"),
        help="Path to /mtl-data/behavior",
    )
    parser.add_argument(
        "--viewpoint-checkpoint",
        type=Path,
        default=Path("/mtl-checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth"),
        help="Path to viewpoint checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/behavior_viewpoint_audit"),
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

    audit_cvb_beef_viewpoint(
        behavior_data_dir=args.behavior_data_dir,
        viewpoint_checkpoint_path=args.viewpoint_checkpoint,
        output_dir=args.output_dir,
        sample_seed=args.sample_seed,
        seqs_per_cell=args.seqs_per_cell,
        device_str=args.device,
    )
