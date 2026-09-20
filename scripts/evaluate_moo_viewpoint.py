# -*- coding: utf-8 -*-
"""
evaluate_moo_viewpoint.py — Evaluate MOO-Trained ResNet-18 on Real Cattle Viewpoint Benchmark

Tests synthetic-to-real transfer:
  - Checkpoint: moo_resnet18_viewpoint.pth (trained on MOO synthetic dataset)
  - Benchmark: 100-sample cross-checked real cattle dataset (ScienceDB, MmCows, SideViewCows2026)
  - Target crops: Extracted using Step 2.1 RT-DETR-L bounding boxes (exact same protocol as zero-shot VLM audit)

Usage:
  python scripts/evaluate_moo_viewpoint.py
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "artifacts", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "moo_resnet18_viewpoint.pth")

MANIFEST_PATH = os.path.join(
    ROOT_DIR, "artifacts", "perception_audit", "viewpoint_expanded_agent_review_manifest.csv"
)
DETECTIONS_PATH = os.path.join(
    ROOT_DIR, "artifacts", "perception_audit", "localization_detections_expanded.csv"
)
OUTPUT_CSV_PATH = os.path.join(
    ROOT_DIR, "artifacts", "perception_audit", "viewpoint_moo_resnet18_evaluation.csv"
)

CLASS_NAMES = ["front", "front-oblique", "side", "rear-oblique", "rear"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def download_checkpoint_if_missing():
    """Download checkpoint from Modal volume if not present locally."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✓ Local checkpoint found: {CHECKPOINT_PATH} ({os.path.getsize(CHECKPOINT_PATH)/1024/1024:.1f} MB)")
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Downloading moo_resnet18_viewpoint.pth from Modal volume 'moo-data'...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        "modal", "volume", "get", "moo-data",
        "moo_resnet18_viewpoint.pth",
        CHECKPOINT_PATH,
        "--profile", "hasinishrak74001",
        "--force"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    if res.returncode != 0 or not os.path.exists(CHECKPOINT_PATH):
        print(f"Error downloading checkpoint: {res.stderr}")
        raise FileNotFoundError(f"Could not retrieve checkpoint from Modal volume.")
    print(f"✓ Checkpoint downloaded successfully to {CHECKPOINT_PATH} ({os.path.getsize(CHECKPOINT_PATH)/1024/1024:.1f} MB)")


def normalize_path(p: str) -> str:
    """Normalize file path for cross-platform matching."""
    if not isinstance(p, str):
        return ""
    p = p.replace("\\", "/").strip().lower()
    for prefix in ["d:/cattle-health-monitoring-multi-task-model/", "d:/"]:
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p


def load_detections(manifest_df: pd.DataFrame) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, str]]:
    """Recover primary cow bounding boxes from Step 2.1 RT-DETR-L detections."""
    if not os.path.exists(DETECTIONS_PATH):
        print(f"Warning: {DETECTIONS_PATH} not found. Using full image fallback.")
        return {}, {str(r["sample_id"]): "full_image_fallback" for _, r in manifest_df.iterrows()}

    det_df = pd.read_csv(DETECTIONS_PATH)
    rt = det_df[det_df["model_name"] == "RT-DETR-L"].copy()
    rt_valid = rt[rt["box_x1"].notna() & rt["box_y1"].notna() & rt["box_x2"].notna() & rt["box_y2"].notna()].copy()

    rt_valid["box_x1"] = pd.to_numeric(rt_valid["box_x1"], errors="coerce")
    rt_valid["box_y1"] = pd.to_numeric(rt_valid["box_y1"], errors="coerce")
    rt_valid["box_x2"] = pd.to_numeric(rt_valid["box_x2"], errors="coerce")
    rt_valid["box_y2"] = pd.to_numeric(rt_valid["box_y2"], errors="coerce")
    rt_valid["confidence"] = pd.to_numeric(rt_valid["confidence"], errors="coerce")
    rt_valid["area"] = (rt_valid["box_x2"] - rt_valid["box_x1"]) * (rt_valid["box_y2"] - rt_valid["box_y1"])
    rt_valid["norm_path"] = rt_valid["image_path"].apply(normalize_path)

    sorted_det = rt_valid.sort_values(by=["norm_path", "area", "confidence"], ascending=[True, False, False])
    primary_map: Dict[str, Dict[str, Any]] = {}
    for norm_p, grp in sorted_det.groupby("norm_path"):
        top = grp.iloc[0]
        primary_map[norm_p] = {
            "box": [float(top["box_x1"]), float(top["box_y1"]), float(top["box_x2"]), float(top["box_y2"])],
            "confidence": float(top["confidence"]),
        }

    sample_boxes: Dict[str, Optional[Dict[str, Any]]] = {}
    sample_modes: Dict[str, str] = {}
    for _, row in manifest_df.iterrows():
        s_id = str(row["sample_id"])
        norm_img_path = normalize_path(row["source_image_path"])
        box_info = primary_map.get(norm_img_path, None)
        sample_boxes[s_id] = box_info
        sample_modes[s_id] = "rtdetr_crop" if box_info is not None else "full_image_fallback"

    return sample_boxes, sample_modes


def get_image_crop(image_path: str, box_info: Optional[Dict[str, Any]]) -> Tuple[Image.Image, str]:
    """Load image and crop to primary cow bounding box if available."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    if box_info is not None and "box" in box_info:
        box = box_info["box"]
        x1 = max(0, int(round(box[0])))
        y1 = max(0, int(round(box[1])))
        x2 = min(w, int(round(box[2])))
        y2 = min(h, int(round(box[3])))
        if x2 > x1 and y2 > y1:
            crop_bgr = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(crop_rgb), "rtdetr_crop"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), "full_image_fallback"


def main():
    print("=" * 70)
    print("  MOO Synthetic-to-Real Viewpoint Transfer Benchmark Evaluation")
    print("=" * 70)

    download_checkpoint_if_missing()

    # Load model
    print("\n[1/4] Loading trained ResNet-18 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = models.resnet18()
    model.fc = nn.Linear(512, 5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    val_acc_synthetic = checkpoint.get("val_acc", 0.0)
    print(f"✓ Model loaded successfully (Synthetic Val Acc was: {val_acc_synthetic*100:.1f}%)")

    # Load benchmark manifest
    print("\n[2/4] Loading real cattle benchmark manifest (N=100)...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"✓ Loaded {len(manifest_df)} samples from {MANIFEST_PATH}")

    sample_boxes, sample_modes = load_detections(manifest_df)
    rtdetr_count = sum(1 for m in sample_modes.values() if m == "rtdetr_crop")
    print(f"✓ Target crops recovered: {rtdetr_count}/{len(manifest_df)} (RT-DETR-L boxes)")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Run inference
    print("\n[3/4] Running inference on 100 real cattle crops...")
    results = []
    start_time = time.time()

    for idx, row in manifest_df.iterrows():
        s_id = str(row["sample_id"])
        img_path = str(row["source_image_path"])
        gt_view = str(row["final_viewpoint"]).strip()
        dataset = str(row["dataset"])
        box_info = sample_boxes.get(s_id, None)

        crop_img, input_mode = get_image_crop(img_path, box_info)
        tensor_img = transform(crop_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_img)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(-probs)
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]

        top1_class = CLASS_NAMES[top1_idx]
        top1_score = float(probs[top1_idx])
        top2_class = CLASS_NAMES[top2_idx]
        top2_score = float(probs[top2_idx])
        margin = top1_score - top2_score

        # Method B Rejection threshold (tau=0.10)
        is_rejected = margin < 0.10
        pred_with_rejection = "unknown / ambiguous" if is_rejected else top1_class

        results.append({
            "sample_id": s_id,
            "dataset": dataset,
            "ground_truth": gt_view,
            "raw_physical_pred": top1_class,
            "raw_physical_score": top1_score,
            "second_pred": top2_class,
            "second_score": top2_score,
            "score_margin": margin,
            "pred_with_margin_rejection": pred_with_rejection,
            "is_rejected": is_rejected,
            "input_mode": input_mode,
            "prob_front": probs[0],
            "prob_front_oblique": probs[1],
            "prob_side": probs[2],
            "prob_rear_oblique": probs[3],
            "prob_rear": probs[4],
        })

    elapsed = time.time() - start_time
    print(f"✓ Inference complete in {elapsed:.2f}s ({elapsed*1000/len(manifest_df):.1f} ms/sample)!")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✓ Saved detailed evaluation CSV to {OUTPUT_CSV_PATH}")

    # [4/4] Compute Metrics & Comparisons
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS & COMPARATIVE BENCHMARK")
    print("=" * 70)

    # 1. Non-ambiguous physical evaluation (N=95)
    non_ambig = res_df[res_df["ground_truth"] != "unknown / ambiguous"].copy()
    y_true_non = non_ambig["ground_truth"]
    y_pred_raw = non_ambig["raw_physical_pred"]

    raw_acc = accuracy_score(y_true_non, y_pred_raw)
    raw_f1 = f1_score(y_true_non, y_pred_raw, average="macro", zero_division=0)
    false_fronts = (non_ambig["raw_physical_pred"] == "front").sum()

    print(f"\n--- Method B: Raw 5-Class Physical Evaluation (N={len(non_ambig)}) ---")
    print(f"  MOO ResNet-18 Accuracy:  {raw_acc*100:.2f}%")
    print(f"  MOO ResNet-18 Macro-F1:  {raw_f1:.4f}")
    print(f"  False Front Predictions: {false_fronts} (True fronts in GT = 0)")

    # Comparison with Zero-Shot VLMs from audit
    print("\n--- Comparative Scoreboard (Raw 5-Class Physical, N=95) ---")
    print(f"  {'Model':<28} | {'Accuracy':<10} | {'Macro-F1':<10} | {'False Fronts':<12}")
    print(f"  {'-'*28}-|-{'-'*10}-|-{'-'*10}-|-{'-'*12}")
    print(f"  {'OpenAI CLIP (ViT-B/32)':<28} | 33.68%     | 0.2711     | 19 / 95")
    print(f"  {'OpenCLIP (LAION-2B)':<28} | 15.79%     | 0.1149     | 13 / 95")
    print(f"  {'Google SigLIP':<28} | 12.63%     | 0.0958     | 66 / 95")
    print(f"  {'MOO ResNet-18 (Ours)':<28} | {raw_acc*100:5.2f}%     | {raw_f1:6.4f}     | {false_fronts:2d} / 95")

    # Per-class breakdown
    print("\n--- Per-Class Classification Report (MOO ResNet-18) ---")
    print(classification_report(y_true_non, y_pred_raw, labels=CLASS_NAMES, zero_division=0))

    # Confusion Matrix
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true_non, y_pred_raw, labels=CLASS_NAMES)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
    print(cm_df)

    # Per-dataset accuracy
    print("\n--- Per-Dataset Physical Accuracy ---")
    for dset, grp in non_ambig.groupby("dataset"):
        d_acc = accuracy_score(grp["ground_truth"], grp["raw_physical_pred"])
        print(f"  {dset:<18}: {d_acc*100:5.2f}% ({len(grp)} samples)")

    # Ambiguous evaluation
    ambig_samples = res_df[res_df["ground_truth"] == "unknown / ambiguous"]
    ambig_recalled = (ambig_samples["pred_with_margin_rejection"] == "unknown / ambiguous").sum()
    print(f"\n--- Ambiguous Margin Rejection (tau=0.10) ---")
    print(f"  Ambiguous Recall: {ambig_recalled}/{len(ambig_samples)} ({ambig_recalled/max(1, len(ambig_samples))*100:.1f}%)")

    # [5/5] Systematic 90-Degree Rotation & Coordinate Frame Search
    print("\n" + "=" * 70)
    print("  SYSTEMATIC 90-DEGREE ROTATION & AXIS-SWAP SEARCH")
    print("=" * 70)
    print("Testing if Blender's 3D cow model was oriented sideways (90° rotated)...")

    candidate_mappings = {
        "Original (0° shift)": lambda p: p,
        "90° Shift (Front->Side, Side->Rear, Rear->Side)": lambda p: (
            "side" if p in ["front", "rear"] else ("rear" if p == "side" else p)
        ),
        "90° Shift (Front/Rear <-> Side, Obliques Swapped)": lambda p: (
            "side" if p in ["front", "rear"] else (
                "rear" if p == "side" else (
                    "rear-oblique" if p == "front-oblique" else "front-oblique"
                )
            )
        ),
        "Full 90° Clockwise Rotation": lambda p: {
            "front": "side", "front-oblique": "rear-oblique",
            "side": "rear", "rear-oblique": "front-oblique", "rear": "front"
        }.get(p, p),
        "Full 90° Counter-Clockwise Rotation": lambda p: {
            "front": "rear", "front-oblique": "front-oblique",
            "side": "front", "rear-oblique": "rear-oblique", "rear": "side"
        }.get(p, p),
        "180° Flip (Front <-> Rear)": lambda p: {
            "front": "rear", "front-oblique": "rear-oblique",
            "side": "side", "rear-oblique": "front-oblique", "rear": "front"
        }.get(p, p),
    }

    print(f"\n  {'Mapping / Coordinate Hypothesis':<52} | {'Accuracy':<10} | {'Macro-F1':<10}")
    print(f"  {'-'*52}-|-{'-'*10}-|-{'-'*10}")
    best_name = "Original"
    best_acc = raw_acc
    best_f1 = raw_f1
    best_preds = y_pred_raw

    for name, map_fn in candidate_mappings.items():
        mapped_pred = y_pred_raw.apply(map_fn)
        acc = accuracy_score(y_true_non, mapped_pred)
        f1 = f1_score(y_true_non, mapped_pred, average="macro", zero_division=0)
        print(f"  {name:<52} | {acc*100:5.2f}%     | {f1:6.4f}")
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_name = name
            best_preds = mapped_pred

    print(f"\n🏆 Best Coordinate Alignment: {best_name}")
    print(f"   Accuracy: {best_acc*100:.2f}% | Macro-F1: {best_f1:.4f}")

    if best_acc > raw_acc:
        print("\n--- Best Alignment Confusion Matrix ---")
        best_cm = confusion_matrix(y_true_non, best_preds, labels=CLASS_NAMES)
        best_cm_df = pd.DataFrame(best_cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
        print(best_cm_df)


if __name__ == "__main__":
    main()
