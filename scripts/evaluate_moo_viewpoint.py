# -*- coding: utf-8 -*-
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
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
DEFAULT_L4_CKPT = os.path.join(CHECKPOINT_DIR, "moo_resnet18_viewpoint8_full_l4.pth")
LEGACY_5CLS_CKPT = os.path.join(CHECKPOINT_DIR, "moo_resnet18_viewpoint.pth")

MANIFEST_PATH = os.path.join(
    ROOT_DIR, "artifacts", "perception_audit", "viewpoint_expanded_agent_review_manifest.csv"
)
DETECTIONS_PATH = os.path.join(
    ROOT_DIR, "artifacts", "perception_audit", "localization_detections_expanded.csv"
)

COARSE_CLASSES = ["front", "front-oblique", "side", "rear-oblique", "rear"]
COARSE_TO_IDX = {name: idx for idx, name in enumerate(COARSE_CLASSES)}

DIR8_TO_COARSE5 = {
    "front": "front",
    "front-left": "front-oblique",
    "front-right": "front-oblique",
    "left": "side",
    "right": "side",
    "back-left": "rear-oblique",
    "back-right": "rear-oblique",
    "back": "rear",
}


def resolve_checkpoint_path() -> Tuple[str, str]:
    """Resolve checkpoint path from CLI or defaults."""
    ckpt_path = DEFAULT_L4_CKPT
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--checkpoint":
            ckpt_path = sys.argv[i + 1]
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_path)

    if not os.path.exists(ckpt_path):
        if os.path.exists(LEGACY_5CLS_CKPT):
            ckpt_path = LEGACY_5CLS_CKPT
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path} or {LEGACY_5CLS_CKPT}. "
                f"Download from Modal volume 'moo-data' first."
            )

    is_l4 = "l4" in os.path.basename(ckpt_path).lower() or "viewpoint8" in os.path.basename(ckpt_path).lower()
    out_csv = os.path.join(
        ROOT_DIR,
        "artifacts",
        "perception_audit",
        "viewpoint_moo_resnet18_l4_evaluation.csv" if is_l4 else "viewpoint_moo_resnet18_evaluation.csv",
    )
    return ckpt_path, out_csv


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

    ckpt_path, output_csv_path = resolve_checkpoint_path()
    print(f"Checkpoint Path: {ckpt_path}")
    print(f"Output CSV Path: {output_csv_path}")

    # Load model
    print("\n[1/4] Loading trained ResNet-18 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", COARSE_CLASSES)
    num_classes = len(class_names)
    is_8class = num_classes == 8

    model = models.resnet18()
    model.fc = nn.Linear(512, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    val_acc_syn = checkpoint.get("best_val_accuracy", checkpoint.get("val_acc", 0.0))
    val_f1_syn = checkpoint.get("best_val_macro_f1", 0.0)
    test_acc_syn = checkpoint.get("test_accuracy", 0.0)
    test_f1_syn = checkpoint.get("test_macro_f1", 0.0)

    print(f"✓ Model loaded successfully ({num_classes} classes: {class_names})")
    print(f"  Synthetic Best Val Acc: {val_acc_syn*100:.2f}% | Val Macro-F1: {val_f1_syn:.4f}")
    if test_acc_syn > 0:
        print(f"  Synthetic Test Acc:     {test_acc_syn*100:.2f}% | Test Macro-F1: {test_f1_syn:.4f} (9,600 images / 100 cow IDs)")

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

        top1_idx = int(np.argmax(probs))
        top1_class = class_names[top1_idx]
        top1_score = float(probs[top1_idx])

        if is_8class:
            coarse_pred = DIR8_TO_COARSE5[top1_class]
            results.append({
                "sample_id": s_id,
                "dataset": dataset,
                "ground_truth": gt_view,
                "pred8": top1_class,
                "pred5": coarse_pred,
                "conf": top1_score,
                "input_mode": input_mode,
            })
        else:
            sorted_indices = np.argsort(-probs)
            top2_idx = sorted_indices[1]
            margin = top1_score - float(probs[top2_idx])
            pred_with_rejection = "unknown / ambiguous" if margin < 0.10 else top1_class
            results.append({
                "sample_id": s_id,
                "dataset": dataset,
                "ground_truth": gt_view,
                "raw_physical_pred": top1_class,
                "raw_physical_score": top1_score,
                "pred_with_margin_rejection": pred_with_rejection,
                "input_mode": input_mode,
            })

    elapsed = time.time() - start_time
    print(f"✓ Inference complete in {elapsed:.2f}s ({elapsed*1000/len(manifest_df):.1f} ms/sample)!")

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv_path, index=False)
    print(f"✓ Saved detailed evaluation CSV to {output_csv_path}")

    # [4/4] Compute Metrics & Comparisons
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (DIAGNOSTIC BENCHMARK)")
    print("=" * 70)

    non_ambig = res_df[res_df["ground_truth"] != "unknown / ambiguous"].copy()
    y_true_non = non_ambig["ground_truth"]
    y_pred_coarse = non_ambig["pred5"] if is_8class else non_ambig["raw_physical_pred"]

    raw_acc = accuracy_score(y_true_non, y_pred_coarse)
    raw_f1 = f1_score(y_true_non, y_pred_coarse, average="macro", zero_division=0)
    false_fronts = int((y_pred_coarse == "front").sum())

    print(f"\n--- Coarse 5-Class Diagnostic Evaluation (N={len(non_ambig)} Non-Ambiguous) ---")
    print(f"  Accuracy:                {raw_acc*100:.2f}% ({accuracy_score(y_true_non, y_pred_coarse, normalize=False)}/{len(non_ambig)})")
    print(f"  Macro-F1:                {raw_f1:.4f}")
    print(f"  False Front Predictions: {false_fronts} (True fronts in GT = 0)")

    print("\n--- Per-Dataset Accuracy ---")
    for dset, grp in non_ambig.groupby("dataset"):
        grp_pred = grp["pred5"] if is_8class else grp["raw_physical_pred"]
        d_acc = accuracy_score(grp["ground_truth"], grp_pred)
        print(f"  {dset:<18}: {d_acc*100:5.2f}% ({len(grp)} samples)")

    print("\n--- Classification Report ---")
    print(classification_report(y_true_non, y_pred_coarse, labels=COARSE_CLASSES, zero_division=0))

    print("--- Confusion Matrix ---")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)
    cm = confusion_matrix(y_true_non, y_pred_coarse, labels=COARSE_CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in COARSE_CLASSES], columns=[f"Pred_{c}" for c in COARSE_CLASSES])
    print(cm_df)

    if is_8class:
        print("\n--- Predicted 8-Direction Distribution on Real Cattle ---")
        print(res_df["pred8"].value_counts())


if __name__ == "__main__":
    main()

