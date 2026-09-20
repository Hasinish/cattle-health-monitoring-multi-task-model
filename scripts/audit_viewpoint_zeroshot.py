"""
audit_viewpoint_zeroshot.py — Comparative Zero-Shot VLM Audit for Cattle Viewpoint

Benchmarks multiple frozen vision-language models on the 60 human-verified cattle viewpoint samples:
  1. openai_clip:     openai/clip-vit-base-patch32 (OpenAI CLIP)
  2. openclip_laion:  laion/CLIP-ViT-B-32-laion2B-s34B-b79K (OpenCLIP LAION-2B)
  3. google_siglip:   google/siglip-base-patch16-224 (Google SigLIP)

Features:
  - Strict visual-only inference (zero metadata access).
  - Target-cow crops recovered from Step 2.1 RT-DETR-L detections (59 matched, 1 fallback).
  - Standardized prompt ensembles across all 6 candidate classes.
  - Method A (explicit ambiguous prompt) vs. Method B (confidence/margin rejection).
  - Metrics: Overall Acc, Macro-F1, Balanced Acc, Per-dataset Acc, Per-class PRF, False Fronts, Ambiguous Recall.
  - Live single-line progress bar with ETA, resumable, and smoke-test mode.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# 1. Constants & Viewpoint Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY_CLASSES = [
    "rear",
    "rear-oblique",
    "side",
    "front-oblique",
    "front",
    "unknown / ambiguous",
]

PROMPT_ENSEMBLES = {
    "rear": [
        "a rear view of a cow",
        "a cow seen from behind",
        "the back of a cow facing away from the camera",
        "a caudal view of cattle showing rump and tailhead",
    ],
    "rear-oblique": [
        "a three-quarter rear view of a cow",
        "a cow viewed diagonally from behind",
        "a rear oblique view of a cow showing flank and hindquarters",
        "a cow angled away from the camera",
    ],
    "side": [
        "a side profile of a cow",
        "a lateral view of a cow",
        "a cow viewed broadside",
        "a full side view of a cow standing or walking",
    ],
    "front-oblique": [
        "a three-quarter front view of a cow",
        "a cow viewed diagonally from the front",
        "a front oblique view of a cow showing head and flank",
        "a cow angled toward the camera",
    ],
    "front": [
        "a front view of a cow",
        "a cow facing directly toward the camera",
        "a cranial view of a cow head and chest",
        "the face and front of a cow looking at the camera",
    ],
    "unknown / ambiguous": [
        "an occluded or ambiguous view of a cow",
        "a blurry, obstructed, or unidentifiable angle of a cow",
        "a cow partially hidden behind stall bars or walls",
        "an ambiguous top-down or heavily occluded view of cattle",
    ],
}

MODEL_CONFIGS = {
    "openai_clip": {
        "family": "OpenAI CLIP",
        "model_id": "openai/clip-vit-base-patch32",
        "type": "clip",
    },
    "openclip_laion": {
        "family": "OpenCLIP LAION-2B",
        "model_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "type": "clip",
    },
    "google_siglip": {
        "family": "Google SigLIP",
        "model_id": "google/siglip-base-patch16-224",
        "type": "siglip",
    },
}


# ---------------------------------------------------------------------------
# 2. Input Preparation: RT-DETR-L Crop Recovery
# ---------------------------------------------------------------------------

def load_primary_cow_boxes(
    localization_csv: str, manifest_df: pd.DataFrame
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Load Step 2.1 RT-DETR-L primary cow bounding box for each sample."""
    if not os.path.exists(localization_csv):
        print(f"[WARN] Localization CSV not found: {localization_csv}. Using full images.")
        return {str(s_id): None for s_id in manifest_df["sample_id"]}

    df_det = pd.read_csv(localization_csv)
    rt = df_det[df_det["model_name"] == "RT-DETR-L"].copy()
    rt_valid = rt[rt["detection_status"] == "detected"].copy()

    rt_valid["box_x1"] = pd.to_numeric(rt_valid["box_x1"], errors="coerce")
    rt_valid["box_y1"] = pd.to_numeric(rt_valid["box_y1"], errors="coerce")
    rt_valid["box_x2"] = pd.to_numeric(rt_valid["box_x2"], errors="coerce")
    rt_valid["box_y2"] = pd.to_numeric(rt_valid["box_y2"], errors="coerce")
    rt_valid["confidence"] = pd.to_numeric(rt_valid["confidence"], errors="coerce")

    rt_valid["area"] = (rt_valid["box_x2"] - rt_valid["box_x1"]) * (
        rt_valid["box_y2"] - rt_valid["box_y1"]
    )
    sorted_det = rt_valid.sort_values(
        by=["sample_id", "area", "confidence"], ascending=[True, False, False]
    )

    primary_map: Dict[str, Dict[str, Any]] = {}
    for s_id, grp in sorted_det.groupby("sample_id"):
        top = grp.iloc[0]
        primary_map[str(s_id)] = {
            "box": [
                float(top["box_x1"]),
                float(top["box_y1"]),
                float(top["box_x2"]),
                float(top["box_y2"]),
            ],
            "confidence": float(top["confidence"]),
        }

    sample_boxes: Dict[str, Optional[Dict[str, Any]]] = {}
    for _, row in manifest_df.iterrows():
        s_id = str(row["sample_id"])
        sample_boxes[s_id] = primary_map.get(s_id, None)

    return sample_boxes


def get_image_crop(
    image_path: str, box_info: Optional[Dict[str, Any]]
) -> Tuple[Image.Image, str]:
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

    # Fallback to full image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), "full_image_fallback"


# ---------------------------------------------------------------------------
# 3. Model Wrapper Classes
# ---------------------------------------------------------------------------

def extract_pooled_features(out: Any) -> torch.Tensor:
    """Extract tensor from raw tensor or Hugging Face BaseModelOutputWithPooling."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0, :]
    return out[0]


class ZeroShotViewpointClassifier:
    """Base class for zero-shot vision-language viewpoint classifiers."""

    def __init__(self, key: str, config: Dict[str, Any], device: str):
        self.key = key
        self.config = config
        self.device = device
        self.model = None
        self.processor = None
        self.text_embeddings = None
        self._load()

    def _load(self):
        raise NotImplementedError

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        raise NotImplementedError


class CLIPViewpointClassifier(ZeroShotViewpointClassifier):
    """Wrapper for OpenAI CLIP and OpenCLIP models via Hugging Face."""

    def _load(self):
        model_id = self.config["model_id"]
        print(f"Loading {self.config['family']} from {model_id} onto {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # Precompute normalized prompt ensemble text embeddings for each class
        self.class_text_embeddings = []
        with torch.no_grad():
            for c in TAXONOMY_CLASSES:
                prompts = PROMPT_ENSEMBLES[c]
                inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = extract_pooled_features(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Ensemble average
                mean_feat = text_features.mean(dim=0, keepdim=True)
                mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
                self.class_text_embeddings.append(mean_feat)
            self.class_text_embeddings = torch.cat(self.class_text_embeddings, dim=0)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = extract_pooled_features(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarities
            sims = (image_features @ self.class_text_embeddings.T).squeeze(0)  # (6,)
            # Temperature scaling matching standard CLIP logit scale (100)
            logits = sims * self.model.logit_scale.exp().clamp(max=100.0)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        dt_ms = (time.perf_counter() - t0) * 1000.0

        scores = {TAXONOMY_CLASSES[i]: float(probs[i]) for i in range(len(TAXONOMY_CLASSES))}
        sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1_class, top1_score = sorted_classes[0]
        top2_class, top2_score = sorted_classes[1]
        margin = top1_score - top2_score

        return {
            "predicted_viewpoint": top1_class,
            "top_score": top1_score,
            "second_class": top2_class,
            "second_score": top2_score,
            "score_margin": margin,
            "runtime_ms": dt_ms,
            "all_scores": scores,
        }


class SigLIPViewpointClassifier(ZeroShotViewpointClassifier):
    """Wrapper for Google SigLIP models via Hugging Face."""

    def _load(self):
        model_id = self.config["model_id"]
        print(f"Loading {self.config['family']} from {model_id} onto {self.device}...")
        self.model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Precompute normalized prompt ensemble text embeddings for each class
        self.class_text_embeddings = []
        with torch.no_grad():
            for c in TAXONOMY_CLASSES:
                prompts = PROMPT_ENSEMBLES[c]
                inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = extract_pooled_features(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_feat = text_features.mean(dim=0, keepdim=True)
                mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
                self.class_text_embeddings.append(mean_feat)
            self.class_text_embeddings = torch.cat(self.class_text_embeddings, dim=0)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = extract_pooled_features(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ self.class_text_embeddings.T).squeeze(0)  # (6,)
            # SigLIP pairwise logit scale & bias if present
            if hasattr(self.model, "logit_scale"):
                logits = sims * self.model.logit_scale.exp()
                if hasattr(self.model, "logit_bias"):
                    logits = logits + self.model.logit_bias
            else:
                logits = sims * 10.0
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        dt_ms = (time.perf_counter() - t0) * 1000.0

        scores = {TAXONOMY_CLASSES[i]: float(probs[i]) for i in range(len(TAXONOMY_CLASSES))}
        sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1_class, top1_score = sorted_classes[0]
        top2_class, top2_score = sorted_classes[1]
        margin = top1_score - top2_score

        return {
            "predicted_viewpoint": top1_class,
            "top_score": top1_score,
            "second_class": top2_class,
            "second_score": top2_score,
            "score_margin": margin,
            "runtime_ms": dt_ms,
            "all_scores": scores,
        }


def create_classifier(key: str, device: str) -> ZeroShotViewpointClassifier:
    cfg = MODEL_CONFIGS[key]
    if cfg["type"] == "clip":
        return CLIPViewpointClassifier(key, cfg, device)
    elif cfg["type"] == "siglip":
        return SigLIPViewpointClassifier(key, cfg, device)
    else:
        raise ValueError(f"Unknown model type: {cfg['type']}")


# ---------------------------------------------------------------------------
# 4. Metrics Evaluation & Reporting
# ---------------------------------------------------------------------------

def evaluate_model_results(df_res: pd.DataFrame, model_key: str) -> Dict[str, Any]:
    """Compute comprehensive performance metrics for a single model run."""
    y_true = df_res["human_viewpoint"].tolist()
    y_pred = df_res["predicted_viewpoint"].tolist()

    # Supported ground-truth classes in the 60 samples
    support_classes = [c for c in TAXONOMY_CLASSES if c in set(y_true)]

    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=support_classes, average="macro", zero_division=0)

    # Per-dataset accuracy
    per_dataset_acc = {}
    for ds in ["ScienceDB", "MmCows", "SideViewCows2026"]:
        sub = df_res[df_res["dataset"] == ds]
        if len(sub) > 0:
            per_dataset_acc[ds] = accuracy_score(sub["human_viewpoint"], sub["predicted_viewpoint"])
        else:
            per_dataset_acc[ds] = 0.0

    # False front predictions (ground truth has 0 front)
    false_fronts = int((df_res["predicted_viewpoint"] == "front").sum())

    # Ambiguous recall (ground truth has 5 ambiguous)
    amb_df = df_res[df_res["human_viewpoint"] == "unknown / ambiguous"]
    amb_recall = (
        float((amb_df["predicted_viewpoint"] == "unknown / ambiguous").mean())
        if len(amb_df) > 0
        else 0.0
    )

    # Confidence and margins
    mean_top_score = float(df_res["top_score"].mean())
    mean_margin = float(df_res["score_margin"].mean())
    mean_runtime_ms = float(df_res["runtime_ms"].mean())

    # Per-class metrics
    clf_report = classification_report(
        y_true, y_pred, labels=support_classes, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=TAXONOMY_CLASSES)

    return {
        "model_key": model_key,
        "family": MODEL_CONFIGS[model_key]["family"],
        "model_id": MODEL_CONFIGS[model_key]["model_id"],
        "n_samples": len(df_res),
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "sciencedb_acc": per_dataset_acc.get("ScienceDB", 0.0),
        "mmcows_acc": per_dataset_acc.get("MmCows", 0.0),
        "sideview_acc": per_dataset_acc.get("SideViewCows2026", 0.0),
        "false_front_count": false_fronts,
        "ambiguous_recall": amb_recall,
        "mean_top1_score": mean_top_score,
        "mean_score_margin": mean_margin,
        "mean_runtime_ms": mean_runtime_ms,
        "classification_report": clf_report,
        "confusion_matrix": cm,
    }


def print_evaluation_summary(metrics: Dict[str, Any]):
    """Print clean terminal scorecard for a model."""
    print("\n" + "=" * 78)
    print(f"  ZERO-SHOT EVALUATION SCORECARD: {metrics['family']} ({metrics['model_id']})")
    print("=" * 78)
    print(f"Samples Evaluated:       {metrics['n_samples']} / 60")
    print(f"Overall Accuracy:        {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Macro-F1 (support > 0):  {metrics['macro_f1']:.4f}")
    print(f"Per-Dataset Accuracy:")
    print(f"  - ScienceDB:           {metrics['sciencedb_acc'] * 100:.2f}%")
    print(f"  - MmCows:              {metrics['mmcows_acc'] * 100:.2f}%")
    print(f"  - SideViewCows2026:    {metrics['sideview_acc'] * 100:.2f}%")
    print(f"False 'front' Count:     {metrics['false_front_count']} (Ground truth has 0 front)")
    print(f"Ambiguous Case Recall:   {metrics['ambiguous_recall'] * 100:.2f}% (5 true ambiguous)")
    print(f"Mean Top-1 Score:        {metrics['mean_top1_score']:.4f}")
    print(f"Mean Top-1 Margin:       {metrics['mean_score_margin']:.4f}")
    print(f"Mean Runtime / Image:    {metrics['mean_runtime_ms']:.1f} ms")

    print("\nConfusion Matrix (Rows = Ground Truth, Cols = Predicted):")
    header = f"{'GT \\ Pred':<22}" + "".join([f"{c[:10]:>11}" for c in TAXONOMY_CLASSES])
    print(header)
    print("-" * len(header))
    cm = metrics["confusion_matrix"]
    for i, c in enumerate(TAXONOMY_CLASSES):
        row_str = f"{c:<22}" + "".join([f"{cm[i, j]:>11}" for j in range(len(TAXONOMY_CLASSES))])
        print(row_str)
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# 5. Main Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Comparative Zero-Shot VLM Audit for Cattle Viewpoint"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="artifacts/perception_audit/viewpoint_manual_review_manifest.csv",
        help="Path to viewpoint review manifest",
    )
    parser.add_argument(
        "--localization-csv",
        type=str,
        default="artifacts/perception_audit/localization_detections_expanded.csv",
        help="Path to Step 2.1 localization detections CSV",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai_clip,openclip_laion,google_siglip",
        help="Comma-separated list of model keys to run",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test on 2 samples (1 ScienceDB, 1 MmCows) to verify pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/perception_audit",
        help="Directory to save per-model CSVs and comparison CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running already completed model CSVs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest_df = pd.read_csv(args.manifest)
    print(f"[OK] Loaded manifest: {len(manifest_df)} rows from {args.manifest}")

    if args.smoke:
        print("[INFO] SMOKE TEST MODE: Selecting 2 representative samples.")
        manifest_df = manifest_df[
            manifest_df["sample_id"].isin(["sample_0001", "sample_0104"])
        ].copy()

    # Load RT-DETR-L primary cow bounding boxes
    sample_boxes = load_primary_cow_boxes(args.localization_csv, manifest_df)
    n_boxes = sum(1 for b in sample_boxes.values() if b is not None)
    print(f"[OK] Recovered RT-DETR-L boxes for {n_boxes} / {len(manifest_df)} samples")

    model_keys = [k.strip() for k in args.models.split(",") if k.strip() in MODEL_CONFIGS]
    print(f"[INFO] Selected models to evaluate: {model_keys} on device: {args.device}")

    all_model_metrics = []

    for model_key in model_keys:
        out_csv = os.path.join(args.output_dir, f"viewpoint_zeroshot_{model_key}.csv")
        if os.path.exists(out_csv) and not args.force and not args.smoke:
            existing_df = pd.read_csv(out_csv)
            if len(existing_df) == len(manifest_df):
                print(f"[SKIP] Model {model_key} already completed ({len(existing_df)} rows). Use --force to re-run.")
                metrics = evaluate_model_results(existing_df, model_key)
                all_model_metrics.append(metrics)
                print_evaluation_summary(metrics)
                continue

        # Initialize model
        classifier = create_classifier(model_key, args.device)

        records = []
        total_samples = len(manifest_df)
        t_start = time.time()

        for idx, (_, row) in enumerate(manifest_df.iterrows(), 1):
            s_id = str(row["sample_id"])
            ds = str(row["dataset"])
            img_path = str(row["source_image_path"])
            human_label = str(row["proposed_viewpoint"])

            box_info = sample_boxes.get(s_id)
            crop_img, input_mode = get_image_crop(img_path, box_info)

            # Predict
            pred = classifier.predict(crop_img)

            records.append({
                "sample_id": s_id,
                "dataset": ds,
                "category_or_subset": row.get("category_or_subset", ""),
                "human_viewpoint": human_label,
                "predicted_viewpoint": pred["predicted_viewpoint"],
                "top_score": round(pred["top_score"], 4),
                "second_viewpoint": pred["second_class"],
                "second_score": round(pred["second_score"], 4),
                "score_margin": round(pred["score_margin"], 4),
                "input_mode": input_mode,
                "model_name": MODEL_CONFIGS[model_key]["model_id"],
                "runtime_ms": round(pred["runtime_ms"], 1),
            })

            # Live Single-Line Progress Display
            elapsed = time.time() - t_start
            ips = idx / elapsed if elapsed > 0 else 0
            eta_sec = (total_samples - idx) / ips if ips > 0 else 0
            pct = (idx / total_samples) * 100
            sys.stdout.write(
                f"\r[{model_key}] [{idx:02d}/{total_samples:02d}] {pct:5.1f}% | "
                f"Sample: {s_id} ({ds[:8]}) | Pred: {pred['predicted_viewpoint']:<14} | "
                f"Top: {pred['top_score']:.3f} | Margin: {pred['score_margin']:.3f} | "
                f"Speed: {ips:.1f} img/s | ETA: {eta_sec:.0f}s"
            )
            sys.stdout.flush()

        sys.stdout.write("\n")

        # Save per-model CSV
        res_df = pd.DataFrame(records)
        res_df.to_csv(out_csv, index=False)
        print(f"[OK] Saved {len(res_df)} predictions to {out_csv}")

        # Free GPU memory
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = evaluate_model_results(res_df, model_key)
        all_model_metrics.append(metrics)
        print_evaluation_summary(metrics)

    # -----------------------------------------------------------------------
    # Comparative Scorecard Table Across All Models
    # -----------------------------------------------------------------------
    if len(all_model_metrics) > 0:
        comp_records = []
        for m in all_model_metrics:
            comp_records.append({
                "model_key": m["model_key"],
                "family": m["family"],
                "model_id": m["model_id"],
                "overall_accuracy": round(m["overall_accuracy"] * 100, 2),
                "macro_f1": round(m["macro_f1"], 4),
                "sciencedb_acc": round(m["sciencedb_acc"] * 100, 2),
                "mmcows_acc": round(m["mmcows_acc"] * 100, 2),
                "sideview_acc": round(m["sideview_acc"] * 100, 2),
                "false_front_count": m["false_front_count"],
                "ambiguous_recall": round(m["ambiguous_recall"] * 100, 2),
                "mean_top1_score": round(m["mean_top1_score"], 4),
                "mean_score_margin": round(m["mean_score_margin"], 4),
                "mean_runtime_ms": round(m["mean_runtime_ms"], 1),
            })
        comp_df = pd.DataFrame(comp_records)
        comp_csv = os.path.join(args.output_dir, "viewpoint_zeroshot_model_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"\n[OK] Wrote comparative model scorecard to {comp_csv}\n")

        print("=" * 100)
        print("  COMPARATIVE ZERO-SHOT MODEL PERFORMANCE TABLE (N=60)")
        print("=" * 100)
        print(comp_df.to_string(index=False))
        print("=" * 100)


if __name__ == "__main__":
    run_benchmark()
