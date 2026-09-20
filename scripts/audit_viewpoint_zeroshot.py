"""
audit_viewpoint_zeroshot.py — Comparative Zero-Shot VLM Audit for Cattle Viewpoint

Benchmarks multiple frozen vision-language models on cattle viewpoint samples:
  1. openai_clip:     openai/clip-vit-base-patch32 (OpenAI CLIP)
  2. openclip_laion:  laion/CLIP-ViT-B-32-laion2B-s34B-b79K (OpenCLIP LAION-2B)
  3. google_siglip:   google/siglip-base-patch16-224 (Google SigLIP)

Evaluation Methods:
  - Method A: Explicit 6-class zero-shot classification (including 'unknown / ambiguous' prompt).
  - Method B: 5-class physical classification ('rear', 'rear-oblique', 'side', 'front-oblique', 'front')
              with confidence/margin rejection thresholding for 'unknown / ambiguous'.
  - Both: Computes and compares both methods simultaneously with threshold sensitivity sweeps.

Features:
  - Strict visual-only inference (zero metadata access).
  - Target-cow crops recovered from Step 2.1 RT-DETR-L detections via normalized path matching.
  - Standardized prompt ensembles across all 6 candidate classes.
  - Dynamic ground truth support: 'final_viewpoint' if present, else 'proposed_viewpoint'.
  - Metrics: Overall Acc, Macro-F1, Per-dataset Acc, Per-class PRF, False Fronts, Ambiguous Recall.
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

PHYSICAL_CLASSES = [
    "rear",
    "rear-oblique",
    "side",
    "front-oblique",
    "front",
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

def normalize_path(p: Any) -> str:
    """Normalize file path for robust cross-platform/case-insensitive matching."""
    return os.path.abspath(os.path.normpath(str(p))).lower()


def load_primary_cow_boxes(
    localization_csv: str, manifest_df: pd.DataFrame
) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, str]]:
    """
    Load Step 2.1 RT-DETR-L primary cow bounding box for each sample by normalized image path.
    Returns:
      sample_boxes: Dict mapping sample_id -> box_info (or None)
      sample_modes: Dict mapping sample_id -> 'rtdetr_crop' or 'full_image_fallback'
    """
    if not os.path.exists(localization_csv):
        print(f"[WARN] Localization CSV not found: {localization_csv}. Using full images.")
        return (
            {str(s_id): None for s_id in manifest_df["sample_id"]},
            {str(s_id): "full_image_fallback" for s_id in manifest_df["sample_id"]},
        )

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
    rt_valid["norm_path"] = rt_valid["image_path"].apply(normalize_path)

    # Sort to select primary cow: largest area first, then highest confidence
    sorted_det = rt_valid.sort_values(
        by=["norm_path", "area", "confidence"], ascending=[True, False, False]
    )

    primary_map: Dict[str, Dict[str, Any]] = {}
    for norm_p, grp in sorted_det.groupby("norm_path"):
        top = grp.iloc[0]
        primary_map[norm_p] = {
            "box": [
                float(top["box_x1"]),
                float(top["box_y1"]),
                float(top["box_x2"]),
                float(top["box_y2"]),
            ],
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
# 3. Model Wrapper Classes Supporting Method A and Method B
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

    def predict(
        self,
        image: Image.Image,
        margin_thresh: float = 0.10,
        conf_thresh: float = 0.30,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class CLIPViewpointClassifier(ZeroShotViewpointClassifier):
    """Wrapper for OpenAI CLIP and OpenCLIP models via Hugging Face."""

    def _load(self):
        model_id = self.config["model_id"]
        print(f"Loading {self.config['family']} from {model_id} onto {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # Precompute normalized prompt ensemble text embeddings for each of the 6 classes
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

    def predict(
        self,
        image: Image.Image,
        margin_thresh: float = 0.10,
        conf_thresh: float = 0.30,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = extract_pooled_features(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarities across all 6 classes
            sims = (image_features @ self.class_text_embeddings.T).squeeze(0)  # (6,)
            logits_6 = sims * self.model.logit_scale.exp().clamp(max=100.0)
            probs_6 = F.softmax(logits_6, dim=-1).cpu().numpy()

            # Method A (6 classes)
            scores_a = {TAXONOMY_CLASSES[i]: float(probs_6[i]) for i in range(6)}
            sorted_a = sorted(scores_a.items(), key=lambda x: x[1], reverse=True)
            top1_a, top1_s_a = sorted_a[0]
            top2_a, top2_s_a = sorted_a[1]
            margin_a = top1_s_a - top2_s_a

            # Method B (5 physical classes only)
            logits_5 = logits_6[:5]
            probs_5 = F.softmax(logits_5, dim=-1).cpu().numpy()
            scores_b = {PHYSICAL_CLASSES[i]: float(probs_5[i]) for i in range(5)}
            sorted_b = sorted(scores_b.items(), key=lambda x: x[1], reverse=True)
            top1_phys, top1_s_phys = sorted_b[0]
            top2_phys, top2_s_phys = sorted_b[1]
            margin_b = top1_s_phys - top2_s_phys

            # Rejection rule for Method B
            is_rejected = (top1_s_phys < conf_thresh) or (margin_b < margin_thresh)
            pred_b = "unknown / ambiguous" if is_rejected else top1_phys

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return {
            # Method A results
            "method_a_viewpoint": top1_a,
            "method_a_top_score": round(top1_s_a, 4),
            "method_a_second_viewpoint": top2_a,
            "method_a_second_score": round(top2_s_a, 4),
            "method_a_margin": round(margin_a, 4),
            "method_a_scores": scores_a,
            # Method B results
            "method_b_viewpoint": pred_b,
            "method_b_raw_physical": top1_phys,
            "method_b_top_score": round(top1_s_phys, 4),
            "method_b_second_viewpoint": top2_phys,
            "method_b_second_score": round(top2_s_phys, 4),
            "method_b_margin": round(margin_b, 4),
            "method_b_is_rejected": is_rejected,
            "method_b_scores": scores_b,
            "runtime_ms": round(dt_ms, 1),
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

    def predict(
        self,
        image: Image.Image,
        margin_thresh: float = 0.10,
        conf_thresh: float = 0.30,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = extract_pooled_features(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ self.class_text_embeddings.T).squeeze(0)  # (6,)
            if hasattr(self.model, "logit_scale"):
                logits_6 = sims * self.model.logit_scale.exp()
                if hasattr(self.model, "logit_bias"):
                    logits_6 = logits_6 + self.model.logit_bias
            else:
                logits_6 = sims * 10.0
            probs_6 = F.softmax(logits_6, dim=-1).cpu().numpy()

            # Method A (6 classes)
            scores_a = {TAXONOMY_CLASSES[i]: float(probs_6[i]) for i in range(6)}
            sorted_a = sorted(scores_a.items(), key=lambda x: x[1], reverse=True)
            top1_a, top1_s_a = sorted_a[0]
            top2_a, top2_s_a = sorted_a[1]
            margin_a = top1_s_a - top2_s_a

            # Method B (5 physical classes only)
            logits_5 = logits_6[:5]
            probs_5 = F.softmax(logits_5, dim=-1).cpu().numpy()
            scores_b = {PHYSICAL_CLASSES[i]: float(probs_5[i]) for i in range(5)}
            sorted_b = sorted(scores_b.items(), key=lambda x: x[1], reverse=True)
            top1_phys, top1_s_phys = sorted_b[0]
            top2_phys, top2_s_phys = sorted_b[1]
            margin_b = top1_s_phys - top2_s_phys

            # Rejection rule for Method B
            is_rejected = (top1_s_phys < conf_thresh) or (margin_b < margin_thresh)
            pred_b = "unknown / ambiguous" if is_rejected else top1_phys

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return {
            # Method A results
            "method_a_viewpoint": top1_a,
            "method_a_top_score": round(top1_s_a, 4),
            "method_a_second_viewpoint": top2_a,
            "method_a_second_score": round(top2_s_a, 4),
            "method_a_margin": round(margin_a, 4),
            "method_a_scores": scores_a,
            # Method B results
            "method_b_viewpoint": pred_b,
            "method_b_raw_physical": top1_phys,
            "method_b_top_score": round(top1_s_phys, 4),
            "method_b_second_viewpoint": top2_phys,
            "method_b_second_score": round(top2_s_phys, 4),
            "method_b_margin": round(margin_b, 4),
            "method_b_is_rejected": is_rejected,
            "method_b_scores": scores_b,
            "runtime_ms": round(dt_ms, 1),
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

def evaluate_predictions(
    y_true: List[str], y_pred: List[str], datasets: List[str], classes: List[str]
) -> Dict[str, Any]:
    """Compute standard classification metrics."""
    support_classes = [c for c in classes if c in set(y_true)]
    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=support_classes, average="macro", zero_division=0)

    per_dataset_acc = {}
    for ds in ["ScienceDB", "MmCows", "SideViewCows2026"]:
        idx_ds = [i for i, d in enumerate(datasets) if d == ds]
        if len(idx_ds) > 0:
            sub_true = [y_true[i] for i in idx_ds]
            sub_pred = [y_pred[i] for i in idx_ds]
            per_dataset_acc[ds] = accuracy_score(sub_true, sub_pred)
        else:
            per_dataset_acc[ds] = 0.0

    gt_fronts = sum(1 for yt in y_true if yt == "front")
    false_fronts = sum(1 for yt, yp in zip(y_true, y_pred) if yp == "front" and yt != "front")

    amb_indices = [i for i, yt in enumerate(y_true) if yt == "unknown / ambiguous"]
    gt_ambiguous = len(amb_indices)
    amb_recall = (
        sum(1 for i in amb_indices if y_pred[i] == "unknown / ambiguous") / gt_ambiguous
        if gt_ambiguous > 0
        else 0.0
    )

    return {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "sciencedb_acc": per_dataset_acc.get("ScienceDB", 0.0),
        "mmcows_acc": per_dataset_acc.get("MmCows", 0.0),
        "sideview_acc": per_dataset_acc.get("SideViewCows2026", 0.0),
        "false_front_count": false_fronts,
        "gt_front_count": gt_fronts,
        "ambiguous_recall": amb_recall,
        "gt_ambiguous_count": gt_ambiguous,
    }


def print_scorecard(title: str, metrics: Dict[str, Any], n_samples: int):
    """Print clean terminal scorecard."""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"Samples Evaluated:       {n_samples}")
    print(f"Overall Accuracy:        {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Macro-F1 (support > 0):  {metrics['macro_f1']:.4f}")
    print(f"Per-Dataset Accuracy:")
    print(f"  - ScienceDB:           {metrics['sciencedb_acc'] * 100:.2f}%")
    print(f"  - MmCows:              {metrics['mmcows_acc'] * 100:.2f}%")
    print(f"  - SideViewCows2026:    {metrics['sideview_acc'] * 100:.2f}%")
    print(f"False 'front' Count:     {metrics['false_front_count']} (Ground truth has {metrics['gt_front_count']} front)")
    print(f"Ambiguous Case Recall:   {metrics['ambiguous_recall'] * 100:.2f}% ({metrics['gt_ambiguous_count']} true ambiguous)")
    if "mean_top1_score" in metrics:
        print(f"Mean Top-1 Score:        {metrics['mean_top1_score']:.4f}")
        print(f"Mean Top-1 Margin:       {metrics['mean_score_margin']:.4f}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# 5. Main Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Comparative Zero-Shot VLM Audit for Cattle Viewpoint (Method A & B)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv",
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
        "--method",
        type=str,
        choices=["method_a", "method_b", "both"],
        default="both",
        help="Evaluation method: method_a (6 classes), method_b (5 physical + rejection), or both",
    )
    parser.add_argument(
        "--margin-thresh",
        type=float,
        default=0.10,
        help="Method B margin rejection threshold (delta < tau -> ambiguous)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.30,
        help="Method B confidence rejection threshold (score < tau -> ambiguous)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test on 2 samples to verify pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/perception_audit/viewpoint_zeroshot_expanded100",
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

    if "final_viewpoint" in manifest_df.columns:
        gt_col = "final_viewpoint"
    elif "proposed_viewpoint" in manifest_df.columns:
        gt_col = "proposed_viewpoint"
    else:
        raise ValueError(
            f"Manifest {args.manifest} has neither 'final_viewpoint' nor 'proposed_viewpoint' column."
        )
    print(f"[INFO] Ground truth column selected: '{gt_col}'")
    print(f"[INFO] Ground truth label distribution (N={len(manifest_df)}):")
    for c, cnt in manifest_df[gt_col].value_counts().items():
        print(f"       - {c:<20}: {cnt}")

    if args.smoke:
        print("[INFO] SMOKE TEST MODE: Selecting 2 representative samples.")
        smoke_ids = [manifest_df["sample_id"].iloc[0], manifest_df["sample_id"].iloc[-1]]
        manifest_df = manifest_df[manifest_df["sample_id"].isin(smoke_ids)].copy()
        print(f"[INFO] Smoke samples selected: {smoke_ids}")

    # Load RT-DETR-L primary cow bounding boxes via normalized path matching
    sample_boxes, sample_modes = load_primary_cow_boxes(args.localization_csv, manifest_df)
    n_boxes = sum(1 for b in sample_boxes.values() if b is not None)
    n_fallbacks = len(manifest_df) - n_boxes
    print(f"[OK] Recovered RT-DETR-L crops: {n_boxes} / {len(manifest_df)}")
    print(f"[INFO] Full-image fallbacks: {n_fallbacks} / {len(manifest_df)}")

    model_keys = [k.strip() for k in args.models.split(",") if k.strip() in MODEL_CONFIGS]
    print(f"[INFO] Selected models to evaluate: {model_keys} on device: {args.device}")
    print(f"[INFO] Evaluation Method: {args.method.upper()} (Margin Thresh: {args.margin_thresh}, Conf Thresh: {args.conf_thresh})")

    all_model_method_a_metrics = []
    all_model_method_b_metrics = []
    all_model_phys_metrics = []

    for model_key in model_keys:
        out_csv = os.path.join(args.output_dir, f"viewpoint_zeroshot_{model_key}.csv")
        
        # Check if re-run is needed
        run_inference = True
        if os.path.exists(out_csv) and not args.force and not args.smoke:
            existing_df = pd.read_csv(out_csv)
            if len(existing_df) == len(manifest_df) and "method_b_viewpoint" in existing_df.columns:
                print(f"[SKIP] Model {model_key} already completed ({len(existing_df)} rows). Use --force to re-run.")
                res_df = existing_df
                run_inference = False

        if run_inference:
            classifier = create_classifier(model_key, args.device)
            records = []
            total_samples = len(manifest_df)
            t_start = time.time()

            for idx, (_, row) in enumerate(manifest_df.iterrows(), 1):
                s_id = str(row["sample_id"])
                ds = str(row["dataset"])
                img_path = str(row["source_image_path"])
                human_label = str(row[gt_col])

                box_info = sample_boxes.get(s_id)
                crop_img, input_mode = get_image_crop(img_path, box_info)

                # Predict both Method A and Method B
                pred = classifier.predict(
                    crop_img, margin_thresh=args.margin_thresh, conf_thresh=args.conf_thresh
                )

                records.append({
                    "sample_id": s_id,
                    "dataset": ds,
                    "category_or_subset": row.get("category_or_subset", ""),
                    "human_viewpoint": human_label,
                    # Method A fields
                    "predicted_viewpoint": pred["method_a_viewpoint"],
                    "top_score": pred["method_a_top_score"],
                    "second_viewpoint": pred["method_a_second_viewpoint"],
                    "second_score": pred["method_a_second_score"],
                    "score_margin": pred["method_a_margin"],
                    # Method B fields
                    "method_b_viewpoint": pred["method_b_viewpoint"],
                    "method_b_raw_physical": pred["method_b_raw_physical"],
                    "method_b_top_score": pred["method_b_top_score"],
                    "method_b_second_viewpoint": pred["method_b_second_viewpoint"],
                    "method_b_second_score": pred["method_b_second_score"],
                    "method_b_margin": pred["method_b_margin"],
                    "method_b_is_rejected": pred["method_b_is_rejected"],
                    "input_mode": input_mode,
                    "model_name": MODEL_CONFIGS[model_key]["model_id"],
                    "runtime_ms": pred["runtime_ms"],
                })

                # Live Single-Line Progress Display
                elapsed = time.time() - t_start
                ips = idx / elapsed if elapsed > 0 else 0
                eta_sec = (total_samples - idx) / ips if ips > 0 else 0
                pct = (idx / total_samples) * 100
                display_pred = pred["method_b_viewpoint"] if args.method == "method_b" else pred["method_a_viewpoint"]
                sys.stdout.write(
                    f"\r[{model_key}] [{idx:>3}/{total_samples}] {pct:5.1f}% | "
                    f"Sample: {s_id} | Pred: {display_pred:<19} | "
                    f"Speed: {ips:.1f} img/s | ETA: {eta_sec:.0f}s"
                )
                sys.stdout.flush()

            sys.stdout.write("\n")
            res_df = pd.DataFrame(records)
            res_df.to_csv(out_csv, index=False)
            print(f"[OK] Saved {len(res_df)} predictions to {out_csv}")

            del classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate Metrics
        y_true = res_df["human_viewpoint"].tolist()
        datasets = res_df["dataset"].tolist()

        # 1. Method A Evaluation
        if args.method in ["method_a", "both"]:
            y_pred_a = res_df["predicted_viewpoint"].tolist()
            mA = evaluate_predictions(y_true, y_pred_a, datasets, TAXONOMY_CLASSES)
            mA["model_key"] = model_key
            mA["family"] = MODEL_CONFIGS[model_key]["family"]
            mA["model_id"] = MODEL_CONFIGS[model_key]["model_id"]
            mA["mean_top1_score"] = float(res_df["top_score"].mean())
            mA["mean_score_margin"] = float(res_df["score_margin"].mean())
            mA["mean_runtime_ms"] = float(res_df["runtime_ms"].mean())
            all_model_method_a_metrics.append(mA)
            print_scorecard(f"METHOD A (6-Class Explicit): {mA['family']}", mA, len(res_df))

        # 2. Method B Evaluation
        if args.method in ["method_b", "both"]:
            # Raw Physical (on physical samples only)
            phys_mask = [yt != "unknown / ambiguous" for yt in y_true]
            y_true_phys = [y_true[i] for i in range(len(y_true)) if phys_mask[i]]
            y_pred_phys = [res_df["method_b_raw_physical"].iloc[i] for i in range(len(y_true)) if phys_mask[i]]
            ds_phys = [datasets[i] for i in range(len(y_true)) if phys_mask[i]]
            
            mPhys = evaluate_predictions(y_true_phys, y_pred_phys, ds_phys, PHYSICAL_CLASSES)
            mPhys["model_key"] = model_key
            mPhys["family"] = MODEL_CONFIGS[model_key]["family"]
            mPhys["model_id"] = MODEL_CONFIGS[model_key]["model_id"]
            mPhys["mean_top1_score"] = float(res_df["method_b_top_score"].mean())
            mPhys["mean_score_margin"] = float(res_df["method_b_margin"].mean())
            all_model_phys_metrics.append(mPhys)
            print_scorecard(f"METHOD B (Raw 5-Class Physical, N={len(y_true_phys)}): {mPhys['family']}", mPhys, len(y_true_phys))

            # Full Method B with Rejection (on all 100 samples)
            y_pred_b = res_df["method_b_viewpoint"].tolist()
            mB = evaluate_predictions(y_true, y_pred_b, datasets, TAXONOMY_CLASSES)
            mB["model_key"] = model_key
            mB["family"] = MODEL_CONFIGS[model_key]["family"]
            mB["model_id"] = MODEL_CONFIGS[model_key]["model_id"]
            mB["false_ambiguous"] = sum(1 for yt, yp in zip(y_true, y_pred_b) if yt != "unknown / ambiguous" and yp == "unknown / ambiguous")
            all_model_method_b_metrics.append(mB)
            print_scorecard(f"METHOD B (6-Class with Rejection tau_margin={args.margin_thresh}): {mB['family']}", mB, len(res_df))

            # Threshold sensitivity sweep table
            print(f"\nMethod B Threshold Sweep for {model_key} (Margin Rejection: delta < tau -> ambiguous):")
            print(f"{'tau_margin':<12} {'Overall Acc (N=100)':<22} {'Amb Recall (N=5)':<18} {'False Amb (N=95)':<18} {'Macro-F1':<10}")
            print("-" * 80)
            for tau in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                sweep_preds = [
                    "unknown / ambiguous" if res_df["method_b_margin"].iloc[i] < tau else res_df["method_b_raw_physical"].iloc[i]
                    for i in range(len(res_df))
                ]
                acc_t = accuracy_score(y_true, sweep_preds)
                f1_t = f1_score(y_true, sweep_preds, labels=TAXONOMY_CLASSES, average="macro", zero_division=0)
                amb_rec_t = sum(1 for yt, yp in zip(y_true, sweep_preds) if yt == "unknown / ambiguous" and yp == "unknown / ambiguous") / 5.0
                false_amb_t = sum(1 for yt, yp in zip(y_true, sweep_preds) if yt != "unknown / ambiguous" and yp == "unknown / ambiguous")
                print(f"{tau:<12.2f} {acc_t*100:<22.2f} {amb_rec_t*100:<18.2f} {false_amb_t:<18} {f1_t:<10.4f}")

    # -----------------------------------------------------------------------
    # Comparative Tables
    # -----------------------------------------------------------------------
    # 1. Method A Comparison Table
    if len(all_model_method_a_metrics) > 0:
        comp_a = []
        for m in all_model_method_a_metrics:
            comp_a.append({
                "model_key": m["model_key"],
                "family": m["family"],
                "overall_accuracy": round(m["overall_accuracy"] * 100, 2),
                "macro_f1": round(m["macro_f1"], 4),
                "sciencedb_acc": round(m["sciencedb_acc"] * 100, 2),
                "mmcows_acc": round(m["mmcows_acc"] * 100, 2),
                "sideview_acc": round(m["sideview_acc"] * 100, 2),
                "false_front_count": m["false_front_count"],
                "ambiguous_recall": round(m["ambiguous_recall"] * 100, 2),
            })
        df_comp_a = pd.DataFrame(comp_a)
        csv_a = os.path.join(args.output_dir, "viewpoint_zeroshot_method_a_comparison.csv")
        df_comp_a.to_csv(csv_a, index=False)
        print(f"\n[OK] Wrote Method A comparison table to {csv_a}")
        print("=" * 100)
        print(f"  METHOD A: COMPARATIVE ZERO-SHOT MODEL PERFORMANCE TABLE (N={len(manifest_df)})")
        print("=" * 100)
        print(df_comp_a.to_string(index=False))
        print("=" * 100)

    # 2. Method B Raw Physical Comparison Table
    if len(all_model_phys_metrics) > 0:
        comp_phys = []
        for m in all_model_phys_metrics:
            comp_phys.append({
                "model_key": m["model_key"],
                "family": m["family"],
                "raw_phys_accuracy": round(m["overall_accuracy"] * 100, 2),
                "raw_macro_f1": round(m["macro_f1"], 4),
                "sciencedb_acc": round(m["sciencedb_acc"] * 100, 2),
                "mmcows_acc": round(m["mmcows_acc"] * 100, 2),
                "sideview_acc": round(m["sideview_acc"] * 100, 2),
                "predicted_fronts": m["false_front_count"],
            })
        df_comp_phys = pd.DataFrame(comp_phys)
        csv_phys = os.path.join(args.output_dir, "viewpoint_zeroshot_method_b_raw_physical_comparison.csv")
        df_comp_phys.to_csv(csv_phys, index=False)
        print(f"\n[OK] Wrote Method B Raw Physical comparison table to {csv_phys}")
        print("=" * 100)
        print(f"  METHOD B: RAW 5-CLASS PHYSICAL PERFORMANCE TABLE (N=95 physical samples, No Rejection)")
        print("=" * 100)
        print(df_comp_phys.to_string(index=False))
        print("=" * 100)

    # 3. Method B Full Comparison Table
    if len(all_model_method_b_metrics) > 0:
        comp_b = []
        for m in all_model_method_b_metrics:
            comp_b.append({
                "model_key": m["model_key"],
                "family": m["family"],
                "overall_accuracy": round(m["overall_accuracy"] * 100, 2),
                "macro_f1": round(m["macro_f1"], 4),
                "sciencedb_acc": round(m["sciencedb_acc"] * 100, 2),
                "mmcows_acc": round(m["mmcows_acc"] * 100, 2),
                "sideview_acc": round(m["sideview_acc"] * 100, 2),
                "false_front_count": m["false_front_count"],
                "ambiguous_recall": round(m["ambiguous_recall"] * 100, 2),
                "false_ambiguous": m["false_ambiguous"],
            })
        df_comp_b = pd.DataFrame(comp_b)
        csv_b = os.path.join(args.output_dir, "viewpoint_zeroshot_method_b_comparison.csv")
        df_comp_b.to_csv(csv_b, index=False)
        print(f"\n[OK] Wrote Method B Rejection comparison table to {csv_b}")
        print("=" * 100)
        print(f"  METHOD B: 6-CLASS REJECTION PERFORMANCE TABLE (N=100, tau_margin={args.margin_thresh})")
        print("=" * 100)
        print(df_comp_b.to_string(index=False))
        print("=" * 100)


if __name__ == "__main__":
    run_benchmark()
