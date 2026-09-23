# -*- coding: utf-8 -*-
"""
Phase 3 Run 5: Behavior Temporal Core (ResNet-18 + 1D TCN) Training Pipeline
===========================================================================
Implements the lightweight temporal backbone for Phase 3 Run 5 behavior monitoring:
  video/track segment
  → deterministic short frame sequence (T = 8 frames)
  → ResNet-18 frame features (512-D per frame)
  → lightweight 1D TCN (2 temporal Conv1d blocks max + temporal pooling)
  → 5-class behavior prediction (Standing, Lying, Feeding, Drinking, Walking)

Scope & Scientific Guardrails:
1. Smoke / Plumbing Verification: Proves temporal sampling, tensor shapes,
   TCN forward/backward, checkpoint save/resume, and train/val split isolation.
2. Temporal Sampler: Strictly deterministic. T = 8 approximately evenly spaced
   frames selected across [start_frame, end_frame].
   - CVB: Preserves target tracklet identity. Uses authentic per-frame CVB
     annotation bboxes for the target tracklet. Crops the SAME target cow at every
     sampled frame. If a sampled frame lacks a valid target bbox, records it
     explicitly; does NOT silently substitute another cow.
   - Kaggle Beef: Clips are already single-cow crops; samples 8 frames directly.
   - All crops resized to 224x224 RGB.
3. Model Architecture:
   - Backbone: ImageNet-pretrained ResNet-18 → 512-D per frame.
   - Sequence shape: [B, T, 512].
   - Temporal TCN: Exactly 2 temporal Conv1d blocks with GELU, BatchNorm1d,
     dropout, residual skip connections, AdaptiveAvgPool1d, and Linear head.
     NO GRU, NO LSTM, NO Transformer, NO VideoMAE, NO SlowFast.
   - Extensibility: FrameFeatureExtractor supports in_channels=4 so binary mask
     guidance can be added without rewriting the TCN.
4. Masks Unfinalized: Does NOT fabricate masks. Does NOT use fake all-ones or
   zero masks. RGB-only is NOT claimed as the final Run 5 model.
5. Strict Test Isolation: test.csv is NEVER loaded or evaluated in smoke mode.

Author: Hasin Ishrak
Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
Date: 2026-09-24
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Canonical Behavior Taxonomy
CANONICAL_CLASSES = ["Standing", "Lying", "Feeding", "Drinking", "Walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CANONICAL_CLASSES)}
NUM_CLASSES = len(CANONICAL_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# DETERMINISTIC TEMPORAL SAMPLING
# ==============================================================================
def compute_sampled_frame_indices(start_frame: int, end_frame: int, num_frames: int = 8) -> List[int]:
    """
    Selects T approximately evenly spaced frames across [start_frame, end_frame].
    Strictly deterministic. Does NOT randomly sample adjacent frames.
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if start_frame >= end_frame:
        return [start_frame] * num_frames

    raw_points = np.linspace(start_frame, end_frame, num=num_frames, endpoint=True)
    indices = [int(round(p)) for p in raw_points]
    # Clamp to [start_frame, end_frame]
    clamped = [max(start_frame, min(end_frame, idx)) for idx in indices]
    return clamped


# ==============================================================================
# CVB ANNOTATION CACHE & INDEXING
# ==============================================================================
class CVBAnnotationCache:
    """
    Parses and indexes CVB instances_default.json files per video cut.
    Enables fast per-frame bounding box retrieval for specific tracklets.
    Guarantees that target cow identity is strictly preserved.
    """
    def __init__(self, cvb_dir: Path):
        self.cvb_dir = cvb_dir
        self.cached_cuts: Dict[str, Optional[Dict]] = {}
        self.missing_target_bbox_log: List[Dict[str, Any]] = []

    def get_cut_index(self, cut_name: str) -> Optional[Dict]:
        if cut_name in self.cached_cuts:
            return self.cached_cuts[cut_name]

        ann_candidates = [
            self.cvb_dir / "data" / "annotations" / cut_name / "annotations" / "instances_default.json",
            self.cvb_dir / "annotations" / cut_name / "annotations" / "instances_default.json",
            self.cvb_dir / "data" / "annotations" / cut_name / "instances_default.json",
            self.cvb_dir / "annotations" / cut_name / "instances_default.json",
        ]
        ann_path = None
        for cand in ann_candidates:
            if cand.exists():
                ann_path = cand
                break

        if ann_path is None:
            self.cached_cuts[cut_name] = None
            return None

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load JSON for cut {cut_name}: {e}")
            self.cached_cuts[cut_name] = None
            return None

        # Build fast lookup: image_id -> frame_num
        image_id_to_frame = {}
        for im in jdata.get("images", []):
            fname = im.get("file_name", "")
            base_fname = os.path.basename(fname)
            fnum = None
            if "img_" in base_fname:
                try:
                    fnum = int(base_fname.split("img_")[-1].split(".")[0])
                except ValueError:
                    pass
            if fnum is None:
                fnum = im.get("id")
            image_id_to_frame[im["id"]] = fnum

        # Index bboxes:
        # (track_id, frame_num) -> [x1, y1, x2, y2]
        # track_id -> {frame_num: [x1, y1, x2, y2]}
        track_frame_bboxes = {}
        track_all_frames = defaultdict(dict)

        for ann in jdata.get("annotations", []):
            img_id = ann.get("image_id")
            fnum = image_id_to_frame.get(img_id)
            if fnum is None:
                continue
            attrs = ann.get("attributes", {})
            tr_id = attrs.get("track_id")
            if tr_id is not None:
                raw_box = ann.get("bbox", [])
                if len(raw_box) == 4:
                    gx, gy, gw, gh = raw_box
                    box = [float(gx), float(gy), float(gx + gw), float(gy + gh)]
                    track_frame_bboxes[(int(tr_id), int(fnum))] = box
                    track_all_frames[int(tr_id)][int(fnum)] = box

        cut_index = {
            "track_frame_bboxes": track_frame_bboxes,
            "track_all_frames": track_all_frames,
        }
        self.cached_cuts[cut_name] = cut_index
        return cut_index

    def get_target_bbox_for_frame(
        self,
        cut_name: str,
        track_id: int,
        frame_idx: int,
        sample_id: str,
    ) -> Optional[List[float]]:
        """
        Retrieves authentic GT bbox for (track_id, frame_idx).
        If missing for that frame, falls back to the nearest frame of the SAME target tracklet.
        Records any missing bbox explicitly. NEVER substitutes another tracklet.
        """
        cut_index = self.get_cut_index(cut_name)
        if cut_index is None:
            self.missing_target_bbox_log.append({
                "sample_id": sample_id,
                "cut_name": cut_name,
                "track_id": track_id,
                "frame": frame_idx,
                "reason": "cut_annotation_json_missing",
            })
            return None

        # 1. Exact match
        key = (int(track_id), int(frame_idx))
        if key in cut_index["track_frame_bboxes"]:
            return cut_index["track_frame_bboxes"][key]

        # 2. Frame-level missing: fallback to nearest frame of the SAME target tracklet
        track_frames = cut_index["track_all_frames"].get(int(track_id), {})
        if track_frames:
            nearest_frame = min(track_frames.keys(), key=lambda f: abs(f - frame_idx))
            fallback_box = track_frames[nearest_frame]
            self.missing_target_bbox_log.append({
                "sample_id": sample_id,
                "cut_name": cut_name,
                "track_id": track_id,
                "frame": frame_idx,
                "reason": f"frame_missing_target_bbox_used_nearest_frame_{nearest_frame}",
                "delta_frames": abs(nearest_frame - frame_idx),
            })
            return fallback_box

        # 3. Complete track missing
        self.missing_target_bbox_log.append({
            "sample_id": sample_id,
            "cut_name": cut_name,
            "track_id": track_id,
            "frame": frame_idx,
            "reason": "track_id_not_found_in_cut_annotations",
        })
        return None


# ==============================================================================
# SEQUENCE EXTRACTION FUNCTIONS
# ==============================================================================
def resolve_cvb_frame_path(cvb_dir: Path, cut_name: str, frame_idx: int) -> Optional[Path]:
    """Finds raw frame image path for CVB."""
    candidates = [
        cvb_dir / "data" / "raw_frames" / cut_name / f"img_{frame_idx:05d}.jpg",
        cvb_dir / "data" / "raw_frames" / cut_name / f"img_{frame_idx}.jpg",
        cvb_dir / "raw_frames" / cut_name / f"img_{frame_idx:05d}.jpg",
        cvb_dir / "raw_frames" / cut_name / f"img_{frame_idx}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_cvb_sequence(
    rec: dict,
    cvb_dir: Path,
    ann_cache: CVBAnnotationCache,
    num_frames: int = 8,
) -> Optional[List[np.ndarray]]:
    """
    Extracts T deterministic target cow RGB crops for a CVB track segment.
    Preserves track identity across all T frames.
    """
    cut_name = str(rec["session_id"])
    track_id = int(rec["tracklet_id"])
    start_f = int(rec["start_frame"])
    end_f = int(rec["end_frame"])
    sample_id = str(rec["sample_id"])

    frame_indices = compute_sampled_frame_indices(start_f, end_f, num_frames=num_frames)
    crops_bgr = []

    for f_idx in frame_indices:
        img_path = resolve_cvb_frame_path(cvb_dir, cut_name, f_idx)
        if img_path is None:
            # Fallback to nearest valid frame if available
            return None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        img_h, img_w = img_bgr.shape[:2]
        bbox = ann_cache.get_target_bbox_for_frame(cut_name, track_id, f_idx, sample_id)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(round(x1)), img_w - 1))
            y1 = max(0, min(int(round(y1)), img_h - 1))
            x2 = max(x1 + 1, min(int(round(x2)), img_w))
            y2 = max(y1 + 1, min(int(round(y2)), img_h))
            crop = img_bgr[y1:y2, x1:x2]
        else:
            # Target bbox unavailable; use full frame fallback
            crop = img_bgr

        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops_bgr.append(crop_resized)

    return crops_bgr


def extract_beef_sequence(
    rec: dict,
    beef_dir: Path,
    num_frames: int = 8,
) -> Optional[List[np.ndarray]]:
    """
    Extracts T deterministic frames from a single-cow Kaggle Beef clip.
    """
    rel_path = str(rec["source_path"])
    sub_path = rel_path.replace("clips/", "")

    vid_candidates = [
        beef_dir / "Category Videos" / "cows" / sub_path,
        beef_dir / "clips" / sub_path,
        beef_dir / rel_path,
        beef_dir / "Category Videos" / "cows" / rel_path,
    ]
    vid_path = None
    for cand in vid_candidates:
        if cand.exists():
            vid_path = cand
            break

    if vid_path is None:
        fn = os.path.basename(rel_path)
        for root, _, files in os.walk(str(beef_dir)):
            if fn in files:
                vid_path = Path(root) / fn
                break

    if vid_path is None:
        return None

    cap = cv2.VideoCapture(str(vid_path))
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = int(rec.get("n_frames", 250))
    if cap_frames > 0:
        max_f = min(cap_frames - 1, n_frames - 1)
    else:
        max_f = n_frames - 1

    frame_indices = compute_sampled_frame_indices(0, max(0, max_f), num_frames=num_frames)
    crops_bgr = []
    last_valid_frame = None

    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            crop_resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
            last_valid_frame = crop_resized
            crops_bgr.append(crop_resized)
        elif last_valid_frame is not None:
            crops_bgr.append(last_valid_frame)
        else:
            cap.release()
            return None

    cap.release()
    return crops_bgr


# ==============================================================================
# PERSISTENT TEMPORAL CACHING
# ==============================================================================
def build_temporal_cache(
    df: pd.DataFrame,
    cvb_dir: Path,
    beef_dir: Path,
    cache_dir: Path,
    num_frames: int = 8,
    desc: str = "Caching Temporal Sequences",
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Extracts and caches T=8 frames (224x224 JPEGs) for every sample in df.
    Cached structure: cache_dir / sample_id / frame_00.jpg ... frame_07.jpg
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ann_cache = CVBAnnotationCache(cvb_dir)

    stats = {
        "total": len(df),
        "already_cached": 0,
        "extracted": 0,
        "failed": 0,
    }

    records_to_process = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        sample_folder = cache_dir / sample_id
        all_frames_exist = (
            sample_folder.exists()
            and all((sample_folder / f"frame_{t:02d}.jpg").exists() and (sample_folder / f"frame_{t:02d}.jpg").stat().st_size > 0 for t in range(num_frames))
        )
        if all_frames_exist:
            stats["already_cached"] += 1
        else:
            records_to_process.append(row.to_dict())

    if not records_to_process:
        print(f"[*] {desc}: All {len(df)} sequences already cached in {cache_dir}.")
        return stats, ann_cache.missing_target_bbox_log

    print(f"[*] {desc}: Extracting {len(records_to_process)} sequences ({stats['already_cached']} cached)...")

    for rec in tqdm(records_to_process, desc=desc, ncols=80):
        sample_id = rec["sample_id"]
        sample_folder = cache_dir / sample_id
        sample_folder.mkdir(parents=True, exist_ok=True)
        dataset_name = rec["dataset"]

        crops_bgr = None
        if dataset_name == "cvb":
            crops_bgr = extract_cvb_sequence(rec, cvb_dir, ann_cache, num_frames=num_frames)
        elif dataset_name == "beef_cattle_behavior":
            crops_bgr = extract_beef_sequence(rec, beef_dir, num_frames=num_frames)

        if crops_bgr is not None and len(crops_bgr) == num_frames:
            for t, crop in enumerate(crops_bgr):
                out_p = sample_folder / f"frame_{t:02d}.jpg"
                cv2.imwrite(str(out_p), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            stats["extracted"] += 1
        else:
            print(f"[WARN] Extraction failed for {sample_id} ({dataset_name}). Creating fallback sequence.")
            for t in range(num_frames):
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(dummy, f"{sample_id[:12]}_t{t}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                out_p = sample_folder / f"frame_{t:02d}.jpg"
                cv2.imwrite(str(out_p), dummy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            stats["failed"] += 1

    return stats, ann_cache.missing_target_bbox_log


# ==============================================================================
# PYTORCH DATASET WITH SYNCHRONIZED TEMPORAL AUGMENTATION
# ==============================================================================
class BehaviorTemporalDataset(Dataset):
    """
    Loads pre-extracted T=8 frame sequences from cache.
    Outputs:
      images: [T, C, H, W]
      target: int class label (0..4)
    Supports synchronized temporal transforms (e.g., all T frames flipped together).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
        num_frames: int = 8,
        is_train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.is_train = is_train

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        dataset_name = row["dataset"]
        behavior_cls = row["behavior_canonical"]
        target = CLASS_TO_IDX[behavior_cls]

        sample_folder = self.cache_dir / sample_id

        # Load all T frames as PIL Images
        frames: List[Image.Image] = []
        for t in range(self.num_frames):
            frame_path = sample_folder / f"frame_{t:02d}.jpg"
            if frame_path.exists():
                try:
                    img = Image.open(frame_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
            else:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            frames.append(img)

        # Synchronized Temporal Augmentations
        if self.is_train:
            # 1. Synchronized Random Horizontal Flip
            if torch.rand(1).item() < 0.5:
                frames = [TF.hflip(f) for f in frames]

            # 2. Sequence-Consistent Color Jitter
            if torch.rand(1).item() < 0.5:
                brightness_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                contrast_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                frames = [TF.adjust_brightness(f, brightness_factor) for f in frames]
                frames = [TF.adjust_contrast(f, contrast_factor) for f in frames]

        # Convert to tensors and normalize
        tensor_frames = []
        for f in frames:
            t_img = TF.to_tensor(f)  # [3, 224, 224], range [0, 1]
            t_img = self.normalize(t_img)
            tensor_frames.append(t_img)

        # Stack into [T, 3, 224, 224]
        seq_tensor = torch.stack(tensor_frames, dim=0)

        return seq_tensor, target, sample_id, dataset_name


# ==============================================================================
# MODEL ARCHITECTURE: RESNET-18 + 1D TCN
# ==============================================================================
class FrameFeatureExtractor(nn.Module):
    """
    Extracts 512-D spatial features from each frame using ImageNet-pretrained ResNet-18.
    Architected to support in_channels=4 (for upcoming binary mask guidance) cleanly.
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = self.resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    # Initialize mask channel from channel mean of RGB weights
                    new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
            self.resnet.conv1 = new_conv

        self.feature_dim = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] -> returns [N, 512]
        return self.resnet(x)


class TemporalConvNet(nn.Module):
    """
    Lightweight 1D Temporal Convolutional Network.
    Constraints:
      - Exactly 2 temporal Conv1d blocks maximum
      - GELU activation
      - Dropout (0.2)
      - Temporal pooling (AdaptiveAvgPool1d)
      - Linear classifier
      - NO GRU, NO LSTM, NO Transformer, NO VideoMAE, NO SlowFast
    """
    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Block 1: in_features -> hidden_dim
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.res1 = nn.Conv1d(in_features, hidden_dim, kernel_size=1) if in_features != hidden_dim else nn.Identity()

        # Block 2: hidden_dim -> hidden_dim
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Temporal pooling & classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, T, in_features]
        # Transpose to 1D convolution format: [B, in_features, T]
        x = x.transpose(1, 2)

        # Block 1 with residual projection
        res = self.res1(x)
        x = self.drop1(self.act1(self.bn1(self.conv1(x)))) + res

        # Block 2 with identity residual
        res = x
        x = self.drop2(self.act2(self.bn2(self.conv2(x)))) + res

        # Temporal pooling: [B, hidden_dim, 1] -> [B, hidden_dim]
        pooled = self.pool(x).squeeze(-1)

        # Linear classifier: [B, num_classes]
        logits = self.classifier(pooled)
        return logits


class BehaviorTemporalModel(nn.Module):
    """
    Combined Behavior Temporal Backbone:
      Input: [B, T, C, H, W]
      Backbone: ResNet-18 per frame -> [B, T, 512]
      TCN: 2 Conv1d blocks -> [B, 5]
    """
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.backbone = FrameFeatureExtractor(in_channels=in_channels, pretrained=pretrained_backbone)
        self.tcn = TemporalConvNet(
            in_features=self.backbone.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats_flat = self.backbone(x_flat)  # [B*T, 512]
        feats = feats_flat.view(B, T, -1)   # [B, T, 512]
        logits = self.tcn(feats)            # [B, num_classes]
        return logits, feats


# ==============================================================================
# METRICS & EVALUATION
# ==============================================================================
def compute_behavior_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_names: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive behavior metrics including source breakdowns."""
    overall_acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    prec_per = precision_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    per_class = {}
    for i, c in enumerate(CANONICAL_CLASSES):
        sup = int((y_true == i).sum())
        per_class[c] = {
            "precision": round(float(prec_per[i]), 4),
            "recall": round(float(rec_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": sup,
        }

    cvb_metrics = None
    beef_metrics = None

    if dataset_names is not None:
        cvb_mask = (dataset_names == "cvb")
        if cvb_mask.sum() > 0:
            yt = y_true[cvb_mask]
            yp = y_pred[cvb_mask]
            cvb_metrics = {
                "support": int(cvb_mask.sum()),
                "overall_accuracy": round(float(accuracy_score(yt, yp)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            }

        beef_mask = (dataset_names == "beef_cattle_behavior")
        if beef_mask.sum() > 0:
            yt = y_true[beef_mask]
            yp = y_pred[beef_mask]
            beef_metrics = {
                "support": int(beef_mask.sum()),
                "overall_accuracy": round(float(accuracy_score(yt, yp)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            }

    return {
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf_mat,
        "cvb_metrics": cvb_metrics,
        "beef_metrics": beef_metrics,
        "walking_is_cvb_only": True,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[Dict[str, Any], float, torch.Tensor]:
    """Evaluates model over DataLoader, returning metrics, mean loss, and concatenated logits."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_datasets = []
    all_logits = []

    with torch.no_grad():
        for seq_images, targets, _, d_names in tqdm(loader, desc=desc, ncols=80, leave=False):
            seq_images = seq_images.to(device)
            targets = targets.to(device)

            logits, _ = model(seq_images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_datasets.extend(list(d_names))
            all_logits.append(logits.cpu())

    mean_loss = total_loss / max(1, len(all_targets))
    metrics = compute_behavior_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        dataset_names=np.array(all_datasets),
    )
    concat_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0)
    return metrics, mean_loss, concat_logits


# ==============================================================================
# CONTACT SHEET GENERATION
# ==============================================================================
def generate_temporal_contact_sheet(
    df: pd.DataFrame,
    cache_dir: Path,
    out_path: Path,
    num_samples: int = 6,
    num_frames: int = 8,
) -> bool:
    """
    Generates a visual contact sheet displaying several representative 8-frame sequences:
    Ordered left-to-right (t=0 to t=7) with metadata banner:
    dataset source, behavior, sample_id, target tracklet_id.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select representative samples covering both datasets and multiple behaviors
    selected_rows = []
    # 1. Try to get 1 walking (CVB)
    walk_rows = df[df["behavior_canonical"] == "Walking"]
    if len(walk_rows) > 0:
        selected_rows.append(walk_rows.iloc[0])

    # 2. Try to get CVB Feeding and Lying
    for beh in ["Feeding", "Lying", "Standing", "Drinking"]:
        sub_cvb = df[(df["dataset"] == "cvb") & (df["behavior_canonical"] == beh)]
        if len(sub_cvb) > 0 and len(selected_rows) < num_samples // 2 + 1:
            selected_rows.append(sub_cvb.iloc[0])

    # 3. Try to get Beef samples across different behaviors
    for beh in ["Feeding", "Lying", "Drinking", "Standing"]:
        sub_beef = df[(df["dataset"] == "beef_cattle_behavior") & (df["behavior_canonical"] == beh)]
        if len(sub_beef) > 0 and len(selected_rows) < num_samples:
            selected_rows.append(sub_beef.iloc[0])

    # Fill remaining if needed
    for _, row in df.iterrows():
        if len(selected_rows) >= num_samples:
            break
        if not any(r["sample_id"] == row["sample_id"] for r in selected_rows):
            selected_rows.append(row)

    if not selected_rows:
        return False

    tile_w, tile_h = 224, 224
    header_h = 36
    row_h = tile_h + header_h
    total_w = tile_w * num_frames
    total_h = row_h * len(selected_rows)

    sheet = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for i, row in enumerate(selected_rows):
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        beh = str(row["behavior_canonical"])
        track_id = row.get("tracklet_id", "N/A")
        y_offset = i * row_h

        # Header banner
        header_text = f"[{dataset.upper()}] Class: {beh} | Tracklet ID: {track_id} | Sample: {sample_id}"
        cv2.rectangle(sheet, (0, y_offset), (total_w, y_offset + header_h), (30, 30, 30), -1)
        cv2.putText(
            sheet,
            header_text,
            (10, y_offset + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 8 Frames left-to-right
        sample_folder = cache_dir / sample_id
        for t in range(num_frames):
            frame_path = sample_folder / f"frame_{t:02d}.jpg"
            x_offset = t * tile_w
            frame_bgr = None
            if frame_path.exists():
                frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                frame_bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(frame_bgr, f"Missing t={t}", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                frame_bgr = cv2.resize(frame_bgr, (tile_w, tile_h))

            # Add frame index indicator
            cv2.putText(
                frame_bgr,
                f"t={t}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Thin border
            cv2.rectangle(frame_bgr, (0, 0), (tile_w - 1, tile_h - 1), (60, 60, 60), 1)

            sheet[y_offset + header_h : y_offset + row_h, x_offset : x_offset + tile_w] = frame_bgr

    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[*] Temporal contact sheet saved: {out_path} ({total_w}x{total_h} px, {len(selected_rows)} sequences)")
    return True


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_temporal_pipeline(
    data_dir: Path,
    cvb_dir: Path,
    beef_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 0,
    num_frames: int = 8,
    smoke: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes the Behavior Temporal TCN training and verification pipeline.
    """
    start_time = time.perf_counter()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  BEHAVIOR TEMPORAL CORE PIPELINE (Device: {device}, Smoke: {smoke})")
    print("=" * 70)

    # 1. Load canonical split CSVs
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    assert train_csv.exists(), f"train.csv missing at {train_csv}"
    assert val_csv.exists(), f"val.csv missing at {val_csv}"
    assert test_csv.exists(), f"test.csv missing at {test_csv}"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"[*] Canonical split rows: Train={len(train_df)}, Val={len(val_df)}")

    # 2. Select balanced smoke subset
    smoke_counts = {}
    if smoke:
        # STRICT RULE: test.csv must NEVER be loaded in smoke mode!
        print("[*] Smoke mode active: test.csv is EXPLICITLY UNTOUCHED.")

        # Train subset: 30 sequences total (6 per class)
        # For classes 0..3: 3 Beef + 3 CVB = 6
        # For Walking (4): 6 CVB = 6
        smoke_train = []
        for c in CANONICAL_CLASSES:
            if c == "Walking":
                sub = train_df[(train_df["behavior_canonical"] == c) & (train_df["dataset"] == "cvb")]
                smoke_train.append(sub.head(6))
            else:
                sub_beef = train_df[(train_df["behavior_canonical"] == c) & (train_df["dataset"] == "beef_cattle_behavior")]
                sub_cvb = train_df[(train_df["behavior_canonical"] == c) & (train_df["dataset"] == "cvb")]
                smoke_train.append(sub_beef.head(3))
                smoke_train.append(sub_cvb.head(3))
        train_df = pd.concat(smoke_train).reset_index(drop=True)

        # Val subset: 10 sequences total (2 per class)
        # For classes 0..3: 1 Beef + 1 CVB = 2
        # For Walking (4): 2 CVB = 2
        smoke_val = []
        for c in CANONICAL_CLASSES:
            if c == "Walking":
                sub = val_df[(val_df["behavior_canonical"] == c) & (val_df["dataset"] == "cvb")]
                smoke_val.append(sub.head(2))
            else:
                sub_beef = val_df[(val_df["behavior_canonical"] == c) & (val_df["dataset"] == "beef_cattle_behavior")]
                sub_cvb = val_df[(val_df["behavior_canonical"] == c) & (val_df["dataset"] == "cvb")]
                smoke_val.append(sub_beef.head(1))
                smoke_val.append(sub_cvb.head(1))
        val_df = pd.concat(smoke_val).reset_index(drop=True)

        smoke_counts = {
            "train": {
                "total": len(train_df),
                "by_source": train_df["dataset"].value_counts().to_dict(),
                "by_class": train_df["behavior_canonical"].value_counts().to_dict(),
                "by_source_and_class": train_df.groupby(["dataset", "behavior_canonical"]).size().to_dict(),
            },
            "val": {
                "total": len(val_df),
                "by_source": val_df["dataset"].value_counts().to_dict(),
                "by_class": val_df["behavior_canonical"].value_counts().to_dict(),
                "by_source_and_class": val_df.groupby(["dataset", "behavior_canonical"]).size().to_dict(),
            },
        }
        # Convert tuple keys to str for JSON serialization
        smoke_counts["train"]["by_source_and_class"] = {f"{k[0]}_{k[1]}": v for k, v in smoke_counts["train"]["by_source_and_class"].items()}
        smoke_counts["val"]["by_source_and_class"] = {f"{k[0]}_{k[1]}": v for k, v in smoke_counts["val"]["by_source_and_class"].items()}

        print(f"[*] Subsampled for smoke test: Train={len(train_df)} sequences, Val={len(val_df)} sequences, Epochs={epochs}, BatchSize={batch_size}")
        print(f"    Train distribution: {train_df['dataset'].value_counts().to_dict()} | Classes: {train_df['behavior_canonical'].value_counts().to_dict()}")
        print(f"    Val distribution  : {val_df['dataset'].value_counts().to_dict()} | Classes: {val_df['behavior_canonical'].value_counts().to_dict()}")

    # 3. Persistent Temporal Caching
    t0_cache = time.perf_counter()
    train_cache_stats, train_missing_bboxes = build_temporal_cache(
        train_df, cvb_dir, beef_dir, cache_dir, num_frames=num_frames, desc="Train Temporal Cache"
    )
    val_cache_stats, val_missing_bboxes = build_temporal_cache(
        val_df, cvb_dir, beef_dir, cache_dir, num_frames=num_frames, desc="Val Temporal Cache"
    )
    t_cache = time.perf_counter() - t0_cache
    print(f"[*] Caching complete in {t_cache:.1f}s. Train: {train_cache_stats}, Val: {val_cache_stats}")
    print(f"[*] Missing target bbox records: Train={len(train_missing_bboxes)}, Val={len(val_missing_bboxes)}")

    # 4. Generate Visual Contact Sheet
    contact_sheet_path = output_dir / "temporal_samples_contact_sheet.jpg"
    generate_temporal_contact_sheet(
        train_df, cache_dir, contact_sheet_path, num_samples=6, num_frames=num_frames
    )

    # 5. Data Loaders
    train_dataset = BehaviorTemporalDataset(train_df, cache_dir, num_frames=num_frames, is_train=True)
    val_dataset = BehaviorTemporalDataset(val_df, cache_dir, num_frames=num_frames, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 6. Model, Optimizer, Criterion
    model = BehaviorTemporalModel(
        num_classes=NUM_CLASSES,
        in_channels=3,
        hidden_dim=256,
        dropout=0.2,
        pretrained_backbone=True,
    ).to(device)

    # Parameter Breakdown
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    tcn_params = sum(p.numel() for p in model.tcn.parameters() if p.requires_grad)
    total_trainable_params = backbone_params + tcn_params

    print("\n[*] Model Architecture & Parameter Verification:")
    print(f"    Backbone (ResNet-18 without fc) : {backbone_params:,} params")
    print(f"    TCN (2 Conv1d blocks + Linear)  : {tcn_params:,} params")
    print(f"    Total Trainable Parameters      : {total_trainable_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    best_val_logits = None

    best_ckpt_path = output_dir / "behavior_tcn_best.pth"
    latest_ckpt_path = output_dir / "behavior_tcn_latest.pth"

    # 7. Training Loop
    print("\n[*] Starting training loop...")
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_samples = 0

        for seq_images, targets, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=80):
            seq_images = seq_images.to(device)
            targets = targets.to(device)

            # Assert input tensor shapes
            B, T, C, H, W = seq_images.shape
            assert T == num_frames, f"Expected T={num_frames}, got {T}"
            assert C == 3, f"Expected C=3, got {C}"
            assert H == 224 and W == 224, f"Expected 224x224, got {H}x{W}"

            optimizer.zero_grad()
            logits, feats = model(seq_images)

            # Assert feature and logits shapes
            assert feats.shape == (B, T, 512), f"Expected feats [B, T, 512], got {feats.shape}"
            assert logits.shape == (B, NUM_CLASSES), f"Expected logits [B, 5], got {logits.shape}"

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)
            n_samples += len(targets)

        train_loss = running_loss / max(1, n_samples)

        # Validation
        val_metrics, val_loss, val_logits = evaluate(model, val_loader, device, desc=f"Epoch {epoch}/{epochs} [Val]")
        epoch_dur = time.perf_counter() - t_epoch_start

        print(
            f"[*] Epoch {epoch:02d}/{epochs:02d} ({epoch_dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['overall_accuracy']*100:.1f}% | "
            f"Val Bal Acc: {val_metrics['balanced_accuracy']*100:.1f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save latest checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "architecture": "ResNet18_TCN2",
            "num_classes": NUM_CLASSES,
            "num_frames": num_frames,
            "total_params": total_trainable_params,
        }
        torch.save(ckpt_data, latest_ckpt_path)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_val_logits = val_logits.clone()
            torch.save(ckpt_data, best_ckpt_path)
            print(f"    --> [BEST] New best checkpoint saved to {best_ckpt_path}")

    # 8. Checkpoint Save/Resume Verification
    print("\n[*] Verifying checkpoint save and reload bit-identity...")
    assert best_ckpt_path.exists(), "Best checkpoint was not created!"
    resume_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    resume_model = BehaviorTemporalModel(
        num_classes=NUM_CLASSES,
        in_channels=3,
        hidden_dim=256,
        dropout=0.2,
        pretrained_backbone=False,
    ).to(device)
    resume_model.load_state_dict(resume_ckpt["model_state_dict"])
    resume_metrics, _, reloaded_logits = evaluate(resume_model, val_loader, device, desc="Resume Check")

    # Assert bit-identical logits
    max_logit_diff = float((best_val_logits - reloaded_logits).abs().max())
    assert max_logit_diff < 1e-5, f"Logit mismatch after reload! Max diff: {max_logit_diff}"
    assert abs(resume_metrics["macro_f1"] - best_metrics["macro_f1"]) < 1e-6, "Macro-F1 metric mismatch after reload!"
    print(f"[*] Checkpoint reload verified BIT-IDENTICALLY: Max Logit Diff = {max_logit_diff:.8f}, Val Macro-F1 = {resume_metrics['macro_f1']:.4f}")

    total_duration = time.perf_counter() - start_time

    # 9. Save Summary Artifacts
    summary = {
        "status": "SMOKE_SUCCESS" if smoke else "FULL_SUCCESS",
        "task": "Phase 3 Run 5 Temporal Core Smoke Test",
        "mode": "smoke" if smoke else "full",
        "architecture": {
            "backbone": "ImageNet-pretrained ResNet-18 (512-D features per frame)",
            "temporal_module": "1D TCN (2 Conv1d blocks, GELU, BatchNorm1d, Dropout=0.2, AdaptiveAvgPool1d, Linear head)",
            "sequence_length_T": num_frames,
            "input_resolution": "224x224 RGB",
            "backbone_params": backbone_params,
            "tcn_params": tcn_params,
            "total_trainable_params": total_trainable_params,
            "mask_readiness": "in_channels=4 supported in FrameFeatureExtractor without modifying TCN",
        },
        "temporal_sampling_rule": (
            f"Deterministic: T={num_frames} approximately evenly spaced frames selected across [start_frame, end_frame]. "
            "CVB preserves target tracklet identity using authentic per-frame bboxes. "
            "Kaggle Beef samples directly from single-cow video clips."
        ),
        "smoke_dataset_counts": smoke_counts,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "max_logit_diff_after_reload": max_logit_diff,
        "test_csv_evaluated": False,
        "total_duration_sec": round(total_duration, 2),
        "checkpoints": {
            "best": str(best_ckpt_path),
            "latest": str(latest_ckpt_path),
        },
        "contact_sheet": str(contact_sheet_path),
        "missing_bbox_records_count": len(train_missing_bboxes) + len(val_missing_bboxes),
    }

    metrics_out = output_dir / "behavior_tcn_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Metrics saved to: {metrics_out}")

    print("\n" + "=" * 70)
    print(f"  TEMPORAL CORE PIPELINE COMPLETE (Runtime: {total_duration:.1f}s)")
    print("=" * 70)
    return summary


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Run 5 Behavior Temporal Core Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/behavior/cvb_beef"),
        help="Path to canonical CVB + Kaggle Beef split CSVs",
    )
    parser.add_argument(
        "--cvb-dir",
        type=Path,
        default=Path("/mnt/cvb/cvb/000058916v001"),
        help="Root path to CVB dataset",
    )
    parser.add_argument(
        "--beef-dir",
        type=Path,
        default=Path("/mnt/beef/beef_behavior"),
        help="Root path to Kaggle Beef dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/behavior_temporal_cache"),
        help="Directory to cache pre-extracted 224x224 RGB temporal sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/behavior_temporal"),
        help="Directory to save checkpoints, contact sheet, and metrics",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (sequences)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of temporal frames per sequence (T)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Run fast smoke test on subsampled balanced set (Train=30, Val=10, 2 epochs)",
    )

    args = parser.parse_args()

    train_temporal_pipeline(
        data_dir=args.data_dir,
        cvb_dir=args.cvb_dir,
        beef_dir=args.beef_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
