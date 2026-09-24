# -*- coding: utf-8 -*-
"""
scripts/train_mtl_e1_hard_shared.py — Phase 3 Run 7: E1 Hard-Shared Multi-Task Learning Control
==============================================================================================
Establishes the canonical hard-shared multi-task baseline across the three core tasks:
  1. ScienceDB Body Condition Scoring (BCS) — 4-channel input, 512-D spatial feature, Ordinal BCE head
  2. CVB + Beef Behavior Recognition (Behavior) — 8-frame sequence, shared spatial feature extractor, 1D TCN, CrossEntropy
  3. SideViewCows2026 Cow Re-Identification (Re-ID) — 4-channel input, 512-D spatial feature, Linear(512, 41), CrossEntropy

Scientific Role:
  Run 7 is the primary E1 control experiment to establish whether forcing the three disparate
  tasks to share the exact same visual representation produces negative transfer relative to
  the selected single-task models (Run 4 BCS, Run 5 Behavior, Run 6 Re-ID).
  It does NOT include adapters, task-private trunks, gates, PCGrad, GradNorm, or dynamic weights.

Architecture:
  - Exactly ONE shared 4-channel ImageNet-pretrained ResNet-18 spatial feature extractor.
    * RGB conv1 weights initialized from ImageNet ResNet-18.
    * 4th mask-channel weights initialized deterministically from the mean of RGB weights.
    * Trainable parameters in shared backbone: 11,179,648 (+3,136 over 3-channel baseline).
  - Task-specific heads:
    * BCS: Cumulative Frank & Hall (2001) Ordinal BCE head (Linear(512, 4), 2,052 params).
    * Behavior: Lightweight 1D Temporal Convolutional Network (TCN, 2 Conv1d blocks + Linear(256, 5), 723,973 params).
    * Re-ID: L2-normalized 512-D representation + Linear(512, 41) classifier (21,033 params).
  - Total E1 parameters: exactly 11,926,706 trainable parameters.

Task-Balanced Update Schedule:
  - One super-step:
      1. Forward BCS batch -> compute BCS loss -> backward(w_bcs * loss_bcs)
      2. Forward Behavior batch -> compute Behavior loss -> backward(w_beh * loss_beh)
      3. Forward Re-ID batch -> compute Re-ID loss -> backward(w_reid * loss_reid)
      4. optimizer.step()
  - Fixed equal task weights: w_bcs = 1.0, w_behavior = 1.0, w_reid = 1.0.
  - Zero unavailable-label padding: batches are task-specific, never mixed with dummy targets.
  - Gradients accumulate directly into the SINGLE shared ResNet-18 visual trunk.

Checkpoint Selection:
  - val_E1_objective = (BCS_val_loss + Behavior_val_loss + ReID_val_loss) / 3.0
  - Model selection strictly by minimum val_E1_objective; canonical test sets remain strictly untouched.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import cv2
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------
# Constants & Taxonomy
# ------------------------------------------------------------------------------
REAL_BCS_CLASSES = [3.25, 3.50, 3.75, 4.00, 4.25]
NUM_BCS_CLASSES = len(REAL_BCS_CLASSES)
BCS_ORDINAL_THRESHOLDS = NUM_BCS_CLASSES - 1  # 4 binary threshold tasks
BCS_LABEL_TO_IDX = {3.25: 0, 3.50: 1, 3.75: 2, 4.00: 3, 4.25: 4}
BCS_IDX_TO_LABEL = {0: 3.25, 1: 3.50, 2: 3.75, 3: 4.00, 4: 4.25}

BEHAVIOR_CLASSES = ["Standing", "Lying", "Feeding", "Drinking", "Walking"]
NUM_BEHAVIOR_CLASSES = len(BEHAVIOR_CLASSES)
BEHAVIOR_CLASS_TO_IDX = {c: i for i, c in enumerate(BEHAVIOR_CLASSES)}
BEHAVIOR_IDX_TO_CLASS = {i: c for i, c in enumerate(BEHAVIOR_CLASSES)}

NUM_REID_TRAIN_COWS = 41

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), timeout=5)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def set_seed(seed: int = 2026):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. ARCHITECTURE: EXACTLY ONE SHARED 4-CHANNEL RESNET-18 VISUAL TRUNK
# ==============================================================================
class ResNet18SharedBackbone(nn.Module):
    """
    Unified 4-channel spatial feature extractor.
    Shared across BCS, Behavior (all 8 frames), and Re-ID.
    Input : [N, 4, 224, 224] (RGB ImageNet-normalized + binary mask in {0.0, 1.0})
    Output: [N, 512] spatial feature vector
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        old_conv1 = base.conv1
        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_conv1.weight
            # Deterministic initialization of 4th mask channel from channel mean of RGB weights
            new_conv1.weight[:, 3:4, :, :] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.conv1 = new_conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)  # [N, 512]


# ------------------------------------------------------------------------------
# Task Heads
# ------------------------------------------------------------------------------
class BCSOrdinalHead(nn.Module):
    """Cumulative Frank & Hall (2001) ordinal head: Linear(512, 4)."""
    def __init__(self, in_features: int = 512, num_classes: int = NUM_BCS_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes - 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # [B, 4]


class BehaviorTemporalTCN(nn.Module):
    """
    Lightweight 1D Temporal Convolutional Network.
    Matches Run 5: exactly 2 temporal Conv1d blocks with GELU, BatchNorm1d,
    dropout=0.2, residual skips, AdaptiveAvgPool1d, and Linear classifier.
    """
    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = NUM_BEHAVIOR_CLASSES,
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

        # Temporal pooling & classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T=8, in_features=512] -> transpose to [B, 512, T]
        x = x.transpose(1, 2)

        # Block 1
        res = self.res1(x)
        x = self.drop1(self.act1(self.bn1(self.conv1(x)))) + res

        # Block 2
        res = x
        x = self.drop2(self.act2(self.bn2(self.conv2(x)))) + res

        # Temporal pooling: [B, hidden_dim, 1] -> [B, hidden_dim]
        pooled = self.pool(x).squeeze(-1)
        logits = self.classifier(pooled)  # [B, 5]
        return logits


class ReIDHead(nn.Module):
    """Re-ID Linear classifier over 41 training cows."""
    def __init__(self, in_features: int = 512, num_classes: int = NUM_REID_TRAIN_COWS):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: raw features [B, 512]
        normalized_embeddings = F.normalize(x, p=2, dim=1)  # [B, 512]
        logits = self.classifier(x)  # [B, 41]
        return logits, normalized_embeddings


# ------------------------------------------------------------------------------
# Unified E1 Hard-Shared Model
# ------------------------------------------------------------------------------
class MTLE1HardSharedModel(nn.Module):
    """
    Run 7 E1 Multi-Task Learning Network:
      One shared 4-channel ResNet-18 spatial feature extractor + 3 task heads.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet18SharedBackbone(pretrained=pretrained)
        self.bcs_head = BCSOrdinalHead(in_features=512, num_classes=NUM_BCS_CLASSES)
        self.behavior_tcn = BehaviorTemporalTCN(in_features=512, hidden_dim=256, num_classes=NUM_BEHAVIOR_CLASSES, dropout=0.2)
        self.reid_head = ReIDHead(in_features=512, num_classes=NUM_REID_TRAIN_COWS)

    def forward_bcs(self, x_bcs: torch.Tensor) -> torch.Tensor:
        """Forward BCS image [B, 4, 224, 224] -> [B, 4] ordinal logits."""
        features = self.backbone(x_bcs)
        logits = self.bcs_head(features)
        return logits

    def forward_behavior(self, x_beh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Behavior sequence [B, 8, 4, 224, 224] -> [B, 5] logits.
        Each of the 8 frames passes through the EXACT SAME shared backbone!
        """
        B, T, C, H, W = x_beh.shape
        x_flat = x_beh.view(B * T, C, H, W)
        feats_flat = self.backbone(x_flat)  # [B*8, 512]
        feats = feats_flat.view(B, T, -1)   # [B, 8, 512]
        logits = self.behavior_tcn(feats)   # [B, 5]
        return logits, feats

    def forward_reid(self, x_reid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Re-ID image [B, 4, 224, 224] -> [B, 41] logits + [B, 512] L2 embeddings."""
        features = self.backbone(x_reid)
        logits, normalized_embeddings = self.reid_head(features)
        return logits, normalized_embeddings

    def count_parameters(self) -> Dict[str, int]:
        backbone_p = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        bcs_p = sum(p.numel() for p in self.bcs_head.parameters() if p.requires_grad)
        beh_p = sum(p.numel() for p in self.behavior_tcn.parameters() if p.requires_grad)
        reid_p = sum(p.numel() for p in self.reid_head.parameters() if p.requires_grad)
        total_p = backbone_p + bcs_p + beh_p + reid_p
        return {
            "shared_backbone_parameters": backbone_p,
            "bcs_head_parameters": bcs_p,
            "behavior_tcn_parameters": beh_p,
            "reid_head_parameters": reid_p,
            "total_trainable_parameters": total_p,
        }

    def assert_hard_sharing(self):
        """Programmatically proves that exactly ONE visual ResNet-18 trunk exists."""
        # 1. Verify backbone is an instance of ResNet18SharedBackbone
        assert isinstance(self.backbone, ResNet18SharedBackbone), "Backbone is not ResNet18SharedBackbone"
        
        # 2. Verify no duplicate Conv2d trunk exists in heads
        for name, module in self.named_children():
            if name != "backbone":
                for sub_name, sub_module in module.named_modules():
                    assert not isinstance(sub_module, nn.Conv2d), (
                        f"Task head '{name}' contains illegal 2D spatial convolution: '{sub_name}'. "
                        f"All spatial convolutions MUST reside exclusively in the shared backbone!"
                    )
        
        # 3. Verify parameter counts match certified numbers
        p_counts = self.count_parameters()
        expected_backbone = 11179648
        expected_bcs = 2052
        expected_beh = 723973
        expected_reid = 21033
        expected_total = 11926706

        assert p_counts["shared_backbone_parameters"] == expected_backbone, (
            f"Backbone param mismatch: expected {expected_backbone}, got {p_counts['shared_backbone_parameters']}"
        )
        assert p_counts["bcs_head_parameters"] == expected_bcs, (
            f"BCS head param mismatch: expected {expected_bcs}, got {p_counts['bcs_head_parameters']}"
        )
        assert p_counts["behavior_tcn_parameters"] == expected_beh, (
            f"Behavior TCN param mismatch: expected {expected_beh}, got {p_counts['behavior_tcn_parameters']}"
        )
        assert p_counts["reid_head_parameters"] == expected_reid, (
            f"Re-ID head param mismatch: expected {expected_reid}, got {p_counts['reid_head_parameters']}"
        )
        assert p_counts["total_trainable_parameters"] == expected_total, (
            f"Total param mismatch: expected {expected_total}, got {p_counts['total_trainable_parameters']}"
        )
        return True


# ==============================================================================
# 2. TARGET CONVERSION & LOSS FUNCTIONS
# ==============================================================================
def ordinal_targets_from_class_indices(
    class_indices: torch.Tensor,
    num_classes: int = NUM_BCS_CLASSES,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Cumulative Frank & Hall (2001) binary threshold targets."""
    batch_size = class_indices.size(0)
    k_minus_1 = num_classes - 1
    thresholds = torch.arange(k_minus_1, device=device).unsqueeze(0).expand(batch_size, -1)
    targets = (class_indices.unsqueeze(1) > thresholds).float()
    return targets


def logits_to_bcs_predictions(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    probs = torch.sigmoid(logits)
    pred_indices = (probs > 0.5).sum(dim=1).cpu().numpy()
    pred_indices = np.clip(pred_indices, 0, NUM_BCS_CLASSES - 1)
    real_bcs = np.array([BCS_IDX_TO_LABEL[idx] for idx in pred_indices])
    return pred_indices, real_bcs


def evaluate_bcs_metrics(y_true_indices: np.ndarray, y_pred_indices: np.ndarray) -> Dict[str, float]:
    y_true_real = np.array([BCS_IDX_TO_LABEL[idx] for idx in y_true_indices])
    y_pred_real = np.array([BCS_IDX_TO_LABEL[idx] for idx in y_pred_indices])

    mae_real = float(np.mean(np.abs(y_true_real - y_pred_real)))
    acc_0 = float(np.mean(y_true_indices == y_pred_indices))
    acc_1 = float(np.mean(np.abs(y_true_indices - y_pred_indices) <= 1))
    bal_acc = float(balanced_accuracy_score(y_true_indices, y_pred_indices))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))

    return {
        "real_mae": round(mae_real, 4),
        "acc_0": round(acc_0, 4),
        "acc_1": round(acc_1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
    }


def evaluate_behavior_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_dict = {BEHAVIOR_CLASSES[i]: round(float(per_class[i]), 4) for i in range(len(per_class))}

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_dict,
    }


def evaluate_reid_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "top1_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
    }


# ==============================================================================
# 3. DATA AUGMENTATION & PREPROCESSING (MATCHED TO RUNS 4, 5, 6)
# ==============================================================================
def apply_gpu_augmentations_bcs(batch: torch.Tensor, is_train: bool = True) -> torch.Tensor:
    """
    BCS GPU Augmentation (Runs 4 matched):
    Input: [B, 4, 224, 224] uint8 on GPU (RGB 0..255, Mask 0..1).
    Output: [B, 4, 224, 224] float32 on GPU.
    """
    if is_train:
        if torch.rand(1).item() < 0.5:
            batch = torch.flip(batch, dims=[-1])
        angle = float(torch.empty(1).uniform_(-15.0, 15.0))
        rgb = TF.rotate(batch[:, :3], angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(batch[:, 3:4], angle, interpolation=TF.InterpolationMode.NEAREST)
        if torch.rand(1).item() < 0.5:
            b_factor = float(torch.empty(1).uniform_(0.9, 1.1))
            rgb = TF.adjust_brightness(rgb, b_factor)
        if torch.rand(1).item() < 0.5:
            c_factor = float(torch.empty(1).uniform_(0.9, 1.1))
            rgb = TF.adjust_contrast(rgb, c_factor)
    else:
        rgb = batch[:, :3]
        mask = batch[:, 3:4]

    rgb_norm = TF.normalize(rgb.float().div(255.0), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return torch.cat([rgb_norm, mask.float()], dim=1)


# ==============================================================================
# 4. DATASETS & RAM PRELOADING
# ==============================================================================
class BCSMTLDataset(Dataset):
    """
    Loads pre-assembled monolithic uint8 tensors from /mtl-data/bcs/
    train_bcs_224.pt (34,369 samples) and val_bcs_224.pt (7,817 samples).
    """
    def __init__(self, pt_path: Path, is_train: bool = True, max_samples: Optional[int] = None):
        assert pt_path.exists(), f"BCS tensor missing: {pt_path}"
        payload = torch.load(pt_path, weights_only=False)
        self.tensors = payload["tensors"]  # [N, 4, 224, 224] uint8
        self.targets = payload["targets"]  # [N] class indices 0..4
        self.raw_labels = payload["raw_labels"]  # [N] float BCS
        self.is_train = is_train

        if max_samples is not None and max_samples < len(self.tensors):
            self.tensors = self.tensors[:max_samples]
            self.targets = self.targets[:max_samples]
            self.raw_labels = self.raw_labels[:max_samples]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # Returns raw uint8 tensor; batched GPU augmentation applied in training loop
        return self.tensors[idx], self.targets[idx], float(self.raw_labels[idx])


class BehaviorMTLDataset(Dataset):
    """
    Loads Behavior sequences from /mtl-data/behavior/
    Supports RAM preloading via ThreadPoolExecutor.
    """
    def __init__(
        self,
        manifest_csv: Path,
        behavior_root: Path,
        is_train: bool = True,
        max_samples: Optional[int] = None,
        num_workers: int = 32,
    ):
        assert manifest_csv.exists(), f"Missing Behavior manifest: {manifest_csv}"
        df = pd.read_csv(manifest_csv)
        if max_samples is not None and max_samples < len(df):
            df = df.head(max_samples)

        self.df = df.reset_index(drop=True)
        self.behavior_root = behavior_root
        self.is_train = is_train
        self.num_frames = 8

        self.samples = []
        for _, row in self.df.iterrows():
            sid = str(row["sample_id"])
            cls_name = str(row["behavior_canonical"])
            cls_idx = BEHAVIOR_CLASS_TO_IDX[cls_name]
            self.samples.append((sid, cls_idx))

        # RAM Preloading into uint8 [N, 8, 4, 224, 224]
        N = len(self.samples)
        print(f"[*] Preloading {N} Behavior sequences into RAM ({'Train' if is_train else 'Val'})...", flush=True)
        t0 = time.time()
        self.cached_seqs = np.empty((N, 8, 4, 224, 224), dtype=np.uint8)

        def _load_seq(idx: int):
            sid, _ = self.samples[idx]
            s_dir = self.behavior_root / sid
            for t in range(8):
                f_p = s_dir / f"frame_{t:02d}.jpg"
                m_p = s_dir / f"mask_{t:02d}.png"
                img_bgr = cv2.imread(str(f_p), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise IOError(f"Missing frame {f_p}")
                if img_bgr.shape[0] != 224 or img_bgr.shape[1] != 224:
                    img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                mask_gray = cv2.imread(str(m_p), cv2.IMREAD_GRAYSCALE)
                if mask_gray is None:
                    raise IOError(f"Missing mask {m_p}")
                if mask_gray.shape[0] != 224 or mask_gray.shape[1] != 224:
                    mask_gray = cv2.resize(mask_gray, (224, 224), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_gray > 127).astype(np.uint8)

                self.cached_seqs[idx, t, :3] = rgb.transpose(2, 0, 1)
                self.cached_seqs[idx, t, 3] = mask_bin

        workers = max(1, min(num_workers, 32))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_load_seq, range(N)))

        elapsed = time.time() - t0
        mb = self.cached_seqs.nbytes / (1024 * 1024)
        print(f"    ✓ Preloaded {N} sequences ({mb:.1f} MB) in {elapsed:.1f}s ({N/max(0.1, elapsed):.1f} seq/s).", flush=True)

        self.cached_tensors = torch.from_numpy(self.cached_seqs)
        self.cached_targets = torch.tensor([s[1] for s in self.samples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # seq: [8, 4, 224, 224] uint8
        seq = self.cached_tensors[idx]
        target = self.cached_targets[idx]
        sid, _ = self.samples[idx]

        # Apply synchronized temporal augmentations
        if self.is_train and torch.rand(1).item() < 0.5:
            seq = torch.flip(seq, dims=[-1])

        rgb = seq[:, :3, :, :]
        mask = seq[:, 3:4, :, :].float()

        if self.is_train and torch.rand(1).item() < 0.5:
            b_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
            c_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
            rgb = TF.adjust_brightness(rgb, b_factor)
            rgb = TF.adjust_contrast(rgb, c_factor)

        rgb_norm = TF.normalize(rgb.float().div(255.0), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        seq_float = torch.cat([rgb_norm, mask], dim=1)  # [8, 4, 224, 224] float32
        return seq_float, target, sid


class ReIDMTLDataset(Dataset):
    """
    Loads SideViewCows2026 Train/Val pairs under Zero-Copy Policy.
    Filters strictly for the 41 representation-learning cows.
    Enforces held_out_cow_overlap == 0 against the 69 evaluation cows.
    Preloads all crops into uint8 RAM tensor.
    """
    def __init__(
        self,
        protocols_dir: Path,
        sideview_root: Path,
        split: str = "train",
        max_samples: Optional[int] = None,
        margin_fraction: float = 0.05,
        num_workers: int = 32,
    ):
        assert protocols_dir.exists(), f"Protocols dir missing: {protocols_dir}"
        assert sideview_root.exists(), f"SideView root missing: {sideview_root}"

        df_a = pd.read_csv(protocols_dir / "protocol_cross_setting.csv")
        train_cows = sorted(df_a.loc[df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())
        eval_cows = set(df_a.loc[~df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())

        assert len(train_cows) == 41, f"Expected 41 training cows, got {len(train_cows)}"
        assert len(eval_cows) == 69, f"Expected 69 evaluation cows, got {len(eval_cows)}"
        assert not set(train_cows).intersection(eval_cows), "LEAKAGE: Train and evaluation cows overlap!"

        self.cow_to_label = {cow: i for i, cow in enumerate(train_cows)}

        df_d = pd.read_csv(protocols_dir / "protocol_closed_set.csv")
        df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
        split_df = df_d_41[df_d_41["closed_set_split"].eq(split)].reset_index(drop=True)

        if max_samples is not None and max_samples < len(split_df):
            split_df = split_df.head(max_samples)

        self.df = split_df
        self.is_train = (split == "train")
        self.margin_fraction = margin_fraction

        self.records = []
        for _, row in self.df.iterrows():
            cow_id = str(row["individual_id"])
            img_rel = str(row["image_path"]).replace("\\", "/").split("sideviewcows2026/")[-1].lstrip("/")
            mask_rel = str(row["mask_path"]).replace("\\", "/").split("sideviewcows2026/")[-1].lstrip("/")
            img_p = sideview_root / img_rel
            mask_p = sideview_root / mask_rel
            self.records.append((str(img_p), str(mask_p), self.cow_to_label[cow_id], cow_id))

        N = len(self.records)
        print(f"[*] Preloading {N} Re-ID pairs into RAM ({split.upper()})...", flush=True)
        t0 = time.time()
        self.cached_crops = np.empty((N, 4, 224, 224), dtype=np.uint8)

        def _load_pair(idx: int):
            img_p, mask_p, _, _ = self.records[idx]
            img_bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
            mask_gray = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask_gray is None:
                raise IOError(f"Failed to read pair {img_p} | {mask_p}")

            mask_bool = mask_gray > 0
            foreground = np.argwhere(mask_bool)
            if foreground.size == 0:
                raise ValueError(f"Empty GT mask in {mask_p}")

            y1, x1 = foreground.min(axis=0).tolist()
            y2_inc, x2_inc = foreground.max(axis=0).tolist()
            x2, y2 = int(x2_inc) + 1, int(y2_inc) + 1
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(math.ceil(bw * self.margin_fraction)), int(math.ceil(bh * self.margin_fraction))

            h, w = mask_gray.shape
            cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
            cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)

            crop_img = img_bgr[cy1:cy2, cx1:cx2]
            crop_mask = mask_gray[cy1:cy2, cx1:cx2]

            crop_img_224 = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_LINEAR)
            crop_mask_224 = cv2.resize(crop_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

            rgb = cv2.cvtColor(crop_img_224, cv2.COLOR_BGR2RGB)
            mask_bin = (crop_mask_224 > 127).astype(np.uint8)

            self.cached_crops[idx, :3] = rgb.transpose(2, 0, 1)
            self.cached_crops[idx, 3] = mask_bin

        workers = max(1, min(num_workers, 32))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_load_pair, range(N)))

        elapsed = time.time() - t0
        mb = self.cached_crops.nbytes / (1024 * 1024)
        print(f"    ✓ Preloaded {N} Re-ID crops ({mb:.1f} MB) in {elapsed:.1f}s ({N/max(0.1, elapsed):.1f} img/s).", flush=True)

        self.cached_tensors = torch.from_numpy(self.cached_crops)
        self.cached_labels = torch.tensor([r[2] for r in self.records], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        crop = self.cached_tensors[idx]
        label = self.cached_labels[idx]
        cow_id = self.records[idx][3]

        if self.is_train and torch.rand(1).item() < 0.5:
            crop = torch.flip(crop, dims=[-1])

        rgb = crop[:3]
        mask = crop[3:4].float()

        if self.is_train and torch.rand(1).item() < 0.5:
            b_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
            c_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
            rgb = TF.adjust_brightness(rgb, b_factor)
            rgb = TF.adjust_contrast(rgb, c_factor)

        rgb_norm = TF.normalize(rgb.float().div(255.0), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        crop_float = torch.cat([rgb_norm, mask], dim=0)  # [4, 224, 224] float32
        return crop_float, label, cow_id


# ==============================================================================
# 5. DETERMINISTIC TASK CYCLER (FOR TASK-BALANCED MTL SCHEDULE)
# ==============================================================================
class DeterministicTaskCycler:
    """
    Deterministic infinite cyclic iterator over a DataLoader.
    Tracks exact epochs cycled and batches drawn.
    """
    def __init__(self, dataloader: DataLoader, name: str = "task"):
        self.dataloader = dataloader
        self.name = name
        self.epoch_cycles = 0
        self.total_batches_drawn = 0
        self.iterator: Optional[Iterator] = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.dataloader)
        try:
            batch = next(self.iterator)
            self.total_batches_drawn += 1
            return batch
        except StopIteration:
            self.epoch_cycles += 1
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
            self.total_batches_drawn += 1
            return batch


# ==============================================================================
# 6. VALIDATION EVALUATION GATES
# ==============================================================================
@torch.no_grad()
def evaluate_mtl_validation(
    model: MTLE1HardSharedModel,
    val_loader_bcs: DataLoader,
    val_loader_beh: DataLoader,
    val_loader_reid: DataLoader,
    criterion_bcs: nn.Module,
    criterion_beh: nn.Module,
    criterion_reid: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates all three tasks independently on their respective validation splits.
    Selects model strictly via predefined val_E1_objective = (loss_bcs + loss_beh + loss_reid) / 3.0.
    """
    model.eval()

    # 1. Evaluate Task A: BCS
    bcs_losses = []
    bcs_trues, bcs_preds = [], []
    for imgs, target_indices, _ in val_loader_bcs:
        imgs = imgs.to(device, non_blocking=True)
        if imgs.dtype == torch.uint8:
            imgs = apply_gpu_augmentations_bcs(imgs, is_train=False)
        target_indices = target_indices.to(device, non_blocking=True)
        targets_ordinal = ordinal_targets_from_class_indices(target_indices, num_classes=NUM_BCS_CLASSES, device=device)

        logits = model.forward_bcs(imgs)
        loss = criterion_bcs(logits, targets_ordinal)
        bcs_losses.append(loss.item())

        preds, _ = logits_to_bcs_predictions(logits)
        bcs_preds.extend(preds.tolist())
        bcs_trues.extend(target_indices.cpu().numpy().tolist())

    val_loss_bcs = float(np.mean(bcs_losses))
    bcs_metrics = evaluate_bcs_metrics(np.array(bcs_trues), np.array(bcs_preds))

    # 2. Evaluate Task B: Behavior
    beh_losses = []
    beh_trues, beh_preds = [], []
    for seqs, targets, _ in val_loader_beh:
        seqs = seqs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, _ = model.forward_behavior(seqs)
        loss = criterion_beh(logits, targets)
        beh_losses.append(loss.item())

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        beh_preds.extend(preds.tolist())
        beh_trues.extend(targets.cpu().numpy().tolist())

    val_loss_beh = float(np.mean(beh_losses))
    beh_metrics = evaluate_behavior_metrics(np.array(beh_trues), np.array(beh_preds))

    # 3. Evaluate Task C: Re-ID
    reid_losses = []
    reid_trues, reid_preds = [], []
    for imgs, targets, _ in val_loader_reid:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, _ = model.forward_reid(imgs)
        loss = criterion_reid(logits, targets)
        reid_losses.append(loss.item())

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        reid_preds.extend(preds.tolist())
        reid_trues.extend(targets.cpu().numpy().tolist())

    val_loss_reid = float(np.mean(reid_losses))
    reid_metrics = evaluate_reid_metrics(np.array(reid_trues), np.array(reid_preds))

    # Composite E1 Validation Objective
    val_e1_objective = float((val_loss_bcs + val_loss_beh + val_loss_reid) / 3.0)

    return {
        "val_e1_objective": round(val_e1_objective, 5),
        "bcs": {
            "val_loss": round(val_loss_bcs, 5),
            **bcs_metrics,
        },
        "behavior": {
            "val_loss": round(val_loss_beh, 5),
            **beh_metrics,
        },
        "reid": {
            "val_loss": round(val_loss_reid, 5),
            **reid_metrics,
        },
    }


# ==============================================================================
# 7. MAIN TRAINING ENGINE
# ==============================================================================
def train_mtl_e1_pipeline(
    bcs_data_dir: Path,
    behavior_data_dir: Path,
    reid_data_dir: Path,
    reid_protocols_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size_bcs: int = 64,
    batch_size_beh: int = 8,
    batch_size_reid: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    smoke: bool = False,
    device_name: Optional[str] = None,
    num_preload_workers: int = 32,
    active_git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main training loop for Run 7: E1 Hard-Shared MTL Control.
    """
    set_seed(2026)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)
    active_sha = active_git_sha if active_git_sha else get_git_commit_sha()

    print("\n" + "=" * 78)
    print("  PHASE 3 RUN 7: E1 HARD-SHARED MULTI-TASK LEARNING CONTROL")
    print(f"  Device           : {device}")
    print(f"  Git Commit SHA   : {active_sha}")
    print(f"  Mode             : {'SMOKE TEST' if smoke else 'FULL 30-EPOCH TRAINING'}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Sizes      : BCS={batch_size_bcs}, Behavior={batch_size_beh}, Re-ID={batch_size_reid}")
    print(f"  Task Weights     : w_BCS=1.0, w_Behavior=1.0, w_ReID=1.0 (Fixed Control)")
    print(f"  Output Dir       : {output_dir}")
    print("=" * 78)

    # 1. Instantiate Unified Model
    model = MTLE1HardSharedModel(pretrained=True).to(device)
    model.assert_hard_sharing()

    param_summary = model.count_parameters()
    print("\n[*] Model Architecture & Verified Parameter Distribution:")
    for k, v in param_summary.items():
        print(f"    {k:<32}: {v:,} params")

    # 2. Build Datasets
    print("\n[*] Initializing task datasets and preloading into RAM...")
    max_s_bcs = 16 if smoke else None
    max_s_beh = 8 if smoke else None
    max_s_reid = 16 if smoke else None

    # Task A: BCS
    train_ds_bcs = BCSMTLDataset(bcs_data_dir / "train_bcs_224.pt", is_train=True, max_samples=max_s_bcs)
    val_ds_bcs = BCSMTLDataset(bcs_data_dir / "val_bcs_224.pt", is_train=False, max_samples=(8 if smoke else None))

    # Task B: Behavior
    train_ds_beh = BehaviorMTLDataset(
        behavior_data_dir / "retained_train.csv",
        behavior_data_dir,
        is_train=True,
        max_samples=max_s_beh,
        num_workers=num_preload_workers,
    )
    val_ds_beh = BehaviorMTLDataset(
        behavior_data_dir / "retained_val.csv",
        behavior_data_dir,
        is_train=False,
        max_samples=(4 if smoke else None),
        num_workers=num_preload_workers,
    )

    # Task C: Re-ID
    train_ds_reid = ReIDMTLDataset(
        reid_protocols_dir,
        reid_data_dir,
        split="train",
        max_samples=max_s_reid,
        num_workers=num_preload_workers,
    )
    val_ds_reid = ReIDMTLDataset(
        reid_protocols_dir,
        reid_data_dir,
        split="val",
        max_samples=(8 if smoke else None),
        num_workers=num_preload_workers,
    )

    print(f"\n[*] Verified Dataset Sizes:")
    print(f"    Task A (BCS)     : Train={len(train_ds_bcs):,}, Val={len(val_ds_bcs):,}")
    print(f"    Task B (Behavior): Train={len(train_ds_beh):,}, Val={len(val_ds_beh):,}")
    print(f"    Task C (Re-ID)   : Train={len(train_ds_reid):,}, Val={len(val_ds_reid):,}")

    # 3. DataLoaders
    # Note: Because uint8 tensors are preloaded into RAM, num_workers=0 eliminates IPC serialization overhead
    train_loader_bcs = DataLoader(train_ds_bcs, batch_size=batch_size_bcs, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_bcs = DataLoader(val_ds_bcs, batch_size=batch_size_bcs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader_beh = DataLoader(train_ds_beh, batch_size=batch_size_beh, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_beh = DataLoader(val_ds_beh, batch_size=batch_size_beh, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader_reid = DataLoader(train_ds_reid, batch_size=batch_size_reid, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_reid = DataLoader(val_ds_reid, batch_size=batch_size_reid, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # 4. Super-Step Scheduling Definition
    # Anchor epoch length to BCS loader length (or smoke steps)
    if smoke:
        super_steps_per_epoch = 2
    else:
        super_steps_per_epoch = len(train_loader_bcs)  # 537 steps per epoch

    bcs_cycler = DeterministicTaskCycler(train_loader_bcs, name="BCS")
    beh_cycler = DeterministicTaskCycler(train_loader_beh, name="Behavior")
    reid_cycler = DeterministicTaskCycler(train_loader_reid, name="ReID")

    print("\n[*] Super-Step Scheduling & Oversampling Analysis:")
    print(f"    Super-steps per epoch    : {super_steps_per_epoch}")
    print(f"    BCS batches/epoch        : {super_steps_per_epoch} (approx {super_steps_per_epoch * batch_size_bcs:,} samples; ratio={super_steps_per_epoch / max(1, len(train_loader_bcs)):.2f}x)")
    print(f"    Behavior batches/epoch   : {super_steps_per_epoch} (approx {super_steps_per_epoch * batch_size_beh:,} sequences; ratio={super_steps_per_epoch / max(1, len(train_loader_beh)):.2f}x)")
    print(f"    Re-ID batches/epoch      : {super_steps_per_epoch} (approx {super_steps_per_epoch * batch_size_reid:,} images; ratio={super_steps_per_epoch / max(1, len(train_loader_reid)):.2f}x)")

    # 5. Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion_bcs = nn.BCEWithLogitsLoss()
    criterion_beh = nn.CrossEntropyLoss()
    criterion_reid = nn.CrossEntropyLoss()

    best_val_objective = float("inf")
    best_metrics: Dict[str, Any] = {}
    epoch_history: List[Dict[str, Any]] = []

    best_ckpt_path = output_dir / ("mtl_e1_smoke_best.pth" if smoke else "mtl_e1_best.pth")
    latest_ckpt_path = output_dir / ("mtl_e1_smoke_latest.pth" if smoke else "mtl_e1_latest.pth")

    # 6. Training Loop
    t_train_start = time.time()
    print("\n[*] Launching MTL update schedule...")

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()

        running_loss_bcs = 0.0
        running_loss_beh = 0.0
        running_loss_reid = 0.0
        running_loss_total = 0.0

        for step in range(1, super_steps_per_epoch + 1):
            optimizer.zero_grad()

            # Task 1: BCS forward & backward
            imgs_bcs, target_indices_bcs, _ = next(bcs_cycler)
            imgs_bcs = imgs_bcs.to(device, non_blocking=True)
            if imgs_bcs.dtype == torch.uint8:
                imgs_bcs = apply_gpu_augmentations_bcs(imgs_bcs, is_train=True)
            target_indices_bcs = target_indices_bcs.to(device, non_blocking=True)
            targets_bcs_ordinal = ordinal_targets_from_class_indices(target_indices_bcs, num_classes=NUM_BCS_CLASSES, device=device)

            logits_bcs = model.forward_bcs(imgs_bcs)
            loss_bcs = criterion_bcs(logits_bcs, targets_bcs_ordinal)
            (1.0 * loss_bcs).backward()

            # Task 2: Behavior forward & backward
            seqs_beh, targets_beh, _ = next(beh_cycler)
            seqs_beh = seqs_beh.to(device, non_blocking=True)
            targets_beh = targets_beh.to(device, non_blocking=True)

            logits_beh, _ = model.forward_behavior(seqs_beh)
            loss_beh = criterion_beh(logits_beh, targets_beh)
            (1.0 * loss_beh).backward()

            # Task 3: Re-ID forward & backward
            imgs_reid, targets_reid, _ = next(reid_cycler)
            imgs_reid = imgs_reid.to(device, non_blocking=True)
            targets_reid = targets_reid.to(device, non_blocking=True)

            logits_reid, _ = model.forward_reid(imgs_reid)
            loss_reid = criterion_reid(logits_reid, targets_reid)
            (1.0 * loss_reid).backward()

            # Verify that gradients have reached the shared backbone
            if smoke and step == 1:
                assert model.backbone.conv1.weight.grad is not None, "Gradients failed to reach shared backbone!"
                assert torch.isfinite(model.backbone.conv1.weight.grad).all(), "Non-finite gradient in shared backbone!"

            # Single optimizer update across all accumulated task gradients
            optimizer.step()

            running_loss_bcs += loss_bcs.item()
            running_loss_beh += loss_beh.item()
            running_loss_reid += loss_reid.item()
            running_loss_total += (loss_bcs.item() + loss_beh.item() + loss_reid.item()) / 3.0

        scheduler.step()

        # Epoch training statistics
        train_loss_bcs = running_loss_bcs / super_steps_per_epoch
        train_loss_beh = running_loss_beh / super_steps_per_epoch
        train_loss_reid = running_loss_reid / super_steps_per_epoch
        train_loss_total = running_loss_total / super_steps_per_epoch

        # Validation Evaluation
        val_eval = evaluate_mtl_validation(
            model=model,
            val_loader_bcs=val_loader_bcs,
            val_loader_beh=val_loader_beh,
            val_loader_reid=val_loader_reid,
            criterion_bcs=criterion_bcs,
            criterion_beh=criterion_beh,
            criterion_reid=criterion_reid,
            device=device,
        )

        epoch_val_objective = val_eval["val_e1_objective"]
        is_best = epoch_val_objective < best_val_objective
        if is_best:
            best_val_objective = epoch_val_objective
            best_metrics = val_eval

        epoch_duration = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "duration_seconds": round(epoch_duration, 1),
            "train_loss_total": round(train_loss_total, 5),
            "train_loss_bcs": round(train_loss_bcs, 5),
            "train_loss_beh": round(train_loss_beh, 5),
            "train_loss_reid": round(train_loss_reid, 5),
            **val_eval,
            "is_best": is_best,
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"TrainObj: {train_loss_total:.4f} | "
            f"ValObj: {epoch_val_objective:.4f} {'[BEST]' if is_best else ''} | "
            f"BCS: val_loss={val_eval['bcs']['val_loss']:.4f} MAE={val_eval['bcs']['real_mae']:.4f} | "
            f"Beh: val_loss={val_eval['behavior']['val_loss']:.4f} F1={val_eval['behavior']['macro_f1']:.4f} | "
            f"ReID: val_loss={val_eval['reid']['val_loss']:.4f} Acc={val_eval['reid']['top1_accuracy']:.4f}",
            flush=True,
        )

        # Checkpoint Saving
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_objective": best_val_objective,
            "best_metrics": best_metrics,
            "epoch_history": epoch_history,
            "parameters": param_summary,
            "git_commit_sha": active_sha,
            "seed": 2026,
            "weights": {"w_bcs": 1.0, "w_beh": 1.0, "w_reid": 1.0},
        }

        torch.save(checkpoint_payload, latest_ckpt_path)
        if is_best:
            torch.save(checkpoint_payload, best_ckpt_path)

    total_training_duration = time.time() - t_train_start
    print("\n" + "=" * 78)
    print(f"  RUN 7 {'SMOKE' if smoke else 'TRAINING'} COMPLETED in {total_training_duration:.1f}s")
    print(f"  Best Val E1 Objective : {best_val_objective:.5f}")
    print(f"  Checkpoints Saved     : {best_ckpt_path} and {latest_ckpt_path}")
    print("=" * 78)

    # 7. Checkpoint Reload Verification
    print("[*] Verifying checkpoint reload bit-identity...")
    model.eval()
    reload_model = MTLE1HardSharedModel(pretrained=False).to(device)
    reload_ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    reload_model.load_state_dict(reload_ckpt["model_state_dict"])
    reload_model.eval()

    # Compare logits on fixed probe sample
    probe_x_bcs = torch.zeros((2, 4, 224, 224), device=device)
    with torch.no_grad():
        orig_logits = model.forward_bcs(probe_x_bcs)
        reload_logits = reload_model.forward_bcs(probe_x_bcs)
        max_logit_diff = float((orig_logits - reload_logits).abs().max().item())

    # Also verify best checkpoint loads into a model cleanly
    best_reload_model = MTLE1HardSharedModel(pretrained=False).to(device)
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    best_reload_model.load_state_dict(best_ckpt["model_state_dict"])
    best_reload_model.eval()

    print(f"    ✓ Checkpoint Reload Max Logit Difference: {max_logit_diff:.8f}")
    assert max_logit_diff < 1e-6, f"Checkpoint reload failed bit-identity: diff={max_logit_diff}"

    summary_results = {
        "git_commit_sha": active_sha,
        "smoke": smoke,
        "epochs": epochs,
        "total_duration_seconds": round(total_training_duration, 1),
        "best_val_objective": best_val_objective,
        "best_metrics": best_metrics,
        "checkpoint_reload_max_logit_diff": max_logit_diff,
        "parameters": param_summary,
        "best_checkpoint_path": str(best_ckpt_path),
        "latest_checkpoint_path": str(latest_ckpt_path),
        "epoch_history": epoch_history,
    }

    # Save summary json
    metrics_json_p = output_dir / ("mtl_e1_smoke_metrics.json" if smoke else "mtl_e1_metrics.json")
    with open(metrics_json_p, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"[*] Saved metrics summary to {metrics_json_p}")

    return summary_results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Run 7: E1 Hard-Shared MTL Control")
    parser.add_argument("--bcs-data-dir", type=Path, default=Path("/mtl-data/bcs"))
    parser.add_argument("--behavior-data-dir", type=Path, default=Path("/mtl-data/behavior"))
    parser.add_argument("--sideview-data-dir", type=Path, default=Path("/sideview/sideviewcows2026"))
    parser.add_argument("--reid-protocols-dir", type=Path, default=Path("datasets/id/sideviewcows2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mtl-checkpoints/mtl_e1_hard_shared"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size-bcs", type=int, default=64)
    parser.add_argument("--batch-size-beh", type=int, default=8)
    parser.add_argument("--batch-size-reid", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Execute 2-epoch cheap smoke test")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_mtl_e1_pipeline(
        bcs_data_dir=args.bcs_data_dir,
        behavior_data_dir=args.behavior_data_dir,
        reid_data_dir=args.sideview_data_dir,
        reid_protocols_dir=args.reid_protocols_dir,
        output_dir=args.output_dir,
        epochs=2 if args.smoke else args.epochs,
        batch_size_bcs=8 if args.smoke else args.batch_size_bcs,
        batch_size_beh=4 if args.smoke else args.batch_size_beh,
        batch_size_reid=8 if args.smoke else args.batch_size_reid,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoke=args.smoke,
        device_name=args.device,
    )
