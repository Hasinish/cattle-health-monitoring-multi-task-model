# -*- coding: utf-8 -*-
"""
scripts/train_mtl_e3_modular.py — Phase 3 Run 8: E3 Modular Multi-Task Learning Architecture
===========================================================================================
Implements the canonical E3 Modular Multi-Task Learning model for direct controlled comparison
with Phase 3 Run 7 (E1 Hard-Shared MTL Control):
  1. ScienceDB Body Condition Scoring (BCS) — 4-channel input, shared trunk -> BCS Adapter -> Ordinal BCE head
  2. CVB + Beef Behavior Recognition (Behavior) — 8-frame sequence, shared trunk -> Behavior Adapter -> 1D TCN -> CrossEntropy
  3. SideViewCows2026 Cow Re-Identification (Re-ID) — 4-channel input, shared trunk -> Re-ID Adapter -> Linear(512, 41) -> CrossEntropy

Scientific Role & Controlled Comparison against Run 7 E1:
  Run 7 demonstrated that naive hard parameter sharing across all three tasks produces severe
  negative transfer on held-out test populations:
    - BCS MAE degraded from 0.1709 (Run 4) to 0.1788 (Run 7 E1)
    - Behavior Macro-F1 degraded from 0.7397 (Run 5) to 0.6866 (Run 7 E1) with minority class Walking collapsing to 0.0408
    - Re-ID Barn mAP degraded from 40.68% (Run 6) to 30.37% (Run 7 E1)
  
  Run 8 (E3 Modular MTL) tests the central thesis hypothesis:
    "Can lightweight task-private residual pathways decouple conflicting gradients and mitigate
     negative transfer while preserving the shared cattle visual manifold?"

Architecture:
  - Exactly ONE shared 4-channel ImageNet-pretrained ResNet-18 spatial feature extractor (11,179,648 params).
  - Exactly THREE lightweight task-private residual bottleneck adapters (395,904 params total; 131,968 each):
      * Adapter Structure: Linear(512, 128) -> LayerNorm(128) -> GELU -> Dropout(0.1) -> Linear(128, 512) + Residual Skip
      * Identity Initialization: The task-private adapters use zero-initialized up-projections, so each adapter initially acts as an identity mapping. At initialization, the adapted task feature therefore equals the output of E3's shared backbone before task-specific adaptation is learned.
  - Exactly THREE task heads (100% matched to Runs 4-7):
      * BCS: Cumulative Frank & Hall (2001) Ordinal BCE head (Linear(512, 4), 2,052 params)
      * Behavior: Lightweight 1D Temporal Convolutional Network (TCN, 2 Conv1d blocks + Linear(256, 5), 723,973 params)
      * Re-ID: Unit L2-normalized 512-D embedding + Linear(512, 41) classifier (21,033 params)
  - Total E3 Trainable Parameters: 12,322,610 (+395,904 params / +3.32% over E1).

Task-Balanced Update Schedule (Identical to E1):
  - Fixed equal task weights: w_bcs = 1.0, w_behavior = 1.0, w_reid = 1.0.
  - Super-step gradient accumulation:
      1. Forward BCS batch -> compute BCS loss -> backward(w_bcs * loss_bcs)
      2. Forward Behavior batch -> compute Behavior loss -> backward(w_beh * loss_beh)
      3. Forward Re-ID batch -> compute Re-ID loss -> backward(w_reid * loss_reid)
      4. optimizer.step()
  - Zero unavailable-label padding: batches are strictly task-specific.
  - Task gradients update their respective private adapter + task head, and accumulate into the shared backbone.

Validation Objective (Identical to E1):
  - val_e3_objective = (BCS_val_loss + Behavior_val_loss + ReID_val_loss) / 3.0
  - Model selection strictly by minimum val_e3_objective; canonical test sets remain strictly untouched.
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
# Constants & Taxonomy (Matched 100% to Runs 4-7)
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
# 1. SHARED VISUAL TRUNK (EXACTLY ONE 4-CHANNEL RESNET-18)
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


# ==============================================================================
# 2. TASK-PRIVATE RESIDUAL BOTTLENECK ADAPTER (MODULAR PATHWAY)
# ==============================================================================
class TaskResidualAdapter(nn.Module):
    """
    Lightweight Task-Private Residual Bottleneck Adapter.
    Enables task-specific feature specialization while preserving the shared visual representation.

    Structure:
      x in R^512
      h = down_proj(x)       # [N, 128]
      h = norm(h)            # LayerNorm(128)
      h = act(h)             # GELU()
      h = drop(h)            # Dropout(0.1)
      h = up_proj(h)         # [N, 512]
      out = x + h            # Residual skip connection

    Parameters:
      down_proj: 512 * 128 + 128 = 65,664
      norm:      128 * 2 = 256
      up_proj:   128 * 512 + 512 = 66,048
      Total per adapter: 131,968 parameters.
    """
    def __init__(self, in_features: int = 512, bottleneck_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim

        self.down_proj = nn.Linear(in_features, bottleneck_dim, bias=True)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, in_features, bias=True)

        # Standard Adapter Identity Initialization (Houlsby et al., 2019):
        # Zero-initialize up_proj weights and bias so at initialization:
        # out == x (identity mapping).
        # The adapted task feature initially equals the shared backbone output before task-specific adaptation is learned.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up_proj(h)
        return residual + h


# ==============================================================================
# 3. TASK HEADS (100% MATCHED TO RUNS 4–7)
# ==============================================================================
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
    Matches Run 5 & Run 7: exactly 2 temporal Conv1d blocks with GELU, BatchNorm1d,
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
        # x: raw adapted features [B, 512]
        normalized_embeddings = F.normalize(x, p=2, dim=1)  # [B, 512]
        logits = self.classifier(x)  # [B, 41]
        return logits, normalized_embeddings


# ==============================================================================
# 4. UNIFIED E3 MODULAR MULTI-TASK MODEL
# ==============================================================================
class MTLE3ModularModel(nn.Module):
    """
    Run 8 E3 Multi-Task Learning Network:
      One shared 4-channel ResNet-18 spatial feature extractor (11,179,648 params)
      + 3 Task-Private Residual Bottleneck Adapters (395,904 params total; 131,968 each)
      + 3 Task Heads (747,058 params total: BCS 2,052 + Behavior 723,973 + Re-ID 21,033)
      = 12,322,610 Total Trainable Parameters (+395,904 params / +3.32% over E1).
    """
    def __init__(
        self,
        pretrained: bool = True,
        bottleneck_dim: int = 128,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        # 1. Single Shared Spatial Backbone
        self.backbone = ResNet18SharedBackbone(pretrained=pretrained)

        # 2. Task-Private Modular Pathways (Residual Adapters)
        self.bcs_adapter = TaskResidualAdapter(512, bottleneck_dim=bottleneck_dim, dropout=adapter_dropout)
        self.behavior_adapter = TaskResidualAdapter(512, bottleneck_dim=bottleneck_dim, dropout=adapter_dropout)
        self.reid_adapter = TaskResidualAdapter(512, bottleneck_dim=bottleneck_dim, dropout=adapter_dropout)

        # 3. Task Heads
        self.bcs_head = BCSOrdinalHead(in_features=512, num_classes=NUM_BCS_CLASSES)
        self.behavior_tcn = BehaviorTemporalTCN(in_features=512, hidden_dim=256, num_classes=NUM_BEHAVIOR_CLASSES, dropout=0.2)
        self.reid_head = ReIDHead(in_features=512, num_classes=NUM_REID_TRAIN_COWS)

    def forward_bcs(self, x_bcs: torch.Tensor) -> torch.Tensor:
        """
        Forward BCS image [B, 4, 224, 224]:
          Shared Trunk -> BCS-Private Adapter -> Ordinal Head -> [B, 4] logits.
        """
        feat_shared = self.backbone(x_bcs)
        feat_private = self.bcs_adapter(feat_shared)
        logits = self.bcs_head(feat_private)
        return logits

    def forward_behavior(self, x_beh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Behavior sequence [B, 8, 4, 224, 224]:
          Each frame -> Shared Trunk -> Behavior-Private Adapter -> [B, 8, 512] -> TCN -> [B, 5] logits.
        """
        B, T, C, H, W = x_beh.shape
        x_flat = x_beh.view(B * T, C, H, W)
        feat_shared = self.backbone(x_flat)
        feat_private = self.behavior_adapter(feat_shared)
        feats = feat_private.view(B, T, -1)
        logits = self.behavior_tcn(feats)
        return logits, feats

    def forward_reid(self, x_reid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Re-ID image [B, 4, 224, 224]:
          Shared Trunk -> Re-ID-Private Adapter -> Re-ID Head -> [B, 41] logits + [B, 512] L2 embeddings.
        """
        feat_shared = self.backbone(x_reid)
        feat_private = self.reid_adapter(feat_shared)
        logits, normalized_embeddings = self.reid_head(feat_private)
        return logits, normalized_embeddings

    def count_parameters(self) -> Dict[str, int]:
        backbone_p = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        bcs_adapter_p = sum(p.numel() for p in self.bcs_adapter.parameters() if p.requires_grad)
        beh_adapter_p = sum(p.numel() for p in self.behavior_adapter.parameters() if p.requires_grad)
        reid_adapter_p = sum(p.numel() for p in self.reid_adapter.parameters() if p.requires_grad)
        total_adapter_p = bcs_adapter_p + beh_adapter_p + reid_adapter_p

        bcs_head_p = sum(p.numel() for p in self.bcs_head.parameters() if p.requires_grad)
        beh_head_p = sum(p.numel() for p in self.behavior_tcn.parameters() if p.requires_grad)
        reid_head_p = sum(p.numel() for p in self.reid_head.parameters() if p.requires_grad)
        total_head_p = bcs_head_p + beh_head_p + reid_head_p

        total_p = backbone_p + total_adapter_p + total_head_p
        return {
            "shared_backbone_parameters": backbone_p,
            "bcs_adapter_parameters": bcs_adapter_p,
            "behavior_adapter_parameters": beh_adapter_p,
            "reid_adapter_parameters": reid_adapter_p,
            "total_task_private_adapter_parameters": total_adapter_p,
            "bcs_head_parameters": bcs_head_p,
            "behavior_tcn_parameters": beh_head_p,
            "reid_head_parameters": reid_head_p,
            "total_task_head_parameters": total_head_p,
            "total_trainable_parameters": total_p,
        }

    def assert_modular_sharing(self):
        """Programmatically proves that:
        1. Exactly ONE shared visual ResNet-18 trunk exists.
        2. Exactly THREE task-private residual adapters exist with bottleneck=128.
        3. NO 2D spatial convolution exists in any adapter or head.
        4. Parameter counts match certified numbers exactly.
        """
        assert isinstance(self.backbone, ResNet18SharedBackbone), "Backbone is not ResNet18SharedBackbone"
        assert isinstance(self.bcs_adapter, TaskResidualAdapter), "bcs_adapter is not TaskResidualAdapter"
        assert isinstance(self.behavior_adapter, TaskResidualAdapter), "behavior_adapter is not TaskResidualAdapter"
        assert isinstance(self.reid_adapter, TaskResidualAdapter), "reid_adapter is not TaskResidualAdapter"

        # Verify no illegal Conv2d exists in adapters or heads
        for name, module in self.named_children():
            if name != "backbone":
                for sub_name, sub_module in module.named_modules():
                    assert not isinstance(sub_module, nn.Conv2d), (
                        f"Module '{name}' contains illegal 2D spatial convolution: '{sub_name}'. "
                        f"All spatial convolutions MUST reside exclusively in the shared backbone!"
                    )

        counts = self.count_parameters()
        expected_backbone = 11179648
        expected_adapter = 131968
        expected_total_adapters = 395904
        expected_bcs_head = 2052
        expected_beh_head = 723973
        expected_reid_head = 21033
        expected_total = 12322610

        assert counts["shared_backbone_parameters"] == expected_backbone, (
            f"Backbone param mismatch: expected {expected_backbone}, got {counts['shared_backbone_parameters']}"
        )
        assert counts["bcs_adapter_parameters"] == expected_adapter, (
            f"BCS adapter param mismatch: expected {expected_adapter}, got {counts['bcs_adapter_parameters']}"
        )
        assert counts["behavior_adapter_parameters"] == expected_adapter, (
            f"Behavior adapter param mismatch: expected {expected_adapter}, got {counts['behavior_adapter_parameters']}"
        )
        assert counts["reid_adapter_parameters"] == expected_adapter, (
            f"Re-ID adapter param mismatch: expected {expected_adapter}, got {counts['reid_adapter_parameters']}"
        )
        assert counts["total_task_private_adapter_parameters"] == expected_total_adapters, (
            f"Total adapter param mismatch: expected {expected_total_adapters}, got {counts['total_task_private_adapter_parameters']}"
        )
        assert counts["bcs_head_parameters"] == expected_bcs_head, (
            f"BCS head param mismatch: expected {expected_bcs_head}, got {counts['bcs_head_parameters']}"
        )
        assert counts["behavior_tcn_parameters"] == expected_beh_head, (
            f"Behavior TCN param mismatch: expected {expected_beh_head}, got {counts['behavior_tcn_parameters']}"
        )
        assert counts["reid_head_parameters"] == expected_reid_head, (
            f"Re-ID head param mismatch: expected {expected_reid_head}, got {counts['reid_head_parameters']}"
        )
        assert counts["total_trainable_parameters"] == expected_total, (
            f"Total param mismatch: expected {expected_total}, got {counts['total_trainable_parameters']}"
        )
        return True


# ==============================================================================
# 5. TARGET CONVERSION & LOSS FUNCTIONS
# ==============================================================================
def ordinal_targets_from_class_indices(
    class_indices: torch.Tensor,
    num_classes: int = NUM_BCS_CLASSES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Converts discrete class index k in {0..K-1} to K-1 binary targets: target_j = 1 if k > j else 0."""
    num_thresholds = num_classes - 1
    B = class_indices.shape[0]
    dev = device if device is not None else class_indices.device

    thresholds = torch.arange(num_thresholds, device=dev).unsqueeze(0).expand(B, -1)
    k_expanded = class_indices.unsqueeze(1).expand(-1, num_thresholds)
    targets_binary = (k_expanded > thresholds).float()
    return targets_binary


def bcs_discrete_predictions_from_ordinal_logits(logits: torch.Tensor) -> np.ndarray:
    """Predicts discrete BCS class index from ordinal threshold logits using P(Y > j) > 0.5."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    predicted_indices = np.sum(binary_preds, axis=1)
    return predicted_indices


def evaluate_bcs_metrics(true_indices: np.ndarray, pred_indices: np.ndarray) -> Dict[str, float]:
    """Computes real BCS MAE, exact accuracy, acc within +/- 0.25 tolerance, and macro-F1."""
    true_labels = np.array([BCS_IDX_TO_LABEL[i] for i in true_indices])
    pred_labels = np.array([BCS_IDX_TO_LABEL[i] for i in pred_indices])

    mae = float(np.mean(np.abs(true_labels - pred_labels)))
    acc_0 = float(accuracy_score(true_indices, pred_indices) * 100.0)
    acc_1 = float(np.mean(np.abs(true_indices - pred_indices) <= 1) * 100.0)
    bal_acc = float(balanced_accuracy_score(true_indices, pred_indices) * 100.0)
    macro_f1 = float(f1_score(true_indices, pred_indices, average="macro", zero_division=0))

    return {
        "real_mae": round(mae, 4),
        "exact_accuracy": round(acc_0, 2),
        "acc_within_1_class": round(acc_1, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 4),
    }


def evaluate_behavior_metrics(true_indices: np.ndarray, pred_indices: np.ndarray) -> Dict[str, Any]:
    """Computes Behavior classification metrics."""
    acc = float(accuracy_score(true_indices, pred_indices) * 100.0)
    bal_acc = float(balanced_accuracy_score(true_indices, pred_indices) * 100.0)
    macro_f1 = float(f1_score(true_indices, pred_indices, average="macro", zero_division=0))
    per_class_f1 = f1_score(true_indices, pred_indices, average=None, zero_division=0)

    class_f1_dict = {
        BEHAVIOR_CLASSES[i]: round(float(per_class_f1[i]), 4)
        for i in range(len(BEHAVIOR_CLASSES))
    }
    return {
        "accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": class_f1_dict,
    }


def evaluate_reid_metrics(true_indices: np.ndarray, pred_indices: np.ndarray) -> Dict[str, float]:
    """Computes Re-ID validation classification metrics."""
    acc = float(accuracy_score(true_indices, pred_indices) * 100.0)
    bal_acc = float(balanced_accuracy_score(true_indices, pred_indices) * 100.0)
    macro_f1 = float(f1_score(true_indices, pred_indices, average="macro", zero_division=0))
    return {
        "top1_accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 4),
    }


# ==============================================================================
# 6. IN-MEMORY HIGH-SPEED DATASETS (100% MATCHED TO RUN 7)
# ==============================================================================
class BCSMTLDataset(Dataset):
    """Loads pre-cached ScienceDB BCS uint8 tensors directly from persistent volume."""
    def __init__(self, tensor_path: Path, is_train: bool = True, max_samples: Optional[int] = None):
        assert tensor_path.exists(), f"BCS tensor missing: {tensor_path}"
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        self.tensors = payload["tensors"]
        self.targets = payload["targets"]
        self.sample_ids = payload.get("sample_ids", [str(i) for i in range(len(self.targets))])

        if max_samples is not None and max_samples < len(self.targets):
            self.tensors = self.tensors[:max_samples]
            self.targets = self.targets[:max_samples]
            self.sample_ids = self.sample_ids[:max_samples]

        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img = self.tensors[idx]
        target = self.targets[idx]
        sid = self.sample_ids[idx]
        return img, target, sid


def apply_gpu_augmentations_bcs(imgs_batch: torch.Tensor, is_train: bool = True) -> torch.Tensor:
    """Transforms [B, 4, 224, 224] uint8 tensor on GPU into normalized float32."""
    B = imgs_batch.shape[0]
    rgb = imgs_batch[:, :3, :, :]
    mask = imgs_batch[:, 3:4, :, :].float()

    if is_train:
        flip_mask = torch.rand(B, device=imgs_batch.device) < 0.5
        if flip_mask.any():
            rgb[flip_mask] = torch.flip(rgb[flip_mask], dims=[-1])
            mask[flip_mask] = torch.flip(mask[flip_mask], dims=[-1])

    rgb_float = rgb.float().div(255.0)
    for c in range(3):
        rgb_float[:, c, :, :] = (rgb_float[:, c, :, :] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    out = torch.cat([rgb_float, mask], dim=1)
    return out


class BehaviorMTLDataset(Dataset):
    """Loads CVB + Kaggle Beef sequences from individual folders and preloads into uint8 RAM tensor."""
    def __init__(
        self,
        manifest_csv: Path,
        behavior_root: Path,
        is_train: bool = True,
        max_samples: Optional[int] = None,
        num_workers: int = 32,
    ):
        assert manifest_csv.exists(), f"Manifest missing: {manifest_csv}"
        assert behavior_root.exists(), f"Behavior root missing: {behavior_root}"

        self.df = pd.read_csv(manifest_csv)
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples).reset_index(drop=True)

        self.behavior_root = behavior_root
        self.is_train = is_train

        self.samples = []
        for _, row in self.df.iterrows():
            sid = str(row["sample_id"])
            cls_name = str(row["behavior_canonical"])
            cls_idx = BEHAVIOR_CLASS_TO_IDX[cls_name]
            self.samples.append((sid, cls_idx))

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
        seq = self.cached_tensors[idx]
        target = self.cached_targets[idx]
        sid, _ = self.samples[idx]

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
    """Loads SideViewCows2026 Train/Val pairs under Zero-Copy Policy."""
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

        self.cow_to_label = {c: i for i, c in enumerate(train_cows)}
        self.label_to_cow = {i: c for i, c in enumerate(train_cows)}

        df_d = pd.read_csv(protocols_dir / "protocol_closed_set.csv")
        df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()

        assert split in ("train", "val"), f"Split must be train or val, got {split}"
        self.df = df_d_41[df_d_41["closed_set_split"].eq(split)].reset_index(drop=True)

        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples).reset_index(drop=True)

        self.sideview_root = sideview_root
        self.is_train = (split == "train")
        self.margin_fraction = margin_fraction

        self.records = []
        for _, row in self.df.iterrows():
            cid = str(row["individual_id"])
            lbl = self.cow_to_label[cid]
            img_rel = str(row["image_path"]).replace("\\", "/").split("sideviewcows2026/")[-1].lstrip("/")
            mask_rel = str(row["mask_path"]).replace("\\", "/").split("sideviewcows2026/")[-1].lstrip("/")
            self.records.append((img_rel, mask_rel, lbl, cid))

        N = len(self.records)
        print(f"[*] Preloading {N} Re-ID crops into RAM ({split.capitalize()})...", flush=True)
        t0 = time.time()
        self.cached_imgs = np.empty((N, 4, 224, 224), dtype=np.uint8)

        def _load_reid(idx: int):
            img_rel, mask_rel, _, _ = self.records[idx]
            img_p = self.sideview_root / img_rel
            mask_p = self.sideview_root / mask_rel

            with Image.open(img_p) as im:
                rgb_im = im.convert("RGB")
            with Image.open(mask_p) as mk:
                mask_gray = mk.convert("L")

            mask_np = np.asarray(mask_gray) > 0
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            if not np.any(rows) or not np.any(cols):
                crop_box = (0, 0, rgb_im.width, rgb_im.height)
            else:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                pad_x = int(round(w * self.margin_fraction))
                pad_y = int(round(h * self.margin_fraction))
                x0 = max(0, xmin - pad_x)
                y0 = max(0, ymin - pad_y)
                x1 = min(rgb_im.width, xmax + 1 + pad_x)
                y1 = min(rgb_im.height, ymax + 1 + pad_y)
                crop_box = (x0, y0, x1, y1)

            img_crop = rgb_im.crop(crop_box).resize((224, 224), resample=Image.BILINEAR)
            mask_crop = mask_gray.crop(crop_box).resize((224, 224), resample=Image.NEAREST)

            rgb_arr = np.asarray(img_crop, dtype=np.uint8)
            mask_arr = (np.asarray(mask_crop, dtype=np.uint8) > 127).astype(np.uint8)

            self.cached_imgs[idx, :3, :, :] = rgb_arr.transpose(2, 0, 1)
            self.cached_imgs[idx, 3, :, :] = mask_arr

        workers = max(1, min(num_workers, 32))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_load_reid, range(N)))

        elapsed = time.time() - t0
        mb = self.cached_imgs.nbytes / (1024 * 1024)
        print(f"    ✓ Preloaded {N} Re-ID pairs ({mb:.1f} MB) in {elapsed:.1f}s ({N/max(0.1, elapsed):.1f} pairs/s).", flush=True)

        self.cached_tensors = torch.from_numpy(self.cached_imgs)
        self.cached_targets = torch.tensor([r[2] for r in self.records], dtype=torch.long)
        self.cow_ids = [r[3] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.cached_tensors[idx]
        target = self.cached_targets[idx]
        cid = self.cow_ids[idx]

        if self.is_train and torch.rand(1).item() < 0.5:
            pair = torch.flip(pair, dims=[-1])

        rgb = pair[:3, :, :]
        mask = pair[3:4, :, :].float()

        if self.is_train and torch.rand(1).item() < 0.5:
            b_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
            c_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
            rgb = TF.adjust_brightness(rgb, b_factor)
            rgb = TF.adjust_contrast(rgb, c_factor)

        rgb_norm = TF.normalize(rgb.float().div(255.0), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        pair_float = torch.cat([rgb_norm, mask], dim=0)  # [4, 224, 224] float32
        return pair_float, target, cid


# ==============================================================================
# 7. VALIDATION EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate_mtl_validation(
    model: MTLE3ModularModel,
    val_loader_bcs: DataLoader,
    val_loader_beh: DataLoader,
    val_loader_reid: DataLoader,
    criterion_bcs: nn.BCEWithLogitsLoss,
    criterion_beh: nn.CrossEntropyLoss,
    criterion_reid: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates validation loss and metrics across all 3 tasks."""
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

        preds = bcs_discrete_predictions_from_ordinal_logits(logits)
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

    # Composite E3 Validation Objective (identical formulation to E1)
    val_e3_objective = float((val_loss_bcs + val_loss_beh + val_loss_reid) / 3.0)

    return {
        "val_e3_objective": round(val_e3_objective, 5),
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
# 8. MAIN TRAINING ENGINE
# ==============================================================================
def train_mtl_e3_pipeline(
    bcs_data_dir: Path,
    behavior_data_dir: Path,
    reid_data_dir: Path,
    reid_protocols_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size_bcs: int = 64,
    batch_size_beh: int = 8,
    batch_size_reid: int = 32,
    bottleneck_dim: int = 128,
    adapter_dropout: float = 0.1,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    smoke: bool = False,
    device_name: Optional[str] = None,
    num_preload_workers: int = 32,
    active_git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main training loop for Run 8: E3 Modular Multi-Task Learning.
    """
    set_seed(2026)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)
    active_sha = active_git_sha if active_git_sha else get_git_commit_sha()

    print("\n" + "=" * 78)
    print("  PHASE 3 RUN 8: E3 MODULAR MULTI-TASK LEARNING")
    print(f"  Device           : {device}")
    print(f"  Git Commit SHA   : {active_sha}")
    print(f"  Mode             : {'SMOKE TEST' if smoke else 'FULL 30-EPOCH TRAINING'}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Sizes      : BCS={batch_size_bcs}, Behavior={batch_size_beh}, Re-ID={batch_size_reid}")
    print(f"  Bottleneck Dim   : {bottleneck_dim}")
    print(f"  Adapter Dropout  : {adapter_dropout}")
    print(f"  Task Weights     : w_BCS=1.0, w_Behavior=1.0, w_ReID=1.0 (Fixed Control)")
    print(f"  Output Dir       : {output_dir}")
    print("=" * 78)

    # 1. Instantiate Unified E3 Modular Model
    model = MTLE3ModularModel(
        pretrained=True,
        bottleneck_dim=bottleneck_dim,
        adapter_dropout=adapter_dropout,
    ).to(device)
    model.assert_modular_sharing()

    param_summary = model.count_parameters()
    print("\n[*] Model Architecture & Verified Parameter Distribution:")
    for k, v in param_summary.items():
        print(f"    {k:<40}: {v:,} params")

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

    # 3. Build DataLoaders
    train_loader_bcs = DataLoader(train_ds_bcs, batch_size=batch_size_bcs, shuffle=True, drop_last=True, num_workers=0)
    val_loader_bcs = DataLoader(val_ds_bcs, batch_size=batch_size_bcs, shuffle=False, drop_last=False, num_workers=0)

    train_loader_beh = DataLoader(train_ds_beh, batch_size=batch_size_beh, shuffle=True, drop_last=True, num_workers=0)
    val_loader_beh = DataLoader(val_ds_beh, batch_size=batch_size_beh, shuffle=False, drop_last=False, num_workers=0)

    train_loader_reid = DataLoader(train_ds_reid, batch_size=batch_size_reid, shuffle=True, drop_last=True, num_workers=0)
    val_loader_reid = DataLoader(val_ds_reid, batch_size=batch_size_reid, shuffle=False, drop_last=False, num_workers=0)

    # 4. Super-Step Schedule
    steps_bcs = len(train_loader_bcs)
    steps_beh = len(train_loader_beh)
    steps_reid = len(train_loader_reid)

    super_steps_per_epoch = 2 if smoke else steps_bcs
    print(f"\n[*] Multi-Task Step Alignment (Epoch = {super_steps_per_epoch} Super-Steps):")
    print(f"    BCS Batches/Epoch       : {steps_bcs}")
    print(f"    Behavior Batches/Epoch  : {steps_beh} (Oversample Ratio: {super_steps_per_epoch/max(1, steps_beh):.2f}x)")
    print(f"    Re-ID Batches/Epoch     : {steps_reid} (Oversample Ratio: {super_steps_per_epoch/max(1, steps_reid):.2f}x)")

    def infinite_iterator(loader: DataLoader) -> Iterator:
        while True:
            for batch in loader:
                yield batch

    bcs_cycler = infinite_iterator(train_loader_bcs)
    beh_cycler = infinite_iterator(train_loader_beh)
    reid_cycler = infinite_iterator(train_loader_reid)

    # 5. Optimizer, Scheduler, and Losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion_bcs = nn.BCEWithLogitsLoss()
    criterion_beh = nn.CrossEntropyLoss()
    criterion_reid = nn.CrossEntropyLoss()

    best_val_objective = float("inf")
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    epoch_history: List[Dict[str, Any]] = []

    best_ckpt_path = output_dir / ("mtl_e3_smoke_best.pth" if smoke else "mtl_e3_best.pth")
    latest_ckpt_path = output_dir / ("mtl_e3_smoke_latest.pth" if smoke else "mtl_e3_latest.pth")

    # 6. Training Loop
    t_train_start = time.time()
    print("\n[*] Launching E3 MTL update schedule...")

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
            if smoke and step == 1:
                assert model.bcs_adapter.down_proj.weight.grad is not None, "BCS adapter received no grad!"
                assert model.behavior_adapter.down_proj.weight.grad is None, "Behavior adapter leaked BCS grad!"
                assert model.reid_adapter.down_proj.weight.grad is None, "ReID adapter leaked BCS grad!"
                assert model.behavior_tcn.classifier.weight.grad is None, "Behavior head leaked BCS grad!"
                assert model.reid_head.classifier.weight.grad is None, "ReID head leaked BCS grad!"

            # Task 2: Behavior forward & backward
            seqs_beh, targets_beh, _ = next(beh_cycler)
            seqs_beh = seqs_beh.to(device, non_blocking=True)
            targets_beh = targets_beh.to(device, non_blocking=True)

            logits_beh, _ = model.forward_behavior(seqs_beh)
            loss_beh = criterion_beh(logits_beh, targets_beh)
            (1.0 * loss_beh).backward()
            if smoke and step == 1:
                assert model.behavior_adapter.down_proj.weight.grad is not None, "Behavior adapter received no grad!"
                assert model.reid_adapter.down_proj.weight.grad is None, "ReID adapter leaked Behavior grad!"
                assert model.reid_head.classifier.weight.grad is None, "ReID head leaked Behavior grad!"

            # Task 3: Re-ID forward & backward
            imgs_reid, targets_reid, _ = next(reid_cycler)
            imgs_reid = imgs_reid.to(device, non_blocking=True)
            targets_reid = targets_reid.to(device, non_blocking=True)

            logits_reid, _ = model.forward_reid(imgs_reid)
            loss_reid = criterion_reid(logits_reid, targets_reid)
            (1.0 * loss_reid).backward()

            # Verify that gradients have reached the shared backbone
            if smoke and step == 1:
                assert model.reid_adapter.down_proj.weight.grad is not None, "ReID adapter received no grad!"
                assert model.backbone.conv1.weight.grad is not None, "Gradients failed to reach shared backbone!"
                assert torch.isfinite(model.backbone.conv1.weight.grad).all(), "Non-finite gradient in shared backbone!"
                print("    [*] Smoke check step 1: forward/backward executed, backbone received grads, task-private pathways isolated ✅", flush=True)

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

        epoch_val_objective = val_eval["val_e3_objective"]
        is_best = epoch_val_objective < best_val_objective
        if is_best:
            best_val_objective = epoch_val_objective
            best_epoch = epoch
            best_metrics = val_eval

        epoch_time = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "epoch_duration_seconds": round(epoch_time, 2),
            "lr": float(scheduler.get_last_lr()[0]),
            "train": {
                "total_loss": round(train_loss_total, 5),
                "bcs_loss": round(train_loss_bcs, 5),
                "behavior_loss": round(train_loss_beh, 5),
                "reid_loss": round(train_loss_reid, 5),
            },
            "val": val_eval,
            "is_best": is_best,
        }
        epoch_history.append(epoch_record)

        marker = " 🏆 (NEW BEST)" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss_total:.4f} | "
            f"Val Objective: {epoch_val_objective:.5f}{marker} | "
            f"BCS MAE: {val_eval['bcs']['real_mae']:.4f} | "
            f"Beh F1: {val_eval['behavior']['macro_f1']:.4f} | "
            f"ReID Acc: {val_eval['reid']['top1_accuracy']:.2f}%",
            flush=True,
        )

        # Checkpoint serialization
        ckpt_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_objective": best_val_objective,
            "val_e3_objective": epoch_val_objective,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "param_summary": param_summary,
            "active_git_sha": active_sha,
            "val_metrics": val_eval,
            "epoch_history": epoch_history,
        }

        torch.save(ckpt_payload, latest_ckpt_path)
        if is_best:
            torch.save(ckpt_payload, best_ckpt_path)

    total_training_time = time.time() - t_train_start

    print("\n" + "=" * 78)
    print("  PHASE 3 RUN 8 E3 MODULAR MTL TRAINING COMPLETED")
    print(f"  Total Duration   : {total_training_time:.1f}s ({total_training_time/60:.2f} mins)")
    print(f"  Best Epoch       : {best_epoch}")
    print(f"  Best Objective   : {best_val_objective:.5f}")
    print(f"  Checkpoints Saved: {best_ckpt_path} & {latest_ckpt_path}")
    print("=" * 78)

    # Smoke Verification: Bit-identical checkpoint reload check
    bit_identical_verified = False
    max_reload_diff = 0.0
    if smoke:
        print("[*] Verifying bit-identical checkpoint reload...", flush=True)
        fresh_model = MTLE3ModularModel(pretrained=False).to(device)
        loaded_ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        fresh_model.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
        fresh_model.eval()
        model.eval()

        dummy_bcs = torch.randn(2, 4, 224, 224, device=device)
        dummy_beh = torch.randn(2, 8, 4, 224, 224, device=device)
        dummy_reid = torch.randn(2, 4, 224, 224, device=device)
        with torch.no_grad():
            diff_bcs = float(torch.max(torch.abs(model.forward_bcs(dummy_bcs) - fresh_model.forward_bcs(dummy_bcs))).item())
            diff_beh = float(torch.max(torch.abs(model.forward_behavior(dummy_beh)[0] - fresh_model.forward_behavior(dummy_beh)[0])).item())
            diff_reid = float(torch.max(torch.abs(model.forward_reid(dummy_reid)[0] - fresh_model.forward_reid(dummy_reid)[0])).item())

        max_reload_diff = max(diff_bcs, diff_beh, diff_reid)
        print(f"    Max Logit Difference on Reload: {max_reload_diff:.8f} (BCS: {diff_bcs:.8f}, Beh: {diff_beh:.8f}, ReID: {diff_reid:.8f})", flush=True)
        assert max_reload_diff < 1e-6, f"Checkpoint reload determinism failed! max_diff={max_reload_diff}"
        print("[*] Checkpoint save/reload verified bit-identically across all 3 tasks (max_diff < 1e-6) ✅", flush=True)
        bit_identical_verified = True

    # Save summary JSON
    summary = {
        "task": "phase3_run8_mtl_e3_modular",
        "experiment_name": "E3 Modular Multi-Task Learning (Task-Private Residual Adapters)",
        "active_git_sha": active_sha,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_objective": best_val_objective,
        "best_metrics": best_metrics,
        "total_training_time_seconds": round(total_training_time, 2),
        "parameter_summary": param_summary,
        "bit_identical_reload_verified": bit_identical_verified,
        "max_reload_diff": max_reload_diff,
        "epoch_history": epoch_history,
    }
    with open(output_dir / "mtl_e3_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Run 8: E3 Modular Multi-Task Learning")
    parser.add_argument("--bcs-data-dir", type=Path, default=Path("/mtl-data/bcs"))
    parser.add_argument("--behavior-data-dir", type=Path, default=Path("/mtl-data/behavior"))
    parser.add_argument("--reid-data-dir", type=Path, default=Path("/sideview/sideviewcows2026"))
    parser.add_argument("--reid-protocols-dir", type=Path, default=Path("/root/datasets/id/sideviewcows2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mtl-checkpoints/mtl_e3_modular"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size-bcs", type=int, default=64)
    parser.add_argument("--batch-size-beh", type=int, default=8)
    parser.add_argument("--batch-size-reid", type=int, default=32)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    train_mtl_e3_pipeline(
        bcs_data_dir=args.bcs_data_dir,
        behavior_data_dir=args.behavior_data_dir,
        reid_data_dir=args.reid_data_dir,
        reid_protocols_dir=args.reid_protocols_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size_bcs=args.batch_size_bcs,
        batch_size_beh=args.batch_size_beh,
        batch_size_reid=args.batch_size_reid,
        bottleneck_dim=args.bottleneck_dim,
        adapter_dropout=args.adapter_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoke=args.smoke,
    )
