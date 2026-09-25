# -*- coding: utf-8 -*-
"""
scripts/train_mtl_e4_pcgrad.py — Phase 3: E4 PCGrad Multi-Task Learning Control
=============================================================================
Implements the canonical E4 PCGrad (Projecting Conflicting Gradients; Yu et al., 2020)
multi-task learning optimization experiment for direct controlled comparison with
Phase 3 Run 7 (E1 Hard-Shared MTL Control):
  1. ScienceDB Body Condition Scoring (BCS) — 4-channel input, shared trunk -> Ordinal BCE head
  2. CVB + Beef Behavior Recognition (Behavior) — 8-frame sequence, shared trunk -> 1D TCN -> CrossEntropy
  3. SideViewCows2026 Cow Re-Identification (Re-ID) — 4-channel input, shared trunk -> Linear(512, 41) -> CrossEntropy

Scientific Role & Controlled Question:
  E4 must answer one controlled scientific question:
    "Does PCGrad improve the hard-shared E1 multi-task model when the architecture,
     data, task heads, training schedule, losses, optimizer, and validation-selection
     rule are otherwise kept matched?"

  Canonical comparison:
    E1: Hard-shared ResNet-18 + ordinary accumulated task gradients + equal task weights.
    E3: Architectural task-private adapter intervention (+3.32% params; 12,322,610 params).
    E4: SAME hard-shared ResNet-18 (11,926,706 params) + PCGrad on SHARED BACKBONE gradients + equal task weights.

  There are:
    - NO adapters
    - NO private ResNet trunks
    - NO gates
    - NO GradNorm
    - NO learned task weights
    - NO uncertainty weighting

Architecture (Identical to E1):
  - Exactly ONE shared 4-channel ImageNet-pretrained ResNet-18 spatial feature extractor (11,179,648 params).
  - Exactly THREE task heads (100% matched to Runs 4-7):
      * BCS: Cumulative Frank & Hall (2001) Ordinal BCE head (Linear(512, 4), 2,052 params)
      * Behavior: Lightweight 1D Temporal Convolutional Network (TCN, 2 Conv1d blocks + Linear(256, 5), 723,973 params)
      * Re-ID: Unit L2-normalized 512-D embedding + Linear(512, 41) classifier (21,033 params)
  - Total E4 Trainable Parameters: exactly 11,926,706 (asserted programmatically).

PCGrad Optimization Surgery (Yu et al., 2020):
  - Applied ONLY to shared backbone parameters (11,179,648 params).
  - Task-specific heads are NOT projected against each other; each head receives ONLY its own task gradient.
  - For each task gradient g_i on the shared backbone:
      For the other task gradients g_j in reproducibly shuffled order (deterministic RNG from seed 2026):
        if dot(g_i, g_j) < 0:
          g_i = g_i - [ dot(g_i, g_j) / (||g_j||^2 + eps) ] * g_j
      Where g_j is the ORIGINAL comparison gradient.
  - Merged shared backbone gradient is computed by SUMMATION:
      g_shared = g_bcs_proj + g_beh_proj + g_reid_proj
  - Single optimizer update per super-step.

Validation Objective (Identical to E1):
  - val_e4_objective = (BCS_val_loss + Behavior_val_loss + ReID_val_loss) / 3.0
  - Model selection strictly by minimum val_e4_objective; canonical test sets remain strictly untouched.
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

# Reuse constants, taxonomy, and datasets from E1 to ensure 100% matched pipeline
from scripts.train_mtl_e1_hard_shared import (
    REAL_BCS_CLASSES,
    NUM_BCS_CLASSES,
    BCS_ORDINAL_THRESHOLDS,
    BCS_LABEL_TO_IDX,
    BCS_IDX_TO_LABEL,
    BEHAVIOR_CLASSES,
    NUM_BEHAVIOR_CLASSES,
    BEHAVIOR_CLASS_TO_IDX,
    BEHAVIOR_IDX_TO_CLASS,
    NUM_REID_TRAIN_COWS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_git_commit_sha,
    set_seed,
    ResNet18SharedBackbone,
    BCSOrdinalHead,
    BehaviorTemporalTCN,
    ReIDHead,
    MTLE1HardSharedModel,
    ordinal_targets_from_class_indices,
    logits_to_bcs_predictions,
    evaluate_bcs_metrics,
    evaluate_behavior_metrics,
    evaluate_reid_metrics,
    apply_gpu_augmentations_bcs,
    BCSMTLDataset,
    BehaviorMTLDataset,
    ReIDMTLDataset,
    DeterministicTaskCycler,
)


# ==============================================================================
# 1. ARCHITECTURE: EXACT E1 HARD-SHARED MODEL (11,926,706 PARAMETERS)
# ==============================================================================
class MTLE4PCGradModel(MTLE1HardSharedModel):
    """
    Run E4 PCGrad Multi-Task Learning Network:
      Exact same architecture as Run 7 E1 Hard-Shared Model:
        - 1 shared 4-channel ResNet-18 backbone (11,179,648 params)
        - 1 cumulative ordinal BCS head (2,052 params)
        - 1 Behavior lightweight 1D TCN (723,973 params)
        - 1 Re-ID linear head + 512-D L2 metric embedding (21,033 params)
      Total trainable parameters: exactly 11,926,706.
      Strictly NO adapters, NO private ResNet trunks, NO gates.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(pretrained=pretrained)


# ==============================================================================
# 2. PCGRAD PROJECTOR & GRADIENT DIAGNOSTICS ENGINE
# ==============================================================================
class PCGradProjector:
    """
    Deterministic Projecting Conflicting Gradients (PCGrad; Yu et al., 2020) engine.
    Applied exclusively to shared backbone parameters.

    Mathematical Formulation:
      Given T task gradients {g_i}_{i=1}^T on the shared backbone:
      For each task i:
        For each other task j in reproducibly shuffled order:
          if dot(g_i, g_j_orig) < 0:
            g_i = g_i - [ dot(g_i, g_j_orig) / (||g_j_orig||^2 + eps) ] * g_j_orig
      Where g_j_orig is the ORIGINAL comparison gradient before projection.
      Merged gradient: g_shared = sum_{i=1}^T g_i.
    """
    def __init__(self, seed: int = 2026, eps: float = 1e-12):
        self.seed = seed
        self.eps = eps
        self.rng = random.Random(seed)

    def get_state(self) -> Any:
        """Preserves PCGrad RNG state for deterministic checkpoint resume."""
        return self.rng.getstate()

    def set_state(self, state: Any) -> None:
        """Restores PCGrad RNG state from checkpoint."""
        self.rng.setstate(state)

    def project_and_diagnose(
        self,
        task_grads: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Projects conflicting gradients and records pre- and post-projection diagnostics.

        Args:
          task_grads: List of 3 1D tensors [g_bcs, g_beh, g_reid] on shared backbone.

        Returns:
          projected_grads: List of 3 1D tensors after PCGrad surgery.
          diagnostics: Detailed diagnostic dict for this super-step.
        """
        num_tasks = len(task_grads)
        assert num_tasks == 3, f"Expected 3 tasks, got {num_tasks}"

        # 1. Pre-projection norms & dot products
        norms = [torch.norm(g, p=2).item() for g in task_grads]
        norm_bcs, norm_beh, norm_reid = norms[0], norms[1], norms[2]

        dot_bcs_beh = torch.dot(task_grads[0], task_grads[1]).item()
        dot_bcs_reid = torch.dot(task_grads[0], task_grads[2]).item()
        dot_beh_reid = torch.dot(task_grads[1], task_grads[2]).item()

        cos_bcs_beh = dot_bcs_beh / (norm_bcs * norm_beh + self.eps)
        cos_bcs_reid = dot_bcs_reid / (norm_bcs * norm_reid + self.eps)
        cos_beh_reid = dot_beh_reid / (norm_beh * norm_reid + self.eps)

        conflict_bcs_beh = 1.0 if dot_bcs_beh < 0.0 else 0.0
        conflict_bcs_reid = 1.0 if dot_bcs_reid < 0.0 else 0.0
        conflict_beh_reid = 1.0 if dot_beh_reid < 0.0 else 0.0

        # 2. PCGrad Projection Surgery
        # Important: use original gradients as reference vectors g_j_orig
        orig_grads = [g.clone() for g in task_grads]
        projected_grads = [g.clone() for g in task_grads]
        projections_triggered = 0

        for i in range(num_tasks):
            other_tasks = [j for j in range(num_tasks) if j != i]
            self.rng.shuffle(other_tasks)
            for j in other_tasks:
                dot_ij = torch.dot(projected_grads[i], orig_grads[j]).item()
                if dot_ij < 0.0:
                    norm_sq = torch.dot(orig_grads[j], orig_grads[j]).item()
                    scale = dot_ij / (norm_sq + self.eps)
                    projected_grads[i] = projected_grads[i] - scale * orig_grads[j]
                    projections_triggered += 1

        # 3. Post-projection diagnostics
        post_norms = [torch.norm(g, p=2).item() for g in projected_grads]
        post_dot_bcs_beh = torch.dot(projected_grads[0], projected_grads[1]).item()
        post_dot_bcs_reid = torch.dot(projected_grads[0], projected_grads[2]).item()
        post_dot_beh_reid = torch.dot(projected_grads[1], projected_grads[2]).item()

        post_cos_bcs_beh = post_dot_bcs_beh / (post_norms[0] * post_norms[1] + self.eps)
        post_cos_bcs_reid = post_dot_bcs_reid / (post_norms[0] * post_norms[2] + self.eps)
        post_cos_beh_reid = post_dot_beh_reid / (post_norms[1] * post_norms[2] + self.eps)

        diagnostics = {
            "norm_bcs": norm_bcs,
            "norm_beh": norm_beh,
            "norm_reid": norm_reid,
            "dot_bcs_beh": dot_bcs_beh,
            "dot_bcs_reid": dot_bcs_reid,
            "dot_beh_reid": dot_beh_reid,
            "cos_bcs_beh": cos_bcs_beh,
            "cos_bcs_reid": cos_bcs_reid,
            "cos_beh_reid": cos_beh_reid,
            "conflict_bcs_beh": conflict_bcs_beh,
            "conflict_bcs_reid": conflict_bcs_reid,
            "conflict_beh_reid": conflict_beh_reid,
            "projections_triggered": float(projections_triggered),
            "post_cos_bcs_beh": post_cos_bcs_beh,
            "post_cos_bcs_reid": post_cos_bcs_reid,
            "post_cos_beh_reid": post_cos_beh_reid,
        }

        return projected_grads, diagnostics


# ==============================================================================
# 3. VALIDATION EVALUATION (EXACTLY MATCHED TO E1)
# ==============================================================================
def evaluate_mtl_validation(
    model: MTLE4PCGradModel,
    val_loader_bcs: DataLoader,
    val_loader_beh: DataLoader,
    val_loader_reid: DataLoader,
    criterion_bcs: nn.Module,
    criterion_beh: nn.Module,
    criterion_reid: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates multi-task validation metrics across all 3 tasks.
    Objective formula matches E1:
      val_e4_objective = (val_loss_bcs + val_loss_beh + val_loss_reid) / 3.0
    """
    model.eval()

    # 1. BCS Validation
    bcs_losses = []
    bcs_preds = []
    bcs_trues = []

    with torch.no_grad():
        for imgs_bcs, target_indices, raw_labels in val_loader_bcs:
            imgs_bcs = imgs_bcs.to(device)
            if imgs_bcs.dtype == torch.uint8:
                imgs_bcs = apply_gpu_augmentations_bcs(imgs_bcs, is_train=False)
            target_indices = target_indices.to(device)
            targets_ordinal = ordinal_targets_from_class_indices(target_indices, num_classes=NUM_BCS_CLASSES, device=device)

            logits = model.forward_bcs(imgs_bcs)
            loss = criterion_bcs(logits, targets_ordinal)
            bcs_losses.append(loss.item())

            preds, _ = logits_to_bcs_predictions(logits)
            bcs_preds.extend(preds.tolist())
            bcs_trues.extend(target_indices.cpu().numpy().tolist())

    val_loss_bcs = float(np.mean(bcs_losses))
    bcs_metrics = evaluate_bcs_metrics(np.array(bcs_trues), np.array(bcs_preds))

    # 2. Behavior Validation
    beh_losses = []
    beh_preds = []
    beh_trues = []

    with torch.no_grad():
        for seqs_beh, targets, _ in val_loader_beh:
            seqs_beh = seqs_beh.to(device)
            targets = targets.to(device)

            logits, _ = model.forward_behavior(seqs_beh)
            loss = criterion_beh(logits, targets)
            beh_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            beh_preds.extend(preds.tolist())
            beh_trues.extend(targets.cpu().numpy().tolist())

    val_loss_beh = float(np.mean(beh_losses))
    beh_metrics = evaluate_behavior_metrics(np.array(beh_trues), np.array(beh_preds))

    # 3. Re-ID Validation
    reid_losses = []
    reid_preds = []
    reid_trues = []

    with torch.no_grad():
        for imgs_reid, targets, _ in val_loader_reid:
            imgs_reid = imgs_reid.to(device)
            targets = targets.to(device)

            logits, _ = model.forward_reid(imgs_reid)
            loss = criterion_reid(logits, targets)
            reid_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            reid_preds.extend(preds.tolist())
            reid_trues.extend(targets.cpu().numpy().tolist())

    val_loss_reid = float(np.mean(reid_losses))
    reid_metrics = evaluate_reid_metrics(np.array(reid_trues), np.array(reid_preds))

    # Composite E4 Validation Objective (matching E1)
    val_e4_objective = float((val_loss_bcs + val_loss_beh + val_loss_reid) / 3.0)

    return {
        "val_e4_objective": round(val_e4_objective, 5),
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
# 4. MAIN TRAINING PIPELINE FOR E4 PCGRAD
# ==============================================================================
def train_mtl_e4_pipeline(
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
    Main training loop for Phase 3 E4 PCGrad Multi-Task Learning Control.
    """
    set_seed(2026)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)
    active_sha = active_git_sha if active_git_sha else get_git_commit_sha()

    print("\n" + "=" * 78)
    print("  PHASE 3: E4 PCGRAD MULTI-TASK LEARNING OPTIMIZATION CONTROL")
    print(f"  Device           : {device}")
    print(f"  Git Commit SHA   : {active_sha}")
    print(f"  Mode             : {'SMOKE TEST' if smoke else 'FULL 30-EPOCH TRAINING'}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Sizes      : BCS={batch_size_bcs}, Behavior={batch_size_beh}, Re-ID={batch_size_reid}")
    print(f"  Task Weights     : w_BCS=1.0, w_Behavior=1.0, w_ReID=1.0 (Fixed Equal)")
    print(f"  PCGrad Target    : Shared ResNet-18 Backbone ONLY (Heads Unprojected)")
    print(f"  Output Dir       : {output_dir}")
    print("=" * 78)

    # 1. Instantiate Model
    model = MTLE4PCGradModel(pretrained=True).to(device)
    model.assert_hard_sharing()

    param_summary = model.count_parameters()
    print("\n[*] Model Architecture & Parameter Verification:")
    for k, v in param_summary.items():
        print(f"    {k:<32}: {v:,} params")
    assert param_summary["total_trainable_parameters"] == 11926706, (
        f"Parameter mismatch! Expected 11,926,706, got {param_summary['total_trainable_parameters']}"
    )

    # 2. Initialize PCGrad Projector
    pcgrad_projector = PCGradProjector(seed=2026, eps=1e-12)

    # 3. Build Datasets
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

    # 4. DataLoaders
    train_loader_bcs = DataLoader(train_ds_bcs, batch_size=batch_size_bcs, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_bcs = DataLoader(val_ds_bcs, batch_size=batch_size_bcs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader_beh = DataLoader(train_ds_beh, batch_size=batch_size_beh, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_beh = DataLoader(val_ds_beh, batch_size=batch_size_beh, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader_reid = DataLoader(train_ds_reid, batch_size=batch_size_reid, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader_reid = DataLoader(val_ds_reid, batch_size=batch_size_reid, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # 5. Super-Step Scheduling Definition
    if smoke:
        super_steps_per_epoch = 2
    else:
        super_steps_per_epoch = len(train_loader_bcs)  # 537 steps per epoch

    bcs_cycler = DeterministicTaskCycler(train_loader_bcs, name="BCS")
    beh_cycler = DeterministicTaskCycler(train_loader_beh, name="Behavior")
    reid_cycler = DeterministicTaskCycler(train_loader_reid, name="ReID")

    print("\n[*] Super-Step Scheduling & Oversampling Analysis:")
    print(f"    Super-steps per epoch    : {super_steps_per_epoch}")
    print(f"    BCS batches/epoch        : {super_steps_per_epoch} (approx {super_steps_per_epoch * batch_size_bcs:,} samples)")
    print(f"    Behavior batches/epoch   : {super_steps_per_epoch} (approx {super_steps_per_epoch * batch_size_beh:,} sequences)")
    print(f"    Re-ID batches/epoch      : {super_steps_per_epoch} (approx {super_steps_per_epoch * batch_size_reid:,} images)")

    # 6. Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion_bcs = nn.BCEWithLogitsLoss()
    criterion_beh = nn.CrossEntropyLoss()
    criterion_reid = nn.CrossEntropyLoss()

    best_val_objective = float("inf")
    best_metrics: Dict[str, Any] = {}
    epoch_history: List[Dict[str, Any]] = []
    pcgrad_diagnostics_history: List[Dict[str, Any]] = []

    best_ckpt_path = output_dir / ("mtl_e4_smoke_best.pth" if smoke else "mtl_e4_best.pth")
    latest_ckpt_path = output_dir / ("mtl_e4_smoke_latest.pth" if smoke else "mtl_e4_latest.pth")

    # 7. Training Loop
    t_train_start = time.time()
    print("\n[*] Launching E4 PCGrad update schedule...")

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()

        running_loss_bcs = 0.0
        running_loss_beh = 0.0
        running_loss_reid = 0.0
        running_loss_total = 0.0

        # Diagnostics accumulators for this epoch
        epoch_diag_acc: Dict[str, float] = {
            "norm_bcs": 0.0,
            "norm_beh": 0.0,
            "norm_reid": 0.0,
            "dot_bcs_beh": 0.0,
            "dot_bcs_reid": 0.0,
            "dot_beh_reid": 0.0,
            "cos_bcs_beh": 0.0,
            "cos_bcs_reid": 0.0,
            "cos_beh_reid": 0.0,
            "conflict_bcs_beh": 0.0,
            "conflict_bcs_reid": 0.0,
            "conflict_beh_reid": 0.0,
            "projections_triggered": 0.0,
            "post_cos_bcs_beh": 0.0,
            "post_cos_bcs_reid": 0.0,
            "post_cos_beh_reid": 0.0,
        }

        for step in range(1, super_steps_per_epoch + 1):
            # ------------------------------------------------------------------
            # Stage A: Task 1 (BCS) Forward & Backward
            # ------------------------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            imgs_bcs, target_indices_bcs, _ = next(bcs_cycler)
            imgs_bcs = imgs_bcs.to(device, non_blocking=True)
            if imgs_bcs.dtype == torch.uint8:
                imgs_bcs = apply_gpu_augmentations_bcs(imgs_bcs, is_train=True)
            target_indices_bcs = target_indices_bcs.to(device, non_blocking=True)
            targets_bcs_ordinal = ordinal_targets_from_class_indices(target_indices_bcs, num_classes=NUM_BCS_CLASSES, device=device)

            logits_bcs = model.forward_bcs(imgs_bcs)
            loss_bcs = criterion_bcs(logits_bcs, targets_bcs_ordinal)
            (1.0 * loss_bcs).backward()

            # Capture BCS gradients
            g_bcs_shared = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.backbone.parameters() if p.requires_grad]
            g_bcs_head = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.bcs_head.parameters() if p.requires_grad]

            # ------------------------------------------------------------------
            # Stage B: Task 2 (Behavior) Forward & Backward
            # ------------------------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            seqs_beh, targets_beh, _ = next(beh_cycler)
            seqs_beh = seqs_beh.to(device, non_blocking=True)
            targets_beh = targets_beh.to(device, non_blocking=True)

            logits_beh, _ = model.forward_behavior(seqs_beh)
            loss_beh = criterion_beh(logits_beh, targets_beh)
            (1.0 * loss_beh).backward()

            # Capture Behavior gradients
            g_beh_shared = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.backbone.parameters() if p.requires_grad]
            g_beh_head = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.behavior_tcn.parameters() if p.requires_grad]

            # ------------------------------------------------------------------
            # Stage C: Task 3 (Re-ID) Forward & Backward
            # ------------------------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            imgs_reid, targets_reid, _ = next(reid_cycler)
            imgs_reid = imgs_reid.to(device, non_blocking=True)
            targets_reid = targets_reid.to(device, non_blocking=True)

            logits_reid, _ = model.forward_reid(imgs_reid)
            loss_reid = criterion_reid(logits_reid, targets_reid)
            (1.0 * loss_reid).backward()

            # Capture Re-ID gradients
            g_reid_shared = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.backbone.parameters() if p.requires_grad]
            g_reid_head = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.reid_head.parameters() if p.requires_grad]

            # Clear model gradients before restoring projected surgery
            optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------------------------
            # Stage D: PCGrad Surgery on Shared Backbone Gradients ONLY
            # ------------------------------------------------------------------
            flat_bcs = torch.cat([g.reshape(-1) for g in g_bcs_shared])
            flat_beh = torch.cat([g.reshape(-1) for g in g_beh_shared])
            flat_reid = torch.cat([g.reshape(-1) for g in g_reid_shared])

            proj_grads, step_diag = pcgrad_projector.project_and_diagnose([flat_bcs, flat_beh, flat_reid])

            # Accumulate diagnostics
            for k, val in step_diag.items():
                epoch_diag_acc[k] += val

            # Merge projected shared backbone gradients by SUMMATION (matches E1 unit-weight accumulation scale)
            merged_shared_grad = proj_grads[0] + proj_grads[1] + proj_grads[2]

            # ------------------------------------------------------------------
            # Stage E: Restore Gradients into Model
            # ------------------------------------------------------------------
            # 1. Restore shared backbone with merged projected gradients
            offset = 0
            for p in model.backbone.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.grad = merged_shared_grad[offset : offset + numel].view_as(p).clone()
                    offset += numel

            # 2. Restore task heads with their own unmodified task gradients (zero cross-talk)
            head_p_idx = 0
            for p in model.bcs_head.parameters():
                if p.requires_grad:
                    p.grad = g_bcs_head[head_p_idx].clone()
                    head_p_idx += 1

            beh_p_idx = 0
            for p in model.behavior_tcn.parameters():
                if p.requires_grad:
                    p.grad = g_beh_head[beh_p_idx].clone()
                    beh_p_idx += 1

            reid_p_idx = 0
            for p in model.reid_head.parameters():
                if p.requires_grad:
                    p.grad = g_reid_head[reid_p_idx].clone()
                    reid_p_idx += 1

            # Verification of gradient distribution
            if smoke and step == 1:
                assert model.backbone.conv1.weight.grad is not None, "Gradients failed to reach shared backbone!"
                assert torch.isfinite(model.backbone.conv1.weight.grad).all(), "Non-finite gradient in shared backbone!"
                assert model.bcs_head.fc.weight.grad is not None, "BCS head grad missing!"
                assert model.behavior_tcn.classifier.weight.grad is not None, "Behavior TCN grad missing!"
                assert model.reid_head.classifier.weight.grad is not None, "Re-ID head grad missing!"

            # ------------------------------------------------------------------
            # Stage F: Single Optimizer Update per Super-Step
            # ------------------------------------------------------------------
            optimizer.step()

            running_loss_bcs += loss_bcs.item()
            running_loss_beh += loss_beh.item()
            running_loss_reid += loss_reid.item()
            running_loss_total += (loss_bcs.item() + loss_beh.item() + loss_reid.item()) / 3.0

        scheduler.step()

        # Aggregate Epoch Diagnostics
        steps_float = float(max(1, super_steps_per_epoch))
        epoch_pcgrad_diag = {
            "epoch": epoch,
            "mean_shared_grad_norm_bcs": round(epoch_diag_acc["norm_bcs"] / steps_float, 6),
            "mean_shared_grad_norm_beh": round(epoch_diag_acc["norm_beh"] / steps_float, 6),
            "mean_shared_grad_norm_reid": round(epoch_diag_acc["norm_reid"] / steps_float, 6),
            "mean_pre_cos_bcs_beh": round(epoch_diag_acc["cos_bcs_beh"] / steps_float, 6),
            "mean_pre_cos_bcs_reid": round(epoch_diag_acc["cos_bcs_reid"] / steps_float, 6),
            "mean_pre_cos_beh_reid": round(epoch_diag_acc["cos_beh_reid"] / steps_float, 6),
            "conflict_frac_bcs_beh": round(epoch_diag_acc["conflict_bcs_beh"] / steps_float, 4),
            "conflict_frac_bcs_reid": round(epoch_diag_acc["conflict_bcs_reid"] / steps_float, 4),
            "conflict_frac_beh_reid": round(epoch_diag_acc["conflict_beh_reid"] / steps_float, 4),
            "mean_projections_triggered_per_step": round(epoch_diag_acc["projections_triggered"] / steps_float, 3),
            "total_projections_triggered": int(epoch_diag_acc["projections_triggered"]),
            "mean_post_cos_bcs_beh": round(epoch_diag_acc["post_cos_bcs_beh"] / steps_float, 6),
            "mean_post_cos_bcs_reid": round(epoch_diag_acc["post_cos_bcs_reid"] / steps_float, 6),
            "mean_post_cos_beh_reid": round(epoch_diag_acc["post_cos_beh_reid"] / steps_float, 6),
        }
        pcgrad_diagnostics_history.append(epoch_pcgrad_diag)

        # Epoch training statistics
        train_loss_bcs = running_loss_bcs / steps_float
        train_loss_beh = running_loss_beh / steps_float
        train_loss_reid = running_loss_reid / steps_float
        train_loss_total = running_loss_total / steps_float

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

        epoch_val_objective = val_eval["val_e4_objective"]
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
            "pcgrad_diagnostics": epoch_pcgrad_diag,
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"TrainObj: {train_loss_total:.4f} | "
            f"ValObj: {epoch_val_objective:.4f} {'[BEST]' if is_best else ''} | "
            f"BCS: val_loss={val_eval['bcs']['val_loss']:.4f} MAE={val_eval['bcs']['real_mae']:.4f} | "
            f"Beh: val_loss={val_eval['behavior']['val_loss']:.4f} F1={val_eval['behavior']['macro_f1']:.4f} | "
            f"ReID: val_loss={val_eval['reid']['val_loss']:.4f} Acc={val_eval['reid']['top1_accuracy']:.4f} | "
            f"PCGrad: proj/step={epoch_pcgrad_diag['mean_projections_triggered_per_step']:.2f} "
            f"conflicts=[BCS-Beh:{epoch_pcgrad_diag['conflict_frac_bcs_beh']:.2f}, "
            f"BCS-ReID:{epoch_pcgrad_diag['conflict_frac_bcs_reid']:.2f}, "
            f"Beh-ReID:{epoch_pcgrad_diag['conflict_frac_beh_reid']:.2f}]",
            flush=True,
        )

        # Checkpoint Saving
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "pcgrad_rng_state": pcgrad_projector.get_state(),
            "best_val_objective": best_val_objective,
            "best_metrics": best_metrics,
            "epoch_history": epoch_history,
            "pcgrad_diagnostics_history": pcgrad_diagnostics_history,
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
    print(f"  PHASE 3 E4 PCGRAD {'SMOKE' if smoke else 'TRAINING'} COMPLETED in {total_training_duration:.1f}s")
    print(f"  Best Val E4 Objective : {best_val_objective:.5f}")
    print(f"  Checkpoints Saved     : {best_ckpt_path} and {latest_ckpt_path}")
    print("=" * 78)

    # 8. Checkpoint Reload Verification
    print("[*] Verifying checkpoint reload bit-identity...")
    model.eval()
    reload_model = MTLE4PCGradModel(pretrained=False).to(device)
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
    best_reload_model = MTLE4PCGradModel(pretrained=False).to(device)
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    best_reload_model.load_state_dict(best_ckpt["model_state_dict"])
    best_reload_model.eval()

    print(f"    ✓ Checkpoint Reload Max Logit Difference: {max_logit_diff:.8f}")
    assert max_logit_diff < 1e-6, f"Checkpoint reload failed bit-identity: diff={max_logit_diff}"

    # Overall diagnostics aggregation across all epochs
    total_steps_all = max(1, len(pcgrad_diagnostics_history) * super_steps_per_epoch)
    total_proj_all = sum(d["total_projections_triggered"] for d in pcgrad_diagnostics_history)
    overall_mean_projections_per_step = round(total_proj_all / float(total_steps_all), 3)

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
        "pcgrad_diagnostics": {
            "epoch_history": pcgrad_diagnostics_history,
            "overall_summary": {
                "total_super_steps": total_steps_all,
                "total_projections_triggered": total_proj_all,
                "overall_mean_projections_per_step": overall_mean_projections_per_step,
            },
        },
    }

    # Save summary json
    metrics_json_p = output_dir / ("mtl_e4_pcgrad_smoke_metrics.json" if smoke else "mtl_e4_metrics.json")
    with open(metrics_json_p, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"[*] Saved metrics summary to {metrics_json_p}")

    return summary_results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: E4 PCGrad MTL Optimization Control")
    parser.add_argument("--bcs-data-dir", type=Path, default=Path("/mtl-data/bcs"))
    parser.add_argument("--behavior-data-dir", type=Path, default=Path("/mtl-data/behavior"))
    parser.add_argument("--sideview-data-dir", type=Path, default=Path("/sideview/sideviewcows2026"))
    parser.add_argument("--reid-protocols-dir", type=Path, default=Path("datasets/id/sideviewcows2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mtl-checkpoints/mtl_e4_pcgrad"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size-bcs", type=int, default=64)
    parser.add_argument("--batch-size-beh", type=int, default=8)
    parser.add_argument("--batch-size-reid", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Execute 2-epoch cheap smoke test")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_mtl_e4_pipeline(
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
