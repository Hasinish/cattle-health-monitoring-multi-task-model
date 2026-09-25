# -*- coding: utf-8 -*-
"""
scripts/evaluate_mtl_e4_held_out.py — Official Held-Out Evaluation of Phase 3 E4 PCGrad Best Checkpoint
======================================================================================================
Evaluates the official frozen Phase 3 E4 PCGrad multi-task learning best checkpoint:
  `/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_best.pth`
across the official held-out evaluation protocols for all three primary tasks:
  1. BCS: Exact matched 7,549 ScienceDB test images from Run 4 (Frank & Hall cumulative ordinal BCE head)
     - Burst-group-disjoint / passage-sequence-safe held-out evaluation (NOT cow-disjoint).
  2. Behavior: Exact matched 780 retained sequences (T=8) from Run 5 (1D TCN head)
     - Source-specific reporting for CVB (N=422) and Beef (N=358). Walking is CVB-only.
  3. Re-ID: Official SideViewCows2026 Protocol A retrieval across 69 held-out cows (512-D unit-L2 embeddings)
     - Gallery: Parlor (36,811 images)
     - Queries: Barn (25,260 images) + Snapshots (607 images)
     - Target masks: released SideView ground-truth / oracle masks (NOT automatic SAM masks).

Zero test leakage:
  - Exact frozen Epoch-3 checkpoint selected by validation objective (val_e4_objective = 0.40098).
  - No retraining, no tuning, no threshold adjustments.
  - Matches Run 4, Run 5, Run 6, Run 7 (E1), and Run 8 (E3) evaluation protocols exactly.
  - Direct 3-way matched comparisons against single-task controls, E1 hard sharing, and E3 modular adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
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
import torchvision.transforms.functional as TF
from tqdm import tqdm

from scripts.train_mtl_e4_pcgrad import (
    MTLE4PCGradModel,
    REAL_BCS_CLASSES,
    NUM_BCS_CLASSES,
    BCS_LABEL_TO_IDX,
    BCS_IDX_TO_LABEL,
    BEHAVIOR_CLASSES,
    NUM_BEHAVIOR_CLASSES,
    BEHAVIOR_CLASS_TO_IDX,
    BEHAVIOR_IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from scripts.train_sideview_reid_perception import SideViewGTMaskReIDDataset


# ------------------------------------------------------------------------------
# 1. BCS Helper Functions & Metrics
# ------------------------------------------------------------------------------
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


def evaluate_bcs_metrics(y_true_indices: np.ndarray, y_pred_indices: np.ndarray) -> Dict[str, Any]:
    y_true_real = np.array([BCS_IDX_TO_LABEL[idx] for idx in y_true_indices])
    y_pred_real = np.array([BCS_IDX_TO_LABEL[idx] for idx in y_pred_indices])

    mae_real = float(np.mean(np.abs(y_true_real - y_pred_real)))
    acc_0 = float(np.mean(y_true_indices == y_pred_indices))
    acc_1 = float(np.mean(np.abs(y_true_indices - y_pred_indices) <= 1))
    bal_acc = float(balanced_accuracy_score(y_true_indices, y_pred_indices))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    precision = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    recall = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(NUM_BCS_CLASSES))).tolist()

    return {
        "real_mae": round(mae_real, 4),
        "acc_0": round(acc_0, 4),
        "acc_1": round(acc_1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm,
    }


# ------------------------------------------------------------------------------
# 2. Behavior Helper Functions & Metrics
# ------------------------------------------------------------------------------
def compute_behavior_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_names: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    overall_acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(NUM_BEHAVIOR_CLASSES))).tolist()

    prec_per = precision_score(y_true, y_pred, average=None, labels=list(range(NUM_BEHAVIOR_CLASSES)), zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_BEHAVIOR_CLASSES)), zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_BEHAVIOR_CLASSES)), zero_division=0)

    per_class = {}
    for i, c in enumerate(BEHAVIOR_CLASSES):
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
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": per_class,
        "per_class_f1": {c: round(float(f1_per[i]), 4) for i, c in enumerate(BEHAVIOR_CLASSES)},
        "confusion_matrix": conf_mat,
        "cvb_metrics": cvb_metrics,
        "beef_metrics": beef_metrics,
        "walking_is_cvb_only": True,
    }


# ------------------------------------------------------------------------------
# 3. Re-ID Protocol A Chunked Retrieval Evaluation
# ------------------------------------------------------------------------------
def evaluate_retrieval_chunked(
    query_feats: torch.Tensor,
    query_ids: List[str],
    gallery_feats: torch.Tensor,
    gallery_ids: List[str],
    chunk_size: int = 1000,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Any]:
    """
    Computes CMC (Rank-1, Rank-5, Rank-10) and mAP via cosine similarity
    on L2-normalized embeddings in query chunks.
    Matches Run 6, E1, E3, and SideViewCows2026 Protocol A exactly.
    """
    num_queries = query_feats.shape[0]
    ranks = {k: 0 for k in topk}
    all_ap = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_feats = query_feats.to(device)
    g_feats = gallery_feats.to(device)
    g_ids = np.array(gallery_ids)
    q_ids = np.array(query_ids)

    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        batch_q = q_feats[start:end]
        sims = torch.mm(batch_q, g_feats.t()).cpu().numpy()

        for i, q_idx in enumerate(range(start, end)):
            target_id = q_ids[q_idx]
            sim_row = sims[i]
            sorted_indices = np.argsort(-sim_row)
            ranked_ids = g_ids[sorted_indices]

            matches = (ranked_ids == target_id)
            if not np.any(matches):
                all_ap.append(0.0)
                continue

            for k in topk:
                if np.any(matches[:k]):
                    ranks[k] += 1

            match_indices = np.where(matches)[0]
            precisions = (np.arange(1, len(match_indices) + 1)) / (match_indices + 1.0)
            all_ap.append(float(np.mean(precisions)))

    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    cmc = {f"rank_{k}": round(float(ranks[k] / num_queries * 100), 2) for k in topk}

    return {
        "num_queries": num_queries,
        "num_gallery": len(gallery_ids),
        "rank_1": cmc.get("rank_1", 0.0),
        "rank_5": cmc.get("rank_5", 0.0),
        "rank_10": cmc.get("rank_10", 0.0),
        "mAP": round(mAP * 100, 2),
    }


# ------------------------------------------------------------------------------
# 4. Main Held-Out Evaluation Function
# ------------------------------------------------------------------------------
def evaluate_run_e4_held_out(
    checkpoint_path: Path,
    bcs_tensor_path: Path,
    behavior_dir: Path,
    reid_root: Path,
    reid_protocol_csv: Path,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  OFFICIAL PHASE 3 E4 PCGRAD MULTI-TASK LEARNING HELD-OUT EVALUATION")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Load frozen model checkpoint
    print(f"\n[*] Loading frozen E4 Best Checkpoint from {checkpoint_path}...")
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MTLE4PCGradModel(pretrained=False).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 11926706, (
        f"Trainable parameter mismatch: expected 11,926,706, got {trainable_params}"
    )
    print(f"  [+] Parameter assertion verified: exactly {trainable_params:,} trainable parameters.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch_val = ckpt.get("epoch", None)
    val_obj = ckpt.get("best_val_objective", ckpt.get("val_e4_objective", None))

    assert epoch_val == 3, f"Expected Epoch == 3, got {epoch_val}"
    assert val_obj is not None and abs(val_obj - 0.40098) < 1e-4, (
        f"Expected val_e4_objective == 0.40098, got {val_obj}"
    )
    print(f"  [+] Checkpoint assertions verified: Epoch = {epoch_val}, val_e4_objective = {val_obj:.5f}")

    # ==========================================================================
    # TASK 1: BCS ON EXACT MATCHED HELD-OUT POPULATION (N=7,549)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("  [TASK 1/3] EVALUATING BCS ON BURST-GROUP-DISJOINT TEST SET (N=7,549)")
    print("-" * 80)
    assert bcs_tensor_path.exists(), f"Missing BCS test tensor: {bcs_tensor_path}"

    t0_bcs = time.time()
    bcs_payload = torch.load(bcs_tensor_path, map_location="cpu", weights_only=False)
    bcs_tensors = bcs_payload["tensors"]  # [7549, 4, 224, 224] uint8
    bcs_targets = bcs_payload["targets"]  # [7549] class index
    n_bcs = len(bcs_targets)
    assert n_bcs == 7549, f"Expected 7549 BCS test samples, got {n_bcs}"
    print(f"  [+] Pre-packed BCS test tensor loaded: {n_bcs} samples in {time.time() - t0_bcs:.2f}s")

    bcs_criterion = nn.BCEWithLogitsLoss()
    bcs_total_loss = 0.0
    all_bcs_preds = []
    all_bcs_trues = []
    bcs_batch_size = 128

    with torch.no_grad():
        for start_idx in range(0, n_bcs, bcs_batch_size):
            end_idx = min(start_idx + bcs_batch_size, n_bcs)
            batch_raw = bcs_tensors[start_idx:end_idx].to(device)
            batch_targets = bcs_targets[start_idx:end_idx].to(device)

            # Preprocessing: ImageNet normalize RGB, binary float mask
            rgb = batch_raw[:, :3].float().div(255.0)
            rgb_norm = TF.normalize(rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            mask = batch_raw[:, 3:4].float()
            imgs = torch.cat([rgb_norm, mask], dim=1)

            logits = model.forward_bcs(imgs)
            targets_ordinal = ordinal_targets_from_class_indices(batch_targets, device=device)
            loss = bcs_criterion(logits, targets_ordinal)
            bcs_total_loss += loss.item() * len(batch_targets)

            pred_indices, _ = logits_to_bcs_predictions(logits)
            all_bcs_preds.extend(pred_indices)
            all_bcs_trues.extend(batch_targets.cpu().numpy())

    bcs_metrics = evaluate_bcs_metrics(np.array(all_bcs_trues), np.array(all_bcs_preds))
    bcs_results = {
        "test_loss": round(bcs_total_loss / n_bcs, 4),
        "total_test_samples": n_bcs,
        **bcs_metrics,
    }
    print(f"  [+] BCS Matched Test Complete:")
    print(f"      Real MAE:    {bcs_results['real_mae']:.4f}")
    print(f"      Acc@0:       {bcs_results['acc_0']*100:.2f}%")
    print(f"      Acc@1:       {bcs_results['acc_1']*100:.2f}%")
    print(f"      Bal Acc:     {bcs_results['balanced_accuracy']*100:.2f}%")
    print(f"      Macro-F1:    {bcs_results['macro_f1']:.4f}")
    print(f"      Test Loss:   {bcs_results['test_loss']:.4f}")

    # ==========================================================================
    # TASK 2: BEHAVIOR ON EXACT RETAINED TEST POPULATION (N=780, T=8)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("  [TASK 2/3] EVALUATING BEHAVIOR ON RETAINED SEQUENCES (N=780, T=8)")
    print("-" * 80)
    retained_csv = behavior_dir / "retained_test.csv"
    assert retained_csv.exists(), f"Missing retained_test.csv at {retained_csv}"
    df_beh = pd.read_csv(retained_csv)
    n_beh = len(df_beh)
    assert n_beh == 780, f"Expected 780 Behavior test sequences, got {n_beh}"
    print(f"  [+] Retained test sequences manifest loaded: {n_beh} sequences")

    t0_beh = time.time()
    cached_beh_pt = behavior_dir.parent / "test_behavior_224.pt"
    if cached_beh_pt.exists():
        print(f"  [*] Loading pre-packed Behavior test tensor from {cached_beh_pt}...")
        cached_payload = torch.load(cached_beh_pt, map_location="cpu", weights_only=False)
        beh_tensors_pt = cached_payload["tensors"]
        beh_labels_pt = cached_payload["labels"]
        beh_dataset_arr = np.array(cached_payload["dataset_names"])
        print(f"  [+] Loaded cached Behavior test tensor ({len(beh_labels_pt)} samples) in {time.time() - t0_beh:.2f}s")
    else:
        print("  [*] Preloading 780 test sequences (8 frames + 8 masks) into RAM with 32 workers...")
        beh_tensors = np.empty((n_beh, 8, 4, 224, 224), dtype=np.uint8)
        beh_labels = [0] * n_beh
        beh_dataset_names = [""] * n_beh

        def _load_single_seq(item: Tuple[int, Any]) -> Tuple[int, np.ndarray, int, str]:
            i, row = item
            sid = str(row["sample_id"])
            seq_folder = behavior_dir / sid
            raw_label = row["behavior_canonical"] if "behavior_canonical" in row else row["label"]
            lbl_idx = BEHAVIOR_CLASS_TO_IDX[raw_label]
            d_name = row["dataset"] if "dataset" in row else row.get("dataset_name", "cvb")

            seq_arr = np.empty((8, 4, 224, 224), dtype=np.uint8)
            for t in range(8):
                f_p = seq_folder / f"frame_{t:02d}.jpg"
                m_p = seq_folder / f"mask_{t:02d}.png"
                bgr = cv2.imread(str(f_p), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask_gray = cv2.imread(str(m_p), cv2.IMREAD_GRAYSCALE)
                mask_bin = (mask_gray > 127).astype(np.uint8)

                seq_arr[t, :3] = rgb.transpose(2, 0, 1)
                seq_arr[t, 3] = mask_bin

            return i, seq_arr, lbl_idx, d_name

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=32) as executor:
            for idx, seq_arr, lbl_idx, d_name in executor.map(_load_single_seq, df_beh.iterrows()):
                beh_tensors[idx] = seq_arr
                beh_labels[idx] = lbl_idx
                beh_dataset_names[idx] = d_name

        beh_tensors_pt = torch.from_numpy(beh_tensors)
        beh_labels_pt = torch.tensor(beh_labels, dtype=torch.long)
        beh_dataset_arr = np.array(beh_dataset_names)
        print(f"  [+] Loaded all 780 sequences ({beh_tensors.nbytes / (1024**2):.1f} MB) in {time.time() - t0_beh:.2f}s")
        try:
            torch.save({
                "tensors": beh_tensors_pt,
                "labels": beh_labels_pt,
                "dataset_names": beh_dataset_names,
            }, cached_beh_pt)
            print(f"  [+] Saved cached Behavior tensor to {cached_beh_pt}")
        except Exception as e:
            print(f"  [!] Note: could not write cache: {e}")

    beh_criterion = nn.CrossEntropyLoss()
    beh_total_loss = 0.0
    all_beh_preds = []
    all_beh_trues = []
    beh_batch_size = 16

    with torch.no_grad():
        for start_idx in range(0, n_beh, beh_batch_size):
            end_idx = min(start_idx + beh_batch_size, n_beh)
            batch_raw = beh_tensors_pt[start_idx:end_idx].to(device)  # [B, 8, 4, 224, 224]
            batch_targets = beh_labels_pt[start_idx:end_idx].to(device)

            B, T, C, H, W = batch_raw.shape
            rgb = batch_raw[:, :, :3].float().div(255.0)
            rgb_flat = rgb.view(B * T, 3, H, W)
            rgb_norm = TF.normalize(rgb_flat, mean=IMAGENET_MEAN, std=IMAGENET_STD).view(B, T, 3, H, W)
            mask = batch_raw[:, :, 3:4].float()
            seqs = torch.cat([rgb_norm, mask], dim=2)  # [B, 8, 4, 224, 224]

            logits, _ = model.forward_behavior(seqs)
            loss = beh_criterion(logits, batch_targets)
            beh_total_loss += loss.item() * len(batch_targets)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_beh_preds.extend(preds)
            all_beh_trues.extend(batch_targets.cpu().numpy())

    beh_metrics = compute_behavior_metrics(
        y_true=np.array(all_beh_trues),
        y_pred=np.array(all_beh_preds),
        dataset_names=beh_dataset_arr,
    )
    beh_results = {
        "test_loss": round(beh_total_loss / n_beh, 4),
        "total_test_samples": n_beh,
        **beh_metrics,
    }
    print(f"  [+] Behavior Matched Test Complete:")
    print(f"      Overall Acc:  {beh_results['overall_accuracy']*100:.2f}%")
    print(f"      Bal Acc:      {beh_results['balanced_accuracy']*100:.2f}%")
    print(f"      Macro-F1:     {beh_results['macro_f1']:.4f}")
    print(f"      Walking F1:   {beh_results['per_class_f1']['Walking']:.4f}")
    print(f"      CVB Acc:      {beh_results['cvb_metrics']['overall_accuracy']*100:.2f}% (F1: {beh_results['cvb_metrics']['macro_f1']:.4f})")
    print(f"      Beef Acc:     {beh_results['beef_metrics']['overall_accuracy']*100:.2f}% (F1: {beh_results['beef_metrics']['macro_f1']:.4f})")
    print(f"      Test Loss:    {beh_results['test_loss']:.4f}")

    # ==========================================================================
    # TASK 3: RE-ID PROTOCOL A RETRIEVAL ON 69 HELD-OUT COWS
    # ==========================================================================
    print("\n" + "-" * 80)
    print("  [TASK 3/3] EVALUATING RE-ID PROTOCOL A RETRIEVAL (69 HELD-OUT COWS)")
    print("-" * 80)
    assert reid_protocol_csv.exists(), f"Missing protocol CSV: {reid_protocol_csv}"
    df_proto = pd.read_csv(reid_protocol_csv)

    gallery_df = df_proto[df_proto["setting_role"] == "gallery"].reset_index(drop=True)
    barn_df = df_proto[df_proto["setting_role"] == "query_barn"].reset_index(drop=True)
    snapshots_df = df_proto[df_proto["setting_role"] == "query_snapshots"].reset_index(drop=True)

    print(f"  [+] Protocol A splits: Gallery={len(gallery_df)}, Query Barn={len(barn_df)}, Query Snapshots={len(snapshots_df)}")
    assert len(gallery_df) == 36811, f"Expected 36811 gallery images, got {len(gallery_df)}"
    assert len(barn_df) == 25260, f"Expected 25260 barn query images, got {len(barn_df)}"
    assert len(snapshots_df) == 607, f"Expected 607 snapshots query images, got {len(snapshots_df)}"

    eval_cows = sorted(df_proto[df_proto["setting_role"].ne("train")]["individual_id"].astype(str).unique())
    assert len(eval_cows) == 69, f"Expected 69 held-out evaluation cows, got {len(eval_cows)}"
    print(f"  [+] Verified exactly {len(eval_cows)} held-out evaluation cows.")

    reid_batch_size = 128
    num_workers = 8
    dummy_map = {c: 0 for c in eval_cows}

    def extract_embs(dataset: Dataset, desc: str) -> Tuple[torch.Tensor, List[str]]:
        loader = DataLoader(dataset, batch_size=reid_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        all_embs = []
        all_ids = []
        pbar = tqdm(loader, desc=f"  [Re-ID] {desc}", file=sys.stdout, leave=True)
        with torch.no_grad():
            for imgs, _, cow_ids in pbar:
                imgs = imgs.to(device)
                _, norm_embs = model.forward_reid(imgs)  # unit-L2 [B, 512]
                all_embs.append(norm_embs.cpu())
                all_ids.extend(cow_ids)
        return torch.cat(all_embs, dim=0), all_ids

    print("  [*] Extracting Parlor Gallery embeddings...")
    g_ds = SideViewGTMaskReIDDataset(gallery_df, reid_root, dummy_map, augment=False)
    g_feats, g_ids = extract_embs(g_ds, "Gallery Parlor")

    print("  [*] Extracting Barn Query embeddings...")
    b_ds = SideViewGTMaskReIDDataset(barn_df, reid_root, dummy_map, augment=False)
    b_feats, b_ids = extract_embs(b_ds, "Query Barn")

    print("  [*] Extracting Snapshots Query embeddings...")
    s_ds = SideViewGTMaskReIDDataset(snapshots_df, reid_root, dummy_map, augment=False)
    s_feats, s_ids = extract_embs(s_ds, "Query Snapshots")

    print("  [*] Computing chunked Protocol A retrieval rankings...")
    barn_retrieval = evaluate_retrieval_chunked(b_feats, b_ids, g_feats, g_ids, chunk_size=1000)
    snap_retrieval = evaluate_retrieval_chunked(s_feats, s_ids, g_feats, g_ids, chunk_size=607)

    reid_results = {
        "query_barn": barn_retrieval,
        "query_snapshots": snap_retrieval,
    }
    print(f"  [+] Re-ID Protocol A Complete:")
    print(f"      Barn -> Parlor:      Rank-1={barn_retrieval['rank_1']:.2f}%, Rank-5={barn_retrieval['rank_5']:.2f}%, Rank-10={barn_retrieval['rank_10']:.2f}%, mAP={barn_retrieval['mAP']:.2f}%")
    print(f"      Snapshots -> Parlor: Rank-1={snap_retrieval['rank_1']:.2f}%, Rank-5={snap_retrieval['rank_5']:.2f}%, Rank-10={snap_retrieval['rank_10']:.2f}%, mAP={snap_retrieval['mAP']:.2f}%")

    # ==========================================================================
    # 5. FAIR MATCHED DELTAS & COMPARISONS (SINGLE-TASK, E1, AND E3)
    # ==========================================================================
    # Single-task references (Runs 4, 5, 6)
    run4_baseline = {
        "population": 7549,
        "real_mae": 0.1709,
        "acc_0": 0.4357,
        "acc_1": 0.8940,
        "balanced_accuracy": 0.3970,
        "macro_f1": 0.4039,
        "test_loss": 0.4403,
    }
    run5_baseline = {
        "population": 780,
        "overall_accuracy": 0.8744,
        "balanced_accuracy": 0.7443,
        "macro_f1": 0.7397,
        "test_loss": 0.4430,
        "per_class_f1": {
            "Standing": 0.7215,
            "Lying": 0.9169,
            "Feeding": 0.9465,
            "Drinking": 0.8682,
            "Walking": 0.2456,
        },
        "cvb_metrics": {"overall_accuracy": 0.8009, "balanced_accuracy": 0.6105, "macro_f1": 0.6188},
        "beef_metrics": {"overall_accuracy": 0.9609, "balanced_accuracy": 0.9426, "macro_f1": 0.9414},
    }
    run6_baseline = {
        "query_barn": {"rank_1": 63.90, "rank_5": 77.10, "rank_10": 82.58, "mAP": 40.68},
        "query_snapshots": {"rank_1": 62.93, "rank_5": 75.29, "rank_10": 81.05, "mAP": 40.42},
    }

    # Run 7 E1 hard-shared reference
    run7_e1_baseline = {
        "bcs": {
            "real_mae": 0.1788,
            "acc_0": 0.4110,
            "acc_1": 0.8844,
            "balanced_accuracy": 0.3570,
            "macro_f1": 0.3605,
            "test_loss": 0.4175,
        },
        "behavior": {
            "overall_accuracy": 0.8500,
            "balanced_accuracy": 0.6730,
            "macro_f1": 0.6866,
            "test_loss": 0.6226,
            "per_class_f1": {
                "Standing": 0.7263,
                "Lying": 0.8924,
                "Feeding": 0.9031,
                "Drinking": 0.8702,
                "Walking": 0.0408,
            },
            "cvb_metrics": {"support": 422, "overall_accuracy": 0.7678, "balanced_accuracy": 0.4890, "macro_f1": 0.5402},
            "beef_metrics": {"support": 358, "overall_accuracy": 0.9469, "balanced_accuracy": 0.9311, "macro_f1": 0.9236},
        },
        "reid": {
            "query_barn": {"rank_1": 57.38, "rank_5": 73.33, "rank_10": 79.79, "mAP": 30.37},
            "query_snapshots": {"rank_1": 57.17, "rank_5": 75.45, "rank_10": 82.70, "mAP": 33.69},
        },
    }

    # Run 8 E3 modular reference
    run8_e3_baseline = {
        "bcs": {
            "real_mae": 0.1916,
            "acc_0": 0.3847,
            "acc_1": 0.8632,
            "balanced_accuracy": 0.3313,
            "macro_f1": 0.3291,
            "test_loss": 0.4555,
        },
        "behavior": {
            "overall_accuracy": 0.8577,
            "balanced_accuracy": 0.6670,
            "macro_f1": 0.6755,
            "test_loss": 0.4193,
            "per_class_f1": {
                "Standing": 0.7417,
                "Lying": 0.9306,
                "Feeding": 0.9016,
                "Drinking": 0.8039,
                "Walking": 0.0000,
            },
            "cvb_metrics": {"support": 422, "overall_accuracy": 0.8033, "balanced_accuracy": 0.6061, "macro_f1": 0.6026},
            "beef_metrics": {"support": 358, "overall_accuracy": 0.9218, "balanced_accuracy": 0.8729, "macro_f1": 0.8849},
        },
        "reid": {
            "query_barn": {"rank_1": 49.08, "rank_5": 67.72, "rank_10": 75.03, "mAP": 28.04},
            "query_snapshots": {"rank_1": 53.71, "rank_5": 72.32, "rank_10": 80.23, "mAP": 28.78},
        },
    }

    # Compute deltas vs Single-Task Baselines
    bcs_deltas_vs_single = {
        "real_mae_delta": round(bcs_results["real_mae"] - run4_baseline["real_mae"], 4),
        "acc_0_delta_pct": round((bcs_results["acc_0"] - run4_baseline["acc_0"]) * 100, 2),
        "acc_1_delta_pct": round((bcs_results["acc_1"] - run4_baseline["acc_1"]) * 100, 2),
        "balanced_acc_delta_pct": round((bcs_results["balanced_accuracy"] - run4_baseline["balanced_accuracy"]) * 100, 2),
        "macro_f1_delta": round(bcs_results["macro_f1"] - run4_baseline["macro_f1"], 4),
        "test_loss_delta": round(bcs_results["test_loss"] - run4_baseline["test_loss"], 4),
    }

    beh_deltas_vs_single = {
        "overall_accuracy_delta_pct": round((beh_results["overall_accuracy"] - run5_baseline["overall_accuracy"]) * 100, 2),
        "balanced_accuracy_delta_pct": round((beh_results["balanced_accuracy"] - run5_baseline["balanced_accuracy"]) * 100, 2),
        "macro_f1_delta": round(beh_results["macro_f1"] - run5_baseline["macro_f1"], 4),
        "test_loss_delta": round(beh_results["test_loss"] - run5_baseline["test_loss"], 4),
        "walking_f1_delta": round(beh_results["per_class_f1"]["Walking"] - run5_baseline["per_class_f1"]["Walking"], 4),
        "cvb_macro_f1_delta": round(beh_results["cvb_metrics"]["macro_f1"] - run5_baseline["cvb_metrics"]["macro_f1"], 4),
        "beef_macro_f1_delta": round(beh_results["beef_metrics"]["macro_f1"] - run5_baseline["beef_metrics"]["macro_f1"], 4),
    }

    reid_deltas_vs_single = {
        "query_barn": {
            "rank_1_delta": round(barn_retrieval["rank_1"] - run6_baseline["query_barn"]["rank_1"], 2),
            "rank_5_delta": round(barn_retrieval["rank_5"] - run6_baseline["query_barn"]["rank_5"], 2),
            "rank_10_delta": round(barn_retrieval["rank_10"] - run6_baseline["query_barn"]["rank_10"], 2),
            "mAP_delta": round(barn_retrieval["mAP"] - run6_baseline["query_barn"]["mAP"], 2),
        },
        "query_snapshots": {
            "rank_1_delta": round(snap_retrieval["rank_1"] - run6_baseline["query_snapshots"]["rank_1"], 2),
            "rank_5_delta": round(snap_retrieval["rank_5"] - run6_baseline["query_snapshots"]["rank_5"], 2),
            "rank_10_delta": round(snap_retrieval["rank_10"] - run6_baseline["query_snapshots"]["rank_10"], 2),
            "mAP_delta": round(snap_retrieval["mAP"] - run6_baseline["query_snapshots"]["mAP"], 2),
        },
    }

    # Compute deltas vs Run 7 E1 Hard-Shared Baseline
    bcs_deltas_vs_e1 = {
        "real_mae_delta": round(bcs_results["real_mae"] - run7_e1_baseline["bcs"]["real_mae"], 4),
        "acc_0_delta_pct": round((bcs_results["acc_0"] - run7_e1_baseline["bcs"]["acc_0"]) * 100, 2),
        "acc_1_delta_pct": round((bcs_results["acc_1"] - run7_e1_baseline["bcs"]["acc_1"]) * 100, 2),
        "balanced_acc_delta_pct": round((bcs_results["balanced_accuracy"] - run7_e1_baseline["bcs"]["balanced_accuracy"]) * 100, 2),
        "macro_f1_delta": round(bcs_results["macro_f1"] - run7_e1_baseline["bcs"]["macro_f1"], 4),
        "test_loss_delta": round(bcs_results["test_loss"] - run7_e1_baseline["bcs"]["test_loss"], 4),
    }

    beh_deltas_vs_e1 = {
        "overall_accuracy_delta_pct": round((beh_results["overall_accuracy"] - run7_e1_baseline["behavior"]["overall_accuracy"]) * 100, 2),
        "balanced_accuracy_delta_pct": round((beh_results["balanced_accuracy"] - run7_e1_baseline["behavior"]["balanced_accuracy"]) * 100, 2),
        "macro_f1_delta": round(beh_results["macro_f1"] - run7_e1_baseline["behavior"]["macro_f1"], 4),
        "test_loss_delta": round(beh_results["test_loss"] - run7_e1_baseline["behavior"]["test_loss"], 4),
        "walking_f1_delta": round(beh_results["per_class_f1"]["Walking"] - run7_e1_baseline["behavior"]["per_class_f1"]["Walking"], 4),
        "cvb_macro_f1_delta": round(beh_results["cvb_metrics"]["macro_f1"] - run7_e1_baseline["behavior"]["cvb_metrics"]["macro_f1"], 4),
        "beef_macro_f1_delta": round(beh_results["beef_metrics"]["macro_f1"] - run7_e1_baseline["behavior"]["beef_metrics"]["macro_f1"], 4),
    }

    reid_deltas_vs_e1 = {
        "query_barn": {
            "rank_1_delta": round(barn_retrieval["rank_1"] - run7_e1_baseline["reid"]["query_barn"]["rank_1"], 2),
            "rank_5_delta": round(barn_retrieval["rank_5"] - run7_e1_baseline["reid"]["query_barn"]["rank_5"], 2),
            "rank_10_delta": round(barn_retrieval["rank_10"] - run7_e1_baseline["reid"]["query_barn"]["rank_10"], 2),
            "mAP_delta": round(barn_retrieval["mAP"] - run7_e1_baseline["reid"]["query_barn"]["mAP"], 2),
        },
        "query_snapshots": {
            "rank_1_delta": round(snap_retrieval["rank_1"] - run7_e1_baseline["reid"]["query_snapshots"]["rank_1"], 2),
            "rank_5_delta": round(snap_retrieval["rank_5"] - run7_e1_baseline["reid"]["query_snapshots"]["rank_5"], 2),
            "rank_10_delta": round(snap_retrieval["rank_10"] - run7_e1_baseline["reid"]["query_snapshots"]["rank_10"], 2),
            "mAP_delta": round(snap_retrieval["mAP"] - run7_e1_baseline["reid"]["query_snapshots"]["mAP"], 2),
        },
    }

    # Compute deltas vs Run 8 E3 Modular Baseline
    bcs_deltas_vs_e3 = {
        "real_mae_delta": round(bcs_results["real_mae"] - run8_e3_baseline["bcs"]["real_mae"], 4),
        "acc_0_delta_pct": round((bcs_results["acc_0"] - run8_e3_baseline["bcs"]["acc_0"]) * 100, 2),
        "acc_1_delta_pct": round((bcs_results["acc_1"] - run8_e3_baseline["bcs"]["acc_1"]) * 100, 2),
        "balanced_acc_delta_pct": round((bcs_results["balanced_accuracy"] - run8_e3_baseline["bcs"]["balanced_accuracy"]) * 100, 2),
        "macro_f1_delta": round(bcs_results["macro_f1"] - run8_e3_baseline["bcs"]["macro_f1"], 4),
        "test_loss_delta": round(bcs_results["test_loss"] - run8_e3_baseline["bcs"]["test_loss"], 4),
    }

    beh_deltas_vs_e3 = {
        "overall_accuracy_delta_pct": round((beh_results["overall_accuracy"] - run8_e3_baseline["behavior"]["overall_accuracy"]) * 100, 2),
        "balanced_accuracy_delta_pct": round((beh_results["balanced_accuracy"] - run8_e3_baseline["behavior"]["balanced_accuracy"]) * 100, 2),
        "macro_f1_delta": round(beh_results["macro_f1"] - run8_e3_baseline["behavior"]["macro_f1"], 4),
        "test_loss_delta": round(beh_results["test_loss"] - run8_e3_baseline["behavior"]["test_loss"], 4),
        "walking_f1_delta": round(beh_results["per_class_f1"]["Walking"] - run8_e3_baseline["behavior"]["per_class_f1"]["Walking"], 4),
        "cvb_macro_f1_delta": round(beh_results["cvb_metrics"]["macro_f1"] - run8_e3_baseline["behavior"]["cvb_metrics"]["macro_f1"], 4),
        "beef_macro_f1_delta": round(beh_results["beef_metrics"]["macro_f1"] - run8_e3_baseline["behavior"]["beef_metrics"]["macro_f1"], 4),
    }

    reid_deltas_vs_e3 = {
        "query_barn": {
            "rank_1_delta": round(barn_retrieval["rank_1"] - run8_e3_baseline["reid"]["query_barn"]["rank_1"], 2),
            "rank_5_delta": round(barn_retrieval["rank_5"] - run8_e3_baseline["reid"]["query_barn"]["rank_5"], 2),
            "rank_10_delta": round(barn_retrieval["rank_10"] - run8_e3_baseline["reid"]["query_barn"]["rank_10"], 2),
            "mAP_delta": round(barn_retrieval["mAP"] - run8_e3_baseline["reid"]["query_barn"]["mAP"], 2),
        },
        "query_snapshots": {
            "rank_1_delta": round(snap_retrieval["rank_1"] - run8_e3_baseline["reid"]["query_snapshots"]["rank_1"], 2),
            "rank_5_delta": round(snap_retrieval["rank_5"] - run8_e3_baseline["reid"]["query_snapshots"]["rank_5"], 2),
            "rank_10_delta": round(snap_retrieval["rank_10"] - run8_e3_baseline["reid"]["query_snapshots"]["rank_10"], 2),
            "mAP_delta": round(snap_retrieval["mAP"] - run8_e3_baseline["reid"]["query_snapshots"]["mAP"], 2),
        },
    }

    full_evaluation_payload = {
        "experiment": "Phase 3 E4 PCGrad MTL Official Held-Out Evaluation",
        "evaluated_checkpoint": str(checkpoint_path),
        "checkpoint_selection_criterion": "Predefined Minimum Validation E4 Objective (Epoch 3, val_e4_objective = 0.40098)",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "populations": {
            "bcs_test_images": n_bcs,
            "behavior_test_sequences": n_beh,
            "reid_protocol_a_gallery": len(gallery_df),
            "reid_protocol_a_query_barn": len(barn_df),
            "reid_protocol_a_query_snapshots": len(snapshots_df),
            "reid_protocol_a_cows": 69,
        },
        "bcs": {
            "e4_pcgrad_matched": bcs_results,
            "run4_baseline_matched": run4_baseline,
            "run7_e1_baseline_matched": run7_e1_baseline["bcs"],
            "run8_e3_baseline_matched": run8_e3_baseline["bcs"],
            "deltas_vs_single_task": bcs_deltas_vs_single,
            "deltas_vs_e1_hard_shared": bcs_deltas_vs_e1,
            "deltas_vs_e3_modular": bcs_deltas_vs_e3,
            "verdict_vs_single": "Degradation" if bcs_deltas_vs_single["real_mae_delta"] > 0 else "Improvement",
            "verdict_vs_e1": "Improvement" if bcs_deltas_vs_e1["real_mae_delta"] < 0 else "Degradation",
            "verdict_vs_e3": "Improvement" if bcs_deltas_vs_e3["real_mae_delta"] < 0 else "Degradation",
        },
        "behavior": {
            "e4_pcgrad_matched": beh_results,
            "run5_baseline_matched": run5_baseline,
            "run7_e1_baseline_matched": run7_e1_baseline["behavior"],
            "run8_e3_baseline_matched": run8_e3_baseline["behavior"],
            "deltas_vs_single_task": beh_deltas_vs_single,
            "deltas_vs_e1_hard_shared": beh_deltas_vs_e1,
            "deltas_vs_e3_modular": beh_deltas_vs_e3,
            "verdict_vs_single": "Improvement" if beh_deltas_vs_single["balanced_accuracy_delta_pct"] > 0 else "Degradation",
            "verdict_vs_e1": "Improvement" if beh_deltas_vs_e1["balanced_accuracy_delta_pct"] > 0 else "Degradation",
            "verdict_vs_e3": "Improvement" if beh_deltas_vs_e3["balanced_accuracy_delta_pct"] > 0 else "Degradation",
        },
        "reid": {
            "e4_pcgrad_protocol_a": reid_results,
            "run6_baseline_protocol_a": run6_baseline,
            "run7_e1_baseline_protocol_a": run7_e1_baseline["reid"],
            "run8_e3_baseline_protocol_a": run8_e3_baseline["reid"],
            "deltas_vs_single_task": reid_deltas_vs_single,
            "deltas_vs_e1_hard_shared": reid_deltas_vs_e1,
            "deltas_vs_e3_modular": reid_deltas_vs_e3,
            "verdict_vs_single": "Improvement" if (reid_deltas_vs_single["query_barn"]["rank_1_delta"] >= 0 and reid_deltas_vs_single["query_snapshots"]["rank_1_delta"] >= 0) else ("Mixed" if (reid_deltas_vs_single["query_barn"]["rank_1_delta"] >= 0 or reid_deltas_vs_single["query_snapshots"]["rank_1_delta"] >= 0) else "Degradation"),
            "verdict_vs_e1": "Improvement" if (reid_deltas_vs_e1["query_barn"]["rank_1_delta"] >= 0 and reid_deltas_vs_e1["query_snapshots"]["rank_1_delta"] >= 0) else ("Mixed" if (reid_deltas_vs_e1["query_barn"]["rank_1_delta"] >= 0 or reid_deltas_vs_e1["query_snapshots"]["rank_1_delta"] >= 0) else "Degradation"),
            "verdict_vs_e3": "Improvement" if (reid_deltas_vs_e3["query_barn"]["rank_1_delta"] >= 0 and reid_deltas_vs_e3["query_snapshots"]["rank_1_delta"] >= 0) else ("Mixed" if (reid_deltas_vs_e3["query_barn"]["rank_1_delta"] >= 0 or reid_deltas_vs_e3["query_snapshots"]["rank_1_delta"] >= 0) else "Degradation"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "mtl_e4_test_evaluation_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_evaluation_payload, f, indent=2)
    print(f"\n[+] Saved full evaluation payload to {out_json}")

    return full_evaluation_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Phase 3 E4 PCGrad Held-Out Evaluation")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_best.pth"))
    parser.add_argument("--bcs-tensor-path", type=Path, default=Path("/mtl-data/bcs/test_bcs_224.pt"))
    parser.add_argument("--behavior-dir", type=Path, default=Path("/mtl-data/behavior_test"))
    parser.add_argument("--reid-root", type=Path, default=Path("/sideview/sideviewcows2026"))
    parser.add_argument("--reid-protocol-csv", type=Path, default=Path("/root/datasets/id/sideviewcows2026/protocol_cross_setting.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mtl-checkpoints/mtl_e4_evaluation"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_run_e4_held_out(
        checkpoint_path=args.checkpoint_path,
        bcs_tensor_path=args.bcs_tensor_path,
        behavior_dir=args.behavior_dir,
        reid_root=args.reid_root,
        reid_protocol_csv=args.reid_protocol_csv,
        output_dir=args.output_dir,
        device=device,
    )
