# -*- coding: utf-8 -*-
"""
Phase 3 Step 4.3 SideViewCows2026 RGB Single-Task Re-ID Baseline Pipeline
=========================================================================
Trains an ImageNet-pretrained ResNet-18 on the canonical SideViewCows2026
re-identification dataset (datasets/id/sideviewcows2026/).

Scientific Requirements:
1. RGB-only baseline: NO segmentation masks, NO SAM priors, NO SuperAnimal pose,
   NO viewpoint estimation, NO temporal features, NO adapters, NO MTL.
2. Input resolution: 224x224 RGB.
3. Canonical Protocol A Alignment:
   - 41 parlor-only cows = representation-learning identities (setting_role == "train").
   - 69 multi-setting cows = completely unseen retrieval evaluation identities.
   - 69 evaluation cows MUST remain strictly untouched during training and tuning.
4. Leakage-Safe Train/Validation Split:
   - Restricts canonical Protocol D (protocol_closed_set.csv) to the 41 Protocol-A training cows.
   - closed_set_split == "train": 12,753 images across 41 cows.
   - closed_set_split == "val": 2,683 images across 41 cows.
   - Sequence-safe: partitions contiguous recording sessions, preventing adjacent-frame leakage.
5. Baseline Architecture:
   - ResNet-18 -> 512-D feature -> L2-normalized embedding.
   - During training: 512-D feature -> Linear(512, 41) classifier -> CrossEntropyLoss.
6. Final Held-Out Retrieval (Full run only, NOT smoke mode):
   - Protocol A gallery: 36,811 parlor images across 69 unseen cows.
   - Query Barn: 25,260 barn images across 69 cows.
   - Query Snapshots: 607 snapshot images across 63 cows.
   - Cosine similarity on L2-normalized embeddings, evaluated in chunks to prevent OOM.
   - Metrics: Rank-1, Rank-5, Rank-10, mAP.
7. Strict Test Isolation:
   - In smoke mode, Protocol A gallery and queries are NEVER loaded.
   - In full mode, Protocol A retrieval is evaluated strictly ONCE after training completes.

Author: Hasin Ishrak
Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
Date: 2026-09-23
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# HASHING & PATH UTILITIES
# ==============================================================================
def compute_file_sha256(filepath: Path) -> str:
    """Compute deterministic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    """Retrieve current Git commit SHA safely."""
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), timeout=5)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def resolve_sideview_image_path(raw_path: str, data_root: Path) -> Optional[Path]:
    """
    Resolves relative/Windows/Linux protocol image path to actual file on disk.
    Example raw_path: 'datasets/id/external/sideviewcows2026/parlor/images/124/124_0001.jpg'
    """
    # Normalize path separators
    clean_p = str(raw_path).replace("\\", "/").strip()
    
    # Strip common prefixes
    prefixes = [
        "datasets/id/external/sideviewcows2026/",
        "datasets/id/sideviewcows2026/",
        "sideviewcows2026/",
        "/data/sideviewcows2026/",
        "data/sideviewcows2026/",
    ]
    rel_p = clean_p
    for pfx in prefixes:
        if rel_p.startswith(pfx):
            rel_p = rel_p[len(pfx):]
            break
    rel_p = rel_p.lstrip("/")

    candidates = [
        data_root / rel_p,
        data_root / "sideviewcows2026" / rel_p,
        REPO_ROOT / clean_p,
        REPO_ROOT / "datasets" / "id" / "external" / "sideviewcows2026" / rel_p,
        Path("/data/sideviewcows2026") / rel_p,
    ]

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand

    return None


# ==============================================================================
# DATASET IMPLEMENTATION
# ==============================================================================
class SideViewReIDDataset(Dataset):
    """
    PyTorch Dataset for SideViewCows2026 RGB images with cow identity labels.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        cow_to_label: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

        if cow_to_label is not None:
            self.cow_to_label = cow_to_label
        else:
            unique_cows = sorted(self.df["individual_id"].astype(str).unique())
            self.cow_to_label = {cow: i for i, cow in enumerate(unique_cows)}

        self.records = []
        for _, row in self.df.iterrows():
            raw_p = str(row["image_path"])
            res_p = resolve_sideview_image_path(raw_p, self.data_root)
            self.records.append({
                "raw_image_path": raw_p,
                "resolved_path": res_p,
                "cow_id": str(row["individual_id"]),
                "label": self.cow_to_label.get(str(row["individual_id"]), -1),
                "subset": str(row.get("subset", "")),
                "recording_id": str(row.get("recording_id", "")),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        resolved = rec.get("resolved_path")
        if resolved is None:
            resolved = resolve_sideview_image_path(rec["raw_image_path"], self.data_root)
        if resolved is None:
            raise FileNotFoundError(f"Failed to resolve image path: {rec['raw_image_path']} under {self.data_root}")

        try:
            with Image.open(resolved) as im:
                img = im.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Corrupt or unreadable image at {resolved}: {e}")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, rec["label"], rec["cow_id"]


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class ResNet18ReIDBaseline(nn.Module):
    """
    Single-Task RGB Identity Baseline:
    ResNet-18 Backbone -> Global Avg Pool -> 512-D Feature -> L2 Normalized Embedding.
    During training: Linear(512, num_classes) classifier head with CrossEntropyLoss.
    """
    def __init__(self, num_classes: int = 41, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.num_classes = num_classes
        self.classifier = nn.Linear(512, num_classes)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts raw 512-D features and L2-normalized 512-D embeddings.
        Returns: (raw_features, normalized_embeddings)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        raw_features = torch.flatten(x, 1)  # (B, 512)
        normalized_embeddings = F.normalize(raw_features, p=2, dim=1)  # (B, 512)
        return raw_features, normalized_embeddings

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during training:
        Returns: (logits, normalized_embeddings)
        """
        raw_features, normalized_embeddings = self.extract_features(x)
        logits = self.classifier(raw_features)
        return logits, normalized_embeddings


# ==============================================================================
# RETRIEVAL EVALUATION IN CHUNKS (PROTOCOL A)
# ==============================================================================
@torch.no_grad()
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
    on L2-normalized embeddings in query chunks to avoid memory bottlenecks.
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
        # Normalized dot product = Cosine Similarity
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


# ==============================================================================
# FEATURE EXTRACTION HELPER
# ==============================================================================
@torch.no_grad()
def extract_dataset_embeddings(
    model: ResNet18ReIDBaseline,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Extracting Embeddings",
) -> Tuple[torch.Tensor, List[str]]:
    """Extracts all 512-D L2-normalized embeddings and cow IDs for a DataLoader."""
    model.eval()
    all_embeddings = []
    all_ids = []

    pbar = tqdm(loader, desc=desc, leave=False, file=sys.stdout, mininterval=1.0)
    for imgs, _, cow_ids in pbar:
        imgs = imgs.to(device)
        _, norm_embs = model.extract_features(imgs)
        all_embeddings.append(norm_embs.cpu())
        all_ids.extend(cow_ids)

    cat_embs = torch.cat(all_embeddings, dim=0)
    return cat_embs, all_ids


# ==============================================================================
# VALIDATION EVALUATION (41 IDENTITIES)
# ==============================================================================
@torch.no_grad()
def evaluate_validation(
    model: ResNet18ReIDBaseline,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates classification performance and session-disjoint retrieval on val set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_embs = []
    all_cow_ids = []

    for imgs, targets, cow_ids in val_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        logits, norm_embs = model(imgs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_embs.append(norm_embs.cpu())
        all_cow_ids.extend(cow_ids)

    n_samples = len(val_loader.dataset)
    mean_loss = total_loss / n_samples
    top1_acc = float(accuracy_score(all_targets, all_preds))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return {
        "val_loss": round(mean_loss, 4),
        "val_top1_acc": round(top1_acc * 100, 2),
        "val_bal_acc": round(bal_acc * 100, 2),
        "val_macro_f1": round(macro_f1, 4),
    }


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_sideview_reid(
    data_root: Path,
    protocols_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    seed: int = 2026,
    smoke: bool = False,
    smoke_samples: int = 100,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes Phase 3 Step 4.3 SideViewCows2026 RGB Re-ID Baseline training.
    """
    # 1. Deterministic Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PHASE 3 STEP 4.3: SIDEVIEWCOWS2026 RGB RE-ID BASELINE TRAINING")
    print(f"  Mode           : {'SMOKE TEST (Subset, Train+Val Only)' if smoke else 'FULL TRAINING'}")
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Data Root      : {data_root}")
    print(f"  Protocols Dir  : {protocols_dir}")
    print(f"  Output Dir     : {output_dir}")
    print(f"  Target Epochs  : {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70)
    sys.stdout.flush()

    # 2. Protocol Integrity & Disjointness Check
    proto_a_path = protocols_dir / "protocol_cross_setting.csv"
    proto_d_path = protocols_dir / "protocol_closed_set.csv"

    assert proto_a_path.exists(), f"Missing Protocol A CSV: {proto_a_path}"
    assert proto_d_path.exists(), f"Missing Protocol D CSV: {proto_d_path}"

    df_a = pd.read_csv(proto_a_path)
    train_cows_a = sorted(df_a[df_a["setting_role"] == "train"]["individual_id"].astype(str).unique())
    eval_cows_a = set(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique())

    assert len(train_cows_a) == 41, f"Expected exactly 41 training cows in Protocol A, got {len(train_cows_a)}"
    assert len(eval_cows_a) == 69, f"Expected exactly 69 evaluation cows in Protocol A, got {len(eval_cows_a)}"
    overlap = set(train_cows_a).intersection(eval_cows_a)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} cows overlap between train and eval in Protocol A!"

    print(f"[*] Protocol A Identity Check: PASS (41 Train Cows, 69 Eval Cows, Disjoint: 100%)")
    sys.stdout.flush()

    # 3. Restrict Protocol D to the 41 Training Cows for Session-Safe Splits
    df_d = pd.read_csv(proto_d_path)
    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows_a)].copy()

    df_train = df_d_41[df_d_41["closed_set_split"] == "train"].reset_index(drop=True)
    df_val = df_d_41[df_d_41["closed_set_split"] == "val"].reset_index(drop=True)

    print(f"[*] Protocol D Restricted Splits: Train = {len(df_train)} imgs across {df_train['individual_id'].nunique()} cows | Val = {len(df_val)} imgs across {df_val['individual_id'].nunique()} cows")
    sys.stdout.flush()

    # Assert expected canonical counts on full runs
    if not smoke:
        assert len(df_train) == 12753, f"Expected 12,753 train images, got {len(df_train)}"
        assert len(df_val) == 2683, f"Expected 2,683 val images, got {len(df_val)}"
        assert df_train["individual_id"].nunique() == 41, "Expected 41 cows in train split"
        assert df_val["individual_id"].nunique() == 41, "Expected 41 cows in val split"

    # Build 41-Identity Label Mapping
    cow_to_label = {cow: i for i, cow in enumerate(train_cows_a)}
    label_to_cow = {i: cow for cow, i in cow_to_label.items()}

    # 4. Smoke Mode Subsetting (STRICT TEST ISOLATION: Protocol A queries NEVER loaded)
    if smoke:
        print(f"\n[SMOKE MODE] Subsetting Train to {smoke_samples} samples and Val to {smoke_samples} samples...")
        df_train = df_train.sample(n=min(smoke_samples, len(df_train)), random_state=seed).reset_index(drop=True)
        df_val = df_val.sample(n=min(smoke_samples, len(df_val)), random_state=seed).reset_index(drop=True)
        print(f"[SMOKE MODE] Smoke Train: {len(df_train)} images | Smoke Val: {len(df_val)} images")
        print(f"[SMOKE MODE] Held-out Protocol A gallery & queries are STRICTLY EXCLUDED.")
        sys.stdout.flush()

    # 5. Transforms & DataLoaders
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = SideViewReIDDataset(df_train, data_root=data_root, cow_to_label=cow_to_label, transform=train_transform)
    val_ds = SideViewReIDDataset(df_val, data_root=data_root, cow_to_label=cow_to_label, transform=val_transform)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # 6. Model, Optimizer, Loss, Scheduler
    model = ResNet18ReIDBaseline(num_classes=41, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_metric = -1.0
    history = []

    # 7. Checkpoint Resume
    if resume_path is not None and resume_path.exists():
        print(f"\n[*] Loading Checkpoint for Resume: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_metric = ckpt.get("best_val_metric", -1.0)
        history = ckpt.get("history", [])
        print(f"[*] Resumed successfully from Epoch {start_epoch - 1} (Best Val Metric: {best_val_metric})")
        sys.stdout.flush()

    # 8. Training Loop
    print("\n" + "-" * 70)
    print("  STARTING TRAINING LOOP")
    print("-" * 70)
    sys.stdout.flush()

    total_training_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=True, file=sys.stdout, mininterval=1.0)
        for imgs, targets, _ in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, norm_embs = model(imgs)

            # Assert shape & L2 normalization on step 1 of epoch 1
            if epoch == 1 and train_total == 0:
                assert logits.shape == (imgs.shape[0], 41), f"Logits shape mismatch: {logits.shape}"
                assert norm_embs.shape == (imgs.shape[0], 512), f"Embedding shape mismatch: {norm_embs.shape}"
                emb_norms = torch.norm(norm_embs, p=2, dim=1)
                assert torch.allclose(emb_norms, torch.ones_like(emb_norms), atol=1e-4), "L2 normalization check failed!"

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += imgs.size(0)

            cur_loss = train_loss / train_total
            cur_acc = (train_correct / train_total) * 100
            pbar.set_postfix({"loss": f"{cur_loss:.4f}", "acc": f"{cur_acc:.1f}%"})

        scheduler.step()
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = (train_correct / train_total) * 100

        # Evaluate on validation set
        val_metrics = evaluate_validation(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_epoch_loss, 4),
            "train_top1_acc": round(train_epoch_acc, 2),
            "val_loss": val_metrics["val_loss"],
            "val_top1_acc": val_metrics["val_top1_acc"],
            "val_bal_acc": val_metrics["val_bal_acc"],
            "val_macro_f1": val_metrics["val_macro_f1"],
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "duration_sec": round(epoch_duration, 1),
        }
        history.append(log_entry)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_top1_acc']:.2f}% | "
            f"Val BalAcc: {val_metrics['val_bal_acc']:.2f}% | Val F1: {val_metrics['val_macro_f1']:.4f}"
        )
        sys.stdout.flush()

        # Checkpoint Management
        is_best = val_metrics["val_top1_acc"] > best_val_metric
        if is_best:
            best_val_metric = val_metrics["val_top1_acc"]

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_metric": best_val_metric,
            "cow_to_label": cow_to_label,
            "label_to_cow": label_to_cow,
            "git_sha": get_git_commit_sha(),
            "config": {
                "num_classes": 41,
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs,
                "seed": seed,
                "smoke": smoke,
            },
            "history": history,
            "val_metrics": val_metrics,
        }

        latest_ckpt_path = output_dir / "reid_baseline_latest.pth"
        torch.save(checkpoint_data, latest_ckpt_path)

        if is_best:
            best_ckpt_path = output_dir / "reid_baseline_best.pth"
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"  --> [*] NEW BEST CHECKPOINT SAVED (Val Acc: {best_val_metric:.2f}%)")
            sys.stdout.flush()

    total_training_duration = time.time() - total_training_start
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETED (Total Duration: {total_training_duration / 60:.1f} mins)")
    print(f"  Best Val Accuracy: {best_val_metric:.2f}%")
    print("=" * 70)
    sys.stdout.flush()

    # 9. Final Held-Out Evaluation on Protocol A (STRICTLY ONCE, FULL MODE ONLY)
    retrieval_results = {}
    if not smoke:
        print("\n" + "=" * 70)
        print("  PROTOCOL A HELD-OUT RETRIEVAL EVALUATION (UNSEEN 69 COWS)")
        print("  Gallery: Milking Parlor (36,811 images)")
        print("  Queries: Barn Video (25,260 images) & Snapshots (607 images)")
        print("=" * 70)
        sys.stdout.flush()

        # Load best model weights
        best_ckpt = torch.load(output_dir / "reid_baseline_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        # Build Protocol A sub-datasets
        df_gallery = df_a[df_a["setting_role"] == "gallery"].reset_index(drop=True)
        df_q_barn = df_a[df_a["setting_role"] == "query_barn"].reset_index(drop=True)
        df_q_snap = df_a[df_a["setting_role"] == "query_snapshots"].reset_index(drop=True)

        gallery_ds = SideViewReIDDataset(df_gallery, data_root=data_root, transform=val_transform)
        q_barn_ds = SideViewReIDDataset(df_q_barn, data_root=data_root, transform=val_transform)
        q_snap_ds = SideViewReIDDataset(df_q_snap, data_root=data_root, transform=val_transform)

        g_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        qb_loader = DataLoader(q_barn_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        qs_loader = DataLoader(q_snap_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print("[*] Extracting Gallery Embeddings (36,811 images)...")
        g_feats, g_ids = extract_dataset_embeddings(model, g_loader, device, desc="Gallery Parlor")

        print("[*] Extracting Query Barn Embeddings (25,260 images)...")
        qb_feats, qb_ids = extract_dataset_embeddings(model, qb_loader, device, desc="Query Barn")

        print("[*] Extracting Query Snapshots Embeddings (607 images)...")
        qs_feats, qs_ids = extract_dataset_embeddings(model, qs_loader, device, desc="Query Snapshots")

        print("\n[*] Evaluating Retrieval: Query Barn -> Gallery Parlor...")
        retrieval_barn = evaluate_retrieval_chunked(qb_feats, qb_ids, g_feats, g_ids, chunk_size=1000)
        print(f"  Query Barn: Rank-1 = {retrieval_barn['rank_1']}% | Rank-5 = {retrieval_barn['rank_5']}% | Rank-10 = {retrieval_barn['rank_10']}% | mAP = {retrieval_barn['mAP']}%")

        print("\n[*] Evaluating Retrieval: Query Snapshots -> Gallery Parlor...")
        retrieval_snap = evaluate_retrieval_chunked(qs_feats, qs_ids, g_feats, g_ids, chunk_size=607)
        print(f"  Query Snapshots: Rank-1 = {retrieval_snap['rank_1']}% | Rank-5 = {retrieval_snap['rank_5']}% | Rank-10 = {retrieval_snap['rank_10']}% | mAP = {retrieval_snap['mAP']}%")

        retrieval_results = {
            "query_barn": retrieval_barn,
            "query_snapshots": retrieval_snap,
        }
    else:
        print("\n[SMOKE MODE] Protocol A held-out retrieval evaluation skipped to maintain strict test isolation.")
        retrieval_results = {"status": "SKIPPED_IN_SMOKE_MODE"}

    # 10. Save Metrics JSON
    final_metrics = {
        "task": "reid_rgb_baseline",
        "dataset": "sideviewcows2026",
        "git_sha": get_git_commit_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_val_top1_acc": best_val_metric,
        "total_training_time_sec": round(total_training_duration, 2),
        "history": history,
        "protocol_a_retrieval": retrieval_results,
    }

    metrics_file = output_dir / "reid_baseline_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n[*] Metrics saved to: {metrics_file}")
    sys.stdout.flush()

    return final_metrics


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="SideViewCows2026 RGB Re-ID Baseline Trainer")
    parser.add_argument("--data-root", type=str, default="datasets/id/external/sideviewcows2026", help="Path to SideViewCows2026 root")
    parser.add_argument("--protocols-dir", type=str, default="datasets/id/sideviewcows2026", help="Path to protocol CSVs directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/reid_baseline", help="Directory to save checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Smoke test flags
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False, help="Run smoke test mode")
    parser.add_argument("--smoke-samples", type=int, default=100, help="Number of samples per split in smoke mode")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    protocols_dir = Path(args.protocols_dir)
    output_dir = Path(args.output_dir)
    resume_path = Path(args.resume) if args.resume else None

    train_sideview_reid(
        data_root=data_root,
        protocols_dir=protocols_dir,
        output_dir=output_dir,
        epochs=args.epochs if (not args.smoke or args.epochs != 30) else 2,
        batch_size=args.batch_size if not args.smoke else 16,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers if not args.smoke else 0,
        seed=args.seed,
        smoke=args.smoke,
        smoke_samples=args.smoke_samples,
        resume_path=resume_path,
    )


if __name__ == "__main__":
    main()
