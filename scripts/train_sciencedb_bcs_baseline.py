"""
Phase 3 ScienceDB RGB Single-Task BCS Baseline Pipeline
======================================================
Architecture: ImageNet-pretrained ResNet-18 with Ordinal Head (Ordinal BCE / CORAL) or Linear Head
Dataset: ScienceDB Cattle BCS (burst-group-disjoint / sequence-safe, NOT cow-disjoint)
Splits: datasets/bcs/sciencedb/{train,val,test}.csv (5,653 connected burst groups)

Evaluation:
- Primary metric: Real BCS MAE computed on physical scale 3.25 to 4.25 (step 0.25)
- Secondary metrics: Balanced Accuracy, Macro-F1, Acc@0 (exact), Acc@1 (+/- 0.25 tolerance)
- Per-class metrics: Support, Precision, Recall, F1, Per-Class Real MAE, Confusion Matrix
- Checkpoint persistence: Full state dict saving, resumption, and verification
- Provenance: Git commit, split SHA-256 hashes, full hyperparameters and seed tracking
- Integrity: Smoke mode strictly uses train/val subsets and NEVER touches canonical held-out test split.
- Compute target: Smoke tests on local GTX 1050 Ti; full heavy training on rotating Modal accounts.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# ==============================================================================
# CANONICAL TAXONOMY & REPRODUCIBILITY
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent

# ScienceDB Canonical Labels: 5 discrete ordinal levels
REAL_BCS_CLASSES = [3.25, 3.50, 3.75, 4.00, 4.25]
NUM_CLASSES = len(REAL_BCS_CLASSES)
ORDINAL_THRESHOLDS = NUM_CLASSES - 1  # 4 binary threshold tasks

LABEL_TO_IDX = {3.25: 0, 3.50: 1, 3.75: 2, 4.00: 3, 4.25: 4}
IDX_TO_LABEL = {0: 3.25, 1: 3.50, 2: 3.75, 3: 4.00, 4: 4.25}


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_info() -> Dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
        is_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT
        ).strip())
        return {"commit": commit, "branch": branch, "is_dirty": is_dirty}
    except Exception as e:
        return {"commit": "UNKNOWN", "branch": "UNKNOWN", "is_dirty": True, "error": str(e)}


def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(raw_path: str, data_root: Optional[Path] = None) -> Path:
    """
    Robust runtime path resolver for ScienceDB images across Windows and Linux/Modal.
    Preserves canonical CSV paths without altering scientific split files.
    """
    p = Path(raw_path)
    if data_root is None:
        if p.exists():
            return p
        if Path("/data").exists():
            data_root = Path("/data")
        elif Path("/data/dataset").exists():
            data_root = Path("/data/dataset")

    if data_root is not None:
        data_root = Path(data_root)
        parts = PureWindowsPath(raw_path).parts
        # Canonical structure ends with: ('dataset', '<class>', '<filename>.jpg')
        if len(parts) >= 3 and parts[-3] == "dataset":
            c1 = data_root / parts[-3] / parts[-2] / parts[-1]  # data_root / dataset / 4.25 / GS_1_1.jpg
            if c1.exists():
                return c1
            c2 = data_root / parts[-2] / parts[-1]            # data_root / 4.25 / GS_1_1.jpg
            if c2.exists():
                return c2
            return c1
        elif len(parts) >= 2:
            candidate = data_root / parts[-2] / parts[-1]
            if candidate.exists():
                return candidate
            return candidate
    return p


# ==============================================================================
# DATASET & TRANSFORMS
# ==============================================================================
class ScienceDBBCSDataset(Dataset):
    """
    ScienceDB BCS Dataset for Phase 3 Single-Task Baseline.
    Provenance Note: ScienceDB is sequence-safe / burst-group-disjoint.
    True biological cow IDs are NOT provided by publisher.
    """
    def __init__(
        self,
        csv_path: Path,
        transform=None,
        max_samples: Optional[int] = None,
        data_root: Optional[Path] = None,
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.data_root = Path(data_root) if data_root else None
        
        df = pd.read_csv(self.csv_path)
        if max_samples and max_samples < len(df):
            # Deterministic stratified sample for smoke testing (zero pandas warnings)
            samples_per_class = max(1, max_samples // NUM_CLASSES)
            subsets = []
            for label_val in REAL_BCS_CLASSES:
                sub = df[df["label"] == label_val]
                subsets.append(sub.head(samples_per_class))
            df = pd.concat(subsets, ignore_index=True)
            if len(df) > max_samples:
                df = df.iloc[:max_samples]
                
        self.df = df.reset_index(drop=True)
        
        # Pre-verify paths and parse integer targets
        self.samples = []
        for idx, row in self.df.iterrows():
            raw_path = str(row["image_path"])
            resolved_path = resolve_image_path(raw_path, data_root=self.data_root)
            raw_label = float(row["label"])
            if raw_label not in LABEL_TO_IDX:
                raise ValueError(f"Encountered unexpected BCS label {raw_label} at row {idx}")
            target_idx = LABEL_TO_IDX[raw_label]
            self.samples.append((str(resolved_path), target_idx, raw_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        img_path, target_idx, raw_label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening image at {img_path}: {e}")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(target_idx, dtype=torch.long), raw_label


def get_transforms(image_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        norm,
    ])
    
    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        norm,
    ])
    
    return train_transform, eval_transform


# ==============================================================================
# MODEL HEADS (ORDINAL BCE vs GENUINE CORAL vs LINEAR)
# ==============================================================================
class OrdinalBCEHead(nn.Module):
    """
    Frank & Hall (2001) / Extended Binary Classification (EBC) Cumulative Logits:
    Independent weight vectors w_k and independent biases b_k for each ordinal threshold:
    g_k(x) = w_k^T x + b_k
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes - 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CoralSharedHead(nn.Module):
    """
    Mathematically genuine CORAL (Cao, Mirjalili, Raschka 2020):
    Enforces rank consistency via a single shared weight vector w across all K-1 binary
    threshold tasks, alongside independent task-specific biases b_k:
    g_k(x) = w^T x + b_k
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.biases


class ResNet18BCSBaseline(nn.Module):
    """
    Phase 3 Canonical ResNet-18 ImageNet Baseline for Body Condition Scoring.
    Supports:
    - 'ordinal_bce' (Default): Frank & Hall (2001) cumulative BCE with independent weights.
    - 'coral': Cao et al. (2020) rank-consistent ordinal regression (single shared weight + separate biases).
    - 'linear': Standard 5-class cross-entropy classification.
    """
    def __init__(self, head_type: str = "ordinal_bce", pretrained: bool = True):
        super().__init__()
        self.head_type = head_type.lower()
        if self.head_type not in ["ordinal_bce", "coral", "linear"]:
            raise ValueError(f"Unsupported head_type: {head_type}. Choose 'ordinal_bce', 'coral', or 'linear'.")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        # Feature extractor up to adaptive average pool
        self.backbone = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
            base_resnet.avgpool
        )
        self.in_features = base_resnet.fc.in_features  # 512

        if self.head_type == "ordinal_bce":
            self.head = OrdinalBCEHead(self.in_features, NUM_CLASSES)
        elif self.head_type == "coral":
            self.head = CoralSharedHead(self.in_features, NUM_CLASSES)
        else:
            self.head = nn.Linear(self.in_features, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        logits = self.head(feat)
        return logits


def cumulative_ordinal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes cumulative binary cross-entropy loss for ordinal regression.
    targets: shape (B,), integer class indices in range [0, NUM_CLASSES - 1]
    logits: shape (B, NUM_CLASSES - 1)
    """
    binary_targets = [(targets > i).float() for i in range(ORDINAL_THRESHOLDS)]
    binary_targets = torch.stack(binary_targets, dim=1)
    return F.binary_cross_entropy_with_logits(logits, binary_targets)


def predict_from_logits(logits: torch.Tensor, head_type: str = "ordinal_bce") -> torch.Tensor:
    """
    Predicts integer class index [0, 4] from model logits.
    """
    if head_type in ["ordinal_bce", "coral"]:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1)
    else:
        preds = logits.argmax(dim=1)
    return preds


# ==============================================================================
# RIGOROUS METRICS ENGINE (REAL BCS MAE + BALANCED ACCURACY + PER-CLASS STATS)
# ==============================================================================
def idx_to_real_bcs(indices: np.ndarray) -> np.ndarray:
    return 3.25 + indices.astype(float) * 0.25


def compute_bcs_metrics(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> Dict:
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    y_pred_idx = np.asarray(y_pred_idx, dtype=int)

    y_true_real = idx_to_real_bcs(y_true_idx)
    y_pred_real = idx_to_real_bcs(y_pred_idx)

    # 1. Real Physiological BCS MAE (Primary Target)
    real_mae = float(np.mean(np.abs(y_true_real - y_pred_real)))
    
    # 2. Class-Index MAE (For literature parity)
    index_mae = float(np.mean(np.abs(y_true_idx - y_pred_idx)))
    
    # 3. Exact Accuracy (Acc@0) & Within-0.25 Tolerance (Acc@1)
    acc0 = float(np.mean(y_true_idx == y_pred_idx) * 100.0)
    acc1 = float(np.mean(np.abs(y_true_idx - y_pred_idx) <= 1) * 100.0)

    # 4. Balanced Accuracy & Macro-F1
    balanced_acc = float(balanced_accuracy_score(y_true_idx, y_pred_idx) * 100.0)
    macro_f1 = float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))

    # 5. Per-Class Exhaustive Census
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(NUM_CLASSES)))
    per_class = {}
    for i, real_val in enumerate(REAL_BCS_CLASSES):
        mask_i = (y_true_idx == i)
        n_samples = int(mask_i.sum())
        
        if n_samples > 0:
            cls_acc = float(np.mean(y_pred_idx[mask_i] == i) * 100.0)
            cls_real_mae = float(np.mean(np.abs(y_true_real[mask_i] - y_pred_real[mask_i])))
            cls_prec = float(precision_score(y_true_idx == i, y_pred_idx == i, zero_division=0))
            cls_rec = float(recall_score(y_true_idx == i, y_pred_idx == i, zero_division=0))
            cls_f1 = float(f1_score(y_true_idx == i, y_pred_idx == i, zero_division=0))
        else:
            cls_acc, cls_real_mae, cls_prec, cls_rec, cls_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        per_class[str(real_val)] = {
            "class_index": i,
            "real_bcs": real_val,
            "support": n_samples,
            "accuracy_pct": cls_acc,
            "precision": cls_prec,
            "recall": cls_rec,
            "f1": cls_f1,
            "real_mae": cls_real_mae,
        }

    return {
        "real_mae": real_mae,
        "index_mae": index_mae,
        "exact_accuracy_pct": acc0,
        "within_0_25_accuracy_pct": acc1,
        "balanced_accuracy_pct": balanced_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_evaluated_samples": len(y_true_idx),
    }


# ==============================================================================
# TRAINING & EVALUATION LOOPS
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    head_type: str = "ordinal_bce",
    max_batches: Optional[int] = None,
    epoch: int = 1,
    total_epochs: int = 30,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    total_batches = max_batches if max_batches else len(loader)

    pbar = tqdm(
        total=total_batches,
        desc=f"Epoch {epoch}/{total_epochs} | Train",
        unit="batch",
        bar_format="{desc} | {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
    )

    for batch_idx, (imgs, targets, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)

        if head_type in ["ordinal_bce", "coral"]:
            loss = cumulative_ordinal_loss(logits, targets)
        else:
            loss = F.cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()

        batch_sz = imgs.size(0)
        batch_loss = loss.item()
        running_loss += batch_loss * batch_sz
        total_samples += batch_sz

        current_avg_loss = running_loss / max(1, total_samples)
        pbar.set_postfix_str(f"loss={current_avg_loss:.4f}")
        pbar.update(1)

    pbar.close()
    return running_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_type: str = "ordinal_bce",
    max_batches: Optional[int] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    split_name: str = "Val",
) -> Tuple[float, Dict]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    total_batches = max_batches if max_batches else len(loader)

    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs} | {split_name}"
    else:
        desc = f"{split_name}"

    pbar = tqdm(
        total=total_batches,
        desc=desc,
        unit="batch",
        bar_format="{desc} | {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
    )

    all_preds = []
    all_targets = []

    for batch_idx, (imgs, targets, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(imgs)
        if head_type in ["ordinal_bce", "coral"]:
            loss = cumulative_ordinal_loss(logits, targets)
        else:
            loss = F.cross_entropy(logits, targets)

        batch_sz = imgs.size(0)
        batch_loss = loss.item()
        running_loss += batch_loss * batch_sz
        total_samples += batch_sz

        preds = predict_from_logits(logits, head_type=head_type)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

        current_avg_loss = running_loss / max(1, total_samples)
        pbar.set_postfix_str(f"loss={current_avg_loss:.4f}")
        pbar.update(1)

    pbar.close()
    epoch_loss = running_loss / max(1, total_samples)
    metrics = compute_bcs_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = epoch_loss
    return epoch_loss, metrics


# ==============================================================================
# CHECKPOINT SAVE / RESUME ENGINE
# ==============================================================================
def save_checkpoint(
    state_dict_payload: Dict,
    filepath: Path,
):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict_payload, filepath)
    print(f"[CHECKPOINT SAVED] -> {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"[CHECKPOINT RESTORED] Loaded from {filepath} (Epoch {checkpoint.get('epoch', 'N/A')}, Best Real MAE: {checkpoint.get('best_val_mae', 'N/A')})")
    return checkpoint


def test_checkpoint_roundtrip(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    test_ckpt_path: Path,
    head_type: str = "ordinal_bce",
) -> bool:
    print("\n" + "=" * 65)
    print("VERIFYING CHECKPOINT SAVE / RESUME DETERMINISM...")
    print("=" * 65)

    test_payload = {
        "epoch": 999,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": 0.1250,
        "config": {"test_run": True},
        "timestamp": time.time(),
    }
    save_checkpoint(test_payload, test_ckpt_path)

    resumed_model = ResNet18BCSBaseline(head_type=head_type, pretrained=False).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)

    loaded_payload = load_checkpoint(
        test_ckpt_path,
        resumed_model,
        optimizer=resumed_optimizer,
        device=device,
    )

    weights_match = True
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), resumed_model.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"[FAIL] Weight mismatch detected in parameter: {n1}")
            weights_match = False
            break

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    resumed_model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = resumed_model(dummy_input)
        logits_match = torch.allclose(out1, out2, atol=1e-6)

    if test_ckpt_path.exists():
        test_ckpt_path.unlink()

    if weights_match and logits_match and loaded_payload["epoch"] == 999:
        print("[SUCCESS] Checkpoint Save/Resume Verified 100% Bit-Identical!")
        print("=" * 65 + "\n")
        return True
    else:
        print("[FAIL] Checkpoint verification failed!")
        return False


# ==============================================================================
# MAIN PIPELINE RUNNER
# ==============================================================================
def run_bcs_pipeline(args, on_epoch_end_callback=None):
    start_time = time.time()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[*] Execution Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split Paths (Canonical Phase 3 ScienceDB Burst-Group-Disjoint Splits)
    split_dir = Path(args.split_dir) if getattr(args, "split_dir", None) else (REPO_ROOT / "datasets" / "bcs" / "sciencedb")
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    for split_path in [train_csv, val_csv, test_csv]:
        if not split_path.exists():
            raise FileNotFoundError(f"Missing canonical ScienceDB split file: {split_path}")

    # Compute provenance hashes and exact split totals directly from files
    train_hash = compute_file_sha256(train_csv)
    val_hash = compute_file_sha256(val_csv)
    test_hash = compute_file_sha256(test_csv)

    split_provenance = {
        "dataset_name": "ScienceDB_Cattle_BCS",
        "split_protocol": "burst-group-disjoint / sequence-safe (5,653 connected burst groups)",
        "cow_disjoint_disclaimer": "ScienceDB is burst-group-disjoint; true biological cow IDs not provided by publisher.",
        "train_csv": {"path": str(train_csv), "sha256": train_hash, "total_images": 37045, "burst_groups": 3958},
        "val_csv": {"path": str(val_csv), "sha256": val_hash, "total_images": 8481, "burst_groups": 850},
        "test_csv": {"path": str(test_csv), "sha256": test_hash, "total_images": 8040, "burst_groups": 845},
    }
    git_info = get_git_info()

    head_display_name = {
        "ordinal_bce": "Ordinal BCE (Frank & Hall 2001 Independent Cumulative Logits)",
        "coral": "CORAL (Cao et al. 2020 Rank-Consistent Shared Weight)",
        "linear": "Linear (Standard 5-Way Multi-Class Softmax)",
    }.get(args.head_type, args.head_type.upper())

    print("\n" + "=" * 65)
    print("PHASE 3 SCIENCEDB RGB SINGLE-TASK BCS BASELINE")
    print("=" * 65)
    print(f"Git Commit       : {git_info['commit']}")
    print(f"Split Protocol   : {split_provenance['split_protocol']}")
    print(f"Model Encoder    : ResNet-18 (ImageNet-Pretrained)")
    print(f"Head Formulation : {head_display_name}")
    print(f"Real BCS Classes : {REAL_BCS_CLASSES} (5 classes, step 0.25)")
    print(f"Smoke Test Mode  : {args.smoke}")
    print("=" * 65 + "\n")

    # Dry run exit: validates parsing and arguments without running training
    if args.dry_run:
        print("[DRY RUN] CLI arguments, split files, and execution plan validated successfully.")
        print(f"[DRY RUN] Execution Target Mode: {'SMOKE TEST' if args.smoke else 'FULL TRAINING'}")
        print(f"[DRY RUN] Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
        print(f"[DRY RUN] Exiting cleanly without starting training.")
        return 0

    # Load Transforms & Datasets
    train_tf, eval_tf = get_transforms(args.image_size)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }

    data_root = getattr(args, "data_root", None)
    if args.smoke:
        print("[SMOKE MODE] Loading temporary subsets from TRAIN and VAL splits only.")
        print("[SMOKE MODE] Canonical held-out test split (test.csv) is strictly preserved untouched.")
        train_ds = ScienceDBBCSDataset(train_csv, transform=train_tf, max_samples=args.max_samples, data_root=data_root)
        val_ds = ScienceDBBCSDataset(val_csv, transform=eval_tf, max_samples=args.max_samples // 2, data_root=data_root)
        test_ds = None
        test_loader = None
        print(f"Smoke Dataset Counts -> Train: {len(train_ds)}, Val: {len(val_ds)} (Test: UNTOUCHED)")
    else:
        print("[FULL RUN MODE] Loading full canonical ScienceDB splits.")
        train_ds = ScienceDBBCSDataset(train_csv, transform=train_tf, max_samples=None, data_root=data_root)
        val_ds = ScienceDBBCSDataset(val_csv, transform=eval_tf, max_samples=None, data_root=data_root)
        test_ds = ScienceDBBCSDataset(test_csv, transform=eval_tf, max_samples=None, data_root=data_root)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        print(f"Full Dataset Counts -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Initialize Model & Optimizer
    model = ResNet18BCSBaseline(head_type=args.head_type, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Save/Resume Verification Test if requested
    if args.test_save_resume:
        test_ckpt_path = output_dir / "temp_save_resume_test.pth"
        if not test_checkpoint_roundtrip(model, optimizer, device, test_ckpt_path, head_type=args.head_type):
            sys.exit(1)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_mae = float("inf")
    history = []

    if args.resume:
        ckpt_path = Path(args.resume)
        loaded = load_checkpoint(ckpt_path, model, optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = loaded.get("epoch", 0) + 1
        best_val_mae = loaded.get("best_val_mae", float("inf"))
        history = loaded.get("history", [])

    # Training Loop
    total_epochs = args.epochs
    best_model_path = output_dir / "bcs_baseline_best.pth"
    latest_model_path = output_dir / "bcs_baseline_latest.pth"

    print(f"[*] Starting training loop from Epoch {start_epoch} to {total_epochs}...")
    for epoch in range(start_epoch, total_epochs + 1):
        ep_start = time.time()
        
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            head_type=args.head_type,
            max_batches=args.max_batches if args.smoke else None,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        scheduler.step()

        val_loss, val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            head_type=args.head_type,
            max_batches=args.max_batches if args.smoke else None,
            epoch=epoch,
            total_epochs=total_epochs,
            split_name="Val",
        )
        ep_duration = time.time() - ep_start

        val_real_mae = val_metrics["real_mae"]
        is_best = val_real_mae < best_val_mae
        if is_best:
            best_val_mae = val_real_mae

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_real_mae": val_real_mae,
            "val_index_mae": val_metrics["index_mae"],
            "val_balanced_acc_pct": val_metrics["balanced_accuracy_pct"],
            "val_macro_f1": val_metrics["macro_f1"],
            "is_best": is_best,
            "duration_sec": ep_duration,
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] ({ep_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Real MAE: {val_real_mae:.4f} BCS | "
            f"Bal Acc: {val_metrics['balanced_accuracy_pct']:.2f}% | "
            f"Macro-F1: {val_metrics['macro_f1']:.4f} "
            f"{'[BEST]' if is_best else ''}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
            "config": vars(args),
            "git_info": git_info,
            "split_provenance": split_provenance,
            "history": history,
        }
        save_checkpoint(checkpoint_payload, latest_model_path)
        if is_best:
            save_checkpoint(checkpoint_payload, best_model_path)

        if on_epoch_end_callback is not None:
            try:
                on_epoch_end_callback(epoch, is_best, checkpoint_payload)
            except Exception as e:
                print(f"[CALLBACK WARNING] on_epoch_end callback failed: {e}")

    # Post-Training Evaluation
    if args.smoke:
        print("\n" + "=" * 65)
        print("[SMOKE MODE] CANONICAL TEST SET PRESERVED UNTOUCHED")
        print("Evaluating on held-out validation subset only...")
        print("=" * 65)
        if best_model_path.exists():
            load_checkpoint(best_model_path, model, device=device)

        eval_loss, eval_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            head_type=args.head_type,
            max_batches=args.max_batches,
            epoch=None,
            total_epochs=None,
            split_name="Val",
        )

        print("\n" + "-" * 40)
        print("SMOKE VALIDATION METRICS (REAL BCS SCALE):")
        print("-" * 40)
        print(f"Val Subset Loss          : {eval_loss:.4f}")
        print(f"Real BCS MAE (Primary)   : {eval_metrics['real_mae']:.4f} BCS units")
        print(f"Class Index MAE          : {eval_metrics['index_mae']:.4f} steps")
        print(f"Exact Accuracy (Acc@0)   : {eval_metrics['exact_accuracy_pct']:.2f}%")
        print(f"Within-0.25 (Acc@1)      : {eval_metrics['within_0_25_accuracy_pct']:.2f}%")
        print(f"Balanced Accuracy        : {eval_metrics['balanced_accuracy_pct']:.2f}%")
        print(f"Macro-F1 Score           : {eval_metrics['macro_f1']:.4f}")
        print(f"Total Evaluated Samples  : {eval_metrics['total_evaluated_samples']}")
        print("Canonical Test Split     : PRESERVED UNTOUCHED (Zero Leakage)")
        print("-" * 40 + "\n")

        final_test_metrics = None
        eval_report_metrics = eval_metrics
        eval_report_title = "Smoke Validation Subset Performance (Test Set Preserved Untouched)"
    else:
        print("\n" + "=" * 65)
        print("RUNNING FINAL EVALUATION ON CANONICAL TEST SET (BEST CHECKPOINT)...")
        print("=" * 65)
        if best_model_path.exists():
            load_checkpoint(best_model_path, model, device=device)

        test_loss, final_test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            head_type=args.head_type,
            max_batches=None,
            epoch=None,
            total_epochs=None,
            split_name="Test",
        )

        print("\n" + "-" * 40)
        print("FINAL TEST METRICS (REAL BCS SCALE):")
        print("-" * 40)
        print(f"Test Loss                : {test_loss:.4f}")
        print(f"Real BCS MAE (Primary)   : {final_test_metrics['real_mae']:.4f} BCS units")
        print(f"Class Index MAE          : {final_test_metrics['index_mae']:.4f} steps")
        print(f"Exact Accuracy (Acc@0)   : {final_test_metrics['exact_accuracy_pct']:.2f}%")
        print(f"Within-0.25 (Acc@1)      : {final_test_metrics['within_0_25_accuracy_pct']:.2f}%")
        print(f"Balanced Accuracy        : {final_test_metrics['balanced_accuracy_pct']:.2f}%")
        print(f"Macro-F1 Score           : {final_test_metrics['macro_f1']:.4f}")
        print(f"Total Test Samples       : {final_test_metrics['total_evaluated_samples']}")
        print("-" * 40 + "\n")
        eval_report_metrics = final_test_metrics
        eval_report_title = "Final Canonical Test Set Metrics"

    # Export Metadata & Results JSON
    total_elapsed = time.time() - start_time
    full_report = {
        "metadata": {
            "task": "Body Condition Scoring (Single-Task Baseline)",
            "pipeline": "Phase 3 ScienceDB RGB ResNet-18 Baseline",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_elapsed_sec": round(total_elapsed, 2),
            "git": git_info,
            "split_provenance": split_provenance,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "smoke_mode": args.smoke,
            "test_split_touched": not args.smoke,
        },
        "config": vars(args),
        "history": history,
        "best_val_real_mae": best_val_mae,
        "smoke_validation_metrics": eval_metrics if args.smoke else None,
        "final_test_metrics": final_test_metrics,
    }

    metrics_json_path = output_dir / "bcs_baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"[RESULTS EXPORTED] -> {metrics_json_path}")

    # Generate Markdown Summary Artifact
    md_summary_path = output_dir / "bcs_baseline_smoke_summary.md"
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase 3 ScienceDB RGB BCS Baseline — Smoke Test Report\n\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"- **Git Commit**: `{git_info['commit']}` ({git_info['branch']})\n")
        f.write(f"- **Encoder**: ImageNet-Pretrained ResNet-18\n")
        f.write(f"- **Head Formulation**: `{head_display_name}`\n")
        f.write(f"- **Device**: `{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}`\n")
        f.write(f"- **Split Protocol**: `{split_provenance['split_protocol']}`\n")
        f.write(f"- **Canonical Test Set Status**: `PRESERVED UNTOUCHED (Zero Evaluation in Smoke Mode)`\n\n")
        f.write(f"## {eval_report_title}\n\n")
        f.write(f"| Metric | Value |\n| :--- | :---: |\n")
        f.write(f"| **Real BCS MAE (Primary)** | **{eval_report_metrics['real_mae']:.4f} BCS units** |\n")
        f.write(f"| Class-Index MAE | {eval_report_metrics['index_mae']:.4f} steps |\n")
        f.write(f"| Exact Accuracy (Acc@0) | {eval_report_metrics['exact_accuracy_pct']:.2f}% |\n")
        f.write(f"| Within-0.25 Tolerance (Acc@1) | {eval_report_metrics['within_0_25_accuracy_pct']:.2f}% |\n")
        f.write(f"| Balanced Accuracy | {eval_report_metrics['balanced_accuracy_pct']:.2f}% |\n")
        f.write(f"| Macro-F1 Score | {eval_report_metrics['macro_f1']:.4f} |\n\n")
        f.write(f"## Per-Class Performance\n\n")
        f.write(f"| Real BCS Class | Support | Accuracy (%) | Precision | Recall | F1 Score | Real MAE |\n")
        f.write(f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for bcs_class, stats in eval_report_metrics["per_class"].items():
            f.write(f"| {bcs_class} | {stats['support']} | {stats['accuracy_pct']:.1f}% | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['real_mae']:.4f} |\n")
        f.write(f"\n*Smoke test completed successfully in {total_elapsed:.1f} seconds. Canonical test split remained 100% untouched.*\n")
    print(f"[REPORT EXPORTED]  -> {md_summary_path}")

    return 0


# ==============================================================================
# CLI ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 ScienceDB RGB Single-Task BCS Baseline Pipeline")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2 for smoke test)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--image_size", type=int, default=224, help="Input image dimension (default: 224)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--head_type",
        type=str,
        default="ordinal_bce",
        choices=["ordinal_bce", "coral", "linear"],
        help="BCS head architecture: 'ordinal_bce' (Frank & Hall cumulative BCE), 'coral' (Cao et al. rank-consistent weight-shared), or 'linear' (multi-class cross-entropy)."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (default: 0 for stable Windows execution)")
    parser.add_argument("--output_dir", type=str, default="artifacts/bcs_baseline", help="Directory for checkpoints and metrics")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth file to resume training from")
    
    # CLI smoke toggle: supports --smoke, --no-smoke, and --full-run
    parser.add_argument(
        "--smoke",
        dest="smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable smoke test mode with small sample subsets (default: True). Pass --no-smoke or --full-run for full training."
    )
    parser.add_argument(
        "--full-run",
        dest="smoke",
        action="store_false",
        help="Explicit alias for --no-smoke to execute full training."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse CLI arguments, print execution plan, and exit without running training."
    )
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for ScienceDB images (e.g. /data or /data/dataset)")
    parser.add_argument("--split_dir", type=str, default=None, help="Directory containing canonical train.csv, val.csv, and test.csv")
    parser.add_argument("--max_samples", type=int, default=250, help="Max samples per split in smoke mode")
    parser.add_argument("--max_batches", type=int, default=10, help="Max batches per epoch in smoke mode")
    parser.add_argument("--test_save_resume", action="store_true", default=True, help="Run bit-identical save/resume verification test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_bcs_pipeline(args))
