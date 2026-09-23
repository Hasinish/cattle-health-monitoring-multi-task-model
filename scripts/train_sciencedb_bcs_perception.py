# -*- coding: utf-8 -*-
"""
scripts/train_sciencedb_bcs_perception.py — Run 4 BCS Perception-Enhanced Training Engine
========================================================================================
Phase 3 Step 4.4 / Run 4: ScienceDB Perception-Enhanced Model (RGB + RT-DETR-L crop + SAM 2.1 mask guidance).

Architecture:
  - ResNet-18 modified for 4-channel input [R, G, B, Mask]
  - Channels 0-2: RGB normalized with ImageNet mean/std
  - Channel 3: SAM 2.1 BINARY foreground mask in {0.0, 1.0} (0 = background, 1 = foreground cow)
  - conv1 modified from nn.Conv2d(3, 64, 7) to nn.Conv2d(4, 64, 7)
    * Weights [:, :3, :, :] copied directly from ImageNet pretrained ResNet-18
    * Weights [:, 3:4, :, :] initialized deterministically via channel mean (or zero)
    * Baseline Run 1 ordinal ResNet-18: 11,178,564 parameters
    * Run 4 4-channel ordinal ResNet-18: 11,181,700 parameters
    * Parameter delta: +3,136 parameters (+0.028%)
  - Head: Ordinal BCE (Frank & Hall, 2001 cumulative head Linear(512, 4))
  - Pose: EXCLUDED from deadline run (unreliable rear anatomy, missing pelvic landmarks)
  - Viewpoint: EXCLUDED from deadline run (cross-domain transfer not yet certified)

Augmentation (Fair Match to Run 1):
  - Resize 224
  - RandomHorizontalFlip(p=0.5) [Synchronized across RGB + Mask]
  - RandomRotation(15 degrees) [Synchronized across RGB + Mask]
  - ColorJitter(brightness=0.1, contrast=0.1) [Applied to RGB ONLY, never mask]

Evaluation:
  - Primary metric: Real BCS MAE computed on scale 3.25 to 4.25 (step 0.25)
  - Secondary metrics: Acc@1 (+/- 0.25), Acc@0 (exact), Balanced Accuracy, Macro-F1
  - Model Selection: Best checkpoint selected by validation Real MAE (train/val ONLY).
  - Test Integrity: Frozen test split is evaluated exactly ONCE post-training. Zero test access during training.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


class ScienceDBPerceptionDataset(Dataset):
    """
    Dataset loader for ScienceDB perception crops and masks.
    Enforces strict perception failure filtering and synchronized augmentations:
      - Rows where detection_status != "detected" or sam_status != "segmented" are EXCLUDED.
      - Fairly matches Run 1 baseline:
        * Resize(224, 224)
        * RandomHorizontalFlip(p=0.5) synchronized across RGB + mask
        * RandomRotation(15 degrees) synchronized across RGB + mask
        * ColorJitter(brightness=0.1, contrast=0.1) on RGB ONLY (never mask)
      - Mask values in {0.0, 1.0} representing BINARY foreground mask guidance.
    """
    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path,
        is_train: bool = True,
        image_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.is_train = is_train
        self.image_size = image_size

        df = pd.read_csv(self.manifest_path)
        total_raw = len(df)

        # Failure Exclusion Policy:
        # Strictly require detection_status == "detected" AND sam_status == "segmented"
        valid_mask = (df["detection_status"] == "detected") & (df["sam_status"] == "segmented")
        n_excluded = int((~valid_mask).sum())
        df = df[valid_mask].copy().reset_index(drop=True)

        split_name = self.manifest_path.stem.replace("_perception", "")
        print(f"  [Filter: {split_name:5s}] Canonical rows: {total_raw} | Excluded perception failures: {n_excluded} | Usable training samples: {len(df)}", flush=True)

        if max_samples and max_samples < len(df):
            # Deterministic stratified sampling for smoke testing
            samples_per_class = max(1, max_samples // NUM_CLASSES)
            subsets = []
            for lbl in REAL_BCS_CLASSES:
                sub = df[df["label"] == lbl].head(samples_per_class)
                subsets.append(sub)
            df = pd.concat(subsets, ignore_index=True)
            if len(df) > max_samples:
                df = df.iloc[:max_samples]

        self.df = df.reset_index(drop=True)
        self.samples = []
        for idx, row in self.df.iterrows():
            crop_p = self.cache_dir / str(row["crop_rel_path"])
            mask_p = self.cache_dir / str(row["mask_rel_path"])
            raw_label = float(row["label"])
            target_idx = LABEL_TO_IDX[raw_label]
            self.samples.append((str(crop_p), str(mask_p), target_idx, raw_label))

        self.rgb_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        crop_path, mask_path, target_idx, raw_label = self.samples[idx]

        # Load RGB crop and grayscale binary mask
        try:
            crop_img = Image.open(crop_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening crop at {crop_path}: {e}")

        try:
            mask_img = Image.open(mask_path).convert("L")
        except Exception as e:
            raise IOError(f"Error opening mask at {mask_path}: {e}")

        # 1. Resize to target dimension
        crop_img = TF.resize(crop_img, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask_img = TF.resize(mask_img, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        # 2. Synchronized augmentations matching Run 1 baseline
        if self.is_train:
            # Synchronized random horizontal flip (p=0.5)
            if np.random.rand() > 0.5:
                crop_img = TF.hflip(crop_img)
                mask_img = TF.hflip(mask_img)

            # Synchronized random rotation (+/- 15 degrees)
            angle = float(np.random.uniform(-15.0, 15.0))
            crop_img = TF.rotate(crop_img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_img = TF.rotate(mask_img, angle, interpolation=TF.InterpolationMode.NEAREST)

            # ColorJitter applied to RGB ONLY, NEVER to mask
            crop_img = self.color_jitter(crop_img)

        # 3. Convert to float tensors
        rgb_tensor = TF.to_tensor(crop_img)  # [3, H, W] in [0.0, 1.0]
        rgb_normed = self.rgb_norm(rgb_tensor)

        mask_tensor = TF.to_tensor(mask_img)  # [1, H, W] in {0.0, 1.0}
        four_channel = torch.cat([rgb_normed, mask_tensor], dim=0)  # [4, H, W]

        return four_channel, torch.tensor(target_idx, dtype=torch.long), raw_label


class OrdinalBCEHead(nn.Module):
    """Cumulative Frank & Hall (2001) ordinal head: Linear(in_features, K-1)."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes - 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ResNet18BCSPerception(nn.Module):
    """
    4-Channel ResNet-18 with ImageNet Pretraining + Deterministic Mask Channel Initialization.
    """
    def __init__(
        self,
        pretrained: bool = True,
        head_type: str = "ordinal_bce",
        mask_init: str = "mean",
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        old_conv1 = base.conv1
        # Create 4-channel conv1
        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )

        with torch.no_grad():
            # Copy ImageNet RGB weights
            new_conv1.weight[:, :3, :, :] = old_conv1.weight
            # Deterministic mask channel initialization
            if mask_init == "mean":
                new_conv1.weight[:, 3:4, :, :] = old_conv1.weight.mean(dim=1, keepdim=True)
            elif mask_init == "zero":
                new_conv1.weight[:, 3:4, :, :] = 0.0
            else:
                raise ValueError(f"Unknown mask_init mode: {mask_init}")

        base.conv1 = new_conv1
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        if head_type == "ordinal_bce":
            self.head = OrdinalBCEHead(in_features=512, num_classes=NUM_CLASSES)
        elif head_type == "linear":
            self.head = nn.Linear(512, NUM_CLASSES)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

        self.head_type = head_type

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
        features = torch.flatten(x, 1)
        logits = self.head(features)
        return logits


def ordinal_targets_from_class_indices(
    class_indices: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Constructs binary threshold targets for ordinal BCE:
    Class 0 (3.25): [0, 0, 0, 0]
    Class 1 (3.50): [1, 0, 0, 0]
    Class 2 (3.75): [1, 1, 0, 0]
    Class 3 (4.00): [1, 1, 1, 0]
    Class 4 (4.25): [1, 1, 1, 1]
    """
    batch_size = class_indices.size(0)
    k_minus_1 = num_classes - 1
    thresholds = torch.arange(k_minus_1, device=device).unsqueeze(0).expand(batch_size, -1)
    targets = (class_indices.unsqueeze(1) > thresholds).float()
    return targets


def logits_to_predictions(logits: torch.Tensor, head_type: str = "ordinal_bce") -> Tuple[np.ndarray, np.ndarray]:
    """Converts raw model logits to predicted class indices and real physical BCS values."""
    if head_type == "ordinal_bce":
        probs = torch.sigmoid(logits)
        # Sum of binary threshold exceedances
        pred_indices = (probs > 0.5).sum(dim=1).cpu().numpy()
    elif head_type == "linear":
        pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    pred_indices = np.clip(pred_indices, 0, NUM_CLASSES - 1)
    real_bcs = np.array([IDX_TO_LABEL[idx] for idx in pred_indices])
    return pred_indices, real_bcs


def evaluate_metrics(y_true_indices: np.ndarray, y_pred_indices: np.ndarray) -> Dict:
    """Calculates all canonical BCS evaluation metrics identical to Run 1 baseline."""
    y_true_real = np.array([IDX_TO_LABEL[idx] for idx in y_true_indices])
    y_pred_real = np.array([IDX_TO_LABEL[idx] for idx in y_pred_indices])

    mae_real = float(np.mean(np.abs(y_true_real - y_pred_real)))
    acc_0 = float(np.mean(y_true_indices == y_pred_indices))
    acc_1 = float(np.mean(np.abs(y_true_indices - y_pred_indices) <= 1))
    bal_acc = float(balanced_accuracy_score(y_true_indices, y_pred_indices))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    precision = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    recall = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(NUM_CLASSES))).tolist()

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


def evaluate_test_split(
    model: nn.Module,
    manifest_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Evaluates best trained checkpoint on the frozen held-out test split exactly once.
    Test metrics are saved separately and do NOT influence model selection.
    """
    test_manifest = manifest_dir / "test_perception.csv"
    if not test_manifest.exists():
        raise FileNotFoundError(f"Missing test perception manifest: {test_manifest}")

    print("\n" + "=" * 75)
    print("  RUN 4: HELD-OUT FROZEN TEST EVALUATION (ONE-TIME POST-TRAINING)")
    print(f"  Test Manifest: {test_manifest}")
    print("=" * 75)

    num_workers = 0 if sys.platform == "win32" else 4
    test_dataset = ScienceDBPerceptionDataset(
        manifest_path=test_manifest,
        cache_dir=cache_dir,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0
    n_test = 0
    all_true_indices, all_pred_indices = [], []

    pbar = tqdm(test_loader, desc="[Test] Held-Out Evaluation", file=sys.stdout, leave=True, dynamic_ncols=True)
    with torch.no_grad():
        for imgs, target_indices, _ in pbar:
            imgs = imgs.to(device)
            target_indices = target_indices.to(device)
            targets = ordinal_targets_from_class_indices(target_indices, device=device)

            logits = model(imgs)
            loss = criterion(logits, targets)
            test_loss += loss.item() * len(target_indices)
            n_test += len(target_indices)

            pred_indices, _ = logits_to_predictions(logits, head_type="ordinal_bce")
            all_pred_indices.extend(pred_indices)
            all_true_indices.extend(target_indices.cpu().numpy())
            pbar.set_postfix(test_loss=f"{loss.item():.4f}")

    avg_test_loss = test_loss / max(1, n_test)
    test_metrics = evaluate_metrics(np.array(all_true_indices), np.array(all_pred_indices))
    test_summary = {
        "test_loss": round(avg_test_loss, 4),
        "total_test_samples": len(test_dataset),
        **test_metrics,
    }

    test_metrics_file = output_dir / "bcs_perception_test_metrics.json"
    with open(test_metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    print(f"\n✓ Held-out Test Real MAE: {test_metrics['real_mae']:.4f} BCS units")
    print(f"✓ Held-out Test Acc@1:    {test_metrics['acc_1']*100:.2f}%")
    print(f"✓ Held-out Test Acc@0:    {test_metrics['acc_0']*100:.2f}%")
    print(f"✓ Held-out Test Bal Acc:  {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"✓ Held-out Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"✓ Saved test metrics to:  {test_metrics_file}")
    print("=" * 75)
    return test_summary


def train_pipeline(
    manifest_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    mask_init: str = "mean",
    smoke: bool = False,
    max_samples: Optional[int] = None,
    eval_test: bool = False,
    device_name: Optional[str] = None,
) -> Dict:
    """
    Main training and validation loop for Run 4 Perception-Enhanced Model.
    Model selection is performed strictly on validation Real MAE using train/val ONLY.
    """
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("  RUN 4: SCIENTEDB BCS PERCEPTION-ENHANCED TRAINING")
    print(f"  Device:         {device}")
    print(f"  Cache Dir:      {cache_dir}")
    print(f"  Manifest Dir:   {manifest_dir}")
    print(f"  Mask Init:      {mask_init} (conv1 weights: +3,136 params)")
    print(f"  Mode:           {'SMOKE TEST (10 samples)' if smoke else 'FULL TRAINING'}")
    print(f"  Epochs:         {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"  Test Eval:      {'YES (post-training one-time)' if (eval_test and not smoke) else 'NO (smoke / train-val only)'}")
    print("=" * 75)

    train_manifest = manifest_dir / "train_perception.csv"
    val_manifest = manifest_dir / "val_perception.csv"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(f"Missing perception manifests in {manifest_dir}")

    n_samples = 10 if smoke else max_samples
    train_dataset = ScienceDBPerceptionDataset(
        manifest_path=train_manifest, cache_dir=cache_dir, is_train=True, max_samples=n_samples,
    )
    val_dataset = ScienceDBPerceptionDataset(
        manifest_path=val_manifest, cache_dir=cache_dir, is_train=False, max_samples=n_samples,
    )

    num_workers = 0 if smoke or sys.platform == "win32" else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\nLoaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.", flush=True)

    model = ResNet18BCSPerception(pretrained=True, head_type="ordinal_bce", mask_init=mask_init)
    model.to(device)

    # Verification: Parameter count comparison
    baseline_params = 11178564  # Run 1 ResNet-18 + Ordinal BCE Head (Linear(512, 4))
    model_params = sum(p.numel() for p in model.parameters())
    param_delta = model_params - baseline_params
    pct_delta = (param_delta / baseline_params) * 100.0
    print(f"Model Parameters: {model_params:,} (Baseline: {baseline_params:,}, Delta: +{param_delta:,} params / +{pct_delta:.3f}%)", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_mae = float("inf")
    best_metrics = {}
    history = []

    best_ckpt_path = output_dir / "bcs_perception_best.pth"
    latest_ckpt_path = output_dir / "bcs_perception_latest.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0

        pbar_train = tqdm(
            train_loader,
            desc=f"Epoch [{epoch:02d}/{epochs:02d}] Train",
            file=sys.stdout,
            leave=False,
            dynamic_ncols=True,
        )
        for imgs, target_indices, _ in pbar_train:
            imgs = imgs.to(device)
            target_indices = target_indices.to(device)
            targets = ordinal_targets_from_class_indices(target_indices, device=device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(target_indices)
            n_train += len(target_indices)
            pbar_train.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train)

        # Validation pass
        model.eval()
        val_loss = 0.0
        n_val = 0
        all_true_indices, all_pred_indices = [], []

        pbar_val = tqdm(
            val_loader,
            desc=f"Epoch [{epoch:02d}/{epochs:02d}] Val",
            file=sys.stdout,
            leave=False,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for imgs, target_indices, _ in pbar_val:
                imgs = imgs.to(device)
                target_indices = target_indices.to(device)
                targets = ordinal_targets_from_class_indices(target_indices, device=device)

                logits = model(imgs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(target_indices)
                n_val += len(target_indices)

                pred_indices, _ = logits_to_predictions(logits, head_type="ordinal_bce")
                all_pred_indices.extend(pred_indices)
                all_true_indices.extend(target_indices.cpu().numpy())
                pbar_val.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / max(1, n_val)
        val_metrics = evaluate_metrics(np.array(all_true_indices), np.array(all_pred_indices))
        val_mae = val_metrics["real_mae"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            **val_metrics,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "epoch_time_s": round(time.time() - t0, 2),
        }
        history.append(epoch_record)

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_metrics = {**epoch_record, "best_epoch": epoch}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_real_mae": val_mae,
                "metrics": val_metrics,
                "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "mask_init": mask_init},
            }, best_ckpt_path)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latest_metrics": epoch_record,
        }, latest_ckpt_path)

        best_marker = " * BEST" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f} BCS | Acc@1: {val_metrics['acc_1']*100:.1f}% | Bal Acc: {val_metrics['balanced_accuracy']*100:.1f}%{best_marker}", flush=True)

    summary = {
        "model_type": "ResNet18BCSPerception_4Ch",
        "parameters": model_params,
        "baseline_parameters": baseline_params,
        "parameter_delta": param_delta,
        "epochs_trained": epochs,
        "best_epoch": best_metrics.get("best_epoch"),
        "best_val_real_mae": best_val_mae,
        "best_metrics": best_metrics,
        "history": history,
    }

    metrics_file = output_dir / "bcs_perception_training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"✓ Training finished. Best Val Real MAE: {best_val_mae:.4f} at Epoch {best_metrics.get('best_epoch')}")
    print(f"✓ Checkpoints saved to: {output_dir}")
    print("=" * 75)

    # Post-training: evaluate held-out test set once with best checkpoint if requested
    if eval_test and not smoke:
        test_manifest = manifest_dir / "test_perception.csv"
        if test_manifest.exists():
            print(f"\n[*] Loading best checkpoint from {best_ckpt_path} for one-time held-out test evaluation...")
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_results = evaluate_test_split(
                model=model,
                manifest_dir=manifest_dir,
                cache_dir=cache_dir,
                output_dir=output_dir,
                batch_size=batch_size,
                device=device,
            )
            summary["test_metrics"] = test_results
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        else:
            print(f"[!] Warning: test manifest not found at {test_manifest}. Skipping test evaluation.")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run 4 ScienceDB BCS Perception Training")
    parser.add_argument("--manifest-dir", type=str, default="artifacts/bcs_perception_smoke/cache/manifests", help="Dir with train_perception.csv and val_perception.csv")
    parser.add_argument("--cache-dir", type=str, default="artifacts/bcs_perception_smoke/cache", help="Perception cache root")
    parser.add_argument("--output-dir", type=str, default="artifacts/bcs_perception_smoke/checkpoints", help="Output checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask-init", type=str, default="mean", choices=["mean", "zero"], help="Initialization of 4th mask channel in conv1")
    parser.add_argument("--smoke", action="store_true", help="Smoke test on small subset")
    parser.add_argument("--eval-test", action="store_true", help="Run one-time test evaluation after training completes")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_pipeline(
        manifest_dir=Path(args.manifest_dir),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_init=args.mask_init,
        smoke=args.smoke,
        eval_test=args.eval_test,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
