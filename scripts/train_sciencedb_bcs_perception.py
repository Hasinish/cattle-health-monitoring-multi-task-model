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
Test Integrity & Wording:
  - Canonical test labels/data were NOT used for training or checkpoint selection.
  - A small test-subset plumbing evaluation was performed during pipeline verification.
  - Final full Run 4 test evaluation remains post-training only.
  - Test metrics must never affect checkpoint or hyperparameter selection.
"""

import os
import sys
import time
import json
import argparse
import concurrent.futures
from pathlib import Path, PureWindowsPath
from typing import Dict, Tuple, List, Optional
import cv2
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
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


def resolve_image_path(raw_path: str, data_root: Optional[Path] = None) -> Path:
    """Resolve raw image path across Windows laptop and Linux/Modal environments."""
    p = Path(raw_path)
    if data_root is None:
        if p.exists():
            return p
        if Path("datasets/bcs/sciencedb_bcs/dataset").exists():
            data_root = Path("datasets/bcs/sciencedb_bcs/dataset")
        elif Path("/data/dataset").exists():
            data_root = Path("/data/dataset")
        elif Path("/data").exists():
            data_root = Path("/data")

    if data_root is not None:
        data_root = Path(data_root)
        parts = PureWindowsPath(raw_path).parts
        if len(parts) >= 3 and parts[-3] == "dataset":
            c1 = data_root / parts[-3] / parts[-2] / parts[-1]
            if c1.exists():
                return c1
            c2 = data_root / parts[-2] / parts[-1]
            if c2.exists():
                return c2
            return c1
        elif len(parts) >= 2:
            candidate = data_root / parts[-2] / parts[-1]
            if candidate.exists():
                return candidate
            return candidate
    return p


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_gpu_augmentations(batch: torch.Tensor, is_train: bool = True) -> torch.Tensor:
    """
    Batched GPU data augmentation and ImageNet normalization.
    Input: batch of shape [B, 4, 224, 224] uint8 on GPU (RGB 0..255, Mask 0..1).
    Output: batch of shape [B, 4, 224, 224] float32 on GPU.
    """
    if is_train:
        # 1. Synchronized Random Horizontal Flip (p=0.5)
        if torch.rand(1).item() < 0.5:
            batch = torch.flip(batch, dims=[-1])

        # 2. Synchronized Random Rotation (+/- 15 degrees)
        angle = float(torch.empty(1).uniform_(-15.0, 15.0))
        rgb = TF.rotate(batch[:, :3], angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(batch[:, 3:4], angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. ColorJitter on RGB ONLY (never mask)
        if torch.rand(1).item() < 0.5:
            b_factor = float(torch.empty(1).uniform_(0.9, 1.1))
            rgb = TF.adjust_brightness(rgb, b_factor)
        if torch.rand(1).item() < 0.5:
            c_factor = float(torch.empty(1).uniform_(0.9, 1.1))
            rgb = TF.adjust_contrast(rgb, c_factor)
    else:
        rgb = batch[:, :3]
        mask = batch[:, 3:4]

    # 4. ImageNet Normalization on RGB, strictly binary float on Mask
    rgb_norm = TF.normalize(rgb.float().div(255.0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.cat([rgb_norm, mask.float()], dim=1)


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
      - In-Memory RAM Preloading: loads all crops/masks into compact uint8 tensors [4, 224, 224] in RAM.
        Zero network disk seeks and zero PIL CPU resizing overhead during training!
    """
    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path,
        is_train: bool = True,
        image_size: int = 224,
        max_samples: Optional[int] = None,
        preload_ram: bool = True,
        num_preload_workers: int = 64,
    ):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.is_train = is_train
        self.image_size = image_size
        self.preload_ram = preload_ram

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
            self.samples.append((str(crop_p), str(mask_p), target_idx, raw_label, str(row["image_path"])))

        self.rgb_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1)

        self.cached_tensors: Optional[torch.Tensor] = None
        self.cached_targets: Optional[torch.Tensor] = None
        self.cached_raw_labels: Optional[torch.Tensor] = None

        split_name = self.manifest_path.stem.replace("_perception", "")
        packed_path = self.cache_dir / "packed" / f"{split_name}_bcs_224.pt"

        if packed_path.exists() and (max_samples is None or max_samples >= len(self.samples)):
            t0 = time.time()
            print(f"[*] Found pre-packed dataset at {packed_path}. Loading directly into RAM...", flush=True)
            payload = torch.load(packed_path, map_location="cpu", weights_only=False)
            self.cached_tensors = payload["tensors"]
            self.cached_targets = payload["targets"]
            self.cached_raw_labels = payload["raw_labels"]
            elapsed = time.time() - t0
            mb = (self.cached_tensors.element_size() * self.cached_tensors.nelement()) / (1024 * 1024)
            print(f"[*] Pre-packed {split_name} loaded: {len(self.cached_tensors)} samples ({mb:.1f} MB) in {elapsed:.1f}s ({mb/max(0.1, elapsed):.1f} MB/s)!", flush=True)
        elif self.preload_ram and len(self.samples) > 0:
            self._preload_into_ram(num_workers=num_preload_workers)

    def _preload_into_ram(self, num_workers: int = 16):
        t0 = time.time()
        split_name = self.manifest_path.stem.replace("_perception", "")
        N = len(self.samples)
        cached_data = np.empty((N, 4, self.image_size, self.image_size), dtype=np.uint8)

        def _load_single(idx: int):
            crop_path, mask_path, _, _, _ = self.samples[idx]
            crop_bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
            if crop_bgr is None:
                raise IOError(f"Failed to read crop at {crop_path}")
            if crop_bgr.shape[0] != self.image_size or crop_bgr.shape[1] != self.image_size:
                crop_224 = cv2.resize(crop_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            else:
                crop_224 = crop_bgr
            rgb_224 = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)

            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise IOError(f"Failed to read mask at {mask_path}")
            if mask_gray.shape[0] != self.image_size or mask_gray.shape[1] != self.image_size:
                mask_224 = cv2.resize(mask_gray, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            else:
                mask_224 = mask_gray
            mask_bin = (mask_224 > 127).astype(np.uint8)

            cached_data[idx, :3] = rgb_224.transpose(2, 0, 1)
            cached_data[idx, 3] = mask_bin

        workers = max(1, min(num_workers, 16))
        print(f"[*] Preloading {split_name} ({N} samples) with {workers} workers...", flush=True)

        log_interval = max(500, N // 20)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_load_single, i) for i in range(N)]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                fut.result()
                if i % log_interval == 0 or i == N:
                    elapsed = time.time() - t0
                    rate = i / max(0.1, elapsed)
                    print(f"  [RAM Preload] {split_name}: {i:,} / {N:,} ({i/N*100:.1f}%) | {elapsed:.1f}s | {rate:.1f} img/s", flush=True)

        self.cached_tensors = torch.from_numpy(cached_data)
        self.cached_targets = torch.tensor([s[2] for s in self.samples], dtype=torch.long)
        self.cached_raw_labels = torch.tensor([s[3] for s in self.samples], dtype=torch.float32)

        elapsed = time.time() - t0
        mb = (self.cached_tensors.element_size() * self.cached_tensors.nelement()) / (1024 * 1024)
        print(f"[*] RAM Preload Complete: {N} samples ({mb:.1f} MB) in {elapsed:.1f}s ({N/max(0.1, elapsed):.1f} samples/s).", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if self.cached_tensors is not None:
            # Sliced in RAM as uint8 tensor [4, 224, 224]; transforms are performed on GPU batch
            return self.cached_tensors[idx], self.cached_targets[idx], float(self.cached_raw_labels[idx])

        # Fallback disk path (when preload_ram=False)
        crop_path, mask_path, target_idx, raw_label, _ = self.samples[idx]

        try:
            crop_img = Image.open(crop_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening crop at {crop_path}: {e}")

        try:
            mask_img = Image.open(mask_path).convert("L")
        except Exception as e:
            raise IOError(f"Error opening mask at {mask_path}: {e}")

        crop_img = TF.resize(crop_img, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask_img = TF.resize(mask_img, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            if np.random.rand() > 0.5:
                crop_img = TF.hflip(crop_img)
                mask_img = TF.hflip(mask_img)

            angle = float(np.random.uniform(-15.0, 15.0))
            crop_img = TF.rotate(crop_img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_img = TF.rotate(mask_img, angle, interpolation=TF.InterpolationMode.NEAREST)

            crop_img = self.color_jitter(crop_img)

        rgb_tensor = TF.to_tensor(crop_img)
        rgb_normed = self.rgb_norm(rgb_tensor)

        mask_tensor = TF.to_tensor(mask_img)
        four_channel = torch.cat([rgb_normed, mask_tensor], dim=0)

        return four_channel, torch.tensor(target_idx, dtype=torch.long), raw_label


class MatchedRGBDataset(Dataset):
    """
    Dataset loader for original RGB ScienceDB images corresponding to the
    EXACT SAME successful-perception subset from test_perception.csv.
    Uses Run 1 baseline evaluation preprocessing:
      - Resize(224, 224)
      - ToTensor()
      - ImageNet Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    def __init__(
        self,
        matched_df: pd.DataFrame,
        data_dir: Optional[Path] = None,
        image_size: int = 224,
        preload_ram: bool = True,
        num_preload_workers: int = 64,
    ):
        self.df = matched_df.reset_index(drop=True)
        self.data_dir = Path(data_dir) if data_dir else None
        self.image_size = image_size
        self.preload_ram = preload_ram
        self.samples = []
        for _, row in self.df.iterrows():
            raw_path = str(row["image_path"])
            resolved = resolve_image_path(raw_path, data_root=self.data_dir)
            raw_label = float(row["label"])
            target_idx = LABEL_TO_IDX[raw_label]
            self.samples.append((str(resolved), raw_path, target_idx, raw_label))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.cached_tensors: Optional[torch.Tensor] = None
        self.cached_targets: Optional[torch.Tensor] = None
        self.cached_raw_labels: Optional[torch.Tensor] = None

        if self.preload_ram and len(self.samples) > 0:
            self._preload_into_ram(num_workers=num_preload_workers)

    def _preload_into_ram(self, num_workers: int = 64):
        t0 = time.time()
        desc = "Preloading Matched RGB Baseline test images into RAM"
        N = len(self.samples)
        cached_data = np.empty((N, 3, self.image_size, self.image_size), dtype=np.uint8)

        def _load_single(idx: int):
            resolved, _, _, _ = self.samples[idx]
            bgr = cv2.imread(resolved, cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"Failed to read image at {resolved}")
            if bgr.shape[0] != self.image_size or bgr.shape[1] != self.image_size:
                r224 = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            else:
                r224 = bgr
            rgb_224 = cv2.cvtColor(r224, cv2.COLOR_BGR2RGB)
            cached_data[idx] = rgb_224.transpose(2, 0, 1)

        workers = max(1, min(num_workers, 64))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(_load_single, range(N)), total=N, desc=desc, ncols=80, file=sys.stdout))

        self.cached_tensors = torch.from_numpy(cached_data)
        self.cached_targets = torch.tensor([s[2] for s in self.samples], dtype=torch.long)
        self.cached_raw_labels = torch.tensor([s[3] for s in self.samples], dtype=torch.float32)

        elapsed = time.time() - t0
        mb = (self.cached_tensors.element_size() * self.cached_tensors.nelement()) / (1024 * 1024)
        print(f"[*] RAM Preload Complete (Matched RGB): {N} samples ({mb:.1f} MB) in {elapsed:.1f}s ({N/max(0.1, elapsed):.1f} samples/s).", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if self.cached_tensors is not None:
            rgb = self.cached_tensors[idx].float().div(255.0)
            norm = TF.normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return norm, self.cached_targets[idx], float(self.cached_raw_labels[idx])

        resolved_p, raw_p, target_idx, raw_label = self.samples[idx]
        try:
            img = Image.open(resolved_p).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening image at {resolved_p} (raw: {raw_p}): {e}")
        tensor = self.transform(img)
        return tensor, torch.tensor(target_idx, dtype=torch.long), raw_label


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


class ResNet18BCSBaseline(nn.Module):
    """
    Phase 3 Canonical ResNet-18 ImageNet Baseline for Body Condition Scoring.
    Frank & Hall (2001) cumulative BCE with independent weights.
    Matches artifacts/bcs_baseline/bcs_baseline_best.pth exactly.
    """
    def __init__(self, head_type: str = "ordinal_bce", pretrained: bool = False):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
        self.head = OrdinalBCEHead(512, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        logits = self.head(feat)
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
    baseline_ckpt_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Evaluates best trained Run 4 checkpoint on successful-perception test samples and
    conducts a fair matched-subset baseline comparison against the existing Run 1 baseline.

    Integrity Protocol:
      - Canonical test labels/data were NOT used for training or checkpoint selection.
      - A small test-subset plumbing evaluation was performed during pipeline verification.
      - Final full Run 4 test evaluation remains post-training only.
      - Test metrics must never affect checkpoint or hyperparameter selection (val Real MAE only).
      - The only statistically and scientifically valid direct comparison is between
        Run 1 matched subset and Run 4 matched subset on the identical image identities.
    """
    test_manifest = manifest_dir / "test_perception.csv"
    if not test_manifest.exists():
        raise FileNotFoundError(f"Missing test perception manifest: {test_manifest}")

    df_test_raw = pd.read_csv(test_manifest)
    total_manifest_samples = len(df_test_raw)
    canonical_test_count = 8040  # Canonical full ScienceDB test split size

    # Filter for successful perception samples: detection_status == 'detected' AND sam_status == 'segmented'
    valid_mask = (df_test_raw["detection_status"] == "detected") & (df_test_raw["sam_status"] == "segmented")
    matched_df = df_test_raw[valid_mask].copy().reset_index(drop=True)
    successful_perception_count = len(matched_df)
    n_det_fail = int((df_test_raw["detection_status"] != "detected").sum())
    n_sam_fail = int(((df_test_raw["detection_status"] == "detected") & (df_test_raw["sam_status"] != "segmented")).sum())
    coverage_pct = round((successful_perception_count / total_manifest_samples) * 100, 2) if total_manifest_samples > 0 else 0.0

    print("\n" + "=" * 75)
    print("  RUN 4: POST-TRAINING TEST EVALUATION & FAIR MATCHED BASELINE COMPARISON")
    print(f"  Test Manifest:               {test_manifest}")
    print(f"  Total Manifest Rows:         {total_manifest_samples}")
    print(f"  Canonical Test Count:        {canonical_test_count}")
    print(f"  Successful Perception Count: {successful_perception_count}")
    print(f"  Excluded Detection Failures: {n_det_fail}")
    print(f"  Excluded SAM Failures:       {n_sam_fail}")
    print(f"  Perception Coverage Rate:    {coverage_pct}%")
    print("  Note: Model selection conducted strictly on validation Real MAE (train/val only).")
    print("=" * 75)

    num_workers = 0 if sys.platform == "win32" else 8

    # --------------------------------------------------------------------------
    # Tier 3: Run 4 Perception-Enhanced Model on Successful Perception Test Samples
    # --------------------------------------------------------------------------
    test_dataset = ScienceDBPerceptionDataset(
        manifest_path=test_manifest,
        cache_dir=cache_dir,
        is_train=False,
        preload_ram=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0
    n_test = 0
    all_true_indices, all_pred_indices = [], []

    pbar = tqdm(test_loader, desc="[Test] Run 4 Perception (Matched)", file=sys.stdout, leave=True, dynamic_ncols=True)
    with torch.no_grad():
        for imgs, target_indices, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            target_indices = target_indices.to(device, non_blocking=True)
            if imgs.dtype == torch.uint8:
                imgs = apply_gpu_augmentations(imgs, is_train=False)
            targets = ordinal_targets_from_class_indices(target_indices, device=device)

            logits = model(imgs)
            loss = criterion(logits, targets)
            test_loss += loss.item() * len(target_indices)
            n_test += len(target_indices)

            pred_indices, _ = logits_to_predictions(logits, head_type="ordinal_bce")
            all_pred_indices.extend(pred_indices)
            all_true_indices.extend(target_indices.cpu().numpy())
            pbar.set_postfix(run4_loss=f"{loss.item():.4f}")

    avg_test_loss = test_loss / max(1, n_test)
    test_metrics = evaluate_metrics(np.array(all_true_indices), np.array(all_pred_indices))
    run4_test_summary = {
        "test_loss": round(avg_test_loss, 4),
        "total_test_samples": len(test_dataset),
        **test_metrics,
    }

    test_metrics_file = output_dir / "bcs_perception_test_metrics.json"
    with open(test_metrics_file, "w", encoding="utf-8") as f:
        json.dump(run4_test_summary, f, indent=2)

    # --------------------------------------------------------------------------
    # Tier 2: Existing Run 1 RGB Baseline Evaluated on the EXACT SAME Subset
    # --------------------------------------------------------------------------
    resolved_baseline_ckpt = None
    if baseline_ckpt_path is not None:
        p = Path(baseline_ckpt_path)
        if p.exists():
            resolved_baseline_ckpt = p
    if resolved_baseline_ckpt is None:
        for candidate in [
            Path("/checkpoints/bcs_baseline/bcs_baseline_best.pth"),
            Path("artifacts/bcs_baseline/bcs_baseline_best.pth"),
        ]:
            if candidate.exists():
                resolved_baseline_ckpt = candidate
                break

    run1_matched_summary = None
    if resolved_baseline_ckpt is not None:
        print(f"\n[*] Evaluating EXISTING Run 1 RGB baseline on EXACT SAME matched test subset ({len(test_dataset)} samples)...", flush=True)
        print(f"    Baseline Checkpoint: {resolved_baseline_ckpt}", flush=True)
        baseline_model = ResNet18BCSBaseline(head_type="ordinal_bce", pretrained=False)
        base_ckpt = torch.load(resolved_baseline_ckpt, map_location=device)
        baseline_model.load_state_dict(base_ckpt["model_state_dict"])
        baseline_model.to(device)
        baseline_model.eval()

        baseline_dataset = MatchedRGBDataset(
            matched_df=test_dataset.df,
            data_dir=data_dir,
            image_size=224,
        )
        # Strict Verification: Assert exact sample identity alignment
        assert len(baseline_dataset) == len(test_dataset), (
            f"Dataset length mismatch: baseline {len(baseline_dataset)} != perception {len(test_dataset)}"
        )
        for idx in range(len(test_dataset)):
            assert baseline_dataset.samples[idx][1] == test_dataset.samples[idx][4], (
                f"Image ID mismatch at index {idx}: {baseline_dataset.samples[idx][1]} != {test_dataset.samples[idx][4]}"
            )

        base_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        base_loss = 0.0
        n_base = 0
        base_true_indices, base_pred_indices = [], []

        pbar_base = tqdm(base_loader, desc="[Test] Run 1 Baseline (Matched)", file=sys.stdout, leave=True, dynamic_ncols=True)
        with torch.no_grad():
            for imgs, target_indices, _ in pbar_base:
                imgs = imgs.to(device)
                target_indices = target_indices.to(device)
                targets = ordinal_targets_from_class_indices(target_indices, device=device)

                logits = baseline_model(imgs)
                loss = criterion(logits, targets)
                base_loss += loss.item() * len(target_indices)
                n_base += len(target_indices)

                pred_indices, _ = logits_to_predictions(logits, head_type="ordinal_bce")
                base_pred_indices.extend(pred_indices)
                base_true_indices.extend(target_indices.cpu().numpy())
                pbar_base.set_postfix(base_loss=f"{loss.item():.4f}")

        avg_base_loss = base_loss / max(1, n_base)
        base_metrics = evaluate_metrics(np.array(base_true_indices), np.array(base_pred_indices))
        run1_matched_summary = {
            "test_loss": round(avg_base_loss, 4),
            "total_test_samples": len(baseline_dataset),
            "checkpoint_path": str(resolved_baseline_ckpt),
            **base_metrics,
        }
    else:
        print(f"\n[!] Notice: Run 1 baseline checkpoint not found at checked paths. Skipping matched baseline evaluation.")

    # --------------------------------------------------------------------------
    # Tier 1: Canonical Run 1 Original Full-Test Result (Reference Only)
    # --------------------------------------------------------------------------
    run1_original_full_summary = {
        "description": "Original Run 1 RGB Baseline evaluated on all 8,040 canonical test images (Reference only; not directly comparable to subset)",
        "checkpoint": "/checkpoints/bcs_baseline/bcs_baseline_best.pth",
        "total_samples": canonical_test_count,
        "real_mae": 0.1848,
        "acc_0": 0.4163,
        "acc_1": 0.8674,
        "balanced_accuracy": 0.4041,
        "macro_f1": 0.4110,
    }

    # --------------------------------------------------------------------------
    # Comprehensive Comparison Compilation
    # --------------------------------------------------------------------------
    comparison_summary = {
        "evaluation_protocol": "Fair Matched-Subset Test Comparison",
        "valid_comparison_statement": (
            "The VALID direct comparison is between '2_run1_matched_perception_subset' and "
            "'3_run4_matched_perception_subset' evaluated on the EXACT SAME image identities where "
            "perception succeeded. The original 8,040-image Run 1 metric is reported for reference only "
            "and is NOT directly comparable to the smaller filtered perception subset."
        ),
        "test_split_integrity_statement": (
            "Canonical test labels and data were NOT used for training or checkpoint selection. "
            "A small test-subset plumbing evaluation was performed during pipeline verification. "
            "Final full Run 4 test evaluation remains post-training only. Test metrics never affect "
            "checkpoint or hyperparameter selection."
        ),
        "coverage_statistics": {
            "canonical_full_test_count": canonical_test_count,
            "manifest_test_rows": total_manifest_samples,
            "successful_perception_test_count": successful_perception_count,
            "excluded_detection_failures": n_det_fail,
            "excluded_sam_failures": n_sam_fail,
            "perception_coverage_percentage": coverage_pct,
        },
        "1_run1_original_full_test": run1_original_full_summary,
        "2_run1_matched_perception_subset": run1_matched_summary,
        "3_run4_matched_perception_subset": run4_test_summary,
    }

    delta_mae = None
    delta_acc1 = None
    delta_acc0 = None
    delta_balacc = None
    delta_f1 = None

    if run1_matched_summary is not None:
        delta_mae = round(test_metrics["real_mae"] - run1_matched_summary["real_mae"], 4)
        delta_acc1 = round((test_metrics["acc_1"] - run1_matched_summary["acc_1"]) * 100, 2)
        delta_acc0 = round((test_metrics["acc_0"] - run1_matched_summary["acc_0"]) * 100, 2)
        delta_balacc = round((test_metrics["balanced_accuracy"] - run1_matched_summary["balanced_accuracy"]) * 100, 2)
        delta_f1 = round(test_metrics["macro_f1"] - run1_matched_summary["macro_f1"], 4)

        comparison_summary["matched_delta_run4_minus_run1"] = {
            "real_mae_delta": delta_mae,
            "acc_1_delta_pct": delta_acc1,
            "acc_0_delta_pct": delta_acc0,
            "balanced_accuracy_delta_pct": delta_balacc,
            "macro_f1_delta": delta_f1,
        }

    comparison_json_file = output_dir / "bcs_perception_matched_test_comparison.json"
    with open(comparison_json_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Write Markdown comparison report
    comparison_md_file = output_dir / "bcs_perception_matched_test_comparison.md"
    with open(comparison_md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 3 BCS Fair Matched-Subset Test Comparison\n\n")
        f.write("> **Scientific Protocol Notice:** The only statistically and scientifically valid direct comparison\n")
        f.write("> is between **Run 1 matched subset** and **Run 4 matched subset** on the EXACT SAME image identities\n")
        f.write("> where perception succeeded. The original 8,040-image Run 1 metric is reported for reference only\n")
        f.write("> and is NOT directly comparable to the smaller filtered perception subset.\n\n")
        f.write("## 1. Test Population & Perception Coverage\n\n")
        f.write(f"- **Canonical Full Test Count**: {canonical_test_count:,} images\n")
        f.write(f"- **Evaluated Test Manifest Rows**: {total_manifest_samples:,} images\n")
        f.write(f"- **Successful Perception Test Count**: {successful_perception_count:,} images\n")
        f.write(f"- **Excluded Detection Failures**: {n_det_fail:,} images (RT-DETR found no cow)\n")
        f.write(f"- **Excluded SAM Failures**: {n_sam_fail:,} images (SAM returned no mask)\n")
        f.write(f"- **Perception Coverage**: {coverage_pct:.2f}%\n\n")
        f.write("## 2. Primary Performance Comparison Table\n\n")
        f.write("| Metric | (1) Run 1 Original Full Test (Ref Only, N=8,040) | (2) Run 1 Matched Subset | (3) Run 4 Matched Subset | Delta (Run 4 - Run 1 Matched) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

        if run1_matched_summary is not None:
            f.write(f"| **Real BCS MAE (Primary)** | **0.1848** | **{run1_matched_summary['real_mae']:.4f}** | **{test_metrics['real_mae']:.4f}** | **{delta_mae:+.4f} BCS units** |\n")
            f.write(f"| **Acc@1 (+/- 0.25 units)** | 86.74% | {run1_matched_summary['acc_1']*100:.2f}% | {test_metrics['acc_1']*100:.2f}% | {delta_acc1:+.2f}% |\n")
            f.write(f"| **Acc@0 (Exact match)** | 41.63% | {run1_matched_summary['acc_0']*100:.2f}% | {test_metrics['acc_0']*100:.2f}% | {delta_acc0:+.2f}% |\n")
            f.write(f"| **Balanced Accuracy** | 40.41% | {run1_matched_summary['balanced_accuracy']*100:.2f}% | {test_metrics['balanced_accuracy']*100:.2f}% | {delta_balacc:+.2f}% |\n")
            f.write(f"| **Macro-F1** | 0.4110 | {run1_matched_summary['macro_f1']:.4f} | {test_metrics['macro_f1']:.4f} | {delta_f1:+.4f} |\n")
        else:
            f.write(f"| **Real BCS MAE (Primary)** | **0.1848** | N/A | **{test_metrics['real_mae']:.4f}** | N/A |\n")
            f.write(f"| **Acc@1 (+/- 0.25 units)** | 86.74% | N/A | {test_metrics['acc_1']*100:.2f}% | N/A |\n")
            f.write(f"| **Acc@0 (Exact match)** | 41.63% | N/A | {test_metrics['acc_0']*100:.2f}% | N/A |\n")
            f.write(f"| **Balanced Accuracy** | 40.41% | N/A | {test_metrics['balanced_accuracy']*100:.2f}% | N/A |\n")
            f.write(f"| **Macro-F1** | 0.4110 | N/A | {test_metrics['macro_f1']:.4f} | N/A |\n")

        f.write("\n## 3. Test Split Integrity Statement\n\n")
        f.write("- Canonical test labels and data were NOT used for training or checkpoint selection.\n")
        f.write("- A small test-subset plumbing evaluation was performed during pipeline verification.\n")
        f.write("- Final full Run 4 test evaluation remains post-training only.\n")
        f.write("- Test metrics must never affect checkpoint or hyperparameter selection.\n")

    print("\n" + "=" * 75)
    print("  MATCHED TEST EVALUATION COMPLETED")
    print(f"  Coverage:                {coverage_pct:.2f}% ({successful_perception_count}/{total_manifest_samples})")
    print(f"  Run 4 Matched Real MAE:  {test_metrics['real_mae']:.4f} BCS units")
    print(f"  Run 4 Matched Acc@1:     {test_metrics['acc_1']*100:.2f}%")
    if run1_matched_summary is not None:
        print(f"  Run 1 Matched Real MAE:  {run1_matched_summary['real_mae']:.4f} BCS units")
        print(f"  Run 1 Matched Acc@1:     {run1_matched_summary['acc_1']*100:.2f}%")
        print(f"  Direct Matched MAE Delta:{delta_mae:+.4f} BCS units")
    print(f"  Saved comparison JSON:   {comparison_json_file}")
    print(f"  Saved comparison MD:     {comparison_md_file}")
    print("=" * 75)

    return comparison_summary


def pack_perception_cache(
    manifest_dir: Path,
    cache_dir: Path,
    splits: List[str] = ["train", "val", "test"],
    image_size: int = 224,
    num_workers: int = 16,
) -> Dict:
    """
    Pack raw perception crops and masks into compact, monolithic .pt files:
      {cache_dir}/packed/{split}_bcs_224.pt
    This bypasses all FUSE network roundtrips during training and enables instant 10s RAM loading.
    """
    import gc
    manifest_dir = Path(manifest_dir)
    cache_dir = Path(cache_dir)
    packed_dir = cache_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    print("\n" + "=" * 75)
    print("  SCIENTEDB PERCEPTION DATASET PACKING ENGINE")
    print(f"  Manifest Dir: {manifest_dir}")
    print(f"  Cache Dir:    {cache_dir}")
    print(f"  Packed Dir:   {packed_dir}")
    print(f"  Splits:       {splits}")
    print(f"  Workers:      {num_workers}")
    print("=" * 75)

    for split in splits:
        manifest_p = manifest_dir / f"{split}_perception.csv"
        if not manifest_p.exists():
            print(f"[!] Manifest not found for split '{split}' at {manifest_p}. Skipping.", flush=True)
            continue

        out_path = packed_dir / f"{split}_bcs_224.pt"
        if out_path.exists():
            print(f"[*] Packed file for '{split}' already exists at {out_path}. Skipping.", flush=True)
            try:
                payload = torch.load(out_path, map_location="cpu", weights_only=False)
                results[split] = {"status": "already_exists", "samples": len(payload["tensors"]), "path": str(out_path)}
            except Exception:
                pass
            continue

        df = pd.read_csv(manifest_p)
        valid_mask = (df["detection_status"] == "detected") & (df["sam_status"] == "segmented")
        df_valid = df[valid_mask].copy().reset_index(drop=True)
        N = len(df_valid)
        print(f"\n[*] Processing split '{split}': {N} valid samples (from {len(df)} canonical rows)...", flush=True)

        samples = []
        for idx, row in df_valid.iterrows():
            crop_p = cache_dir / str(row["crop_rel_path"])
            mask_p = cache_dir / str(row["mask_rel_path"])
            raw_label = float(row["label"])
            target_idx = LABEL_TO_IDX[raw_label]
            samples.append((str(crop_p), str(mask_p), target_idx, raw_label, str(row["image_path"])))

        cached_data = np.empty((N, 4, image_size, image_size), dtype=np.uint8)

        def _load_one(idx: int):
            crop_path, mask_path, _, _, _ = samples[idx]
            crop_bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
            if crop_bgr is None:
                raise IOError(f"Failed to read crop at {crop_path}")
            if crop_bgr.shape[0] != image_size or crop_bgr.shape[1] != image_size:
                crop_224 = cv2.resize(crop_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            else:
                crop_224 = crop_bgr
            rgb_224 = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)

            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise IOError(f"Failed to read mask at {mask_path}")
            if mask_gray.shape[0] != image_size or mask_gray.shape[1] != image_size:
                mask_224 = cv2.resize(mask_gray, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            else:
                mask_224 = mask_gray
            mask_bin = (mask_224 > 127).astype(np.uint8)

            cached_data[idx, :3] = rgb_224.transpose(2, 0, 1)
            cached_data[idx, 3] = mask_bin

        t0 = time.time()
        workers = max(1, min(num_workers, 16))
        log_interval = max(500, N // 20)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_load_one, i) for i in range(N)]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                fut.result()
                if i % log_interval == 0 or i == N:
                    elapsed = time.time() - t0
                    rate = i / max(0.1, elapsed)
                    print(f"  [Packing {split}] {i:,} / {N:,} ({i/N*100:.1f}%) | {elapsed:.1f}s | {rate:.1f} img/s", flush=True)

        tensors = torch.from_numpy(cached_data)
        targets = torch.tensor([s[2] for s in samples], dtype=torch.long)
        raw_labels = torch.tensor([s[3] for s in samples], dtype=torch.float32)

        payload = {
            "tensors": tensors,
            "targets": targets,
            "raw_labels": raw_labels,
            "split": split,
            "num_samples": N,
        }
        print(f"[*] Saving packed {split} tensor to {out_path}...", flush=True)
        torch.save(payload, out_path)
        elapsed_total = time.time() - t0
        mb = out_path.stat().st_size / (1024 * 1024)
        print(f"✓ Packed {split}: {N} samples ({mb:.1f} MB) in {elapsed_total:.1f}s -> {out_path}", flush=True)

        results[split] = {"status": "packed", "samples": N, "size_mb": round(mb, 1), "elapsed_s": round(elapsed_total, 1), "path": str(out_path)}
        del cached_data, tensors, targets, raw_labels, payload
        gc.collect()

    print("\n" + "=" * 75)
    print("  PACKING COMPLETE FOR ALL REQUESTED SPLITS")
    print("=" * 75)
    return results


def train_pipeline(
    manifest_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    baseline_ckpt_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    mask_init: str = "mean",
    smoke: bool = False,
    max_samples: Optional[int] = None,
    eval_test: bool = False,
    device_name: Optional[str] = None,
    preload_ram: bool = True,
    num_preload_workers: int = 64,
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
    print(f"  Preload RAM:    {preload_ram} (workers: {num_preload_workers})")
    print(f"  Test Eval:      {'YES (post-training one-time)' if (eval_test and not smoke) else 'NO (smoke / train-val only)'}")
    print("=" * 75)

    train_manifest = manifest_dir / "train_perception.csv"
    val_manifest = manifest_dir / "val_perception.csv"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(f"Missing perception manifests in {manifest_dir}")

    n_samples = 10 if smoke else max_samples
    train_dataset = ScienceDBPerceptionDataset(
        manifest_path=train_manifest,
        cache_dir=cache_dir,
        is_train=True,
        max_samples=n_samples,
        preload_ram=preload_ram,
        num_preload_workers=num_preload_workers,
    )
    val_dataset = ScienceDBPerceptionDataset(
        manifest_path=val_manifest,
        cache_dir=cache_dir,
        is_train=False,
        max_samples=n_samples,
        preload_ram=preload_ram,
        num_preload_workers=num_preload_workers,
    )

    num_workers = 0 if preload_ram else (0 if smoke or sys.platform == "win32" else 8)
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
            imgs = imgs.to(device, non_blocking=True)
            target_indices = target_indices.to(device, non_blocking=True)
            if imgs.dtype == torch.uint8:
                imgs = apply_gpu_augmentations(imgs, is_train=True)
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
                imgs = imgs.to(device, non_blocking=True)
                target_indices = target_indices.to(device, non_blocking=True)
                if imgs.dtype == torch.uint8:
                    imgs = apply_gpu_augmentations(imgs, is_train=False)
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

    # Post-training: evaluate held-out test split with best checkpoint if requested
    if eval_test and not smoke:
        test_manifest = manifest_dir / "test_perception.csv"
        if test_manifest.exists():
            print(f"\n[*] Loading best checkpoint from {best_ckpt_path} for post-training test evaluation...")
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_results = evaluate_test_split(
                model=model,
                manifest_dir=manifest_dir,
                cache_dir=cache_dir,
                output_dir=output_dir,
                baseline_ckpt_path=baseline_ckpt_path,
                data_dir=data_dir,
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
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="Path to Run 1 baseline checkpoint for fair matched comparison")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing original ScienceDB images (for Run 1 matched eval)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask-init", type=str, default="mean", choices=["mean", "zero"], help="Initialization of 4th mask channel in conv1")
    parser.add_argument("--smoke", action="store_true", help="Smoke test on small subset")
    parser.add_argument("--no-preload-ram", action="store_true", help="Disable RAM preloading and read directly from disk")
    parser.add_argument("--eval-test", action="store_true", help="Run test evaluation after training completes")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_pipeline(
        manifest_dir=Path(args.manifest_dir),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        baseline_ckpt_path=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_init=args.mask_init,
        smoke=args.smoke,
        eval_test=args.eval_test,
        device_name=args.device,
        preload_ram=not args.no_preload_ram,
    )


if __name__ == "__main__":
    main()
