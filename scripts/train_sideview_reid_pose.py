# -*- coding: utf-8 -*-
"""SideViewCows2026 Re-ID + Pose Ablation Training Pipeline.

Scientific Ablation:
  Tests whether pretrained quadruped anatomical pose information adds discriminative
  value over the Run 6 GT-mask perception baseline:

  Run 6:               4-channel ResNet-18 [R, G, B, Mask] (512-D) -> Linear(512, 41)
  Run 6 + Pose (New):  4-channel ResNet-18 (512-D) + Pose MLP (156 -> 64-D) -> Fused(576-D) -> Linear(576, 41)

Representations:
  - Visual: identical to Run 6 (GT-mask cow crop + 5% margin -> 224x224 -> [R, G, B, Mask]).
  - Pose: 39 normalized keypoints (x_norm, y_norm, conf, is_valid) normalized to [0, 1] relative
    to the crop box. Missing keypoints/failed detections explicitly represented with is_valid=0.
  - Zero metadata leakage: no cow IDs, camera IDs, filenames, recording IDs, or subset IDs.

Protocols (Strict Isolation):
  - Training/Val: 41 Protocol-A representation-learning cows (Protocol D: 12,753 train / 2,683 val).
  - Held-out Evaluation: 69 Protocol-A cows (Parlor gallery, Barn queries, Snapshot queries).
    Strictly isolated and NEVER loaded during training or smoke testing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from scripts.train_sideview_reid_baseline import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        evaluate_retrieval_chunked,
        get_git_commit_sha,
        resolve_sideview_image_path,
    )
    from scripts.train_sideview_reid_perception import (
        ResNet18ReIDPerception,
        _mask_bbox_with_margin,
        _validate_pair_paths,
    )
except ModuleNotFoundError:
    from train_sideview_reid_baseline import (  # type: ignore
        IMAGENET_MEAN,
        IMAGENET_STD,
        evaluate_retrieval_chunked,
        get_git_commit_sha,
        resolve_sideview_image_path,
    )
    from train_sideview_reid_perception import (  # type: ignore
        ResNet18ReIDPerception,
        _mask_bbox_with_margin,
        _validate_pair_paths,
    )

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST
NUM_SUPERANIMAL_KEYPOINTS = 39
POSE_FEATURE_DIM = NUM_SUPERANIMAL_KEYPOINTS * 4  # 156 (x, y, conf, is_valid)
POSE_EMBEDDING_DIM = 64
FUSED_EMBEDDING_DIM = 512 + POSE_EMBEDDING_DIM  # 576


# ==============================================================================
# 1. POSE FEATURE EXTRACTOR HELPER
# ==============================================================================
class SuperAnimalPoseFeatureExtractor:
    """Extracts normalized 156-D pose vectors using frozen SuperAnimal ResNet-50."""

    def __init__(self, device: str = "cuda"):
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
            create_superanimal_inference_runners,
        )

        self.device = device
        self.pose_runner, self.det_runner, model_cfg = create_superanimal_inference_runners(
            superanimal_name="superanimal_quadruped",
            model_name="resnet_50",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            max_individuals=1,
            batch_size=1,
            detector_batch_size=1,
            device=device,
        )
        self.bodyparts = model_cfg["metadata"]["bodyparts"]

    def extract_crop_pose_feature(self, crop_rgb_np: np.ndarray) -> np.ndarray:
        """Extracts normalized 156-D float32 vector for a single cow crop.

        Format for each keypoint: [x_norm, y_norm, confidence, is_valid]
        Coordinates are normalized to [0, 1] relative to (crop_w, crop_h).
        If detector fails or keypoints are missing: returns 0.0 with is_valid=0.0.
        """
        crop_h, crop_w = crop_rgb_np.shape[:2]
        feat = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)

        try:
            det_preds = self.det_runner.inference([crop_rgb_np]) if self.det_runner is not None else None
            has_box = False
            if det_preds is not None and len(det_preds) > 0:
                bboxes = det_preds[0].get("bboxes", [])
                scores = det_preds[0].get("bbox_scores", [])
                if len(bboxes) > 0 and (len(scores) == 0 or np.max(scores) > 0.0):
                    has_box = True

            if not has_box and self.det_runner is not None:
                return feat  # is_valid = 0.0 for all points

            pose_inputs = [(crop_rgb_np, det_preds[0])] if det_preds is not None else [crop_rgb_np]
            pose_preds = self.pose_runner.inference(pose_inputs)
            if not pose_preds:
                return feat

            pred = pose_preds[0]
            raw_bpts = pred.get("bodyparts", None)
            if raw_bpts is not None:
                bpts_arr = np.asarray(raw_bpts)
                if bpts_arr.ndim == 3:
                    bpts_arr = bpts_arr[0]
                for k_idx in range(min(len(self.bodyparts), len(bpts_arr))):
                    kx = float(bpts_arr[k_idx, 0])
                    ky = float(bpts_arr[k_idx, 1])
                    kc = float(bpts_arr[k_idx, 2])
                    x_norm = max(0.0, min(1.0, kx / max(crop_w, 1)))
                    y_norm = max(0.0, min(1.0, ky / max(crop_h, 1)))
                    base_idx = k_idx * 4
                    feat[base_idx : base_idx + 4] = [x_norm, y_norm, kc, 1.0]
            else:
                coords = pred.get("coordinates", [])
                conf_arr = pred.get("confidence", [])
                if len(coords) > 0 and len(conf_arr) > 0:
                    c_xy = coords[0] if len(coords.shape) == 3 else coords
                    c_conf = conf_arr[0] if len(conf_arr.shape) == 2 else conf_arr
                    for k_idx in range(min(len(self.bodyparts), len(c_xy))):
                        kx = float(c_xy[k_idx][0])
                        ky = float(c_xy[k_idx][1])
                        kc = float(c_conf[k_idx])
                        x_norm = max(0.0, min(1.0, kx / max(crop_w, 1)))
                        y_norm = max(0.0, min(1.0, ky / max(crop_h, 1)))
                        base_idx = k_idx * 4
                        feat[base_idx : base_idx + 4] = [x_norm, y_norm, kc, 1.0]

        except Exception as e:
            pass

        return feat


# ==============================================================================
# 2. DATASET: RUN 6 CROP + POSE EMBEDDING VECTOR
# ==============================================================================
class SideViewReIDPoseDataset(Dataset):
    """SideView Re-ID dataset yielding 4-channel image tensor + 156-D pose feature vector."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        cow_to_label: Dict[str, int],
        augment: bool,
        pose_dict: Optional[Dict[str, np.ndarray]] = None,
        pose_extractor: Optional[SuperAnimalPoseFeatureExtractor] = None,
        margin_fraction: float = 0.05,
        image_size: int = 224,
    ) -> None:
        required = {"image_path", "mask_path", "individual_id"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.cow_to_label = cow_to_label
        self.augment = augment
        self.pose_dict = pose_dict or {}
        self.pose_extractor = pose_extractor
        self.margin_fraction = margin_fraction
        self.image_size = image_size
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

        self.records: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            cow_id = str(row["individual_id"])
            if cow_id not in cow_to_label:
                raise ValueError(f"Cow {cow_id} not in cow_to_label mapping")
            raw_img = str(row["image_path"])
            raw_mask = str(row["mask_path"])
            self.records.append({
                "raw_image_path": raw_img,
                "raw_mask_path": raw_mask,
                "resolved_image_path": resolve_sideview_image_path(raw_img, self.data_root),
                "resolved_mask_path": resolve_sideview_image_path(raw_mask, self.data_root),
                "cow_id": cow_id,
                "label": cow_to_label[cow_id],
                "subset": str(row.get("subset", "")),
                "recording_id": str(row.get("recording_id", "")),
            })

    def __len__(self) -> int:
        return len(self.records)

    def _load_source(self, idx: int) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
        rec = self.records[idx]
        img_p = rec["resolved_image_path"] or resolve_sideview_image_path(rec["raw_image_path"], self.data_root)
        mask_p = rec["resolved_mask_path"] or resolve_sideview_image_path(rec["raw_mask_path"], self.data_root)
        if img_p is None or mask_p is None:
            raise FileNotFoundError(f"Missing image or mask: {rec['raw_image_path']}")

        _validate_pair_paths(img_p, mask_p, rec["cow_id"])
        with Image.open(img_p) as im:
            rgb_im = im.convert("RGB")
        with Image.open(mask_p) as mk:
            mask_gray = mk.convert("L")

        mask_np = np.asarray(mask_gray) > 0
        crop_box = _mask_bbox_with_margin(mask_np, self.margin_fraction)
        img_crop = rgb_im.crop(crop_box)
        mask_crop = mask_gray.crop(crop_box)

        metadata = {
            "cow_id": rec["cow_id"],
            "raw_image_path": rec["raw_image_path"],
            "crop_box": crop_box,
            "crop_size": img_crop.size,
        }
        return img_crop, mask_crop, metadata

    def _to_tensor(self, img_crop: Image.Image, mask_crop: Image.Image, apply_aug: bool) -> torch.Tensor:
        img_resized = img_crop.resize((self.image_size, self.image_size), resample=RESAMPLE_BILINEAR)
        mask_resized = mask_crop.resize((self.image_size, self.image_size), resample=RESAMPLE_NEAREST)

        # Do NOT apply horizontal flip to pose-guided Re-ID to preserve left/right limb asymmetry
        if apply_aug:
            img_resized = self.color_jitter(img_resized)

        rgb_tensor = TF.to_tensor(img_resized)
        rgb_tensor = TF.normalize(rgb_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask_np = (np.asarray(mask_resized) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return torch.cat([rgb_tensor, mask_tensor], dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        rec = self.records[idx]
        img_crop, mask_crop, meta = self._load_source(idx)
        img_tensor = self._to_tensor(img_crop, mask_crop, apply_aug=self.augment)

        # Retrieve pose feature vector
        raw_path = rec["raw_image_path"]
        if raw_path in self.pose_dict:
            pose_vec = self.pose_dict[raw_path]
        elif self.pose_extractor is not None:
            pose_vec = self.pose_extractor.extract_crop_pose_feature(np.asarray(img_crop))
            self.pose_dict[raw_path] = pose_vec
        else:
            pose_vec = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)

        pose_tensor = torch.from_numpy(pose_vec.copy()).to(torch.float32)
        return img_tensor, pose_tensor, rec["label"], rec["cow_id"]


# ==============================================================================
# 3. ARCHITECTURE: RESNET-18 (512-D) + POSE MLP (64-D) -> FUSED (576-D)
# ==============================================================================
class PoseMLP(nn.Module):
    """Transforms 156-D normalized keypoint coordinates & confidences into a 64-D embedding."""

    def __init__(self, in_features: int = POSE_FEATURE_DIM, out_features: int = POSE_EMBEDDING_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18ReIDPoseAblation(nn.Module):
    """Run 6 Perception Spatial Trunk fused with Pose MLP embedding."""

    def __init__(
        self,
        num_classes: int = 41,
        pose_in_features: int = POSE_FEATURE_DIM,
        pose_emb_dim: int = POSE_EMBEDDING_DIM,
        pretrained: bool = True,
    ):
        super().__init__()
        # Visual spatial trunk matching Run 6 exactly
        self.visual_backbone = ResNet18ReIDPerception(num_classes=num_classes, pretrained=pretrained)
        self.visual_backbone.classifier = nn.Identity()  # Remove unused Run 6 linear classifier
        self.visual_dim = 512
        self.pose_dim = pose_emb_dim
        self.fused_dim = self.visual_dim + self.pose_dim  # 512 + 64 = 576

        # Pose branch
        self.pose_mlp = PoseMLP(in_features=pose_in_features, out_features=pose_emb_dim)

        # Fused classifier
        self.classifier = nn.Linear(self.fused_dim, num_classes)

    def extract_visual_feature(self, visual_input: torch.Tensor) -> torch.Tensor:
        x = self.visual_backbone.conv1(visual_input)
        x = self.visual_backbone.bn1(x)
        x = self.visual_backbone.relu(x)
        x = self.visual_backbone.maxpool(x)

        x = self.visual_backbone.layer1(x)
        x = self.visual_backbone.layer2(x)
        x = self.visual_backbone.layer3(x)
        x = self.visual_backbone.layer4(x)

        x = self.visual_backbone.avgpool(x)
        return torch.flatten(x, 1)  # [B, 512]

    def forward(
        self, visual_input: torch.Tensor, pose_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            visual_input: [B, 4, 224, 224] (RGB + GT binary mask)
            pose_input:   [B, 156] (39 normalized keypoints * (x_norm, y_norm, conf, is_valid))

        Returns:
            logits:         [B, 41] classification logits
            norm_embedding: [B, 576] unit L2-normalized embedding for retrieval
        """
        f_vis = self.extract_visual_feature(visual_input)   # [B, 512]
        f_pose = self.pose_mlp(pose_input)                 # [B, 64]

        f_fused = torch.cat([f_vis, f_pose], dim=1)        # [B, 576]
        norm_embedding = F.normalize(f_fused, p=2, dim=1)  # [B, 576]
        logits = self.classifier(f_fused)                  # [B, 41]

        return logits, norm_embedding


def get_parameter_counts() -> Dict[str, int]:
    model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False)
    vis_params = sum(p.numel() for p in model.visual_backbone.parameters() if p.requires_grad)
    pose_params = sum(p.numel() for p in model.pose_mlp.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "visual_backbone_trainable_parameters": vis_params,
        "pose_mlp_trainable_parameters": pose_params,
        "classifier_trainable_parameters": cls_params,
        "total_trainable_parameters": total_params,
    }


# ==============================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
@torch.no_grad()
def evaluate_validation_pose(
    model: ResNet18ReIDPoseAblation,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    preds_all: List[int] = []
    targets_all: List[int] = []

    for vis_x, pose_x, targets, _ in loader:
        vis_x = vis_x.to(device)
        pose_x = pose_x.to(device)
        targets = targets.to(device)

        logits, _ = model(vis_x, pose_x)
        total_loss += criterion(logits, targets).item() * vis_x.size(0)
        preds_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

    count = len(loader.dataset)
    return {
        "val_loss": round(total_loss / count, 4),
        "val_top1_acc": round(accuracy_score(targets_all, preds_all) * 100, 2),
        "val_bal_acc": round(balanced_accuracy_score(targets_all, preds_all) * 100, 2),
        "val_macro_f1": round(f1_score(targets_all, preds_all, average="macro", zero_division=0), 4),
    }


def precompute_pose_features_subset(
    df: pd.DataFrame,
    data_root: Path,
    device_str: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Pre-extracts 156-D pose vectors for a subset of samples to avoid GPU contention."""
    extractor = SuperAnimalPoseFeatureExtractor(device=device_str)
    pose_dict: Dict[str, np.ndarray] = {}

    print(f"[*] Pre-computing SuperAnimal pose features for {len(df)} samples...")
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing Pose"):
        raw_img = str(row["image_path"])
        raw_mask = str(row["mask_path"])
        cow_id = str(row["individual_id"])

        img_p = resolve_sideview_image_path(raw_img, data_root)
        mask_p = resolve_sideview_image_path(raw_mask, data_root)
        if img_p is None or mask_p is None:
            pose_dict[raw_img] = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
            continue

        with Image.open(img_p) as im:
            rgb_im = im.convert("RGB")
        with Image.open(mask_p) as mk:
            mask_gray = mk.convert("L")

        mask_np = np.asarray(mask_gray) > 0
        crop_box = _mask_bbox_with_margin(mask_np, margin_fraction=0.05)
        crop_rgb_np = np.asarray(rgb_im.crop(crop_box))

        vec = extractor.extract_crop_pose_feature(crop_rgb_np)
        pose_dict[raw_img] = vec

    print(f"[OK] Precomputed {len(pose_dict)} pose features in {time.time() - t0:.2f}s")
    # Clean up extractor memory
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pose_dict


def train_sideview_reid_pose(
    data_root: Path,
    protocols_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    seed: int = 2026,
    smoke: bool = False,
    smoke_samples: int = 64,
    pose_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("SIDEVIEWCOWS2026 RE-ID + SUPERANIMAL POSE ABLATION TRAINING")
    print(f"Mode: {'SMOKE TEST (TRAIN/VAL ONLY)' if smoke else 'FULL ABLATION TRAINING'}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Seed: {seed}")
    print("Architecture: 4-Channel ResNet-18 (512-D) + Pose MLP (156->64) -> Fused (576-D)")
    print("=" * 76)

    # 1. Protocols & Disjointness Enforcement
    proto_a_path = protocols_dir / "protocol_cross_setting.csv"
    proto_d_path = protocols_dir / "protocol_closed_set.csv"
    df_a = pd.read_csv(proto_a_path)
    df_d = pd.read_csv(proto_d_path)

    train_cows = sorted(df_a[df_a["setting_role"] == "train"]["individual_id"].astype(str).unique())
    eval_cows = sorted(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique())

    if len(train_cows) != 41 or len(eval_cows) != 69:
        raise AssertionError("Cow counts mismatch in Protocol A")
    if set(train_cows).intersection(set(eval_cows)):
        raise AssertionError("Train/Eval cow identity overlap detected!")

    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train_full = df_d_41[df_d_41["closed_set_split"] == "train"].reset_index(drop=True)
    df_val_full = df_d_41[df_d_41["closed_set_split"] == "val"].reset_index(drop=True)

    if smoke:
        df_train = df_train_full.sample(n=min(smoke_samples, len(df_train_full)), random_state=seed).reset_index(drop=True)
        df_val = df_val_full.sample(n=min(smoke_samples, len(df_val_full)), random_state=seed).reset_index(drop=True)
    else:
        df_train = df_train_full
        df_val = df_val_full

    print(f"[OK] Training cows: {len(train_cows)} | Train samples: {len(df_train)} | Val samples: {len(df_val)}")
    cow_to_label = {c: i for i, c in enumerate(train_cows)}

    # 2. Pose Feature Preparation
    pose_dict: Dict[str, np.ndarray] = {}
    if pose_cache_path is not None and pose_cache_path.exists():
        print(f"[*] Loading precomputed pose cache from: {pose_cache_path}")
        pose_dict = torch.load(pose_cache_path, map_location="cpu", weights_only=False)
        print(f"[OK] Loaded {len(pose_dict)} cached pose features.")
    else:
        # Precompute pose features for the active train + val subset
        combined_df = pd.concat([df_train, df_val], ignore_index=True)
        pose_dict = precompute_pose_features_subset(
            combined_df, data_root=data_root, device_str=str(device)
        )

    # 3. Datasets and Loaders
    train_dataset = SideViewReIDPoseDataset(
        df=df_train,
        data_root=data_root,
        cow_to_label=cow_to_label,
        augment=True,
        pose_dict=pose_dict,
    )
    val_dataset = SideViewReIDPoseDataset(
        df=df_val,
        data_root=data_root,
        cow_to_label=cow_to_label,
        augment=False,
        pose_dict=pose_dict,
    )

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

    # 4. Model, Optimizer, Loss
    model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=True).to(device)
    param_counts = get_parameter_counts()
    print(f"[*] Trainable Parameters: {param_counts['total_trainable_parameters']:,}")
    print(f"    - Visual Spatial Trunk: {param_counts['visual_backbone_trainable_parameters']:,}")
    print(f"    - Pose MLP Branch:      {param_counts['pose_mlp_trainable_parameters']:,}")
    print(f"    - Fused Classifier:     {param_counts['classifier_trainable_parameters']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    history = []
    best_val_acc = -1.0
    best_checkpoint_path = output_dir / ("reid_pose_smoke_best.pth" if smoke else "reid_pose_best.pth")

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_preds: List[int] = []
        train_targets: List[int] = []

        for vis_x, pose_x, targets, _ in train_loader:
            vis_x = vis_x.to(device)
            pose_x = pose_x.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(vis_x, pose_x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * vis_x.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            train_targets.extend(targets.cpu().tolist())

        scheduler.step()
        train_loss_avg = train_loss / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds) * 100

        val_metrics = evaluate_validation_pose(model, val_loader, criterion, device)
        val_acc = val_metrics["val_top1_acc"]
        ep_dur = time.time() - ep_t0

        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss_avg, 4),
            "train_top1_acc": round(train_acc, 2),
            **val_metrics,
            "duration_s": round(ep_dur, 2),
        }
        history.append(ep_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_dur:.1f}s] "
            f"Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} Acc: {val_acc:.2f}% F1: {val_metrics['val_macro_f1']:.4f}"
        )

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "param_counts": param_counts,
                    "cow_to_label": cow_to_label,
                    "seed": seed,
                },
                best_checkpoint_path,
            )

    # 6. Checkpoint Determinism & Reload Verification
    print("\n[*] Verifying Checkpoint Reload Determinism...")
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    fresh_model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False).to(device)
    fresh_model.load_state_dict(ckpt["model_state_dict"])
    fresh_model.eval()
    model.eval()

    sample_vis, sample_pose, _, _ = next(iter(val_loader))
    sample_vis = sample_vis.to(device)
    sample_pose = sample_pose.to(device)

    with torch.no_grad():
        out_orig, emb_orig = model(sample_vis, sample_pose)
        out_fresh, emb_fresh = fresh_model(sample_vis, sample_pose)
        diff_logits = (out_orig - out_fresh).abs().max().item()
        diff_embs = (emb_orig - emb_fresh).abs().max().item()

    print(f"[OK] Bit-Identical Reload Verification: Max Logit Diff = {diff_logits:.8f}, Max Emb Diff = {diff_embs:.8f}")
    if diff_logits > 1e-6:
        raise AssertionError(f"Checkpoint reload produced divergent outputs: {diff_logits}")

    # Verify unit-L2 norm
    l2_norms = torch.norm(emb_orig, p=2, dim=1).cpu().numpy()
    print(f"[OK] Embedding L2 Norm Min: {l2_norms.min():.6f}, Max: {l2_norms.max():.6f} (Expected: 1.000000)")

    metrics_out = {
        "status": "SMOKE_CERTIFIED" if smoke else "TRAINING_COMPLETE",
        "condition": "SideViewCows2026 GT-Mask + SuperAnimal Pose (ResNet-50)",
        "epochs": epochs,
        "best_epoch": ckpt["epoch"],
        "best_val_metrics": ckpt["val_metrics"],
        "trainable_parameters": param_counts,
        "checkpoint_reload_max_logit_diff": diff_logits,
        "checkpoint_reload_max_emb_diff": diff_embs,
        "test_protocol_a_evaluated": False,  # Held-out cows strictly untouched
        "history": history,
        "best_checkpoint_path": str(best_checkpoint_path),
        "duration_total_s": round(time.perf_counter() - t_start, 2),
    }

    metrics_json_path = output_dir / ("reid_pose_smoke_metrics.json" if smoke else "reid_pose_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[OK] Saved metrics to {metrics_json_path}")

    return metrics_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SideView Re-ID + Pose Ablation")
    parser.add_argument("--data-root", type=Path, default=Path("/data/sideviewcows2026"))
    parser.add_argument("--protocols-dir", type=Path, default=REPO_ROOT / "datasets" / "id" / "sideviewcows2026")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "reid_pose_ablation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    args = parser.parse_args()

    train_sideview_reid_pose(
        data_root=args.data_root,
        protocols_dir=args.protocols_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoke=args.smoke,
        smoke_samples=args.smoke_samples,
    )
