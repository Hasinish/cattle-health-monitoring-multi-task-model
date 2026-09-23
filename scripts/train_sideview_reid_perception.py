# -*- coding: utf-8 -*-
"""Run 6 SideViewCows2026 GT-mask perception-enhanced Re-ID.

This is the deadline oracle-segmentation condition.  Every input is built from
the dataset-provided target-cow mask:

    RGB + matching GT mask -> mask-derived box + 5% margin -> aligned crop
    -> 224x224 -> [R, G, B, binary mask]

It deliberately does not run a detector, SAM, pose, viewpoint, attention, or
temporal modeling.  The model and optimization remain matched to the Run 3 RGB
baseline except for ResNet-18 conv1 accepting a fourth input channel.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

try:
    from scripts.train_sideview_reid_baseline import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        ResNet18ReIDBaseline,
        evaluate_retrieval_chunked,
        extract_dataset_embeddings,
        get_git_commit_sha,
        resolve_sideview_image_path,
    )
except ModuleNotFoundError:  # Direct invocation: python scripts/<file>.py
    from train_sideview_reid_baseline import (  # type: ignore
        IMAGENET_MEAN,
        IMAGENET_STD,
        ResNet18ReIDBaseline,
        evaluate_retrieval_chunked,
        extract_dataset_embeddings,
        get_git_commit_sha,
        resolve_sideview_image_path,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def _validate_pair_paths(image_path: Path, mask_path: Path, cow_id: str) -> None:
    """Reject a protocol row whose RGB/mask provenance does not align."""
    if image_path.stem != mask_path.stem:
        raise ValueError(
            f"RGB/mask stem mismatch for cow {cow_id}: {image_path.name} vs {mask_path.name}"
        )
    if image_path.parent.name != mask_path.parent.name:
        raise ValueError(
            f"RGB/mask cow-directory mismatch: {image_path.parent} vs {mask_path.parent}"
        )
    if image_path.parent.name != str(cow_id):
        raise ValueError(
            f"Protocol cow {cow_id} disagrees with pair directory {image_path.parent.name}"
        )


def _mask_bbox_with_margin(mask: np.ndarray, margin_fraction: float) -> Tuple[int, int, int, int]:
    """Return an in-bounds, PIL-style exclusive crop box from a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    height, width = mask.shape
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        raise ValueError("GT mask is empty; refusing to fabricate a crop")
    if foreground.shape[0] == height * width:
        raise ValueError("GT mask is full-one; refusing an uninformative mask/crop")

    y1, x1 = foreground.min(axis=0).tolist()
    y2_inc, x2_inc = foreground.max(axis=0).tolist()
    x2 = int(x2_inc) + 1
    y2 = int(y2_inc) + 1
    x1 = int(x1)
    y1 = int(y1)

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Non-positive target-mask box: {(x1, y1, x2, y2)}")

    margin_x = int(math.ceil(box_width * margin_fraction))
    margin_y = int(math.ceil(box_height * margin_fraction))
    crop_box = (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )
    cx1, cy1, cx2, cy2 = crop_box
    if not (0 <= cx1 < cx2 <= width and 0 <= cy1 < cy2 <= height):
        raise ValueError(f"Out-of-bounds/degenerate crop {crop_box} for {(width, height)}")
    return crop_box


class SideViewGTMaskReIDDataset(Dataset):
    """SideView target-cow crop plus explicit GT binary-mask channel."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        cow_to_label: Dict[str, int],
        augment: bool,
        margin_fraction: float = 0.05,
        image_size: int = 224,
    ) -> None:
        required = {"image_path", "mask_path", "individual_id"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Protocol missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.cow_to_label = cow_to_label
        self.augment = augment
        self.margin_fraction = margin_fraction
        self.image_size = image_size
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1
        )
        self.records: List[Dict[str, Any]] = []

        for _, row in self.df.iterrows():
            cow_id = str(row["individual_id"])
            if cow_id not in cow_to_label:
                raise ValueError(f"Cow {cow_id} is outside the 41-cow training mapping")
            raw_image_path = str(row["image_path"])
            raw_mask_path = str(row["mask_path"])
            self.records.append(
                {
                    "raw_image_path": raw_image_path,
                    "raw_mask_path": raw_mask_path,
                    "resolved_image_path": resolve_sideview_image_path(raw_image_path, self.data_root),
                    "resolved_mask_path": resolve_sideview_image_path(raw_mask_path, self.data_root),
                    "cow_id": cow_id,
                    "label": cow_to_label[cow_id],
                    "subset": str(row.get("subset", "")),
                    "recording_id": str(row.get("recording_id", "")),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def _load_source(
        self, idx: int
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Dict[str, Any]]:
        rec = self.records[idx]
        image_path = rec["resolved_image_path"]
        mask_path = rec["resolved_mask_path"]
        if image_path is None:
            image_path = resolve_sideview_image_path(rec["raw_image_path"], self.data_root)
        if mask_path is None:
            mask_path = resolve_sideview_image_path(rec["raw_mask_path"], self.data_root)
        if image_path is None:
            raise FileNotFoundError(f"Missing RGB image: {rec['raw_image_path']}")
        if mask_path is None:
            raise FileNotFoundError(f"Missing GT mask: {rec['raw_mask_path']}")

        _validate_pair_paths(image_path, mask_path, rec["cow_id"])
        try:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            with Image.open(mask_path) as handle:
                mask_gray = handle.convert("L")
        except Exception as exc:
            raise RuntimeError(f"Unreadable RGB/mask pair {image_path} | {mask_path}: {exc}") from exc

        if image.size != mask_gray.size:
            raise ValueError(
                f"RGB/mask size mismatch for {image_path.name}: {image.size} vs {mask_gray.size}"
            )

        mask_np = np.asarray(mask_gray) > 0
        crop_box = _mask_bbox_with_margin(mask_np, self.margin_fraction)
        binary_mask = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")
        image_crop = image.crop(crop_box)
        mask_crop = binary_mask.crop(crop_box)
        if image_crop.size != mask_crop.size:
            raise AssertionError("RGB and GT mask were not cropped with identical coordinates")

        x1, y1, x2, y2 = crop_box
        metadata = {
            "cow_id": rec["cow_id"],
            "recording_id": rec["recording_id"],
            "subset": rec["subset"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size_wh": [image.width, image.height],
            "crop_box_xyxy_exclusive": [x1, y1, x2, y2],
            "crop_size_wh_before_resize": [x2 - x1, y2 - y1],
            "foreground_pixels": int(mask_np.sum()),
            "foreground_fraction": float(mask_np.mean()),
        }
        return image, binary_mask, image_crop, mask_crop, metadata

    def _to_tensor(
        self, image_crop: Image.Image, mask_crop: Image.Image, apply_augmentation: bool
    ) -> torch.Tensor:
        image_crop = image_crop.resize(
            (self.image_size, self.image_size), resample=RESAMPLE_BILINEAR
        )
        mask_crop = mask_crop.resize(
            (self.image_size, self.image_size), resample=RESAMPLE_NEAREST
        )

        if apply_augmentation and random.random() < 0.5:
            image_crop = TF.hflip(image_crop)
            mask_crop = TF.hflip(mask_crop)
        if apply_augmentation:
            image_crop = self.color_jitter(image_crop)  # RGB only; never alter mask colors.

        rgb_tensor = TF.to_tensor(image_crop)
        rgb_tensor = TF.normalize(rgb_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_np = np.asarray(mask_crop) > 0
        mask_tensor = torch.from_numpy(mask_np.copy()).to(torch.float32).unsqueeze(0)
        unique_mask_values = set(torch.unique(mask_tensor).tolist())
        if not unique_mask_values.issubset({0.0, 1.0}):
            raise AssertionError(f"Mask interpolation broke binarity: {unique_mask_values}")
        sample = torch.cat([rgb_tensor, mask_tensor], dim=0)
        if sample.shape != (4, self.image_size, self.image_size):
            raise AssertionError(f"Expected [4,{self.image_size},{self.image_size}], got {sample.shape}")
        return sample

    def audit_sample(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, _, image_crop, mask_crop, metadata = self._load_source(idx)
        tensor = self._to_tensor(image_crop, mask_crop, apply_augmentation=False)
        metadata["mask_values_after_resize"] = sorted(torch.unique(tensor[3]).tolist())
        return tensor, metadata

    def visual_sample(
        self, idx: int
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Dict[str, Any]]:
        image, full_mask, image_crop, mask_crop, metadata = self._load_source(idx)
        final_crop = image_crop.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)
        final_mask = mask_crop.resize((self.image_size, self.image_size), RESAMPLE_NEAREST)
        overlay = final_crop.copy()
        green = Image.new("RGB", overlay.size, (0, 255, 0))
        alpha = final_mask.point(lambda value: 96 if value > 0 else 0)
        overlay.paste(green, mask=alpha)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, self.image_size - 1, self.image_size - 1), outline=(255, 220, 0), width=3)
        return image, full_mask, overlay, final_crop, metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        _, _, image_crop, mask_crop, _ = self._load_source(idx)
        tensor = self._to_tensor(image_crop, mask_crop, apply_augmentation=self.augment)
        rec = self.records[idx]
        return tensor, rec["label"], rec["cow_id"]


class ResNet18ReIDPerception(ResNet18ReIDBaseline):
    """Run 3 ResNet-18 with only conv1 expanded from RGB to RGB+mask."""

    def __init__(self, num_classes: int = 41, pretrained: bool = True) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        rgb_conv = self.conv1
        four_channel_conv = nn.Conv2d(
            4,
            rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            four_channel_conv.weight[:, :3].copy_(rgb_conv.weight)
            four_channel_conv.weight[:, 3:4].copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
        self.conv1 = four_channel_conv


def _parameter_counts() -> Dict[str, int]:
    baseline = ResNet18ReIDBaseline(num_classes=41, pretrained=False)
    perception = ResNet18ReIDPerception(num_classes=41, pretrained=False)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters() if parameter.requires_grad)
    perception_count = sum(parameter.numel() for parameter in perception.parameters() if parameter.requires_grad)
    return {
        "run3_rgb_trainable_parameters": baseline_count,
        "run6_rgb_mask_trainable_parameters": perception_count,
        "fourth_channel_parameter_difference": perception_count - baseline_count,
    }


def _integrity_audit(datasets: List[SideViewGTMaskReIDDataset]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        for idx in range(len(dataset)):
            tensor, metadata = dataset.audit_sample(idx)
            if tensor.shape != (4, 224, 224):
                raise AssertionError(f"Integrity audit tensor shape failed: {tensor.shape}")
            records.append(metadata)
    return {
        "pairs_verified": len(records),
        "invalid_pairs": 0,
        "all_rgb_paths_exist": True,
        "all_mask_paths_exist": True,
        "all_masks_nonempty": True,
        "all_crops_in_bounds": True,
        "all_rgb_mask_crop_coordinates_identical": True,
        "all_resized_masks_binary": True,
        "example_records": records[:8],
        "crop_width_range_before_resize": [
            min(record["crop_size_wh_before_resize"][0] for record in records),
            max(record["crop_size_wh_before_resize"][0] for record in records),
        ],
        "crop_height_range_before_resize": [
            min(record["crop_size_wh_before_resize"][1] for record in records),
            max(record["crop_size_wh_before_resize"][1] for record in records),
        ],
    }


def _fit_panel(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, method=RESAMPLE_BILINEAR, color=(20, 20, 20))


def build_contact_sheet(
    dataset: SideViewGTMaskReIDDataset, output_path: Path, max_examples: int = 6
) -> Dict[str, Any]:
    selected: List[int] = []
    seen_recordings = set()
    for idx, record in enumerate(dataset.records):
        recording_id = record["recording_id"]
        if recording_id not in seen_recordings:
            selected.append(idx)
            seen_recordings.add(recording_id)
        if len(selected) == max_examples:
            break

    panel_width = 224
    header_height = 52
    row_height = header_height + 224
    sheet = Image.new("RGB", (panel_width * 4, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    rows = []
    labels = ["Original RGB", "GT binary mask", "Crop + mask overlay", "Final 224x224 crop"]

    for row_idx, idx in enumerate(selected):
        image, mask, overlay, final_crop, metadata = dataset.visual_sample(idx)
        panels = [_fit_panel(image), _fit_panel(mask), overlay, final_crop]
        y0 = row_idx * row_height
        title = (
            f"cow={metadata['cow_id']} | recording={metadata['recording_id']} | "
            f"crop={metadata['crop_size_wh_before_resize'][0]}x{metadata['crop_size_wh_before_resize'][1]}"
        )
        draw.text((5, y0 + 3), title, fill="black", font=font)
        for col_idx, (label, panel) in enumerate(zip(labels, panels)):
            x0 = col_idx * panel_width
            draw.text((x0 + 5, y0 + 25), label, fill="black", font=font)
            sheet.paste(panel, (x0, y0 + header_height))
        rows.append(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90, optimize=True)
    return {"path": str(output_path), "examples": len(rows), "rows": rows}


@torch.no_grad()
def evaluate_validation(
    model: ResNet18ReIDPerception,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    predictions: List[int] = []
    targets_all: List[int] = []
    for samples, targets, _ in loader:
        samples = samples.to(device)
        targets = targets.to(device)
        logits, _ = model(samples)
        total_loss += criterion(logits, targets).item() * samples.size(0)
        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    count = len(loader.dataset)
    return {
        "val_loss": round(total_loss / count, 4),
        "val_top1_acc": round(accuracy_score(targets_all, predictions) * 100, 2),
        "val_bal_acc": round(balanced_accuracy_score(targets_all, predictions) * 100, 2),
        "val_macro_f1": round(f1_score(targets_all, predictions, average="macro", zero_division=0), 4),
    }


def train_sideview_reid_perception(
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
    smoke_samples: int = 64,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
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
    print("RUN 6: SIDEVIEWCOWS2026 GT/ORACLE-MASK PERCEPTION-ENHANCED RE-ID")
    print(f"Mode: {'LOCAL SMOKE (TRAIN/VAL ONLY)' if smoke else 'FULL TRAINING + FINAL RETRIEVAL'}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("Representation: mask-derived 5% crop + explicit binary mask channel")
    print("=" * 76)

    protocol_a_path = protocols_dir / "protocol_cross_setting.csv"
    protocol_d_path = protocols_dir / "protocol_closed_set.csv"
    if not protocol_a_path.exists() or not protocol_d_path.exists():
        raise FileNotFoundError(f"Missing SideView protocol files under {protocols_dir}")

    df_a = pd.read_csv(protocol_a_path)
    train_cows = sorted(
        df_a.loc[df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique()
    )
    evaluation_rows = df_a.loc[~df_a["setting_role"].eq("train")]
    evaluation_cows = set(evaluation_rows["individual_id"].astype(str).unique())
    if len(train_cows) != 41 or len(evaluation_cows) != 69:
        raise AssertionError(
            f"Protocol A identity counts changed: {len(train_cows)} train / {len(evaluation_cows)} eval"
        )
    if set(train_cows).intersection(evaluation_cows):
        raise AssertionError("Protocol A train/evaluation identity overlap detected")

    df_d = pd.read_csv(protocol_d_path)
    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train_full = df_d_41[df_d_41["closed_set_split"].eq("train")].reset_index(drop=True)
    df_val_full = df_d_41[df_d_41["closed_set_split"].eq("val")].reset_index(drop=True)
    if len(df_train_full) != 12753 or len(df_val_full) != 2683:
        raise AssertionError(
            f"Canonical Protocol D counts changed: {len(df_train_full)} train / {len(df_val_full)} val"
        )
    if df_train_full["individual_id"].astype(str).nunique() != 41:
        raise AssertionError("Training split no longer contains exactly 41 identities")
    if df_val_full["individual_id"].astype(str).nunique() != 41:
        raise AssertionError("Validation split no longer contains exactly 41 identities")

    if smoke:
        df_train = df_train_full.sample(
            n=min(smoke_samples, len(df_train_full)), random_state=seed
        ).reset_index(drop=True)
        df_val = df_val_full.sample(
            n=min(smoke_samples, len(df_val_full)), random_state=seed
        ).reset_index(drop=True)
    else:
        df_train = df_train_full
        df_val = df_val_full

    accessed_cows = set(df_train["individual_id"].astype(str)).union(
        df_val["individual_id"].astype(str)
    )
    if not accessed_cows.issubset(set(train_cows)):
        raise AssertionError("Smoke subset contains a held-out Protocol A evaluation cow")

    cow_to_label = {cow: index for index, cow in enumerate(train_cows)}
    label_to_cow = {index: cow for cow, index in cow_to_label.items()}
    train_dataset = SideViewGTMaskReIDDataset(
        df_train, data_root, cow_to_label, augment=True
    )
    val_dataset = SideViewGTMaskReIDDataset(
        df_val, data_root, cow_to_label, augment=False
    )

    integrity = _integrity_audit([train_dataset, val_dataset])
    contact_sheet = build_contact_sheet(
        val_dataset, output_dir / "gt_mask_crop_contact_sheet.jpg"
    )
    print(
        f"Integrity PASS: {integrity['pairs_verified']} real RGB-mask pairs; "
        "0 invalid; masks binary after nearest-neighbor resize"
    )

    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model = ResNet18ReIDPerception(num_classes=41, pretrained=True).to(device)
    counts = _parameter_counts()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_val_metric = -1.0
    history: List[Dict[str, Any]] = []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_metric = float(checkpoint["best_val_metric"])
        history = list(checkpoint.get("history", []))

    shape_checks: Dict[str, Any] = {}
    training_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
            file=sys.stdout,
            mininterval=1.0,
        )
        for samples, targets, _ in progress:
            samples = samples.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_features, embeddings = model.extract_features(samples)
            logits = model.classifier(raw_features)

            if not shape_checks:
                norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
                if samples.shape[1:] != (4, 224, 224):
                    raise AssertionError(f"Input shape failed: {samples.shape}")
                if raw_features.shape != (samples.shape[0], 512):
                    raise AssertionError(f"Raw feature shape failed: {raw_features.shape}")
                if embeddings.shape != (samples.shape[0], 512):
                    raise AssertionError(f"Embedding shape failed: {embeddings.shape}")
                if logits.shape != (samples.shape[0], 41):
                    raise AssertionError(f"Logit shape failed: {logits.shape}")
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
                    raise AssertionError("Embedding L2 norms are not approximately one")
                shape_checks = {
                    "input_shape": list(samples.shape),
                    "raw_feature_shape": list(raw_features.shape),
                    "embedding_shape": list(embeddings.shape),
                    "logits_shape": list(logits.shape),
                    "embedding_norm_min": float(norms.min().item()),
                    "embedding_norm_max": float(norms.max().item()),
                }

            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            if not any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise AssertionError("Backward pass produced no finite gradients")
            optimizer.step()

            batch_count = samples.size(0)
            train_count += batch_count
            train_loss_sum += float(loss.item()) * batch_count
            train_correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
            progress.set_postfix(
                loss=f"{train_loss_sum / train_count:.4f}",
                acc=f"{100 * train_correct / train_count:.1f}%",
            )

        scheduler.step()
        validation = evaluate_validation(model, val_loader, criterion, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss_sum / train_count, 4),
            "train_top1_acc": round(100 * train_correct / train_count, 2),
            **validation,
            "lr": float(scheduler.get_last_lr()[0]),
            "duration_sec": round(time.perf_counter() - epoch_start, 2),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        is_best = validation["val_top1_acc"] > best_val_metric
        if is_best:
            best_val_metric = validation["val_top1_acc"]
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_metric": best_val_metric,
            "cow_to_label": cow_to_label,
            "label_to_cow": label_to_cow,
            "history": history,
            "config": {
                "representation": "GT/oracle target-mask crop + explicit binary mask channel",
                "margin_fraction": 0.05,
                "num_classes": 41,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
                "smoke": smoke,
            },
            "git_sha": get_git_commit_sha(),
        }
        latest_path = output_dir / "reid_perception_latest.pth"
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, output_dir / "reid_perception_best.pth")

    training_seconds = time.perf_counter() - training_start

    # Checkpoint round-trip on a fixed, unaugmented validation batch.
    model.eval()
    fixed_samples, _, _ = next(iter(val_loader))
    fixed_samples = fixed_samples.to(device)
    with torch.no_grad():
        logits_before, _ = model(fixed_samples)
    reloaded_model = ResNet18ReIDPerception(num_classes=41, pretrained=False).to(device)
    reloaded_checkpoint = torch.load(
        output_dir / "reid_perception_latest.pth", map_location=device, weights_only=False
    )
    reloaded_model.load_state_dict(reloaded_checkpoint["model_state_dict"])
    reloaded_model.eval()
    with torch.no_grad():
        logits_after, _ = reloaded_model(fixed_samples)
    max_logit_difference = float((logits_before - logits_after).abs().max().item())
    bit_identical = bool(torch.equal(logits_before, logits_after))
    if max_logit_difference != 0.0:
        raise AssertionError(
            f"Checkpoint reload changed logits (max difference {max_logit_difference})"
        )

    retrieval_results: Dict[str, Any]
    held_out_images_loaded = 0
    if smoke:
        retrieval_results = {
            "status": "NOT_LOADED_OR_EVALUATED_IN_SMOKE_MODE",
            "gallery_loaded": False,
            "query_barn_loaded": False,
            "query_snapshots_loaded": False,
        }
    else:
        best_checkpoint = torch.load(
            output_dir / "reid_perception_best.pth", map_location=device, weights_only=False
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()
        gallery_df = df_a[df_a["setting_role"].eq("gallery")].reset_index(drop=True)
        barn_df = df_a[df_a["setting_role"].eq("query_barn")].reset_index(drop=True)
        snapshots_df = df_a[df_a["setting_role"].eq("query_snapshots")].reset_index(drop=True)
        held_out_images_loaded = len(gallery_df) + len(barn_df) + len(snapshots_df)
        evaluation_loaders = []
        for frame in (gallery_df, barn_df, snapshots_df):
            dataset = SideViewGTMaskReIDDataset(
                frame, data_root, {cow: 0 for cow in evaluation_cows}, augment=False
            )
            evaluation_loaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            )
        gallery_features, gallery_ids = extract_dataset_embeddings(
            model, evaluation_loaders[0], device, desc="Gallery Parlor"
        )
        barn_features, barn_ids = extract_dataset_embeddings(
            model, evaluation_loaders[1], device, desc="Query Barn"
        )
        snapshot_features, snapshot_ids = extract_dataset_embeddings(
            model, evaluation_loaders[2], device, desc="Query Snapshots"
        )
        retrieval_results = {
            "query_barn": evaluate_retrieval_chunked(
                barn_features, barn_ids, gallery_features, gallery_ids
            ),
            "query_snapshots": evaluate_retrieval_chunked(
                snapshot_features, snapshot_ids, gallery_features, gallery_ids,
                chunk_size=607,
            ),
        }

    metrics = {
        "task": "run6_sideview_gt_mask_perception_reid",
        "status": "smoke_certified_full_training_pending" if smoke else "full_training_complete",
        "representation": {
            "condition": "GT/oracle segmentation-guided Re-ID representation",
            "automatic_deployment_pipeline": False,
            "mask_source": "SideViewCows2026 target-cow ground truth",
            "crop_source": "bounding box derived from the same GT target-cow mask",
            "crop_margin_fraction": 0.05,
            "rgb_mask_crop_coordinates_identical": True,
            "rgb_normalization": "ImageNet",
            "mask_normalization": "none; binary float {0,1}",
            "rgb_multiplied_by_mask": False,
        },
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "protocol": {
            "canonical_train_images": len(df_train_full),
            "canonical_val_images": len(df_val_full),
            "representation_learning_cows": len(train_cows),
            "held_out_evaluation_cows": len(evaluation_cows),
            "smoke_train_images": len(df_train),
            "smoke_val_images": len(df_val),
            "smoke_unique_cows_accessed": len(accessed_cows),
            "accessed_cows_subset_of_41_training_cows": True,
            "held_out_images_loaded": held_out_images_loaded,
        },
        "integrity": integrity,
        "contact_sheet": contact_sheet,
        "shape_checks": shape_checks,
        "parameter_counts": counts,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "history": history,
            "total_training_time_sec": round(training_seconds, 2),
            "loss_finite": True,
            "backward_pass_verified": True,
        },
        "total_pipeline_runtime_sec": round(time.perf_counter() - pipeline_start, 2),
        "checkpoint_reload": {
            "checkpoint_saved": True,
            "bit_identical_logits": bit_identical,
            "max_logit_difference": max_logit_difference,
        },
        "protocol_a_retrieval": retrieval_results,
        "git_sha_at_execution": get_git_commit_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = output_dir / "reid_perception_smoke_metrics.json" if smoke else output_dir / "reid_perception_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {metrics_path}")
    print(f"Checkpoint reload max-logit difference: {max_logit_difference:.8f}")
    if smoke:
        print("Protocol A gallery/query images loaded: 0")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 6 SideViewCows2026 GT-mask perception Re-ID trainer"
    )
    parser.add_argument(
        "--data-root", default="datasets/id/external/sideviewcows2026", type=Path
    )
    parser.add_argument(
        "--protocols-dir", default="datasets/id/sideviewcows2026", type=Path
    )
    parser.add_argument(
        "--output-dir", default="artifacts/reid_perception", type=Path
    )
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--resume", default=None, type=Path)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke-samples", default=64, type=int)
    args = parser.parse_args()

    train_sideview_reid_perception(
        data_root=args.data_root,
        protocols_dir=args.protocols_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        seed=args.seed,
        smoke=args.smoke,
        smoke_samples=args.smoke_samples,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
