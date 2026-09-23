# -*- coding: utf-8 -*-
"""
train_moo_real_viewpoint.py — 3-Class Fine-Tuning of MOO-Pretrained ResNet-18 on Real Cattle Viewpoint Crops

Architecture:
  - Backbone: Pretrained ResNet-18 loaded from MOO synthetic checkpoint (moo_resnet18_viewpoint8_full_l4.pth)
  - Head: Fresh Linear(512, 3) classification head for ['front', 'side', 'rear']
  - Loss: CrossEntropyLoss
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4) with CosineAnnealingLR
  - Data: Derived RT-DETR cow crops from self_clean_v1_rtdetr_crop/

Augmentation & Viewpoint Preservation:
  - Bilateral symmetry justification: In a 3-class viewpoint taxonomy ('front', 'side', 'rear'),
    left-right horizontal flipping maps left-side to right-side (both are 'side'),
    and cranial/caudal symmetry preserves 'front' and 'rear'. Therefore, RandomHorizontalFlip(p=0.5)
    is geometrically valid and strictly label-preserving.
  - RandomResizedCrop(224, scale=(0.85, 1.0))
  - ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

Usage:
  python scripts/train_moo_real_viewpoint.py --smoke
  python scripts/train_moo_real_viewpoint.py --epochs 20 --batch-size 32
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, classification_report

CLASS_NAMES = ["front", "side", "rear"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

SEED = 2026


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ViewpointCropDataset(Dataset):
    """
    Dataset loader for RT-DETR viewpoint crops.
    Resolves image paths either relative to root or data_dir.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path, transform: Optional[T.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        clean_id = str(row["clean_id"])
        cls_name = str(row["class"])
        target = CLASS_TO_IDX[cls_name]

        # Resolve image path
        # Try local subpath relative to data_dir
        crop_filename = f"{clean_id}.jpg"
        img_path = self.data_dir / cls_name / crop_filename
        if not img_path.exists():
            # Fallback to crop_path from manifest
            crop_rel = str(row.get("crop_path", "")).replace("\\", "/")
            img_path = self.data_dir / Path(crop_rel).name
            if not img_path.exists():
                # Direct check
                img_path = Path(crop_rel)

        if not img_path.exists():
            raise FileNotFoundError(f"Crop image not found: {img_path} for clean_id {clean_id}")

        img = Image.open(str(img_path)).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)

        return tensor, target, clean_id


def build_transforms():
    """
    Construct train and validation image transforms.
    """
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),  # Geometrically sound for front/side/rear
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def load_moo_pretrained_resnet18(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load learned ResNet-18 backbone from MOO synthetic checkpoint,
    and initialize a fresh 3-class classification head.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MOO checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    model = models.resnet18()
    # Filter out the 8-class classification head
    backbone_weights = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(backbone_weights, strict=False)

    # Attach fresh 3-class head
    model.fc = nn.Linear(512, 3)
    model = model.to(device)

    print(f"✓ Loaded MOO backbone from: {ckpt_path.name}")
    print(f"  Old classes in MOO checkpoint: {checkpoint.get('class_names')}")
    print(f"  New classification head:       Linear(512, 3) -> {CLASS_NAMES}")
    print(f"  Backbone missing keys: {missing} | Unexpected: {unexpected}")

    meta = {
        "source_checkpoint": ckpt_path.name,
        "source_best_val_f1": checkpoint.get("best_val_macro_f1", 0.0),
        "source_classes": checkpoint.get("class_names"),
        "target_classes": CLASS_NAMES,
    }
    return model, meta


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> Dict[str, Any]:
    """Evaluate model on validation split and compute balanced metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"{desc:5s}", leave=False, file=sys.stdout, dynamic_ncols=True)
    with torch.no_grad():
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = total_loss / max(1, total_samples)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_recall = {CLASS_NAMES[i]: float(recalls[i]) if i < len(recalls) else 0.0 for i in range(len(CLASS_NAMES))}

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_recall": per_class_recall,
        "n_samples": total_samples,
    }


def train_pipeline(
    data_dir: Path,
    moo_ckpt_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device_name: str = "",
    smoke: bool = False,
    smoke_samples: int = 32,
) -> Dict[str, Any]:
    """
    Execute full fine-tuning training and validation schedule.
    """
    set_seed(SEED)

    if not device_name:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  MOO -> REAL CATTLE VIEWPOINT 3-CLASS FINE-TUNING")
    print(f"  Mode:            {'SMOKE TEST (Readiness Verification)' if smoke else 'FULL TRAINING'}")
    print(f"  Device:          {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Data Directory:  {data_dir}")
    print(f"  MOO Checkpoint:  {moo_ckpt_path}")
    print(f"  Output Dir:      {output_dir}")
    print(f"  Epochs:          {epochs if not smoke else 2}")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Learning Rate:   {lr}")
    print("=" * 80)

    # 1. Load splits (STRICT TEST SET PROTECTION: test.csv is NEVER loaded during training)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    if not train_csv.exists():
        train_csv = data_dir / "splits" / "train.csv"
        val_csv = data_dir / "splits" / "val.csv"

    assert train_csv.exists(), f"Missing train.csv at {train_csv}"
    assert val_csv.exists(), f"Missing val.csv at {val_csv}"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"\n[1/4] Loaded splits (TEST SET STRICTLY EXCLUDED):")
    print(f"  - Train: {len(train_df)} samples across classes: {dict(train_df['class'].value_counts())}")
    print(f"  - Val:   {len(val_df)} samples across classes: {dict(val_df['class'].value_counts())}")

    if smoke:
        # Sample tiny balanced subsets for smoke test
        train_df = train_df.groupby("class").head(10).reset_index(drop=True)
        val_df = val_df.groupby("class").head(5).reset_index(drop=True)
        epochs = 2
        print(f"  [SMOKE] Downsampled to Train={len(train_df)}, Val={len(val_df)}, Epochs=2")

    # 2. Datasets and DataLoaders
    train_transform, val_transform = build_transforms()
    train_dataset = ViewpointCropDataset(train_df, data_dir, transform=train_transform)
    val_dataset = ViewpointCropDataset(val_df, data_dir, transform=val_transform)

    num_workers = 2 if (device.type == "cuda" and not sys.platform.startswith("win")) else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. Model initialization
    print("\n[2/4] Initializing model from MOO checkpoint...")
    model, moo_meta = load_moo_pretrained_resnet18(moo_ckpt_path, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training loop with live tqdm
    print("\n[3/4] Launching training...")
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    history = []

    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )

        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            n_train_samples += bs
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        scheduler.step()
        epoch_train_loss = running_loss / max(1, n_train_samples)

        # Validation evaluation
        val_res = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch:02d} [Val]")

        # Print per-epoch summary
        cur_lr = scheduler.get_last_lr()[0]
        recalls_str = " ".join([f"{k[:1].upper()}:{v*100:.1f}%" for k, v in val_res["per_class_recall"].items()])
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} | "
            f"Val Acc: {val_res['accuracy']*100:.2f}% | "
            f"Val Bal Acc: {val_res['balanced_accuracy']*100:.2f}% | "
            f"Val Macro-F1: {val_res['macro_f1']:.4f} ({recalls_str}) | "
            f"lr: {cur_lr:.1e}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["accuracy"],
            "val_balanced_accuracy": val_res["balanced_accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "per_class_recall": val_res["per_class_recall"],
            "lr": cur_lr,
        }
        history.append(epoch_record)

        # Save Best Checkpoint (PRIMARY = Val Macro-F1)
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            best_epoch = epoch
            best_metrics = val_res

            best_ckpt_path = output_dir / "viewpoint_resnet18_real_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "val_macro_f1": val_res["macro_f1"],
                "val_balanced_accuracy": val_res["balanced_accuracy"],
                "val_accuracy": val_res["accuracy"],
                "per_class_recall": val_res["per_class_recall"],
                "moo_metadata": moo_meta,
                "seed": SEED,
            }, best_ckpt_path)

        # Save latest checkpoint
        latest_ckpt_path = output_dir / "viewpoint_resnet18_real_latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "class_to_idx": CLASS_TO_IDX,
            "latest_val_macro_f1": val_res["macro_f1"],
            "seed": SEED,
        }, latest_ckpt_path)

    total_time = time.time() - t_train_start
    print(f"\n✓ Training completed in {total_time:.1f}s ({total_time / max(1, epochs):.1f}s/epoch)")

    # 5. Export metrics JSON and summary
    print("\n[4/4] Archiving metrics and training report...")
    metrics_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_metrics": best_metrics,
        "total_epochs": epochs,
        "total_time_seconds": round(total_time, 2),
        "history": history,
        "moo_meta": moo_meta,
        "class_names": CLASS_NAMES,
    }

    import json
    metrics_json_path = output_dir / "viewpoint_training_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("=" * 80)
    print("  TRAINING SUMMARY SCORECARD")
    print("=" * 80)
    print(f"  Best Epoch:           {best_epoch}/{epochs}")
    print(f"  Best Val Macro-F1:    {best_val_macro_f1:.4f}")
    if best_metrics:
        print(f"  Best Val Bal Acc:     {best_metrics.get('balanced_accuracy', 0.0)*100:.2f}%")
        print(f"  Best Val Accuracy:    {best_metrics.get('accuracy', 0.0)*100:.2f}%")
        for cls_name, rec in best_metrics.get("per_class_recall", {}).items():
            print(f"    - Recall {cls_name:5s}:     {rec*100:.1f}%")
    print(f"  Best Checkpoint:      {output_dir / 'viewpoint_resnet18_real_best.pth'}")
    print(f"  Metrics JSON:         {metrics_json_path}")
    print("=" * 80)

    return metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MOO ResNet-18 on real viewpoint crops.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--data-dir", type=str, default="datasets/viewpoint/self_clean_v1_rtdetr_crop", help="Path to cropped viewpoint dataset.")
    parser.add_argument("--moo-ckpt", type=str, default="artifacts/checkpoints/moo_resnet18_viewpoint8_full_l4.pth", help="Path to MOO synthetic checkpoint.")
    parser.add_argument("--output-dir", type=str, default="artifacts/viewpoint_real_finetune", help="Output directory for checkpoints.")
    parser.add_argument("--device", type=str, default="", help="Device ('cuda' or 'cpu').")
    parser.add_argument("--smoke", action="store_true", help="Run 2-epoch smoke test.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    moo_ckpt = Path(args.moo_ckpt)
    out_dir = Path(args.output_dir)

    train_pipeline(
        data_dir=data_dir,
        moo_ckpt_path=moo_ckpt,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device_name=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
