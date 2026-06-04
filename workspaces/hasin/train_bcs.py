import os
import time
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
SEED          = 42
PERSON_NAME   = "Hasin"
BASE_MODEL    = "ResNet-18"
TIMM_MODEL    = "resnet18"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\hasin"
DRYAD_CSV     = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
SCIDB_CSV     = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"

NUM_CLASSES   = 5       # ordinal classes 0-4
CORAL_OUTPUTS = NUM_CLASSES - 1   # 4 output nodes
BATCH_SIZE    = 64      # doubled from 32 — safe with AMP on RTX 4080 16GB
NUM_WORKERS   = 8
EPOCHS        = 30
LR            = 1e-3

DRYAD_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
SCIDB_LABEL_MAP = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4}

DEVICE = torch.device("cuda")

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# DATASET
# ============================================================
class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_map, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        # CRITICAL: dict lookup — never use int() or float() directly on the label
        raw = row["label"]
        try:
            key = int(raw)
        except (ValueError, TypeError):
            key = float(raw)
        if key not in self.label_map:
            key = float(raw)
        label = self.label_map[key]

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def get_transforms():
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ToTensor(),
        norm,
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf

def build_loaders(csv_path, label_map):
    train_tf, eval_tf = get_transforms()
    train_ds = BCSDataset(csv_path, "train", label_map, train_tf)
    val_ds   = BCSDataset(csv_path, "val",   label_map, eval_tf)
    test_ds  = BCSDataset(csv_path, "test",  label_map, eval_tf)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

# ============================================================
# CBAM ATTENTION
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ============================================================
# MODEL
# ============================================================
class BCSModel(nn.Module):
    def __init__(self):
        super().__init__()
        # global_pool='' gives spatial feature maps needed for CBAM
        self.backbone = timm.create_model(
            TIMM_MODEL, pretrained=True, num_classes=0, global_pool=""
        )
        self.backbone = self.backbone.to(DEVICE)

        # Dynamically detect feature dimension — never hardcode
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, 3, 224, 224).to(DEVICE))
            feature_dim = dummy.shape[1]

        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feature_dim, CORAL_OUTPUTS)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ============================================================
# CORAL LOSS + METRICS
# ============================================================
def coral_loss(logits, labels):
    sets = [(labels > i).float() for i in range(NUM_CLASSES - 1)]
    targets = torch.stack(sets, dim=1)
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    acc0   = float(np.mean(y_true == y_pred) * 100)
    acc1   = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100)
    return mae, acc0, acc1

# ============================================================
# TRAIN ONE EPOCH (AMP)
# ============================================================
def train_epoch(model, loader, optimizer, scaler, epoch, dataset_name):
    model.train()
    running_loss = 0.0
    total        = 0

    bar = tqdm(loader, desc=f"[{dataset_name}] Epoch {epoch:02d}/{EPOCHS} Train", leave=False)
    for imgs, labels in bar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss   = coral_loss(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        total        += imgs.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total

# ============================================================
# EVALUATE (AMP)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, desc):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(imgs)

        preds = coral_predict(logits)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return compute_metrics(all_labels, all_preds)

# ============================================================
# TRAINING RUN
# ============================================================
def run_training(dataset_name, csv_path, label_map, checkpoint_name):
    print(f"\n{'='*70}")
    print(f"  STARTING: {dataset_name}")
    print(f"{'='*70}")

    ckpt_path    = os.path.join(WORKSPACE_DIR, checkpoint_name)
    train_loader, val_loader, test_loader = build_loaders(csv_path, label_map)

    model     = BCSModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler    = torch.amp.GradScaler("cuda")

    best_val_mae  = float("inf")
    best_val_metrics = None
    train_losses  = []
    milestone     = {}          # {10: loss, 20: loss, 30: loss}

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch, dataset_name)
        scheduler.step()
        train_losses.append(train_loss)

        if epoch in (10, 20, 30):
            milestone[epoch] = train_loss

        val_mae, val_acc0, val_acc1 = evaluate(model, val_loader, f"[{dataset_name}] Epoch {epoch:02d} Val")
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.6f} | Val MAE: {val_mae:.4f} | Val ±0: {val_acc0:.2f}% | Val ±1: {val_acc1:.2f}%")

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            best_val_metrics = (val_mae, val_acc0, val_acc1)
            torch.save(model.state_dict(), ckpt_path)

    elapsed_mins = (time.time() - start) / 60.0

    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_mae, test_acc0, test_acc1 = evaluate(model, test_loader, f"[{dataset_name}] Test")

    print(f"\n  {dataset_name} DONE | Best Val MAE: {best_val_mae:.4f} | Test MAE: {test_mae:.4f} | Time: {elapsed_mins:.2f} mins")

    return {
        "dataset_name":   dataset_name,
        "loss_ep10":      milestone.get(10, "N/A"),
        "loss_ep20":      milestone.get(20, "N/A"),
        "loss_ep30":      milestone.get(30, "N/A"),
        "final_loss":     train_losses[-1],
        "val_mae":        best_val_metrics[0],
        "val_acc0":       best_val_metrics[1],
        "val_acc1":       best_val_metrics[2],
        "test_mae":       test_mae,
        "test_acc0":      test_acc0,
        "test_acc1":      test_acc1,
        "ckpt_path":      ckpt_path,
        "time_mins":      elapsed_mins,
        "train_losses":   train_losses,
    }

# ============================================================
# OUTPUT FILES
# ============================================================
def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def save_results(dryad, scidb):
    def block(r):
        return f"""DATASET: {r['dataset_name']}
EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {fmt(r['loss_ep10'])}
LOSS AT EPOCH 20: {fmt(r['loss_ep20'])}
LOSS AT EPOCH 30: {fmt(r['loss_ep30'])}
FINAL TRAIN LOSS: {fmt(r['final_loss'])}
VAL MAE: {fmt(r['val_mae'])}
VAL ACCURACY +-0 (exact match): {r['val_acc0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {r['val_acc1']:.2f}%
TEST MAE: {fmt(r['test_mae'])}
TEST ACCURACY +-0: {r['test_acc0']:.2f}%
TEST ACCURACY +-1: {r['test_acc1']:.2f}%
CHECKPOINT PATH: {r['ckpt_path']}
TRAINING TIME (mins): {r['time_mins']:.2f}
ANY ISSUES ENCOUNTERED: None"""

    report = f"""---CONTEXT 3 BCS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL}

{block(dryad)}

{block(scidb)}
---END CONTEXT 3---
"""
    out_path = os.path.join(WORKSPACE_DIR, "bcs_results.txt")
    with open(out_path, "w") as f:        # WRITE mode — never append
        f.write(report)
    print(f"\nResults saved: {out_path}")
    print(report)


def save_loss_curve(dryad, scidb):
    epochs = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, dryad["train_losses"], label="Dryad Train Loss",   color="steelblue")
    plt.plot(epochs, scidb["train_losses"], label="ScienceDB Train Loss", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("CORAL Loss")
    plt.title(f"BCS Training Loss — {PERSON_NAME} ({BASE_MODEL})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(WORKSPACE_DIR, "bcs_loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {out_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Check GPU drivers.")

    # Enable cudnn autotuner — finds fastest conv algorithms for fixed input size
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print(f"  BCS TRAINING — {PERSON_NAME} — {BASE_MODEL}")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print(f"  Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: ON")
    print("=" * 70)

    dryad_result = run_training("Dryad",     DRYAD_CSV, DRYAD_LABEL_MAP, "dryad_bcs_best.pth")
    scidb_result = run_training("ScienceDB", SCIDB_CSV, SCIDB_LABEL_MAP,  "sciencedb_bcs_best.pth")

    save_results(dryad_result, scidb_result)
    save_loss_curve(dryad_result, scidb_result)

    print("\n" + "=" * 70)
    print("  ALL TRAINING COMPLETE")
    print(f"  Dryad    — Test MAE: {dryad_result['test_mae']:.4f}")
    print(f"  ScienceDB — Test MAE: {scidb_result['test_mae']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()