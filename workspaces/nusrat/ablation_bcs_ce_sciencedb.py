"""Ablation script: standard Cross-Entropy for BCS on ScienceDB (optimized for 100% GPU utilization)."""
import os
import time
import random
import warnings
import csv
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)  # Prevents OpenCV multithreading deadlock with DataLoader

SEED = 42
PERSON_NAME = "Nusrat"
BASE_MODEL = "EfficientNetB0"
TIMM_MODEL_NAME = "efficientnet_b0"

BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")
SCIENCE_DB_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "sciencedb_bcs_ce_best.pth")
RESULTS_TXT = os.path.join(WORKSPACE_DIR, "bcs_ce_sciencedb_ablation_results.txt")
LOSS_CURVE_PNG = os.path.join(WORKSPACE_DIR, "bcs_ce_sciencedb_ablation_loss_curve.png")

NUM_CLASSES = 5
BATCH_SIZE = 256  # Reduced from 512 to prevent VRAM swapping to system RAM
EPOCHS = 10       # Reduced to 10 to meet deadline
LR = 2e-3         # Scaled learning rate for larger batch size
STEP_SIZE = 10
GAMMA = 0.5
DEVICE = torch.device("cuda")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False  # Disabled to prevent 5-10 minute startup delays

class BCSDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.samples = []
        bcs_map = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4}
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['split'] == split:
                    p = row['image_path']
                    if not os.path.isabs(p):
                        p = os.path.join(BASE_DIR, p)
                    self.samples.append((p, bcs_map[float(row['label'])]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)

def get_transforms(split):
    if split == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0, value=(0,0,0)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 1)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=[-2, -1])
        max_ = x.max(dim=-1)[0].max(dim=-1)[0]
        a = self.fc(avg)
        m = self.fc(max_)
        return x * self.sig(a + m).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)
        m, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([a, m], dim=1)))

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class BCSModelCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(TIMM_MODEL_NAME, pretrained=True, num_classes=0, global_pool="")
        fd = self.backbone.num_features
        self.cbam = CBAM(fd)
        self.head = nn.Linear(fd, NUM_CLASSES)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = self.cbam(f)
        f = f.mean(dim=[-2, -1])  # Equivalent to AdaptiveAvgPool2d(1) + flatten(1)
        return self.head(f)

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"ScienceDB | Epoch {epoch:02d}/{EPOCHS} | Train", leave=True)
    for images, labels in progress:
        # Transfer memory format to channels_last for maximum Tensor Core layout speed
        images = images.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Autocast AMP
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, split_name):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_labels, all_preds = [], []

    progress = tqdm(loader, desc=f"ScienceDB | {split_name}", leave=True)
    for images, labels in progress:
        images = images.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    exact = accuracy_score(all_labels, all_preds)
    pm1 = np.mean(np.abs(all_labels - all_preds) <= 1)

    return {
        "loss": running_loss / total_samples,
        "mae": float(mae),
        "acc_exact": float(exact),
        "acc_within_1": float(pm1)
    }

def main():
    set_seed(SEED)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    print("=" * 80)
    print("BCS CE SCIENCEDB ABLATION SCRIPT (MAX GPU OPTIMIZED)")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | Device: {DEVICE}")
    print("=" * 80)

    train_loader = DataLoader(BCSDataset(SCIENCE_DB_CSV, "train", get_transforms("train")),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(BCSDataset(SCIENCE_DB_CSV, "val", get_transforms("val")),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(BCSDataset(SCIENCE_DB_CSV, "test", get_transforms("test")),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = BCSModelCE().to(DEVICE)
    
    # 1. Memory Format Optimization: Use Channels Last layout
    model = model.to(memory_format=torch.channels_last)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"--> Resuming weights from existing checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler = torch.amp.GradScaler('cuda')

    best_val_mae = float("inf")
    best_epoch = None
    best_val_metrics = None
    train_losses, val_losses = [], []
    loss_at_10, loss_at_20, loss_at_30 = None, None, None

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        val_metrics = evaluate(model, val_loader, f"Val Epoch {epoch:02d}")

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        scheduler.step()

        if epoch == 10: loss_at_10 = train_loss
        if epoch == 20: loss_at_20 = train_loss
        if epoch == 30: loss_at_30 = train_loss

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val MAE: {val_metrics['mae']:.4f} | Val Acc: {val_metrics['acc_exact']*100:.2f}%")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            # Uncompile helper before saving weights to keep checkpoint loadable
            torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), CHECKPOINT_PATH)
            print(f"--> Saved best checkpoint at epoch {epoch}")

    total_time_mins = (time.time() - start_time) / 60.0

    print("Evaluating best model on Test set...")
    best_model = BCSModelCE().to(DEVICE).to(memory_format=torch.channels_last)
    best_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    best_model.eval()
    test_metrics = evaluate(best_model, test_loader, "Test Evaluation")

    report = f"""---CONTEXT 3 BCS CE SCIENCEDB ABLATION---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL}
DATASET: ScienceDB (Cross-Entropy Loss)

EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {loss_at_10:.6f}
LOSS AT EPOCH 20: {loss_at_20:.6f}
LOSS AT EPOCH 30: {loss_at_30:.6f}
FINAL TRAIN LOSS: {train_losses[-1]:.6f}
VAL MAE: {best_val_metrics['mae']:.6f}
VAL ACCURACY +-0 (exact match): {best_val_metrics['acc_exact']:.6f}
VAL ACCURACY +-1 (within 1 class): {best_val_metrics['acc_within_1']:.6f}
TEST MAE: {test_metrics['mae']:.6f}
TEST ACCURACY +-0: {test_metrics['acc_exact']:.6f}
TEST ACCURACY +-1: {test_metrics['acc_within_1']:.6f}
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {total_time_mins:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    # Save Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_losses, label="ScienceDB CE Train Loss")
    plt.plot(range(1, EPOCHS+1), val_losses, label="ScienceDB CE Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ScienceDB BCS CE Ablation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_PNG, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
