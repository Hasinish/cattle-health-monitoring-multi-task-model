import os
import time
import random
import warnings
from pathlib import Path

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
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

warnings.filterwarnings("ignore")

SEED = 42
PERSON_NAME = "Nusrat"
BASE_MODEL = "EfficientNetB0"
TIMM_MODEL_NAME = "efficientnet_b0"

BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")

DRYAD_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "bcs_index.csv")
SCIENCEDB_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")

DRYAD_CHECKPOINT = os.path.join(WORKSPACE_DIR, "dryad_bcs_best.pth")
SCIENCEDB_CHECKPOINT = os.path.join(WORKSPACE_DIR, "sciencedb_bcs_best.pth")

RESULTS_TXT = os.path.join(WORKSPACE_DIR, "bcs_results.txt")
LOSS_CURVE_PNG = os.path.join(WORKSPACE_DIR, "bcs_loss_curve.png")

NUM_CLASSES = 5
CORAL_OUTPUTS = NUM_CLASSES - 1

BATCH_SIZE = 128  # Increased from 32 to 128 to maximize GPU usage on RTX 4080 SUPER
EPOCHS = 30
LR = 1e-3
STEP_SIZE = 10
GAMMA = 0.5

DEVICE = torch.device("cuda")

DRYAD_LABEL_MAP = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
}

SCIENCEDB_LABEL_MAP = {
    3.25: 0,
    3.5: 1,
    3.75: 2,
    4.0: 3,
    4.25: 4,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_map, transform=None):
        self.csv_path = csv_path
        self.split = split
        self.label_map = label_map
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        raw_label = row["label"]

        try:
            raw_label_key = int(raw_label)
        except Exception:
            raw_label_key = float(raw_label)

        if raw_label_key not in self.label_map:
            raw_label_key = float(raw_label)

        label = self.label_map[raw_label_key]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(split):
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            normalize,
        ])

    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize,
    ])


def build_loaders(csv_path, label_map):
    train_dataset = BCSDataset(csv_path, "train", label_map, get_transforms("train"))
    val_dataset = BCSDataset(csv_path, "val", label_map, get_transforms("val"))
    test_dataset = BCSDataset(csv_path, "test", label_map, get_transforms("test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# CBAM ATTENTION
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        hidden_channels = max(in_channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))

        attention = self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        attention = self.sigmoid(
            self.conv(torch.cat([avg, max_], dim=1))
        )

        return x * attention


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

        self.backbone = timm.create_model(
            TIMM_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        self.feature_dim = self.backbone.num_features

        self.cbam = CBAM(self.feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.feature_dim, CORAL_OUTPUTS)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)
        logits = self.head(pooled)

        return logits


# ============================================================
# CORAL LOSS + METRICS
# ============================================================

def coral_loss(logits, labels, num_classes):
    sets = []

    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)

    labels_stacked = torch.stack(sets, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, labels_stacked)

    return loss


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    acc_exact = accuracy_score(y_true, y_pred)
    acc_within_1 = np.mean(np.abs(y_true - y_pred) <= 1)

    return {
        "mae": float(mae),
        "acc_exact": float(acc_exact),
        "acc_within_1": float(acc_within_1),
    }


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, optimizer, epoch, dataset_name):
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=f"{dataset_name} | Epoch {epoch:02d}/{EPOCHS} | Train",
        leave=True,
    )

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = coral_loss(logits, labels, NUM_CLASSES)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

    epoch_loss = running_loss / total_samples

    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, dataset_name, split_name):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_preds = []

    progress = tqdm(
        loader,
        desc=f"{dataset_name} | {split_name}",
        leave=True,
    )

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = coral_loss(logits, labels, NUM_CLASSES)

        preds = coral_predict(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

    avg_loss = running_loss / total_samples
    metrics = compute_metrics(all_labels, all_preds)

    metrics["loss"] = float(avg_loss)

    return metrics


def run_training(dataset_name, csv_path, label_map, checkpoint_path):
    print("=" * 80)
    print(f"STARTING DATASET: {dataset_name}")
    print(f"CSV: {csv_path}")
    print(f"CHECKPOINT: {checkpoint_path}")
    print("=" * 80)

    train_loader, val_loader, test_loader = build_loaders(csv_path, label_map)

    model = BCSModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    best_val_mae = float("inf")
    best_epoch = None
    best_val_metrics = None

    train_losses = []
    val_losses = []

    loss_at_epoch_10 = None
    loss_at_epoch_20 = None
    loss_at_epoch_30 = None

    start_time = time.time()
    issue_text = "None"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            dataset_name=dataset_name,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            dataset_name=dataset_name,
            split_name=f"Val Epoch {epoch:02d}/{EPOCHS}",
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        if epoch == 10:
            loss_at_epoch_10 = train_loss
        if epoch == 20:
            loss_at_epoch_20 = train_loss
        if epoch == 30:
            loss_at_epoch_30 = train_loss

        print(
            f"{dataset_name} | Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f} | "
            f"Val Acc +-0: {val_metrics['acc_exact']:.6f} | "
            f"Val Acc +-1: {val_metrics['acc_within_1']:.6f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_val_metrics = val_metrics

            torch.save(
                {
                    "person_name": PERSON_NAME,
                    "base_model": BASE_MODEL,
                    "dataset_name": dataset_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "label_map": label_map,
                    "num_classes": NUM_CLASSES,
                    "timm_model_name": TIMM_MODEL_NAME,
                },
                checkpoint_path,
            )

            print(
                f"{dataset_name} | New best checkpoint saved at epoch {epoch} "
                f"with Val MAE {best_val_mae:.6f}"
            )

    total_time_mins = (time.time() - start_time) / 60.0

    print(f"{dataset_name} | Loading best checkpoint for final test evaluation...")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        dataset_name=dataset_name,
        split_name="Test",
    )

    final_train_loss = train_losses[-1]

    print("=" * 80)
    print(f"COMPLETED DATASET: {dataset_name}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Best Val MAE: {best_val_metrics['mae']:.6f}")
    print(f"Best Val Acc +-0: {best_val_metrics['acc_exact']:.6f}")
    print(f"Best Val Acc +-1: {best_val_metrics['acc_within_1']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test Acc +-0: {test_metrics['acc_exact']:.6f}")
    print(f"Test Acc +-1: {test_metrics['acc_within_1']:.6f}")
    print(f"Training Time Mins: {total_time_mins:.2f}")
    print("=" * 80)

    return {
        "dataset_name": dataset_name,
        "epochs_trained": EPOCHS,
        "loss_at_epoch_10": loss_at_epoch_10,
        "loss_at_epoch_20": loss_at_epoch_20,
        "loss_at_epoch_30": loss_at_epoch_30,
        "final_train_loss": final_train_loss,
        "best_epoch": best_epoch,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": checkpoint_path,
        "training_time_mins": total_time_mins,
        "issues": issue_text,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ============================================================
# OUTPUT FILES
# ============================================================

def fmt_float(value):
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def write_context3_report(dryad_result, sciencedb_result):
    def block(result):
        return f"""DATASET: {result['dataset_name']}
EPOCHS TRAINED: {result['epochs_trained']}
LOSS AT EPOCH 10: {fmt_float(result['loss_at_epoch_10'])}
LOSS AT EPOCH 20: {fmt_float(result['loss_at_epoch_20'])}
LOSS AT EPOCH 30: {fmt_float(result['loss_at_epoch_30'])}
FINAL TRAIN LOSS: {fmt_float(result['final_train_loss'])}
VAL MAE: {fmt_float(result['val_metrics']['mae'])}
VAL ACCURACY +-0 (exact match): {fmt_float(result['val_metrics']['acc_exact'])}
VAL ACCURACY +-1 (within 1 class): {fmt_float(result['val_metrics']['acc_within_1'])}
TEST MAE: {fmt_float(result['test_metrics']['mae'])}
TEST ACCURACY +-0: {fmt_float(result['test_metrics']['acc_exact'])}
TEST ACCURACY +-1: {fmt_float(result['test_metrics']['acc_within_1'])}
CHECKPOINT PATH: {result['checkpoint_path']}
TRAINING TIME (mins): {result['training_time_mins']:.2f}
ANY ISSUES ENCOUNTERED: {result['issues']}"""

    report = f"""---CONTEXT 3 BCS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL}

{block(dryad_result)}

{block(sciencedb_result)}
---END CONTEXT 3---
"""

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Context 3 report saved to: {RESULTS_TXT}")
    print(report)


def save_loss_curve(dryad_result, sciencedb_result):
    plt.figure(figsize=(10, 6))

    epochs = list(range(1, EPOCHS + 1))

    plt.plot(
        epochs,
        dryad_result["train_losses"],
        label="Dryad Train Loss",
    )
    plt.plot(
        epochs,
        dryad_result["val_losses"],
        label="Dryad Val Loss",
    )
    plt.plot(
        epochs,
        sciencedb_result["train_losses"],
        label="ScienceDB Train Loss",
    )
    plt.plot(
        epochs,
        sciencedb_result["val_losses"],
        label="ScienceDB Val Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BCS Training and Validation Loss Curves - Nusrat - EfficientNetB0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(LOSS_CURVE_PNG, dpi=300)
    plt.close()

    print(f"Loss curve saved to: {LOSS_CURVE_PNG}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script must run on GPU as required."
        )

    print("=" * 80)
    print("BCS TRAINING SCRIPT")
    print(f"Person: {PERSON_NAME}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Timm Model: {TIMM_MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Workspace: {WORKSPACE_DIR}")
    print("=" * 80)

    dryad_result = run_training(
        dataset_name="Dryad",
        csv_path=DRYAD_CSV,
        label_map=DRYAD_LABEL_MAP,
        checkpoint_path=DRYAD_CHECKPOINT,
    )

    sciencedb_result = run_training(
        dataset_name="ScienceDB",
        csv_path=SCIENCEDB_CSV,
        label_map=SCIENCEDB_LABEL_MAP,
        checkpoint_path=SCIENCEDB_CHECKPOINT,
    )

    write_context3_report(dryad_result, sciencedb_result)
    save_loss_curve(dryad_result, sciencedb_result)

    print("=" * 80)
    print("ALL BCS TRAINING COMPLETED SUCCESSFULLY")
    print(f"Checkpoint 1: {DRYAD_CHECKPOINT}")
    print(f"Checkpoint 2: {SCIENCEDB_CHECKPOINT}")
    print(f"Results TXT: {RESULTS_TXT}")
    print(f"Loss Curve: {LOSS_CURVE_PNG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
