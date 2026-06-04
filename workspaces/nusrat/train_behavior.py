import os
import time
import random
import multiprocessing

import cv2
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_loss_curve.png"

NUM_CLASSES = 7
BATCH_SIZE = 128
NUM_WORKERS = 8
MAX_EPOCHS = 30
PATIENCE = 10
RANDOM_SEED = 42

CLASS_NAMES = [
    "Walking",
    "Standing",
    "Feeding head up",
    "Feeding head down",
    "Licking",
    "Drinking",
    "Lying",
]


class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=42)
            for _, group in df.groupby('label')
        ]).reset_index(drop=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row['label']) - 1
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, eval_transform


def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro',
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    per_class_accuracy = []
    for class_idx in range(NUM_CLASSES):
        total = cm[class_idx].sum()
        correct = cm[class_idx, class_idx]
        acc = (correct / total) * 100 if total > 0 else 0.0
        per_class_accuracy.append(acc)

    return macro_f1, per_class_accuracy


def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Behavior Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()


def format_loss(epoch_losses, epoch):
    value = epoch_losses.get(epoch)
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def format_acc(value):
    return f"{value:.2f}%"


def write_results(
    actual_epochs_trained,
    epoch_losses,
    final_train_loss,
    val_f1,
    val_acc,
    test_f1,
    test_acc,
    training_time_mins,
    early_stop_epoch,
):
    early_stop_text = str(early_stop_epoch) if early_stop_epoch is not None else "N/A"
    text = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {format_loss(epoch_losses, 10)}
LOSS AT EPOCH 20: {format_loss(epoch_losses, 20)}
LOSS AT EPOCH 30: {format_loss(epoch_losses, 30)}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL MACRO F1: {val_f1:.6f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {format_acc(val_acc[0])}
  Class 2 (Standing): {format_acc(val_acc[1])}
  Class 3 (Feeding head up): {format_acc(val_acc[2])}
  Class 4 (Feeding head down): {format_acc(val_acc[3])}
  Class 5 (Licking): {format_acc(val_acc[4])}
  Class 6 (Drinking): {format_acc(val_acc[5])}
  Class 7 (Lying): {format_acc(val_acc[6])}
TEST MACRO F1: {test_f1:.6f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {format_acc(test_acc[0])}
  Class 2 (Standing): {format_acc(test_acc[1])}
  Class 3 (Feeding head up): {format_acc(test_acc[2])}
  Class 4 (Feeding head down): {format_acc(test_acc[3])}
  Class 5 (Licking): {format_acc(test_acc[4])}
  Class 6 (Drinking): {format_acc(test_acc[5])}
  Class 7 (Lying): {format_acc(test_acc[6])}
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stop_text}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


def main():
    start_time = time.time()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    set_seed(RANDOM_SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_transform, eval_transform = build_transforms()

    train_dataset = BehaviorDataset(CSV_PATH, "train", train_transform)
    val_dataset = BehaviorDataset(CSV_PATH, "val", eval_transform)
    test_dataset = BehaviorDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True)
    val_loader = build_loader(val_dataset, shuffle=False)
    test_loader = build_loader(test_dataset, shuffle=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device)
    model = model.to(device, non_blocking=True)

    criterion = FocalLoss(gamma=2, alpha=0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    epochs_no_improve = 0
    best_val_f1 = -1.0
    train_losses = []
    epoch_losses = {}
    early_stop_epoch = None
    actual_epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/30"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)
        epoch_losses[epoch] = train_loss

        val_f1, _ = evaluate(model, val_loader, device, "Validating")
        scheduler.step()

        actual_epochs_trained = epoch

        print(f"Epoch {epoch}/30 | Loss: {train_loss:.6f} | Val Macro F1: {val_f1:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                early_stop_epoch = epoch
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                break

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_f1, val_acc = evaluate(model, val_loader, device, "Validating")
    test_f1, test_acc = evaluate(model, test_loader, device, "Testing")

    save_loss_curve(train_losses)

    training_time_mins = (time.time() - start_time) / 60
    final_train_loss = train_losses[-1]

    write_results(
        actual_epochs_trained,
        epoch_losses,
        final_train_loss,
        val_f1,
        val_acc,
        test_f1,
        test_acc,
        training_time_mins,
        early_stop_epoch,
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
