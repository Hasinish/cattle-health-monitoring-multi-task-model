import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import os
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)

import timm
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration
PERSON_NAME = "Namira"
BASE_MODEL_DISPLAY = "MobileNetV3-Small"
MODEL_NAME = "mobilenetv3_small_100"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\namira"
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_cropped_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\namira\lameness_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\namira\lameness_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\namira\lameness_loss_curve.png"

BATCH_SIZE = 512
NUM_WORKERS = 8
MAX_EPOCHS = 10
RANDOM_SEED = 42

class LamenessDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
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
        label = float(row['label'])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + max_).view(b, c, 1, 1)

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

class LamenessModel(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, 1)

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

def build_loader(dataset, shuffle, persistent=True):
    actual_persistent = persistent if NUM_WORKERS > 0 else False
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=actual_persistent,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

def evaluate_model(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze(1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    normal_acc = (cm[0, 0] / cm[0].sum() * 100) if cm[0].sum() > 0 else 0.0
    lame_acc = (cm[1, 1] / cm[1].sum() * 100) if cm[1].sum() > 0 else 0.0
    return auc, acc, f1, [normal_acc, lame_acc]

def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Lameness Training Loss Curve — {PERSON_NAME} ({BASE_MODEL_DISPLAY})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def write_results(actual_epochs_trained, final_train_loss, val_auc, val_acc, val_f1, val_class_acc, test_auc, test_acc, test_f1, test_class_acc, training_time_mins):
    text = f"""---CONTEXT 3 LAMENESS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: {actual_epochs_trained}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL AUC: {val_auc:.6f}
VAL ACCURACY: {val_acc * 100:.2f}%
VAL F1 SCORE: {val_f1:.6f}
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): {val_class_acc[0]:.2f}%
  Class 1 (Lame): {val_class_acc[1]:.2f}%
TEST AUC: {test_auc:.6f}
TEST ACCURACY: {test_acc * 100:.2f}%
TEST F1 SCORE: {test_f1:.6f}
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): {test_class_acc[0]:.2f}%
  Class 1 (Lame): {test_class_acc[1]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
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
    train_dataset = LamenessDataset(CSV_PATH, "train", train_transform)
    val_dataset = LamenessDataset(CSV_PATH, "val", eval_transform)
    test_dataset = LamenessDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME} | Model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = LamenessModel(MODEL_NAME, device).to(device, non_blocking=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = -1.0
    train_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)
        val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Validating")
        scheduler.step()
        print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f} | Val Acc: {val_acc*100:.2f}%")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Best Val Eval")
    test_auc, test_acc, test_f1, test_class_acc = evaluate_model(model, test_loader, device, "Test Eval")
    save_loss_curve(train_losses)
    training_time_mins = (time.time() - start_time) / 60
    write_results(MAX_EPOCHS, train_losses[-1], val_auc, val_acc, val_f1, val_class_acc, test_auc, test_acc, test_f1, test_class_acc, training_time_mins)
    print(f"Done. Checkpoint: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
