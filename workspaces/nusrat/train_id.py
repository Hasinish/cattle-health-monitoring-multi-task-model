import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes

import os
import csv
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import timm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\id\id_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\id_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\id_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\id_loss_curve.png"

NUM_CLASSES = 46
BATCH_SIZE = 64
NUM_WORKERS = 0  # Set to 0 to prevent Windows multiprocessing deadlocks and worker unexpected exit crashes
MAX_EPOCHS = 10
RANDOM_SEED = 42

class CowIDDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.samples.append({
                        'image_path': row['image_path'],
                        'label': int(row['label']),
                        'cow_id': row['cow_id']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = item['label']
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

class CowIDModel(nn.Module):
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

def build_loader(dataset, shuffle, persistent=True):
    # If NUM_WORKERS is 0, persistent_workers must be False
    actual_persistent = persistent if NUM_WORKERS > 0 else False
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=actual_persistent,
    )

def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return acc

def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Cow ID Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def format_loss(epoch_losses, epoch):
    value = epoch_losses.get(epoch)
    if value is None:
        return "N/A"
    return f"{value:.6f}"

def write_results(
    actual_epochs_trained,
    epoch_losses,
    final_train_loss,
    val_acc,
    test_acc,
    training_time_mins,
):
    text = f"""---CONTEXT 3 ID---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: OpenCows2020
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {format_loss(epoch_losses, 10)}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL TOP-1 ACCURACY: {val_acc * 100:.2f}%
TEST TOP-1 ACCURACY: {test_acc * 100:.2f}%
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

    train_dataset = CowIDDataset(CSV_PATH, "train", train_transform)
    val_dataset = CowIDDataset(CSV_PATH, "val", eval_transform)
    test_dataset = CowIDDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = CowIDModel(MODEL_NAME, NUM_CLASSES, device)
    model = model.to(device, non_blocking=True)

    # Freeze backbone parameters to prevent overfitting
    # We will only train CBAM and the classifier head
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = -1.0
    train_losses = []
    epoch_losses = {}
    actual_epochs_trained = 0

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

        val_acc = evaluate(model, val_loader, device, "Validating")
        scheduler.step()

        actual_epochs_trained = epoch

        print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {train_loss:.6f} | Val Top-1 Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_acc = evaluate(model, val_loader, device, "Validating Best Val")
    test_acc = evaluate(model, test_loader, device, "Testing Best Val")

    save_loss_curve(train_losses)

    training_time_mins = (time.time() - start_time) / 60
    final_train_loss = train_losses[-1]

    write_results(
        actual_epochs_trained,
        epoch_losses,
        final_train_loss,
        val_acc,
        test_acc,
        training_time_mins,
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
