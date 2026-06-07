import os
import csv
import time
import cv2
import timm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# Configurations
PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\CBVD-5\cbvd_cropped_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\cbvd_behavior_results.txt"

NUM_CLASSES = 7
BATCH_SIZE = 64
NUM_WORKERS = 0  # Safe for Windows
RANDOM_SEED = 42

# Mapped classes of interest in CBVD-5:
#   1 -> Class 2 (Standing)
#   3 -> Class 4 (Feeding head down)
#   5 -> Class 6 (Drinking)
#   6 -> Class 7 (Lying)
MAPPED_CLASSES = [1, 3, 5, 6]

class CBVDDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
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

# Attention layers (matching train_behavior.py)
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

class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Set up transform
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset = CBVDDataset(CSV_PATH, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Total CBVD-5 evaluation samples: {len(dataset)}")

    # Load Model
    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device)
    model = model.to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # Evaluate
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating CBVD-5"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    # Compute metrics on the MAPPED_CLASSES only to keep F1 accurate
    # Note: If model predicts a class outside MAPPED_CLASSES, it is treated as a false positive
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro',
        labels=MAPPED_CLASSES,
        zero_division=0
    )
    
    # Calculate confusion matrix on MAPPED_CLASSES
    # We map labels to a 4x4 matrix specifically for the 4 classes
    cm = confusion_matrix(all_labels, all_preds, labels=MAPPED_CLASSES)

    per_class_accuracy = []
    for idx, class_label in enumerate(MAPPED_CLASSES):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        acc = (correct / total) * 100 if total > 0 else 0.0
        per_class_accuracy.append(acc)

    # Output text results
    text = f"""---CONTEXT 3 CBVD-5 EVALUATION---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CBVD-5 (2000 balanced images)
VAL MACRO F1 ON CBVD-5: {macro_f1:.6f}
PER-CLASS ACCURACY ON CBVD-5:
  Class 2 (Standing): {per_class_accuracy[0]:.2f}%
  Class 4 (Feeding head down): {per_class_accuracy[1]:.2f}%
  Class 6 (Drinking): {per_class_accuracy[2]:.2f}%
  Class 7 (Lying): {per_class_accuracy[3]:.2f}%
TRAINING TIME (mins): N/A (Evaluation Only)
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n" + text)
    print(f"\nSaved results to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
