import os
import time
import random
import numpy as np
import pandas as pd

import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PERSON_NAME = "Shouvik"
BASE_MODEL_DISPLAY = "DenseNet121"
MODEL_NAME = "densenet121"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\shouvik"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "behavior_best.pth")
RESULTS_PATH = os.path.join(WORKSPACE_DIR, "behavior_results.txt")
LOSS_CURVE_PATH = os.path.join(WORKSPACE_DIR, "behavior_loss_curve.png")

NUM_CLASSES = 7
BATCH_SIZE = 128
NUM_WORKERS = 4
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

# ============================================================
# REPRODUCIBILITY & SETUP
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

# ============================================================
# DATASET
# ============================================================
class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        # Mandatory sampling to fix class imbalance and make training fast
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=RANDOM_SEED)
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
        label = int(row['label']) - 1  # 1-7 to 0-6
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

# ============================================================
# CBAM ATTENTION MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(in_channels // reduction, 1), in_channels)
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
# FOCAL LOSS
# ============================================================
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

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
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

# ============================================================
# METRICS EVALUATION
# ============================================================
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
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    per_class_acc = []
    for i in range(NUM_CLASSES):
        class_total = np.sum(cm[i, :])
        acc = (cm[i, i] / class_total * 100.0) if class_total > 0 else 0.0
        per_class_acc.append(acc)
        
    return macro_f1, per_class_acc

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
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
    
    # Datasets & Loaders
    train_ds = BehaviorDataset(CSV_PATH, "train", train_transform)
    val_ds = BehaviorDataset(CSV_PATH, "val", eval_transform)
    test_ds = BehaviorDataset(CSV_PATH, "test", eval_transform)
    
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_ds, shuffle=True, persistent_workers=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=False, **loader_kwargs)
    
    # Build Model
    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    epochs_no_improve = 0
    actual_epochs_trained = 0
    early_stopping_epoch = "N/A"
    
    epoch_losses = []
    loss_milestones = {10: "N/A", 20: "N/A", 30: "N/A"}
    
    start_time = time.time()
    print("="*60)
    print(f"  BEHAVIOR TRAINING — {PERSON_NAME} — {BASE_MODEL_DISPLAY}")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print("="*60)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{MAX_EPOCHS:02d}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        
        if epoch in loss_milestones:
            loss_milestones[epoch] = f"{epoch_loss:.6f}"
            
        # Validation evaluation
        val_f1, val_accs = evaluate(model, val_loader, device, "Validating")
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Loss: {epoch_loss:.6f} | Val Macro F1: {val_f1:.4f}")
        
        # Checkpoint Saving & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                early_stopping_epoch = str(epoch)
                actual_epochs_trained = epoch
                break
                
        actual_epochs_trained = epoch

    end_time = time.time()
    training_time_mins = (end_time - start_time) / 60.0
    final_train_loss = f"{epoch_losses[-1]:.6f}"
    
    # Save Loss Curve
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{PERSON_NAME} - Behavior Training Loss')
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    
    # Load best checkpoint and evaluate on Test set
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    final_val_f1, final_val_per_class = evaluate(model, val_loader, device, "Evaluating Val")
    test_f1, test_per_class = evaluate(model, test_loader, device, "Evaluating Test")
    
    # Context 3 Report content
    report_content = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {loss_milestones.get(10, "N/A")}
LOSS AT EPOCH 20: {loss_milestones.get(20, "N/A")}
LOSS AT EPOCH 30: {loss_milestones.get(30, "N/A")}
FINAL TRAIN LOSS: {final_train_loss}
VAL MACRO F1: {final_val_f1:.4f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {final_val_per_class[0]:.2f}%
  Class 2 (Standing): {final_val_per_class[1]:.2f}%
  Class 3 (Feeding head up): {final_val_per_class[2]:.2f}%
  Class 4 (Feeding head down): {final_val_per_class[3]:.2f}%
  Class 5 (Licking): {final_val_per_class[4]:.2f}%
  Class 6 (Drinking): {final_val_per_class[5]:.2f}%
  Class 7 (Lying): {final_val_per_class[6]:.2f}%
TEST MACRO F1: {test_f1:.4f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {test_per_class[0]:.2f}%
  Class 2 (Standing): {test_per_class[1]:.2f}%
  Class 3 (Feeding head up): {test_per_class[2]:.2f}%
  Class 4 (Feeding head down): {test_per_class[3]:.2f}%
  Class 5 (Licking): {test_per_class[4]:.2f}%
  Class 6 (Drinking): {test_per_class[5]:.2f}%
  Class 7 (Lying): {test_per_class[6]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stopping_epoch}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    with open(RESULTS_PATH, "w") as f:
        f.write(report_content)
        
    print(f"Training complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
