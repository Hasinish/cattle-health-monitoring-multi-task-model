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

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# GPU Optimization settings
torch.backends.cudnn.benchmark = True

class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        
        # Mandatory sampling code to fix class imbalance and make training fast
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
        img_path = row['image_path']
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = int(row['label']) - 1
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# CBAM Attention Modules
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

# Complete Model Architecture
class BehaviorNetwork(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=7):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        self.backbone = self.backbone.to(device)
        
        # Dynamically detect feature dimensions
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, 3, 224, 224).to(device))
            feature_dim = dummy.shape[1]
            
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Focal Loss Module
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

def evaluate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(7):
        class_total = np.sum(cm[i, :])
        if class_total > 0:
            acc = (cm[i, i] / class_total) * 100.0
        else:
            acc = 0.0
        per_class_acc.append(acc)
        
    return macro_f1, per_class_acc

def main():
    # Setup directories
    workspace_dir = r"D:\T25301094 P2\workspaces\hasin"
    os.makedirs(workspace_dir, exist_ok=True)
    
    csv_path = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
    checkpoint_path = os.path.join(workspace_dir, "behavior_best.pth")
    results_txt_path = os.path.join(workspace_dir, "behavior_results.txt")
    loss_curve_path = os.path.join(workspace_dir, "behavior_loss_curve.png")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define Transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Initialize Datasets and Loaders
    train_dataset = BehaviorDataset(csv_path, 'train', transform=train_transform)
    val_dataset = BehaviorDataset(csv_path, 'val', transform=val_test_transform)
    test_dataset = BehaviorDataset(csv_path, 'test', transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=False, prefetch_factor=2
    )
    
    # Initialize Framework Components
    model = BehaviorNetwork(model_name='resnet18', num_classes=7).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    # Tracking variables
    MAX_EPOCHS = 30
    PATIENCE = 10
    epochs_no_improve = 0
    best_val_f1 = -1.0
    actual_epochs_trained = 0
    early_stopping_epoch = "N/A"
    
    epoch_losses = []
    loss_milestones = {10: "N/A", 20: "N/A", 30: "N/A"}
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
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
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        
        # Track specific milestone losses
        if epoch in loss_milestones:
            loss_milestones[epoch] = f"{epoch_loss:.4f}"
            
        # Evaluate Validation metrics
        val_f1, val_per_class = evaluate_metrics(model, val_loader, device)
        print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {epoch_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        # Early stopping verification block
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
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
    final_train_loss = f"{epoch_losses[-1]:.4f}"
    
    # Plot training loss curve
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Behavior Training Loss Curve')
    plt.legend()
    plt.savefig(loss_curve_path)
    plt.close()
    
    # Evaluate best checkpoint on Val and Test sets
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    final_val_f1, final_val_per_class = evaluate_metrics(model, val_loader, device)
    test_f1, test_per_class = evaluate_metrics(model, test_loader, device)
    
    # Auto-generate Context 3 Report file
    report_content = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
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
CHECKPOINT PATH: {checkpoint_path}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stopping_epoch}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(results_txt_path, 'w') as f:
        f.write(report_content)

if __name__ == '__main__':
    main()