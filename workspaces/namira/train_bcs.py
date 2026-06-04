import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# --- Configuration & Seed ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CBAM Architecture ---
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

# --- Model Definitions ---
class MultiTaskBCSModel(nn.Module):
    def __init__(self):
        super().__init__()
        # FIX APPLIED HERE: num_classes=0 explicitly removes the classifier head
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0, global_pool='')
        
        # Calculate feature map channels
        dummy_tensor = torch.randn(1, 3, 224, 224)
        out = self.backbone(dummy_tensor)
        in_channels = out.shape[1]
        
        self.cbam = CBAM(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def coral_loss(logits, labels, num_classes=5):
    sets = []
    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)
    labels_stacked = torch.stack(sets, dim=1).to(device)
    return F.binary_cross_entropy_with_logits(logits, labels_stacked)

# --- Dataset Handler ---
class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_mapping, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'label']
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        mapped_label = self.label_mapping[label]
        return img, torch.tensor(mapped_label, dtype=torch.long)

# --- Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Training Loop Engine ---
def run_training_cycle(dataset_name, csv_path, label_mapping, best_ckpt_path):
    print(f"\n======================================")
    print(f" INITIALIZING TRAINING: {dataset_name.upper()}")
    print(f"======================================")
    
    train_ds = BCSDataset(csv_path, 'train', label_mapping, train_transforms)
    val_ds = BCSDataset(csv_path, 'val', label_mapping, val_test_transforms)
    test_ds = BCSDataset(csv_path, 'test', label_mapping, val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = MultiTaskBCSModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30
    best_val_mae = float('inf')
    loss_history = []
    tracker = {'loss_10': 0, 'loss_20': 0, 'loss_30': 0}
    
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [{dataset_name}]")
        for imgs, labels in progress:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(imgs)
            loss = coral_loss(logits, labels, 5)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            progress.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        epoch_loss = running_loss / len(train_ds)
        loss_history.append((epoch, epoch_loss))
        
        if epoch in [10, 20, 30]:
            tracker[f'loss_{epoch}'] = epoch_loss

        # Validation Step
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        val_mae = np.mean(np.abs(val_preds - val_labels))
        val_acc_0 = np.mean(val_preds == val_labels) * 100
        val_acc_1 = np.mean(np.abs(val_preds - val_labels) <= 1) * 100
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_ckpt_path)

    # Test Evaluation (using best model)
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    test_mae = np.mean(np.abs(test_preds - test_labels))
    test_acc_0 = np.mean(test_preds == test_labels) * 100
    test_acc_1 = np.mean(np.abs(test_preds - test_labels) <= 1) * 100
    
    total_mins = (time.time() - start_time) / 60

    metrics = {
        'epochs': epochs,
        'loss_10': tracker['loss_10'],
        'loss_20': tracker['loss_20'],
        'loss_30': tracker['loss_30'],
        'val_mae': best_val_mae,
        'val_acc_0': val_acc_0,
        'val_acc_1': val_acc_1,
        'test_mae': test_mae,
        'test_acc_0': test_acc_0,
        'test_acc_1': test_acc_1,
        'time': total_mins,
        'history': loss_history
    }
    
    return metrics

if __name__ == '__main__':
    # Workspace Config
    workspace = r"D:\T25301094 P2\workspaces\namira"
    
    # Dryad Config
    dryad_csv = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
    dryad_map = {2:0, 3:1, 4:2, 5:3, 6:4}
    dryad_ckpt = os.path.join(workspace, "dryad_bcs_best.pth")
    
    # ScienceDB Config
    sciencedb_csv = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"
    sciencedb_map = {3.25:0, 3.5:1, 3.75:2, 4.0:3, 4.25:4}
    sciencedb_ckpt = os.path.join(workspace, "sciencedb_bcs_best.pth")

    dryad_metrics = run_training_cycle("Dryad", dryad_csv, dryad_map, dryad_ckpt)
    science_metrics = run_training_cycle("ScienceDB", sciencedb_csv, sciencedb_map, sciencedb_ckpt)

    # --- Generate Loss Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in dryad_metrics['history']], [x[1] for x in dryad_metrics['history']], label='Dryad Train Loss', marker='o')
    plt.plot([x[0] for x in science_metrics['history']], [x[1] for x in science_metrics['history']], label='ScienceDB Train Loss', marker='s')
    plt.title('CORAL Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(workspace, "bcs_loss_curve.png"))
    print("\nLoss curve saved.")

    # --- Generate Context 3 Report ---
    report_content = f"""---CONTEXT 3 BCS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small

DATASET: Dryad
EPOCHS TRAINED: {dryad_metrics['epochs']}
LOSS AT EPOCH 10: {dryad_metrics['loss_10']:.4f}
LOSS AT EPOCH 20: {dryad_metrics['loss_20']:.4f}
LOSS AT EPOCH 30: {dryad_metrics['loss_30']:.4f}
FINAL TRAIN LOSS: {dryad_metrics['loss_30']:.4f}
VAL MAE: {dryad_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {dryad_metrics['val_acc_0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {dryad_metrics['val_acc_1']:.2f}%
TEST MAE: {dryad_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {dryad_metrics['test_acc_0']:.2f}%
TEST ACCURACY +-1: {dryad_metrics['test_acc_1']:.2f}%
CHECKPOINT PATH: {dryad_ckpt}
TRAINING TIME (mins): {dryad_metrics['time']:.2f}
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: {science_metrics['epochs']}
LOSS AT EPOCH 10: {science_metrics['loss_10']:.4f}
LOSS AT EPOCH 20: {science_metrics['loss_20']:.4f}
LOSS AT EPOCH 30: {science_metrics['loss_30']:.4f}
FINAL TRAIN LOSS: {science_metrics['loss_30']:.4f}
VAL MAE: {science_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {science_metrics['val_acc_0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {science_metrics['val_acc_1']:.2f}%
TEST MAE: {science_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {science_metrics['test_acc_0']:.2f}%
TEST ACCURACY +-1: {science_metrics['test_acc_1']:.2f}%
CHECKPOINT PATH: {sciencedb_ckpt}
TRAINING TIME (mins): {science_metrics['time']:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    
    with open(os.path.join(workspace, "bcs_results.txt"), "w") as f:
        f.write(report_content)
    print("Context 3 report generated successfully.")