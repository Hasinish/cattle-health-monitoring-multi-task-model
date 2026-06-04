import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt

# ==========================================
# HYPERPARAMETERS & SEEDING
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
NUM_CLASSES = 5  # 5 ordinal classes for both datasets
NUM_CORAL_OUTPUTS = NUM_CLASSES - 1  # 4 output nodes

# Workspace configurations
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\bithi"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# ==========================================
# CBAM ATTENTION MODULE
# ==========================================
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

# ==========================================
# RESNET-50 BACKBONE + ATTENTION + HEAD
# ==========================================
class BCSModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        # Load backbone with feature maps preserved (global_pool='')
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        in_channels = self.backbone.num_features
        
        self.cbam = CBAM(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, NUM_CORAL_OUTPUTS)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.cbam(features)
        features = self.pool(features).squeeze(-1).squeeze(-1)
        logits = self.fc(features)
        return logits

# ==========================================
# LOSS AND PREDICTION FUNCTIONS
# ==========================================
def coral_loss(logits, labels, num_classes=NUM_CLASSES):
    sets = []
    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)
    labels_stacked = torch.stack(sets, dim=1)
    loss = F.binary_cross_entropy_with_logits(logits, labels_stacked)
    return loss

def predict_class(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)

# ==========================================
# DATASET IMPLEMENTATION
# ==========================================
class CattleBCSDataset(Dataset):
    def __init__(self, csv_path, split, label_mapping, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.label_mapping = label_mapping
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Open and ensure 3 channels
        img = Image.open(img_path).convert('RGB')
        
        raw_label = row['label']
        label = self.label_mapping[raw_label]
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

# ==========================================
# TRAINING AND EVALUATION ENGINE
# ==========================================
def train_and_evaluate(dataset_name, csv_path, label_mapping, checkpoint_name):
    print(f"\n==================================================")
    print(f"STARTING TRAINING ON DATASET: {dataset_name}")
    print(f"==================================================")
    
    # Preprocessing pipelines
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataloaders
    train_dataset = CattleBCSDataset(csv_path, 'train', label_mapping, train_transform)
    val_dataset = CattleBCSDataset(csv_path, 'val', label_mapping, val_test_transform)
    test_dataset = CattleBCSDataset(csv_path, 'test', label_mapping, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = BCSModel(model_name='resnet50', pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_mae = float('inf')
    loss_history = []
    
    # Dictionary to capture progress milestones for reporting
    metrics_report = {
        'loss_ep10': 0.0, 'loss_ep20': 0.0, 'loss_ep30': 0.0, 'final_train_loss': 0.0,
        'val_mae': 0.0, 'val_acc0': 0.0, 'val_acc1': 0.0,
        'test_mae': 0.0, 'test_acc0': 0.0, 'test_acc1': 0.0,
        'time_mins': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # Progress bar setup
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = coral_loss(logits, labels, NUM_CLASSES)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        scheduler.step()
        
        # Capture specific epoch metrics
        if epoch == 10: metrics_report['loss_ep10'] = epoch_loss
        if epoch == 20: metrics_report['loss_ep20'] = epoch_loss
        if epoch == 30: metrics_report['loss_ep30'] = epoch_loss
        metrics_report['final_train_loss'] = epoch_loss
        
        # Validation evaluation loop
        model.eval()
        val_abs_errors, val_exact_matches, val_close_matches = [], [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                preds = predict_class(logits)
                
                abs_err = torch.abs(preds - labels)
                val_abs_errors.extend(abs_err.cpu().numpy())
                val_exact_matches.extend((abs_err == 0).cpu().numpy())
                val_close_matches.extend((abs_err <= 1).cpu().numpy())
                
        val_mae = np.mean(val_abs_errors)
        val_acc0 = np.mean(val_exact_matches) * 100
        val_acc1 = np.mean(val_close_matches) * 100
        
        print(f"Epoch {epoch} Summary -> Train Loss: {epoch_loss:.4f} | Val MAE: {val_mae:.4f} | Val Acc+-0: {val_acc0:.2f}%")
        
        # Save best model logic
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            metrics_report['val_mae'] = val_mae
            metrics_report['val_acc0'] = val_acc0
            metrics_report['val_acc1'] = val_acc1
            torch.save(model.state_dict(), os.path.join(WORKSPACE_DIR, checkpoint_name))
            
    end_time = time.time()
    metrics_report['time_mins'] = (end_time - start_time) / 60
    
    # Final Testing Evaluation using the saved best weights
    print(f"\nEvaluating final checkpoint on Test Split...")
    best_model_path = os.path.join(WORKSPACE_DIR, checkpoint_name)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_abs_errors, test_exact_matches, test_close_matches = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = predict_class(logits)
            
            abs_err = torch.abs(preds - labels)
            test_abs_errors.extend(abs_err.cpu().numpy())
            test_exact_matches.extend((abs_err == 0).cpu().numpy())
            test_close_matches.extend((abs_err <= 1).cpu().numpy())
            
    metrics_report['test_mae'] = np.mean(test_abs_errors)
    metrics_report['test_acc0'] = np.mean(test_exact_matches) * 100
    metrics_report['test_acc1'] = np.mean(test_close_matches) * 100
    
    return metrics_report, loss_history

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # RUN 1: Dryad Dataset
    dryad_csv = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
    dryad_mapping = {2:0, 3:1, 4:2, 5:3, 6:4}
    dryad_metrics, dryad_loss = train_and_evaluate("Dryad", dryad_csv, dryad_mapping, "dryad_bcs_best.pth")
    
    # RUN 2: ScienceDB Dataset
    sciencedb_csv = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"
    sciencedb_mapping = {3.25:0, 3.5:1, 3.75:2, 4.0:3, 4.25:4}
    sciencedb_metrics, sciencedb_loss = train_and_evaluate("ScienceDB", sciencedb_csv, sciencedb_mapping, "sciencedb_bcs_best.pth")
    
    # 1. Plot and save Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), dryad_loss, label='Dryad Train Loss', color='blue', linestyle='-')
    plt.plot(range(1, EPOCHS+1), sciencedb_loss, label='ScienceDB Train Loss', color='orange', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('BCS Training Loss Curves (ResNet-50)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(WORKSPACE_DIR, "bcs_loss_curve.png"))
    plt.close()
    
    # 2. Compile Context 3 Report String
    report_output = f"""---CONTEXT 3 BCS---
PERSON NAME: Bithi
BASE MODEL: ResNet-50

DATASET: Dryad
EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {dryad_metrics['loss_ep10']:.4f}
LOSS AT EPOCH 20: {dryad_metrics['loss_ep20']:.4f}
LOSS AT EPOCH 30: {dryad_metrics['loss_ep30']:.4f}
FINAL TRAIN LOSS: {dryad_metrics['final_train_loss']:.4f}
VAL MAE: {dryad_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {dryad_metrics['val_acc0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {dryad_metrics['val_acc1']:.2f}%
TEST MAE: {dryad_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {dryad_metrics['test_acc0']:.2f}%
TEST ACCURACY +-1: {dryad_metrics['test_acc1']:.2f}%
CHECKPOINT PATH: D:\\T25301094 P2\\workspaces\\bithi\\dryad_bcs_best.pth
TRAINING TIME (mins): {dryad_metrics['time_mins']:.2f}
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {sciencedb_metrics['loss_ep10']:.4f}
LOSS AT EPOCH 20: {sciencedb_metrics['loss_ep20']:.4f}
LOSS AT EPOCH 30: {sciencedb_metrics['loss_ep30']:.4f}
FINAL TRAIN LOSS: {sciencedb_metrics['final_train_loss']:.4f}
VAL MAE: {sciencedb_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {sciencedb_metrics['val_acc0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {sciencedb_metrics['val_acc1']:.2f}%
TEST MAE: {sciencedb_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {sciencedb_metrics['test_acc0']:.2f}%
TEST ACCURACY +-1: {sciencedb_metrics['test_acc1']:.2f}%
CHECKPOINT PATH: D:\\T25301094 P2\\workspaces\\bithi\\sciencedb_bcs_best.pth
TRAINING TIME (mins): {sciencedb_metrics['time_mins']:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""
    
    # Save text report to file
    with open(os.path.join(WORKSPACE_DIR, "bcs_results.txt"), "w") as f:
        f.write(report_output)
        
    print(f"\nExecution Finished! Results and loss curves successfully compiled in {WORKSPACE_DIR}")