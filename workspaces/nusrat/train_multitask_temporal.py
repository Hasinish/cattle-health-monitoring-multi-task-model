import os
import csv
import cv2
import time
import random
import warnings
from collections import defaultdict
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score

# ============================================================
# CONFIG
# ============================================================
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)  # Prevent OpenCV multithreading deadlocks in DataLoader

SEED = 42
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")

BCS_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")
BEHAVIOR_CSV = os.path.join(BASE_DIR, "datasets", "behavior", "behavior_index.csv")
LAMENESS_CSV = os.path.join(BASE_DIR, "datasets", "lameness", "lameness_cropped_index.csv")
ID_CSV = os.path.join(BASE_DIR, "datasets", "id", "id_index.csv")

CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "multitask_temporal_best.pth")
RESULTS_PATH = os.path.join(WORKSPACE_DIR, "multitask_temporal_results.txt")

NUM_WORKERS = 0  # CRITICAL for Windows to avoid deadlocks
BATCH_SIZE = 8   # Sequence models require smaller batch sizes to prevent GPU OOM
PHASE_EPOCHS = 5  # Epochs for individual head training
JOINT_EPOCHS = 10 # Epochs for joint fine-tuning
LR_HEADS = 1e-3
LR_JOINT = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Weights for Joint Fine-Tuning
W_BCS = 0.35
W_BEH = 0.35
W_LAM = 0.15
W_ID = 0.15

# ============================================================
# UTILS & TRANSFORMS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

def get_transforms(split):
    if split == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================
# SPATIOTEMPORAL DATASETS
# ============================================================

class BCSSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.samples = []
        bcs_map = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    img_path = row['image_path']
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(BASE_DIR, img_path)
                    label = bcs_map[float(row['label'])]
                    self.samples.append((img_path, label))
                    
        # To balance speed, we sample a representative subset for joint training
        if split == 'train':
            random.seed(SEED)
            random.shuffle(self.samples)
            self.samples = self.samples[:4000]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        # Replicate static image 20 times to form a sequence
        frames = torch.stack([image] * 20)
        return frames, torch.tensor(label, dtype=torch.long)


class IDSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.samples = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    img_path = row['image_path']
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(BASE_DIR, img_path)
                    label = int(row['label'])
                    self.samples.append((img_path, label))
                    
        if split == 'train':
            random.seed(SEED)
            random.shuffle(self.samples)
            self.samples = self.samples[:4000]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        # Replicate static image 20 times to form a sequence
        frames = torch.stack([image] * 20)
        return frames, torch.tensor(label, dtype=torch.long)


class BehaviorSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.sequences = []
        
        # Group frames by their sequence prefix (excluding frame index)
        seq_groups = defaultdict(list)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    img_path = row['image_path']
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(BASE_DIR, img_path)
                    label = int(row['label']) - 1
                    
                    stem = Path(img_path).stem
                    parts = stem.split('_')
                    # Prefix grouping key
                    seq_prefix = "_".join(parts[:-1])
                    frame_idx = int(parts[-1])
                    
                    seq_groups[seq_prefix].append((img_path, label, frame_idx))
                    
        for seq_prefix, items in seq_groups.items():
            items = sorted(items, key=lambda x: x[2])  # Sort by frame index
            self.sequences.append(items)
            
        if split == 'train':
            random.seed(SEED)
            random.shuffle(self.sequences)
            # Cap at 2000 sequence samples for balanced and fast training
            self.sequences = self.sequences[:2000]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        items = self.sequences[idx]
        total_frames = len(items)
        
        # Sample exactly 20 frames uniformly
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
        
        frames = []
        label = items[0][1]
        for idx_to_load in indices:
            img_path = items[idx_to_load][0]
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Missing image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image=image)["image"]
            frames.append(image)
            
        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


class LamenessSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.video_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.video_data.append({
                        'image_path': row['image_path'],
                        'label': float(row['label']),
                        'cow_id': row['cow_id']
                    })
        self.video_ids = sorted(list(set(item['cow_id'] for item in self.video_data)))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        v_id = self.video_ids[idx]
        v_samples = [item for item in self.video_data if item['cow_id'] == v_id]
        v_samples = sorted(v_samples, key=lambda x: x['image_path'])
        
        frames = []
        label = v_samples[0]['label']
        total_frames = len(v_samples)
        
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
        
        for idx_to_load in indices:
            img_path = v_samples[idx_to_load]['image_path']
            if not os.path.isabs(img_path):
                img_path = os.path.join(BASE_DIR, img_path)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Missing image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image=image)["image"]
            frames.append(image)
            
        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)


def build_loader(csv_path, split, task_type, shuffle):
    transform = get_transforms(split)
    if task_type == 'bcs':
        dataset = BCSSequenceDataset(csv_path, split, transform)
    elif task_type == 'behavior':
        dataset = BehaviorSequenceDataset(csv_path, split, transform)
    elif task_type == 'lameness':
        dataset = LamenessSequenceDataset(csv_path, split, transform)
    elif task_type == 'id':
        dataset = IDSequenceDataset(csv_path, split, transform)
        
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

# ============================================================
# CBAM VISUAL ATTENTION
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
# SPATIOTEMPORAL MULTI-TASK MODEL
# ============================================================

class MultiTaskTemporalModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Backbone extracts 2D features per frame
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Sequential/Temporal heads (LSTMs)
        self.behavior_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.behavior_classifier = nn.Linear(hidden_dim, 7)
        
        self.lameness_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.lameness_classifier = nn.Linear(hidden_dim, 1)
        
        # 2D Linear Heads (replicated frame-by-frame)
        self.bcs_head = nn.Linear(feature_dim, 4)
        self.id_head = nn.Linear(feature_dim, 46)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len=20, 3, 224, 224)
        batch_size, seq_len, c, h, w = x.shape
        x_flattened = x.view(batch_size * seq_len, c, h, w)
        
        # Backbone + attention + pooling
        features = self.backbone.forward_features(x_flattened)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to sequence: (batch_size, seq_len, feature_dim)
        seq_features = pooled.view(batch_size, seq_len, -1)
        
        # 1. Behavior prediction (Temporal final step)
        beh_out, _ = self.behavior_lstm(seq_features)
        behavior_logits = self.behavior_classifier(beh_out[:, -1, :])  # (batch_size, 7)
        
        # 2. Lameness prediction (Temporal final step)
        lame_out, _ = self.lameness_lstm(seq_features)
        lameness_logits = self.lameness_classifier(lame_out[:, -1, :])  # (batch_size, 1)
        
        # 3. BCS predictions (Frame-by-frame averaged predictions)
        flat_features = seq_features.view(batch_size * seq_len, -1)
        flat_bcs_logits = self.bcs_head(flat_features)  # (batch * seq_len, 4)
        bcs_logits = flat_bcs_logits.view(batch_size, seq_len, -1).mean(dim=1)  # (batch_size, 4)
        
        # 4. ID predictions (Frame-by-frame averaged predictions during training)
        flat_id_logits = self.id_head(flat_features)  # (batch * seq_len, 46)
        id_logits = flat_id_logits.view(batch_size, seq_len, -1).mean(dim=1)  # (batch_size, 46)
        
        return {
            'bcs': bcs_logits,
            'behavior': behavior_logits,
            'lameness': lameness_logits,
            'id': id_logits
        }

# ============================================================
# LOSSES & METRICS
# ============================================================

def coral_loss(logits, labels, num_classes=5):
    sets = []
    for i in range(num_classes - 1):
        sets.append((labels > i).float())
    labels_stacked = torch.stack(sets, dim=1)
    return F.binary_cross_entropy_with_logits(logits, labels_stacked)

def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

focal_criterion = FocalLoss()
bce_criterion = nn.BCEWithLogitsLoss()
ce_criterion = nn.CrossEntropyLoss()

# ============================================================
# TRAINING LOOPS
# ============================================================

def train_single_task(model, loader, optimizer, task_name, epochs):
    print(f"\n--- Training {task_name.upper()} Head (Backbone Frozen) ---")
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [{task_name}]")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)[task_name]
                
                if task_name == 'bcs':
                    loss = coral_loss(outputs, labels)
                elif task_name == 'behavior':
                    loss = focal_criterion(outputs, labels)
                elif task_name == 'lameness':
                    loss = bce_criterion(outputs.squeeze(-1), labels)
                elif task_name == 'id':
                    loss = ce_criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

def train_joint(model, loaders, optimizer, epochs):
    print(f"\n--- Phase 6: Joint Fine-tuning (Backbone Unfrozen) ---")
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        model.train()
        iters = {k: iter(v) for k, v in loaders.items()}
        running_losses = {k: 0.0 for k in loaders.keys()}
        steps = {k: 0 for k in loaders.keys()}
        
        pbar = tqdm(range(max([len(v) for v in loaders.values()])), desc=f"Epoch {epoch+1}/{epochs} [Joint]")
        
        for _ in pbar:
            optimizer.zero_grad()
            total_loss = 0
            
            for task_name, w in [('bcs', W_BCS), ('behavior', W_BEH), ('lameness', W_LAM), ('id', W_ID)]:
                try:
                    images, labels = next(iters[task_name])
                except StopIteration:
                    iters[task_name] = iter(loaders[task_name])
                    images, labels = next(iters[task_name])
                
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)[task_name]
                    
                    if task_name == 'bcs':
                        loss = coral_loss(outputs, labels)
                    elif task_name == 'behavior':
                        loss = focal_criterion(outputs, labels)
                    elif task_name == 'lameness':
                        loss = bce_criterion(outputs.squeeze(-1), labels)
                    elif task_name == 'id':
                        loss = ce_criterion(outputs, labels)
                    
                    scaled_loss = w * loss
                
                scaler.scale(scaled_loss).backward()
                running_losses[task_name] += loss.item()
                steps[task_name] += 1
                total_loss += (w * loss.item())
                
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'L_joint': f"{total_loss:.4f}"})

@torch.no_grad()
def evaluate_all(model, test_loaders):
    model.eval()
    results = {}
    print("\n--- Evaluating Multi-Task Temporal Model ---")
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Spatiotemporal Multi-Task Model Evaluation Results\n")
        f.write("="*40 + "\n\n")
        
        for task_name, loader in test_loaders.items():
            all_labels, all_preds, all_probs = [], [], []
            for images, labels in tqdm(loader, desc=f"Eval {task_name}"):
                images = images.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)[task_name]
                
                if task_name == 'bcs':
                    preds = coral_predict(outputs).cpu().numpy()
                elif task_name == 'lameness':
                    probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    all_probs.extend(probs.tolist())
                else: # behavior and id
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.tolist())
                
            # Metrics
            if task_name == 'bcs':
                mae = mean_absolute_error(all_labels, all_preds)
                acc = accuracy_score(all_labels, all_preds)
                res_str = f"BCS - MAE: {mae:.4f}, Exact Acc: {acc:.4f}"
            elif task_name == 'behavior':
                f1 = f1_score(all_labels, all_preds, average='macro')
                res_str = f"Behavior - Macro F1: {f1:.4f}"
            elif task_name == 'lameness':
                acc = accuracy_score(all_labels, all_preds)
                auc = roc_auc_score(all_labels, all_probs)
                res_str = f"Lameness - Acc: {acc:.4f}, AUC: {auc:.4f}"
            elif task_name == 'id':
                acc = accuracy_score(all_labels, all_preds)
                res_str = f"Cow ID - Acc: {acc:.4f}"
                
            print(res_str)
            f.write(res_str + "\n")
            results[task_name] = res_str

    print(f"\nResults saved to {RESULTS_PATH}")
    return results

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print(f"Using device: {DEVICE}")
    print("Building Spatiotemporal DataLoaders...")
    train_loaders = {
        'bcs': build_loader(BCS_CSV, 'train', 'bcs', True),
        'behavior': build_loader(BEHAVIOR_CSV, 'train', 'behavior', True),
        'lameness': build_loader(LAMENESS_CSV, 'train', 'lameness', True),
        'id': build_loader(ID_CSV, 'train', 'id', True)
    }
    
    test_loaders = {
        'bcs': build_loader(BCS_CSV, 'test', 'bcs', False),
        'behavior': build_loader(BEHAVIOR_CSV, 'test', 'behavior', False),
        'lameness': build_loader(LAMENESS_CSV, 'test', 'lameness', False),
        'id': build_loader(ID_CSV, 'test', 'id', False)
    }
    
    model = MultiTaskTemporalModel().to(DEVICE)
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Phase 2: BCS
    optimizer = torch.optim.Adam(list(model.bcs_head.parameters()), lr=LR_HEADS)
    train_single_task(model, train_loaders['bcs'], optimizer, 'bcs', PHASE_EPOCHS)
    
    # Phase 3: Behavior
    optimizer = torch.optim.Adam(list(model.behavior_lstm.parameters()) + list(model.behavior_classifier.parameters()), lr=LR_HEADS)
    train_single_task(model, train_loaders['behavior'], optimizer, 'behavior', PHASE_EPOCHS)
    
    # Phase 4: Lameness
    optimizer = torch.optim.Adam(list(model.lameness_lstm.parameters()) + list(model.lameness_classifier.parameters()), lr=LR_HEADS)
    train_single_task(model, train_loaders['lameness'], optimizer, 'lameness', PHASE_EPOCHS)
    
    # Phase 5: ID
    optimizer = torch.optim.Adam(list(model.id_head.parameters()), lr=LR_HEADS)
    train_single_task(model, train_loaders['id'], optimizer, 'id', PHASE_EPOCHS)
    
    # Phase 6: Joint Fine-Tuning
    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_JOINT)
    train_joint(model, train_loaders, optimizer, JOINT_EPOCHS)
    
    # Save Model
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel saved to {CHECKPOINT_PATH}")
    
    # Evaluate
    evaluate_all(model, test_loaders)

if __name__ == "__main__":
    main()
