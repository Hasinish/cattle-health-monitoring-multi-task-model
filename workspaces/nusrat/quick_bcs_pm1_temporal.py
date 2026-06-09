"""Quick script to compute BCS ±1 accuracy from saved multitask temporal checkpoint."""
import os, csv, cv2, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")
BCS_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "multitask_temporal_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Minimal Dataset ────────────────────────────────────────────
class BCSTemporalDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.samples = []
        bcs_map = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4}
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['split'] == split:
                    p = row['image_path']
                    if not os.path.isabs(p):
                        p = os.path.join(BASE_DIR, p)
                    self.samples.append((p, bcs_map[float(row['label'])]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        # Replicate static image 20 times to form a sequence
        frames = torch.stack([img] * 20)
        return frames, torch.tensor(label, dtype=torch.long)

# ── Minimal Model (same as train_multitask_temporal.py) ────────
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

    def forward(self, x): return self.sa(self.ca(x))

class MultiTaskTemporalModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.behavior_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.behavior_classifier = nn.Linear(hidden_dim, 7)
        
        self.lameness_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.lameness_classifier = nn.Linear(hidden_dim, 1)
        
        self.bcs_head = nn.Linear(feature_dim, 4)
        self.id_head = nn.Linear(feature_dim, 46)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x_flattened = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone.forward_features(x_flattened)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)
        seq_features = pooled.view(batch_size, seq_len, -1)
        
        # We only need bcs for this evaluation
        flat_features = seq_features.view(batch_size * seq_len, -1)
        flat_bcs_logits = self.bcs_head(flat_features)
        bcs_logits = flat_bcs_logits.view(batch_size, seq_len, -1).mean(dim=1)
        return bcs_logits

if __name__ == '__main__':
    transform = A.Compose([A.Resize(224,224),
                           A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                           ToTensorV2()])

    # Use a batch size of 16 to be safe on memory/OOM
    loader = DataLoader(BCSTemporalDataset(BCS_CSV, 'test', transform),
                        batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    model = MultiTaskTemporalModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # ── Eval ───────────────────────────────────────────────────────
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="BCS Temporal Eval"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    mae      = np.mean(np.abs(all_preds - all_labels))
    exact    = np.mean(all_preds == all_labels)
    pm1      = np.mean(np.abs(all_preds - all_labels) <= 1)

    print(f"\n=== BCS ±1 Accuracy (Spatiotemporal Multi-Task Model) ===")
    print(f"  MAE        : {mae:.4f}")
    print(f"  Exact Acc  : {exact*100:.2f}%")
    print(f"  ±1 Acc     : {pm1*100:.2f}%")
