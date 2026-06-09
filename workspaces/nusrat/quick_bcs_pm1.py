"""Quick script to compute BCS ±1 accuracy from saved multitask checkpoint."""
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
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "multitask_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Minimal Dataset ────────────────────────────────────────────
class BCSDataset(Dataset):
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
        return img, torch.tensor(label, dtype=torch.long)

# ── Minimal Model (same as train_multitask.py) ─────────────────
class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
        h = max(c//r, 1)
        self.fc = nn.Sequential(nn.Linear(c,h), nn.ReLU(), nn.Linear(h,c))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.fc(self.avg(x).squeeze(-1).squeeze(-1))
        m = self.fc(self.max(x).squeeze(-1).squeeze(-1))
        return x * self.sig(a+m).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3); self.sig = nn.Sigmoid()
    def forward(self, x):
        a = torch.mean(x,1,keepdim=True); m,_ = torch.max(x,1,keepdim=True)
        return x * self.sig(self.conv(torch.cat([a,m],1)))

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__(); self.ca = ChannelAttention(c); self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="")
        fd = self.backbone.num_features
        self.cbam = CBAM(fd); self.pool = nn.AdaptiveAvgPool2d(1)
        self.bcs_head = nn.Linear(fd, 4)   # CORAL: 4 = num_classes-1
        self.behavior_head = nn.Linear(fd, 7)
        self.lameness_head = nn.Linear(fd, 1)
        self.id_head = nn.Linear(fd, 46)
    def forward(self, x):
        f = self.pool(self.cbam(self.backbone.forward_features(x))).flatten(1)
        return {'bcs': self.bcs_head(f), 'behavior': self.behavior_head(f),
                'lameness': self.lameness_head(f), 'id': self.id_head(f)}

if __name__ == '__main__':
    transform = A.Compose([A.Resize(224,224),
                           A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                           ToTensorV2()])

    loader = DataLoader(BCSDataset(BCS_CSV, 'test', transform),
                        batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = MultiTaskModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()

    # ── Eval ───────────────────────────────────────────────────────
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="BCS Eval"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)['bcs']
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    mae      = np.mean(np.abs(all_preds - all_labels))
    exact    = np.mean(all_preds == all_labels)
    pm1      = np.mean(np.abs(all_preds - all_labels) <= 1)

    print(f"\n=== BCS ±1 Accuracy (Multi-Task Model) ===")
    print(f"  MAE        : {mae:.4f}")
    print(f"  Exact Acc  : {exact*100:.2f}%")
    print(f"  ±1 Acc     : {pm1*100:.2f}%")
