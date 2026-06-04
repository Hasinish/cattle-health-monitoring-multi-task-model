import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes

import os
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration
PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "ResNet18-LSTM"
MODEL_NAME = "resnet18"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_loss_curve.png"

BATCH_SIZE = 16  # Increased to leverage 16GB VRAM (processes 16 * 20 = 320 frames per batch)
NUM_WORKERS = 2  # Low workers to be safe with memory and deadlocks
MAX_EPOCHS = 15
RANDOM_SEED = 42
HIDDEN_DIM = 64

class VideoSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        # Get unique video IDs in this split
        self.video_ids = sorted(df['cow_id'].unique())
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        v_id = self.video_ids[idx]
        v_df = self.df[self.df['cow_id'] == v_id].sort_values('image_path').reset_index(drop=True)
        
        frames = []
        label = float(v_df.iloc[0]['label'])
        
        # Sub-sample exactly 20 frames evenly spaced to prevent OOM and redundancy
        total_frames = len(v_df)
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
        
        for idx_to_load in indices:
            img_path = v_df.iloc[idx_to_load]['image_path']
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform is not None:
                # Note: using the same transform across the sequence
                # For basic normalizations this is fine. For spatial augs, 
                # using seed ensures same crop/flip is applied to the entire clip.
                state = random.getstate()
                image = self.transform(image=image)["image"]
                random.setstate(state)
                
            frames.append(image)
            
        frames_tensor = torch.stack(frames)  # shape: (20, 3, 224, 224)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        # Extract features without classification head
        backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
            
        self.backbone = backbone
        # LSTM processes the frame feature vectors
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        # Flatten batch and seq dimensions to run through backbone
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # LSTM output
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # Take the final output step
        last_step_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Output logit
        logits = self.classifier(last_step_out)  # (batch_size, 1)
        return logits

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
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
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
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
                probs = torch.sigmoid(outputs).view(-1)
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
    plt.title("Spatiotemporal Lameness Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def write_results(actual_epochs_trained, final_train_loss, val_auc, val_acc, val_f1, val_class_acc, test_auc, test_acc, test_f1, test_class_acc, training_time_mins):
    text = f"""---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CattleLameness (20 frames video sequences)
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

    train_dataset = VideoSequenceDataset(CSV_PATH, "train", train_transform)
    val_dataset = VideoSequenceDataset(CSV_PATH, "val", eval_transform)
    test_dataset = VideoSequenceDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples (videos): {len(train_dataset)}")
    print(f"Val samples (videos): {len(val_dataset)}")
    print(f"Test samples (videos): {len(test_dataset)}")

    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device, non_blocking=True)

    # Freeze backbone parameters to prevent overfitting on the small dataset
    # We will only train the LSTM and classification head
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
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
                loss = criterion(outputs.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)

        val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Validating")
        scheduler.step()

        print(f"Epoch {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f} | Val Acc: {val_acc * 100:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Evaluating Best Val")
    test_auc, test_acc, test_f1, test_class_acc = evaluate_model(model, test_loader, device, "Testing Best Val")

    save_loss_curve(train_losses)
    training_time_mins = (time.time() - start_time) / 60

    write_results(
        MAX_EPOCHS,
        train_losses[-1],
        val_auc,
        val_acc,
        val_f1,
        val_class_acc,
        test_auc,
        test_acc,
        test_f1,
        test_class_acc,
        training_time_mins
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
