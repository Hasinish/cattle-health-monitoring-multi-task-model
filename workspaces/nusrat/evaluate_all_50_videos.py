import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Configuration
MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 64
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_cropped_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth"
OUTPUT_REPORT_PATH = r"D:\T25301094 P2\workspaces\nusrat\all_50_videos_evaluation.txt"
DECISION_THRESHOLD = 0.50  # Calibrated decision threshold for cropped model

class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        last_step_out = lstm_out[:, -1, :]
        logits = self.classifier(last_step_out)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")

    # Load dataset index using standard csv to avoid pyarrow crash
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset index not found at: {CSV_PATH}")
    
    import csv
    video_data = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_data.append({
                'image_path': row['image_path'],
                'label': int(row['label']),
                'cow_id': row['cow_id'],
                'split': row['split']
            })
            
    # Get all unique cow (video) IDs
    video_ids = sorted(list(set(item['cow_id'] for item in video_data)))
    print(f"Found {len(video_ids)} unique video sequences to evaluate.")

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    results = []

    # Run inference for all 50 videos
    for v_id in tqdm(video_ids, desc="Evaluating videos"):
        v_samples = [item for item in video_data if item['cow_id'] == v_id]
        v_samples = sorted(v_samples, key=lambda x: x['image_path'])
        split = v_samples[0]['split']
        label = v_samples[0]['label']

        # Sample 20 frames evenly
        total_frames = len(v_samples)
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20

        frames = []
        for idx_to_load in indices:
            img_path = v_samples[idx_to_load]['image_path']
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)["image"]
            frames.append(image)

        # Shape: (1, 20, 3, 224, 224)
        input_tensor = torch.stack(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                outputs = model(input_tensor)
                prob = torch.sigmoid(outputs).view(-1).item()

        pred_label = 1 if prob >= DECISION_THRESHOLD else 0
        correct = (pred_label == label)

        results.append({
            "cow_id": v_id,
            "split": split,
            "true_label": label,
            "pred_probability": prob,
            "pred_label": pred_label,
            "correct": correct
        })

    # Compute metrics globally and per split
    splits = ["train", "val", "test"]
    split_metrics = {}
    
    for s in splits:
        s_results = [r for r in results if r['split'] == s]
        if len(s_results) > 0:
            labels = np.array([r['true_label'] for r in s_results])
            probs = np.array([r['pred_probability'] for r in s_results])
            preds = np.array([r['pred_label'] for r in s_results])
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = float('nan')
            
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            split_metrics[s] = {
                "count": len(s_results),
                "accuracy": acc,
                "f1": f1,
                "auc": auc,
                "cm": cm
            }

    # Global metrics
    g_labels = np.array([r['true_label'] for r in results])
    g_probs = np.array([r['pred_probability'] for r in results])
    g_preds = np.array([r['pred_label'] for r in results])
    g_acc = accuracy_score(g_labels, g_preds)
    g_f1 = f1_score(g_labels, g_preds, zero_division=0)
    g_auc = roc_auc_score(g_labels, g_probs)
    g_cm = confusion_matrix(g_labels, g_preds, labels=[0, 1])

    # Build the report string
    report = []
    report.append("=" * 70)
    report.append("          EVALUATION REPORT: ALL 50 CATTLE VIDEOS")
    report.append("=" * 70)
    report.append(f"Model: {MODEL_NAME}-LSTM")
    report.append(f"Checkpoint: {CHECKPOINT_PATH}")
    report.append(f"Calibrated Threshold: {DECISION_THRESHOLD:.2f}\n")

    report.append("DETAILED TABLE:")
    report.append("-" * 75)
    report.append(f"{'Cow ID':<15} | {'Split':<8} | {'True Label':<12} | {'Pred Prob':<10} | {'Pred Label':<10} | {'Status':<8}")
    report.append("-" * 75)
    for r in results:
        t_lbl = "Lame" if r['true_label'] == 1 else "Normal"
        p_lbl = "Lame" if r['pred_label'] == 1 else "Normal"
        status = "CORRECT" if r['correct'] else "WRONG"
        report.append(f"{r['cow_id']:<15} | {r['split']:<8} | {t_lbl:<12} | {r['pred_probability']:0.4f}     | {p_lbl:<10} | {status:<8}")
    report.append("-" * 75)
    report.append("\nSUMMARY METRICS PER SPLIT:")
    
    for s, metrics in split_metrics.items():
        report.append(f"\n--- {s.upper()} SPLIT ({metrics['count']} videos) ---")
        report.append(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        report.append(f"  F1 Score: {metrics['f1']:.4f}")
        report.append(f"  AUC:      {metrics['auc']:.4f}")
        report.append(f"  Confusion Matrix:")
        report.append(f"    [[TN: {metrics['cm'][0,0]}, FP: {metrics['cm'][0,1]}],")
        report.append(f"     [FN: {metrics['cm'][1,0]}, TP: {metrics['cm'][1,1]}]]")

    report.append("\n" + "=" * 70)
    report.append("--- OVERALL GLOBAL METRICS (All 50 videos) ---")
    report.append(f"  Overall Accuracy: {g_acc*100:.2f}%")
    report.append(f"  Overall F1 Score: {g_f1:.4f}")
    report.append(f"  Overall AUC:      {g_auc:.4f}")
    report.append(f"  Overall Confusion Matrix:")
    report.append(f"    [[TN: {g_cm[0,0]}, FP: {g_cm[0,1]}],")
    report.append(f"     [FN: {g_cm[1,0]}, TP: {g_cm[1,1]}]]")
    report.append("=" * 70)

    report_text = "\n".join(report)
    
    # Save report
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Print to console
    print(report_text)
    print(f"\nSaved evaluation report to: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    main()
