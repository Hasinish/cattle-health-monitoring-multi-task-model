import os
import glob
import cv2
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Model Configuration
MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 64
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth"
DECISION_THRESHOLD = 0.70  # Calibrated decision threshold

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

def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def load_frames_from_directory(dir_path):
    # Support sorting frame files numerically or alphabetically
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    
    if not files:
        raise FileNotFoundError(f"Error: No image files found in directory {dir_path}")
    
    # Sort files naturally/numerically
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in glob.os.path.split(x)[1].split('_')])
    
    frames = []
    for file in files:
        img = cv2.imread(file)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict Lameness from Video or Frame Directory")
    parser.add_argument("--path", type=str, required=True, help="Path to input video file (.mp4, .avi) OR image frame directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load model
    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model checkpoint loaded successfully from: {CHECKPOINT_PATH}")

    # Detect if path is a directory, a glob pattern matching files, or a video file
    if os.path.isdir(args.path):
        print(f"Loading frames from directory: {args.path}")
        frames = load_frames_from_directory(args.path)
    elif glob.glob(args.path) and len(glob.glob(args.path)) > 1:
        print(f"Loading frames matching pattern: {args.path}")
        files = glob.glob(args.path)
        files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in os.path.split(x)[1].split('_')])
        frames = []
        for file in files:
            img = cv2.imread(file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
    else:
        print(f"Loading frames from video file: {args.path}")
        frames = load_frames_from_video(args.path)

    total_frames = len(frames)
    print(f"Loaded {total_frames} frames.")
    
    if total_frames == 0:
        raise ValueError("Error: Loaded frame sequence is empty.")

    # Sparse sample exactly 20 frames evenly spaced
    indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
    sampled_frames = [frames[i] for i in indices]

    # Normalize and compile transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    processed_frames = [transform(image=img)["image"] for img in sampled_frames]
    input_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)  # Add batch dimension: (1, 20, 3, 224, 224)

    # Perform inference
    with torch.no_grad():
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
            outputs = model(input_tensor)
            prob = torch.sigmoid(outputs).item()

    prediction = "Lame" if prob >= DECISION_THRESHOLD else "Normal"

    print("\n" + "="*40)
    print("           INFERENCE RESULTS            ")
    print("="*40)
    print(f"Source: {os.path.basename(args.path)}")
    print(f"Calibrated Threshold: {DECISION_THRESHOLD:.2f}")
    print(f"Lameness Probability: {prob * 100:.2f}%")
    print(f"Final Prediction:     {prediction.upper()}")
    print("="*40)

if __name__ == "__main__":
    main()
