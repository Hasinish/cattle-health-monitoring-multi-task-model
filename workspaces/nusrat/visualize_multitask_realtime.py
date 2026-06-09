import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")
MULTITASK_CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "multitask_best.pth")

# Defaulting to Downloaded YouTube Cow Video
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "youtube_cow_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "youtube_realtime_demo.mp4")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (224, 224)

# Mappings
BCS_CLASSES = [3.25, 3.5, 3.75, 4.0, 4.25]
BEHAVIOR_CLASSES = [
    "Walking",
    "Standing",
    "Feeding head up",
    "Feeding head down",
    "Licking",
    "Drinking",
    "Lying"
]

# ============================================================
# MULTI-TASK MODEL ARCHITECTURE (Matches train_multitask.py)
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

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.bcs_head = nn.Linear(feature_dim, 4)
        self.behavior_head = nn.Linear(feature_dim, 7)
        self.lameness_head = nn.Linear(feature_dim, 1)
        self.id_head = nn.Linear(feature_dim, 46)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)
        return {
            'bcs': self.bcs_head(pooled),
            'behavior': self.behavior_head(pooled),
            'lameness': self.lameness_head(pooled),
            'id': self.id_head(pooled)
        }

# ============================================================
# INFERENCE PIPELINE
# ============================================================

def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)

def main():
    print(f"Loading YOLOv8 cropper from {YOLO_MODEL_PATH}...")
    detector = YOLO(YOLO_MODEL_PATH)
    
    print(f"Loading MultiTaskModel checkpoint from {MULTITASK_CHECKPOINT_PATH}...")
    model = MultiTaskModel().to(DEVICE)
    if os.path.exists(MULTITASK_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(MULTITASK_CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("WARNING: Checkpoint not found! Running visualization with random/untrained weights.")
    model.eval()

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"Processing video: {INPUT_VIDEO_PATH}")
    print(f"Saving demo video to: {OUTPUT_VIDEO_PATH}")

    for _ in tqdm(range(total_frames), desc="Running Multi-Task Inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect Cow
        yolo_results = detector(frame, verbose=False)[0]
        cow_box = None
        max_area = 0

        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            if class_id == 19: # 19 is 'cow' in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    cow_box = (x1, y1, x2, y2)

        # Step 2: Crop & Classify Cow if detected
        if cow_box is not None:
            x1, y1, x2, y2 = cow_box
            cow_crop = frame[y1:y2, x1:x2]
            
            if cow_crop.size > 0:
                # Preprocess for MultiTaskModel
                rgb_crop = cv2.cvtColor(cow_crop, cv2.COLOR_BGR2RGB)
                input_tensor = transform(image=rgb_crop)["image"].unsqueeze(0).to(DEVICE)

                # Forward pass
                with torch.no_grad():
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(input_tensor)

                # Predictions
                bcs_idx = coral_predict(outputs['bcs']).item()
                bcs_score = BCS_CLASSES[min(bcs_idx, len(BCS_CLASSES)-1)]

                behavior_probs = F.softmax(outputs['behavior'], dim=1).squeeze(0)
                behavior_idx = torch.argmax(behavior_probs).item()
                behavior_conf = behavior_probs[behavior_idx].item()
                behavior = BEHAVIOR_CLASSES[behavior_idx]

                lame_prob = torch.sigmoid(outputs['lameness']).item()
                lameness = "Lame" if lame_prob > 0.5 else "Normal"
                
                cow_id = torch.argmax(outputs['id'], dim=1).item()

                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw background label plate
                overlay = frame.copy()
                plate_y1 = max(y1 - 100, 0)
                cv2.rectangle(overlay, (x1, plate_y1), (x1 + 240, y1), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Put prediction labels on the video
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"ID: Cow #{cow_id}", (x1 + 10, plate_y1 + 20), font, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"BCS: {bcs_score}", (x1 + 10, plate_y1 + 40), font, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, f"Behavior: {behavior} ({behavior_conf*100:.1f}%)", (x1 + 10, plate_y1 + 60), font, 0.55, (255, 100, 255), 2)
                cv2.putText(frame, f"Lameness: {lameness} ({lame_prob:.2f})", (x1 + 10, plate_y1 + 80), font, 0.55, (100, 100, 255) if lameness == "Lame" else (100, 255, 100), 2)

        out.write(frame)
        
        # Display the frame in real-time
        try:
            cv2.imshow("Multi-Task Cow Monitor (Press 'q' to Quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nReal-time preview stopped early by user.")
                break
        except cv2.error:
            pass

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("\nVisual inference test completed successfully!")

if __name__ == "__main__":
    main()
