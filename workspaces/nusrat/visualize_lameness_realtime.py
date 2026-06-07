import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from tqdm import tqdm

# Configurations
BASE_DIR = r"D:\T25301094 P2"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
LAME_CHECKPOINT_PATH = os.path.join(BASE_DIR, "workspaces", "nusrat", "spatiotemporal_lameness_efficientnet_best.pth")
ID_CHECKPOINT_PATH = os.path.join(BASE_DIR, "workspaces", "nusrat", "id_best.pth")
CSV_PATH = os.path.join(BASE_DIR, "datasets", "id", "id_index.csv")
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "cut_cow_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "cut_cow_realtime_detection.mp4")

MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 64
DECISION_THRESHOLD = 0.50
TARGET_SIZE = (224, 224)
SEQ_LEN = 20
NUM_CLASSES_ID = 46

# Attention Modules for CowIDModel
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

# CowIDModel definition
class CowIDModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
        return x

# CNN-LSTM Lameness Model definition
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

def draw_premium_overlay(img, text_lines, x, y, colors):
    font_scale = 0.75
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    box_w = 0
    line_heights = []
    
    for line in text_lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        box_w = max(box_w, w)
        line_heights.append(h + baseline + 6)
    
    box_w += 20
    box_h = sum(line_heights) + 15
    
    # Background semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - 25), (x + box_w, y + box_h - 25), (0, 0, 0), -1)
    
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw texts with their individual colors
    curr_y = y - 5
    for idx, line in enumerate(text_lines):
        color = colors[idx] if idx < len(colors) else (255, 255, 255)
        cv2.putText(img, line, (x, curr_y), font, font_scale, color, thickness)
        curr_y += line_heights[idx]

def build_embedding_database(model, csv_path, device):
    print("Building cow embedding database from training images...")
    
    # Group image paths by label
    cow_images = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'train':
                lbl = int(row['label'])
                if lbl not in cow_images:
                    cow_images[lbl] = []
                if len(cow_images[lbl]) < 5:
                    cow_images[lbl].append(row['image_path'])
                    
    # Extract features and average them per cow class
    database = {}
    model.eval()
    
    # Simple transform matching train_id.py eval_transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    with torch.no_grad():
        for lbl, paths in tqdm(cow_images.items(), desc="Registering Cows"):
            tensors = []
            for path in paths:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = transform(image=img)["image"]
                    tensors.append(tensor)
            if tensors:
                batch = torch.stack(tensors).to(device)
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                    # Get features before final classifier
                    features = model.extract_features(batch) # shape: (N, 1280)
                    mean_feature = features.mean(dim=0)
                    mean_feature = F.normalize(mean_feature, p=2, dim=0) # Normalize to unit sphere
                database[lbl] = mean_feature
                
    print(f"Database built successfully. Registered classes: {len(database)}\n")
    return database

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Bounding Box, Cow ID & Lameness Visualizer")
    parser.add_argument("--input", type=str, default=INPUT_VIDEO_PATH, help="Path to input video file")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to save annotated video")
    parser.add_argument("--stride", type=int, default=8, help="Temporal stride to sub-sample frames")
    parser.add_argument("--threshold", type=float, default=0.65, help="Cosine similarity threshold for registration verification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLO detector
    print("Loading YOLOv8 detector...")
    yolo = YOLO(YOLO_MODEL_PATH)

    # Load Cow ID model
    print("Loading Cow ID baseline model...")
    id_model = CowIDModel(MODEL_NAME, NUM_CLASSES_ID, device)
    id_model = id_model.to(device)
    if not os.path.exists(ID_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Cow ID checkpoint not found at: {ID_CHECKPOINT_PATH}")
    id_model.load_state_dict(torch.load(ID_CHECKPOINT_PATH, map_location=device, weights_only=True))
    id_model.eval()

    # Build embedding database
    embedding_db = build_embedding_database(id_model, CSV_PATH, device)

    # Load Lameness model
    print("Loading Lameness CNN-LSTM model...")
    lame_model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    lame_model = lame_model.to(device)
    if not os.path.exists(LAME_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Lameness checkpoint not found at: {LAME_CHECKPOINT_PATH}")
    lame_model.load_state_dict(torch.load(LAME_CHECKPOINT_PATH, map_location=device, weights_only=True))
    lame_model.eval()
    print("Models loaded successfully.")

    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open input video {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"\nProcessing Video:")
    print(f"  Source:     {args.input}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps}")
    print(f"  Output:     {args.output}\n")

    # Image preprocessing transform
    transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Sliding window buffer of cropped cow frames (holds preprocessed tensors)
    cropped_buffer = []
    show_window = True
    last_best_box = None
    
    # Store running ID results to display stable values
    last_cow_id_str = "Unknown"
    last_cow_id_prob = 0.0

    # Auto-registration variables
    unknown_embeddings_buffer = []
    next_class_id = NUM_CLASSES_ID  # Starts at 46 (which represents indices 0-45)
    
    # Calculate required consecutive frames of observation for 5 seconds
    required_unknown_frames = int(5.0 * fps / args.stride)
    print(f"Registration Window: 5.0s of video space ({required_unknown_frames} processed frames at stride {args.stride})")

    # Process frame-by-frame
    for i in tqdm(range(total_frames), desc="Running Real-Time Inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = yolo(frame, verbose=False)[0]

        best_box = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            # Class ID 19 is 'cow'
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        # Update last known bounding box for tracking fallback
        if best_box is not None:
            last_best_box = best_box
        
        # Determine crop region
        crop_box = best_box if best_box is not None else last_best_box

        cow_crop = None
        if crop_box is not None:
            cx1, cy1, cx2, cy2 = crop_box
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(width, cx2)
            cy2 = min(height, cy2)
            if cx2 > cx1 and cy2 > cy1:
                cow_crop = frame[cy1:cy2, cx1:cx2]

        # Process crop frame
        if cow_crop is not None and cow_crop.size > 0:
            rgb_crop = cv2.cvtColor(cow_crop, cv2.COLOR_BGR2RGB)
            transformed = transform(image=rgb_crop)["image"]

            # Sub-sample frames temporally according to STRIDE to match dataset time span
            if i % args.stride == 0:
                cropped_buffer.append(transformed)
                if len(cropped_buffer) > SEQ_LEN:
                    cropped_buffer.pop(0)

                # Run Cow ID prediction using vector embedding Cosine Similarity
                input_crop_tensor = transformed.unsqueeze(0).to(device)
                with torch.no_grad():
                    with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                        live_features = id_model.extract_features(input_crop_tensor) # shape: (1, 1280)
                        live_features = F.normalize(live_features, p=2, dim=1) # normalize to unit sphere
                        
                # Compute Cosine Similarity against all registered classes
                best_similarity = -1.0
                best_class = -1
                for lbl, reg_feature in embedding_db.items():
                    sim = torch.dot(live_features[0], reg_feature).item()
                    if sim > best_similarity:
                        best_similarity = sim
                        best_class = lbl
                
                # Check threshold for open-set registration verification
                if best_similarity >= args.threshold:
                    last_cow_id_str = f"{(best_class + 1):03d}"
                    last_cow_id_prob = best_similarity
                    # Clear the unknown buffer because we successfully recognized a registered cow
                    unknown_embeddings_buffer.clear()
                else:
                    last_cow_id_str = "Unknown Cow"
                    last_cow_id_prob = best_similarity
                    
                    # Store features for auto-registration
                    unknown_embeddings_buffer.append(live_features[0].cpu())
                    
                    # If we collect 5 seconds of consecutive unknown frames, register as a new cow
                    if len(unknown_embeddings_buffer) >= required_unknown_frames:
                        new_cow_vector = torch.stack(unknown_embeddings_buffer).mean(dim=0).to(device)
                        new_cow_vector = F.normalize(new_cow_vector, p=2, dim=0)
                        
                        # Register in memory database
                        embedding_db[next_class_id] = new_cow_vector
                        print(f"\n>>> [AUTO-REGISTRATION] Registered New Cow in database as ID: {next_class_id + 1:03d} after 5.0 seconds of observation!")
                        
                        # Set current outputs to match the new class
                        last_cow_id_str = f"{next_class_id + 1:03d}"
                        last_cow_id_prob = 1.0 # Initial similarity is 1.0 since it matches itself
                        
                        next_class_id += 1
                        unknown_embeddings_buffer.clear()

        # Draw YOLO box on the frame
        box_color = (0, 255, 0) # Green bounding box
        if best_box is not None:
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), box_color, 3)
        elif last_best_box is not None:
            # Draw tracking box if reusing previous box
            cv2.rectangle(frame, (last_best_box[0], last_best_box[1]), (last_best_box[2], last_best_box[3]), (255, 128, 0), 1)

        # Run spatiotemporal lameness prediction if buffer is full
        text_lines = []
        text_colors = []

        # Add Cow ID info to overlay
        text_lines.append(f"Cow ID: {last_cow_id_str} ({last_cow_id_prob*100:.1f}%)")
        text_colors.append((255, 255, 255)) # White

        if len(cropped_buffer) == SEQ_LEN:
            input_seq_tensor = torch.stack(cropped_buffer).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                    lame_logits = lame_model(input_seq_tensor)
                    lame_prob = torch.sigmoid(lame_logits).view(-1).item()

            prediction = "LAME" if lame_prob >= DECISION_THRESHOLD else "NORMAL"
            gait_color = (0, 0, 255) if prediction == "LAME" else (0, 255, 0)
            text_lines.append(f"Gait: {prediction} ({lame_prob*100:.1f}%)")
            text_colors.append(gait_color)
        else:
            text_lines.append(f"Gait: Buffering Stride... ({len(cropped_buffer)}/{SEQ_LEN})")
            text_colors.append((255, 255, 0)) # Cyan/yellow

        # Draw the premium transparent background overlay
        text_x = best_box[0] if best_box is not None else (last_best_box[0] if last_best_box is not None else 30)
        text_y = max(40, (best_box[1] - 40) if best_box is not None else (last_best_box[1] - 40 if last_best_box is not None else 50))
        
        draw_premium_overlay(frame, text_lines, text_x, text_y, text_colors)

        # Write annotated frame to output video
        out.write(frame)

        # Display window
        if show_window:
            try:
                cv2.imshow("Real-Time Unified Multi-Task Inference (Press 'q' to Quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing stopped early by user.")
                    break
            except cv2.error:
                print("\nWarning: Headless environment or OpenCV without GUI support detected.")
                print("Running in HEADLESS mode. Video is still being written to the output file.")
                show_window = False

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print(f"\nProcessing Complete!")
    print(f"Saved annotated video to: {args.output}")

if __name__ == "__main__":
    main()
