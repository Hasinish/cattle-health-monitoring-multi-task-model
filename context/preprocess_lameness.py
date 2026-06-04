import os
import cv2
import csv
import random
from pathlib import Path

# Paths
BASE_DIR = r"D:\T25301094 P2"
LAME_DIR = Path(BASE_DIR) / "datasets" / "lameness" / "CattleLameness" / "Data" / "Lame"
NORMAL_DIR = Path(BASE_DIR) / "datasets" / "lameness" / "CattleLameness" / "Data" / "Normal"
OUTPUT_FRAMES_DIR = Path(BASE_DIR) / "datasets" / "lameness" / "frames"
OUTPUT_CSV = Path(BASE_DIR) / "datasets" / "lameness" / "lameness_index.csv"

# Configuration
TARGET_SIZE = (224, 224)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Ensure output frames directory exists
OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def process_video_folder(directory, label, class_prefix):
    video_files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4')])
    
    # Shuffle videos to perform video-wise (cow-wise) split
    random.shuffle(video_files)
    
    num_videos = len(video_files)
    train_end = int(num_videos * 0.70)
    val_end = train_end + int(num_videos * 0.15)
    
    video_data = []
    
    for idx, video_file in enumerate(video_files):
        video_path = directory / video_file
        video_id = f"{class_prefix}_{idx + 1}"
        
        # Determine split
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'val'
        else:
            split = 'test'
            
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video {video_file}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Empty video {video_file}")
            cap.release()
            continue
            
        # Read and save all frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to 224x224
            resized_frame = cv2.resize(frame, TARGET_SIZE)
            
            # Save frame image
            frame_filename = f"{video_id}_frame_{frame_idx}.jpg"
            frame_path = OUTPUT_FRAMES_DIR / frame_filename
            cv2.imwrite(str(frame_path), resized_frame)
            
            video_data.append({
                'image_path': str(frame_path),
                'label': label,
                'cow_id': video_id,
                'split': split
            })
            frame_idx += 1
            
        cap.release()
        print(f"  Processed {video_file}: extracted {frame_idx} frames.")
        
    return video_data

print("Extracting Lame video frames (ALL)...")
lame_data = process_video_folder(LAME_DIR, 1, "Lame")

print("Extracting Normal video frames (ALL)...")
normal_data = process_video_folder(NORMAL_DIR, 0, "Normal")

all_data = lame_data + normal_data

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    for item in all_data:
        writer.writerow([item['image_path'], item['label'], item['cow_id'], item['split']])

# Summary statistics
splits = {'train': 0, 'val': 0, 'test': 0}
labels = {'Lame (1)': 0, 'Normal (0)': 0}
for item in all_data:
    splits[item['split']] += 1
    if item['label'] == 1:
        labels['Lame (1)'] += 1
    else:
        labels['Normal (0)'] += 1

print(f"\nPreprocessing Complete!")
print(f"Total frames extracted: {len(all_data)}")
print(f"Split distribution: Train={splits['train']} | Val={splits['val']} | Test={splits['test']}")
print(f"Class distribution: Lame={labels['Lame (1)']} | Normal={labels['Normal (0)']}")
print(f"CSV index file written to: {OUTPUT_CSV}")
