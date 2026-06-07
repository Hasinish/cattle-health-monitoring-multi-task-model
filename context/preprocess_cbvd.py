import os
import csv
import json
import random
import cv2
from pathlib import Path
from collections import defaultdict

# CONFIG
BASE_DIR = r"d:\T25301094 P2"
DATASET_ROOT = Path(BASE_DIR) / "datasets" / "behavior" / "CBVD-5"
OUTPUT_CROP_DIR = DATASET_ROOT / "cropped"
OUTPUT_CSV = DATASET_ROOT / "cbvd_cropped_index.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
OUTPUT_CROP_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset Root: {DATASET_ROOT}")
print(f"Output Crop Dir: {OUTPUT_CROP_DIR}")
print(f"Output CSV: {OUTPUT_CSV}")

# Parse CSV
csv_path = DATASET_ROOT / "CBVD-5.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

# Groups by label (0, 1, 2, 3, 4 -> mapped to MmCows labels)
# Label mapping:
#   '1' in option_ids -> 6 (Lying)
#   '3' in option_ids -> 5 (Drinking)
#   '2' in option_ids -> 3 (Feeding)
#   '0' in option_ids -> 1 (Standing)
grouped_rows = defaultdict(list)

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        if row[0].startswith('#'):
            continue
        
        # row: metadata_id, file_list, flags, temporal_coordinates, spatial_coordinates, metadata
        metadata_id = row[0]
        try:
            filename = json.loads(row[1])[0]
            spatial = json.loads(row[4])
            meta = json.loads(row[5])
            
            if "1" not in meta:
                continue
                
            val = meta["1"]
            
            # Label mapping logic
            if '1' in val:      # Lying down
                mapped_label = 6
            elif '3' in val:    # Drinking
                mapped_label = 5
            elif '2' in val:    # Foraging/Feeding
                mapped_label = 3
            elif '0' in val:    # Standing
                mapped_label = 1
            else:
                continue
                
            grouped_rows[mapped_label].append({
                'metadata_id': metadata_id,
                'filename': filename,
                'spatial': spatial,
                'label': mapped_label
            })
        except Exception as e:
            pass

print("\nParsed Label Counts:")
for lbl, items in sorted(grouped_rows.items()):
    print(f"  Mapped Label {lbl}: {len(items)} items")

# Select a balanced subset of up to 500 images per class
selected_items = []
target_size_per_class = 500

for lbl, items in sorted(grouped_rows.items()):
    random.shuffle(items)
    selected = items[:target_size_per_class]
    selected_items.extend(selected)
    print(f"  Selected {len(selected)} items for Label {lbl}")

print(f"\nTotal selected items for cropping: {len(selected_items)}")

# Crop and save images
csv_rows = []
success_count = 0

for idx, item in enumerate(selected_items):
    filename = item['filename']
    metadata_id = item['metadata_id']
    spatial = item['spatial']
    label = item['label']
    
    # Locate full-resolution source image
    try:
        video_id = int(filename.split('_')[0])
    except ValueError:
        continue
        
    if video_id >= 700:
        img_path = DATASET_ROOT / "labelframes_add" / "labelframes" / filename
    else:
        img_path = DATASET_ROOT / "labelframes" / "labelframes" / filename
        
    if not img_path.exists():
        continue
        
    # Read full-resolution image
    img = cv2.imread(str(img_path))
    if img is None:
        continue
        
    # Bounding Box Coordinates: [shape_id, x, y, width, height]
    x = int(float(spatial[1]))
    y = int(float(spatial[2]))
    w = int(float(spatial[3]))
    h = int(float(spatial[4]))
    
    img_h, img_w = img.shape[:2]
    cx1 = max(0, x)
    cy1 = max(0, y)
    cx2 = min(img_w, x + w)
    cy2 = min(img_h, y + h)
    
    if cx2 <= cx1 or cy2 <= cy1:
        continue
        
    # Crop and Resize
    crop = img[cy1:cy2, cx1:cx2]
    resized_crop = cv2.resize(crop, (224, 224))
    
    # Save Crop
    crop_filename = f"{metadata_id}.jpg"
    crop_path = OUTPUT_CROP_DIR / crop_filename
    cv2.imwrite(str(crop_path), resized_crop)
    
    # Save CSV Row: image_path, label, cow_id, split
    csv_rows.append([str(crop_path), str(label), f"CBVD_{video_id}", "test"])
    success_count += 1
    
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(selected_items)} images...")

# Write new CSV index
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(csv_rows)

print(f"\nCropping Complete!")
print(f"Successfully cropped and saved {success_count} images.")
print(f"CSV index written to: {OUTPUT_CSV}")
