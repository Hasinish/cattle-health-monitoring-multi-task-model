import os
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

# CONFIG
BASE_DIR = r"d:\T25301094 P2"
DATASET_ROOT = Path(BASE_DIR) / "datasets" / "id" / "opencow2020-DatasetNinja"
OUTPUT_CSV = Path(BASE_DIR) / "datasets" / "id" / "id_index.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

train_dir = DATASET_ROOT / "identification-train" / "img"
test_dir = DATASET_ROOT / "identification-test" / "img"

print(f"Dataset root: {DATASET_ROOT}")
print(f"Output CSV path: {OUTPUT_CSV}")

rows = []

# Process Train Set to split into train and val splits
train_images_by_cow = defaultdict(list)
if train_dir.exists():
    for filename in sorted(os.listdir(train_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Filename format: [class_id]_[image_id].jpg
            parts = filename.split('_')
            if len(parts) >= 2:
                cow_id = parts[0]
                image_path = train_dir / filename
                train_images_by_cow[cow_id].append(str(image_path))

# Split cow-wise/image-wise within training set
# Total 46 classes
all_cows = sorted(list(train_images_by_cow.keys()))
print(f"Found {len(all_cows)} cow classes in training directory.")

for cow_id in all_cows:
    img_paths = train_images_by_cow[cow_id]
    # Shuffle images for this cow class to avoid order bias
    random.shuffle(img_paths)
    
    n = len(img_paths)
    train_count = int(n * 0.85)
    
    # 0-indexed label corresponding to cow class
    label = int(cow_id) - 1
    
    for idx, path in enumerate(img_paths):
        split = 'train' if idx < train_count else 'val'
        rows.append([path, str(label), cow_id, split])

# Process Test Set to assign to test split
test_count = 0
if test_dir.exists():
    for filename in sorted(os.listdir(test_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            parts = filename.split('_')
            if len(parts) >= 2:
                cow_id = parts[0]
                image_path = test_dir / filename
                label = int(cow_id) - 1
                rows.append([str(image_path), str(label), cow_id, 'test'])
                test_count += 1

print(f"Total test images indexed: {test_count}")

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

# Print Summary Statistics
splits = Counter([row[3] for row in rows])
print(f"\nPreprocessing Complete!")
print(f"Total entries in CSV: {len(rows)}")
print(f"Split distribution: Train={splits['train']} | Val={splits['val']} | Test={splits['test']}")
print(f"CSV index file written to: {OUTPUT_CSV}")
