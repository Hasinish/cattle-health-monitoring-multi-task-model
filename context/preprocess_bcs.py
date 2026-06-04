import os
import random
import shutil
import csv
from pathlib import Path

# CONFIG
DATASET_ROOT = r"D:\T25301094 P2\datasets\bcs\dryad_bcs\Total_sorted_DGE_images\Total_sorted_DGE_images"
OUTPUT_CSV = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
VALID_CLASSES = ['2', '3', '4', '5', '6']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Step 1: Collect all cows and their images per class
cow_data = {}  # cow_id -> list of (image_path, label)

for label in VALID_CLASSES:
    class_dir = Path(DATASET_ROOT) / label
    if not class_dir.exists():
        continue
    for cow_folder in class_dir.iterdir():
        if not cow_folder.is_dir():
            continue
        cow_id = cow_folder.name
        images = list(cow_folder.glob("*.tif"))
        if len(images) == 0:
            continue
        if cow_id not in cow_data:
            cow_data[cow_id] = []
        for img in images:
            cow_data[cow_id].append((str(img), label))

# Step 2: Cow-wise split
all_cows = list(cow_data.keys())
random.shuffle(all_cows)

n = len(all_cows)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_cows = set(all_cows[:train_end])
val_cows = set(all_cows[train_end:val_end])
test_cows = set(all_cows[val_end:])

# Step 3: Write CSV
rows = []
for cow_id, entries in cow_data.items():
    if cow_id in train_cows:
        split = 'train'
    elif cow_id in val_cows:
        split = 'val'
    else:
        split = 'test'
    for img_path, label in entries:
        rows.append([img_path, label, cow_id, split])

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

# Step 4: Print summary
print(f"Total cows: {len(all_cows)}")
print(f"Train cows: {len(train_cows)} | Val cows: {len(val_cows)} | Test cows: {len(test_cows)}")

splits = {'train': 0, 'val': 0, 'test': 0}
for row in rows:
    splits[row[3]] += 1
print(f"Train images: {splits['train']} | Val images: {splits['val']} | Test images: {splits['test']}")
print(f"Total images: {len(rows)}")
print(f"CSV saved to: {OUTPUT_CSV}")