import os
import random
import csv
from pathlib import Path
from collections import defaultdict, Counter

DATASET_ROOT = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs\dataset"
OUTPUT_CSV = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"
VALID_CLASSES = ['3.25', '3.5', '3.75', '4.0', '4.25']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Step 1: Collect all cows and images
cow_data = defaultdict(list)

for label in VALID_CLASSES:
    class_dir = Path(DATASET_ROOT) / label
    if not class_dir.exists():
        continue
    for img_file in class_dir.glob("*.jpg"):
        stem = img_file.stem
        if stem.startswith('GS_') or stem.startswith('YM_'):
            parts = stem.split('_')
            cow_id = f"{parts[0]}_{parts[1]}"
        elif stem.startswith('L-i') or stem.startswith('R-i'):
            cow_id = 'i' + stem[3:]  # L-i1035 and R-i1035 = same cow
        else:
            cow_id = stem
        cow_data[cow_id].append((str(img_file), label))

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

# Step 4: Summary
splits = {'train': 0, 'val': 0, 'test': 0}
for row in rows:
    splits[row[3]] += 1

print(f"Total cows: {len(all_cows)}")
print(f"Train cows: {len(train_cows)} | Val: {len(val_cows)} | Test: {len(test_cows)}")
print(f"Train images: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")
print(f"Total images: {len(rows)}")

print("\nClass distribution:")
labels = [r[1] for r in rows]
for cls, count in sorted(Counter(labels).items()):
    print(f"  BCS {cls}: {count} images")

print("\nSample rows:")
for row in rows[:3]:
    print(row)