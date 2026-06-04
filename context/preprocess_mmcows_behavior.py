import os
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

DATASET_ROOT = r"D:\T25301094 P2\datasets\behavior\mmcows\cropped_bboxes\cropped_bboxes\behaviors"
OUTPUT_CSV = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
VALID_CLASSES = ['1','2','3','4','5','6','7']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

cow_data = defaultdict(list)

for label in VALID_CLASSES:
    class_dir = Path(DATASET_ROOT) / label
    for img_file in class_dir.glob("*.jpg"):
        parts = img_file.stem.split('_')
        cow_id = parts[2]
        cow_data[cow_id].append((str(img_file), label))

all_cows = list(cow_data.keys())
random.shuffle(all_cows)

n = len(all_cows)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_cows = set(all_cows[:train_end])
val_cows = set(all_cows[train_end:val_end])
test_cows = set(all_cows[val_end:])

rows = []
for cow_id, entries in cow_data.items():
    split = 'train' if cow_id in train_cows else ('val' if cow_id in val_cows else 'test')
    for img_path, label in entries:
        rows.append([img_path, label, cow_id, split])

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

splits = {'train': 0, 'val': 0, 'test': 0}
for row in rows:
    splits[row[3]] += 1

print(f"Total cows: {len(all_cows)}")
print(f"Train: {len(train_cows)} cows | Val: {len(val_cows)} cows | Test: {len(test_cows)} cows")
print(f"Train images: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")
print(f"Total: {len(rows)}")
print("\nClass distribution:")
labels = [r[1] for r in rows]
for cls, count in sorted(Counter(labels).items()):
    print(f"  Class {cls}: {count} images")