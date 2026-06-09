import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_index.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")
CROPPED_DIR = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped")
YOLO_MODEL = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")
TARGET_SIZE = (224, 224)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV}")

    os.makedirs(CROPPED_DIR, exist_ok=True)

    print("Loading YOLOv8 detector...")
    detector = YOLO(YOLO_MODEL)

    print(f"Reading index file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    cropped_records = []
    skipped_count = 0

    print(f"Processing {len(df)} images...")
    # Loop over all images in the index
    for idx, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Cropping ScienceDB")):
        img_path = row[1]['image_path']
        label = row[1]['label']
        cow_id = row[1]['cow_id']
        split = row[1]['split']

        if not os.path.exists(img_path):
            skipped_count += 1
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            skipped_count += 1
            continue

        # Run YOLO detection
        results = detector(img, verbose=False)[0]

        cow_crop = None
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

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cow_crop = img[y1:y2, x1:x2]

        # Save output path
        filename = f"crop_{idx}_{os.path.basename(img_path)}"
        output_path = os.path.join(CROPPED_DIR, filename)

        # Write crop or fallback to original resized
        if cow_crop is not None and cow_crop.size > 0:
            resized_crop = cv2.resize(cow_crop, TARGET_SIZE)
            cv2.imwrite(output_path, resized_crop)
        else:
            # Fallback to resizing original image if YOLO misses
            resized_orig = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(output_path, resized_orig)

        cropped_records.append({
            "image_path": output_path,
            "label": label,
            "cow_id": cow_id,
            "split": split
        })

    # Save new cropped index CSV
    out_df = pd.DataFrame(cropped_records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCropping Complete!")
    print(f"  Cropped images saved to: {CROPPED_DIR}")
    print(f"  New index CSV written to: {OUTPUT_CSV}")
    print(f"  Skipped/Missing files:    {skipped_count}")

if __name__ == "__main__":
    main()
