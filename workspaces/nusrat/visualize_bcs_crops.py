import os
import cv2
import random
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_index.csv")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "workspaces", "nusrat", "bcs_crop_samples.png")
YOLO_MODEL = "yolov8n.pt"
PANEL_SIZE = (224, 224)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV}")

    print("Loading YOLOv8 detector...")
    detector = YOLO(YOLO_MODEL)

    print(f"Reading index file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Pick 5 random rows to visualize
    random.seed(42)  # For reproducibility
    sampled_rows = df.sample(n=5).reset_index(drop=True)

    rows_images = []

    for idx, row in enumerate(sampled_rows.iterrows()):
        img_path = row[1]['image_path']
        print(f"Processing image {idx+1}/5: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image: {img_path}")
            continue

        # Keep a copy of the original for drawing the box
        original_visual = img.copy()

        # Run YOLO detection
        results = detector(img, verbose=False)[0]

        cow_crop = None
        box_coords = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            # Class ID 19 is 'cow'
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    box_coords = (x1, y1, x2, y2)

        if box_coords is not None:
            x1, y1, x2, y2 = box_coords
            cow_crop = img[y1:y2, x1:x2]

        # Draw box and label on original image copy
        if box_coords is not None:
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(original_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_visual, "Cow", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Process panels
        left_panel = cv2.resize(original_visual, PANEL_SIZE)
        
        if cow_crop is not None and cow_crop.size > 0:
            right_panel = cv2.resize(cow_crop, PANEL_SIZE)
        else:
            # Fallback if YOLO misses: show resized original with red warning border
            right_panel = cv2.resize(img, PANEL_SIZE)
            cv2.rectangle(right_panel, (0, 0), (PANEL_SIZE[0]-1, PANEL_SIZE[1]-1), (0, 0, 255), 3)
            cv2.putText(right_panel, "No Cow Detected", (10, PANEL_SIZE[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Label both panels
        cv2.putText(left_panel, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(right_panel, "YOLO Crop", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Stitched row: horizontally concatenate Left and Right
        row_concat = np.hstack((left_panel, right_panel))
        rows_images.append(row_concat)

    if not rows_images:
        print("Error: No images were successfully processed.")
        return

    # Vertically concatenate all rows
    grid_visual = np.vstack(rows_images)

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE, grid_visual)
    print(f"\nSaved visualization grid to: {OUTPUT_IMAGE}")

    # Display in window
    try:
        print("Displaying crop samples. Press any key to close the window...")
        cv2.imshow("Cattle Crop Samples (Left: Original, Right: YOLO Crop)", grid_visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("\nNote: Running in headless mode. Bypassing window display.")

if __name__ == "__main__":
    main()
