# -*- coding: utf-8 -*-
"""
scripts/build_bcs_perception_contact_sheet.py
============================================
Generates 4-panel visual contact sheet:
[Original + BBox | Cattle Crop | SAM 2.1 Binary Mask | Guided Model Input]
across 5 representative ScienceDB BCS classes (3.25, 3.50, 3.75, 4.00, 4.25).
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def build_contact_sheet(
    manifest_csv: str = "artifacts/bcs_perception_smoke/cache/manifests/train_perception.csv",
    cache_dir: str = "artifacts/bcs_perception_smoke/cache",
    out_path: str = "docs/audits/assets/bcs_perception/bcs_perception_contact_sheet.jpg",
):
    df = pd.read_csv(manifest_csv)
    cache_path = Path(cache_dir)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Pick 1 sample per class (total 5 rows)
    classes = [3.25, 3.50, 3.75, 4.00, 4.25]
    selected_rows = []
    for c in classes:
        sub = df[df["label"] == c]
        if len(sub) > 0:
            selected_rows.append(sub.iloc[0])

    row_strips = []
    tile_h = 240
    tile_w = 320

    for row in selected_rows:
        orig_p = Path(row["image_path"])
        crop_p = cache_path / row["crop_rel_path"]
        mask_p = cache_path / row["mask_rel_path"]

        orig_bgr = cv2.imread(str(orig_p))
        crop_bgr = cv2.imread(str(crop_p))
        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # Panel 1: Original + BBox
        p1 = orig_bgr.copy()
        if row["detection_status"] == "detected" and pd.notna(row["bbox_x1"]):
            x1, y1, x2, y2 = int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
            cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 165, 255), 3) # Orange box
            cv2.putText(p1, f"Cow conf={row['confidence']:.2f}", (x1 + 5, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(p1, f"Orig (BCS {row['label']})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Panel 2: Crop
        p2 = crop_bgr.copy()
        cv2.putText(p2, f"Crop ({p2.shape[1]}x{p2.shape[0]})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Panel 3: Mask
        p3 = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(p3, f"SAM 2.1 Mask ({row['mask_area_ratio']*100:.1f}%)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Panel 4: Guided Model Input (RGB + Green mask overlay simulating 4-channel input)
        p4 = crop_bgr.copy()
        mask_bool = mask_gray > 127
        overlay = p4.copy()
        overlay[mask_bool] = [0, 230, 0] # Semi-transparent green
        cv2.addWeighted(overlay, 0.40, p4, 0.60, 0, p4)
        cv2.putText(p4, "4-Ch Guided Input", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Resize all to tile_w x tile_h
        panels = [
            cv2.resize(p1, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
            cv2.resize(p2, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
            cv2.resize(p3, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
            cv2.resize(p4, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
        ]
        strip = np.hstack(panels)
        row_strips.append(strip)

    # Master contact sheet: 5 rows
    grid = np.vstack(row_strips)

    # Top title banner
    banner_h = 50
    banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
    title_text = "ScienceDB Run 4 Perception Preprocessing: [Original + BBox | Crop | SAM 2.1 Mask | 4-Ch Guided Input]"
    cv2.putText(banner, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    contact_sheet = np.vstack([banner, grid])
    cv2.imwrite(str(out_file), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"✓ Saved contact sheet: {out_file} ({contact_sheet.shape[1]}x{contact_sheet.shape[0]} px)")

if __name__ == "__main__":
    build_contact_sheet()
