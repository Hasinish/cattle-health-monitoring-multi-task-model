# -*- coding: utf-8 -*-
"""
crop_self_viewpoint_rtdetr.py — Generate Derived RT-DETR-L Cow-Cropped Real Viewpoint Dataset

Non-destructive preprocessing pipeline for:
  Source: datasets/viewpoint/self_clean_v1/ (880 clean images)
  Target: datasets/viewpoint/self_clean_v1_rtdetr_crop/
          front/
          side/
          rear/
          metadata/

Pipeline:
  1. Load clean manifest (880 included images).
  2. For each image, run RT-DETR-L cattle detection (COCO class 19, conf >= 0.25).
  3. Select primary cow:
     - 1 cow: use detection
     - >1 cows: choose largest cow box by area
     - 0 cows: mark as 'detection_failed', exclude from cropped dataset, record in failures CSV
  4. Expand bbox with small proportional margin (default 5% on all sides) to preserve extremities.
  5. Clamp to image boundaries and crop RGB cow region.
  6. Save crop in canonical class subfolder with standardized JPEG quality 95.
  7. Preserve 100% provenance including clean_id, class, duplicate_group_id, sha256, URL, domain.
  8. Output full manifest, failure CSV, audit report, and visual contact sheet.

Usage:
  python scripts/crop_self_viewpoint_rtdetr.py --smoke
  python scripts/crop_self_viewpoint_rtdetr.py
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import RTDETR

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_MANIFEST_PATH = ROOT_DIR / "datasets" / "viewpoint" / "self_clean_v1" / "metadata" / "manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "datasets" / "viewpoint" / "self_clean_v1_rtdetr_crop"
ASSETS_DIR = ROOT_DIR / "docs" / "audits" / "assets" / "viewpoint_self_clean_rtdetr_crops"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RT-DETR-L cow crops for real viewpoint dataset.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on a tiny balanced subset (15 images).")
    parser.add_argument("--smoke-samples-per-class", type=int, default=5, help="Samples per class in smoke mode.")
    parser.add_argument("--conf", type=float, default=0.25, help="RT-DETR detection confidence threshold.")
    parser.add_argument("--margin", type=float, default=0.05, help="Proportional bbox margin expansion (0.05 = 5%%).")
    parser.add_argument("--device", type=str, default="", help="Inference device ('cuda' or 'cpu'). Auto-detected if empty.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory path.")
    return parser.parse_args()


def load_clean_dataset(manifest_path: Path) -> pd.DataFrame:
    """Load canonical clean manifest and filter included rows."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    clean_df = df[df["inclusion_status"] == "included"].copy()
    if len(clean_df) != 880:
        print(f"Warning: Expected 880 included images, found {len(clean_df)}")
    return clean_df


def crop_with_margin(
    img_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
    margin_ratio: float = 0.05,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Expand bounding box by proportional margin and crop.
    box: (x1, y1, x2, y2)
    Returns: (crop_bgr, (cx1, cy1, cx2, cy2))
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio

    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(w, int(round(x2 + pad_x)))
    cy2 = min(h, int(round(y2 + pad_y)))

    crop = img_bgr[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)


def generate_contact_sheet(
    samples_info: List[Dict[str, Any]],
    output_path: Path,
    title: str = "RT-DETR-L Cattle Viewpoint Crop Quality Audit",
):
    """
    Generate visual contact sheet with original (bbox drawn) and final crop.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples_info)
    if n == 0:
        return

    tile_w, tile_h = 320, 240
    cols = 6
    rows = int(np.ceil(n / cols))
    banner_h = 60
    sheet_w = cols * tile_w
    sheet_h = rows * tile_h + banner_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(20, 22, 25))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except IOError:
        font = ImageFont.load_default()
        font_sm = font

    # Draw header banner
    draw.rectangle([(0, 0), (sheet_w, banner_h)], fill=(30, 34, 42))
    draw.text((15, 12), title, fill=(255, 255, 255), font=font)
    draw.text((15, 34), f"N={n} representative samples | Model: RT-DETR-L (COCO class 19) | Margin: 5% proportional", fill=(170, 185, 200), font=font_sm)

    for idx, s in enumerate(samples_info):
        r = idx // cols
        c = idx % cols
        x_off = c * tile_w
        y_off = banner_h + r * tile_h

        crop_path = s["crop_path"]
        if not os.path.exists(crop_path):
            continue

        crop_img = Image.open(crop_path).convert("RGB")
        # Resize to fit in tile preserving aspect ratio with letterbox
        c_w, c_h = crop_img.size
        scale = min((tile_w - 10) / c_w, (tile_h - 40) / c_h)
        nw, nh = max(1, int(c_w * scale)), max(1, int(c_h * scale))
        crop_res = crop_img.resize((nw, nh), Image.Resampling.LANCZOS)

        paste_x = x_off + (tile_w - nw) // 2
        paste_y = y_off + 5 + ((tile_h - 40) - nh) // 2
        sheet.paste(crop_res, (paste_x, paste_y))

        # Tile boundary
        draw.rectangle([(x_off, y_off), (x_off + tile_w - 1, y_off + tile_h - 1)], outline=(50, 55, 65), width=1)

        # Bottom label
        cls_color = (100, 220, 120) if s["class"] == "front" else ((100, 180, 255) if s["class"] == "side" else (255, 160, 90))
        label_text = f"{s['clean_id']} [{s['class'].upper()}]"
        sub_text = f"conf: {s['conf']:.2f} | cows: {s['cows']} | {s['crop_w']}x{s['crop_h']}"
        draw.rectangle([(x_off + 1, y_off + tile_h - 32), (x_off + tile_w - 2, y_off + tile_h - 2)], fill=(28, 30, 36))
        draw.text((x_off + 6, y_off + tile_h - 30), label_text, fill=cls_color, font=font_sm)
        draw.text((x_off + 6, y_off + tile_h - 16), sub_text, fill=(180, 185, 190), font=font_sm)

    sheet.save(str(output_path), quality=92)
    print(f"✓ Saved contact sheet: {output_path} ({sheet_w}x{sheet_h} px)")


def main():
    args = parse_args()

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_root = Path(args.output_dir)
    if args.smoke:
        output_root = output_root.parent / f"{output_root.name}_smoke"

    print("=" * 80)
    print("  RT-DETR-L Cattle Viewpoint Crop Generator")
    print(f"  Mode:            {'SMOKE TEST' if args.smoke else 'FULL RUN'}")
    print(f"  Device:          {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"  Confidence:      {args.conf}")
    print(f"  BBox Margin:     {args.margin * 100:.1f}% proportional expansion")
    print(f"  Source Manifest: {SOURCE_MANIFEST_PATH}")
    print(f"  Target Output:   {output_root}")
    print("=" * 80)

    # 1. Load clean dataset manifest
    df_clean = load_clean_dataset(SOURCE_MANIFEST_PATH)
    total_clean = len(df_clean)
    print(f"\n[1/5] Loaded source manifest: {total_clean} included images")
    print(f"  - front: {len(df_clean[df_clean['normalized_class'] == 'front'])}")
    print(f"  - side:  {len(df_clean[df_clean['normalized_class'] == 'side'])}")
    print(f"  - rear:  {len(df_clean[df_clean['normalized_class'] == 'rear'])}")

    # Handle smoke mode
    if args.smoke:
        smoke_rows = []
        for cls_name in ["front", "side", "rear"]:
            cls_df = df_clean[df_clean["normalized_class"] == cls_name]
            smoke_rows.append(cls_df.head(args.smoke_samples_per_class))
        df_clean = pd.concat(smoke_rows, ignore_index=True)
        print(f"  [SMOKE] Subset sampled: {len(df_clean)} images ({args.smoke_samples_per_class} per class)")

    # 2. Setup target directories
    for cls_name in ["front", "side", "rear"]:
        (output_root / cls_name).mkdir(parents=True, exist_ok=True)
    meta_dir = output_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load RT-DETR-L model
    print("\n[2/5] Loading RT-DETR-L model...")
    t_load_start = time.time()
    model = RTDETR("rtdetr-l.pt")
    # Warmup CUDA if applicable
    if device == "cuda":
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, conf=args.conf, classes=[19], device=device, verbose=False)
    print(f"✓ RT-DETR-L ready in {time.time() - t_load_start:.2f}s")

    # 4. Process each image with live tqdm
    print("\n[3/5] Executing cattle localization, margin cropping, and export...")

    crop_manifest_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    representative_samples: List[Dict[str, Any]] = []

    class_success_counts = {"front": 0, "side": 0, "rear": 0}
    multi_cow_count = 0
    total_images = len(df_clean)

    pbar = tqdm(
        df_clean.iterrows(),
        total=total_images,
        desc="RT-DETR cropping",
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    t_process_start = time.time()

    for idx, row in pbar:
        clean_id = str(row["clean_id"])
        norm_class = str(row["normalized_class"])
        src_clean_rel = str(row["clean_path"]).replace("\\", "/")
        src_clean_path = ROOT_DIR / src_clean_rel
        dup_group_id = str(row["duplicate_group_id"])
        source_sha256 = str(row["sha256"])
        source_url = str(row.get("source_url", ""))
        source_domain = str(row.get("source_domain", ""))

        if not src_clean_path.exists():
            failure_records.append({
                "clean_id": clean_id,
                "class": norm_class,
                "source_clean_path": src_clean_rel,
                "duplicate_group_id": dup_group_id,
                "source_sha256": source_sha256,
                "source_url": source_url,
                "source_domain": source_domain,
                "detection_status": "error_load",
                "detection_count": 0,
                "failure_reason": "source_file_not_found",
                "source_width": 0,
                "source_height": 0,
                "source_file_size_bytes": 0,
            })
            continue

        src_size_bytes = src_clean_path.stat().st_size
        img_bgr = cv2.imread(str(src_clean_path))
        if img_bgr is None:
            failure_records.append({
                "clean_id": clean_id,
                "class": norm_class,
                "source_clean_path": src_clean_rel,
                "duplicate_group_id": dup_group_id,
                "source_sha256": source_sha256,
                "source_url": source_url,
                "source_domain": source_domain,
                "detection_status": "error_load",
                "detection_count": 0,
                "failure_reason": "image_decode_failed",
                "source_width": 0,
                "source_height": 0,
                "source_file_size_bytes": src_size_bytes,
            })
            continue

        h, w = img_bgr.shape[:2]

        # Run RT-DETR-L detection (class 19 = cow in COCO)
        res = model(img_bgr, conf=args.conf, classes=[19], device=device, verbose=False)
        boxes_obj = res[0].boxes

        n_cows = len(boxes_obj)
        if n_cows == 0:
            failure_records.append({
                "clean_id": clean_id,
                "class": norm_class,
                "source_clean_path": src_clean_rel,
                "duplicate_group_id": dup_group_id,
                "source_sha256": source_sha256,
                "source_url": source_url,
                "source_domain": source_domain,
                "detection_status": "detection_failed",
                "detection_count": 0,
                "failure_reason": "zero_cattle_detected",
                "source_width": w,
                "source_height": h,
                "source_file_size_bytes": src_size_bytes,
            })
        else:
            if n_cows > 1:
                multi_cow_count += 1

            xyxy = boxes_obj.xyxy.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

            # Select largest cow bounding box by area
            primary_idx = int(np.argmax(areas))
            prim_box = (
                float(xyxy[primary_idx, 0]),
                float(xyxy[primary_idx, 1]),
                float(xyxy[primary_idx, 2]),
                float(xyxy[primary_idx, 3]),
            )
            prim_conf = float(confs[primary_idx])

            # Apply proportional margin and crop
            crop_bgr, (cx1, cy1, cx2, cy2) = crop_with_margin(img_bgr, prim_box, margin_ratio=args.margin)
            crop_h, crop_w = crop_bgr.shape[:2]

            # Save crop with standardized high quality JPEG (quality=95)
            crop_filename = f"{clean_id}.jpg"
            crop_dest_path = output_root / norm_class / crop_filename
            cv2.imwrite(str(crop_dest_path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            crop_size_bytes = crop_dest_path.stat().st_size

            crop_rel_path = f"datasets/viewpoint/{output_root.name}/{norm_class}/{crop_filename}"

            crop_manifest_records.append({
                "clean_id": clean_id,
                "class": norm_class,
                "source_clean_path": src_clean_rel,
                "crop_path": crop_rel_path,
                "duplicate_group_id": dup_group_id,
                "source_sha256": source_sha256,
                "source_url": source_url,
                "source_domain": source_domain,
                "detection_status": "success",
                "detection_count": n_cows,
                "selected_confidence": round(prim_conf, 4),
                "bbox_x1": round(prim_box[0], 2),
                "bbox_y1": round(prim_box[1], 2),
                "bbox_x2": round(prim_box[2], 2),
                "bbox_y2": round(prim_box[3], 2),
                "crop_x1": cx1,
                "crop_y1": cy1,
                "crop_x2": cx2,
                "crop_y2": cy2,
                "bbox_margin_rule": f"proportional_{int(args.margin * 100)}pct",
                "source_width": w,
                "source_height": h,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "source_file_size_bytes": src_size_bytes,
                "crop_file_size_bytes": crop_size_bytes,
            })

            class_success_counts[norm_class] += 1

            # Keep representative samples for contact sheet (up to 6 per class)
            if sum(1 for s in representative_samples if s["class"] == norm_class) < 6:
                representative_samples.append({
                    "clean_id": clean_id,
                    "class": norm_class,
                    "crop_path": str(crop_dest_path),
                    "conf": prim_conf,
                    "cows": n_cows,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                })

        # Update live tqdm stats
        n_succ = len(crop_manifest_records)
        n_fail = len(failure_records)
        pbar.set_postfix_str(
            f"F:{class_success_counts['front']} S:{class_success_counts['side']} R:{class_success_counts['rear']} | "
            f"succ={n_succ} fail={n_fail}"
        )

    t_process_total = time.time() - t_process_start
    print(f"\n✓ Inference & cropping completed in {t_process_total:.2f}s ({t_process_total / max(1, total_images) * 1000:.1f}ms/img)")

    # 5. Export metadata CSVs
    print("\n[4/5] Writing manifests and audit reports...")
    crop_df = pd.DataFrame(crop_manifest_records)
    fail_df = pd.DataFrame(failure_records)

    manifest_csv_path = meta_dir / "crop_manifest.csv"
    failures_csv_path = meta_dir / "detection_failures.csv"

    crop_df.to_csv(manifest_csv_path, index=False)
    fail_df.to_csv(failures_csv_path, index=False)

    print(f"✓ Saved crop manifest: {manifest_csv_path} ({len(crop_df)} rows)")
    print(f"✓ Saved failure report: {failures_csv_path} ({len(fail_df)} rows)")

    # 6. Calculate size reduction metrics
    total_src_bytes = sum(r["source_file_size_bytes"] for r in crop_manifest_records) + sum(r["source_file_size_bytes"] for r in failure_records)
    total_crop_bytes = sum(r["crop_file_size_bytes"] for r in crop_manifest_records)
    src_mb = total_src_bytes / (1024 * 1024)
    src_gb = total_src_bytes / (1024 * 1024 * 1024)
    crop_mb = total_crop_bytes / (1024 * 1024)
    crop_gb = total_crop_bytes / (1024 * 1024 * 1024)
    pct_reduction = ((total_src_bytes - total_crop_bytes) / max(1, total_src_bytes)) * 100.0

    mean_conf = float(crop_df["selected_confidence"].mean()) if len(crop_df) > 0 else 0.0
    median_conf = float(crop_df["selected_confidence"].median()) if len(crop_df) > 0 else 0.0

    # 7. Write crop report markdown
    report_md_path = meta_dir / "crop_report.md"
    report_content = f"""# RT-DETR-L Cattle Viewpoint Crop Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Mode:** {'SMOKE TEST' if args.smoke else 'FULL RUN'}  
**Source Dataset:** `datasets/viewpoint/self_clean_v1/`  
**Derived Target:** `datasets/viewpoint/{output_root.name}/`  
**Localization Model:** Pretrained RT-DETR-L (`rtdetr-l.pt`, COCO Class 19 Cattle)  
**Confidence Threshold:** {args.conf}  
**Margin Rule:** Proportional {int(args.margin * 100)}% on all bounding box boundaries  
**Provenance File:** `datasets/viewpoint/self_clean_v1/metadata/manifest.csv`

---

## 1. Executive Summary & Size Reduction

| Metric | Source (`self_clean_v1`) | Derived (`{output_root.name}`) | Delta / Reduction |
| :--- | :---: | :---: | :---: |
| **Total Images** | {total_images} | {len(crop_df)} successful crops | {len(fail_df)} failed detections |
| **Total Size (MB)** | {src_mb:.2f} MB | {crop_mb:.2f} MB | **-{pct_reduction:.1f}% reduction** |
| **Total Size (GB)** | {src_gb:.3f} GB | {crop_gb:.3f} GB | **-{pct_reduction:.1f}% reduction** |

---

## 2. Crop Counts by Viewpoint Class

| Class | Source Included | Successful Crops | Failed Detections | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **`front`** | {len(df_clean[df_clean['normalized_class'] == 'front'])} | {class_success_counts['front']} | {len(fail_df[fail_df['class'] == 'front']) if len(fail_df) > 0 else 0} | {(class_success_counts['front'] / max(1, len(df_clean[df_clean['normalized_class'] == 'front']))) * 100:.1f}% |
| **`side`** | {len(df_clean[df_clean['normalized_class'] == 'side'])} | {class_success_counts['side']} | {len(fail_df[fail_df['class'] == 'side']) if len(fail_df) > 0 else 0} | {(class_success_counts['side'] / max(1, len(df_clean[df_clean['normalized_class'] == 'side']))) * 100:.1f}% |
| **`rear`** | {len(df_clean[df_clean['normalized_class'] == 'rear'])} | {class_success_counts['rear']} | {len(fail_df[fail_df['class'] == 'rear']) if len(fail_df) > 0 else 0} | {(class_success_counts['rear'] / max(1, len(df_clean[df_clean['normalized_class'] == 'rear']))) * 100:.1f}% |
| **TOTAL** | **{total_images}** | **{len(crop_df)}** | **{len(fail_df)}** | **{(len(crop_df) / max(1, total_images)) * 100:.1f}%** |

---

## 3. Localization Diagnostics

- **Multi-Cow Images:** {multi_cow_count} ({multi_cow_count / max(1, total_images) * 100:.1f}% of samples contained >1 cattle detection; primary cow selected by maximum bbox area).
- **Mean Detector Confidence:** {mean_conf:.4f}
- **Median Detector Confidence:** {median_conf:.4f}
- **Detection Failures:** {len(fail_df)} (recorded in `metadata/detection_failures.csv`; excluded from training dataset).

---

## 4. Preprocessing & Crop Rule Specification

1. **Detection:** Pretrained RT-DETR-L (`rtdetr-l.pt`) applied to full RGB images with confidence threshold >= {args.conf} filtering COCO class 19 (`cow`).
2. **Target Selection:** For images with multiple cattle detections, the candidate bounding box maximizing physical area (x2 - x1) * (y2 - y1) is designated as the primary animal.
3. **Margin Expansion:** Bounding box coordinates [x1, y1, x2, y2] are expanded by a proportional margin of {int(args.margin * 100)}%:
   - pad_x = 0.05 * (x2 - x1)
   - pad_y = 0.05 * (y2 - y1)
   - cx1 = max(0, round(x1 - pad_x))
   - cy1 = max(0, round(y1 - pad_y))
   - cx2 = min(W, round(x2 + pad_x))
   - cy2 = min(H, round(y2 + pad_y))
4. **Encoding:** Cropped BGR arrays are encoded to JPEG format at Quality 95, preventing high-frequency compression artifacts while dramatically reducing storage footprint.
5. **Leakage Protection:** `duplicate_group_id` from the source manifest is strictly preserved in `crop_manifest.csv` for downstream split partition design.

---

## 5. Artifact Registry

- `metadata/crop_manifest.csv`: Master crop manifest with bounding box and dimensional provenance.
- `metadata/detection_failures.csv`: Complete record of images with zero RT-DETR cattle detections.
- `metadata/crop_report.md`: This comprehensive audit report.
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip() + "\n")
    print(f"✓ Saved crop report: {report_md_path}")

    # 8. Generate visual contact sheet
    print("\n[5/5] Generating visual contact sheet...")
    cs_name = "crop_quality_smoke_contact_sheet.jpg" if args.smoke else "crop_quality_contact_sheet.jpg"
    cs_path = ASSETS_DIR / cs_name
    generate_contact_sheet(
        representative_samples,
        cs_path,
        title=f"RT-DETR-L Cattle Viewpoint Crop Quality Audit ({'Smoke' if args.smoke else 'Full'})",
    )

    # 9. Print final summary scorecard
    print("\n" + "=" * 80)
    print("  RT-DETR-L CROP GENERATION SUMMARY")
    print("=" * 80)
    print(f"  Source Dataset:           {total_images} images ({src_mb:.2f} MB / {src_gb:.3f} GB)")
    print(f"  Derived Cropped Dataset:  {len(crop_df)} images ({crop_mb:.2f} MB / {crop_gb:.3f} GB)")
    print(f"  Storage Size Reduction:   -{pct_reduction:.1f}%")
    print(f"  Successful Crops:         {len(crop_df)} / {total_images} ({(len(crop_df) / max(1, total_images)) * 100:.1f}%)")
    print(f"    - front:                {class_success_counts['front']}")
    print(f"    - side:                 {class_success_counts['side']}")
    print(f"    - rear:                 {class_success_counts['rear']}")
    print(f"  Detection Failures:       {len(fail_df)}")
    print(f"  Multi-Cow Detections:     {multi_cow_count}")
    print(f"  Mean Confidence:          {mean_conf:.4f} (median: {median_conf:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
