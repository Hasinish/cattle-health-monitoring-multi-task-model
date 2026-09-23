# -*- coding: utf-8 -*-
"""
scripts/build_sciencedb_perception_cache.py — ScienceDB Perception Preprocessing Engine
=======================================================================================
Phase 3 Step 4.4 / Run 4: Generates frozen perception cache for ScienceDB BCS.

Pipeline:
  ScienceDB 1080p/576p Image
    -> RT-DETR-L Cow Localization (conf >= 0.25, COCO class 19)
    -> Deterministic Primary Cow Selection (max area, confidence tie-breaker)
    -> Crop with fixed documented 5% proportional margin
    -> SAM 2.1 instance segmentation using detector box prompt
    -> Save crop image (.jpg) + BINARY foreground mask (.png) + provenance metadata

Dataset Splits:
  Strictly preserves canonical ScienceDB split CSVs (train.csv, val.csv, test.csv)
  partitioned by 5,653 repaired burst groups. Zero alterations to split assignments.

Mask Semantics:
  Ultralytics SAM 2.1 returns binary boolean masks {False, True}.
  Stored as 8-bit PNG: 0 = background, 255 = cow foreground.
  Documented strictly as BINARY foreground mask guidance (never soft probability).

Failure Handling:
  - If zero RT-DETR detections: recorded explicitly as 'no_detection',
    sam_status='upstream_localization_failure'. No detector box is fabricated.
    Zero fake full-image crops or fake zero-masks are saved to disk.
  - If SAM fails to return mask: recorded explicitly as sam_status='sam_no_mask'.
    Zero fake zero-masks are saved to disk.
  - Manifest records 100% of samples and exact provenance.
  - Downstream training dataset strictly filters for:
    detection_status == 'detected' AND sam_status == 'segmented'

Resumability:
  - Skips already completed valid crop + mask pairs on disk.
  - Preserves existing manifest records and appends/updates incrementally.
  - Periodically commits manifest and progress.

Usage:
  # Local smoke test (10 samples across 5 classes):
  python scripts/build_sciencedb_perception_cache.py --smoke --cache-dir artifacts/bcs_perception_smoke/cache

  # Full generation on Modal cloud volume:
  python scripts/build_sciencedb_perception_cache.py --cache-dir /cache --data-dir /data/dataset
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path, PureWindowsPath
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

try:
    from ultralytics import RTDETR, SAM
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

REAL_BCS_CLASSES = [3.25, 3.50, 3.75, 4.00, 4.25]
COCO_COW_CLASS_ID = 19


def resolve_image_path(raw_path: str, data_root: Optional[Path] = None) -> Path:
    """Resolve raw image path across Windows laptop and Linux/Modal environments."""
    p = Path(raw_path)
    if data_root is None:
        if p.exists():
            return p
        if Path("datasets/bcs/sciencedb_bcs/dataset").exists():
            data_root = Path("datasets/bcs/sciencedb_bcs/dataset")
        elif Path("/data/dataset").exists():
            data_root = Path("/data/dataset")
        elif Path("/data").exists():
            data_root = Path("/data")

    if data_root is not None:
        data_root = Path(data_root)
        parts = PureWindowsPath(raw_path).parts
        if len(parts) >= 3 and parts[-3] == "dataset":
            c1 = data_root / parts[-3] / parts[-2] / parts[-1]
            if c1.exists():
                return c1
            c2 = data_root / parts[-2] / parts[-1]
            if c2.exists():
                return c2
            return c1
        elif len(parts) >= 2:
            candidate = data_root / parts[-2] / parts[-1]
            if candidate.exists():
                return candidate
            return candidate
    return p


def select_primary_cow(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    img_w: int,
    img_h: int,
    conf_thresh: float = 0.25,
    margin_ratio: float = 0.05,
) -> Optional[Dict]:
    """
    Select primary cow by maximum bounding box area, breaking ties with confidence.
    Expands box by documented proportional margin (default: 5%).
    """
    valid_mask = (classes == COCO_COW_CLASS_ID) & (scores >= conf_thresh)
    if not np.any(valid_mask):
        return None

    v_boxes = boxes_xyxy[valid_mask]
    v_scores = scores[valid_mask]

    areas = (v_boxes[:, 2] - v_boxes[:, 0]) * (v_boxes[:, 3] - v_boxes[:, 1])
    # Primary cow: sort by area descending, confidence descending
    sort_idx = np.lexsort((-v_scores, -areas))
    best_idx = sort_idx[0]

    b_x1, b_y1, b_x2, b_y2 = v_boxes[best_idx]
    b_conf = float(v_scores[best_idx])

    # 5% proportional margin expansion
    box_w = b_x2 - b_x1
    box_h = b_y2 - b_y1
    mw = margin_ratio * box_w
    mh = margin_ratio * box_h

    crop_x1 = max(0, int(round(b_x1 - mw)))
    crop_y1 = max(0, int(round(b_y1 - mh)))
    crop_x2 = min(img_w, int(round(b_x2 + mw)))
    crop_y2 = min(img_h, int(round(b_y2 + mh)))

    return {
        "box_xyxy": [float(b_x1), float(b_y1), float(b_x2), float(b_y2)],
        "crop_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "confidence": b_conf,
        "num_cow_detections": int(np.sum(valid_mask)),
    }


def process_image(
    img_bgr: np.ndarray,
    rtdetr_model: RTDETR,
    sam_model: SAM,
    conf_thresh: float = 0.25,
    margin_ratio: float = 0.05,
    device: str = "cuda",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Execute localization -> primary cow crop -> SAM 2.1 segmentation on single image.
    Returns: (crop_bgr, crop_mask_uint8, provenance_metadata)
    On failure: returns (None, None, meta) or (crop_bgr, None, meta). Zero fabricated boxes or masks.
    """
    img_h, img_w = img_bgr.shape[:2]

    # Step 1: RT-DETR-L Localization
    rtdetr_res = rtdetr_model(img_bgr, conf=conf_thresh, verbose=False)
    boxes = rtdetr_res[0].boxes
    if boxes is not None and len(boxes) > 0:
        b_xyxy = boxes.xyxy.cpu().numpy()
        b_conf = boxes.conf.cpu().numpy()
        b_cls = boxes.cls.cpu().numpy()
        primary_info = select_primary_cow(
            b_xyxy, b_conf, b_cls, img_w, img_h,
            conf_thresh=conf_thresh, margin_ratio=margin_ratio,
        )
    else:
        primary_info = None

    if primary_info is None:
        # Failure: No cow detected. Explicit failure recording; DO NOT fabricate box or crop/mask.
        meta = {
            "detection_status": "no_detection",
            "sam_status": "upstream_localization_failure",
            "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
            "crop_x1": None, "crop_y1": None, "crop_x2": None, "crop_y2": None,
            "confidence": 0.0,
            "num_cow_detections": 0,
            "mask_area_ratio": 0.0,
            "crop_w": None, "crop_h": None,
        }
        return None, None, meta

    # Step 2: SAM 2.1 Segmentation using detector box prompt
    raw_box = primary_info["box_xyxy"]
    crop_x1, crop_y1, crop_x2, crop_y2 = primary_info["crop_xyxy"]
    crop_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    sam_res = sam_model(img_bgr, bboxes=[raw_box], device=device, verbose=False)
    if sam_res[0].masks is not None and len(sam_res[0].masks) > 0:
        # SAM 2.1 data is boolean tensor of shape [1, img_h, img_w]
        raw_mask = sam_res[0].masks.data[0].cpu().numpy().astype(bool)
        # Crop mask to match image crop
        crop_mask_bool = raw_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = (crop_mask_bool.astype(np.uint8)) * 255
        sam_status = "segmented"
        mask_area_ratio = float(np.mean(crop_mask_bool))
    else:
        # SAM failure: Explicit failure recording; DO NOT fabricate zero mask.
        crop_mask = None
        sam_status = "sam_no_mask"
        mask_area_ratio = 0.0

    meta = {
        "detection_status": "detected",
        "sam_status": sam_status,
        "bbox_x1": round(raw_box[0], 1), "bbox_y1": round(raw_box[1], 1),
        "bbox_x2": round(raw_box[2], 1), "bbox_y2": round(raw_box[3], 1),
        "crop_x1": crop_x1, "crop_y1": crop_y1, "crop_x2": crop_x2, "crop_y2": crop_y2,
        "confidence": round(primary_info["confidence"], 4),
        "num_cow_detections": primary_info["num_cow_detections"],
        "mask_area_ratio": round(mask_area_ratio, 4),
        "crop_w": crop_img.shape[1], "crop_h": crop_img.shape[0],
    }
    return crop_img, crop_mask, meta


def run_cache_generation(
    data_dir: Path,
    splits_dir: Path,
    cache_dir: Path,
    rtdetr_weights: str = "rtdetr-l.pt",
    sam_weights: str = "sam2.1_s.pt",
    conf_thresh: float = 0.25,
    margin_ratio: float = 0.05,
    split_to_run: str = "all",
    smoke: bool = False,
    max_samples: Optional[int] = None,
    save_interval: int = 50,
    commit_callback: Optional[Callable[[], None]] = None,
    device: str = "cuda",
) -> Dict:
    """
    Main cache generation loop across splits with resumability, live tqdm, and explicit failure isolation.
    """
    print("\n" + "=" * 75)
    print("  SCIENTEDB PERCEPTION CACHE GENERATION PIPELINE")
    print(f"  Target Volume / Directory: {cache_dir}")
    print(f"  Splits Directory:          {splits_dir}")
    print(f"  Mode:                      {'SMOKE (Sampled)' if smoke else 'FULL'}")
    print(f"  Device:                    {device}")
    print("=" * 75)

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = cache_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Perception Models
    print(f"\n[1/3] Loading RT-DETR-L ({rtdetr_weights})...", flush=True)
    rtdetr_model = RTDETR(rtdetr_weights)

    print(f"[2/3] Loading SAM 2.1 ({sam_weights})...", flush=True)
    sam_model = SAM(sam_weights)

    splits = ["train", "val", "test"] if split_to_run == "all" else [split_to_run]
    overall_stats = {}

    for split in splits:
        csv_file = splits_dir / f"{split}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing canonical split CSV: {csv_file}")

        df = pd.read_csv(csv_file)
        print(f"\n[3/3] Processing split '{split}' ({len(df)} total canonical samples)...", flush=True)

        if smoke:
            # Deterministic stratified 2 samples per BCS class for smoke testing
            samples_per_class = 2 if max_samples is None else max(1, max_samples // len(REAL_BCS_CLASSES))
            subsets = []
            for bcs in REAL_BCS_CLASSES:
                sub = df[df["label"] == bcs].head(samples_per_class)
                subsets.append(sub)
            df_proc = pd.concat(subsets, ignore_index=True)
            print(f"  [*] Smoke mode active: processing {len(df_proc)} samples across all 5 BCS classes.")
        elif max_samples:
            df_proc = df.head(max_samples)
        else:
            df_proc = df

        split_crop_dir = cache_dir / "crops" / split
        split_mask_dir = cache_dir / "masks" / split
        split_crop_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        # Resumability: check existing manifest records
        manifest_csv = manifest_dir / f"{split}_perception.csv"
        existing_records = {}
        if manifest_csv.exists():
            try:
                ex_df = pd.read_csv(manifest_csv)
                for _, ex_row in ex_df.iterrows():
                    existing_records[str(ex_row["image_path"])] = ex_row.to_dict()
                print(f"  [*] Found existing manifest with {len(existing_records)} records. Checking for resumability...")
            except Exception as e:
                print(f"  [!] Notice: could not parse existing manifest ({e}), starting fresh for {split}.")

        records_map = dict(existing_records)
        n_skipped = 0
        n_detected = 0
        n_segmented = 0
        n_det_fail = 0
        n_sam_fail = 0

        t0 = time.time()
        pbar = tqdm(
            df_proc.iterrows(),
            total=len(df_proc),
            desc=f"Cache [{split:5s}]",
            file=sys.stdout,
            dynamic_ncols=True,
        )

        for i, (_, row) in enumerate(pbar, 1):
            raw_path = str(row["image_path"])
            bcs_label = float(row["label"])
            burst_group = str(row["burst_group_id"])
            resolved_p = resolve_image_path(raw_path, data_root=data_dir)

            if not resolved_p.exists():
                raise FileNotFoundError(f"Image not found: {resolved_p} (raw: {raw_path})")

            # Determine class-specific output subdirectory
            cls_str = f"{bcs_label:.2f}"
            (split_crop_dir / cls_str).mkdir(parents=True, exist_ok=True)
            (split_mask_dir / cls_str).mkdir(parents=True, exist_ok=True)

            stem = resolved_p.stem
            out_crop_rel = f"crops/{split}/{cls_str}/{stem}.jpg"
            out_mask_rel = f"masks/{split}/{cls_str}/{stem}.png"
            out_crop_abs = cache_dir / out_crop_rel
            out_mask_abs = cache_dir / out_mask_rel

            # Resumption check: if already processed and files exist on disk
            if raw_path in existing_records:
                ex = existing_records[raw_path]
                det_st = ex.get("detection_status")
                sam_st = ex.get("sam_status")
                if det_st == "detected" and sam_st == "segmented":
                    c_p = cache_dir / str(ex.get("crop_rel_path", ""))
                    m_p = cache_dir / str(ex.get("mask_rel_path", ""))
                    if c_p.exists() and c_p.stat().st_size > 0 and m_p.exists() and m_p.stat().st_size > 0:
                        n_skipped += 1
                        n_detected += 1
                        n_segmented += 1
                        pbar.set_postfix(skip=n_skipped, det=n_detected, seg=n_segmented, fail=n_det_fail + n_sam_fail)
                        continue
                elif det_st == "detected":
                    # RT-DETR succeeded, but SAM failed
                    n_skipped += 1
                    n_detected += 1
                    n_sam_fail += 1
                    pbar.set_postfix(skip=n_skipped, det=n_detected, seg=n_segmented, fail=n_det_fail + n_sam_fail)
                    continue
                else:
                    # Detection failure: RT-DETR found no valid cow (upstream localization failure)
                    n_skipped += 1
                    n_det_fail += 1
                    pbar.set_postfix(skip=n_skipped, det=n_detected, seg=n_segmented, fail=n_det_fail + n_sam_fail)
                    continue

            img_bgr = cv2.imread(str(resolved_p))
            if img_bgr is None:
                raise IOError(f"Failed to read image at {resolved_p}")

            crop_img, crop_mask, meta = process_image(
                img_bgr, rtdetr_model, sam_model,
                conf_thresh=conf_thresh, margin_ratio=margin_ratio, device=device,
            )

            # Only save files to disk for successful outputs (never fabricate crops or masks)
            if meta["detection_status"] == "detected":
                n_detected += 1
                cv2.imwrite(str(out_crop_abs), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_crop_rel = out_crop_rel
                if meta["sam_status"] == "segmented":
                    n_segmented += 1
                    cv2.imwrite(str(out_mask_abs), crop_mask)
                    saved_mask_rel = out_mask_rel
                else:
                    # RT-DETR succeeded, but SAM failed
                    n_sam_fail += 1
                    saved_mask_rel = None
            else:
                # RT-DETR found no valid cow (upstream localization failure)
                n_det_fail += 1
                saved_crop_rel = None
                saved_mask_rel = None

            rec = {
                "image_path": raw_path,
                "label": bcs_label,
                "burst_group_id": burst_group,
                "crop_rel_path": saved_crop_rel,
                "mask_rel_path": saved_mask_rel,
                **meta,
            }
            records_map[raw_path] = rec
            pbar.set_postfix(skip=n_skipped, det=n_detected, seg=n_segmented, fail=n_det_fail + n_sam_fail)

            # Periodic manifest save & commit
            if i % save_interval == 0:
                ordered_records = [records_map[str(rp)] for rp in df_proc["image_path"] if str(rp) in records_map]
                pd.DataFrame(ordered_records).to_csv(manifest_csv, index=False)
                if commit_callback is not None:
                    try:
                        commit_callback()
                    except Exception:
                        pass

        # Split complete: order records exactly matching df_proc
        ordered_records = [records_map[str(rp)] for rp in df_proc["image_path"] if str(rp) in records_map]
        out_df = pd.DataFrame(ordered_records)
        out_df.to_csv(manifest_csv, index=False)
        if commit_callback is not None:
            try:
                commit_callback()
            except Exception:
                pass

        print(f"\n  ✓ Saved split manifest: {manifest_csv} ({len(out_df)} records, {n_skipped} skipped/reused)")
        print(f"  ✓ Split Perception Summary: Detected={n_detected} | SegmentedSuccess={n_segmented} | DetectionFail={n_det_fail} | SamFail={n_sam_fail}", flush=True)

        split_summary = {
            "split": split,
            "total_processed": len(df_proc),
            "skipped_reused": n_skipped,
            "detected": n_detected,
            "detection_failure": n_det_fail,
            "sam_failure": n_sam_fail,
            "segmented_success": n_segmented,
            "detection_failures": n_det_fail,
            "sam_failures": n_sam_fail,
            "segmented": n_segmented,
            "det_rate_pct": round(n_detected / len(df_proc) * 100, 2) if len(df_proc) > 0 else 0,
            "seg_success_rate_pct": round(n_segmented / len(df_proc) * 100, 2) if len(df_proc) > 0 else 0,
            "seg_rate_pct": round(n_segmented / len(df_proc) * 100, 2) if len(df_proc) > 0 else 0,
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        overall_stats[split] = split_summary

    summary_file = manifest_dir / "cache_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2)
    print(f"\n✓ Saved cache summary: {summary_file}")
    if commit_callback is not None:
        try:
            commit_callback()
        except Exception:
            pass
    print("=" * 75)
    return overall_stats


def main():
    parser = argparse.ArgumentParser(description="ScienceDB Perception Cache Builder")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing ScienceDB dataset images")
    parser.add_argument("--splits-dir", type=str, default="datasets/bcs/sciencedb", help="Directory containing train/val/test.csv")
    parser.add_argument("--cache-dir", type=str, default="artifacts/bcs_perception_smoke/cache", help="Output cache directory")
    parser.add_argument("--rtdetr-weights", type=str, default="rtdetr-l.pt", help="RT-DETR-L weights path")
    parser.add_argument("--sam-weights", type=str, default="sam2.1_s.pt", help="SAM 2.1 weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="RT-DETR detection confidence threshold")
    parser.add_argument("--margin", type=float, default=0.05, help="Proportional crop margin expansion ratio")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"], help="Split to process")
    parser.add_argument("--smoke", action="store_true", help="Smoke test on small subset (10 samples across 5 classes)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    parser.add_argument("--device", type=str, default=None, help="Computation device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_cache_generation(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        splits_dir=Path(args.splits_dir),
        cache_dir=Path(args.cache_dir),
        rtdetr_weights=args.rtdetr_weights,
        sam_weights=args.sam_weights,
        conf_thresh=args.conf,
        margin_ratio=args.margin,
        split_to_run=args.split,
        smoke=args.smoke,
        max_samples=args.max_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
