# -*- coding: utf-8 -*-
"""
Phase 3 Run 5: Behavior Perception Cache Builder (CVB + Kaggle Beef)
===================================================================
Generates the authentic perception smoke cache:
  - T=8 deterministic temporal frames per sequence
  - CVB:
    * Authentic target tracklet GT bbox for THAT specific frame
    * NO nearest-frame fallback in perception mode (missing exact bbox = failure)
    * Prompt SAM 2.1 Small using exact GT bbox
    * Spatially aligned RGB crop + binary cow mask crop resized to 224x224
    * NEVER choose another cow; NEVER use full-frame fallback
  - Kaggle Beef:
    * Single-cow 224x224 frame
    * RT-DETR-L cow detector (conf >= 0.25, COCO cow class 19)
    * A5 policy: largest detected cow bbox + positive center point -> SAM 2.1 Small
    * Fallback if no cow detected: image center point (112, 112) -> SAM 2.1 Small
    * Authentic 224x224 RGB frame + real SAM binary mask
  - Strict Data Integrity:
    * Binary masks saved with values {0, 255} (loaded as {0, 1} float tensors)
    * ZERO fake, zero-filled, or all-ones dummy masks
    * ZERO dummy black RGB sequences
    * Any unrecoverable frame/SAM failure marks sequence as failed and excludes it
    * Detailed frame-level audit log: perception_manifest.csv

Author: Hasin Ishrak
Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
Date: 2026-09-24
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================================================================
# DETERMINISTIC TEMPORAL SAMPLING
# ==============================================================================
def compute_sampled_frame_indices(start_frame: int, end_frame: int, num_frames: int = 8) -> List[int]:
    """
    Selects T approximately evenly spaced frames across [start_frame, end_frame].
    Strictly deterministic. Does NOT randomly sample adjacent frames.
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if start_frame >= end_frame:
        return [start_frame] * num_frames

    raw_points = np.linspace(start_frame, end_frame, num=num_frames, endpoint=True)
    indices = [int(round(p)) for p in raw_points]
    clamped = [max(start_frame, min(end_frame, idx)) for idx in indices]
    return clamped


def get_benchmark_sequence_subset(train_df: pd.DataFrame, max_candidates: int = 300) -> pd.DataFrame:
    """
    Selects a deterministic, representative, balanced interleaved subset of CVB and
    Kaggle Beef sequences from train.csv for speed benchmarking on L4 vs L40S.
    Guarantees:
      - Strictly deterministic (seed 2026)
      - Alternating mixture of CVB and Beef sequences
      - Balanced representation across all 5 canonical behaviors
      - CVB-only Walking behavior is properly represented
      - Exactly identical sequence order for both L4 and L40S benchmarks
    """
    cvb_df = train_df[train_df["dataset"] == "cvb"].copy().sample(frac=1.0, random_state=2026).reset_index(drop=True)
    beef_df = train_df[train_df["dataset"] == "beef_cattle_behavior"].copy().sample(frac=1.0, random_state=2026).reset_index(drop=True)

    classes = ["Standing", "Lying", "Feeding", "Drinking", "Walking"]
    cvb_by_class = {c: cvb_df[cvb_df["behavior_canonical"] == c].to_dict("records") for c in classes}
    beef_by_class = {c: beef_df[beef_df["behavior_canonical"] == c].to_dict("records") for c in classes}

    interleaved = []
    max_len = max(
        max((len(v) for v in cvb_by_class.values()), default=0),
        max((len(v) for v in beef_by_class.values()), default=0),
    )
    for i in range(max_len):
        for c in classes:
            if i < len(cvb_by_class[c]):
                interleaved.append(cvb_by_class[c][i])
                if len(interleaved) >= max_candidates:
                    break
            if c != "Walking" and i < len(beef_by_class[c]):
                interleaved.append(beef_by_class[c][i])
                if len(interleaved) >= max_candidates:
                    break
        if len(interleaved) >= max_candidates:
            break

    bench_df = pd.DataFrame(interleaved[:max_candidates]).reset_index(drop=True)
    return bench_df


def save_cache_manifest_and_summary(
    cache_dir: Path,
    frame_records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    retained_df: pd.DataFrame,
    failed_df: Optional[pd.DataFrame] = None,
    split_name: Optional[str] = None,
):
    """
    Persists progressive or final perception manifest and summary to cache_dir.
    Ensures interrupted caching sessions leave a consistent on-disk audit trail.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if frame_records:
        manifest_df = pd.DataFrame(frame_records)
        manifest_df.to_csv(cache_dir / "perception_manifest.csv", index=False)

    summary_data = {
        "stats": {k: v for k, v in stats.items() if k not in ("failed_records", "failed_df")},
        "total_sequences_requested": stats.get("total_requested", len(retained_df)),
        "total_sequences_retained": len(retained_df),
        "total_sequences_excluded": len(failed_df) if failed_df is not None else stats.get("failed_sequences", 0),
        "total_real_masks_generated": stats.get("total_real_masks_generated", 0),
        "cvb_frames_generated": stats.get("cvb_frames_generated", 0),
        "beef_a5_frames_generated": stats.get("beef_a5_frames_generated", 0),
        "beef_fallback_frames_generated": stats.get("beef_fallback_frames_generated", 0),
        "already_cached_count": stats.get("already_cached", 0),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_dir / "perception_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    if split_name:
        retained_df.to_csv(cache_dir / f"retained_{split_name}.csv", index=False)
        if failed_df is not None and len(failed_df) > 0:
            failed_df.to_csv(cache_dir / f"failed_{split_name}.csv", index=False)


# ==============================================================================
# CVB EXACT TARGET-TRACKLET ANNOTATION CACHE
# ==============================================================================
class CVBExactAnnotationCache:
    """
    Indexes CVB instances_default.json files per video cut.
    Provides EXACT per-frame target bbox lookup: (track_id, frame_idx) -> [x1, y1, x2, y2].
    CRITICAL: Perception mode strictly prohibits nearest-frame fallback.
    Missing exact bbox returns None and marks perception failure.
    """
    def __init__(self, cvb_dir: Path):
        self.cvb_dir = cvb_dir
        self.cached_cuts: Dict[str, Optional[Dict[Tuple[int, int], List[float]]]] = {}

    def get_cut_index(self, cut_name: str) -> Optional[Dict[Tuple[int, int], List[float]]]:
        if cut_name in self.cached_cuts:
            return self.cached_cuts[cut_name]

        ann_candidates = [
            self.cvb_dir / "data" / "annotations" / cut_name / "annotations" / "instances_default.json",
            self.cvb_dir / "annotations" / cut_name / "annotations" / "instances_default.json",
            self.cvb_dir / "data" / "annotations" / cut_name / "instances_default.json",
            self.cvb_dir / "annotations" / cut_name / "instances_default.json",
        ]
        ann_path = None
        for cand in ann_candidates:
            if cand.exists():
                ann_path = cand
                break

        if ann_path is None:
            self.cached_cuts[cut_name] = None
            return None

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load JSON for cut {cut_name}: {e}")
            self.cached_cuts[cut_name] = None
            return None

        # Build image_id -> frame_num map
        image_id_to_frame = {}
        for im in jdata.get("images", []):
            fname = im.get("file_name", "")
            base_fname = os.path.basename(fname)
            fnum = None
            if "img_" in base_fname:
                try:
                    fnum = int(base_fname.split("img_")[-1].split(".")[0])
                except ValueError:
                    pass
            if fnum is None:
                fnum = im.get("id")
            image_id_to_frame[im["id"]] = fnum

        # Index bboxes: (track_id, frame_num) -> [x1, y1, x2, y2]
        track_frame_bboxes = {}
        for ann in jdata.get("annotations", []):
            img_id = ann.get("image_id")
            fnum = image_id_to_frame.get(img_id)
            if fnum is None:
                continue
            attrs = ann.get("attributes", {})
            tr_id = attrs.get("track_id")
            if tr_id is not None:
                raw_box = ann.get("bbox", [])
                if len(raw_box) == 4:
                    gx, gy, gw, gh = raw_box
                    box = [float(gx), float(gy), float(gx + gw), float(gy + gh)]
                    track_frame_bboxes[(int(tr_id), int(fnum))] = box

        self.cached_cuts[cut_name] = track_frame_bboxes
        return track_frame_bboxes

    def get_exact_target_bbox(self, cut_name: str, track_id: int, frame_idx: int) -> Optional[List[float]]:
        cut_index = self.get_cut_index(cut_name)
        if cut_index is None:
            return None
        return cut_index.get((int(track_id), int(frame_idx)))


# ==============================================================================
# PATH RESOLVERS
# ==============================================================================
def resolve_cvb_frame_path(cvb_dir: Path, cut_name: str, frame_idx: int) -> Optional[Path]:
    candidates = [
        cvb_dir / "data" / "raw_frames" / cut_name / f"img_{frame_idx:05d}.jpg",
        cvb_dir / "data" / "raw_frames" / cut_name / f"img_{frame_idx}.jpg",
        cvb_dir / "raw_frames" / cut_name / f"img_{frame_idx:05d}.jpg",
        cvb_dir / "raw_frames" / cut_name / f"img_{frame_idx}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_beef_video_path(beef_dir: Path, rel_path: str) -> Optional[Path]:
    sub_path = rel_path.replace("clips/", "")
    candidates = [
        beef_dir / "Category Videos" / "cows" / sub_path,
        beef_dir / "clips" / sub_path,
        beef_dir / rel_path,
        beef_dir / "Category Videos" / "cows" / rel_path,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    fn = os.path.basename(rel_path)
    for root, _, files in os.walk(str(beef_dir)):
        if fn in files:
            return Path(root) / fn
    return None


# ==============================================================================
# PERCEPTION EXTRACTION: CVB
# ==============================================================================
def extract_cvb_perception_sequence(
    rec: dict,
    cvb_dir: Path,
    ann_cache: CVBExactAnnotationCache,
    sam_model: Any,
    num_frames: int = 8,
    device: str = "cuda",
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], List[Dict[str, Any]]]:
    """
    Extracts T=8 spatially aligned RGB cow crops and real SAM 2.1 binary masks for CVB.
    Rules:
      - Uses exact authentic target GT bbox for each sampled frame.
      - Never substitutes another cow or nearest frame.
      - Prompts SAM 2.1 Small using the exact target GT bbox on the full frame.
      - Crops both RGB and mask to target bbox, then resizes both to 224x224.
      - Requires non-empty mask (>0 pixels, <total pixels).
    Returns (rgb_crops, mask_crops, frame_records).
    If any frame fails, returns (None, None, frame_records).
    """
    cut_name = str(rec["session_id"])
    track_id = int(rec["tracklet_id"])
    start_f = int(rec["start_frame"])
    end_f = int(rec["end_frame"])
    sample_id = str(rec["sample_id"])

    frame_indices = compute_sampled_frame_indices(start_f, end_f, num_frames=num_frames)
    rgb_crops = []
    mask_crops = []
    frame_records = []

    for t_step, f_idx in enumerate(frame_indices):
        frame_meta = {
            "sample_id": sample_id,
            "dataset": "cvb",
            "t": t_step,
            "frame_index": f_idx,
            "tracklet_id": track_id,
            "bbox": None,
            "prompt_strategy": "cvb_gt_bbox",
            "fallback_used": False,
            "mask_success": False,
            "mask_pixels": 0,
            "mask_area_ratio": 0.0,
            "failure_reason": None,
        }

        # 1. Resolve raw 1080p frame
        img_path = resolve_cvb_frame_path(cvb_dir, cut_name, f_idx)
        if img_path is None:
            frame_meta["failure_reason"] = "frame_image_not_found"
            frame_records.append(frame_meta)
            return None, None, frame_records

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            frame_meta["failure_reason"] = "frame_decode_failed"
            frame_records.append(frame_meta)
            return None, None, frame_records

        img_h, img_w = img_bgr.shape[:2]

        # 2. Retrieve exact target GT bbox
        bbox = ann_cache.get_exact_target_bbox(cut_name, track_id, f_idx)
        if bbox is None:
            frame_meta["failure_reason"] = "exact_target_bbox_missing"
            frame_records.append(frame_meta)
            return None, None, frame_records

        # Sanitize bbox coordinates
        x1 = max(0, min(int(round(bbox[0])), img_w - 1))
        y1 = max(0, min(int(round(bbox[1])), img_h - 1))
        x2 = max(x1 + 1, min(int(round(bbox[2])), img_w))
        y2 = max(y1 + 1, min(int(round(bbox[3])), img_h))
        clean_box = [float(x1), float(y1), float(x2), float(y2)]
        frame_meta["bbox"] = json.dumps(clean_box)

        # 3. Prompt SAM 2.1 Small using target GT bbox on full frame
        try:
            sam_res = sam_model(img_bgr, bboxes=[clean_box], device=device, verbose=False)
            if sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
                raw_mask = sam_res[0].masks.data[0].cpu().numpy().astype(bool)
                if raw_mask.shape != (img_h, img_w):
                    full_mask = cv2.resize(raw_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    full_mask = raw_mask
            else:
                full_mask = None
        except Exception as e:
            frame_meta["failure_reason"] = f"sam_inference_error_{str(e)[:30]}"
            frame_records.append(frame_meta)
            return None, None, frame_records

        if full_mask is None or np.sum(full_mask) == 0:
            frame_meta["failure_reason"] = "sam_returned_empty_mask"
            frame_records.append(frame_meta)
            return None, None, frame_records

        # 4. Spatially aligned cow crop
        rgb_crop = img_bgr[y1:y2, x1:x2]
        mask_crop = full_mask[y1:y2, x1:x2]

        if np.sum(mask_crop) == 0:
            frame_meta["failure_reason"] = "crop_mask_empty"
            frame_records.append(frame_meta)
            return None, None, frame_records

        # 5. Resize both to 224x224
        rgb_224 = cv2.resize(rgb_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask_224 = cv2.resize(mask_crop.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)

        pix_count = int(np.sum(mask_224 > 0))
        if pix_count == 0 or pix_count == (224 * 224):
            frame_meta["failure_reason"] = "mask_trivial_zero_or_all_ones"
            frame_records.append(frame_meta)
            return None, None, frame_records

        frame_meta["mask_success"] = True
        frame_meta["mask_pixels"] = pix_count
        frame_meta["mask_area_ratio"] = round(float(pix_count) / (224 * 224), 4)

        rgb_crops.append(rgb_224)
        mask_crops.append((mask_224 > 0).astype(np.uint8))
        frame_records.append(frame_meta)

    return rgb_crops, mask_crops, frame_records


# ==============================================================================
# PERCEPTION EXTRACTION: KAGGLE BEEF
# ==============================================================================
def extract_beef_perception_sequence(
    rec: dict,
    beef_dir: Path,
    rtdetr_model: Any,
    sam_model: Any,
    num_frames: int = 8,
    device: str = "cuda",
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], List[Dict[str, Any]]]:
    """
    Extracts T=8 frames and real SAM 2.1 binary masks for Kaggle Beef clips.
    Rules:
      - Beef clips are already single-cow 224x224 crops.
      - Runs RT-DETR-L on the frame.
      - A5 policy: if cow detected, prompt SAM 2.1 with largest box + center point.
      - Fallback: if no cow detected, prompt SAM 2.1 with image center point (112, 112).
      - Zero fabricated crops or fake masks.
    Returns (rgb_frames, mask_frames, frame_records).
    If any frame fails, returns (None, None, frame_records).
    """
    rel_path = str(rec["source_path"])
    sample_id = str(rec["sample_id"])
    vid_path = resolve_beef_video_path(beef_dir, rel_path)

    frame_records = []
    if vid_path is None:
        frame_records.append({
            "sample_id": sample_id,
            "dataset": "beef_cattle_behavior",
            "t": 0,
            "frame_index": 0,
            "tracklet_id": None,
            "bbox": None,
            "prompt_strategy": "beef_unresolved",
            "fallback_used": False,
            "mask_success": False,
            "mask_pixels": 0,
            "mask_area_ratio": 0.0,
            "failure_reason": "video_file_not_found",
        })
        return None, None, frame_records

    cap = cv2.VideoCapture(str(vid_path))
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = int(rec.get("n_frames", 250))
    max_f = min(cap_frames - 1, n_frames - 1) if cap_frames > 0 else n_frames - 1

    frame_indices = compute_sampled_frame_indices(0, max(0, max_f), num_frames=num_frames)
    rgb_frames = []
    mask_frames = []

    for t_step, f_idx in enumerate(frame_indices):
        frame_meta = {
            "sample_id": sample_id,
            "dataset": "beef_cattle_behavior",
            "t": t_step,
            "frame_index": f_idx,
            "tracklet_id": None,
            "bbox": None,
            "prompt_strategy": None,
            "fallback_used": False,
            "mask_success": False,
            "mask_pixels": 0,
            "mask_area_ratio": 0.0,
            "failure_reason": None,
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            cap.release()
            frame_meta["failure_reason"] = "video_read_frame_failed"
            frame_records.append(frame_meta)
            return None, None, frame_records

        img_h, img_w = frame_bgr.shape[:2]

        # 1. Run RT-DETR-L detection
        try:
            rt_res = rtdetr_model(frame_bgr, conf=0.25, classes=[19], verbose=False, device=device)
            boxes_obj = rt_res[0].boxes
        except Exception as e:
            cap.release()
            frame_meta["failure_reason"] = f"rtdetr_inference_error_{str(e)[:30]}"
            frame_records.append(frame_meta)
            return None, None, frame_records

        # 2. Determine prompt strategy: A5 vs Fallback
        if len(boxes_obj) > 0:
            xyxy_arr = boxes_obj.xyxy.cpu().numpy()
            areas = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) * (xyxy_arr[:, 3] - xyxy_arr[:, 1])
            best_idx = int(np.argmax(areas))
            largest_box = [round(float(c), 1) for c in xyxy_arr[best_idx].tolist()]
            box_cx = round(float((largest_box[0] + largest_box[2]) / 2.0), 1)
            box_cy = round(float((largest_box[1] + largest_box[3]) / 2.0), 1)

            pts = np.array([[[box_cx, box_cy]]], dtype=np.float32)
            lbls = np.array([[1]], dtype=np.int32)
            prompt_strategy = "beef_A5_rtdetr_box_center_point"
            fallback_used = False
            used_bbox = largest_box

            try:
                sam_res = sam_model(
                    frame_bgr,
                    bboxes=[largest_box],
                    points=pts,
                    labels=lbls,
                    device=device,
                    verbose=False,
                )
            except Exception as e:
                cap.release()
                frame_meta["failure_reason"] = f"sam_a5_error_{str(e)[:30]}"
                frame_records.append(frame_meta)
                return None, None, frame_records
        else:
            # Fallback: center point (112, 112)
            cx, cy = 112.0, 112.0
            pts = np.array([[[cx, cy]]], dtype=np.float32)
            lbls = np.array([[1]], dtype=np.int32)
            prompt_strategy = "beef_center_point_fallback"
            fallback_used = True
            used_bbox = None

            try:
                sam_res = sam_model(
                    frame_bgr,
                    points=pts,
                    labels=lbls,
                    device=device,
                    verbose=False,
                )
            except Exception as e:
                cap.release()
                frame_meta["failure_reason"] = f"sam_fallback_error_{str(e)[:30]}"
                frame_records.append(frame_meta)
                return None, None, frame_records

        frame_meta["prompt_strategy"] = prompt_strategy
        frame_meta["fallback_used"] = fallback_used
        frame_meta["bbox"] = json.dumps(used_bbox) if used_bbox else None

        # 3. Parse SAM mask
        if sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
            raw_mask = sam_res[0].masks.data[0].cpu().numpy().astype(bool)
            if raw_mask.shape != (img_h, img_w):
                mask_224 = cv2.resize(raw_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_224 = raw_mask.astype(np.uint8)
        else:
            mask_224 = None

        if mask_224 is None or np.sum(mask_224) == 0:
            cap.release()
            frame_meta["failure_reason"] = "sam_returned_empty_mask"
            frame_records.append(frame_meta)
            return None, None, frame_records

        # Ensure 224x224
        if (img_h, img_w) != (224, 224):
            frame_224 = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
            mask_224 = cv2.resize(mask_224, (224, 224), interpolation=cv2.INTER_NEAREST)
        else:
            frame_224 = frame_bgr

        pix_count = int(np.sum(mask_224 > 0))
        if pix_count == 0 or pix_count == (224 * 224):
            cap.release()
            frame_meta["failure_reason"] = "mask_trivial_zero_or_all_ones"
            frame_records.append(frame_meta)
            return None, None, frame_records

        frame_meta["mask_success"] = True
        frame_meta["mask_pixels"] = pix_count
        frame_meta["mask_area_ratio"] = round(float(pix_count) / (224 * 224), 4)

        rgb_frames.append(frame_224)
        mask_frames.append((mask_224 > 0).astype(np.uint8))
        frame_records.append(frame_meta)

    cap.release()
    return rgb_frames, mask_frames, frame_records


# ==============================================================================
# BATCH PERCEPTION CACHE BUILDER
# ==============================================================================
def build_behavior_perception_cache(
    df: pd.DataFrame,
    cvb_dir: Path,
    beef_dir: Path,
    cache_dir: Path,
    rtdetr_model: Optional[Any] = None,
    sam_model: Optional[Any] = None,
    num_frames: int = 8,
    device: str = "cuda",
    desc: str = "Caching Perception Sequences",
    commit_callback: Optional[Any] = None,
    commit_interval_sec: float = 60.0,
    time_limit_sec: Optional[float] = None,
    clean_corrupt_folders: bool = True,
    save_progressive_manifest: bool = True,
    split_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Builds the authentic perception cache for df (train, val, or benchmark sequences).
    Saves:
      cache_dir / sample_id / frame_00.jpg ... frame_07.jpg
      cache_dir / sample_id / mask_00.png ... mask_07.png
      cache_dir / sample_id / perception_metadata.json
    Binary masks saved as {0, 255} uint8 PNGs.
    Strict failure policy: any failed sequence is excluded; zero dummy black or fake masks.
    Interruption-safe: validates on-disk sequences, purges corrupted partial folders,
    periodically commits persistent volume, and progressively saves manifests.
    Returns:
      retained_df, summary_dict, all_frame_records
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ann_cache = CVBExactAnnotationCache(cvb_dir)

    # Initialize models if not provided
    if rtdetr_model is None or sam_model is None:
        from ultralytics import RTDETR, SAM
        print(f"[*] Initializing RT-DETR-L and SAM 2.1 Small on {device}...")
        if rtdetr_model is None:
            rtdetr_model = RTDETR("rtdetr-l.pt")
        if sam_model is None:
            sam_model = SAM("sam2.1_s.pt")

    stats = {
        "total_requested": len(df),
        "already_cached": 0,
        "extracted_success": 0,
        "failed_sequences": 0,
        "cvb_frames_generated": 0,
        "beef_a5_frames_generated": 0,
        "beef_fallback_frames_generated": 0,
        "total_real_masks_generated": 0,
    }

    all_frame_records = []
    failed_records = []
    successful_indices = []

    last_commit_time = time.time()
    loop_start_time = time.perf_counter()

    print(f"[*] {desc}: Processing {len(df)} candidate sequences...")

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc, ncols=80)):
        sample_id = row["sample_id"]
        dataset_name = row["dataset"]
        sample_folder = cache_dir / sample_id

        # Check if already completely cached with valid perception_metadata.json
        meta_path = sample_folder / "perception_metadata.json"
        already_valid = False
        if sample_folder.exists() and meta_path.exists():
            frames_ok = all(
                (sample_folder / f"frame_{t:02d}.jpg").exists()
                and (sample_folder / f"frame_{t:02d}.jpg").stat().st_size > 0
                and (sample_folder / f"mask_{t:02d}.png").exists()
                and (sample_folder / f"mask_{t:02d}.png").stat().st_size > 0
                for t in range(num_frames)
            )
            if frames_ok:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        cached_records = json.load(f)
                    assert isinstance(cached_records, list) and len(cached_records) == num_frames, "Invalid record length"
                    # Strict provenance assertions: reject placeholder provenance
                    for rec_meta in cached_records:
                        assert rec_meta.get("prompt_strategy") not in ("beef_cached", None), "Placeholder prompt_strategy detected"
                        assert rec_meta.get("bbox") != "already_cached", "Placeholder bbox detected"
                        assert "frame_index" in rec_meta, "Missing frame_index"
                        assert rec_meta.get("mask_success") is True, "mask_success is False in cached record"

                    stats["already_cached"] += 1
                    stats["extracted_success"] += 1
                    successful_indices.append(idx)
                    all_frame_records.extend(cached_records)

                    for rec_meta in cached_records:
                        strat = rec_meta.get("prompt_strategy")
                        if strat == "cvb_gt_bbox":
                            stats["cvb_frames_generated"] += 1
                        elif strat == "beef_A5_rtdetr_box_center_point":
                            stats["beef_a5_frames_generated"] += 1
                        elif strat == "beef_center_point_fallback":
                            stats["beef_fallback_frames_generated"] += 1
                        stats["total_real_masks_generated"] += 1
                    already_valid = True
                except Exception as e:
                    print(f"[WARN] Sequence {sample_id} cache metadata invalid ({e}). Cleansing folder for fresh extraction.")
                    already_valid = False

        if already_valid:
            # Check periodic volume commit
            if commit_callback is not None and (time.time() - last_commit_time >= commit_interval_sec):
                current_retained = df.iloc[successful_indices].reset_index(drop=True)
                current_failed = pd.DataFrame(failed_records)
                if save_progressive_manifest:
                    save_cache_manifest_and_summary(cache_dir, all_frame_records, stats, current_retained, current_failed, split_name=split_name)
                try:
                    commit_callback(stats["extracted_success"], stats["already_cached"])
                except Exception as ce:
                    print(f"[WARN] Commit callback warning: {ce}")
                last_commit_time = time.time()
            continue

        # If sample_folder exists but was partial/corrupted, purge it cleanly
        if sample_folder.exists() and clean_corrupt_folders:
            shutil.rmtree(sample_folder, ignore_errors=True)

        # Real perception extraction
        rgb_seq = None
        mask_seq = None
        frame_records = []

        rec = row.to_dict()
        if dataset_name == "cvb":
            rgb_seq, mask_seq, frame_records = extract_cvb_perception_sequence(
                rec, cvb_dir, ann_cache, sam_model, num_frames=num_frames, device=device
            )
        elif dataset_name == "beef_cattle_behavior":
            rgb_seq, mask_seq, frame_records = extract_beef_perception_sequence(
                rec, beef_dir, rtdetr_model, sam_model, num_frames=num_frames, device=device
            )

        all_frame_records.extend(frame_records)

        if rgb_seq is not None and mask_seq is not None and len(rgb_seq) == num_frames and len(mask_seq) == num_frames:
            # Write frames and masks to disk
            sample_folder.mkdir(parents=True, exist_ok=True)
            for t in range(num_frames):
                out_rgb = sample_folder / f"frame_{t:02d}.jpg"
                out_mask = sample_folder / f"mask_{t:02d}.png"
                cv2.imwrite(str(out_rgb), rgb_seq[t], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(out_mask), (mask_seq[t] * 255).astype(np.uint8))

                # Update stats
                rec_meta = frame_records[t]
                if rec_meta["prompt_strategy"] == "cvb_gt_bbox":
                    stats["cvb_frames_generated"] += 1
                elif rec_meta["prompt_strategy"] == "beef_A5_rtdetr_box_center_point":
                    stats["beef_a5_frames_generated"] += 1
                elif rec_meta["prompt_strategy"] == "beef_center_point_fallback":
                    stats["beef_fallback_frames_generated"] += 1
                stats["total_real_masks_generated"] += 1

            # Persist authentic per-frame perception metadata
            meta_path = sample_folder / "perception_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(frame_records, f, indent=2)

            stats["extracted_success"] += 1
            successful_indices.append(idx)
        else:
            stats["failed_sequences"] += 1
            last_reason = frame_records[-1].get("failure_reason") if frame_records else "extraction_failed"
            failed_records.append({
                "sample_id": sample_id,
                "dataset": dataset_name,
                "behavior_canonical": row.get("behavior_canonical"),
                "failure_reason": last_reason,
            })
            print(f"[EXCLUDE] Sequence {sample_id} ({dataset_name}) failed perception ({last_reason}). Excluding from dataset.")

        # Periodic commit callback & progressive manifest update
        if commit_callback is not None and (time.time() - last_commit_time >= commit_interval_sec):
            current_retained = df.iloc[successful_indices].reset_index(drop=True)
            current_failed = pd.DataFrame(failed_records)
            if save_progressive_manifest:
                save_cache_manifest_and_summary(cache_dir, all_frame_records, stats, current_retained, current_failed, split_name=split_name)
            try:
                commit_callback(stats["extracted_success"], stats["already_cached"])
            except Exception as ce:
                print(f"[WARN] Commit callback warning: {ce}")
            last_commit_time = time.time()

        # Benchmark time limit check: break cleanly between sequences
        if time_limit_sec is not None:
            elapsed_bench = time.perf_counter() - loop_start_time
            if elapsed_bench >= time_limit_sec:
                print(f"[*] Benchmark time limit reached ({elapsed_bench:.1f}s >= {time_limit_sec:.1f}s). Stopping cleanly between sequences.")
                break

    retained_df = df.iloc[successful_indices].reset_index(drop=True)
    failed_df = pd.DataFrame(failed_records)
    stats["failed_records"] = failed_records
    stats["failed_df"] = failed_df

    if save_progressive_manifest:
        save_cache_manifest_and_summary(cache_dir, all_frame_records, stats, retained_df, failed_df, split_name=split_name)

    if commit_callback is not None:
        try:
            commit_callback(stats["extracted_success"], stats["already_cached"])
        except Exception as ce:
            print(f"[WARN] Final commit callback warning: {ce}")

    return retained_df, stats, all_frame_records
