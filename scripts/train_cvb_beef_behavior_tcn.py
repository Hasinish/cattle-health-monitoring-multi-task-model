# -*- coding: utf-8 -*-
"""
Phase 3 Run 5: Behavior Temporal Core (ResNet-18 + 1D TCN) Training Pipeline
===========================================================================
Implements the lightweight temporal backbone for Phase 3 Run 5 behavior monitoring:
  video/track segment
  → deterministic short frame sequence (T = 8 frames)
  → ResNet-18 frame features (512-D per frame)
  → lightweight 1D TCN (2 temporal Conv1d blocks max + temporal pooling)
  → 5-class behavior prediction (Standing, Lying, Feeding, Drinking, Walking)

Scope & Scientific Guardrails:
1. Smoke / Plumbing Verification: Proves temporal sampling, tensor shapes,
   TCN forward/backward, checkpoint save/resume, and train/val split isolation.
2. Temporal Sampler: Strictly deterministic. T = 8 approximately evenly spaced
   frames selected across [start_frame, end_frame].
   - CVB: Preserves target tracklet identity. Uses authentic per-frame CVB
     annotation bboxes for the target tracklet. Crops the SAME target cow at every
     sampled frame. If a sampled frame lacks a valid target bbox, records it
     explicitly; does NOT silently substitute another cow.
   - Kaggle Beef: Clips are already single-cow crops; samples 8 frames directly.
   - All crops resized to 224x224 RGB.
3. Model Architecture:
   - Backbone: ImageNet-pretrained ResNet-18 → 512-D per frame.
   - Sequence shape: [B, T, 512].
   - Temporal TCN: Exactly 2 temporal Conv1d blocks with GELU, BatchNorm1d,
     dropout, residual skip connections, AdaptiveAvgPool1d, and Linear head.
     NO GRU, NO LSTM, NO Transformer, NO VideoMAE, NO SlowFast.
   - Extensibility: FrameFeatureExtractor supports in_channels=4 so binary mask
     guidance can be added without rewriting the TCN.
4. Masks Unfinalized: Does NOT fabricate masks. Does NOT use fake all-ones or
   zero masks. RGB-only is NOT claimed as the final Run 5 model.
5. Strict Test Isolation: test.csv is NEVER loaded or evaluated in smoke mode.

Author: Hasin Ishrak
Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
Date: 2026-09-24
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Canonical Behavior Taxonomy
CANONICAL_CLASSES = ["Standing", "Lying", "Feeding", "Drinking", "Walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CANONICAL_CLASSES)}
NUM_CLASSES = len(CANONICAL_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    # Clamp to [start_frame, end_frame]
    clamped = [max(start_frame, min(end_frame, idx)) for idx in indices]
    return clamped


# ==============================================================================
# CVB ANNOTATION CACHE & INDEXING
# ==============================================================================
class CVBAnnotationCache:
    """
    Parses and indexes CVB instances_default.json files per video cut.
    Enables fast per-frame bounding box retrieval for specific tracklets.
    Guarantees that target cow identity is strictly preserved.
    """
    def __init__(self, cvb_dir: Path):
        self.cvb_dir = cvb_dir
        self.cached_cuts: Dict[str, Optional[Dict]] = {}
        self.missing_target_bbox_log: List[Dict[str, Any]] = []

    def get_cut_index(self, cut_name: str) -> Optional[Dict]:
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

        # Build fast lookup: image_id -> frame_num
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

        # Index bboxes:
        # (track_id, frame_num) -> [x1, y1, x2, y2]
        # track_id -> {frame_num: [x1, y1, x2, y2]}
        track_frame_bboxes = {}
        track_all_frames = defaultdict(dict)

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
                    track_all_frames[int(tr_id)][int(fnum)] = box

        cut_index = {
            "track_frame_bboxes": track_frame_bboxes,
            "track_all_frames": track_all_frames,
        }
        self.cached_cuts[cut_name] = cut_index
        return cut_index

    def get_target_bbox_for_frame(
        self,
        cut_name: str,
        track_id: int,
        frame_idx: int,
        sample_id: str,
    ) -> Optional[List[float]]:
        """
        Retrieves authentic GT bbox for (track_id, frame_idx).
        If missing for that frame, falls back to the nearest frame of the SAME target tracklet.
        Records any missing bbox explicitly. NEVER substitutes another tracklet.
        """
        cut_index = self.get_cut_index(cut_name)
        if cut_index is None:
            self.missing_target_bbox_log.append({
                "sample_id": sample_id,
                "cut_name": cut_name,
                "track_id": track_id,
                "frame": frame_idx,
                "reason": "cut_annotation_json_missing",
            })
            return None

        # 1. Exact match
        key = (int(track_id), int(frame_idx))
        if key in cut_index["track_frame_bboxes"]:
            return cut_index["track_frame_bboxes"][key]

        # 2. Frame-level missing: fallback to nearest frame of the SAME target tracklet
        track_frames = cut_index["track_all_frames"].get(int(track_id), {})
        if track_frames:
            nearest_frame = min(track_frames.keys(), key=lambda f: abs(f - frame_idx))
            fallback_box = track_frames[nearest_frame]
            self.missing_target_bbox_log.append({
                "sample_id": sample_id,
                "cut_name": cut_name,
                "track_id": track_id,
                "frame": frame_idx,
                "reason": f"frame_missing_target_bbox_used_nearest_frame_{nearest_frame}",
                "delta_frames": abs(nearest_frame - frame_idx),
            })
            return fallback_box

        # 3. Complete track missing
        self.missing_target_bbox_log.append({
            "sample_id": sample_id,
            "cut_name": cut_name,
            "track_id": track_id,
            "frame": frame_idx,
            "reason": "track_id_not_found_in_cut_annotations",
        })
        return None


# ==============================================================================
# SEQUENCE EXTRACTION FUNCTIONS
# ==============================================================================
def resolve_cvb_frame_path(cvb_dir: Path, cut_name: str, frame_idx: int) -> Optional[Path]:
    """Finds raw frame image path for CVB."""
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


def extract_cvb_sequence(
    rec: dict,
    cvb_dir: Path,
    ann_cache: CVBAnnotationCache,
    num_frames: int = 8,
) -> Optional[List[np.ndarray]]:
    """
    Extracts T deterministic target cow RGB crops for a CVB track segment.
    Preserves track identity across all T frames.
    """
    cut_name = str(rec["session_id"])
    track_id = int(rec["tracklet_id"])
    start_f = int(rec["start_frame"])
    end_f = int(rec["end_frame"])
    sample_id = str(rec["sample_id"])

    frame_indices = compute_sampled_frame_indices(start_f, end_f, num_frames=num_frames)
    crops_bgr = []

    for f_idx in frame_indices:
        img_path = resolve_cvb_frame_path(cvb_dir, cut_name, f_idx)
        if img_path is None:
            # Fallback to nearest valid frame if available
            return None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        img_h, img_w = img_bgr.shape[:2]
        bbox = ann_cache.get_target_bbox_for_frame(cut_name, track_id, f_idx, sample_id)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(round(x1)), img_w - 1))
            y1 = max(0, min(int(round(y1)), img_h - 1))
            x2 = max(x1 + 1, min(int(round(x2)), img_w))
            y2 = max(y1 + 1, min(int(round(y2)), img_h))
            crop = img_bgr[y1:y2, x1:x2]
        else:
            # Target bbox unavailable; use full frame fallback
            crop = img_bgr

        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops_bgr.append(crop_resized)

    return crops_bgr


def extract_beef_sequence(
    rec: dict,
    beef_dir: Path,
    num_frames: int = 8,
) -> Optional[List[np.ndarray]]:
    """
    Extracts T deterministic frames from a single-cow Kaggle Beef clip.
    """
    rel_path = str(rec["source_path"])
    sub_path = rel_path.replace("clips/", "")

    vid_candidates = [
        beef_dir / "Category Videos" / "cows" / sub_path,
        beef_dir / "clips" / sub_path,
        beef_dir / rel_path,
        beef_dir / "Category Videos" / "cows" / rel_path,
    ]
    vid_path = None
    for cand in vid_candidates:
        if cand.exists():
            vid_path = cand
            break

    if vid_path is None:
        fn = os.path.basename(rel_path)
        for root, _, files in os.walk(str(beef_dir)):
            if fn in files:
                vid_path = Path(root) / fn
                break

    if vid_path is None:
        return None

    cap = cv2.VideoCapture(str(vid_path))
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = int(rec.get("n_frames", 250))
    if cap_frames > 0:
        max_f = min(cap_frames - 1, n_frames - 1)
    else:
        max_f = n_frames - 1

    frame_indices = compute_sampled_frame_indices(0, max(0, max_f), num_frames=num_frames)
    crops_bgr = []
    last_valid_frame = None

    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            crop_resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
            last_valid_frame = crop_resized
            crops_bgr.append(crop_resized)
        elif last_valid_frame is not None:
            crops_bgr.append(last_valid_frame)
        else:
            cap.release()
            return None

    cap.release()
    return crops_bgr


# ==============================================================================
# PERSISTENT TEMPORAL CACHING
# ==============================================================================
def build_temporal_cache(
    df: pd.DataFrame,
    cvb_dir: Path,
    beef_dir: Path,
    cache_dir: Path,
    num_frames: int = 8,
    desc: str = "Caching Temporal Sequences",
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Extracts and caches T=8 frames (224x224 JPEGs) for every sample in df.
    Cached structure: cache_dir / sample_id / frame_00.jpg ... frame_07.jpg
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ann_cache = CVBAnnotationCache(cvb_dir)

    stats = {
        "total": len(df),
        "already_cached": 0,
        "extracted": 0,
        "failed": 0,
    }

    records_to_process = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        sample_folder = cache_dir / sample_id
        all_frames_exist = (
            sample_folder.exists()
            and all((sample_folder / f"frame_{t:02d}.jpg").exists() and (sample_folder / f"frame_{t:02d}.jpg").stat().st_size > 0 for t in range(num_frames))
        )
        if all_frames_exist:
            stats["already_cached"] += 1
        else:
            records_to_process.append(row.to_dict())

    if not records_to_process:
        print(f"[*] {desc}: All {len(df)} sequences already cached in {cache_dir}.")
        return stats, ann_cache.missing_target_bbox_log

    print(f"[*] {desc}: Extracting {len(records_to_process)} sequences ({stats['already_cached']} cached)...")

    for rec in tqdm(records_to_process, desc=desc, ncols=80):
        sample_id = rec["sample_id"]
        sample_folder = cache_dir / sample_id
        sample_folder.mkdir(parents=True, exist_ok=True)
        dataset_name = rec["dataset"]

        crops_bgr = None
        if dataset_name == "cvb":
            crops_bgr = extract_cvb_sequence(rec, cvb_dir, ann_cache, num_frames=num_frames)
        elif dataset_name == "beef_cattle_behavior":
            crops_bgr = extract_beef_sequence(rec, beef_dir, num_frames=num_frames)

        if crops_bgr is not None and len(crops_bgr) == num_frames:
            for t, crop in enumerate(crops_bgr):
                out_p = sample_folder / f"frame_{t:02d}.jpg"
                cv2.imwrite(str(out_p), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            stats["extracted"] += 1
        else:
            print(f"[WARN] Extraction failed for {sample_id} ({dataset_name}). Creating fallback sequence.")
            for t in range(num_frames):
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(dummy, f"{sample_id[:12]}_t{t}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                out_p = sample_folder / f"frame_{t:02d}.jpg"
                cv2.imwrite(str(out_p), dummy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            stats["failed"] += 1

    return stats, ann_cache.missing_target_bbox_log


# ==============================================================================
# PYTORCH DATASET WITH SYNCHRONIZED TEMPORAL AUGMENTATION
# ==============================================================================
# ==============================================================================
# PYTORCH DATASET WITH SYNCHRONIZED TEMPORAL AUGMENTATION
# ==============================================================================
class BehaviorTemporalDataset(Dataset):
    """
    Loads pre-extracted T=8 frame sequences from cache.
    Outputs:
      images: [T, C, H, W] where C=3 for 'rgb' mode and C=4 for 'rgb_mask' mode
      target: int class label (0..4)
    Supports synchronized temporal transforms (e.g., all T frames flipped together).
    In perception mode ('rgb_mask'):
      - missing RGB or mask -> hard failure (no black image fallback)
      - empty or all-ones mask -> hard failure
      - synchronized flip on RGB AND mask
      - color jitter on RGB ONLY
      - ImageNet normalize on RGB ONLY; mask is strictly {0.0, 1.0} float
    Supports in-memory RAM preloading (preload_ram=True):
      - Preloads all sequences into compact uint8 tensors in RAM via ThreadPoolExecutor
      - Completely eliminates network volume / NFS I/O thrashing during training epochs
      - Accelerates epoch throughput from minutes down to ~12-15 seconds per epoch.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
        num_frames: int = 8,
        is_train: bool = False,
        input_mode: str = "rgb_mask",
        preload_ram: bool = True,
        num_preload_workers: int = 32,
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.is_train = is_train
        self.input_mode = input_mode
        self.preload_ram = preload_ram

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Integrity check for perception mode if NOT preloading into RAM
        if self.input_mode == "rgb_mask" and not self.preload_ram:
            missing_items = []
            for _, row in self.df.iterrows():
                sid = str(row["sample_id"])
                s_dir = self.cache_dir / sid
                if not s_dir.exists():
                    missing_items.append(f"{sid} (folder missing)")
                    continue
                for t in range(self.num_frames):
                    f_p = s_dir / f"frame_{t:02d}.jpg"
                    m_p = s_dir / f"mask_{t:02d}.png"
                    if not f_p.exists() or f_p.stat().st_size == 0 or not m_p.exists() or m_p.stat().st_size == 0:
                        missing_items.append(f"{sid}_t{t}")
                        break
            if missing_items:
                raise RuntimeError(
                    f"Perception cache integrity check FAILED: {len(missing_items)} missing/empty sequence items in {self.cache_dir}. "
                    f"First missing items: {missing_items[:5]}. No silent black or dummy fallback allowed in perception mode!"
                )

        self.cached_tensors: Optional[List[torch.Tensor]] = None
        if self.preload_ram:
            self._preload_into_ram(num_workers=num_preload_workers)

    def _preload_into_ram(self, num_workers: int = 32):
        t0 = time.perf_counter()
        split_name = "Train" if self.is_train else "Val"
        desc = f"Preloading {split_name} sequences into RAM"
        sids = [str(r["sample_id"]) for _, r in self.df.iterrows()]
        num_seqs = len(sids)
        self.cached_tensors = [None] * num_seqs

        def _load_single(idx: int, sid: str) -> Tuple[int, torch.Tensor]:
            s_dir = self.cache_dir / sid
            if not s_dir.exists():
                raise RuntimeError(f"Sample folder missing at {s_dir}")

            if self.input_mode == "rgb_mask":
                seq_np = np.empty((self.num_frames, 4, 224, 224), dtype=np.uint8)
                for t in range(self.num_frames):
                    f_p = s_dir / f"frame_{t:02d}.jpg"
                    m_p = s_dir / f"mask_{t:02d}.png"
                    if not f_p.exists() or f_p.stat().st_size == 0 or not m_p.exists() or m_p.stat().st_size == 0:
                        raise RuntimeError(f"Missing/empty frame or mask for {sid} at t={t}")

                    bgr = cv2.imread(str(f_p), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"Failed to decode RGB frame {f_p}")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    mask_cv = cv2.imread(str(m_p), cv2.IMREAD_GRAYSCALE)
                    if mask_cv is None:
                        raise RuntimeError(f"Failed to decode mask {m_p}")

                    pix_count = int(np.sum(mask_cv > 127))
                    if pix_count == 0:
                        raise RuntimeError(f"Empty mask detected for {sid} at t={t}. Zero masks strictly prohibited!")
                    if pix_count == mask_cv.size:
                        raise RuntimeError(f"All-ones mask detected for {sid} at t={t}. All-one masks strictly prohibited!")

                    mask_binary = (mask_cv > 127).astype(np.uint8)
                    seq_np[t, :3, :, :] = rgb.transpose(2, 0, 1)
                    seq_np[t, 3, :, :] = mask_binary
                return idx, torch.from_numpy(seq_np)
            elif self.input_mode == "rgb":
                seq_np = np.empty((self.num_frames, 3, 224, 224), dtype=np.uint8)
                for t in range(self.num_frames):
                    f_p = s_dir / f"frame_{t:02d}.jpg"
                    if f_p.exists():
                        bgr = cv2.imread(str(f_p), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        else:
                            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                    seq_np[t, :3, :, :] = rgb.transpose(2, 0, 1)
                return idx, torch.from_numpy(seq_np)
            else:
                raise ValueError(f"Unknown input_mode: {self.input_mode}")

        max_workers = max(1, min(num_workers, 32))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_load_single, i, sid): i
                for i, sid in enumerate(sids)
            }
            for fut in tqdm(concurrent.futures.as_completed(future_to_idx), total=num_seqs, desc=desc, ncols=80):
                res_idx, res_tensor = fut.result()
                self.cached_tensors[res_idx] = res_tensor

        assert all(t is not None for t in self.cached_tensors), "RAM preload incomplete: some tensors are None!"
        elapsed = time.perf_counter() - t0
        total_mb = sum(t.element_size() * t.nelement() for t in self.cached_tensors) / (1024 * 1024)
        rate = num_seqs / max(0.1, elapsed)
        print(f"[*] RAM Preload Complete: {num_seqs} sequences ({total_mb:.1f} MB) into memory in {elapsed:.1f}s ({rate:.1f} seq/s).")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        dataset_name = row["dataset"]
        behavior_cls = row["behavior_canonical"]
        target = CLASS_TO_IDX[behavior_cls]

        # Fast in-memory path (zero disk seeks, vectorized transforms)
        if self.cached_tensors is not None:
            seq = self.cached_tensors[idx]  # [T, C, 224, 224] uint8
            if self.input_mode == "rgb_mask":
                # 1. Synchronized Random Horizontal Flip (RGB AND Mask)
                if self.is_train and torch.rand(1).item() < 0.5:
                    seq = torch.flip(seq, dims=[-1])

                rgb = seq[:, :3, :, :]
                mask = seq[:, 3:4, :, :].to(dtype=torch.float32)

                # 2. Sequence-Consistent Color Jitter (RGB ONLY; never alter mask)
                if self.is_train and torch.rand(1).item() < 0.5:
                    b_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    c_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    rgb = TF.adjust_brightness(rgb, b_factor)
                    rgb = TF.adjust_contrast(rgb, c_factor)

                rgb_norm = TF.normalize(
                    rgb.to(dtype=torch.float32).div(255.0),
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                )
                seq_tensor = torch.cat([rgb_norm, mask], dim=1)  # [T, 4, 224, 224] float32
                return seq_tensor, target, sample_id, dataset_name

            elif self.input_mode == "rgb":
                if self.is_train and torch.rand(1).item() < 0.5:
                    seq = torch.flip(seq, dims=[-1])
                rgb = seq[:, :3, :, :]
                if self.is_train and torch.rand(1).item() < 0.5:
                    b_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    c_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    rgb = TF.adjust_brightness(rgb, b_factor)
                    rgb = TF.adjust_contrast(rgb, c_factor)
                seq_tensor = TF.normalize(
                    rgb.to(dtype=torch.float32).div(255.0),
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                )
                return seq_tensor, target, sample_id, dataset_name
            else:
                raise ValueError(f"Unknown input_mode: {self.input_mode}")

        # Fallback disk path (when preload_ram=False)
        sample_folder = self.cache_dir / sample_id

        if self.input_mode == "rgb":
            frames: List[Image.Image] = []
            for t in range(self.num_frames):
                frame_path = sample_folder / f"frame_{t:02d}.jpg"
                if frame_path.exists():
                    try:
                        img = Image.open(frame_path).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (224, 224), (0, 0, 0))
                else:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                frames.append(img)

            if self.is_train:
                if torch.rand(1).item() < 0.5:
                    frames = [TF.hflip(f) for f in frames]
                if torch.rand(1).item() < 0.5:
                    brightness_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    contrast_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    frames = [TF.adjust_brightness(f, brightness_factor) for f in frames]
                    frames = [TF.adjust_contrast(f, contrast_factor) for f in frames]

            tensor_frames = [self.normalize(TF.to_tensor(f)) for f in frames]
            seq_tensor = torch.stack(tensor_frames, dim=0)  # [T, 3, 224, 224]
            return seq_tensor, target, sample_id, dataset_name

        elif self.input_mode == "rgb_mask":
            frames_rgb = []
            frames_mask = []
            for t in range(self.num_frames):
                frame_path = sample_folder / f"frame_{t:02d}.jpg"
                mask_path = sample_folder / f"mask_{t:02d}.png"

                if not frame_path.exists() or not mask_path.exists():
                    raise RuntimeError(f"Missing perception frame or mask for {sample_id} at t={t}")

                try:
                    img = Image.open(frame_path).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"Failed to read RGB frame {frame_path}: {e}")

                mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_cv is None:
                    raise RuntimeError(f"Failed to read binary mask {mask_path}")

                pix_count = int(np.sum(mask_cv > 127))
                if pix_count == 0:
                    raise RuntimeError(f"Empty mask detected for {sample_id} at t={t}. Zero masks strictly prohibited!")
                if pix_count == mask_cv.size:
                    raise RuntimeError(f"All-ones mask detected for {sample_id} at t={t}. All-one masks strictly prohibited!")

                mask_pil = Image.fromarray(mask_cv)
                frames_rgb.append(img)
                frames_mask.append(mask_pil)

            if self.is_train:
                if torch.rand(1).item() < 0.5:
                    frames_rgb = [TF.hflip(f) for f in frames_rgb]
                    frames_mask = [TF.hflip(m) for m in frames_mask]

                if torch.rand(1).item() < 0.5:
                    b_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    c_factor = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
                    frames_rgb = [TF.adjust_brightness(f, b_factor) for f in frames_rgb]
                    frames_rgb = [TF.adjust_contrast(f, c_factor) for f in frames_rgb]

            tensor_frames_4ch = []
            for f_rgb, f_mask in zip(frames_rgb, frames_mask):
                t_rgb = TF.to_tensor(f_rgb)          # [3, 224, 224], range [0, 1]
                t_rgb = self.normalize(t_rgb)        # ImageNet normalization
                m_np = np.array(f_mask)
                t_mask = torch.from_numpy((m_np > 127).astype(np.float32)).unsqueeze(0)  # [1, 224, 224], strictly {0.0, 1.0}
                t_4ch = torch.cat([t_rgb, t_mask], dim=0)  # [4, 224, 224]
                tensor_frames_4ch.append(t_4ch)

            seq_tensor = torch.stack(tensor_frames_4ch, dim=0)  # [T=8, 4, 224, 224]
            return seq_tensor, target, sample_id, dataset_name
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}. Choose 'rgb' or 'rgb_mask'.")




# ==============================================================================
# MODEL ARCHITECTURE: RESNET-18 + 1D TCN
# ==============================================================================
class FrameFeatureExtractor(nn.Module):
    """
    Extracts 512-D spatial features from each frame using ImageNet-pretrained ResNet-18.
    Architected to support in_channels=4 (for upcoming binary mask guidance) cleanly.
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = self.resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    # Initialize mask channel from channel mean of RGB weights
                    new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
            self.resnet.conv1 = new_conv

        self.feature_dim = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] -> returns [N, 512]
        return self.resnet(x)


class TemporalConvNet(nn.Module):
    """
    Lightweight 1D Temporal Convolutional Network.
    Constraints:
      - Exactly 2 temporal Conv1d blocks maximum
      - GELU activation
      - Dropout (0.2)
      - Temporal pooling (AdaptiveAvgPool1d)
      - Linear classifier
      - NO GRU, NO LSTM, NO Transformer, NO VideoMAE, NO SlowFast
    """
    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Block 1: in_features -> hidden_dim
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.res1 = nn.Conv1d(in_features, hidden_dim, kernel_size=1) if in_features != hidden_dim else nn.Identity()

        # Block 2: hidden_dim -> hidden_dim
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Temporal pooling & classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, T, in_features]
        # Transpose to 1D convolution format: [B, in_features, T]
        x = x.transpose(1, 2)

        # Block 1 with residual projection
        res = self.res1(x)
        x = self.drop1(self.act1(self.bn1(self.conv1(x)))) + res

        # Block 2 with identity residual
        res = x
        x = self.drop2(self.act2(self.bn2(self.conv2(x)))) + res

        # Temporal pooling: [B, hidden_dim, 1] -> [B, hidden_dim]
        pooled = self.pool(x).squeeze(-1)

        # Linear classifier: [B, num_classes]
        logits = self.classifier(pooled)
        return logits


class BehaviorTemporalModel(nn.Module):
    """
    Combined Behavior Temporal Backbone:
      Input: [B, T, C, H, W]
      Backbone: ResNet-18 per frame -> [B, T, 512]
      TCN: 2 Conv1d blocks -> [B, 5]
    """
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.backbone = FrameFeatureExtractor(in_channels=in_channels, pretrained=pretrained_backbone)
        self.tcn = TemporalConvNet(
            in_features=self.backbone.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats_flat = self.backbone(x_flat)  # [B*T, 512]
        feats = feats_flat.view(B, T, -1)   # [B, T, 512]
        logits = self.tcn(feats)            # [B, num_classes]
        return logits, feats


# ==============================================================================
# METRICS & EVALUATION
# ==============================================================================
def compute_behavior_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_names: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive behavior metrics including source breakdowns."""
    overall_acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    prec_per = precision_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    per_class = {}
    for i, c in enumerate(CANONICAL_CLASSES):
        sup = int((y_true == i).sum())
        per_class[c] = {
            "precision": round(float(prec_per[i]), 4),
            "recall": round(float(rec_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": sup,
        }

    cvb_metrics = None
    beef_metrics = None

    if dataset_names is not None:
        cvb_mask = (dataset_names == "cvb")
        if cvb_mask.sum() > 0:
            yt = y_true[cvb_mask]
            yp = y_pred[cvb_mask]
            cvb_metrics = {
                "support": int(cvb_mask.sum()),
                "overall_accuracy": round(float(accuracy_score(yt, yp)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            }

        beef_mask = (dataset_names == "beef_cattle_behavior")
        if beef_mask.sum() > 0:
            yt = y_true[beef_mask]
            yp = y_pred[beef_mask]
            beef_metrics = {
                "support": int(beef_mask.sum()),
                "overall_accuracy": round(float(accuracy_score(yt, yp)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            }

    return {
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf_mat,
        "cvb_metrics": cvb_metrics,
        "beef_metrics": beef_metrics,
        "walking_is_cvb_only": True,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[Dict[str, Any], float, torch.Tensor]:
    """Evaluates model over DataLoader, returning metrics, mean loss, and concatenated logits."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_datasets = []
    all_logits = []

    with torch.no_grad():
        for seq_images, targets, _, d_names in tqdm(loader, desc=desc, ncols=80, leave=False):
            seq_images = seq_images.to(device)
            targets = targets.to(device)

            logits, _ = model(seq_images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_datasets.extend(list(d_names))
            all_logits.append(logits.cpu())

    mean_loss = total_loss / max(1, len(all_targets))
    metrics = compute_behavior_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        dataset_names=np.array(all_datasets),
    )
    concat_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0)
    return metrics, mean_loss, concat_logits


# ==============================================================================
# CONTACT SHEET GENERATION
# ==============================================================================
def generate_temporal_contact_sheet(
    df: pd.DataFrame,
    cache_dir: Path,
    out_path: Path,
    num_samples: int = 6,
    num_frames: int = 8,
) -> bool:
    """
    Generates a visual contact sheet displaying several representative 8-frame sequences:
    Ordered left-to-right (t=0 to t=7) with metadata banner:
    dataset source, behavior, sample_id, target tracklet_id.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select representative samples covering both datasets and multiple behaviors
    selected_rows = []
    # 1. Try to get 1 walking (CVB)
    walk_rows = df[df["behavior_canonical"] == "Walking"]
    if len(walk_rows) > 0:
        selected_rows.append(walk_rows.iloc[0])

    # 2. Try to get CVB Feeding and Lying
    for beh in ["Feeding", "Lying", "Standing", "Drinking"]:
        sub_cvb = df[(df["dataset"] == "cvb") & (df["behavior_canonical"] == beh)]
        if len(sub_cvb) > 0 and len(selected_rows) < num_samples // 2 + 1:
            selected_rows.append(sub_cvb.iloc[0])

    # 3. Try to get Beef samples across different behaviors
    for beh in ["Feeding", "Lying", "Drinking", "Standing"]:
        sub_beef = df[(df["dataset"] == "beef_cattle_behavior") & (df["behavior_canonical"] == beh)]
        if len(sub_beef) > 0 and len(selected_rows) < num_samples:
            selected_rows.append(sub_beef.iloc[0])

    # Fill remaining if needed
    for _, row in df.iterrows():
        if len(selected_rows) >= num_samples:
            break
        if not any(r["sample_id"] == row["sample_id"] for r in selected_rows):
            selected_rows.append(row)

    if not selected_rows:
        return False

    tile_w, tile_h = 224, 224
    header_h = 36
    row_h = tile_h + header_h
    total_w = tile_w * num_frames
    total_h = row_h * len(selected_rows)

    sheet = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for i, row in enumerate(selected_rows):
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        beh = str(row["behavior_canonical"])
        track_id = row.get("tracklet_id", "N/A")
        y_offset = i * row_h

        # Header banner
        header_text = f"[{dataset.upper()}] Class: {beh} | Tracklet ID: {track_id} | Sample: {sample_id}"
        cv2.rectangle(sheet, (0, y_offset), (total_w, y_offset + header_h), (30, 30, 30), -1)
        cv2.putText(
            sheet,
            header_text,
            (10, y_offset + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 8 Frames left-to-right
        sample_folder = cache_dir / sample_id
        for t in range(num_frames):
            frame_path = sample_folder / f"frame_{t:02d}.jpg"
            x_offset = t * tile_w
            frame_bgr = None
            if frame_path.exists():
                frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                frame_bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(frame_bgr, f"Missing t={t}", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                frame_bgr = cv2.resize(frame_bgr, (tile_w, tile_h))

            # Add frame index indicator
            cv2.putText(
                frame_bgr,
                f"t={t}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Thin border
            cv2.rectangle(frame_bgr, (0, 0), (tile_w - 1, tile_h - 1), (60, 60, 60), 1)

            sheet[y_offset + header_h : y_offset + row_h, x_offset : x_offset + tile_w] = frame_bgr

    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[*] Temporal contact sheet saved: {out_path} ({total_w}x{total_h} px, {len(selected_rows)} sequences)")
    return True


def generate_perception_contact_sheet(
    df: pd.DataFrame,
    cache_dir: Path,
    out_path: Path,
    num_samples: int = 6,
    num_frames: int = 8,
) -> bool:
    """
    Generates visual contact sheet for perception mode showing aligned RGB + SAM mask:
    For each representative sequence:
      - Header banner: [DATASET] Class | Tracklet ID | Sample ID
      - Row A: Clean RGB crop (t=0..7)
      - Row B: RGB + SAM 2.1 mask overlay (green transparent fill + contour, t=0..7)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    walk_rows = df[df["behavior_canonical"] == "Walking"]
    if len(walk_rows) > 0:
        selected_rows.append(walk_rows.iloc[0])

    for beh in ["Feeding", "Lying", "Standing", "Drinking"]:
        sub_cvb = df[(df["dataset"] == "cvb") & (df["behavior_canonical"] == beh)]
        if len(sub_cvb) > 0 and len(selected_rows) < num_samples // 2 + 1:
            selected_rows.append(sub_cvb.iloc[0])

    for beh in ["Feeding", "Lying", "Drinking", "Standing"]:
        sub_beef = df[(df["dataset"] == "beef_cattle_behavior") & (df["behavior_canonical"] == beh)]
        if len(sub_beef) > 0 and len(selected_rows) < num_samples:
            selected_rows.append(sub_beef.iloc[0])

    for _, row in df.iterrows():
        if len(selected_rows) >= num_samples:
            break
        if not any(r["sample_id"] == row["sample_id"] for r in selected_rows):
            selected_rows.append(row)

    if not selected_rows:
        return False

    tile_w, tile_h = 224, 224
    header_h = 32
    seq_h = header_h + tile_h * 2  # Header + RGB row + Mask overlay row
    total_w = tile_w * num_frames
    total_h = seq_h * len(selected_rows)

    sheet = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for i, row in enumerate(selected_rows):
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        beh = str(row["behavior_canonical"])
        track_id = row.get("tracklet_id", "N/A")
        y_seq = i * seq_h

        # Header banner
        header_text = f"[{dataset.upper()}] Class: {beh} | Tracklet: {track_id} | Sample: {sample_id}"
        cv2.rectangle(sheet, (0, y_seq), (total_w, y_seq + header_h), (35, 35, 35), -1)
        cv2.putText(sheet, header_text, (10, y_seq + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        sample_folder = cache_dir / sample_id
        for t in range(num_frames):
            x_offset = t * tile_w
            f_path = sample_folder / f"frame_{t:02d}.jpg"
            m_path = sample_folder / f"mask_{t:02d}.png"

            frame_bgr = cv2.imread(str(f_path)) if f_path.exists() else None
            mask_gray = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE) if m_path.exists() else None

            if frame_bgr is None:
                frame_bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            else:
                frame_bgr = cv2.resize(frame_bgr, (tile_w, tile_h))

            # Row A: Clean RGB
            row_a_y = y_seq + header_h
            p_rgb = frame_bgr.copy()
            cv2.putText(p_rgb, f"t={t} RGB", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(p_rgb, (0, 0), (tile_w - 1, tile_h - 1), (60, 60, 60), 1)
            sheet[row_a_y : row_a_y + tile_h, x_offset : x_offset + tile_w] = p_rgb

            # Row B: RGB + SAM 2.1 Mask Overlay
            row_b_y = row_a_y + tile_h
            p_mask = frame_bgr.copy()
            if mask_gray is not None and np.sum(mask_gray > 127) > 0:
                mask_bool = mask_gray > 127
                overlay = p_mask.copy()
                overlay[mask_bool] = (overlay[mask_bool] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
                p_mask = overlay
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(p_mask, contours, -1, (0, 255, 0), 1)
                ar = float(np.sum(mask_bool)) / (tile_w * tile_h)
                cv2.putText(p_mask, f"t={t} Mask (ar={ar:.2f})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(p_mask, f"t={t} NO MASK", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(p_mask, (0, 0), (tile_w - 1, tile_h - 1), (60, 60, 60), 1)
            sheet[row_b_y : row_b_y + tile_h, x_offset : x_offset + tile_w] = p_mask

    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[*] Perception contact sheet saved: {out_path} ({total_w}x{total_h} px, {len(selected_rows)} sequences)")
    return True


# ==============================================================================
# SHARED DETERMINISTIC SMOKE SUBSET SELECTOR
# ==============================================================================
def get_balanced_smoke_subset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministically selects the shared canonical 40-sequence balanced smoke subset:
      - Train: 30 sequences (6 per class)
        * For Standing, Lying, Feeding, Drinking: 3 Beef + 3 CVB = 6 per class
        * For Walking: 6 CVB = 6
      - Val: 10 sequences (2 per class)
        * For Standing, Lying, Feeding, Drinking: 1 Beef + 1 CVB = 2 per class
        * For Walking: 2 CVB = 2
    Deterministic: uses df head/filtering on canonical splits.
    Guarantees selected cache IDs == selected training IDs.
    """
    smoke_train = []
    for c in CANONICAL_CLASSES:
        if c == "Walking":
            sub = train_df[(train_df["behavior_canonical"] == c) & (train_df["dataset"] == "cvb")]
            smoke_train.append(sub.head(6))
        else:
            sub_beef = train_df[(train_df["behavior_canonical"] == c) & (train_df["dataset"] == "beef_cattle_behavior")]
            sub_cvb = train_df[(train_df["behavior_canonical"] == c) & (train_df["dataset"] == "cvb")]
            smoke_train.append(sub_beef.head(3))
            smoke_train.append(sub_cvb.head(3))
    train_sub = pd.concat(smoke_train).reset_index(drop=True)

    smoke_val = []
    for c in CANONICAL_CLASSES:
        if c == "Walking":
            sub = val_df[(val_df["behavior_canonical"] == c) & (val_df["dataset"] == "cvb")]
            smoke_val.append(sub.head(2))
        else:
            sub_beef = val_df[(val_df["behavior_canonical"] == c) & (val_df["dataset"] == "beef_cattle_behavior")]
            sub_cvb = val_df[(val_df["behavior_canonical"] == c) & (val_df["dataset"] == "cvb")]
            smoke_val.append(sub_beef.head(1))
            smoke_val.append(sub_cvb.head(1))
    val_sub = pd.concat(smoke_val).reset_index(drop=True)

    return train_sub, val_sub


# ==============================================================================
# REPRODUCIBILITY & PROVENANCE HELPERS
# ==============================================================================
def set_reproducibility_seeds(seed: int = 2026):
    """
    Configures deterministic seeds across Python random, NumPy, and PyTorch (CPU & CUDA)
    and enforces cuDNN deterministic behavior consistent with thesis methodology.
    """
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_file_sha256(file_path: Optional[Path]) -> Optional[str]:
    """Computes deterministic SHA-256 hash of a file if it exists and is non-empty."""
    if file_path is None:
        return None
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha(fallback: str = "UNKNOWN") -> str:
    """Retrieves current git commit SHA for provenance recording."""
    env_sha = os.environ.get("GIT_COMMIT_SHA")
    if env_sha:
        return env_sha.strip()
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return fallback


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_temporal_pipeline(
    data_dir: Path,
    cvb_dir: Path,
    beef_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 0,
    num_frames: int = 8,
    smoke: bool = True,
    input_mode: str = "rgb_mask",
    seed: int = 2026,
    git_commit_sha: Optional[str] = None,
    device: Optional[torch.device] = None,
    epoch_commit_callback: Optional[Any] = None,
    preload_ram: bool = True,
    num_preload_workers: int = 32,
) -> Dict[str, Any]:
    """
    Executes the Behavior Temporal TCN training and verification pipeline.
    Supports:
      input_mode='rgb'      -> 3-channel [B, T=8, 3, 224, 224] (historical RGB temporal baseline)
      input_mode='rgb_mask' -> 4-channel [B, T=8, 4, 224, 224] (real Run 5 perception model)
    """
    start_time = time.perf_counter()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Strict reproducibility seeds
    set_reproducibility_seeds(seed)
    active_git_sha = git_commit_sha or get_git_commit_sha()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    in_channels = 4 if input_mode == "rgb_mask" else 3

    print("\n" + "=" * 70)
    print(f"  BEHAVIOR TEMPORAL PIPELINE: Mode={input_mode} (in_channels={in_channels}), Device={device}, Smoke={smoke}")
    print(f"  Provenance: Seed={seed}, cuDNN Deterministic=True, Git SHA={active_git_sha}, PreloadRAM={preload_ram}")
    print("=" * 70)

    # 1. Load canonical split CSVs if not passed in
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    assert train_csv.exists(), f"train.csv missing at {train_csv}"
    assert val_csv.exists(), f"val.csv missing at {val_csv}"
    assert test_csv.exists(), f"test.csv missing at {test_csv}"

    canonical_train_sha256 = compute_file_sha256(train_csv)
    canonical_val_sha256 = compute_file_sha256(val_csv)

    if train_df is None:
        train_df = pd.read_csv(train_csv)
    if val_df is None:
        val_df = pd.read_csv(val_csv)
    print(f"[*] Input split rows: Train={len(train_df)}, Val={len(val_df)}")
    print(f"[*] Canonical split SHA-256: train.csv={canonical_train_sha256[:16]}..., val.csv={canonical_val_sha256[:16]}...")

    # 2. Select balanced smoke subset if smoke and full split was passed
    smoke_counts = {}
    if smoke:
        # STRICT RULE: test.csv must NOT be parsed, loaded into a DataFrame, sampled, used for tuning, or evaluated!
        print("[*] Smoke mode active: test.csv is EXPLICITLY UNTOUCHED.")

        # If train_df has > 30 rows, filter to balanced smoke subset
        if len(train_df) > 30:
            train_df, val_df = get_balanced_smoke_subset(train_df, val_df)

        smoke_counts = {
            "train": {
                "total": len(train_df),
                "by_source": train_df["dataset"].value_counts().to_dict(),
                "by_class": train_df["behavior_canonical"].value_counts().to_dict(),
                "by_source_and_class": train_df.groupby(["dataset", "behavior_canonical"]).size().to_dict(),
            },
            "val": {
                "total": len(val_df),
                "by_source": val_df["dataset"].value_counts().to_dict(),
                "by_class": val_df["behavior_canonical"].value_counts().to_dict(),
                "by_source_and_class": val_df.groupby(["dataset", "behavior_canonical"]).size().to_dict(),
            },
        }
        smoke_counts["train"]["by_source_and_class"] = {f"{k[0]}_{k[1]}": v for k, v in smoke_counts["train"]["by_source_and_class"].items()}
        smoke_counts["val"]["by_source_and_class"] = {f"{k[0]}_{k[1]}": v for k, v in smoke_counts["val"]["by_source_and_class"].items()}

        print(f"[*] Subsampled for smoke test: Train={len(train_df)} sequences, Val={len(val_df)} sequences, Epochs={epochs}, BatchSize={batch_size}")
        print(f"    Train distribution: {train_df['dataset'].value_counts().to_dict()} | Classes: {train_df['behavior_canonical'].value_counts().to_dict()}")
        print(f"    Val distribution  : {val_df['dataset'].value_counts().to_dict()} | Classes: {val_df['behavior_canonical'].value_counts().to_dict()}")

    # 3. Cache handling
    train_missing_bboxes = []
    val_missing_bboxes = []
    if input_mode == "rgb_mask":
        # IN PERCEPTION MODE: Do NOT run legacy build_temporal_cache!
        # Expect pre-built real perception cache from build_behavior_perception_cache.py
        print(f"[*] Perception mode ('rgb_mask'): Bypassing legacy build_temporal_cache(). Using perception cache at {cache_dir}.")
        train_sids = set(train_df["sample_id"].astype(str))
        val_sids = set(val_df["sample_id"].astype(str))
        cached_dirs = set(os.listdir(cache_dir)) if cache_dir.exists() else set()

        missing_train = train_sids - cached_dirs
        missing_val = val_sids - cached_dirs
        assert not missing_train, f"Perception cache missing {len(missing_train)} train sequences! First: {list(missing_train)[:3]}"
        assert not missing_val, f"Perception cache missing {len(missing_val)} val sequences! First: {list(missing_val)[:3]}"
        print(f"[*] Verified perception cache: all {len(train_df)} train and {len(val_df)} val sequences present.")
    else:
        # IN RGB MODE: Run legacy temporal cache builder
        t0_cache = time.perf_counter()
        train_cache_stats, train_missing_bboxes = build_temporal_cache(
            train_df, cvb_dir, beef_dir, cache_dir, num_frames=num_frames, desc="Train Temporal Cache"
        )
        val_cache_stats, val_missing_bboxes = build_temporal_cache(
            val_df, cvb_dir, beef_dir, cache_dir, num_frames=num_frames, desc="Val Temporal Cache"
        )
        t_cache = time.perf_counter() - t0_cache
        print(f"[*] Caching complete in {t_cache:.1f}s. Train: {train_cache_stats}, Val: {val_cache_stats}")
        print(f"[*] Missing target bbox records: Train={len(train_missing_bboxes)}, Val={len(val_missing_bboxes)}")

    # 4. Generate Visual Contact Sheet
    if input_mode == "rgb_mask":
        contact_sheet_path = output_dir / "temporal_perception_contact_sheet.jpg"
        generate_perception_contact_sheet(
            train_df, cache_dir, contact_sheet_path, num_samples=6, num_frames=num_frames
        )
    else:
        contact_sheet_path = output_dir / "temporal_samples_contact_sheet.jpg"
        generate_temporal_contact_sheet(
            train_df, cache_dir, contact_sheet_path, num_samples=6, num_frames=num_frames
        )

    # 5. Data Loaders
    train_dataset = BehaviorTemporalDataset(
        train_df,
        cache_dir,
        num_frames=num_frames,
        is_train=True,
        input_mode=input_mode,
        preload_ram=preload_ram,
        num_preload_workers=num_preload_workers,
    )
    val_dataset = BehaviorTemporalDataset(
        val_df,
        cache_dir,
        num_frames=num_frames,
        is_train=False,
        input_mode=input_mode,
        preload_ram=preload_ram,
        num_preload_workers=num_preload_workers,
    )

    actual_train_workers = 0 if preload_ram else num_workers
    actual_val_workers = 0 if preload_ram else num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_train_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_val_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 6. Model, Optimizer, Criterion
    model = BehaviorTemporalModel(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        hidden_dim=256,
        dropout=0.2,
        pretrained_backbone=True,
    ).to(device)

    # Parameter Verification
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    tcn_params = sum(p.numel() for p in model.tcn.parameters() if p.requires_grad)
    total_trainable_params = backbone_params + tcn_params

    if input_mode == "rgb_mask":
        expected_params = 11903621
        assert total_trainable_params == expected_params, (
            f"Param mismatch for 4-channel model! Expected {expected_params:,}, got {total_trainable_params:,}"
        )
    else:
        expected_params = 11900485
        assert total_trainable_params == expected_params, (
            f"Param mismatch for 3-channel model! Expected {expected_params:,}, got {total_trainable_params:,}"
        )

    print("\n[*] Model Architecture & Parameter Verification:")
    print(f"    Input Mode                      : {input_mode} (in_channels={in_channels})")
    print(f"    Backbone (ResNet-18 without fc) : {backbone_params:,} params")
    print(f"    TCN (2 Conv1d blocks + Linear)  : {tcn_params:,} params")
    print(f"    Total Trainable Parameters      : {total_trainable_params:,} params (Exact Verified)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    best_val_logits = None

    best_ckpt_path = output_dir / "behavior_tcn_best.pth"
    latest_ckpt_path = output_dir / "behavior_tcn_latest.pth"

    # Retained split provenance & exact sample IDs
    retained_train_csv = cache_dir / "retained_train.csv"
    retained_val_csv = cache_dir / "retained_val.csv"
    retained_train_sha256 = compute_file_sha256(retained_train_csv)
    retained_val_sha256 = compute_file_sha256(retained_val_csv)

    retained_train_sample_ids = [str(sid) for sid in train_df["sample_id"].tolist()]
    retained_val_sample_ids = [str(sid) for sid in val_df["sample_id"].tolist()]

    reproducibility_meta = {
        "seed": seed,
        "python_hash_seed": str(seed),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "git_commit_sha": active_git_sha,
        "canonical_train_csv_sha256": canonical_train_sha256,
        "canonical_val_csv_sha256": canonical_val_sha256,
        "retained_train_csv_sha256": retained_train_sha256,
        "retained_val_csv_sha256": retained_val_sha256,
        "retained_train_sample_ids": retained_train_sample_ids,
        "retained_val_sample_ids": retained_val_sample_ids,
    }

    epoch_history = []

    # 7. Training Loop
    print("\n[*] Starting training loop...")
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_samples = 0

        for seq_images, targets, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=80):
            seq_images = seq_images.to(device)
            targets = targets.to(device)

            # Assert input tensor shapes
            B, T, C, H, W = seq_images.shape
            assert T == num_frames, f"Expected T={num_frames}, got {T}"
            assert C == in_channels, f"Expected C={in_channels}, got {C}"
            assert H == 224 and W == 224, f"Expected 224x224, got {H}x{W}"

            optimizer.zero_grad()
            logits, feats = model(seq_images)

            # Assert feature and logits shapes
            assert feats.shape == (B, T, 512), f"Expected feats [B, T, 512], got {feats.shape}"
            assert logits.shape == (B, NUM_CLASSES), f"Expected logits [B, 5], got {logits.shape}"

            loss = criterion(logits, targets)
            assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)
            n_samples += len(targets)

        train_loss = running_loss / max(1, n_samples)

        # Validation
        val_metrics, val_loss, val_logits = evaluate(model, val_loader, device, desc=f"Epoch {epoch}/{epochs} [Val]")
        epoch_dur = time.perf_counter() - t_epoch_start

        print(
            f"[*] Epoch {epoch:02d}/{epochs:02d} ({epoch_dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['overall_accuracy']*100:.1f}% | "
            f"Val Bal Acc: {val_metrics['balanced_accuracy']*100:.1f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_overall_accuracy": round(float(val_metrics["overall_accuracy"]), 6),
            "val_balanced_accuracy": round(float(val_metrics["balanced_accuracy"]), 6),
            "val_macro_f1": round(float(val_metrics["macro_f1"]), 6),
            "val_macro_precision": round(float(val_metrics["macro_precision"]), 6),
            "val_macro_recall": round(float(val_metrics["macro_recall"]), 6),
            "val_per_class_f1": {k: round(float(v), 6) for k, v in val_metrics["per_class_f1"].items()},
            "duration_sec": round(epoch_dur, 2),
        }
        epoch_history.append(epoch_record)

        # Save latest checkpoint with full reproducibility & history metadata
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "architecture": "ResNet18_TCN2",
            "input_mode": input_mode,
            "in_channels": in_channels,
            "num_classes": NUM_CLASSES,
            "num_frames": num_frames,
            "total_params": total_trainable_params,
            "reproducibility": reproducibility_meta,
            "epoch_history": list(epoch_history),
        }
        torch.save(ckpt_data, latest_ckpt_path)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_val_logits = val_logits.clone()
            torch.save(ckpt_data, best_ckpt_path)
            print(f"    --> [BEST] New best checkpoint saved to {best_ckpt_path}")

        if epoch_commit_callback is not None:
            try:
                epoch_commit_callback(epoch, val_metrics["macro_f1"] >= best_macro_f1)
            except Exception as ce:
                print(f"[WARN] Epoch commit callback warning: {ce}")

    # 8. Checkpoint Save/Resume Verification
    print("\n[*] Verifying checkpoint save and reload bit-identity...")
    assert best_ckpt_path.exists(), "Best checkpoint was not created!"
    resume_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    resume_model = BehaviorTemporalModel(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        hidden_dim=256,
        dropout=0.2,
        pretrained_backbone=False,
    ).to(device)
    resume_model.load_state_dict(resume_ckpt["model_state_dict"])
    resume_metrics, _, reloaded_logits = evaluate(resume_model, val_loader, device, desc="Resume Check")

    # Assert bit-identical logits
    max_logit_diff = float((best_val_logits - reloaded_logits).abs().max())
    assert max_logit_diff < 1e-5, f"Logit mismatch after reload! Max diff: {max_logit_diff}"
    assert abs(resume_metrics["macro_f1"] - best_metrics["macro_f1"]) < 1e-6, "Macro-F1 metric mismatch after reload!"
    print(f"[*] Checkpoint reload verified BIT-IDENTICALLY: Max Logit Diff = {max_logit_diff:.8f}, Val Macro-F1 = {resume_metrics['macro_f1']:.4f}")

    total_duration = time.perf_counter() - start_time

    # 9. Save Summary Artifacts
    summary = {
        "status": "SMOKE_SUCCESS" if smoke else "FULL_SUCCESS",
        "task": f"Phase 3 Run 5 Behavior ({input_mode.upper()}) Smoke Test",
        "mode": "smoke" if smoke else "full",
        "input_mode": input_mode,
        "in_channels": in_channels,
        "architecture": {
            "backbone": f"ImageNet-pretrained ResNet-18 ({in_channels}-channel conv1, 512-D features per frame)",
            "temporal_module": "1D TCN (2 Conv1d blocks, GELU, BatchNorm1d, Dropout=0.2, AdaptiveAvgPool1d, Linear head)",
            "sequence_length_T": num_frames,
            "input_resolution": f"224x224 ({in_channels} channels)",
            "backbone_params": backbone_params,
            "tcn_params": tcn_params,
            "total_trainable_params": total_trainable_params,
        },
        "temporal_sampling_rule": (
            f"Deterministic: T={num_frames} approximately evenly spaced frames selected across [start_frame, end_frame]. "
            "CVB preserves target tracklet identity using authentic per-frame bboxes. "
            "Kaggle Beef samples directly from single-cow video clips."
        ),
        "smoke_dataset_counts": smoke_counts,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "max_logit_diff_after_reload": max_logit_diff,
        "test_set_protection": "Canonical test.csv was not parsed, loaded into a DataFrame, sampled, used for tuning, or evaluated.",
        "test_csv_evaluated": False,
        "total_duration_sec": round(total_duration, 2),
        "checkpoints": {
            "best": str(best_ckpt_path),
            "latest": str(latest_ckpt_path),
        },
        "contact_sheet": str(contact_sheet_path),
        "missing_bbox_records_count": len(train_missing_bboxes) + len(val_missing_bboxes),
        "reproducibility": reproducibility_meta,
        "retained_train_sample_ids": retained_train_sample_ids,
        "retained_val_sample_ids": retained_val_sample_ids,
        "epoch_history": epoch_history,
    }

    metrics_out = output_dir / "behavior_tcn_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Metrics saved to: {metrics_out}")

    print("\n" + "=" * 70)
    print(f"  BEHAVIOR PIPELINE COMPLETE (Runtime: {total_duration:.1f}s)")
    print("=" * 70)
    if epoch_commit_callback is not None:
        try:
            epoch_commit_callback(epochs, True)
        except Exception as ce:
            print(f"[WARN] Final commit callback warning: {ce}")

    return summary


# ==============================================================================
# MATCHED DATASET FOR RUN 2 BASELINE EVALUATION
# ==============================================================================
class MatchedBehaviorRGBDataset(Dataset):
    """
    Dataset to load single-frame RGB midpoint crops for Run 2 ResNet-18 baseline evaluation.
    STRICT REQUIREMENT: Loads the authentic historical Run 2 cached crops directly from:
      run2_cache_dir / f"{sample_id}.jpg"
    (where Run 2 saved crops at /checkpoints/behavior_cache/{sample_id}.jpg during baseline training).
    Zero fallback, zero dummy crops, zero on-the-fly video decoding.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        run2_cache_dir: Path,
        image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.run2_cache_dir = Path(run2_cache_dir)
        self.image_size = image_size
        self.sample_ids = self.df["sample_id"].tolist()
        self.labels = [CLASS_TO_IDX[b] for b in self.df["behavior_canonical"]]
        self.dataset_names = self.df["dataset"].tolist()

        assert self.run2_cache_dir.exists(), f"Run 2 cache directory does not exist: {self.run2_cache_dir}"

        # Assert all requested sample crops exist and are non-empty
        missing = [
            sid for sid in self.sample_ids
            if not (self.run2_cache_dir / f"{sid}.jpg").exists() or (self.run2_cache_dir / f"{sid}.jpg").stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(
                f"MatchedBehaviorRGBDataset failed: {len(missing)} / {len(self.sample_ids)} "
                f"Run 2 cache files are missing or 0 bytes in {self.run2_cache_dir}! "
                f"Missing sample IDs (first 20): {missing[:20]}. "
                "Silent fallback or reconstruction is strictly prohibited."
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        sample_id = self.sample_ids[idx]
        target = self.labels[idx]
        d_name = self.dataset_names[idx]

        cache_path = self.run2_cache_dir / f"{sample_id}.jpg"
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing authentic Run 2 cache file: {cache_path}")

        image = Image.open(cache_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, target, sample_id, d_name


# ==============================================================================
# RUN 2 RGB BASELINE VS RUN 5 PERCEPTION+TCN MATCHED EVALUATION
# ==============================================================================
def evaluate_matched_run2_vs_run5(
    run5_checkpoint_path: Path,
    run2_checkpoint_path: Path,
    retained_test_df: pd.DataFrame,
    perception_cache_dir: Path,
    output_dir: Path,
    cvb_dir: Optional[Path] = None,
    beef_dir: Optional[Path] = None,
    run2_cache_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    num_frames: int = 8,
    preload_ram: bool = True,
    num_preload_workers: int = 32,
) -> Dict[str, Any]:
    """
    Evaluates both the frozen Run 5 Perception+TCN model and the historical
    Run 2 RGB baseline model on the EXACT SAME retained test sequences.
    Produces fair head-to-head comparison metrics and structured markdown report.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    assert run5_checkpoint_path.exists(), f"Run 5 checkpoint missing at {run5_checkpoint_path}"
    assert run2_checkpoint_path.exists(), f"Run 2 checkpoint missing at {run2_checkpoint_path}"
    assert len(retained_test_df) > 0, "Empty retained_test_df provided for evaluation!"
    assert run2_cache_dir is not None and run2_cache_dir.exists(), f"Run 2 cache directory missing at {run2_cache_dir}"

    # Strict audit: verify that authentic historical Run 2 cache file exists for EVERY retained test ID
    missing_run2_crops = [
        sid for sid in retained_test_df["sample_id"]
        if not (run2_cache_dir / f"{sid}.jpg").exists() or (run2_cache_dir / f"{sid}.jpg").stat().st_size == 0
    ]
    if missing_run2_crops:
        raise RuntimeError(
            f"CRITICAL: Authentic Run 2 baseline cache missing {len(missing_run2_crops)} / {len(retained_test_df)} "
            f"retained test samples at {run2_cache_dir}! Missing sample IDs (first 20): {missing_run2_crops[:20]}... "
            "Silent reconstruction or fallback is strictly prohibited. Historical Run 2 cached crops are required."
        )
    print(f"[*] Verified authentic Run 2 test cache: all {len(retained_test_df)} retained test crops present in {run2_cache_dir}.")

    print("\n" + "=" * 70)
    print(f"  FAIR MATCHED TEST EVALUATION: Run 5 Perception+TCN vs Run 2 RGB Baseline")
    print(f"  Retained Test Sequences: {len(retained_test_df)}")
    print(f"  Device: {device}")
    print("=" * 70)

    # 1. Evaluate Run 5 Perception+TCN Model
    print("\n[*] Step 1: Evaluating Run 5 Perception+TCN (4-channel [B, 8, 4, 224, 224])...")
    run5_ckpt = torch.load(run5_checkpoint_path, map_location=device, weights_only=False)
    run5_model = BehaviorTemporalModel(
        num_classes=NUM_CLASSES,
        in_channels=4,
        hidden_dim=256,
        dropout=0.2,
        pretrained_backbone=False,
    ).to(device)
    run5_model.load_state_dict(run5_ckpt["model_state_dict"])
    run5_model.eval()

    run5_dataset = BehaviorTemporalDataset(
        retained_test_df,
        perception_cache_dir,
        num_frames=num_frames,
        is_train=False,
        input_mode="rgb_mask",
        preload_ram=preload_ram,
        num_preload_workers=num_preload_workers,
    )
    run5_loader = DataLoader(
        run5_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    run5_metrics, run5_loss, _ = evaluate(
        run5_model, run5_loader, device, desc="Evaluating Run 5 Perception+TCN"
    )

    print(f"[*] Run 5 Matched Test Results:")
    print(f"    Overall Accuracy : {run5_metrics['overall_accuracy']*100:.2f}%")
    print(f"    Balanced Accuracy: {run5_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro-F1         : {run5_metrics['macro_f1']:.4f}")
    print(f"    Test Loss        : {run5_loss:.4f}")

    # 2. Evaluate Run 2 RGB Baseline Model on EXACT SAME Samples
    print("\n[*] Step 2: Evaluating Run 2 RGB Baseline on identical retained test samples...")
    run2_model = models.resnet18(weights=None)
    run2_model.fc = nn.Linear(512, NUM_CLASSES)
    run2_ckpt = torch.load(run2_checkpoint_path, map_location=device, weights_only=False)
    run2_model.load_state_dict(run2_ckpt["model_state_dict"])
    run2_model = run2_model.to(device)
    run2_model.eval()

    run2_dataset = MatchedBehaviorRGBDataset(
        retained_test_df,
        run2_cache_dir=run2_cache_dir,
        image_size=224,
    )
    run2_loader = DataLoader(
        run2_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    run2_loss_total = 0.0
    run2_preds = []
    run2_targets = []
    run2_datasets = []

    with torch.no_grad():
        for images, targets, _, d_names in tqdm(run2_loader, desc="Evaluating Run 2 RGB (Matched)", ncols=80):
            images = images.to(device)
            targets = targets.to(device)
            logits = run2_model(images)
            loss = criterion(logits, targets)
            run2_loss_total += loss.item() * len(targets)
            preds = logits.argmax(dim=1).cpu().numpy()
            run2_preds.extend(preds)
            run2_targets.extend(targets.cpu().numpy())
            run2_datasets.extend(list(d_names))

    mean_run2_loss = run2_loss_total / max(1, len(run2_targets))
    run2_matched_metrics = compute_behavior_metrics(
        y_true=np.array(run2_targets),
        y_pred=np.array(run2_preds),
        dataset_names=np.array(run2_datasets),
    )

    print(f"[*] Run 2 Matched Test Results:")
    print(f"    Overall Accuracy : {run2_matched_metrics['overall_accuracy']*100:.2f}%")
    print(f"    Balanced Accuracy: {run2_matched_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro-F1         : {run2_matched_metrics['macro_f1']:.4f}")
    print(f"    Test Loss        : {mean_run2_loss:.4f}")

    # 3. Canonical Historical Run 2 Reference (from Run 2 certificate)
    run2_canonical_historical = {
        "candidate_count": 809,
        "overall_accuracy": 0.8888,
        "balanced_accuracy": 0.7172,
        "macro_f1": 0.7413,
        "test_loss": 0.5312,
        "per_class": {
            "Lying": {"precision": 0.9634, "recall": 0.9472, "f1": 0.9552, "support": 284},
            "Feeding": {"precision": 0.8931, "recall": 0.9538, "f1": 0.9225, "support": 281},
            "Drinking": {"precision": 0.8254, "recall": 0.8615, "f1": 0.8430, "support": 58},
            "Standing": {"precision": 0.7785, "recall": 0.7585, "f1": 0.7684, "support": 160},
            "Walking": {"precision": 0.2500, "recall": 0.1923, "f1": 0.2174, "support": 26, "note": "CVB-only"},
        },
        "dataset_breakdown": {
            "cvb": {"overall_accuracy": 0.8412, "balanced_accuracy": 0.6402, "macro_f1": 0.6541, "support": 381},
            "beef_cattle_behavior": {"overall_accuracy": 0.9406, "balanced_accuracy": 0.8988, "macro_f1": 0.9113, "support": 428},
        },
    }

    # 4. Deltas & Structured Report
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matched_sample_count": len(retained_test_df),
        "run2_canonical_historical": run2_canonical_historical,
        "run2_matched_subset": {
            "overall_accuracy": run2_matched_metrics["overall_accuracy"],
            "balanced_accuracy": run2_matched_metrics["balanced_accuracy"],
            "macro_f1": run2_matched_metrics["macro_f1"],
            "test_loss": round(mean_run2_loss, 4),
            "per_class": run2_matched_metrics["per_class"],
            "cvb_metrics": run2_matched_metrics["cvb_metrics"],
            "beef_metrics": run2_matched_metrics["beef_metrics"],
            "confusion_matrix": run2_matched_metrics["confusion_matrix"],
        },
        "run5_perception_tcn_matched_subset": {
            "overall_accuracy": run5_metrics["overall_accuracy"],
            "balanced_accuracy": run5_metrics["balanced_accuracy"],
            "macro_f1": run5_metrics["macro_f1"],
            "test_loss": round(run5_loss, 4),
            "per_class": run5_metrics["per_class"],
            "cvb_metrics": run5_metrics["cvb_metrics"],
            "beef_metrics": run5_metrics["beef_metrics"],
            "confusion_matrix": run5_metrics["confusion_matrix"],
        },
        "matched_delta_run5_minus_run2": {
            "overall_accuracy": round(run5_metrics["overall_accuracy"] - run2_matched_metrics["overall_accuracy"], 4),
            "balanced_accuracy": round(run5_metrics["balanced_accuracy"] - run2_matched_metrics["balanced_accuracy"], 4),
            "macro_f1": round(run5_metrics["macro_f1"] - run2_matched_metrics["macro_f1"], 4),
            "test_loss": round(run5_loss - mean_run2_loss, 4),
            "per_class_f1_delta": {
                c: round(run5_metrics["per_class"][c]["f1"] - run2_matched_metrics["per_class"][c]["f1"], 4)
                for c in CANONICAL_CLASSES
            },
        },
    }

    # Save JSON metrics
    metrics_path = output_dir / "run5_test_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(run5_metrics, f, indent=2)

    comp_path = output_dir / "run2_vs_run5_matched_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Generate Markdown Table Report
    md_path = output_dir / "run2_vs_run5_matched_comparison.md"
    d_acc = comparison["matched_delta_run5_minus_run2"]["overall_accuracy"] * 100
    d_bal = comparison["matched_delta_run5_minus_run2"]["balanced_accuracy"] * 100
    d_f1 = comparison["matched_delta_run5_minus_run2"]["macro_f1"]

    md_content = f"""# Phase 3 Run 5 vs Run 2 Matched Behavior Test Comparison

**Evaluation Date:** {comparison['timestamp']}  
**Matched Test Sample Count:** {len(retained_test_df)} (from 809 canonical candidates)

---

## 1. High-Level Performance Comparison

| Metric | Run 2 RGB Full (Historical) | Run 2 RGB Matched (N={len(retained_test_df)}) | Run 5 Perception+TCN (N={len(retained_test_df)}) | Delta (Run 5 - Run 2 Matched) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | {run2_canonical_historical['overall_accuracy']*100:.2f}% | {run2_matched_metrics['overall_accuracy']*100:.2f}% | **{run5_metrics['overall_accuracy']*100:.2f}%** | **{d_acc:+.2f}%** |
| **Balanced Accuracy** | {run2_canonical_historical['balanced_accuracy']*100:.2f}% | {run2_matched_metrics['balanced_accuracy']*100:.2f}% | **{run5_metrics['balanced_accuracy']*100:.2f}%** | **{d_bal:+.2f}%** |
| **Macro-F1** | {run2_canonical_historical['macro_f1']:.4f} | {run2_matched_metrics['macro_f1']:.4f} | **{run5_metrics['macro_f1']:.4f}** | **{d_f1:+.4f}** |
| **Test Loss** | {run2_canonical_historical['test_loss']:.4f} | {mean_run2_loss:.4f} | **{run5_loss:.4f}** | **{comparison['matched_delta_run5_minus_run2']['test_loss']:+.4f}** |

---

## 2. Per-Class F1 Score Comparison (Matched Subset)

| Behavior Class | Run 2 RGB Matched F1 | Run 5 Perception+TCN F1 | Delta (Run 5 - Run 2) | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for c in CANONICAL_CLASSES:
        f1_r2 = run2_matched_metrics["per_class"][c]["f1"]
        f1_r5 = run5_metrics["per_class"][c]["f1"]
        delta_f = f1_r5 - f1_r2
        sup = run5_metrics["per_class"][c]["support"]
        flag = " *(CVB-only)*" if c == "Walking" else ""
        md_content += f"| **{c}**{flag} | {f1_r2:.4f} | **{f1_r5:.4f}** | **{delta_f:+.4f}** | {sup} |\n"

    md_content += f"""
---

## 3. Dataset Source Breakdown (Matched Subset)

| Dataset | Metric | Run 2 RGB Matched | Run 5 Perception+TCN |
| :--- | :--- | :--- | :--- |
| **CVB (Barn CCTV, Multi-Cow)** | Accuracy | {run2_matched_metrics['cvb_metrics']['overall_accuracy']*100:.2f}% | **{run5_metrics['cvb_metrics']['overall_accuracy']*100:.2f}%** |
| | Macro-F1 | {run2_matched_metrics['cvb_metrics']['macro_f1']:.4f} | **{run5_metrics['cvb_metrics']['macro_f1']:.4f}** |
| **Kaggle Beef (Single-Cow Clips)** | Accuracy | {run2_matched_metrics['beef_metrics']['overall_accuracy']*100:.2f}% | **{run5_metrics['beef_metrics']['overall_accuracy']*100:.2f}%** |
| | Macro-F1 | {run2_matched_metrics['beef_metrics']['macro_f1']:.4f} | **{run5_metrics['beef_metrics']['macro_f1']:.4f}** |

---
*Generated automatically by Phase 3 Run 5 Test Evaluation Suite.*
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[*] Comparison report saved to {md_path}")
    print(f"[*] Comparison JSON saved to {comp_path}")
    return comparison


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Run 5 Behavior Temporal Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/behavior/cvb_beef"),
        help="Path to canonical CVB + Kaggle Beef split CSVs",
    )
    parser.add_argument(
        "--cvb-dir",
        type=Path,
        default=Path("/mnt/cvb/cvb/000058916v001"),
        help="Root path to CVB dataset",
    )
    parser.add_argument(
        "--beef-dir",
        type=Path,
        default=Path("/mnt/beef/beef_behavior"),
        help="Root path to Kaggle Beef dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/behavior_perception_cache"),
        help="Directory to cache pre-extracted temporal sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/behavior_perception"),
        help="Directory to save checkpoints, contact sheet, and metrics",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["rgb", "rgb_mask"],
        default="rgb_mask",
        help="Input mode: 'rgb' (3 channels, legacy) or 'rgb_mask' (4 channels, perception)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (sequences)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of temporal frames per sequence (T)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Explicit training seed for Python, NumPy, PyTorch CPU & CUDA",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Run fast smoke test on subsampled balanced set (Train=30, Val=10, 2 epochs)",
    )

    args = parser.parse_args()

    train_temporal_pipeline(
        data_dir=args.data_dir,
        cvb_dir=args.cvb_dir,
        beef_dir=args.beef_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        smoke=args.smoke,
        input_mode=args.input_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

