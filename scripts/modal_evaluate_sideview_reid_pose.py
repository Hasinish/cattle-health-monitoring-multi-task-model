# -*- coding: utf-8 -*-
"""High-Speed Batched & Distributed Protocol A Evaluation for SideView Re-ID + Pose.

Target Profile   : dryousufmozumder (healthy $20.83 balance)
Dataset Volume   : sideview-data (mounted at /data)
Checkpoint Volume: reid-checkpoints (mounted at /checkpoints)
Model Checkpoint : /checkpoints/sideview_reid_pose_ablation/reid_pose_best.pth
Pose Cache       : /checkpoints/sideview_pose_cache/pose_features_v1.pt

Design:
  1. Chunks missing Protocol A samples (up to 62,678 images) across a parallel pool
     of modest Tesla T4 GPU workers on Modal.
  2. Each worker processes batches of crops through SuperAnimal with live progress bars.
  3. Incremental chunk results are merged and persisted to /checkpoints/sideview_pose_cache/.
  4. Best checkpoint (Epoch 23, Val Acc 99.18%) extracts 576-D unit-L2 embeddings
     for Gallery Parlor (36,811), Barn Queries (25,260), and Snapshot Queries (607).
  5. Computes official Protocol A retrieval metrics (Rank-1, Rank-5, Rank-10, mAP)
     and saves local metrics JSON and report.

Usage:
  python -m modal run --profile dryousufmozumder scripts/modal_evaluate_sideview_reid_pose.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _orig_commonpath = os.path.commonpath

    def _safe_commonpath(paths):
        try:
            return _orig_commonpath(paths)
        except ValueError:
            return ""

    os.path.commonpath = _safe_commonpath

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

data_vol = modal.Volume.from_name("sideview-data")
checkpoint_vol = modal.Volume.from_name("reid-checkpoints", create_if_missing=True)

pose_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "deeplabcut>=3.0.0",
        "dlclibrary",
        "opencv-python-headless",
        "pillow",
        "pandas",
        "numpy<2.0.0",
        "scikit-learn",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_baseline.py"),
        remote_path="/root/scripts/train_sideview_reid_baseline.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_perception.py"),
        remote_path="/root/scripts/train_sideview_reid_perception.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_pose.py"),
        remote_path="/root/scripts/train_sideview_reid_pose.py",
    )
)

app = modal.App("sideview-reid-pose-evaluation", image=pose_image)


# ==============================================================================
# 1. PARALLEL WORKER: EXTRACT POSE FOR A CHUNK OF IMAGES
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=3600,
    cpu=4.0,
    max_containers=12,  # Upgraded concurrency limit: 12 T4 GPUs
)
def extract_pose_chunk_remote(
    chunk_rows: List[Dict[str, str]],
    chunk_idx: int,
    total_chunks: int,
    batch_size: int = 16,
) -> str:
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from scripts.train_sideview_reid_perception import (
        _mask_bbox_with_margin,
        resolve_sideview_image_path,
    )
    from scripts.train_sideview_reid_pose import (
        POSE_FEATURE_DIM,
        SuperAnimalPoseFeatureExtractor,
    )

    data_root = Path("/data/sideviewcows2026")
    chunk_dir = Path("/checkpoints/sideview_pose_cache/chunks")
    chunk_file = chunk_dir / f"chunk_{chunk_idx:02d}.pt"

    # Fast skip if already extracted and committed on volume
    if chunk_file.exists() and chunk_file.stat().st_size > 1024:
        try:
            cached_res = torch.load(chunk_file, map_location="cpu", weights_only=False)
            if len(cached_res) >= len(chunk_rows):
                print(f"Chunk {chunk_idx + 1:02d}/{total_chunks:02d}: 100% (already cached on volume ✅)")
                return f"chunk_{chunk_idx:02d}.pt"
        except Exception:
            pass

    extractor = SuperAnimalPoseFeatureExtractor(device="cuda")
    results: Dict[str, List[float]] = {}

    pbar = tqdm(
        total=len(chunk_rows),
        desc=f"Chunk {chunk_idx + 1:02d}/{total_chunks:02d}",
        file=sys.stdout,
        mininterval=2.0,
    )

    # Process in micro-batches
    for i in range(0, len(chunk_rows), batch_size):
        batch = chunk_rows[i : i + batch_size]
        valid_crops: List[np.ndarray] = []
        valid_keys: List[str] = []

        for row in batch:
            raw_img = row["image_path"]
            raw_mask = row["mask_path"]
            img_p = resolve_sideview_image_path(raw_img, data_root)
            mask_p = resolve_sideview_image_path(raw_mask, data_root)

            if img_p is None or mask_p is None:
                results[raw_img] = [0.0] * POSE_FEATURE_DIM
                continue

            try:
                with Image.open(img_p) as im:
                    rgb_im = im.convert("RGB")
                with Image.open(mask_p) as mk:
                    mask_gray = mk.convert("L")

                mask_np = np.asarray(mask_gray) > 0
                crop_box = _mask_bbox_with_margin(mask_np, margin_fraction=0.05)
                crop_rgb_np = np.asarray(rgb_im.crop(crop_box))

                valid_crops.append(crop_rgb_np)
                valid_keys.append(raw_img)
            except Exception:
                results[raw_img] = [0.0] * POSE_FEATURE_DIM

        # Extract features for valid crops
        for key, crop in zip(valid_keys, valid_crops):
            try:
                vec = extractor.extract_crop_pose_feature(crop)
                results[key] = vec.tolist()
            except Exception:
                results[key] = [0.0] * POSE_FEATURE_DIM

        pbar.update(len(batch))

    pbar.close()

    # Atomically persist this chunk to persistent volume immediately
    chunk_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, chunk_file)
    checkpoint_vol.commit()
    print(f"Chunk {chunk_idx + 1:02d}/{total_chunks:02d}: 100% [SAVED {len(results)} poses to volume]")

    return f"chunk_{chunk_idx:02d}.pt"


# ==============================================================================
# 2. MASTER EVALUATION COORDINATOR
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=14400,  # 4-hour master timeout: zero timeout risk
    cpu=4.0,
    memory=16384,
)
def evaluate_protocol_a_master(
    chunk_size: int = 1500,
    eval_batch_size: int = 128,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from scripts.train_sideview_reid_baseline import evaluate_retrieval_chunked
    from scripts.train_sideview_reid_pose import (
        ResNet18ReIDPoseAblation,
        SideViewReIDPoseDataset,
        extract_dataset_embeddings_pose,
        get_parameter_counts,
    )

    data_root = Path("/data/sideviewcows2026")
    proto_dir = Path("/root/datasets/id/sideviewcows2026")
    df_a = pd.read_csv(proto_dir / "protocol_cross_setting.csv")

    eval_cows = sorted(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique())
    gallery_df = df_a[df_a["setting_role"] == "gallery"].reset_index(drop=True)
    barn_df = df_a[df_a["setting_role"] == "query_barn"].reset_index(drop=True)
    snapshots_df = df_a[df_a["setting_role"] == "query_snapshots"].reset_index(drop=True)

    print("\n" + "=" * 76)
    print("  SIDEVIEW RE-ID + POSE PROTOCOL A EVALUATION (dryousufmozumder)")
    print("=" * 76)
    print(f"[*] Total Evaluation Cows: {len(eval_cows)} (Unseen)")
    print(f"[*] Parlor Gallery:   {len(gallery_df):,} images")
    print(f"[*] Barn Queries:     {len(barn_df):,} images")
    print(f"[*] Snapshot Queries: {len(snapshots_df):,} images")
    print(f"[*] Total Protocol A: {len(gallery_df) + len(barn_df) + len(snapshots_df):,} images")

    # Load existing pose cache
    cache_path = Path("/checkpoints/sideview_pose_cache/pose_features_v1.pt")
    pose_dict: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            pose_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in loaded.items()}
            print(f"[OK] Loaded {len(pose_dict):,} existing cached poses from {cache_path}")
        except Exception as e:
            print(f"[WARNING] Could not load {cache_path}: {e}")

    # Identify missing Protocol A samples
    proto_a_df = pd.concat([gallery_df, snapshots_df, barn_df], ignore_index=True)
    missing_df = proto_a_df[~proto_a_df["image_path"].isin(pose_dict.keys())].reset_index(drop=True)
    print(f"[*] Missing Protocol A Samples To Extract: {len(missing_df):,} / {len(proto_a_df):,}")

    chunk_dir = Path("/checkpoints/sideview_pose_cache/chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if len(missing_df) > 0:
        missing_records = missing_df[["image_path", "mask_path"]].to_dict(orient="records")
        chunks = [
            missing_records[i : i + chunk_size]
            for i in range(0, len(missing_records), chunk_size)
        ]
        total_chunks = len(chunks)
        print(f"[*] Distributing {len(missing_records):,} images across {total_chunks} parallel chunks (up to 12 T4 workers)...")

        chunk_args = [
            (chunk, c_idx, total_chunks)
            for c_idx, chunk in enumerate(chunks)
        ]

        t0 = time.time()
        # Use Modal .starmap to distribute across worker pool
        chunk_results = list(extract_pose_chunk_remote.starmap(chunk_args))
        elapsed = time.time() - t0
        print(f"\n[OK] Parallel pose extraction completed in {elapsed:.1f}s ({elapsed / 60:.1f} mins)!")

        # Assemble individual chunk files from volume into pose_dict
        print(f"[*] Assembling all {total_chunks} chunk files from {chunk_dir} into master cache...")
        new_count = 0
        for c_idx in range(total_chunks):
            c_file = chunk_dir / f"chunk_{c_idx:02d}.pt"
            if c_file.exists():
                try:
                    c_dict = torch.load(c_file, map_location="cpu", weights_only=False)
                    for k, v in c_dict.items():
                        if k not in pose_dict:
                            pose_dict[k] = v
                            new_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed reading {c_file}: {e}")
            else:
                print(f"[WARNING] Chunk file {c_file} not found on volume!")

        print(f"[OK] Added {new_count:,} new poses. Total pose cache size: {len(pose_dict):,}")

        # Persist updated monolithic pose cache to volume
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dict_to_save = {k: np.asarray(v, dtype=np.float32) for k, v in pose_dict.items()}
        torch.save(dict_to_save, cache_path)
        checkpoint_vol.commit()
        print(f"[OK] Persisted master pose cache to {cache_path} and committed volume.")
    else:
        print("[OK] All 62,678 Protocol A samples already exist in pose cache!")

    # Convert all poses to np.ndarray float32
    final_pose_dict = {
        k: (v if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.float32))
        for k, v in pose_dict.items()
    }

    # Load Model Checkpoint
    model_path = Path("/checkpoints/sideview_reid_pose_ablation/reid_pose_best.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"\n[*] Loading Trained Checkpoint from: {model_path}")
    ckpt = torch.load(model_path, map_location="cuda", weights_only=False)
    best_epoch = ckpt.get("epoch", "unknown")
    val_metrics = ckpt.get("val_metrics", {})
    print(f"[OK] Checkpoint Best Epoch: {best_epoch} | Val Acc: {val_metrics.get('val_top1_acc')}% | Val Macro-F1: {val_metrics.get('val_macro_f1')}")

    model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_label_map = {c: 0 for c in eval_cows}

    print("\n[*] Building DataLoaders for Embedding Extraction (Batch Size: 128)...")
    gallery_dataset = SideViewReIDPoseDataset(
        df=gallery_df,
        data_root=data_root,
        cow_to_label=dummy_label_map,
        augment=False,
        pose_dict=final_pose_dict,
    )
    snapshots_dataset = SideViewReIDPoseDataset(
        df=snapshots_df,
        data_root=data_root,
        cow_to_label=dummy_label_map,
        augment=False,
        pose_dict=final_pose_dict,
    )
    barn_dataset = SideViewReIDPoseDataset(
        df=barn_df,
        data_root=data_root,
        cow_to_label=dummy_label_map,
        augment=False,
        pose_dict=final_pose_dict,
    )

    gallery_loader = DataLoader(gallery_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=8)
    snapshots_loader = DataLoader(snapshots_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=8)
    barn_loader = DataLoader(barn_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=8)

    print("[*] Extracting Parlor Gallery 576-D Embeddings (36,811 images)...")
    gallery_features, gallery_ids = extract_dataset_embeddings_pose(
        model, gallery_loader, torch.device("cuda"), desc="Gallery Parlor Embeddings"
    )

    print("[*] Extracting Snapshot Query 576-D Embeddings (607 images)...")
    snapshot_features, snapshot_ids = extract_dataset_embeddings_pose(
        model, snapshots_loader, torch.device("cuda"), desc="Query Snapshots Embeddings"
    )

    print("[*] Evaluating Protocol A Retrieval (Snapshots -> Parlor)...")
    snapshots_eval = evaluate_retrieval_chunked(
        query_features=snapshot_features,
        query_ids=snapshot_ids,
        gallery_features=gallery_features,
        gallery_ids=gallery_ids,
        chunk_size=607,
    )

    print(f"\n>>> SNAPSHOTS -> PARLOR RETRIEVAL RESULT:")
    print(f"    Rank-1:  {snapshots_eval['rank_1']:.2f}%")
    print(f"    Rank-5:  {snapshots_eval['rank_5']:.2f}%")
    print(f"    Rank-10: {snapshots_eval['rank_10']:.2f}%")
    print(f"    mAP:     {snapshots_eval['mAP']:.2f}%")

    print("\n[*] Extracting Barn Query 576-D Embeddings (25,260 images)...")
    barn_features, barn_ids = extract_dataset_embeddings_pose(
        model, barn_loader, torch.device("cuda"), desc="Query Barn Embeddings"
    )

    print("[*] Evaluating Protocol A Retrieval (Barn -> Parlor)...")
    barn_eval = evaluate_retrieval_chunked(
        query_features=barn_features,
        query_ids=barn_ids,
        gallery_features=gallery_features,
        gallery_ids=gallery_ids,
    )

    print(f"\n>>> BARN -> PARLOR RETRIEVAL RESULT:")
    print(f"    Rank-1:  {barn_eval['rank_1']:.2f}%")
    print(f"    Rank-5:  {barn_eval['rank_5']:.2f}%")
    print(f"    Rank-10: {barn_eval['rank_10']:.2f}%")
    print(f"    mAP:     {barn_eval['mAP']:.2f}%")

    retrieval_metrics = {
        "task": "sideviewcows2026_reid_pose_ablation",
        "condition": "SideViewCows2026 GT-Mask + SuperAnimal Pose (ResNet-50)",
        "profile": "dryousufmozumder",
        "best_epoch": best_epoch,
        "val_metrics_at_best": val_metrics,
        "parameter_counts": get_parameter_counts(),
        "retrieval_results": {
            "query_snapshots": snapshots_eval,
            "query_barn": barn_eval,
        },
        "held_out_protocol_a_images_loaded": len(gallery_df) + len(barn_df) + len(snapshots_df),
    }

    # Save to checkpoints volume
    out_dir = Path("/checkpoints/sideview_reid_pose_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reid_pose_metrics.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_metrics, f, indent=2)
    checkpoint_vol.commit()

    return retrieval_metrics


# ==============================================================================
# LOCAL ENTRYPOINT (SHOWS LIVE STREAMING PROGRESS)
# ==============================================================================
@app.local_entrypoint()
def main(chunk_size: int = 1500):
    print("\n" + "=" * 76)
    print("  LAUNCHING PROTOCOL A EVALUATION ON PROFILE: dryousufmozumder")
    print("=" * 76)
    print("[LOCAL] Distributing pose extraction and calculating retrieval metrics...")
    metrics = evaluate_protocol_a_master.remote(chunk_size=chunk_size)

    local_out = REPO_ROOT / "artifacts" / "reid_pose_ablation"
    local_out.mkdir(parents=True, exist_ok=True)
    metrics_path = local_out / "reid_pose_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    snap = metrics["retrieval_results"]["query_snapshots"]
    barn = metrics["retrieval_results"]["query_barn"]

    print("\n" + "=" * 76)
    print("  FINAL PROTOCOL A RETRIEVAL RESULTS (POSE ABLATION):")
    print("=" * 76)
    print(f"  Snapshots -> Parlor: Rank-1: {snap['rank_1']:.2f}% | Rank-5: {snap['rank_5']:.2f}% | Rank-10: {snap['rank_10']:.2f}% | mAP: {snap['mAP']:.2f}%")
    print(f"  Barn -> Parlor:      Rank-1: {barn['rank_1']:.2f}% | Rank-5: {barn['rank_5']:.2f}% | Rank-10: {barn['rank_10']:.2f}% | mAP: {barn['mAP']:.2f}%")
    print(f"\n[OK] Metrics saved locally to: {metrics_path}")
    print("=" * 76 + "\n")
