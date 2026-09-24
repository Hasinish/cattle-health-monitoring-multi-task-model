# -*- coding: utf-8 -*-
"""Modal Hydration and Verification Pipeline for CBVD-5 Behavior Dataset.

Source: Li et al., CBVD-5 (Nature Scientific Reports 2024)
Kaggle: fandaoerji/cbvd-5cow-behavior-video-dataset
Target: Modal Volume 'cbvd5-data' mounted at /data/cbvd5/

Dataset Details:
  - 887 MP4 videos (687 primary + 200 supplementary across 107 biological cows)
  - 206,100 mini frames + 5,322 annotated label frames
  - Total Size: ~14.8 GB

Usage:
  # Download to Modal volume on profile hasinishrak2015:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_cbvd5.py::download

  # Download to Modal volume on profile dryousufmozumder:
  modal run --profile dryousufmozumder scripts/modal_hydrate_cbvd5.py::download

  # Verify on either profile:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_cbvd5.py::verify
  modal run --profile dryousufmozumder scripts/modal_hydrate_cbvd5.py::verify
"""

from __future__ import annotations

import os
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, Any

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent storage volume for CBVD-5
volume = modal.Volume.from_name("cbvd5-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/cbvd5"
KAGGLE_HANDLE = "fandaoerji/cbvd-5cow-behavior-video-dataset"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kagglehub", "tqdm")
)

app = modal.App("cbvd5-behavior-pipeline", image=image)


# ==============================================================================
# DOWNLOAD AND EXTRACTION LOGIC
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=7200,  # 2 hours max
    cpu=2.0,
    memory=4096,  # 4 GB RAM
)
def download_cbvd5_remote() -> Dict[str, Any]:
    """Download CBVD-5 via kagglehub into persistent Modal volume."""
    import kagglehub

    print("=" * 70)
    print("  CBVD-5 KAGGLE BEHAVIOR CLOUD PIPELINE")
    print(f"  Dataset Handle: {KAGGLE_HANDLE}")
    print(f"  Target Mount:   {DATASET_DIR}")
    print("=" * 70)

    # Check if already downloaded & populated
    if os.path.exists(DATASET_DIR):
        existing_mp4s = sum(
            1 for root, _, files in os.walk(DATASET_DIR)
            for f in files if f.lower().endswith(".mp4")
        )
        if existing_mp4s >= 800:
            print(f"✓ CBVD-5 is already hydrated in volume! ({existing_mp4s} MP4 videos found).")
            return {"status": "already_hydrated", "mp4_count": existing_mp4s}

    print("\n[1/2] Downloading from Kaggle via kagglehub (unauthenticated cloud pipe)...")
    t0 = time.time()
    cache_path = kagglehub.dataset_download(KAGGLE_HANDLE)
    elapsed_dl = time.time() - t0
    print(f"✓ Download complete in {elapsed_dl:.1f}s! Downloaded to: {cache_path}")

    # Sync from kagglehub cache to persistent volume
    print(f"\n[2/2] Syncing dataset into persistent volume: {DATASET_DIR}...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    t0_sync = time.time()
    copied_files = 0
    total_bytes = 0

    for root, dirs, files in os.walk(cache_path):
        rel_dir = os.path.relpath(root, cache_path)
        dest_root = os.path.join(DATASET_DIR, rel_dir) if rel_dir != "." else DATASET_DIR
        os.makedirs(dest_root, exist_ok=True)

        for f in files:
            src_file = os.path.join(root, f)
            dest_file = os.path.join(dest_root, f)
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
                copied_files += 1
                try:
                    total_bytes += os.path.getsize(dest_file)
                except OSError:
                    pass

    elapsed_sync = time.time() - t0_sync
    print(f"✓ Synced {copied_files:,} files ({total_bytes / (1024**3):.2f} GB) in {elapsed_sync:.1f}s!")

    volume.commit()
    print("✓ Volume committed successfully!")
    return {"status": "success", "copied_files": copied_files}


# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_cbvd5_remote() -> Dict[str, Any]:
    """Verify extracted CBVD-5 dataset in Modal volume."""
    print("=" * 70)
    print("  CBVD-5 KAGGLE BEHAVIOR PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {DATASET_DIR}")
    print("=" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory {DATASET_DIR} does not exist in volume.")
        return {"status": "not_found", "total_files": 0}

    total_files = 0
    total_bytes = 0
    mp4_files = []
    image_files = []

    for root, _, files in os.walk(DATASET_DIR):
        total_files += len(files)
        for f in files:
            p = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
            fl = f.lower()
            if fl.endswith(".mp4"):
                mp4_files.append(p)
            elif fl.endswith((".jpg", ".png", ".jpeg")):
                image_files.append(p)

    size_gb = total_bytes / (1024 ** 3)
    print(f"✓ Total Files Found:    {total_files:,}")
    print(f"✓ Total Volume Storage: {size_gb:.2f} GB")
    print(f"✓ MP4 Videos:           {len(mp4_files):,} (expected ~887)")
    print(f"✓ Mini / Frame Images:  {len(image_files):,} (expected ~206,100)")

    is_verified = len(mp4_files) >= 800
    status_str = "verified" if is_verified else "incomplete"
    print(f"\n[STATUS] CBVD-5 volume status: {status_str.upper()} ({len(mp4_files)}/887 MP4s)")

    return {
        "status": status_str,
        "total_files": total_files,
        "size_gb": round(size_gb, 2),
        "mp4_count": len(mp4_files),
        "image_count": len(image_files),
    }


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def download():
    """Run full CBVD-5 download and hydration."""
    res = download_cbvd5_remote.remote()
    print(f"\n✓ Download result: {res}")


@app.local_entrypoint()
def verify():
    """Run standalone verification."""
    res = verify_cbvd5_remote.remote()
    print(f"\n✓ Verification result: {res}")
