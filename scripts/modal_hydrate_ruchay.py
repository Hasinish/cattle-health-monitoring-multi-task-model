# -*- coding: utf-8 -*-
"""Modal Hydration and Verification Pipeline for Ruchay 2026 Cattle BCS Dataset.

Source: Ruchay et al., Zenodo Record 20290988 (DOI: 10.5281/zenodo.20290988)
Title: "RGB-D Image Dataset for Automated Body Condition Scoring of Dairy Cows in Rotary Milking Parlors"
Target: Modal Volume 'ruchay-data' mounted at /data/ruchay2026/

Dataset Details:
  - Total Raw Archive: 77.74 GB across 5 session archives + Dataset.xlsx (1,025 cows, 25,700 RGB-D pairs)
  - Session 1: 06.12.2024.zip   (9.09 GB)
  - Session 2: 20.02.2025.zip   (12.88 GB)
  - Session 3: 20.03.2025.zip   (15.74 GB)
  - Session 4 Part 1: 27.03.2025.1.zip (19.93 GB)
  - Session 4 Part 2: 27.03.2025.2.zip (20.10 GB)
  - Metadata:  Dataset.xlsx     (83 KB)

Usage:
  # Download all sessions sequentially (recommended for a full hydration terminal):
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::download_all

  # Download specific session independently:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::download_archive --name 06.12.2024.zip
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::download_archive --name 20.02.2025.zip
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::download_archive --name 20.03.2025.zip
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::download_archive --name 27.03.2025.1.zip
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::download_archive --name 27.03.2025.2.zip

  # Verify dataset on Modal volume:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_ruchay.py::verify
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent storage volume for Ruchay BCS
volume = modal.Volume.from_name("ruchay-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/ruchay2026"

RUCHAY_FILES: Dict[str, Dict[str, Any]] = {
    "Dataset.xlsx": {
        "url": "https://zenodo.org/api/records/20290988/files/Dataset.xlsx/content",
        "size": 85117,
        "is_archive": False,
    },
    "06.12.2024.zip": {
        "url": "https://zenodo.org/api/records/20290988/files/06.12.2024.zip/content",
        "size": 9761921345,
        "is_archive": True,
        "session": "06.12.2024",
    },
    "20.02.2025.zip": {
        "url": "https://zenodo.org/api/records/20290988/files/20.02.2025.zip/content",
        "size": 13833894767,
        "is_archive": True,
        "session": "20.02.2025",
    },
    "20.03.2025.zip": {
        "url": "https://zenodo.org/api/records/20290988/files/20.03.2025.zip/content",
        "size": 16900406793,
        "is_archive": True,
        "session": "20.03.2025",
    },
    "27.03.2025.1.zip": {
        "url": "https://zenodo.org/api/records/20290988/files/27.03.2025.1.zip/content",
        "size": 21398188177,
        "is_archive": True,
        "session": "27.03.2025_part1",
    },
    "27.03.2025.2.zip": {
        "url": "https://zenodo.org/api/records/20290988/files/27.03.2025.2.zip/content",
        "size": 21577717441,
        "is_archive": True,
        "session": "27.03.2025_part2",
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2", "unzip")
    .pip_install("tqdm", "pillow", "openpyxl")
)

app = modal.App("ruchay-bcs-pipeline", image=image)


# ==============================================================================
# DOWNLOAD AND EXTRACTION LOGIC
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=7200,  # 2 hours max per session
    cpu=2.0,
    memory=4096,  # 4 GB RAM
)
def download_ruchay_file(file_name: str) -> Dict[str, Any]:
    """Download a single Ruchay archive/file via aria2c, extract, and cleanup archive."""
    import subprocess
    import threading

    if file_name not in RUCHAY_FILES:
        raise ValueError(f"Unknown file: {file_name}. Valid choices: {list(RUCHAY_FILES.keys())}")

    info = RUCHAY_FILES[file_name]
    os.makedirs(DATASET_DIR, exist_ok=True)
    local_file = os.path.join(DATASET_DIR, file_name)
    aria2_control = local_file + ".aria2"

    print("=" * 70)
    print(f"  RUCHAY 2026 BCS CLOUD PIPELINE: {file_name}")
    print(f"  Expected Size: {info['size'] / (1024**3):.2f} GB")
    print(f"  Target Mount:  {DATASET_DIR}")
    print("=" * 70)

    # 1. Download
    is_complete = os.path.exists(local_file) and not os.path.exists(aria2_control)
    if is_complete and os.path.getsize(local_file) == info["size"]:
        print(f"✓ {file_name} already fully downloaded.")
    else:
        print(f"[*] Downloading {file_name} with 16 parallel connections...")
        stop_event = threading.Event()

        def periodic_commit():
            while not stop_event.wait(60):
                try:
                    volume.commit()
                    print(f"  [Checkpoint] Periodic volume commit completed.", flush=True)
                except Exception as ce:
                    print(f"  [Checkpoint Warning] {ce}", flush=True)

        commit_thread = threading.Thread(target=periodic_commit, daemon=True)
        commit_thread.start()

        cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-j", "16",
            "-k", "2M",
            "--file-allocation=falloc",
            "--summary-interval=10",
            "--check-certificate=false",
            "-c",
            "-o", file_name,
            "-d", DATASET_DIR,
            info["url"],
        ]

        t0 = time.time()
        try:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise RuntimeError(f"aria2c failed with code {res.returncode}")
        finally:
            stop_event.set()
            commit_thread.join(timeout=2.0)

        elapsed = time.time() - t0
        mb_s = (info["size"] / (1024 * 1024)) / max(elapsed, 0.001)
        print(f"✓ Download complete in {elapsed:.1f}s ({mb_s:.1f} MB/s)!")
        volume.commit()

    # 2. Extract if archive
    if info["is_archive"]:
        print(f"[*] Extracting {file_name} into {DATASET_DIR}...")
        t0 = time.time()
        res_unzip = subprocess.run(["unzip", "-q", "-o", local_file, "-d", DATASET_DIR])
        if res_unzip.returncode != 0:
            raise RuntimeError(f"unzip failed with code {res_unzip.returncode}")

        elapsed_extract = time.time() - t0
        print(f"✓ Extracted in {elapsed_extract:.1f}s!")

        # 3. Cleanup source zip to preserve volume quota
        print(f"[*] Cleaning up {file_name} archive...")
        try:
            os.remove(local_file)
            print(f"✓ Removed {file_name} (freed {info['size'] / (1024**3):.2f} GB).")
        except Exception as e:
            print(f"Warning: could not remove zip: {e}")

        volume.commit()
        print("✓ Volume committed successfully!")

    return {"file": file_name, "status": "success"}


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=14400,  # 4 hours max for all sessions
    cpu=2.0,
    memory=4096,
)
def download_all_ruchay() -> Dict[str, Any]:
    """Download and extract all Ruchay 2026 BCS sessions sequentially."""
    print("=" * 70)
    print("  RUCHAY 2026 BCS FULL HYDRATION (ALL SESSIONS)")
    print("  Total Expected Data: 77.74 GB")
    print("=" * 70)

    # First metadata
    download_ruchay_file.local("Dataset.xlsx")

    # Then each session archive
    archives = [
        "06.12.2024.zip",
        "20.02.2025.zip",
        "20.03.2025.zip",
        "27.03.2025.1.zip",
        "27.03.2025.2.zip",
    ]

    for arc in archives:
        print(f"\n>>> Processing {arc}...")
        download_ruchay_file.local(arc)

    print("\n✓ All Ruchay archives downloaded, extracted, and committed!")
    return {"status": "all_completed"}


# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_ruchay() -> Dict[str, Any]:
    """Verify extracted Ruchay 2026 BCS dataset in Modal volume."""
    from PIL import Image

    print("=" * 70)
    print("  RUCHAY 2026 CATTLE BCS PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {DATASET_DIR}")
    print("=" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory {DATASET_DIR} does not exist in volume.")
        return {"status": "not_found", "total_files": 0}

    total_files = 0
    total_bytes = 0
    rgb_images = []
    depth_images = []
    session_dirs = []

    for item in os.listdir(DATASET_DIR):
        p = os.path.join(DATASET_DIR, item)
        if os.path.isdir(p):
            session_dirs.append(item)

    for root, _, files in os.walk(DATASET_DIR):
        total_files += len(files)
        for f in files:
            p = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
            fl = f.lower()
            if fl.endswith(".png") or fl.endswith(".jpg"):
                if "depth" in fl:
                    depth_images.append(p)
                else:
                    rgb_images.append(p)

    size_gb = total_bytes / (1024 ** 3)
    meta_xlsx = os.path.join(DATASET_DIR, "Dataset.xlsx")
    meta_exists = os.path.exists(meta_xlsx)

    print(f"✓ Total Files Found:    {total_files:,}")
    print(f"✓ Total Volume Storage: {size_gb:.2f} GB")
    print(f"✓ Session Folders:      {len(session_dirs)} ({session_dirs})")
    print(f"✓ RGB Images:           {len(rgb_images):,}")
    print(f"✓ Depth Maps:           {len(depth_images):,}")
    print(f"✓ Metadata Spreadsheet: {'FOUND (Dataset.xlsx)' if meta_exists else 'MISSING'}")

    sample_images = (rgb_images[:3] + depth_images[:3])
    if sample_images:
        print("\nSample Image Decodes:")
        for p in sample_images:
            try:
                with Image.open(p) as img:
                    rel_p = os.path.relpath(p, DATASET_DIR)
                    print(f"  ✓ {rel_p}: format={img.format}, size={img.size}, mode={img.mode}")
            except Exception as e:
                print(f"  ✗ {p}: {e}")

    # Check for leftover zips
    zips_left = [f for f in os.listdir(DATASET_DIR) if f.endswith(".zip")]
    if zips_left:
        print(f"\n[WARNING] Leftover zip archives detected: {zips_left}")
    else:
        print("\n✓ All temporary zip archives successfully cleaned up!")

    return {
        "status": "verified" if total_files > 1000 and meta_exists else "incomplete",
        "total_files": total_files,
        "size_gb": round(size_gb, 2),
        "sessions": session_dirs,
        "rgb_count": len(rgb_images),
        "depth_count": len(depth_images),
    }


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def download_all():
    """Run full sequential download of all Ruchay sessions."""
    download_all_ruchay.remote()


@app.local_entrypoint()
def download_archive(name: str):
    """Run download of a specific session archive."""
    download_ruchay_file.remote(name)


@app.local_entrypoint()
def verify():
    """Run standalone verification."""
    res = verify_ruchay.remote()
    print(f"\n✓ Verification result: {res}")
