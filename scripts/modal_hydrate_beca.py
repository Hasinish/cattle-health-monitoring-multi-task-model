# -*- coding: utf-8 -*-
"""Modal Hydration and Verification Pipeline for BECA Dataset (BECA-D & BECA-L).

Source: Figshare file 63928608 (DOI: 10.6084/m9.figshare.32070171)
Title: "BECA: A Computer Vision Dataset for Long-Term Recognition In Beef Cattle"
Target: Modal Volume 'beca-data' mounted at /data/beca/

Dataset Details:
  - Archive: BECA.zip (20,309,712,996 bytes / ~18.91 GB)
  - Sub-datasets:
    - BECA-D: 16,889 images across 5,661 beef cattle (scale/pretraining benchmark)
    - BECA-L: 12,172 images across 103 beef cattle tracked over 5 continuous months (long-term benchmark)
  - Total extracted images: 29,061 images across 5,764 total cattle identities

Usage:
  # Download and extract on profile hasinishrak2015:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_beca.py::download

  # Download and extract on profile dryousufmozumder:
  modal run --profile dryousufmozumder scripts/modal_hydrate_beca.py::download

  # Verify dataset on Modal volume:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_beca.py::verify
  modal run --profile dryousufmozumder scripts/modal_hydrate_beca.py::verify
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

# Persistent storage volume for BECA
volume = modal.Volume.from_name("beca-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/beca"

FIGSHARE_URL = "https://ndownloader.figshare.com/files/63928608"
EXPECTED_ARCHIVE_SIZE = 20309712996  # 18.91 GB

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2", "unzip")
    .pip_install("tqdm", "pillow")
)

app = modal.App("beca-reid-pipeline", image=image)


# ==============================================================================
# DOWNLOAD AND EXTRACTION LOGIC
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=7200,  # 2 hours max
    cpu=2.0,
    memory=4096,  # 4 GB RAM
)
def download_beca_remote() -> Dict[str, Any]:
    """Download BECA.zip (18.91 GB) from Figshare via 16-connection aria2c, extract, and cleanup."""
    import subprocess
    import threading

    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, "BECA.zip")
    aria2_control = zip_path + ".aria2"

    print("=" * 70)
    print("  BECA (BECA-D & BECA-L) RE-ID CLOUD PIPELINE")
    print(f"  Source Endpoint: {FIGSHARE_URL}")
    print(f"  Target Mount:    {DATASET_DIR}")
    print(f"  Archive Size:    {EXPECTED_ARCHIVE_SIZE / (1024**3):.2f} GB")
    print("=" * 70)

    # Check if already extracted
    beca_d = os.path.join(DATASET_DIR, "BECA-D")
    beca_l = os.path.join(DATASET_DIR, "BECA-L")
    if os.path.exists(beca_d) and os.path.exists(beca_l):
        d_imgs = sum(len(files) for _, _, files in os.walk(beca_d))
        l_imgs = sum(len(files) for _, _, files in os.walk(beca_l))
        if d_imgs >= 15000 and l_imgs >= 10000:
            print(f"✓ BECA is already extracted in volume! (BECA-D: {d_imgs:,}, BECA-L: {l_imgs:,})")
            return {"status": "already_extracted", "beca_d": d_imgs, "beca_l": l_imgs}

    # Step 1: Download
    is_complete = os.path.exists(zip_path) and not os.path.exists(aria2_control) and os.path.getsize(zip_path) == EXPECTED_ARCHIVE_SIZE
    if not is_complete:
        print("\n[1/2] Downloading BECA.zip with 16 parallel connections...")
        stop_event = threading.Event()

        def periodic_commit():
            while not stop_event.wait(60):
                try:
                    volume.commit()
                    print("  [Checkpoint] Periodic volume commit completed.", flush=True)
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
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "-c",
            "-o", "BECA.zip",
            "-d", DATASET_DIR,
            FIGSHARE_URL,
        ]

        t0 = time.time()
        try:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise RuntimeError(f"aria2c download failed with exit code {res.returncode}")
        finally:
            stop_event.set()
            commit_thread.join(timeout=2.0)

        elapsed = time.time() - t0
        mb_s = (EXPECTED_ARCHIVE_SIZE / (1024 * 1024)) / max(elapsed, 0.001)
        print(f"✓ Download complete in {elapsed:.1f}s ({mb_s:.1f} MB/s)!")
        volume.commit()
    else:
        print(f"✓ BECA.zip already downloaded ({os.path.getsize(zip_path) / (1024**3):.2f} GB).")

    # Step 2: Extract
    print(f"\n[2/2] Extracting BECA.zip into {DATASET_DIR}...")
    t0 = time.time()
    res_unzip = subprocess.run(["unzip", "-q", "-o", zip_path, "-d", DATASET_DIR])
    if res_unzip.returncode != 0:
        raise RuntimeError(f"unzip failed with exit code {res_unzip.returncode}")

    elapsed_extract = time.time() - t0
    print(f"✓ Extraction complete in {elapsed_extract:.1f}s!")

    # Step 3: Cleanup zip
    print("[*] Cleaning up BECA.zip to reclaim volume space...")
    try:
        os.remove(zip_path)
        print(f"✓ Removed BECA.zip (freed {EXPECTED_ARCHIVE_SIZE / (1024**3):.2f} GB).")
    except Exception as e:
        print(f"Warning: could not remove zip: {e}")

    volume.commit()
    print("✓ Volume committed successfully!")
    return {"status": "success"}


# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_beca_remote() -> Dict[str, Any]:
    """Verify extracted BECA dataset (both BECA-D and BECA-L) in Modal volume."""
    from PIL import Image

    print("=" * 70)
    print("  BECA (BECA-D & BECA-L) RE-ID PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {DATASET_DIR}")
    print("=" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory {DATASET_DIR} does not exist in volume.")
        return {"status": "not_found", "total_files": 0}

    beca_d = os.path.join(DATASET_DIR, "BECA-D")
    beca_l = os.path.join(DATASET_DIR, "BECA-L")

    d_imgs = []
    l_imgs = []
    d_cows = set()
    l_cows = set()

    if os.path.exists(beca_d):
        for root, _, files in os.walk(beca_d):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    p = os.path.join(root, f)
                    d_imgs.append(p)
                    d_cows.add(os.path.basename(root))

    if os.path.exists(beca_l):
        for root, _, files in os.walk(beca_l):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    p = os.path.join(root, f)
                    l_imgs.append(p)
                    l_cows.add(os.path.basename(root))

    total_images = len(d_imgs) + len(l_imgs)
    print(f"✓ Total Images Found: {total_images:,}")
    print(f"  - BECA-D (Scale Benchmark):     {len(d_imgs):,} images across {len(d_cows)} cows (expected 16,889 / 5,661 cows)")
    print(f"  - BECA-L (Longitudinal 5-month): {len(l_imgs):,} images across {len(l_cows)} cows (expected 12,172 / 103 cows)")

    sample_images = (d_imgs[:2] + l_imgs[:2])
    if sample_images:
        print("\nSample Image Decodes:")
        for p in sample_images:
            try:
                with Image.open(p) as img:
                    rel_p = os.path.relpath(p, DATASET_DIR)
                    print(f"  ✓ {rel_p}: format={img.format}, size={img.size}, mode={img.mode}")
            except Exception as e:
                print(f"  ✗ {p}: {e}")

    zip_present = os.path.exists(os.path.join(DATASET_DIR, "BECA.zip"))
    print(f"\nArchive BECA.zip cleanup status: {'STILL PRESENT' if zip_present else 'CLEANED UP (space reclaimed)'}")

    is_verified = len(d_imgs) >= 15000 and len(l_imgs) >= 10000
    status_str = "verified" if is_verified else "incomplete"
    print(f"\n[STATUS] BECA volume status: {status_str.upper()} ({total_images} total images)")

    return {
        "status": status_str,
        "total_images": total_images,
        "beca_d_images": len(d_imgs),
        "beca_d_cows": len(d_cows),
        "beca_l_images": len(l_imgs),
        "beca_l_cows": len(l_cows),
    }


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def download():
    """Run full BECA download, extraction, and verification."""
    res = download_beca_remote.remote()
    print(f"\n✓ Download result: {res}")


@app.local_entrypoint()
def verify():
    """Run standalone verification."""
    res = verify_beca_remote.remote()
    print(f"\n✓ Verification result: {res}")
