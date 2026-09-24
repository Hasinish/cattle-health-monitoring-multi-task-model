# -*- coding: utf-8 -*-
"""Modal Hydration and Verification Pipeline for OpenCows2020 Dataset (Legacy Baseline).

Source: Kaggle (amibhavsar/open-cow-2020 & andrewmvd/opencows2020)
Paper: Gao et al., CVPRW 2020 (DOI: 10.1109/CVPRW50498.2020.00140)
Target: Modal Volume 'opencows-data' mounted at /data/opencow2020-DatasetNinja/

Dataset Details:
  - 4,736 images across 46 cows (Train: 3,586, Val: 654, Official Test: 496)
  - Format: identification-train/img/ and identification-test/img/
  - Total Size: ~3.2 GB

Usage:
  # Download to Modal volume on profile hasinishrak2015:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_opencows.py::download

  # Download to Modal volume on profile dryousufmozumder:
  modal run --profile dryousufmozumder scripts/modal_hydrate_opencows.py::download

  # Verify on either profile:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_opencows.py::verify
  modal run --profile dryousufmozumder scripts/modal_hydrate_opencows.py::verify
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

# Persistent storage volume for OpenCows2020
volume = modal.Volume.from_name("opencows-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/opencow2020-DatasetNinja"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kagglehub", "tqdm", "pillow")
)

app = modal.App("opencows-reid-pipeline", image=image)


# ==============================================================================
# DOWNLOAD AND EXTRACTION LOGIC
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=3600,  # 1 hour max
    cpu=2.0,
    memory=4096,  # 4 GB RAM
)
def download_opencows_remote() -> Dict[str, Any]:
    """Download OpenCows2020 via kagglehub into persistent Modal volume."""
    import kagglehub

    print("=" * 70)
    print("  OPENCOWS2020 (LEGACY BASELINE) CLOUD PIPELINE")
    print(f"  Target Mount: {DATASET_DIR}")
    print("=" * 70)

    train_dst = os.path.join(DATASET_DIR, "identification-train", "img")
    test_dst = os.path.join(DATASET_DIR, "identification-test", "img")

    # Check if already organized
    if os.path.exists(train_dst) and os.path.exists(test_dst):
        n_train = len([f for f in os.listdir(train_dst) if f.lower().endswith((".jpg", ".png"))])
        n_test = len([f for f in os.listdir(test_dst) if f.lower().endswith((".jpg", ".png"))])
        if n_train >= 4000 and n_test >= 400:
            print(f"✓ OpenCows2020 is already organized in volume! (Train: {n_train:,}, Test: {n_test:,})")
            return {"status": "already_organized", "train_count": n_train, "test_count": n_test}

    print("\n[1/2] Downloading amibhavsar/open-cow-2020 via kagglehub...")
    t0 = time.time()
    cache_path = Path(kagglehub.dataset_download("amibhavsar/open-cow-2020"))
    elapsed_dl = time.time() - t0
    print(f"✓ Download complete in {elapsed_dl:.1f}s! Cache path: {cache_path}")

    # Locate identification images root
    id_img_root = None
    for cand in [
        cache_path / "identification" / "images",
        cache_path / "10m32xl88x2b61zlkkgz3fml17" / "identification" / "images",
        cache_path / "images",
    ]:
        if cand.exists() and (cand / "train").exists():
            id_img_root = cand
            break

    if not id_img_root:
        for p in cache_path.rglob("train"):
            if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
                id_img_root = p.parent
                break

    if not id_img_root:
        raise RuntimeError(f"Could not locate identification/images within {cache_path}")

    print(f"\n[2/2] Formatting and organizing images into {DATASET_DIR}...")
    train_src = id_img_root / "train"
    test_src = id_img_root / "test"

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    def copy_split(src_dir: Path, dst_dir: str, split_name: str) -> int:
        copied = 0
        for cow_folder in sorted(src_dir.iterdir()):
            if not cow_folder.is_dir():
                continue
            try:
                cow_id = str(int(cow_folder.name))
            except ValueError:
                continue
            for img in cow_folder.glob("*.jpg"):
                dest_file = os.path.join(dst_dir, f"{cow_id}_{img.name}")
                if not os.path.exists(dest_file):
                    shutil.copy2(img, dest_file)
                copied += 1
        print(f"  ✓ Formatted and copied {copied:,} {split_name} images.")
        return copied

    n_tr = copy_split(train_src, train_dst, "train")
    n_te = copy_split(test_src, test_dst, "test")

    volume.commit()
    print("✓ Volume committed successfully!")
    return {"status": "success", "train_copied": n_tr, "test_copied": n_te}


# ==============================================================================
# VERIFICATION FUNCTION
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_opencows_remote() -> Dict[str, Any]:
    """Verify formatted OpenCows2020 dataset in Modal volume."""
    from PIL import Image

    print("=" * 70)
    print("  OPENCOWS2020 (LEGACY BASELINE) PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {DATASET_DIR}")
    print("=" * 70)

    train_dst = os.path.join(DATASET_DIR, "identification-train", "img")
    test_dst = os.path.join(DATASET_DIR, "identification-test", "img")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        print(f"[ERROR] Split directories not found in {DATASET_DIR}.")
        return {"status": "not_found", "total_images": 0}

    tr_files = [f for f in os.listdir(train_dst) if f.lower().endswith((".jpg", ".png"))]
    te_files = [f for f in os.listdir(test_dst) if f.lower().endswith((".jpg", ".png"))]

    total_images = len(tr_files) + len(te_files)
    tr_cows = {f.split("_")[0] for f in tr_files if "_" in f}
    te_cows = {f.split("_")[0] for f in te_files if "_" in f}

    total_bytes = sum(
        os.path.getsize(os.path.join(d, f))
        for d, flist in [(train_dst, tr_files), (test_dst, te_files)]
        for f in flist
    )
    size_gb = total_bytes / (1024 ** 3)

    print(f"✓ Total Images Found: {total_images:,} (expected 4,736)")
    print(f"✓ Total Volume Space: {size_gb:.2f} GB")
    print(f"  - Train Partition:  {len(tr_files):,} images across {len(tr_cows)} cows (expected 4,240 / 46 cows)")
    print(f"  - Test Partition:   {len(te_files):,} images across {len(te_cows)} cows (expected 496 / 46 cows)")

    sample_images = (
        [os.path.join(train_dst, f) for f in tr_files[:2]]
        + [os.path.join(test_dst, f) for f in te_files[:2]]
    )
    if sample_images:
        print("\nSample Image Decodes:")
        for p in sample_images:
            try:
                with Image.open(p) as img:
                    rel_p = os.path.relpath(p, DATASET_DIR)
                    print(f"  ✓ {rel_p}: format={img.format}, size={img.size}, mode={img.mode}")
            except Exception as e:
                print(f"  ✗ {p}: {e}")

    is_verified = total_images >= 4700 and len(tr_cows) == 46
    status_str = "verified" if is_verified else "incomplete"
    print(f"\n[STATUS] OpenCows2020 volume status: {status_str.upper()} ({total_images}/4,736 images)")

    return {
        "status": status_str,
        "total_images": total_images,
        "train_images": len(tr_files),
        "test_images": len(te_files),
        "unique_cows": len(tr_cows),
        "size_gb": round(size_gb, 2),
    }


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def download():
    """Run full OpenCows2020 download and organization."""
    res = download_opencows_remote.remote()
    print(f"\n✓ Download result: {res}")


@app.local_entrypoint()
def verify():
    """Run standalone verification."""
    res = verify_opencows_remote.remote()
    print(f"\n✓ Verification result: {res}")
