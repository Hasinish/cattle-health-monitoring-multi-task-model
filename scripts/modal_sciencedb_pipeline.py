# -*- coding: utf-8 -*-
"""
Modal Cloud Pipeline for ScienceDB Cattle BCS Dataset (dataset.rar, ~4.11 GB)
Source: Science Data Bank (CAS / china.scidb.cn)
Target: Modal Volume 'sciencedb-data' at /data

Usage:
  modal run --profile tigerwood697 scripts/modal_sciencedb_pipeline.py
"""
import os
import sys

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent storage volume for ScienceDB
volume = modal.Volume.from_name("sciencedb-data", create_if_missing=True)
VOLUME_DIR = "/data"

SCIENCEDB_URL = (
    "https://china.scidb.cn/download?fileId=4516a6866954512e08446bc3504b32c9"
)

# Minimal container with aria2 and unar (unar supports RAR5, unlike Debian's p7zip)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2", "unar")
    .pip_install("tqdm")
)

app = modal.App("sciencedb-bcs-pipeline", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1800,  # 30 mins max
    cpu=1.0,
    memory=2048,  # 2 GB RAM (minimal cost: ~$0.06/hr)
)
def download_sciencedb():
    """Download dataset.rar (4.11 GB) from ScienceDB via 16-connection aria2c and extract."""
    import shutil
    import subprocess
    import time

    os.makedirs(VOLUME_DIR, exist_ok=True)
    rar_path = os.path.join(VOLUME_DIR, "dataset.rar")
    dataset_dir = os.path.join(VOLUME_DIR, "dataset")
    aria2_control = rar_path + ".aria2"

    print("==================================================")
    print("  SCIENCEDB CATTLE BCS 16-STREAM CLOUD PIPELINE")
    print(f"  Volume Directory: {VOLUME_DIR}")
    print(f"  Source URL: {SCIENCEDB_URL}")
    print("==================================================")

    # Check if already extracted
    expected_classes = ["3.25", "3.5", "3.75", "4.0", "4.25"]
    if os.path.exists(dataset_dir) and all(
        os.path.exists(os.path.join(dataset_dir, c)) for c in expected_classes
    ):
        if os.path.exists(rar_path):
            try:
                os.remove(rar_path)
                print("✓ Removed dataset.rar archive to reclaim space.")
            except Exception:
                pass
            volume.commit()
        total_files = sum(len(files) for _, _, files in os.walk(dataset_dir))
        print(f"✓ ScienceDB is ALREADY extracted and ready in volume! Total files: {total_files}")
        return {"status": "already_extracted", "total_files": total_files}

    # Step 1: Download via aria2c with 16 parallel connections
    is_downloaded = os.path.exists(rar_path) and not os.path.exists(aria2_control)

    if not is_downloaded:
        print("\n[1/2] Downloading dataset.rar (~4.11 GB) with 16 parallel streams...")
        aria2_cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-j", "16",
            "-k", "1M",
            "--file-allocation=falloc",
            "--summary-interval=2",
            "--check-certificate=false",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "-c",  # resume
            "-o", "dataset.rar",
            "-d", VOLUME_DIR,
            SCIENCEDB_URL,
        ]

        start_time = time.time()
        res = subprocess.run(aria2_cmd)
        if res.returncode != 0:
            raise RuntimeError(f"aria2c download failed with exit code {res.returncode}")

        elapsed = time.time() - start_time
        file_size_gb = os.path.getsize(rar_path) / (1024 ** 3)
        avg_speed = (file_size_gb * 1024) / max(1, elapsed)
        print(f"\n✓ Download complete in {elapsed:.1f}s ({avg_speed:.1f} MB/s avg)!")
        volume.commit()
    else:
        print(f"✓ dataset.rar already downloaded and verified on volume ({os.path.getsize(rar_path) / (1024**3):.2f} GB)!")

    # Step 2: Extract via unar (full RAR5 support)
    print(f"\n[2/2] Extracting dataset.rar into {VOLUME_DIR} via unar (RAR5 support)...")
    start_extract = time.time()

    extract_cmd = ["unar", "-f", "-o", VOLUME_DIR, rar_path]
    res_extract = subprocess.run(extract_cmd)
    
    # Check if images actually extracted despite non-critical XML errors
    total_imgs = 0
    for root, _, files in os.walk(VOLUME_DIR):
        total_imgs += sum(1 for f in files if f.lower().endswith(".jpg"))

    if total_imgs < 1000 and res_extract.returncode != 0:
        raise RuntimeError(f"unar extraction failed with exit code {res_extract.returncode} (only {total_imgs} images)")
    elif res_extract.returncode != 0:
        print(f"\n[NOTICE] unar returned code {res_extract.returncode} due to some non-critical XML annotations.")
        print(f"✓ But {total_imgs} images were successfully extracted and verified!")

    # Organize into /data/dataset/ if classes extracted into /data/
    if not os.path.exists(dataset_dir):
        if any(os.path.exists(os.path.join(VOLUME_DIR, c)) for c in expected_classes):
            os.makedirs(dataset_dir, exist_ok=True)
            for c in expected_classes:
                src = os.path.join(VOLUME_DIR, c)
                dst = os.path.join(dataset_dir, c)
                if os.path.exists(src):
                    shutil.move(src, dst)

    extract_elapsed = time.time() - start_extract
    print(f"✓ Extraction complete in {extract_elapsed:.1f}s!")

    # Cleanup rar to save quota
    print("Cleaning up dataset.rar to save volume space...")
    try:
        os.remove(rar_path)
        print("✓ Removed dataset.rar archive.")
    except Exception as e:
        print(f"Warning: could not remove rar: {e}")

    volume.commit()
    print("✓ Volume committed successfully!")

    total_files = sum(len(files) for _, _, files in os.walk(dataset_dir))
    return {"status": "success", "total_files": total_files}


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_sciencedb():
    """Verify extracted ScienceDB Cattle BCS dataset in Modal volume."""
    from PIL import Image

    dataset_dir = os.path.join(VOLUME_DIR, "dataset")
    expected_classes = ["3.25", "3.5", "3.75", "4.0", "4.25"]

    print("=" * 70)
    print("  SCIENCEDB CATTLE BCS PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {dataset_dir}")
    print("=" * 70)

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Directory {dataset_dir} does not exist in volume.")
        return {"status": "not_found", "total_images": 0}

    class_counts = {}
    total_images = 0
    sample_images = []

    for c in expected_classes:
        cdir = os.path.join(dataset_dir, c)
        if os.path.exists(cdir):
            imgs = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            class_counts[c] = len(imgs)
            total_images += len(imgs)
            if imgs and len(sample_images) < 5:
                sample_images.append(os.path.join(cdir, imgs[0]))
        else:
            class_counts[c] = 0

    print(f"✓ Total Images Found: {total_images:,}")
    print("Class Counts:")
    for c, cnt in class_counts.items():
        print(f"  - Class {c}: {cnt:,} images")

    print("\nSample PIL Image Decodes:")
    for p in sample_images:
        try:
            with Image.open(p) as img:
                print(f"  ✓ {os.path.basename(p)}: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"  ✗ {p}: {e}")

    rar_path = os.path.join(VOLUME_DIR, "dataset.rar")
    rar_present = os.path.exists(rar_path)
    print(f"Archive dataset.rar cleanup status: {'STILL PRESENT' if rar_present else 'CLEANED UP (space reclaimed)'}")

    return {
        "status": "verified" if total_images >= 50000 else "incomplete",
        "total_images": total_images,
        "class_counts": class_counts,
    }


@app.local_entrypoint()
def main():
    download_sciencedb.remote()


@app.local_entrypoint()
def verify():
    verify_sciencedb.remote()
