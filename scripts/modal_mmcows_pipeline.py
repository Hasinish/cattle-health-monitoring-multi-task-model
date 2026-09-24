# -*- coding: utf-8 -*-
"""
Modal Cloud Pipeline for MmCows Behavior Dataset (cropped_bboxes.zip)
Source: Hugging Face (neis-lab/mmcows)
Target: Modal Volume 'mmcows-data' at /data

Usage:
  modal run --profile tigerwood697 scripts/modal_mmcows_pipeline.py::download_mmcows
"""
import os
import sys

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent storage volume for MmCows
volume = modal.Volume.from_name("mmcows-data", create_if_missing=True)
VOLUME_DIR = "/data"

# Minimal container image with unzip and Rust hf_transfer
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install("huggingface_hub[hf_transfer]", "hf_transfer", "tqdm")
)

app = modal.App("mmcows-behavior-pipeline", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1800,  # 30 mins max
    cpu=1.0,
    memory=2048,  # 2 GB RAM (minimal cost)
)
def download_mmcows(hf_token: str = ""):
    """Download cropped_bboxes.zip (12.7 GB) from Hugging Face using Rust hf_transfer."""
    import subprocess
    import time
    import zipfile

    token = hf_token or os.environ.get("HF_TOKEN", "")

    os.makedirs(VOLUME_DIR, exist_ok=True)
    zip_path = os.path.join(VOLUME_DIR, "cropped_bboxes.zip")
    extracted_dir = os.path.join(VOLUME_DIR, "cropped_bboxes")
    aria2_control = zip_path + ".aria2"

    print("==================================================")
    print("  MMCOWS BEHAVIOR HUGGING FACE CLOUD PIPELINE (RUST HF_TRANSFER)")
    print(f"  Volume Directory: {VOLUME_DIR}")
    if token:
        print("  Auth Status: Bearer Token Attached (CDN priority!)")
    else:
        print("  Auth Status: Unauthenticated")
    print("==================================================")

    # Check if already extracted
    if os.path.exists(extracted_dir):
        total_files = sum(len(files) for _, _, files in os.walk(extracted_dir))
        print(f"✓ MmCows is already extracted in volume! Total files: {total_files}")
        return {"status": "already_extracted", "total_files": total_files}

    # Clean any stale aria2 files if present
    if os.path.exists(aria2_control):
        try:
            os.remove(aria2_control)
        except Exception:
            pass

    # Step 1: Download using Rust hf_transfer
    is_complete_zip = (
        os.path.exists(zip_path)
        and zipfile.is_zipfile(zip_path)
    )

    if not is_complete_zip:
        print("\n[1/2] Downloading cropped_bboxes.zip (~12.7 GB) via Rust hf_transfer...")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import hf_hub_download

        start_time = time.time()
        try:
            downloaded_file = hf_hub_download(
                repo_id="neis-lab/mmcows",
                filename="cropped_bboxes.zip",
                repo_type="dataset",
                token=token if token else None,
                local_dir=VOLUME_DIR,
            )
        except Exception as e:
            print(f"Error during hf_hub_download: {e}")
            raise

        elapsed = time.time() - start_time
        downloaded_gb = os.path.getsize(downloaded_file) / (1024 ** 3)
        speed_mb = (downloaded_gb * 1024) / max(1, elapsed)
        print(f"\n✓ Rust hf_transfer complete in {elapsed:.1f}s ({speed_mb:.1f} MB/s avg)!")
        volume.commit()
    else:
        print(f"✓ cropped_bboxes.zip already downloaded and verified ({os.path.getsize(zip_path) / (1024**3):.2f} GB)")

    # Step 2: Extract archive
    print(f"\n[2/2] Extracting cropped_bboxes.zip into {VOLUME_DIR}...")
    start_unzip = time.time()
    unzip_cmd = ["unzip", "-q", "-o", zip_path, "-d", VOLUME_DIR]
    res_unzip = subprocess.run(unzip_cmd)
    if res_unzip.returncode != 0:
        raise RuntimeError(f"unzip failed with exit code {res_unzip.returncode}")

    unzip_elapsed = time.time() - start_unzip
    print(f"✓ Extraction complete in {unzip_elapsed:.1f}s!")

    # Remove zip to save volume storage quota
    print("Cleaning up cropped_bboxes.zip to preserve volume space...")
    try:
        os.remove(zip_path)
        print("✓ Removed cropped_bboxes.zip archive.")
    except Exception as e:
        print(f"Warning: could not remove zip: {e}")

    volume.commit()
    print("✓ Volume committed successfully!")

    total_files = sum(len(files) for _, _, files in os.walk(extracted_dir))
    return {"status": "success", "total_files": total_files}


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_mmcows():
    """Verify extracted MmCows behavior dataset in Modal volume."""
    from PIL import Image

    extracted_dir = os.path.join(VOLUME_DIR, "cropped_bboxes")
    behaviors_dir = os.path.join(extracted_dir, "behaviors")

    print("=" * 70)
    print("  MMCOWS BEHAVIOR DATASET PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {extracted_dir}")
    print("=" * 70)

    if not os.path.exists(extracted_dir):
        print(f"[ERROR] Directory {extracted_dir} does not exist in volume.")
        return {"status": "not_found", "total_files": 0}

    total_files = 0
    total_bytes = 0
    class_counts = {}
    sample_images = []
    unique_cows = set()

    for root, _, files in os.walk(extracted_dir):
        total_files += len(files)
        for f in files:
            p = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                parts = f.split("_")
                if len(parts) >= 3:
                    unique_cows.add(parts[2])
                if len(sample_images) < 5:
                    sample_images.append(p)

    if os.path.exists(behaviors_dir):
        for c in sorted(os.listdir(behaviors_dir)):
            cdir = os.path.join(behaviors_dir, c)
            if os.path.isdir(cdir):
                class_counts[c] = len(os.listdir(cdir))

    size_gb = total_bytes / (1024 ** 3)
    print(f"✓ Total Files Found: {total_files:,}")
    print(f"✓ Total Disk Space:  {size_gb:.2f} GB")
    print(f"✓ Unique Cow IDs:    {len(unique_cows)} cows identified")
    if class_counts:
        print("Behavior Class Counts:")
        for c, count in class_counts.items():
            print(f"  - Class {c}: {count:,} images")

    print("\nSample PIL Image Decodes:")
    for p in sample_images:
        try:
            with Image.open(p) as img:
                print(f"  ✓ {os.path.basename(p)}: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"  ✗ {p}: {e}")

    zip_path = os.path.join(VOLUME_DIR, "cropped_bboxes.zip")
    zip_present = os.path.exists(zip_path)
    print(f"Archive cropped_bboxes.zip cleanup status: {'STILL PRESENT' if zip_present else 'CLEANED UP (space reclaimed)'}")

    return {
        "status": "verified" if total_files >= 200000 else "incomplete",
        "total_files": total_files,
        "size_gb": round(size_gb, 2),
        "unique_cows": len(unique_cows),
    }


@app.local_entrypoint()
def main(hf_token: str = ""):
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token and os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass

    if token:
        print(f"✓ Automatically loaded HF_TOKEN from .env ({token[:8]}...)")
    else:
        print("Notice: No HF_TOKEN found in .env or arguments. Proceeding as guest.")

    download_mmcows.remote(hf_token=token)


@app.local_entrypoint()
def verify():
    verify_mmcows.remote()
