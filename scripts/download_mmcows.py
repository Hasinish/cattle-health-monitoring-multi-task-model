"""
Automated downloader and extractor for MmCows Behavior Dataset (cropped_bboxes.zip)
Source: Hugging Face (neis-lab/mmcows)
Target: datasets/behavior/mmcows/
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "datasets" / "behavior" / "mmcows"
PREPROCESS_SCRIPT = REPO_ROOT / "context" / "preprocess_mmcows_behavior.py"

TARGET_DIR.mkdir(parents=True, exist_ok=True)

def download_and_extract_mmcows():
    print("==================================================")
    print("  MMCOWS BEHAVIOR DATASET RESTORATION (HUGGING FACE)")
    print(f"  Target: {TARGET_DIR}")
    print("==================================================")
    
    # Check if already preprocessed
    csv_index = REPO_ROOT / "datasets" / "behavior" / "behavior_index.csv"
    if csv_index.exists() and csv_index.stat().st_size > 1000:
        print(f"[FOUND] behavior_index.csv already exists at {csv_index}")
        print("Dataset is already preprocessed and ready to train!")
        return

    # Check if huggingface_hub is available
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    zip_path = TARGET_DIR / "cropped_bboxes.zip"

    # Step 1: Download cropped_bboxes.zip (12.74 GB)
    if not zip_path.exists():
        print("Downloading cropped_bboxes.zip (~12.7 GB) from Hugging Face...")
        print("Repository: neis-lab/mmcows")
        downloaded_file = hf_hub_download(
            repo_id="neis-lab/mmcows",
            filename="cropped_bboxes.zip",
            repo_type="dataset",
            local_dir=str(TARGET_DIR),
        )
        print(f"Downloaded to: {downloaded_file}")
    else:
        print(f"[FOUND] Existing archive at {zip_path}")

    # Step 2: Extract archive
    print(f"\nExtracting {zip_path.name} into {TARGET_DIR}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Extraction complete!")
    if zip_path.exists():
        print("Cleaning up cropped_bboxes.zip archive to save disk space...")
        try:
            zip_path.unlink()
        except Exception:
            pass

    # Step 3: Run preprocessor
    print(f"\nRunning behavior preprocessor: {PREPROCESS_SCRIPT.name}...")
    subprocess.check_call([sys.executable, str(PREPROCESS_SCRIPT)])
    print("\n[SUCCESS] MmCows behavior dataset is fully restored and indexed!")

if __name__ == "__main__":
    download_and_extract_mmcows()
