"""
MASTER DATASET RESTORATION AND PREPROCESSING PIPELINE
One single script to download, extract, and index all datasets directly into datasets/

Usage:
  python scripts/download_all.py --all
  python scripts/download_all.py --task lameness
  python scripts/download_all.py --task behavior
  python scripts/download_all.py --task id
  python scripts/download_all.py --task bcs
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"

def run_step(title, func):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    try:
        func()
        print(f"[SUCCESS] {title} completed successfully!")
    except Exception as e:
        print(f"[ERROR] {title} failed: {e}")

# -----------------------------------------------------------------------------
# 1. LAMENESS (Mendeley CattleLameness)
# -----------------------------------------------------------------------------
def restore_lameness():
    lame_dir = DATASETS_DIR / "lameness" / "CattleLameness"
    lame_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if not (lame_dir / ".git").exists():
        print(f"Cloning CattleLameness to {lame_dir}...")
        subprocess.check_call(["git", "clone", "https://github.com/fahimsohan/CattleLameness", str(lame_dir)])
    else:
        print(f"[FOUND] CattleLameness repository already exists at {lame_dir}")
        
    script = REPO_ROOT / "context" / "preprocess_lameness.py"
    print(f"Extracting video frames and generating CSV index via {script.name}...")
    subprocess.check_call([sys.executable, str(script)])

# -----------------------------------------------------------------------------
# 2. BEHAVIOR (MmCows cropped_bboxes)
# -----------------------------------------------------------------------------
def restore_behavior():
    script = REPO_ROOT / "scripts" / "download_mmcows.py"
    print(f"Executing MmCows downloader: {script.name}...")
    subprocess.check_call([sys.executable, str(script)])

# -----------------------------------------------------------------------------
# 3. COW ID (OpenCows2020)
# -----------------------------------------------------------------------------
def restore_id(url=None):
    script = REPO_ROOT / "scripts" / "download_opencows.py"
    cmd = [sys.executable, str(script)]
    if url:
        cmd.append(url)
    print(f"Executing OpenCows downloader: {script.name}...")
    subprocess.check_call(cmd)

# -----------------------------------------------------------------------------
# 4. BCS (Dryad BCS)
# -----------------------------------------------------------------------------
def restore_bcs():
    bcs_target = DATASETS_DIR / "bcs" / "dryad_bcs"
    bcs_target.mkdir(parents=True, exist_ok=True)
    zip_path = bcs_target / "Total_sorted_DGE_images.zip"
    downloads_zip = Path.home() / "Downloads" / "Total_sorted_DGE_images.zip"
    
    # Check if images already extracted
    dge_images = list(bcs_target.glob("*.jpg")) + list(bcs_target.glob("*.png")) + list((bcs_target / "Total_sorted_DGE_images").glob("*.jpg"))
    if len(dge_images) > 1000:
        print(f"[FOUND] Dryad BCS images already extracted ({len(dge_images)} images found).")
    else:
        import zipfile
        if zip_path.exists():
            print(f"Extracting {zip_path.name} to {bcs_target}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(bcs_target)
            print("Extracted successfully!")
        elif downloads_zip.exists():
            print(f"Found archive in Downloads: {downloads_zip}")
            print(f"Extracting to {bcs_target}...")
            with zipfile.ZipFile(downloads_zip, 'r') as zf:
                zf.extractall(bcs_target)
            print("Extracted successfully!")
        else:
            print("Dryad BCS archive not found.")
            print("Note: Place Total_sorted_DGE_images.zip or extract DGE images into:")
            print(f"  {bcs_target}")
            print("Or keep it in your Downloads folder: ~/Downloads/Total_sorted_DGE_images.zip")

    script = REPO_ROOT / "context" / "preprocess_bcs.py"
    if script.exists():
        print(f"Running BCS preprocessor: {script.name}...")
        subprocess.check_call([sys.executable, str(script)])

# -----------------------------------------------------------------------------
# VERIFICATION
# -----------------------------------------------------------------------------
def verify_datasets():
    print("\n" + "=" * 60)
    print("  VERIFYING RESTORED DATASETS & CSV INDICES")
    print("=" * 60)
    indices = {
        "Lameness": DATASETS_DIR / "lameness" / "lameness_index.csv",
        "Behavior": DATASETS_DIR / "behavior" / "behavior_index.csv",
        "Cow ID": DATASETS_DIR / "id" / "id_index.csv",
        "BCS": DATASETS_DIR / "bcs" / "bcs_index.csv",
    }
    for name, path in indices.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f) - 1
            print(f"  [OK] {name:<10}: {path} ({lines:,} samples indexed)")
        else:
            print(f"  [PENDING] {name:<10}: CSV not found at {path}")

def main():
    parser = argparse.ArgumentParser(description="Master dataset restoration for Cattle Health Monitoring")
    parser.add_argument("--all", action="store_true", help="Download and restore all datasets")
    parser.add_argument("--task", choices=["lameness", "behavior", "id", "bcs"], help="Restore specific task dataset")
    parser.add_argument("--id-url", type=str, default=None, help="Direct download URL for OpenCows2020 if needed")
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.print_help()
        return

    if args.all or args.task == "lameness":
        run_step("LAMENESS DATASET RESTORATION", restore_lameness)

    if args.all or args.task == "behavior":
        run_step("BEHAVIOR (MMCOWS) DATASET RESTORATION", restore_behavior)

    if args.all or args.task == "id":
        run_step("COW ID (OPENCOWS2020) RESTORATION", lambda: restore_id(args.id_url))

    if args.all or args.task == "bcs":
        run_step("BCS DATASET RESTORATION", restore_bcs)

    verify_datasets()

if __name__ == "__main__":
    main()
