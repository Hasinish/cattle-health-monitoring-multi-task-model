"""
Downloader and Organizer for CBVD-5 (Cattle Behavior Video Dataset).

Dataset: fandaoerji/cbvd-5cow-behavior-video-dataset (Kaggle)
Paper: Li et al., CBVD (2024), Nature Scientific Reports
Scientific Role: Primary external behavior benchmark for larger-herd cross-domain evaluation.

Features:
- Downloads dataset via kagglehub API
- Syncs/copies files into canonical datasets/behavior/external/cbvd5/
- Inspects extracted directory layout, classes, video counts, and size
- Generates clean inventory summary
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = REPO_ROOT / "datasets" / "behavior" / "external" / "cbvd5"
KAGGLE_DATASET_HANDLE = "fandaoerji/cbvd-5cow-behavior-video-dataset"


def download_and_organize(dest_dir: Path, copy_to_target: bool = True):
    print("=" * 70)
    print("  CBVD-5 KAGGLE DATASET DOWNLOADER & ORGANIZER")
    print(f"  Dataset Handle:   {KAGGLE_DATASET_HANDLE}")
    print(f"  Target Directory: {dest_dir}")
    print("=" * 70)

    try:
        import kagglehub
    except ImportError:
        print("[ERROR] 'kagglehub' package is not installed.")
        print("Please install it with: pip install kagglehub")
        sys.exit(1)

    print(f"\n[1/3] Downloading latest version from Kaggle via kagglehub...")
    cache_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
    cache_dir = Path(cache_path)
    print(f"[SUCCESS] Downloaded to Kaggle cache: {cache_dir}")

    if not copy_to_target:
        print("\n[NOTE] --no-copy specified. Files remain in Kaggle cache.")
        return cache_dir

    print(f"\n[2/3] Syncing files to target directory: {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy files from cache to target
    copied_count = 0
    total_bytes = 0
    for src_item in cache_dir.rglob("*"):
        rel_path = src_item.relative_to(cache_dir)
        target_item = dest_dir / rel_path

        if src_item.is_dir():
            target_item.mkdir(parents=True, exist_ok=True)
        else:
            total_bytes += src_item.stat().st_size
            if not target_item.exists() or target_item.stat().st_size != src_item.stat().st_size:
                shutil.copy2(src_item, target_item)
                copied_count += 1

    print(f"[SUCCESS] Synced files to {dest_dir}")
    print(f"Total size: {total_bytes / (1024**2):.2f} MB ({total_bytes / (1024**3):.3f} GB)")

    print(f"\n[3/3] Inspecting dataset structure...")
    subdirs = [d for d in dest_dir.iterdir() if d.is_dir()]
    files = [f for f in dest_dir.iterdir() if f.is_file()]

    print(f"Top-level directories ({len(subdirs)}):")
    for d in sorted(subdirs):
        n_files = len(list(d.rglob("*.*")))
        print(f"  - {d.name}/ ({n_files} files)")

    print(f"Top-level files ({len(files)}):")
    for f in sorted(files):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 70)
    print("[COMPLETE] CBVD-5 dataset is ready for external behavior benchmarking!")
    print("=" * 70)
    return dest_dir


def main():
    parser = argparse.ArgumentParser(description="Download and organize CBVD-5 dataset from Kaggle")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_TARGET_DIR), help="Target destination directory")
    parser.add_argument("--no-copy", action="store_true", help="Keep downloaded files in Kaggle cache only")
    args = parser.parse_args()

    download_and_organize(Path(args.dest), copy_to_target=not args.no_copy)


if __name__ == "__main__":
    main()
