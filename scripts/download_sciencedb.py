"""
Automated downloader and extractor for ScienceDB Cattle BCS Dataset (Primary BCS Task)
URL: https://china.scidb.cn/download?fileId=4516a6866954512e08446bc3504b32c9&traceId=ae1cb76e-9f3b-48b0-9c51-2750c6266d25
Dataset: 53,566 RGB rear-view images across 5 classes (3.25, 3.5, 3.75, 4.0, 4.25)
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import urllib.request
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
BCS_DIR = DATASETS_DIR / "bcs"
SCIENCEDB_DIR = BCS_DIR / "sciencedb_bcs"
ARCHIVE_PATH = SCIENCEDB_DIR / "dataset.rar"
EXPECTED_DATASET_DIR = SCIENCEDB_DIR / "dataset"

DEFAULT_URL = (
    "https://china.scidb.cn/download?fileId=4516a6866954512e08446bc3504b32c9&traceId=ae1cb76e-9f3b-48b0-9c51-2750c6266d25"
)

UNRAR_CANDIDATES = [
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files\WinRAR\WinRAR.exe",
    r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "unrar",
    "7z",
    "tar",
]


def find_extractor():
    for candidate in UNRAR_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        elif shutil.which(candidate):
            return candidate
    return None


def download_file(url, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    }
    
    existing_bytes = 0
    if temp_path.exists():
        existing_bytes = temp_path.stat().st_size
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"[RESUME] Found partial download of {existing_bytes / (1024**2):.2f} MB. Attempting to resume...")

    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            total_size = response.headers.get("Content-Length")
            
            if status == 206:  # Partial Content
                total_bytes = existing_bytes + int(total_size)
                mode = "ab"
            elif status == 200:
                total_bytes = int(total_size) if total_size else None
                existing_bytes = 0
                mode = "wb"
            else:
                mode = "wb"
                total_bytes = int(total_size) if total_size else None

            print(f"Downloading {target_path.name} ({total_bytes / (1024**3):.2f} GB)..." if total_bytes else f"Downloading {target_path.name}...")
            
            with open(temp_path, mode) as out_file:
                with tqdm(
                    total=total_bytes,
                    initial=existing_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=target_path.name,
                ) as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1 MB chunks
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))

        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)
        print(f"[SUCCESS] Download completed: {target_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def extract_rar(archive_path, extract_dir):
    extractor = find_extractor()
    if not extractor:
        print("[ERROR] No RAR extractor found! Please install WinRAR or 7-Zip.")
        print(f"Archive is saved at: {archive_path}")
        return False

    print(f"Extracting {archive_path.name} using: {extractor}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = []
    if "unrar.exe" in extractor.lower():
        cmd = [extractor, "x", "-y", "-o+", str(archive_path), str(extract_dir)]
    elif "winrar.exe" in extractor.lower():
        cmd = [extractor, "x", "-y", "-ibck", str(archive_path), str(extract_dir)]
    elif "7z" in extractor.lower():
        cmd = [extractor, "x", str(archive_path), f"-o{extract_dir}", "-y"]
    elif "tar" in extractor.lower():
        cmd = [extractor, "-xf", str(archive_path), "-C", str(extract_dir)]
    else:
        cmd = [extractor, "x", str(archive_path), str(extract_dir)]

    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Extraction exited with code {res.returncode}")
        return False
        
    print("[SUCCESS] Extraction completed successfully!")
    return True


def verify_and_restructure():
    # Verify expected classes
    expected_classes = ['3.25', '3.5', '3.75', '4.0', '4.25']
    
    # Check if dataset/ is directly inside sciencedb_bcs
    if not EXPECTED_DATASET_DIR.exists():
        # Check if classes are directly in sciencedb_bcs
        found_in_root = all((SCIENCEDB_DIR / cls_name).exists() for cls_name in expected_classes)
        if found_in_root:
            print("[INFO] Restructuring class folders into 'dataset/' subdirectory...")
            EXPECTED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
            for cls_name in expected_classes:
                src = SCIENCEDB_DIR / cls_name
                dst = EXPECTED_DATASET_DIR / cls_name
                if src.exists():
                    shutil.move(str(src), str(dst))

    # Verify counts
    total_imgs = 0
    for cls_name in expected_classes:
        cls_dir = EXPECTED_DATASET_DIR / cls_name
        if cls_dir.exists():
            cnt = len(list(cls_dir.glob("*.jpg")))
            print(f"  Class {cls_name}: {cnt} images")
            total_imgs += cnt
        else:
            print(f"  [MISSING] Class {cls_name} not found in {EXPECTED_DATASET_DIR}")

    print(f"[INFO] Total ScienceDB images found: {total_imgs}")
    return total_imgs > 1000


def run_preprocessor():
    script = REPO_ROOT / "context" / "preprocess_sciencedb_bcs.py"
    if script.exists():
        print(f"\nRunning ScienceDB preprocessor: {script.name}...")
        subprocess.check_call([sys.executable, str(script)])
        print("[SUCCESS] ScienceDB index generated!")


def main():
    parser = argparse.ArgumentParser(description="Download and extract ScienceDB Cattle BCS dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="Direct download URL for ScienceDB dataset.rar")
    parser.add_argument("--extract-only", action="store_true", help="Skip download and extract existing dataset.rar")
    parser.add_argument("--clean-archive", action="store_true", help="Delete dataset.rar after successful extraction")
    args = parser.parse_args()

    print("=" * 60)
    print("  ScienceDB Cattle BCS Dataset Downloader & Restorer")
    print("=" * 60)

    # Step 1: Download
    if not args.extract_only:
        if not ARCHIVE_PATH.exists():
            success = download_file(args.url, ARCHIVE_PATH)
            if not success:
                print("\n[ERROR] Download could not complete. You can rerun this script to resume.")
                sys.exit(1)
        else:
            print(f"[FOUND] Archive already exists at {ARCHIVE_PATH}")

    # Step 2: Extract
    if ARCHIVE_PATH.exists():
        success = extract_rar(ARCHIVE_PATH, SCIENCEDB_DIR)
        if not success:
            sys.exit(1)

    # Step 3: Verify and restructure
    ok = verify_and_restructure()
    if not ok:
        print("[WARNING] Could not verify all 5 class folders under dataset/. Please inspect manually.")
    
    # Step 4: Run preprocessor to generate CSV
    run_preprocessor()

    # Optional archive cleanup
    if args.clean_archive and ARCHIVE_PATH.exists():
        print(f"Cleaning up {ARCHIVE_PATH.name} to reclaim disk space...")
        ARCHIVE_PATH.unlink()

    print("\n" + "=" * 60)
    print("  ScienceDB BCS Dataset is Ready for Phase 3 Training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
