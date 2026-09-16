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
# 4. BCS — PRIMARY: ScienceDB, SECONDARY: Dryad
# -----------------------------------------------------------------------------
DRYAD_STREAM_URL = "https://datadryad.org/downloads/file_stream/2391628"

def restore_sciencedb(url=None):
    script = REPO_ROOT / "scripts" / "download_sciencedb.py"
    cmd = [sys.executable, str(script)]
    if url:
        cmd.extend(["--url", url])
    print(f"Executing ScienceDB downloader: {script.name}...")
    subprocess.check_call(cmd)

def restore_dryad():
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
        import time
        import webbrowser

        if not zip_path.exists() and not downloads_zip.exists():
            print("\nDryad BCS archive not found locally.")
            print(f"Opening direct download stream in your browser:")
            print(f"  {DRYAD_STREAM_URL}")
            print("\nDryad uses JavaScript Proof-of-Work (Anubis) bot protection,")
            print("so the download must be triggered through a real browser window.")
            webbrowser.open(DRYAD_STREAM_URL)
            print("Browser launched! Waiting for Total_sorted_DGE_images.zip in Downloads folder...")
            
            # Wait for download to appear and complete
            start_wait = time.time()
            crdownload_seen = False
            while time.time() - start_wait < 900:
                if downloads_zip.exists():
                    time.sleep(2)
                    break
                inprogress = list((Path.home() / "Downloads").glob("*Total_sorted_DGE*.crdownload")) + \
                             list((Path.home() / "Downloads").glob("*Total_sorted_DGE*.part"))
                if inprogress:
                    crdownload_seen = True
                    sys.stdout.write(f"\rDownloading in browser: {inprogress[0].stat().st_size / (1024**2):.1f} MB received...")
                    sys.stdout.flush()
                elif crdownload_seen:
                    time.sleep(2)
                    if downloads_zip.exists():
                        break
                time.sleep(2)
            print("")

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
            print("Download not completed or archive not found.")
            print(f"Please ensure Total_sorted_DGE_images.zip is in {bcs_target} or ~/Downloads/")
            return

    script = REPO_ROOT / "context" / "preprocess_bcs.py"
    if script.exists():
        print(f"Running Dryad BCS preprocessor: {script.name}...")
        subprocess.check_call([sys.executable, str(script)])

def restore_bcs(sciencedb_url=None):
    print("\n--- [PRIMARY BCS] Restoring ScienceDB Dataset ---")
    restore_sciencedb(sciencedb_url)
    print("\n--- [SECONDARY BCS] Restoring Dryad Dataset ---")
    restore_dryad()

# -----------------------------------------------------------------------------
# VERIFICATION
# -----------------------------------------------------------------------------
def verify_datasets():
    print("\n" + "=" * 60)
    print("  VERIFYING RESTORED DATASETS & CSV INDICES")
    print("=" * 60)
    indices = {
        "Behavior (MmCows)": DATASETS_DIR / "behavior" / "behavior_index.csv",
        "Cow ID (OpenCows)": DATASETS_DIR / "id" / "id_index.csv",
        "BCS (ScienceDB)": DATASETS_DIR / "bcs" / "sciencedb_bcs_index.csv",
        "BCS (Dryad)": DATASETS_DIR / "bcs" / "bcs_index.csv",
        "Lameness (Mendeley)": DATASETS_DIR / "lameness" / "lameness_index.csv",
    }
    for name, path in indices.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f) - 1
            print(f"  [OK] {name:<22}: {path.name} ({lines:,} samples indexed)")
        else:
            print(f"  [PENDING] {name:<22}: CSV not found at {path.name}")

def main():
    parser = argparse.ArgumentParser(description="Master dataset restoration for Cattle Health Monitoring")
    parser.add_argument("--all", action="store_true", help="Download and restore all active datasets")
    parser.add_argument("--task", choices=["bcs", "sciencedb", "dryad", "behavior", "id", "lameness"], help="Restore specific task dataset")
    parser.add_argument("--id-url", type=str, default=None, help="Direct download URL for OpenCows2020 if needed")
    parser.add_argument("--sciencedb-url", type=str, default=None, help="Direct download URL for ScienceDB dataset.rar")
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.print_help()
        return

    if args.all or args.task in ["bcs", "sciencedb"]:
        run_step("BCS PRIMARY (SCIENCEDB) RESTORATION", lambda: restore_sciencedb(args.sciencedb_url))

    if args.all or args.task in ["bcs", "dryad"]:
        run_step("BCS SECONDARY (DRYAD) RESTORATION", restore_dryad)

    if args.all or args.task == "behavior":
        run_step("BEHAVIOR (MMCOWS) RESTORATION", restore_behavior)

    if args.all or args.task == "id":
        run_step("COW ID (OPENCOWS2020) RESTORATION", lambda: restore_id(args.id_url))

    if args.task == "lameness":
        run_step("LAMENESS (HISTORICAL MENDELEY) RESTORATION", restore_lameness)

    verify_datasets()

if __name__ == "__main__":
    main()
