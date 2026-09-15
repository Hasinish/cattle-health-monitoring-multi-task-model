"""
Automated downloader and extractor for OpenCows2020 dataset
Supports official dataset_tools API and direct URL download.
"""

import os
import sys
import tarfile
import zipfile
import shutil
import urllib.request
from pathlib import Path

# Paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
ID_DIR = DATASETS_DIR / "id"
TARGET_DIR = ID_DIR / "opencow2020-DatasetNinja"

ID_DIR.mkdir(parents=True, exist_ok=True)


def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path.name} to {extract_to}...")
    if archive_path.suffix in ['.tar', '.gz', '.tgz'] or '.tar' in archive_path.name:
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(path=extract_to)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    print("Extraction complete!")


def download_via_dataset_tools():
    print("Attempting download using official dataset_tools...")
    try:
        import dataset_tools as dtools
    except ImportError:
        print("Installing dataset-tools...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "dataset-tools"])
        import dataset_tools as dtools

    print("Downloading OpenCow2020 via dataset-tools API...")
    dtools.download(dataset="OpenCow2020", dst_dir=str(ID_DIR))
    print("Download finished via dataset_tools!")

    # Normalize folder name if needed
    for possible in [ID_DIR / "OpenCow2020", ID_DIR / "opencow2020", ID_DIR / "OpenCows2020"]:
        if possible.exists() and not TARGET_DIR.exists():
            print(f"Standardizing folder name: {possible.name} -> {TARGET_DIR.name}...")
            possible.rename(TARGET_DIR)


def download_via_url(url):
    print("Downloading via direct URL...")
    archive_path = ID_DIR / "opencow2020-DatasetNinja.tar"
    
    # Download with progress bar
    from tqdm import tqdm
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="OpenCows2020") as t:
        urllib.request.urlretrieve(url, filename=str(archive_path), reporthook=t.update_to)
    
    print("Download complete! Extracting...")
    extract_archive(archive_path, ID_DIR)


def main():
    print("==================================================")
    print("  OPENCOWS2020 AUTOMATED DOWNLOAD & SETUP")
    print(f"  Target: {TARGET_DIR}")
    print("==================================================")

    # If already extracted and valid
    if (TARGET_DIR / "identification-train" / "img").exists():
        print(f"Dataset already exists at {TARGET_DIR}!")
    else:
        # Check if URL was passed as command-line argument
        if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
            download_via_url(sys.argv[1])
        else:
            # Check if tar exists locally in datasets/id or Downloads
            local_tar = ID_DIR / "opencow2020-DatasetNinja.tar"
            downloads_tar = Path.home() / "Downloads" / "opencow2020-DatasetNinja.tar"
            
            if local_tar.exists():
                extract_archive(local_tar, ID_DIR)
            elif downloads_tar.exists():
                print(f"Found archive in Downloads: {downloads_tar}")
                extract_archive(downloads_tar, ID_DIR)
            else:
                try:
                    download_via_dataset_tools()
                except Exception as e:
                    print(f"dataset-tools download failed: {e}")
                    print("\nRun this script with the direct URL:")
                    print("  python scripts/download_opencows.py \"<direct_url>\"")
                    return

    # Run preprocessor
    print("\nRunning preprocess_id.py to generate index CSV...")
    import subprocess
    prep_script = REPO_ROOT / "context" / "preprocess_id.py"
    subprocess.check_call([sys.executable, str(prep_script)])
    print("\n[SUCCESS] OpenCows2020 is fully downloaded, extracted, and indexed!")


if __name__ == "__main__":
    main()
