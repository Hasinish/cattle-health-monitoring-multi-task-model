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


def download_via_kagglehub():
    print("Downloading amibhavsar/open-cow-2020 via kagglehub...")
    import kagglehub
    path = Path(kagglehub.dataset_download("amibhavsar/open-cow-2020"))
    print(f"Kagglehub dataset path: {path}")
    
    # Locate identification/images
    id_img_root = None
    for cand in [
        path / "identification" / "images",
        path / "10m32xl88x2b61zlkkgz3fml17" / "identification" / "images",
        path / "images",
    ]:
        if cand.exists() and (cand / "train").exists():
            id_img_root = cand
            break
            
    if not id_img_root:
        for p in path.rglob("train"):
            if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
                id_img_root = p.parent
                break
                
    if not id_img_root:
        raise RuntimeError(f"Could not locate identification images in {path}")
        
    print(f"Found identification images at: {id_img_root}")
    train_src = id_img_root / "train"
    test_src = id_img_root / "test"
    
    train_dst = TARGET_DIR / "identification-train" / "img"
    test_dst = TARGET_DIR / "identification-test" / "img"
    train_dst.mkdir(parents=True, exist_ok=True)
    test_dst.mkdir(parents=True, exist_ok=True)
    
    def copy_split(src_dir, dst_dir, split_name):
        print(f"Copying and formatting {split_name} images to {dst_dir}...")
        count = 0
        for cow_folder in sorted(src_dir.iterdir()):
            if not cow_folder.is_dir():
                continue
            try:
                cow_id = str(int(cow_folder.name))
            except ValueError:
                continue
            for img in cow_folder.glob("*.jpg"):
                dst_file = dst_dir / f"{cow_id}_{img.name}"
                if not dst_file.exists():
                    shutil.copy2(img, dst_file)
                count += 1
        print(f"Formatted and copied {count} {split_name} images.")
        
    copy_split(train_src, train_dst, "train")
    copy_split(test_src, test_dst, "test")


DEFAULT_URL = "https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMTg3Nl9PcGVuQ293MjAyMC9vcGVuY293MjAyMC1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICI0WEZmblFEMDBPVm5yNWoyREVSbE1TMkxKeTFpVHg2bThUWHdDc2dSK1NnPSJ9?response-content-disposition=attachment%3B%20filename%3D%22opencow2020-DatasetNinja.tar%22"

def main():
    print("==================================================")
    print("  OPENCOWS2020 AUTOMATED DOWNLOAD & SETUP")
    print(f"  Target: {TARGET_DIR}")
    print("==================================================")

    # If already extracted and valid
    if (TARGET_DIR / "identification-train" / "img").exists() and len(list((TARGET_DIR / "identification-train" / "img").glob("*.jpg"))) > 100:
        print(f"Dataset already exists at {TARGET_DIR}!")
    else:
        # Check if URL was passed as command-line argument
        if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
            download_via_url(sys.argv[1])
        else:
            # Check local tar archives first
            local_tar = ID_DIR / "opencow2020-DatasetNinja.tar"
            downloads_tar = Path.home() / "Downloads" / "opencow2020-DatasetNinja.tar"
            
            if local_tar.exists():
                extract_archive(local_tar, ID_DIR)
            elif downloads_tar.exists():
                print(f"Found archive in Downloads: {downloads_tar}")
                extract_archive(downloads_tar, ID_DIR)
            else:
                # 1. Try Kagglehub first (fastest, most reliable)
                try:
                    download_via_kagglehub()
                except Exception as e:
                    print(f"Kagglehub download failed: {e}")
                    # 2. Fall back to direct S3 download
                    print("[FALLBACK] Switching to direct S3 download URL...")
                    try:
                        download_via_url(DEFAULT_URL)
                    except Exception as e2:
                        print(f"Direct S3 download failed: {e2}")
                        # 3. Fall back to dataset-tools
                        download_via_dataset_tools()

    # Verify target directory exists
    if not (TARGET_DIR / "identification-train" / "img").exists():
        raise RuntimeError(f"OpenCows2020 directory structure missing at {TARGET_DIR}")

    # Run preprocessor
    print("\nRunning preprocess_id.py to generate index CSV...")
    import subprocess
    prep_script = REPO_ROOT / "context" / "preprocess_id.py"
    subprocess.check_call([sys.executable, str(prep_script)])
    print("\n[SUCCESS] OpenCows2020 is fully downloaded, extracted, and indexed!")


if __name__ == "__main__":
    main()
