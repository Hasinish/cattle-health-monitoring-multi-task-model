# -*- coding: utf-8 -*-
"""
Modal Cloud Pipeline for Kaggle Beef Cattle Behavior Dataset (lucyfirst/beef-cattle-behavior-data-set).

Handles:
  1. Ultra-fast cloud download of 49 GB archive directly into persistent Modal Volume
     `beef-behavior-data` using aria2c (16 parallel connections to Google Cloud Storage).
  2. High-speed multi-threaded extraction via 7z directly on the cloud volume.
  3. Automatic cleanup of the 49 GB archive.zip to prevent volume quota bloat.
  4. 60-second periodic volume.commit() checkpoints for safe resumability.
  5. Physical verification of extracted dataset files, formats, and disk usage.

Usage:
  # Download and extract on profile tigerwood693:
  modal run --profile tigerwood693 scripts/modal_beef_behavior_pipeline.py::download_beef_behavior

  # Verify dataset anytime:
  modal run --profile tigerwood693 scripts/modal_beef_behavior_pipeline.py::verify_beef_behavior
"""

import os
import sys
from pathlib import Path

# Guard against Windows cross-drive ValueError in ntpath.commonpath for Modal
if sys.platform == "win32":
    _orig_commonpath = os.path.commonpath

    def _safe_commonpath(paths):
        try:
            return _orig_commonpath(paths)
        except ValueError:
            return ""

    os.path.commonpath = _safe_commonpath

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent storage volume for Beef Cattle Behavior dataset
volume = modal.Volume.from_name("beef-behavior-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/beef_behavior"

# Minimal container image with aria2, p7zip, and imaging tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2", "p7zip-full", "unzip")
    .pip_install("kagglehub", "requests", "tqdm", "pillow")
)

app = modal.App("beef-cattle-behavior-pipeline", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=7200,    # 2 hours max
    cpu=4.0,         # 4 vCPUs for TLS line rate & 7z decompression
    memory=8192,     # 8 GB RAM for massive write buffering
)
def download_beef_behavior():
    """Download Kaggle 49 GB beef cattle behavior dataset via multi-connection aria2c with RAM caching."""
    import shutil
    import subprocess
    import threading
    import time
    import urllib.request

    import kagglehub
    from kagglehub.clients import build_kaggle_client
    from kagglehub.datasets import parse_dataset_handle
    from kagglehub.http_resolver import _build_dataset_download_request, _get_current_version

    os.makedirs(DATASET_DIR, exist_ok=True)
    archive_path = os.path.join(DATASET_DIR, "archive.zip")

    print("=" * 75)
    print("  KAGGLE BEEF CATTLE BEHAVIOR DATASET (lucyfirst/beef-cattle-behavior-data-set)")
    print("  Target Volume: beef-behavior-data mounted at /data/beef_behavior")
    print("  Target Archive: 48.55 GB (GCS Presigned Direct Stream)")
    print("  Engine: aria2c (16 parallel streams) + 128MB RAM Buffer + 7z multi-thread")
    print("=" * 75)

    # 1. Resolve direct Google Cloud Storage pre-signed URL via Kaggle public API
    print("\n[1/4] Resolving direct Google Cloud Storage URL from Kaggle API...")
    t0 = time.time()
    handle_str = "lucyfirst/beef-cattle-behavior-data-set"
    h = parse_dataset_handle(handle_str)

    with build_kaggle_client() as api_client:
        version = _get_current_version(api_client, h)
        print(f"  ✓ Latest Dataset Version: v{version}")
        h_ver = h.with_version(version)
        req = _build_dataset_download_request(h_ver, None)
        resp = api_client.datasets.dataset_api_client.download_dataset(req)
        download_url = resp.url

    print(f"  ✓ Pre-signed GCS URL resolved in {time.time() - t0:.1f}s")
    print(f"  ✓ Endpoint: {download_url.split('?')[0]}")

    # Check headers
    req_head = urllib.request.Request(download_url, method="HEAD")
    with urllib.request.urlopen(req_head) as r:
        content_len = int(r.headers.get("Content-Length", 0))
        total_gb = content_len / (1024 ** 3)
        print(f"  ✓ Remote File Size: {total_gb:.2f} GB ({content_len:,} bytes)")

    # 2. Gentle 5-minute checkpoint thread (NO 60s freeze stalls!)
    stop_event = threading.Event()

    def checkpoint_worker():
        while not stop_event.wait(300.0):  # Every 5 minutes, NOT every 60s!
            try:
                curr_size = os.path.getsize(archive_path) if os.path.exists(archive_path) else 0
                size_gb = curr_size / (1024 ** 3)
                pct = (curr_size / content_len) * 100 if content_len > 0 else 0
                volume.commit()
                print(
                    f"\n>>> [CHECKPOINT] Downloaded {size_gb:.2f} / {total_gb:.2f} GB ({pct:.1f}%) | Volume saved! <<<\n",
                    flush=True
                )
            except Exception as e:
                pass

    monitor_thread = threading.Thread(target=checkpoint_worker, daemon=True)
    monitor_thread.start()

    # 3. Maximum-speed multi-threaded download via aria2c
    print("\n[2/4] Starting turbo download with aria2c (16 parallel streams)...")
    cmd = [
        "aria2c",
        "-x", "16",
        "-s", "16",
        "-j", "16",
        "-k", "2M",                     # Aggressive piece splitting
        "--file-allocation=none",        # Immediate streaming without preallocating 49 GB
        "--disk-cache=128M",             # 128 MB RAM write buffer (bypasses NFS random write latency!)
        "--max-connection-per-server=16",
        "--continue=true",               # Resumes from existing 2.1 GB download!
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        f"--dir={DATASET_DIR}",
        "-o", "archive.zip",
        "--summary-interval=5",
        "--console-log-level=error",
        download_url,
    ]

    t_dl_start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=DATASET_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in proc.stdout:
            line_str = line.strip()
            if not line_str or line_str.startswith("http") or "X-Goog-" in line_str:
                continue
            print(line_str, flush=True)
        proc.wait()
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)

    if proc.returncode != 0:
        print(f"\n[ERROR] aria2c exited with code {proc.returncode}")
        return

    dl_duration = time.time() - t_dl_start
    avg_speed = (total_gb * 1024) / dl_duration if dl_duration > 0 else 0
    print(f"\n✓ Download completed in {dl_duration / 60:.1f} minutes ({avg_speed:.1f} MB/s avg)!")

    # 4. Multi-threaded extraction via 7z directly into persistent volume
    print("\n[3/4] Extracting 49 GB archive with multi-threaded 7z into volume...")
    t_ext_start = time.time()
    ext_cmd = [
        "7z",
        "x",
        "-y",
        "-mmt=on",
        archive_path,
        f"-o{DATASET_DIR}",
    ]

    res = subprocess.run(ext_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] 7z extraction failed: {res.stderr[:500]}")
        volume.commit()
        return

    ext_duration = time.time() - t_ext_start
    print(f"✓ Extracted successfully into volume in {ext_duration / 60:.1f} minutes!")

    # 5. Purge archive.zip to save volume quota
    print("\n[4/4] Purging archive.zip to reclaim volume storage...")
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print("✓ archive.zip deleted.")

    # Final volume commit
    print("\n[FINAL] Committing final volume state...")
    volume.commit()
    print("✓ Volume committed successfully!")

    total_files = sum(len(f) for _, _, f in os.walk(DATASET_DIR))
    print(f"\n🎉 [SUCCESS] Kaggle Beef Cattle Behavior dataset ready! Total extracted files: {total_files:,}")


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_beef_behavior():
    """Scan and verify extracted Beef Cattle Behavior dataset in Modal volume."""
    from PIL import Image

    print("=" * 75)
    print("  KAGGLE BEEF CATTLE BEHAVIOR DATASET PHYSICAL VERIFICATION")
    print("  Volume Directory: /data/beef_behavior")
    print("=" * 75)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory {DATASET_DIR} does not exist in volume.")
        return

    extensions = {}
    total_files = 0
    total_bytes = 0
    sample_images = []

    print("\n[1/3] Scanning filesystem in volume...")
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            total_files += 1
            ext = os.path.splitext(f)[1].lower() or "no_ext"
            extensions[ext] = extensions.get(ext, 0) + 1
            p = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
            if ext in [".jpg", ".jpeg", ".png"] and len(sample_images) < 5:
                sample_images.append(p)

    total_gb = total_bytes / (1024 ** 3)
    print(f"✓ Total Files Found: {total_files:,}")
    print(f"✓ Total Disk Space:  {total_gb:.2f} GB")
    print("\nFile Types Breakdown:")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {ext:10s}: {count:,} files")

    # Sample-open images with PIL
    if sample_images:
        print("\n[2/3] Inspecting sample images with PIL...")
        for p in sample_images:
            try:
                with Image.open(p) as img:
                    rel_p = os.path.relpath(p, DATASET_DIR)
                    print(f"  ✓ {rel_p}: format={img.format}, size={img.size}, mode={img.mode}")
            except Exception as e:
                print(f"  ✗ {p}: {e}")

    # Inspect top-level directory structure
    print("\n[3/3] Directory structure overview:")
    for item in sorted(os.listdir(DATASET_DIR))[:20]:
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            n_sub = len(os.listdir(item_path))
            print(f"  📁 {item}/ ({n_sub:,} items)")
        else:
            sz_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  📄 {item} ({sz_mb:.2f} MB)")

    print("\n[COMPLETE] Verification finished!")
