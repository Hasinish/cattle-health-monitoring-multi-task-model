# -*- coding: utf-8 -*-
"""
Modal Cloud Pipeline for CVB (Cattle Visual Behaviors) Dataset.

Handles:
  1. Ultra-fast cloud download of 226,344 CVB files (225,829 frames + 503 JSON annotations)
     into persistent Modal Volume `cvb-data` using aria2c (16 parallel streams).
  2. Live progress bar / periodic summary reporting to terminal.
  3. Periodic volume.commit() checkpoints every 60s so progress is resumable.
  4. Physical verification of downloaded frames, annotations, and disk usage.

Usage:
  # Download to Modal volume on profile tigerwood693:
  modal run --profile tigerwood693 scripts/modal_cvb_pipeline.py::download_cvb

  # Verify downloaded files anytime:
  modal run --profile tigerwood693 scripts/modal_cvb_pipeline.py::verify_cvb
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

# Persistent storage volume for CVB dataset
volume = modal.Volume.from_name("cvb-data", create_if_missing=True)
VOLUME_DIR = "/data"
CVB_DIR = "/data/cvb"

# Minimal container image with aria2 and imaging tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2")
    .pip_install("tqdm", "pillow")
)

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    GZ_LOCAL = REPO_ROOT / "58916v001.txt.gz"
    if GZ_LOCAL.exists():
        image = image.add_local_file(str(GZ_LOCAL), remote_path="/root/58916v001.txt.gz")

app = modal.App("cvb-dataset-pipeline", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=7200,  # 2 hours max
    cpu=1.0,       # Minimal container resources (AGENTS.md rule)
    memory=2048,   # 2 GB RAM (AGENTS.md rule)
)
def download_cvb():
    """Download CVB dataset via aria2c with live progress and periodic volume commits."""
    import gzip
    import shutil
    import subprocess
    import threading
    import time

    os.makedirs(CVB_DIR, exist_ok=True)
    input_txt = "/tmp/58916v001.txt"

    print("=" * 70)
    print("  CVB (CATTLE VISUAL BEHAVIORS) MODAL CLOUD DOWNLOADER")
    print("  Target Volume: cvb-data mounted at /data/cvb")
    print("  CPU: 1.0 | RAM: 2048 MB (Minimal Cost Tier)")
    print("=" * 70)

    # Decompress input file
    print("\n[1/3] Decompressing download link manifest...")
    t0 = time.time()
    with gzip.open("/root/58916v001.txt.gz", "rb") as f_in, open(input_txt, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"✓ Decompressed {os.path.getsize(input_txt) / (1024 * 1024):.2f} MB link manifest in {time.time() - t0:.1f}s")

    # Count existing files in CVB_DIR
    print("\n[2/3] Checking existing files in /data/cvb...")
    existing_files = sum(len(files) for _, _, files in os.walk(CVB_DIR))
    print(f"  Existing files in volume: {existing_files:,}")

    # Launch periodic volume commit & disk monitor thread
    stop_event = threading.Event()

    def checkpoint_worker():
        while not stop_event.wait(60.0):
            try:
                # Count files and size
                f_count = 0
                total_bytes = 0
                for root, _, files in os.walk(CVB_DIR):
                    f_count += len(files)
                    for f in files:
                        try:
                            total_bytes += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                size_gb = total_bytes / (1024 ** 3)
                volume.commit()
                pct = (f_count / 226344) * 100
                print(
                    f"\n>>> [CHECKPOINT] {f_count:,} / 226,344 files ({pct:.1f}%) | "
                    f"Disk: {size_gb:.2f} GB | Volume committed successfully! <<<\n",
                    flush=True
                )
            except Exception as e:
                print(f"[WARN] Checkpoint error: {e}", flush=True)

    monitor_thread = threading.Thread(target=checkpoint_worker, daemon=True)
    monitor_thread.start()

    # Launch aria2c
    print("\n[3/3] Starting aria2c with 64 parallel connections (Turbo Mode)...")
    print("      Streaming live progress summary below:\n")

    cmd = [
        "aria2c",
        "-x", "16",
        "-j", "64",
        "-s", "16",
        f"--input-file={input_txt}",
        f"--dir={CVB_DIR}",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--download-result=hide",
        "--summary-interval=5",
        "--console-log-level=error",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=CVB_DIR,  # Ensures relative paths in 58916v001.txt download inside /data/cvb!
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in proc.stdout:
            line_str = line.strip()
            if not line_str:
                continue
            # Filter out noisy Amazon presigned S3 URLs from terminal
            if line_str.startswith("FILE:") or "X-Amz-" in line_str or line_str.startswith("---") or line_str.startswith("http"):
                continue
            print(line_str, flush=True)
        proc.wait()
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)

    # Final volume commit
    print("\n[FINAL] Committing final volume state...")
    volume.commit()
    print("✓ Volume committed!")

    if proc.returncode == 0:
        print("\n🎉 [SUCCESS] All CVB files downloaded successfully!")
    else:
        print(f"\n[INFO] aria2c exited with code {proc.returncode}.")


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_cvb():
    """Verify downloaded CVB files, formats, resolutions, and disk usage."""
    import json
    from PIL import Image

    print("=" * 70)
    print("  CVB (CATTLE VISUAL BEHAVIORS) PHYSICAL VERIFICATION")
    print("  Volume Directory: /data/cvb")
    print("=" * 70)

    if not os.path.exists(CVB_DIR):
        print(f"[ERROR] Directory {CVB_DIR} does not exist in volume.")
        return

    total_files = 0
    total_bytes = 0
    jpg_files = []
    json_files = []

    print("\n[1/3] Scanning files in /data/cvb...")
    for root, _, files in os.walk(CVB_DIR):
        total_files += len(files)
        for f in files:
            p = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
            if f.endswith(".jpg"):
                jpg_files.append(p)
            elif f.endswith(".json"):
                json_files.append(p)

    size_gb = total_bytes / (1024 ** 3)
    print(f"✓ Total Files Found: {total_files:,}")
    print(f"  - JPEG Frames:    {len(jpg_files):,}")
    print(f"  - JSON Files:     {len(json_files):,}")
    print(f"  - Total Disk:     {size_gb:.2f} GB")

    # Sample-open 5 JPEG images
    print("\n[2/3] Sample-opening JPEG frames with PIL...")
    for p in jpg_files[:5]:
        try:
            with Image.open(p) as img:
                print(f"  ✓ {os.path.basename(p)}: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"  ✗ {p}: {e}")

    # Inspect 1 JSON annotation file
    print("\n[3/3] Inspecting sample JSON annotation...")
    if json_files:
        try:
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  File: {os.path.basename(json_files[0])}")
            print(f"  Keys: {list(data.keys())}")
            if "categories" in data:
                print(f"  Categories: {[c.get('name') for c in data['categories']]}")
            if "annotations" in data:
                print(f"  Total annotations in clip: {len(data['annotations'])}")
        except Exception as e:
            print(f"  ✗ Error reading JSON: {e}")

    print("\n[COMPLETE] CVB verification finished!")
