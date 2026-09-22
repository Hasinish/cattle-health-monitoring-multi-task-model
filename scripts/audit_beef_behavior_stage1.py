# -*- coding: utf-8 -*-
"""
Fast Forensic Inspection of Kaggle Beef Behavior Dataset on Modal Volume beef-behavior-data.
Stage 1: Top-level inspection, inode analysis, directory structure, and archive status.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
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

volume = modal.Volume.from_name("beef-behavior-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/beef_behavior"

image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("beef-behavior-audit-stage1", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def inspect_volume_top_level():
    """Inspect top-level volume filesystem, directories, and archive state."""
    import subprocess

    print("=" * 80)
    print("  KAGGLE BEEF CATTLE BEHAVIOR DATASET: STAGE 1 FAST INSPECTION")
    print("  Volume Mount: /data")
    print("=" * 80)

    # 1. df -h and df -i
    print("\n--- [1] Disk & Inode Usage (df -h / df -i) ---")
    res = subprocess.run(["df", "-h", VOLUME_DIR], capture_output=True, text=True)
    print(res.stdout.strip())
    res_i = subprocess.run(["df", "-i", VOLUME_DIR], capture_output=True, text=True)
    print(res_i.stdout.strip())

    # 2. Check /data/ contents
    print("\n--- [2] Contents of /data/ ---")
    if os.path.exists(VOLUME_DIR):
        for item in sorted(os.listdir(VOLUME_DIR)):
            p = os.path.join(VOLUME_DIR, item)
            is_dir = os.path.isdir(p)
            sz = os.path.getsize(p) if not is_dir else 0
            print(f"  {'[DIR]' if is_dir else '[FILE]'} {item} ({sz / (1024*1024):.2f} MB)" if not is_dir else f"  {'[DIR]' if is_dir else '[FILE]'} {item}/")
    else:
        print("[ERROR] /data does not exist!")

    # 3. Check /data/beef_behavior/ contents
    print("\n--- [3] Contents of /data/beef_behavior/ ---")
    if os.path.exists(DATASET_DIR):
        for item in sorted(os.listdir(DATASET_DIR)):
            p = os.path.join(DATASET_DIR, item)
            is_dir = os.path.isdir(p)
            if is_dir:
                try:
                    sub_items = os.listdir(p)
                    print(f"  [DIR]  {item}/ ({len(sub_items):,} direct children)")
                    # show first 5 sub items
                    for s in sub_items[:5]:
                        sp = os.path.join(p, s)
                        s_dir = os.path.isdir(sp)
                        print(f"         - {'[DIR]' if s_dir else '[FILE]'} {s}")
                    if len(sub_items) > 5:
                        print(f"         ... and {len(sub_items) - 5} more items")
                except Exception as e:
                    print(f"  [DIR]  {item}/ (error listing: {e})")
            else:
                sz = os.path.getsize(p)
                print(f"  [FILE] {item} ({sz / (1024*1024):.2f} MB; {sz:,} bytes)")
    else:
        print("[ERROR] /data/beef_behavior does not exist!")

    # 4. Check for archive files
    archive_zip = os.path.join(DATASET_DIR, "archive.zip")
    archive_aria2 = os.path.join(DATASET_DIR, "archive.zip.aria2")
    print("\n--- [4] Archive & In-Flight Status ---")
    print(f"  archive.zip exists: {os.path.exists(archive_zip)}")
    if os.path.exists(archive_zip):
        sz_gb = os.path.getsize(archive_zip) / (1024**3)
        print(f"  archive.zip size: {sz_gb:.3f} GB ({os.path.getsize(archive_zip):,} bytes)")
    print(f"  archive.zip.aria2 exists: {os.path.exists(archive_aria2)}")
    if os.path.exists(archive_aria2):
        print(f"  [NOTE] Download was in progress or interrupted!")

    # 5. Fast summary of subdirectories
    print("\n--- [5] Deep Inspection of Extracted Directories (maxdepth 3) ---")
    cmd = ["find", DATASET_DIR, "-maxdepth", "3", "-type", "d"]
    res_d = subprocess.run(cmd, capture_output=True, text=True)
    dirs = res_d.stdout.strip().split("\n")
    print(f"Total directories (depth <= 3): {len(dirs)}")
    for d in dirs[:30]:
        if d:
            rel = os.path.relpath(d, DATASET_DIR)
            print(f"  {rel}/")
    if len(dirs) > 30:
        print(f"  ... and {len(dirs) - 30} more directories")


@app.local_entrypoint()
def main():
    inspect_volume_top_level.remote()


if __name__ == "__main__":
    main()
