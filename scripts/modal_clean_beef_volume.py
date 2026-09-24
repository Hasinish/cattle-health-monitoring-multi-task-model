# -*- coding: utf-8 -*-
"""Clean Kaggle Beef volume on hasinishrak2015, preserving only what is required for training.

1. Verifies all 4,337 MP4 video clips in /data/beef_behavior/Category Videos/cows/.
2. Completely wipes the bloated /data/beef_behavior/Labelframes/ directory to reclaim 495,000+ inodes.
3. Purges redundant 45.22 GB archive.zip (already backed up on dryousufmozumder) to save volume storage.
4. Commits the volume and prints updated disk & inode health.

Usage:
    modal run --profile hasinishrak2015 scripts/modal_clean_beef_volume.py::clean_and_verify
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
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

image = modal.Image.debian_slim(python_version="3.11").apt_install("procps")

app = modal.App("clean-beef-volume-hasinishrak", image=image)


@app.function(
    volumes={"/data": volume},
    timeout=1800,  # 30 mins
    cpu=4.0,
    memory=8192,
)
def clean_and_verify_remote() -> Dict[str, Any]:
    print("=" * 75)
    print("  KAGGLE BEEF VOLUME CLEANUP & TRAINING DATA CERTIFICATION (hasinishrak2015)")
    print("=" * 75)

    beef_dir = Path("/data/beef_behavior")
    cat_dir = beef_dir / "Category Videos" / "cows"
    lf_dir = beef_dir / "Labelframes"
    archive_path = beef_dir / "archive.zip"

    # 1. Audit Category Videos (The actual training dataset)
    print("\n[1/4] Auditing training videos in Category Videos/cows/ ...")
    if not cat_dir.exists():
        raise FileNotFoundError(f"Missing {cat_dir}! Cannot proceed with cleanup.")

    mp4_counts: Dict[str, int] = {}
    total_bytes = 0
    total_mp4s = 0

    for cls_dir in sorted(cat_dir.iterdir()):
        if cls_dir.is_dir():
            clips = list(cls_dir.glob("*.mp4"))
            mp4_counts[cls_dir.name] = len(clips)
            total_mp4s += len(clips)
            for c in clips:
                total_bytes += c.stat().st_size

    print(f"  ✓ Total Training MP4 Clips: {total_mp4s:,} ({total_bytes / (1024**3):.2f} GB)")
    for cls_name, count in mp4_counts.items():
        print(f"    - {cls_name}: {count:,} clips")

    if total_mp4s < 4300:
        raise AssertionError(f"Expected ~4,337 clips, found only {total_mp4s}! Aborting.")

    # 2. Wipe redundant Labelframes directory to reclaim inodes
    print("\n[2/4] Wiping redundant Labelframes/ to reclaim 495,000+ inodes...")
    t0 = time.time()
    if lf_dir.exists():
        shutil.rmtree(lf_dir)
        print(f"  ✓ Labelframes/ deleted in {time.time() - t0:.1f}s!")
    else:
        print("  - Labelframes/ does not exist (already clean).")

    # 3. Purge redundant archive.zip to reclaim 45 GB volume storage
    print("\n[3/4] Purging redundant archive.zip...")
    if archive_path.exists():
        archive_gb = archive_path.stat().st_size / (1024**3)
        archive_path.unlink()
        print(f"  ✓ archive.zip ({archive_gb:.2f} GB) deleted to reclaim storage.")
    else:
        print("  - archive.zip does not exist (already purged).")

    # 4. Commit volume and report final health
    print("\n[4/4] Committing volume state to Modal...")
    volume.commit()
    print("  ✓ Volume committed successfully!")

    res_df = subprocess.run(["df", "-h", "/data"], capture_output=True, text=True)
    res_dfi = subprocess.run(["df", "-i", "/data"], capture_output=True, text=True)

    print(f"\n[FINAL DISK HEALTH]:\n{res_df.stdout}")
    print(f"[FINAL INODE HEALTH]:\n{res_dfi.stdout}")

    return {
        "status": "SUCCESS",
        "total_mp4_clips": total_mp4s,
        "classes": mp4_counts,
        "video_dataset_gb": round(total_bytes / (1024**3), 2),
        "disk_usage": res_df.stdout.strip(),
        "inode_usage": res_dfi.stdout.strip(),
    }


@app.local_entrypoint()
def clean_and_verify():
    res = clean_and_verify_remote.remote()
    print("\n[LOCAL] Cleanup & Verification Succeeded!")
    print(f"  - Total Training MP4s: {res['total_mp4_clips']:,}")
    print(f"  - Dataset Size:        {res['video_dataset_gb']} GB")
    print(f"  - Classes:             {res['classes']}")
