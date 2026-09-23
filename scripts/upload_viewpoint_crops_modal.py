# -*- coding: utf-8 -*-
"""
upload_viewpoint_crops_modal.py — Upload RT-DETR Cropped Viewpoint Dataset to Modal Volume

Uploads:
  Source: datasets/viewpoint/self_clean_v1_rtdetr_crop/ (879 crops + metadata + splits)
  Target: Modal Volume 'viewpoint-real-data'
  Target Layout:
    /data/self_clean_v1_rtdetr_crop/
        front/
        side/
        rear/
        metadata/
        splits/

Features:
  - Resumable: Scans remote volume first; skips already-uploaded files with matching byte size.
  - Live progress: tqdm terminal progress displaying files completed, MB transferred, speed, ETA, and current class.
  - Chunked batch upload: Uploads in batches of 25 files per transaction.
  - Remote verification: Executes zero-GPU minimal CPU (1.0 CPU, 1024MB RAM) verification function verifying:
      * front = 392
      * side = 222
      * rear = 265
      * total = 879
      * 0 zero-byte files
      * random PIL image decodes across all 3 classes
      * metadata and splits verified

Usage:
  python scripts/upload_viewpoint_crops_modal.py
  python scripts/upload_viewpoint_crops_modal.py --verify-only
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from tqdm import tqdm

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_DATASET_DIR = ROOT_DIR / "datasets" / "viewpoint" / "self_clean_v1_rtdetr_crop"
VOLUME_NAME = "viewpoint-real-data"
REMOTE_PREFIX = "/self_clean_v1_rtdetr_crop"

BATCH_SIZE = 25  # files per volume transaction batch

# Modal container image for cheap CPU verification
verify_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pillow", "pandas", "tqdm")
)
app = modal.App("viewpoint-dataset-uploader")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload and verify viewpoint crops on Modal.")
    parser.add_argument("--verify-only", action="store_true", help="Only run remote verification without uploading.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Files per batch upload transaction.")
    return parser.parse_args()


def get_remote_file_map(vol: modal.Volume) -> Dict[str, int]:
    """Scan existing remote files and sizes on the volume."""
    print("Scanning remote volume for existing files to enable resumability...", flush=True)
    remote_map: Dict[str, int] = {}
    try:
        entries = vol.listdir(REMOTE_PREFIX, recursive=True)
        for e in entries:
            # FileEntry type: FILE is 1, DIRECTORY is 2
            if getattr(e.type, "value", e.type) == 1:
                norm_p = "/" + e.path.replace("\\", "/").lstrip("/")
                remote_map[norm_p] = e.size
        print(f"✓ Found {len(remote_map)} existing files on remote volume.", flush=True)
    except Exception as e:
        print(f"[*] Remote directory empty or not yet created: {e}", flush=True)
    return remote_map


def collect_local_files() -> List[Tuple[Path, str, int, str]]:
    """
    Collect all files to upload: (local_path, remote_path, size_bytes, category)
    """
    files_to_upload = []

    # 1. Image files (front, side, rear)
    for cls_name in ["front", "side", "rear"]:
        cls_dir = LOCAL_DATASET_DIR / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            sz = img_path.stat().st_size
            rem_p = f"{REMOTE_PREFIX}/{cls_name}/{img_path.name}"
            files_to_upload.append((img_path, rem_p, sz, cls_name))

    # 2. Metadata files
    meta_dir = LOCAL_DATASET_DIR / "metadata"
    if meta_dir.exists():
        for mf in sorted(meta_dir.glob("*.*")):
            sz = mf.stat().st_size
            rem_p = f"{REMOTE_PREFIX}/metadata/{mf.name}"
            files_to_upload.append((mf, rem_p, sz, "metadata"))

    # 3. Splits files
    splits_dir = LOCAL_DATASET_DIR / "splits"
    if splits_dir.exists():
        for sf in sorted(splits_dir.glob("*.*")):
            sz = sf.stat().st_size
            rem_p = f"{REMOTE_PREFIX}/splits/{sf.name}"
            files_to_upload.append((sf, rem_p, sz, "splits"))

    # Also root split files if present
    for rf_name in ["train.csv", "val.csv", "test.csv", "split_report.md"]:
        rf = LOCAL_DATASET_DIR / rf_name
        if rf.exists():
            sz = rf.stat().st_size
            rem_p = f"{REMOTE_PREFIX}/{rf_name}"
            files_to_upload.append((rf, rem_p, sz, "root_splits"))

    return files_to_upload


@app.function(
    volumes={"/data": volume},
    cpu=1.0,
    memory=1024,
    timeout=300,
    image=verify_image,
)
def verify_remote_dataset():
    """
    Minimal CPU-only verification directly inside Modal volume.
    Asserts exact counts, zero byte files, PIL readability, and metadata presence.
    """
    import random
    from PIL import Image
    import pandas as pd

    print("\n" + "=" * 70, flush=True)
    print("  MODAL CLOUD REAL VIEWPOINT DATASET AUDIT & VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    root = Path("/data/self_clean_v1_rtdetr_crop")
    assert root.exists(), f"Remote root directory not found: {root}"

    # 1. Verify class directories and counts
    counts = {}
    expected_counts = {"front": 392, "side": 222, "rear": 265}
    zero_byte_files = []

    for cls_name, expected in expected_counts.items():
        cls_dir = root / cls_name
        assert cls_dir.exists(), f"Class directory missing: {cls_dir}"
        img_files = list(cls_dir.glob("*.jpg"))
        counts[cls_name] = len(img_files)

        for f in img_files:
            if f.stat().st_size == 0:
                zero_byte_files.append(str(f))

        print(f"[*] Class '{cls_name:5s}': {counts[cls_name]} files (expected {expected})", flush=True)
        assert counts[cls_name] == expected, f"Count mismatch for '{cls_name}': expected {expected}, got {counts[cls_name]}"

    total_images = sum(counts.values())
    print(f"\n[*] Total verified cropped images: {total_images} / 879", flush=True)
    assert total_images == 879, f"Total image count mismatch: expected 879, got {total_images}"

    print(f"[*] Zero-byte check: {len(zero_byte_files)} zero-byte files detected.", flush=True)
    assert len(zero_byte_files) == 0, f"Found zero-byte files: {zero_byte_files[:5]}"

    # 2. Random PIL decode verification across all classes
    print("\n[*] Executing random PIL decode verification (5 per class)...", flush=True)
    random.seed(2026)
    for cls_name in expected_counts:
        img_files = list((root / cls_name).glob("*.jpg"))
        sampled = random.sample(img_files, 5)
        for s in sampled:
            with Image.open(s) as img:
                w, h = img.size
                assert w > 0 and h > 0, f"Corrupted image dimensions for {s}"
                assert img.mode in ("RGB", "L"), f"Unexpected mode {img.mode} for {s}"
        print(f"  ✓ {cls_name:5s}: 5/5 random samples successfully decoded with PIL.", flush=True)

    # 3. Verify metadata files
    print("\n[*] Verifying metadata & split manifests...", flush=True)
    meta_dir = root / "metadata"
    assert (meta_dir / "crop_manifest.csv").exists(), "crop_manifest.csv missing"
    assert (meta_dir / "detection_failures.csv").exists(), "detection_failures.csv missing"
    assert (meta_dir / "crop_report.md").exists(), "crop_report.md missing"

    # Verify split files
    for split_name in ["train.csv", "val.csv", "test.csv"]:
        sp = root / split_name
        if not sp.exists():
            sp = root / "splits" / split_name
        assert sp.exists(), f"Split file missing: {split_name}"
        df_sp = pd.read_csv(sp)
        print(f"  ✓ {split_name:10s}: {len(df_sp)} rows verified.", flush=True)

    volume.commit()
    print("\n" + "=" * 70, flush=True)
    print("  ✓ ALL REMOTE VERIFICATION CHECKS PASSED WITH 100% SUCCESS!", flush=True)
    print("=" * 70, flush=True)
    return {
        "status": "PASSED",
        "counts": counts,
        "total_images": total_images,
        "zero_byte_count": len(zero_byte_files),
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("  MODAL VIEWPOINT CROPPED DATASET UPLOADER")
    print(f"  Local Source:  {LOCAL_DATASET_DIR}")
    print(f"  Modal Volume:  {VOLUME_NAME}")
    print(f"  Remote Prefix: {REMOTE_PREFIX}")
    print(f"  Batch Size:    {args.batch_size} files/batch")
    print("=" * 80)

    if not args.verify_only:
        # Collect all local files
        all_files = collect_local_files()
        total_files = len(all_files)
        total_bytes = sum(sz for _, _, sz, _ in all_files)
        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_bytes / (1024 * 1024 * 1024)

        print(f"\n[1/4] Found {total_files} local files ({total_mb:.2f} MB / {total_gb:.3f} GB).")

        # Scan remote volume for resumability
        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        remote_map = get_remote_file_map(vol)

        pending_files = []
        skipped_count = 0
        skipped_bytes = 0

        for local_p, rem_p, sz, cat in all_files:
            if rem_p in remote_map and remote_map[rem_p] == sz:
                skipped_count += 1
                skipped_bytes += sz
            else:
                pending_files.append((local_p, rem_p, sz, cat))

        if skipped_count > 0:
            print(f"✓ Resuming upload: {skipped_count}/{total_files} files already on volume ({skipped_bytes / (1024*1024):.2f} MB skipped).")
        print(f"[*] Uploading remaining {len(pending_files)} files ({sum(sz for _, _, sz, _ in pending_files) / (1024*1024):.2f} MB)...\n")

        if pending_files:
            # Batch upload with live tqdm progress
            transferred_bytes = skipped_bytes
            uploaded_count = skipped_count
            start_time = time.time()

            pbar = tqdm(
                total=total_files,
                initial=skipped_count,
                desc="Uploading crops",
                unit="files",
                file=sys.stdout,
                dynamic_ncols=True,
            )

            # Chunk into batches
            batch_size = args.batch_size
            for b_idx in range(0, len(pending_files), batch_size):
                chunk = pending_files[b_idx : b_idx + batch_size]
                chunk_bytes = sum(sz for _, _, sz, _ in chunk)

                t_batch_start = time.time()
                with vol.batch_upload(force=True) as batch:
                    for local_p, rem_p, sz, cat in chunk:
                        batch.put_file(str(local_p), rem_p)

                uploaded_count += len(chunk)
                transferred_bytes += chunk_bytes
                pbar.update(len(chunk))

                elapsed = time.time() - start_time
                speed_mb_s = (transferred_bytes - skipped_bytes) / (1024 * 1024) / max(0.1, elapsed)
                rem_bytes = total_bytes - transferred_bytes
                eta_s = int(rem_bytes / (speed_mb_s * 1024 * 1024)) if speed_mb_s > 0 else 0
                mins, secs = divmod(eta_s, 60)

                cur_cat = chunk[-1][3]
                pbar.set_postfix_str(
                    f"{transferred_bytes/(1024*1024*1024):.2f}/{total_gb:.2f}GB | "
                    f"{speed_mb_s:.1f}MB/s | {cur_cat} | ETA:{mins:02d}:{secs:02d}"
                )

            pbar.close()
            total_elapsed = time.time() - start_time
            print(f"\n✓ All {total_files} files staged on volume in {total_elapsed:.1f}s!")

    # Remote verification execution
    print("\n[2/4] Executing remote Modal verification function (minimal CPU container)...")
    with modal.enable_output():
        with app.run():
            res = verify_remote_dataset.remote()
            print("Remote verification result:", res)


if __name__ == "__main__":
    main()
