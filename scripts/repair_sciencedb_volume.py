# -*- coding: utf-8 -*-
"""
ScienceDB Dataset Repair and Exhaustive Integrity Verification on Modal
========================================================================
Verifies the 1,753-image patch applied to Modal volume 'sciencedb-data',
overwriting 0-byte corrupted files, and executes an exhaustive 53,566-image
PIL readability audit across all 5 canonical BCS classes.
"""

import sys
import os
from pathlib import Path

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

volume = modal.Volume.from_name("sciencedb-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pillow", "pandas", "tqdm")
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "bcs" / "sciencedb"),
        remote_path="/root/datasets/bcs/sciencedb",
    )
    .add_local_file(
        str(REPO_ROOT / "artifacts" / "bcs_baseline" / "sciencedb_volume_zero_bytes.json"),
        remote_path="/root/sciencedb_volume_zero_bytes.json",
    )
)

app = modal.App("repair-sciencedb-volume", image=image)


@app.function(
    volumes={"/data": volume},
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def apply_patch_and_verify_exhaustive():
    import json
    import zipfile
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    import pandas as pd
    from tqdm import tqdm

    print("\n" + "=" * 70, flush=True)
    print("  SCIENCEDB MODAL VOLUME REPAIR & EXHAUSTIVE VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    dataset_dir = Path("/data/dataset")
    patch_zip = Path("/data/sciencedb_patch_1753.zip")
    manifest_path = Path("/root/sciencedb_volume_zero_bytes.json")

    with open(manifest_path, "r") as f:
        affected_files = json.load(f)

    # -------------------------------------------------------------------------
    # 1. PRE-REPAIR & EXTRACTION CHECK
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Checking Patch & Repair State for {len(affected_files)} affected files...", flush=True)
    if patch_zip.exists():
        print(f"[*] Found patch archive {patch_zip} ({patch_zip.stat().st_size / (1024*1024):.2f} MB).", flush=True)
        with zipfile.ZipFile(patch_zip, "r") as zf:
            infolist = zf.infolist()
            assert len(infolist) == 1753, f"Expected 1753 files in zip, found {len(infolist)}"
            zf.extractall(dataset_dir)
        patch_zip.unlink()
        volume.commit()
        print("[*] Successfully extracted and committed patch files.", flush=True)
    else:
        print("[*] Patch archive already extracted and committed on volume.", flush=True)

    # -------------------------------------------------------------------------
    # 2. VERIFY THE 1,753 REPAIRED FILES INDIVIDUALLY
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Verifying all {len(affected_files)} repaired files individually...", flush=True)
    repaired_ok = 0
    for rel in affected_files:
        p = dataset_dir / rel
        assert p.exists(), f"File {p} does not exist!"
        sz = p.stat().st_size
        assert sz > 0, f"File {p} is still 0 bytes!"
        try:
            with Image.open(p) as img:
                w, h = img.size
                assert w > 0 and h > 0
                repaired_ok += 1
        except Exception as e:
            raise RuntimeError(f"Repaired file {p} cannot be opened with PIL: {e}")

    print(f"[*] 100% of patched files verified: {repaired_ok}/1753 opened successfully with PIL.", flush=True)

    # -------------------------------------------------------------------------
    # 3. EXHAUSTIVE INTEGRITY VERIFICATION ACROSS ALL 53,566 IMAGES
    # -------------------------------------------------------------------------
    print("\n[3/5] Running EXHAUSTIVE integrity audit across ALL ScienceDB images on volume...", flush=True)
    expected_classes = {
        "3.25": 7536,
        "3.5": 13256,
        "3.75": 14255,
        "4.0": 12556,
        "4.25": 5963,
    }
    expected_grand_total = 53566

    actual_class_counts = {}
    all_image_paths = []
    zero_byte_count = 0

    for c, expected_n in expected_classes.items():
        c_dir = dataset_dir / c
        assert c_dir.exists(), f"Class directory {c_dir} is missing!"
        files = [f for f in c_dir.iterdir() if f.suffix.lower() == ".jpg"]
        actual_class_counts[c] = len(files)
        assert len(files) == expected_n, f"Class {c} count mismatch! Expected {expected_n}, got {len(files)}"
        for f in files:
            if f.stat().st_size == 0:
                zero_byte_count += 1
            all_image_paths.append(f)

    grand_total = len(all_image_paths)
    print(f"[*] Total image files found across all 5 classes: {grand_total} (Expected: {expected_grand_total})", flush=True)
    assert grand_total == expected_grand_total, f"Grand total mismatch: {grand_total} vs {expected_grand_total}"
    assert zero_byte_count == 0, f"Found {zero_byte_count} zero-byte files after repair!"
    print(f"[*] Post-repair zero-byte count across ALL {grand_total} images: {zero_byte_count}", flush=True)

    print("\nClass breakdown:", flush=True)
    for c, n in actual_class_counts.items():
        print(f"    - Class {c}: {n} images (Expected: {expected_classes[c]}) -> MATCH", flush=True)

    # Fast multi-threaded PIL verification with chunksize & explicit flushing
    print(f"\n[*] Testing PIL Image.open() on ALL {grand_total} images (64 worker threads)...", flush=True)
    
    def check_pil(path):
        try:
            with Image.open(path) as img:
                w, h = img.size
                return w > 0 and h > 0
        except Exception:
            return False

    results = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        for idx, ok in enumerate(pool.map(check_pil, all_image_paths, chunksize=64)):
            results.append(ok)
            if (idx + 1) % 5000 == 0 or (idx + 1) == grand_total:
                valid_so_far = sum(results)
                print(f"[*] Verified {idx + 1}/{grand_total} images ({((idx + 1)/grand_total)*100:.1f}%) | PIL valid: {valid_so_far} | PIL failed: {len(results) - valid_so_far}", flush=True)

    pil_valid = sum(results)
    pil_failed = grand_total - pil_valid
    print(f"\n[*] Exhaustive PIL Audit Result: {pil_valid}/{grand_total} images opened successfully.", flush=True)
    assert pil_failed == 0, f"{pil_failed} images failed PIL open check!"
    assert pil_valid == expected_grand_total, f"Expected {expected_grand_total} valid images, got {pil_valid}"

    # -------------------------------------------------------------------------
    # 4. VERIFY CANONICAL SPLIT HASHES AND SAMPLE RESOLUTION
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying canonical split CSVs and SHA-256 hashes...")
    splits_dir = Path("/root/datasets/bcs/sciencedb")
    expected_hashes = {
        "train": "9f6b0bc716e01a2ab22208daff1c49e49fd450a4d7cf0a57b2979275ed33497a",
        "val": "e223e3c4c081ca5c9f993f7156dc791df97b6ea6d011b8b4b6f068590b3d975d",
        "test": "eae459e031d06c4b1150ce2cbcdcb8259724b070b831341222b15c99e3626e5f",
    }
    split_details = {}
    for s_name, exp_hash in expected_hashes.items():
        sp = splits_dir / f"{s_name}.csv"
        assert sp.exists(), f"Split file {sp} not found!"
        with open(sp, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == exp_hash, f"Hash mismatch on {s_name}.csv: {actual_hash} vs {exp_hash}"
        df = pd.read_csv(sp)
        split_details[s_name] = {"rows": len(df), "sha256": actual_hash}
        print(f"[*] Split {s_name}.csv: {len(df)} rows, hash {actual_hash[:16]}... -> MATCH")

    # -------------------------------------------------------------------------
    # 5. FINAL VOLUME COMMIT & CERTIFICATION
    # -------------------------------------------------------------------------
    print("\n[5/5] Finalizing volume commit...")
    volume.commit()
    print("\n" + "=" * 70)
    print("  EXHAUSTIVE INTEGRITY VERIFICATION: 100% PASS")
    print(f"  Pre-Repair Zero-Byte Count : 1,753")
    print(f"  Post-Repair Zero-Byte Count: {zero_byte_count}")
    print(f"  Total PIL-Readable Images  : {pil_valid} / {grand_total}")
    print("=" * 70 + "\n")

    return {
        "success": True,
        "pre_zero_count": 1753,
        "post_zero_count": zero_byte_count,
        "grand_total": grand_total,
        "pil_valid": pil_valid,
        "class_counts": actual_class_counts,
        "split_hashes_match": True,
    }


@app.local_entrypoint()
def main():
    res = apply_patch_and_verify_exhaustive.remote()
    print("Remote execution result:", res)
    if res.get("success"):
        print("\n[SUCCESS] ScienceDB dataset volume is 100% repaired and exhaustively verified!")
    else:
        print("\n[FAILED] Repair verification failed!")
        sys.exit(1)
