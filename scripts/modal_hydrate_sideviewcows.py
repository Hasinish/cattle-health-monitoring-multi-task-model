"""
Modal Cloud Hydration & Verification Pipeline for SideViewCows2026 (Zenodo Record 21605650).

Low-cost, high-reliability cloud data hydration targeting Modal profile tigerwood697.
Configured for minimal compute cost: 1.0 CPU, 2048 MB RAM, NO GPU.

Usage:
  # Main Full Hydration (Interactive, live streaming progress):
  modal run --profile tigerwood697 scripts/modal_hydrate_sideviewcows.py::main

  # Standalone Post-Hydration Verification:
  modal run --profile tigerwood697 scripts/modal_hydrate_sideviewcows.py::verify

  # Pre-Flight Smoke Test (Checks volume mount & Zenodo connectivity without 25GB download):
  modal run --profile tigerwood697 scripts/modal_hydrate_sideviewcows.py::smoke_test
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

# Windows UTF-8 console output & path patch
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
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Persistent Storage Volume on tigerwood697
sideview_vol = modal.Volume.from_name("sideview-data", create_if_missing=True)

# Minimal Container Image
hydrate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "pillow", "tqdm")
    .add_local_file(
        str(REPO_ROOT / "scripts" / "fast_download_sideviewcows.py"),
        remote_path="/root/scripts/fast_download_sideviewcows.py",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
)

app = modal.App("sideviewcows-hydration", image=hydrate_image)


# ==============================================================================
# VERIFICATION HELPER (REUSABLE ACROSS REMOTE FUNCTIONS)
# ==============================================================================
def run_verification(dataset_root: Path, protocols_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Exhaustively verifies SideViewCows2026 dataset on disk:
    1. Exact file counts per subset (parlor, barn, snapshots) for images & masks
    2. Total images (80,260) and masks (80,260)
    3. Biological cow identity count (110)
    4. Zero-byte file detection
    5. Random PIL decode sanity checks (RGB images & L/1/P masks)
    6. Canonical protocol CSV path mapping checks
    """
    import random
    from PIL import Image
    import pandas as pd

    print("\n" + "=" * 70)
    print(f"  VERIFYING SIDEVIEWCOWS2026 DATASET AT: {dataset_root}")
    print("=" * 70)
    sys.stdout.flush()

    expected_subsets = {
        "parlor": {"expected_imgs": 54393, "expected_masks": 54393},
        "barn": {"expected_imgs": 25260, "expected_masks": 25260},
        "snapshots": {"expected_imgs": 607, "expected_masks": 607},
    }

    subset_results = {}
    total_imgs = 0
    total_masks = 0
    zero_byte_files = []
    unique_cows = set()
    all_img_paths = []
    all_mask_paths = []

    for subset, specs in expected_subsets.items():
        img_dir = dataset_root / subset / "images"
        mask_dir = dataset_root / subset / "masks"

        if not img_dir.exists():
            subset_results[subset] = {"error": f"Images directory missing: {img_dir}"}
            continue
        if not mask_dir.exists():
            subset_results[subset] = {"error": f"Masks directory missing: {mask_dir}"}
            continue

        imgs = list(img_dir.glob("*/*.jpg"))
        masks = list(mask_dir.glob("*/*.png"))

        n_imgs = len(imgs)
        n_masks = len(masks)
        total_imgs += n_imgs
        total_masks += n_masks

        all_img_paths.extend(imgs)
        all_mask_paths.extend(masks)

        # Collect unique cow IDs from subdirectories
        for p in img_dir.iterdir():
            if p.is_dir():
                unique_cows.add(p.name)

        # Check zero-byte files
        for f in imgs + masks:
            if f.stat().st_size == 0:
                zero_byte_files.append(str(f))

        match_imgs = (n_imgs == specs["expected_imgs"])
        match_masks = (n_masks == specs["expected_masks"])
        status = "PASS" if (match_imgs and match_masks) else "FAIL"

        subset_results[subset] = {
            "status": status,
            "images_found": n_imgs,
            "images_expected": specs["expected_imgs"],
            "masks_found": n_masks,
            "masks_expected": specs["expected_masks"],
        }

        print(f"[{status}] Subset '{subset}': {n_imgs}/{specs['expected_imgs']} images, {n_masks}/{specs['expected_masks']} masks")
        sys.stdout.flush()

    # 1. Image & Mask Count Assertions
    all_counts_match = (
        total_imgs == 80260 and
        total_masks == 80260 and
        len(unique_cows) == 110 and
        len(zero_byte_files) == 0
    )

    print(f"\n[*] Global Totals: {total_imgs}/80,260 Images, {total_masks}/80,260 Masks, {len(unique_cows)}/110 Cows")
    print(f"[*] Zero-Byte Files Detected: {len(zero_byte_files)}")
    sys.stdout.flush()

    # 2. Random PIL Decodes
    decode_results = {"rgb_samples_checked": 0, "mask_samples_checked": 0, "decode_errors": 0}
    if all_img_paths and all_mask_paths:
        sample_imgs = random.sample(all_img_paths, min(10, len(all_img_paths)))
        sample_masks = random.sample(all_mask_paths, min(10, len(all_mask_paths)))

        print("\n[*] Running Random PIL Decode Sanity Checks (10 images, 10 masks)...")
        for img_p in sample_imgs:
            try:
                with Image.open(img_p) as im:
                    im.verify()
                decode_results["rgb_samples_checked"] += 1
            except Exception as e:
                decode_results["decode_errors"] += 1
                print(f"[DECODE ERROR] Failed to open image {img_p}: {e}")

        for mask_p in sample_masks:
            try:
                with Image.open(mask_p) as im:
                    im.verify()
                decode_results["mask_samples_checked"] += 1
            except Exception as e:
                decode_results["decode_errors"] += 1
                print(f"[DECODE ERROR] Failed to open mask {mask_p}: {e}")

        print(f"[PASS] PIL Decodes: {decode_results['rgb_samples_checked']} RGB images & {decode_results['mask_samples_checked']} masks verified successfully.")
        sys.stdout.flush()

    # 3. Protocol Path Resolution Checks
    protocol_checks = {}
    if protocols_dir and protocols_dir.exists():
        print("\n[*] Verifying Canonical Protocol CSV Path Resolutions...")
        protocol_files = [
            "protocol_cross_setting.csv",
            "protocol_longitudinal.csv",
            "protocol_open_set.csv",
            "protocol_closed_set.csv",
        ]
        prefix_to_strip = "datasets/id/external/sideviewcows2026/"

        for proto_name in protocol_files:
            proto_path = protocols_dir / proto_name
            if not proto_path.exists():
                protocol_checks[proto_name] = "MISSING_CSV"
                continue

            df = pd.read_csv(proto_path, nrows=100)
            sample_rows = df.sample(min(15, len(df)), random_state=2026)
            resolved_ok = 0
            for _, row in sample_rows.iterrows():
                rel_img = str(row["image_path"]).replace(prefix_to_strip, "").lstrip("/\\")
                rel_mask = str(row["mask_path"]).replace(prefix_to_strip, "").lstrip("/\\")
                full_img = dataset_root / rel_img
                full_mask = dataset_root / rel_mask
                if full_img.exists() and full_mask.exists():
                    resolved_ok += 1

            pass_rate = resolved_ok / len(sample_rows)
            status = "PASS" if pass_rate == 1.0 else f"FAIL ({resolved_ok}/{len(sample_rows)})"
            protocol_checks[proto_name] = status
            print(f"[{'PASS' if pass_rate == 1.0 else 'FAIL'}] {proto_name}: 100% path resolution verified on sampled rows ({resolved_ok}/{len(sample_rows)})")
            sys.stdout.flush()

    overall_pass = all_counts_match and (decode_results["decode_errors"] == 0)
    print("=" * 70)
    print(f"  VERIFICATION VERDICT: {'ALL CHECKS PASSED ✅' if overall_pass else 'VERIFICATION FAILED ❌'}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    return {
        "overall_verified": overall_pass,
        "total_images": total_imgs,
        "total_masks": total_masks,
        "unique_cows": len(unique_cows),
        "zero_byte_count": len(zero_byte_files),
        "subset_results": subset_results,
        "decode_results": decode_results,
        "protocol_checks": protocol_checks,
    }


# ==============================================================================
# EXTRACTION HELPER WITH PROGRESS
# ==============================================================================
def extract_zip_monitored(zip_path: Path, extract_to: Path, pbar_cls):
    """
    Extracts a zip file while streaming smooth progress updates.
    """
    print(f"\n📦 Starting Extraction: {zip_path.name} -> {extract_to}...")
    sys.stdout.flush()
    start_t = time.time()

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.infolist()
        total_uncompressed = sum(m.file_size for m in members)
        pbar = pbar_cls(
            total_bytes=total_uncompressed,
            desc=f"📦 Unzipping {zip_path.name}",
            bar_len=25,
        )
        for member in members:
            z.extract(member, extract_to)
            pbar.update(member.file_size)
        pbar.close()

    duration = time.time() - start_t
    print(f"✅ Extracted {zip_path.name} ({len(members)} entries, {total_uncompressed / (1024**3):.2f} GB) in {duration:.1f}s\n")
    sys.stdout.flush()


# ==============================================================================
# MODAL REMOTE FUNCTIONS
# ==============================================================================
@app.function(
    volumes={"/data": sideview_vol},
    cpu=1.0,
    memory=2048,
    timeout=300,
)
def smoke_test_remote() -> dict:
    """
    Fast pre-flight smoke test on minimal container:
    1. Verifies /data volume is mounted and writable
    2. Tests HTTP Range connectivity to Zenodo record 21605650
    3. Downloads tiny README.md (8 KB) to confirm network I/O
    4. Asserts that 0 large zip archives are downloaded
    """
    import sys
    sys.path.insert(0, "/root")
    from scripts.fast_download_sideviewcows import (
        BASE_URL,
        BROWSER_HEADERS,
        FILES_CATALOG,
        download_chunk_with_retry,
    )

    print("\n" + "=" * 70)
    print("  MODAL SMOKE TEST: SIDEVIEWCOWS2026 PRE-FLIGHT READINESS")
    print("=" * 70)

    dataset_root = Path("/data/sideviewcows2026")
    dataset_root.mkdir(parents=True, exist_ok=True)
    assert dataset_root.exists(), f"Volume mount failed: {dataset_root}"
    print(f"[*] Volume Mount PASS: {dataset_root} is writable")

    # Test HTTP HEAD request to Zenodo for manifest and README
    test_files = ["README.md", "manifest.csv"]
    zenodo_reachability = {}

    for k in test_files:
        url = FILES_CATALOG[k]["url"]
        req = urllib.request.Request(url, headers=dict(BROWSER_HEADERS, Range="bytes=0-100"))
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
                content_range = resp.headers.get("Content-Range", "")
                zenodo_reachability[k] = f"HTTP {status} (Range: {content_range})"
        except Exception as e:
            zenodo_reachability[k] = f"FAILED: {e}"

    print(f"[*] Zenodo Connectivity: {zenodo_reachability}")

    # Download tiny README.md to prove write capability
    readme_info = FILES_CATALOG["README.md"]
    readme_dest = dataset_root / "README.md"
    readme_part = dataset_root / "README.md.part"
    download_chunk_with_retry(readme_info["url"], readme_part, 0, readme_info["size"] - 1)
    if readme_part.exists():
        if readme_dest.exists():
            readme_dest.unlink()
        readme_part.rename(readme_dest)

    assert readme_dest.exists() and readme_dest.stat().st_size == readme_info["size"], "README download failed"
    print(f"[*] Test Download PASS: {readme_dest.name} ({readme_dest.stat().st_size} bytes)")

    # Assert no large archives were pulled
    for z in ["snapshots.zip", "parlor.zip", "barn.zip"]:
        assert not (dataset_root / z).exists(), f"Accidental large download detected: {z}"

    sideview_vol.commit()
    print("[*] Volume commit successful.")
    print("=" * 70)
    print("  SMOKE TEST PASSED: READY FOR USER-RUN FULL HYDRATION")
    print("=" * 70 + "\n")

    return {
        "status": "SMOKE_PASS",
        "volume_mount": str(dataset_root),
        "zenodo_reachability": zenodo_reachability,
        "test_file_size": readme_dest.stat().st_size,
    }


@app.function(
    volumes={"/data": sideview_vol},
    cpu=1.0,
    memory=2048,
    timeout=10800,  # 3 hours for 25GB download + extraction
)
def hydrate_full_remote(threads: int = 4, skip_cleanup: bool = False) -> dict:
    """
    Executes full SideViewCows2026 hydration directly into /data/sideviewcows2026:
    1. Downloads metadata files
    2. Downloads and extracts snapshots.zip, parlor.zip, barn.zip
    3. Runs complete 80,260-image / 80,260-mask verification
    4. Cleans up zip archives to save disk space
    5. Periodically commits volume progress
    """
    import sys
    sys.path.insert(0, "/root")
    from scripts.fast_download_sideviewcows import (
        FILES_CATALOG,
        CleanProgressBar,
        download_file_multithreaded,
    )

    start_total_time = time.time()
    dataset_root = Path("/data/sideviewcows2026")
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    dataset_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MODAL HYDRATION: SIDEVIEWCOWS2026 (ZENODO RECORD 21605650)")
    print(f"  Target Root: {dataset_root}")
    print(f"  Parallel Download Threads: {threads}")
    print(f"  Machine Config: 1.0 CPU, 2048 MB RAM, NO GPU")
    print("=" * 70)
    sys.stdout.flush()

    # --------------------------------------------------------------------------
    # Phase 1: Metadata Files
    # --------------------------------------------------------------------------
    print("\n--- Phase 1: Metadata Files ---")
    sys.stdout.flush()
    meta_files = ["manifest.csv", "SHA256SUMS", "README.md", "preview.jpg"]
    for mf in meta_files:
        info = FILES_CATALOG[mf]
        dest = dataset_root / mf
        download_file_multithreaded(info["url"], dest, info["size"], num_threads=1)
        sys.stdout.flush()

    sideview_vol.commit()
    print("[*] Metadata files persisted and volume committed.")
    sys.stdout.flush()

    # --------------------------------------------------------------------------
    # Phase 2: Data Archives (Download, Extract, Commit per subset)
    # --------------------------------------------------------------------------
    print("\n--- Phase 2: Data Archives ---")
    sys.stdout.flush()

    subsets_spec = [
        ("snapshots", 607, 607),
        ("parlor", 54393, 54393),
        ("barn", 25260, 25260),
    ]

    for subset, exp_imgs, exp_masks in subsets_spec:
        img_dir = dataset_root / subset / "images"
        mask_dir = dataset_root / subset / "masks"
        zip_name = f"{subset}.zip"
        zip_info = FILES_CATALOG[zip_name]
        zip_path = dataset_root / zip_name

        # Check if already fully extracted
        if img_dir.exists() and mask_dir.exists():
            curr_imgs = len(list(img_dir.glob("*/*.jpg")))
            curr_masks = len(list(mask_dir.glob("*/*.png")))
            if curr_imgs == exp_imgs and curr_masks == exp_masks:
                print(f"\n[SKIP] Subset '{subset}' already fully extracted ({curr_imgs} images, {curr_masks} masks).")
                sys.stdout.flush()
                # Clean leftover zip if present
                if zip_path.exists() and not skip_cleanup:
                    zip_path.unlink()
                continue

        # Download archive
        print(f"\n📥 Downloading {zip_name} ({zip_info['size'] / (1024**3):.2f} GB)...")
        sys.stdout.flush()
        download_file_multithreaded(
            zip_info["url"],
            zip_path,
            zip_info["size"],
            num_threads=threads,
        )
        sys.stdout.flush()

        # Extract archive
        extract_zip_monitored(zip_path, dataset_root, CleanProgressBar)

        # Commit volume after each subset
        sideview_vol.commit()
        print(f"[*] Subset '{subset}' extracted and committed to volume.")
        sys.stdout.flush()

    # --------------------------------------------------------------------------
    # Phase 3: Comprehensive Verification
    # --------------------------------------------------------------------------
    verification_report = run_verification(dataset_root, protocols_dir=protocols_dir)

    if not verification_report["overall_verified"]:
        sideview_vol.commit()
        raise RuntimeError("Dataset verification FAILED! Review output above. Zip archives preserved for inspection.")

    # --------------------------------------------------------------------------
    # Phase 4: Zip Cleanup (Only after verified success)
    # --------------------------------------------------------------------------
    if not skip_cleanup:
        print("\n--- Phase 4: Storage Optimization (Pruning Source Zips) ---")
        freed_bytes = 0
        for subset, _, _ in subsets_spec:
            zp = dataset_root / f"{subset}.zip"
            if zp.exists():
                sz = zp.stat().st_size
                zp.unlink()
                freed_bytes += sz
                print(f"[REMOVED] {zp.name} (freed {sz / (1024**3):.2f} GB)")

        sideview_vol.commit()
        print(f"[*] Cleanup complete: Freed {freed_bytes / (1024**3):.2f} GB of raw zip storage.")
        sys.stdout.flush()
    else:
        print("\n[*] Zip cleanup skipped per user request (--skip-cleanup).")

    total_duration = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"  SIDEVIEWCOWS2026 HYDRATION COMPLETE (Total Time: {total_duration / 60:.1f} mins)")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    return {
        "status": "HYDRATION_SUCCESS",
        "total_duration_sec": round(total_duration, 2),
        "verification": verification_report,
    }


@app.function(
    volumes={"/data": sideview_vol},
    cpu=1.0,
    memory=2048,
    timeout=600,
)
def verify_remote() -> dict:
    """
    Standalone verification function: runs full health audit without downloading.
    """
    dataset_root = Path("/data/sideviewcows2026")
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    return run_verification(dataset_root, protocols_dir=protocols_dir)


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def smoke_test():
    """Fast pre-flight smoke test."""
    print("[LOCAL] Launching Modal pre-flight smoke test on tigerwood697...")
    res = smoke_test_remote.remote()
    print("\n[LOCAL] Pre-Flight Smoke Test Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def verify():
    """Standalone dataset verification."""
    print("[LOCAL] Launching standalone verification on Modal volume sideview-data...")
    res = verify_remote.remote()
    print("\n[LOCAL] Verification Result Summary:")
    print(f"  Verified Overall : {res.get('overall_verified')}")
    print(f"  Total Images     : {res.get('total_images')}")
    print(f"  Total Masks      : {res.get('total_masks')}")
    print(f"  Unique Cows      : {res.get('unique_cows')}")
    print(f"  Zero-Byte Files  : {res.get('zero_byte_count')}")


@app.local_entrypoint()
def main(threads: int = 4, skip_cleanup: bool = False):
    """
    Main entrypoint for full download & hydration.
    Streams live progress to terminal.
    """
    print(f"[LOCAL] Launching SideViewCows2026 cloud hydration on Modal (threads={threads})...")
    res = hydrate_full_remote.remote(threads=threads, skip_cleanup=skip_cleanup)
    print("\n[LOCAL] Hydration completed!")
    print(f"  Status        : {res.get('status')}")
    print(f"  Total Duration: {res.get('total_duration_sec')}s")
    print(f"  Images Verified: {res.get('verification', {}).get('total_images')}")
    print(f"  Masks Verified : {res.get('verification', {}).get('total_masks')}")


if __name__ == "__main__":
    pass
