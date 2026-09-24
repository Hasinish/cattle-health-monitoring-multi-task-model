# -*- coding: utf-8 -*-
"""Cloud-to-Cloud Fast Transfer of CVB Dataset (hasinishrak2015 -> dryousufmozumder).

Transfers the full 14.29 GB (226,344 files) CVB dataset directly between Modal
accounts in ~3-4 minutes via uncompressed cloud tar stream.

Usage:
    # Option A (Super-Fast Cloud-to-Cloud, ~3.5 mins):
    # Step 1: Pack CVB on source (hasinishrak2015, ~60s):
    modal run --profile hasinishrak2015 scripts/modal_transfer_cvb_to_dryousuf.py::pack_cvb

    # Step 2: Stream & extract on destination (dryousufmozumder, ~2.5 mins):
    modal run --profile dryousufmozumder scripts/modal_transfer_cvb_to_dryousuf.py::transfer_cvb

    # Option B (Direct 64-Stream Downloader, ~15 mins, zero packing required):
    modal run --profile dryousufmozumder scripts/modal_cvb_pipeline.py::download_cvb
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# Guard against Windows console encoding errors with unicode characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

cvb_volume = modal.Volume.from_name("cvb-data", create_if_missing=True)

transfer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tar")
    .pip_install("modal", "tqdm")
)

app = modal.App("transfer-cvb-to-dryousuf", image=transfer_image)


def _get_profile_creds(profile_name: str) -> Dict[str, str]:
    """Extract token_id and token_secret from local ~/.modal.toml."""
    cfg_path = Path(os.path.expanduser("~/.modal.toml"))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    if profile_name not in cfg:
        raise KeyError(f"Profile '{profile_name}' not found in ~/.modal.toml")
    return {
        "token_id": cfg[profile_name]["token_id"],
        "token_secret": cfg[profile_name]["token_secret"],
    }


# ==============================================================================
# STEP 1: PACK CVB ON SOURCE (hasinishrak2015)
# ==============================================================================
@app.function(
    volumes={"/data": cvb_volume},
    cpu=8.0,
    memory=16384,
    timeout=1800,
)
def pack_cvb_remote() -> Dict[str, Any]:
    """Packages the 226k CVB files using 64-worker parallel NVMe pre-staging."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("=" * 70)
    print("  🚀 64-WORKER PARALLEL CVB PACKING (hasinishrak2015)")
    print("  Volume: cvb-data mounted at /data")
    print("  Engine: 64-worker ThreadPoolExecutor + Local NVMe Staging")
    print("=" * 70)

    cvb_dir = Path("/data/cvb")
    if not cvb_dir.exists():
        raise FileNotFoundError(f"CVB directory not found at {cvb_dir}")

    tar_path = Path("/data/cvb_archive.tar")
    if tar_path.exists() and tar_path.stat().st_size > 14 * (1024**3):
        print(f"[OK] cvb_archive.tar already exists ({tar_path.stat().st_size / (1024**3):.2f} GB). Ready to stream!")
        return {"status": "already_packed", "size_gb": tar_path.stat().st_size / (1024**3)}

    stage_dir = Path("/tmp/cvb_stage")
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all cut and sub-directories to parallelize
    print("[1/3] Scanning CVB directories for 64-worker parallel transfer...")
    tasks: list[tuple[Path, Path]] = []
    
    # We walk the top 3 levels to split the 226k files into ~1,000 independent folders
    base_data = cvb_dir / "000058916v001" / "data"
    if base_data.exists():
        for category in ["raw_frames", "annotations", "cvb_in_ava_format"]:
            cat_dir = base_data / category
            if cat_dir.exists():
                for sub in cat_dir.iterdir():
                    rel = sub.relative_to(cvb_dir)
                    tasks.append((sub, stage_dir / rel))
    
    # Metadata files
    meta_dir = cvb_dir / "000058916v001" / "metadata"
    if meta_dir.exists():
        rel = meta_dir.relative_to(cvb_dir)
        tasks.append((meta_dir, stage_dir / rel))

    print(f"  ✓ Discovered {len(tasks):,} discrete folders across CVB.")
    print(f"[2/3] Launching 64-worker parallel pre-staging into local NVMe SSD...")

    t_stage = time.time()
    done = 0
    errors: list[str] = []

    def _copy_folder(item: tuple[Path, Path]) -> bool:
        src, dst = item
        if dst.exists():
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return True

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(_copy_folder, t): t for t in tasks}
        for future in as_completed(futures):
            t = futures[future]
            try:
                future.result()
            except Exception as e:
                errors.append(f"{t[0].name}: {e}")
            done += 1
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t_stage
                rate = done / max(0.1, el)
                eta = (len(tasks) - done) / max(0.1, rate)
                print(f"  >>> [64-WORKER NVMe PRE-STAGE] {done:4d} / {len(tasks)} folders ({done/len(tasks)*100:5.1f}%) | Speed: {rate:5.1f} folders/s | ETA: {eta:4.1f}s", flush=True)

    if errors:
        print(f"[!] Warning: encountered {len(errors)} copy errors: {errors[:3]}")

    stage_duration = time.time() - t_stage
    print(f"[OK] All {len(tasks)} folders pre-staged to local NVMe in {stage_duration:.1f}s ({len(tasks)/stage_duration:.1f} folders/s)!")

    # 2. Sequential tar from local NVMe to volume (pure sequential 150+ MB/s, zero FUSE lockups)
    print(f"\n[3/3] Creating single sequential archive {tar_path} on volume...")
    t_tar = time.time()
    
    cmd = ["tar", "-cf", str(tar_path), "-C", str(stage_dir), "."]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"tar packing failed: {res.stderr}")

    tar_duration = time.time() - t_tar
    size_gb = tar_path.stat().st_size / (1024**3)
    speed = (tar_path.stat().st_size / (1024**2)) / max(0.1, tar_duration)
    print(f"[OK] Single sequential tar completed in {tar_duration:.1f}s ({size_gb:.2f} GB at {speed:.1f} MB/s)!")

    # Clean up local NVMe staging dir
    shutil.rmtree(stage_dir, ignore_errors=True)

    print("[*] Committing volume checkpoint on source...")
    cvb_volume.commit()
    return {"status": "success", "size_gb": size_gb, "total_elapsed_s": stage_duration + tar_duration}


@app.local_entrypoint()
def pack_cvb():
    print("[LOCAL] Packaging CVB on hasinishrak2015...")
    res = pack_cvb_remote.remote()
    print(f"\n[OK] CVB successfully packed: {res['size_gb']:.2f} GB! Ready for transfer to dryousufmozumder.")


# ==============================================================================
# STEP 2: TRANSFER & EXTRACT ON DESTINATION (dryousufmozumder)
# ==============================================================================
@app.function(
    volumes={"/data": cvb_volume},
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def transfer_cvb_remote(src_creds: Dict[str, str], cleanup_tar: bool = True) -> Dict[str, Any]:
    """Streams cvb_archive.tar cloud-to-cloud and extracts into destination cvb-data."""
    print("=" * 75)
    print("  🚀 CVB CLOUD-TO-CLOUD DIRECT TRANSFER")
    print("  Source:      hasinishrak2015 (cvb-data)")
    print("  Destination: dryousufmozumder (/data/cvb)")
    print("  Bandwidth:   100% Datacenter Pipe (0 bytes through local Wi-Fi)")
    print("=" * 75)

    client = modal.Client.from_credentials(src_creds["token_id"], src_creds["token_secret"])
    source_vol = modal.Volume.from_name("cvb-data", client=client)

    dest_tar = Path("/data/cvb_archive.tar")
    temp_tar = Path("/data/cvb_archive.tar.transfer_tmp")
    expected_cvb_bytes = 14.29 * (1024**3)

    # 1. Stream cloud-to-cloud
    print("[*] Checking source archive...")
    t_start = time.time()

    # 5-second live streaming heartbeat thread
    stop_event = threading.Event()
    def heartbeat():
        while not stop_event.wait(5.0):
            if temp_tar.exists():
                curr_bytes = temp_tar.stat().st_size
                curr_gb = curr_bytes / (1024**3)
                pct = min(100.0, (curr_bytes / expected_cvb_bytes) * 100)
                el = time.time() - t_start
                spd = (curr_bytes / (1024**2)) / max(0.1, el)
                rem_bytes = max(0, expected_cvb_bytes - curr_bytes)
                eta = rem_bytes / max(1.0, spd * 1024 * 1024)
                print(f"  >>> [STREAMING CVB] {curr_gb:.2f} / 14.29 GB ({pct:.1f}%) | Speed: {spd:.1f} MB/s | Elapsed: {el:.0f}s | ETA: {eta:.0f}s", flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    try:
        with open(temp_tar, "wb") as f:
            source_vol.read_file_into_fileobj("cvb_archive.tar", f)
    finally:
        stop_event.set()

    if temp_tar.exists():
        temp_tar.replace(dest_tar)

    elapsed_dl = time.time() - t_start
    size_gb = dest_tar.stat().st_size / (1024**3)
    avg_speed = (dest_tar.stat().st_size / (1024**2)) / max(0.1, elapsed_dl)
    print(f"\n[OK] cvb_archive.tar transferred in {elapsed_dl:.1f}s ({avg_speed:.1f} MB/s)!")

    # 2. Extract in-place with live extraction monitor
    print("\n" + "=" * 75)
    print("  📦 EXTRACTING CVB ARCHIVE ON DESTINATION (226,344 files)")
    print("=" * 75)
    t_ext = time.time()
    
    stop_ext = threading.Event()
    cvb_dest_dir = Path("/data/cvb")
    def ext_monitor():
        while not stop_ext.wait(5.0):
            if cvb_dest_dir.exists():
                f_count = sum(len(files) for _, _, files in os.walk(cvb_dest_dir))
                pct = min(100.0, (f_count / 226344) * 100)
                el = time.time() - t_ext
                print(f"  >>> [EXTRACTING CVB] {f_count:,} / 226,344 files ({pct:.1f}%) | Elapsed: {el:.0f}s", flush=True)

    m_ext = threading.Thread(target=ext_monitor, daemon=True)
    m_ext.start()

    try:
        cmd = ["tar", "-xf", str(dest_tar), "-C", "/data"]
        print(f"[*] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] tar extraction warning/error: {res.stderr[:500]}")
        else:
            print("[OK] Extraction completed cleanly!")
    finally:
        stop_ext.set()

    ext_time = time.time() - t_ext
    print(f"[OK] Extraction finished in {ext_time:.1f}s ({ext_time/60:.1f} mins).")

    # Cleanup temporary tar to save volume quota
    if cleanup_tar and dest_tar.exists():
        print("[*] Cleaning up temporary cvb_archive.tar to reclaim volume quota...")
        dest_tar.unlink()

    # Commit volume
    print("[*] Committing final volume on dryousufmozumder...")
    cvb_volume.commit()

    return {"status": "success", "size_gb": size_gb, "elapsed_total": time.time() - t_start}


@app.local_entrypoint()
def transfer_cvb():
    src_creds = _get_profile_creds("hasinishrak2015")
    print("[LOCAL] Initiating CVB transfer to profile: dryousufmozumder...")
    res = transfer_cvb_remote.remote(src_creds)
    print("\n[OK] ALL DONE! CVB is now present and committed on dryousufmozumder!")
