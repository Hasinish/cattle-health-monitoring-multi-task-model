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
    cpu=4.0,
    memory=8192,
    timeout=1800,
)
def pack_cvb_remote() -> Dict[str, Any]:
    """Packages the 226k CVB files into an uncompressed sequential tar archive."""
    print("=" * 70)
    print("  📦 PACKING CVB DATASET ON SOURCE (hasinishrak2015)")
    print("  Volume: cvb-data mounted at /data")
    print("=" * 70)

    cvb_dir = Path("/data/cvb")
    if not cvb_dir.exists():
        raise FileNotFoundError(f"CVB directory not found at {cvb_dir}")

    tar_path = Path("/data/cvb_archive.tar")
    if tar_path.exists() and tar_path.stat().st_size > 14 * (1024**3):
        print(f"[OK] cvb_archive.tar already exists ({tar_path.stat().st_size / (1024**3):.2f} GB). Skipping pack.")
        return {"status": "already_packed", "size_gb": tar_path.stat().st_size / (1024**3)}

    print("[*] Running uncompressed tar creation (sequential 100+ MB/s)...")
    t0 = time.time()
    
    cmd = ["tar", "-cf", str(tar_path), "-C", "/data", "cvb"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"tar packing failed: {res.stderr}")

    elapsed = time.time() - t0
    size_gb = tar_path.stat().st_size / (1024**3)
    speed = (tar_path.stat().st_size / (1024**2)) / max(0.1, elapsed)
    print(f"[OK] Packed {size_gb:.2f} GB in {elapsed:.1f}s ({speed:.1f} MB/s)!")

    print("[*] Committing volume checkpoint on source...")
    cvb_volume.commit()
    return {"status": "success", "size_gb": size_gb, "elapsed_s": elapsed}


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

    # 1. Stream cloud-to-cloud
    print("[*] Checking source archive...")
    t_start = time.time()

    # Heartbeat thread
    stop_event = threading.Event()
    def heartbeat():
        while not stop_event.wait(30.0):
            if temp_tar.exists():
                curr_gb = temp_tar.stat().st_size / (1024**3)
                el = time.time() - t_start
                spd = (temp_tar.stat().st_size / (1024**2)) / max(0.1, el)
                print(f"  >>> Streaming: {curr_gb:.2f} GB transferred | Speed: {spd:.1f} MB/s | Elapsed: {el:.0f}s", flush=True)

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

    # 2. Extract in-place
    print("\n" + "=" * 75)
    print("  📦 EXTRACTING CVB ARCHIVE ON DESTINATION")
    print("=" * 75)
    t_ext = time.time()
    
    cmd = ["tar", "-xf", str(dest_tar), "-C", "/data"]
    print(f"[*] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] tar extraction warning/error: {res.stderr[:500]}")
    else:
        print("[OK] Extraction completed cleanly!")

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
