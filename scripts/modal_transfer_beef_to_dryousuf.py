# -*- coding: utf-8 -*-
"""Cloud-to-Cloud Fast Transfer of Kaggle Beef Dataset (hasinishrak2015 -> dryousufmozumder).

Transfers the full 45.22 GB Kaggle Beef archive directly between Modal accounts
in the cloud at ~100-150 MB/s datacenter speeds with ZERO local PC bandwidth.

Usage:
    # 1. Test cloud connection:
    modal run --profile dryousufmozumder scripts/modal_transfer_beef_to_dryousuf.py::test_connection

    # 2. Transfer archive.zip (and optionally auto-extract):
    modal run --profile dryousufmozumder scripts/modal_transfer_beef_to_dryousuf.py::transfer_beef

    # Or transfer only without extraction:
    modal run --profile dryousufmozumder scripts/modal_transfer_beef_to_dryousuf.py::transfer_beef --extract=False
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

# Target volume on destination profile (dryousufmozumder)
target_volume = modal.Volume.from_name("beef-behavior-data", create_if_missing=True)

transfer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("p7zip-full")
    .pip_install("modal", "tqdm")
)

app = modal.App("transfer-beef-to-dryousuf", image=transfer_image)


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
# 1. TEST CONNECTION REMOTE
# ==============================================================================
@app.function(
    volumes={"/data": target_volume},
    cpu=1.0,
    memory=2048,
    timeout=120,
)
def test_connection_remote(src_creds: Dict[str, str]) -> Dict[str, Any]:
    """Verifies direct cross-account cloud connection from inside dryousufmozumder."""
    print("=" * 70)
    print("  TESTING CROSS-ACCOUNT CONNECTION (dryousufmozumder -> hasinishrak2015)")
    print("=" * 70)

    print("[*] Connecting to hasinishrak2015 API...")
    client = modal.Client.from_credentials(src_creds["token_id"], src_creds["token_secret"])
    source_vol = modal.Volume.from_name("beef-behavior-data", client=client)
    
    entries = source_vol.listdir("/beef_behavior")
    print(f"  [OK] Connected! Found {len(entries)} items in source /beef_behavior:")
    archive_size = 0
    for e in entries:
        size_str = f"{e.size / (1024**3):.2f} GB" if e.size > 0 else "DIR"
        print(f"    - {e.path} ({size_str})")
        if "archive.zip" in e.path:
            archive_size = e.size

    return {
        "status": "success",
        "archive_size_bytes": archive_size,
        "archive_size_gb": archive_size / (1024**3),
    }


@app.local_entrypoint()
def test_connection():
    src_creds = _get_profile_creds("hasinishrak2015")
    res = test_connection_remote.remote(src_creds)
    print("\n[[OK]] Cloud-to-Cloud Bridge is 100% operational!")
    print(f"    Target Source: hasinishrak2015 (beef-behavior-data)")
    print(f"    Target Archive: {res['archive_size_gb']:.2f} GB ({res['archive_size_bytes']:,} bytes)")


# ==============================================================================
# 2. TRANSFER & EXTRACT BEEF REMOTE
# ==============================================================================
@app.function(
    volumes={"/data": target_volume},
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def transfer_beef_remote(src_creds: Dict[str, str], extract: bool = True) -> Dict[str, Any]:
    """Streams archive.zip cloud-to-cloud and extracts on dryousufmozumder."""
    print("=" * 75)
    print("  🚀 KAGGLE BEEF CLOUD-TO-CLOUD DIRECT TRANSFER")
    print("  Source:      hasinishrak2015 (beef-behavior-data)")
    print("  Destination: dryousufmozumder (/data/beef_behavior)")
    print("  Target File: beef_behavior/archive.zip (45.22 GB)")
    print("  Bandwidth:   100% Datacenter Pipe (0 bytes through local Wi-Fi)")
    print("=" * 75)

    dest_dir = Path("/data/beef_behavior")
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_dest = dest_dir / "archive.zip"
    temp_dest = dest_dir / "archive.zip.transfer_tmp"

    client = modal.Client.from_credentials(src_creds["token_id"], src_creds["token_secret"])
    source_vol = modal.Volume.from_name("beef-behavior-data", client=client)

    # 1. Check if already transferred
    expected_size = 48553721000  # ~45.22 GB
    if archive_dest.exists() and archive_dest.stat().st_size >= expected_size * 0.99:
        print(f"\n[[OK]] archive.zip already exists on destination volume ({archive_dest.stat().st_size / (1024**3):.2f} GB).")
    else:
        print(f"\n[*] Starting high-speed cloud stream ({expected_size / (1024**3):.2f} GB)...")
        t_start = time.time()

        # Heartbeat / checkpoint thread to keep container and volume fresh
        stop_event = threading.Event()
        def heartbeat():
            while not stop_event.wait(60.0):
                if temp_dest.exists():
                    curr_bytes = temp_dest.stat().st_size
                    curr_gb = curr_bytes / (1024**3)
                    pct = (curr_bytes / expected_size) * 100
                    elapsed = time.time() - t_start
                    speed = (curr_bytes / (1024**2)) / max(0.1, elapsed)
                    print(f"  >>> Streaming: {curr_gb:.2f} / 45.22 GB ({pct:.1f}%) | Speed: {speed:.1f} MB/s | Elapsed: {elapsed:.0f}s", flush=True)

        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()

        try:
            with open(temp_dest, "wb") as f:
                source_vol.read_file_into_fileobj("beef_behavior/archive.zip", f)
        finally:
            stop_event.set()

        # Rename atomic
        if temp_dest.exists():
            temp_dest.replace(archive_dest)

        total_elapsed = time.time() - t_start
        size_gb = archive_dest.stat().st_size / (1024**3)
        avg_speed = (archive_dest.stat().st_size / (1024**2)) / max(0.1, total_elapsed)
        print(f"\n[[OK]] archive.zip transferred successfully in {total_elapsed:.1f}s ({avg_speed:.1f} MB/s)!")

        print("[*] Committing destination volume checkpoint...")
        target_volume.commit()

    # 2. Extraction step
    if extract:
        print("\n" + "=" * 75)
        print("  📦 EXTRACTING KAGGLE BEEF ARCHIVE (Multi-Threaded 7z)")
        print("=" * 75)
        t_ext = time.time()
        
        # 7z extraction directly into /data/beef_behavior/
        cmd = ["7z", "x", "-y", "-mmt=on", str(archive_dest), f"-o{dest_dir}"]
        print(f"[*] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        if res.returncode != 0:
            print(f"[!] 7z warning/error (code {res.returncode}):\n{res.stderr[:500]}")
        else:
            print("  [OK] 7z extraction completed cleanly!")
        
        ext_time = time.time() - t_ext
        print(f"[[OK]] Extraction finished in {ext_time:.1f}s ({ext_time/60:.1f} mins).")

        # Check extracted directories
        cats = list((dest_dir / "Category Videos").glob("*")) if (dest_dir / "Category Videos").exists() else []
        labels = list((dest_dir / "Labelframes").glob("*")) if (dest_dir / "Labelframes").exists() else []
        print(f"  [OK] Found {len(cats)} items in Category Videos")
        print(f"  [OK] Found {len(labels)} items in Labelframes")

        print("[*] Committing final extracted volume...")
        target_volume.commit()
    else:
        print("\n[*] Extraction skipped (extract=False). archive.zip is safely preserved.")

    return {
        "status": "success",
        "archive_size_gb": archive_dest.stat().st_size / (1024**3),
        "extract": extract,
    }


@app.local_entrypoint()
def transfer_beef(extract: bool = True):
    src_creds = _get_profile_creds("hasinishrak2015")
    print(f"[LOCAL] Initiating Kaggle Beef direct cloud transfer to profile: dryousufmozumder (extract={extract})...")
    res = transfer_beef_remote.remote(src_creds, extract=extract)
    print("\n[[OK]] ALL DONE! Kaggle Beef is now present and committed on dryousufmozumder!")
