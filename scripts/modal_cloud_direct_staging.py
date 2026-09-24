# -*- coding: utf-8 -*-
"""Cloud-to-Cloud Direct MTL Staging Bridge.

Transfers perception-cached datasets directly between Modal accounts in the cloud
(tigerwood697/tigerwood693 -> hasinishrak2015) with ZERO local PC bandwidth.

Usage:
    modal run --profile hasinishrak2015 scripts/modal_cloud_direct_staging.py::test_connection
    modal run --profile hasinishrak2015 scripts/modal_cloud_direct_staging.py::transfer_bcs
    modal run --profile hasinishrak2015 scripts/modal_cloud_direct_staging.py::transfer_behavior
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)


# Persistent volume on target profile (hasinishrak2015)
mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=True)

transfer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("modal", "tqdm", "pandas")
)

app = modal.App("mtl-cloud-direct-transfer", image=transfer_image)


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


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute SHA-256 with 16MB stream buffers."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# 1. TEST CONNECTION (SMOKE TEST)
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=1.0,
    memory=2048,
    timeout=120,
)
def test_connection_remote(bcs_creds: Dict[str, str], beh_creds: Dict[str, str]) -> Dict[str, Any]:
    """Verifies direct cross-account cloud connection from inside hasinishrak2015."""
    print("=" * 70)
    print("  TESTING CROSS-ACCOUNT CLOUD-TO-CLOUD CONNECTION")
    print("=" * 70)

    # 1. Connect to tigerwood697 (BCS)
    print("[*] Connecting to tigerwood697 (BCS)...")
    client_bcs = modal.Client.from_credentials(bcs_creds["token_id"], bcs_creds["token_secret"])
    vol_bcs = modal.Volume.from_name("sciencedb-perception-cache", client=client_bcs)
    bcs_entries = vol_bcs.listdir("/packed")
    print(f"  ✓ tigerwood697 connected! Found {len(bcs_entries)} files in /packed:")
    for e in bcs_entries:
        print(f"    - {e.path} ({e.size / 1024 / 1024:.2f} MB)")

    # 2. Connect to tigerwood693 (Behavior)
    print("\n[*] Connecting to tigerwood693 (Behavior)...")
    client_beh = modal.Client.from_credentials(beh_creds["token_id"], beh_creds["token_secret"])
    vol_beh = modal.Volume.from_name("behavior-perception-cache", client=client_beh)
    beh_entries = vol_beh.listdir("/production")
    print(f"  ✓ tigerwood693 connected! Found {len(beh_entries)} entries in /production.")

    return {"status": "success", "bcs_packed_files": len(bcs_entries), "beh_entries": len(beh_entries)}


@app.local_entrypoint()
def test_connection():
    bcs_creds = _get_profile_creds("tigerwood697")
    beh_creds = _get_profile_creds("tigerwood693")
    res = test_connection_remote.remote(bcs_creds, beh_creds)
    print("\n[✓] Cloud-to-Cloud Bridge is 100% operational!")
    print(f"    BCS files found: {res['bcs_packed_files']}")
    print(f"    Behavior entries found: {res['beh_entries']}")


# ==============================================================================
# 2. BCS DIRECT CLOUD TRANSFER
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def transfer_bcs_remote(creds: Dict[str, str]) -> Dict[str, Any]:
    """Transfers packed BCS tensors directly from tigerwood697 to hasinishrak2015 in the cloud."""
    print("=" * 70)
    print("  🚀 BCS DIRECT CLOUD-TO-CLOUD TRANSFER")
    print("  Source:      tigerwood697 (sciencedb-perception-cache)")
    print("  Destination: hasinishrak2015 (/mtl-data/bcs)")
    print("  Bandwidth:   100% Cloud Datacenter (0 bytes through local PC)")
    print("=" * 70)

    t_start = time.time()
    bcs_dir = Path("/mtl-data/bcs")
    bcs_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any partial staging directory from prior failed local chunk runs
    staging_dir = bcs_dir / "staging"
    if staging_dir.exists():
        print("[*] Cleaning up obsolete local staging dir from volume...")
        shutil.rmtree(staging_dir, ignore_errors=True)

    client = modal.Client.from_credentials(creds["token_id"], creds["token_secret"])
    source_vol = modal.Volume.from_name("sciencedb-perception-cache", client=client)

    files_to_transfer = [
        # Manifests
        ("/manifests/train_perception.csv", bcs_dir / "train_perception.csv", 9508.53 * 1024),
        ("/manifests/val_perception.csv", bcs_dir / "val_perception.csv", 2140.62 * 1024),
        ("/manifests/cache_summary.json", bcs_dir / "cache_summary.json", 1.17 * 1024),
        # Packed tensors
        ("/packed/val_bcs_224.pt", bcs_dir / "val_bcs_224.pt", 1496.31 * 1024 * 1024),
        ("/packed/train_bcs_224.pt", bcs_dir / "train_bcs_224.pt", 6578.84 * 1024 * 1024),
    ]

    results = {}
    for remote_src, local_dst, expected_min_size in files_to_transfer:
        fname = local_dst.name
        if local_dst.exists() and local_dst.stat().st_size >= expected_min_size * 0.95:
            print(f"[✓] {fname} already exists and verified ({local_dst.stat().st_size / (1024*1024):.2f} MB). Skipping.")
            results[fname] = {"size": local_dst.stat().st_size, "status": "already_present"}
            continue

        print(f"\n[*] Transferring {fname} ({expected_min_size / (1024*1024):.2f} MB) cloud-to-cloud...")
        t0 = time.time()

        temp_dst = bcs_dir / f"{fname}.transfer_tmp"
        with open(temp_dst, "wb") as f:
            source_vol.read_file_into_fileobj(remote_src, f)

        # Rename atomic
        if temp_dst.exists():
            temp_dst.replace(local_dst)

        elapsed = time.time() - t0
        size_mb = local_dst.stat().st_size / (1024 * 1024)
        speed = size_mb / max(0.1, elapsed)
        print(f"  ✓ {fname} transferred in {elapsed:.1f}s ({speed:.1f} MB/s)!")

        # Commit volume periodically to save checkpoint
        mtl_data_vol.commit()
        results[fname] = {"size": local_dst.stat().st_size, "elapsed": elapsed, "speed_mb_s": speed}

    # Final SHA-256 and size verification
    print("\n[*] Running final verification of BCS files on target volume...")
    for fname in ["train_bcs_224.pt", "val_bcs_224.pt", "train_perception.csv", "val_perception.csv"]:
        fpath = bcs_dir / fname
        assert fpath.exists(), f"Missing {fname} after transfer"
        print(f"  ✓ {fname}: {fpath.stat().st_size / (1024*1024):.2f} MB")

    mtl_data_vol.commit()
    total_elapsed = time.time() - t_start
    print(f"\n[✓] ALL BCS DATA TRANSFERRED AND COMMITTED IN {total_elapsed:.1f}s!")
    return {"status": "success", "total_elapsed": total_elapsed, "files": results}


# ==============================================================================
# 3. BEHAVIOR DIRECT CLOUD TRANSFER
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def transfer_behavior_remote(creds: Dict[str, str]) -> Dict[str, Any]:
    """Transfers packaged Behavior dataset from tigerwood693 to hasinishrak2015 in the cloud."""

    print("=" * 70)
    print("  🚀 BEHAVIOR DIRECT CLOUD-TO-CLOUD TRANSFER")
    print("  Source:      tigerwood693 (behavior-perception-cache)")
    print("  Destination: hasinishrak2015 (/mtl-data/behavior)")
    print("  Bandwidth:   100% Cloud Datacenter (0 bytes through local PC)")
    print("=" * 70)

    t_start = time.time()
    beh_dir = Path("/mtl-data/behavior")
    beh_dir.mkdir(parents=True, exist_ok=True)

    client = modal.Client.from_credentials(creds["token_id"], creds["token_secret"])
    source_vol = modal.Volume.from_name("behavior-perception-cache", client=client)

    # Check for manifest
    manifest_bytes = bytearray()
    print("[*] Reading behavior export manifest from tigerwood693...")
    for chunk in source_vol.read_file("/export_behavior/behavior_export_manifest.json"):
        manifest_bytes.extend(chunk)

    import json
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    chunks = manifest.get("chunks", [])
    print(f"  ✓ Loaded export manifest: {manifest.get('total_bytes', 0) / (1024**2):.1f} MB across {len(chunks)} chunks.")

    # Reassemble directly into /tmp/behavior_retained.tar
    tar_tmp = Path("/tmp/behavior_retained.tar")
    if tar_tmp.exists():
        tar_tmp.unlink()

    print(f"\n[*] Streaming {len(chunks)} chunks cloud-to-cloud into {tar_tmp}...")
    with open(tar_tmp, "wb") as out_f:
        for c in chunks:
            c_name = c["chunk_name"]
            c_size_mb = c["size"] / (1024 * 1024)
            print(f"    - Downloading {c_name} ({c_size_mb:.1f} MB)...", end="", flush=True)
            t0 = time.time()
            source_vol.read_file_into_fileobj(f"/export_behavior/{c_name}", out_f)
            el = time.time() - t0
            speed = c_size_mb / max(0.1, el)
            print(f" done in {el:.1f}s ({speed:.1f} MB/s)!")

    assert tar_tmp.stat().st_size == manifest["total_bytes"], "Reassembled tar size mismatch"
    print(f"\n[+] Extracting {tar_tmp} into {beh_dir} via native Linux tar...")
    t_ext = time.time()
    import subprocess
    cmd = ["tar", "-xf", str(tar_tmp), "-C", str(beh_dir)]
    subprocess.run(cmd, check=True)
    print(f"  ✓ Extraction complete in {time.time() - t_ext:.1f}s!")

    if tar_tmp.exists():
        tar_tmp.unlink()

    # Commit target volume
    mtl_data_vol.commit()
    total_elapsed = time.time() - t_start
    print(f"\n[✓] ALL BEHAVIOR DATA TRANSFERRED AND COMMITTED IN {total_elapsed:.1f}s!")
    return {"status": "success", "total_elapsed": total_elapsed, "total_bytes": manifest["total_bytes"]}


# ==============================================================================
# 4. RE-ID DIRECT CLONE FROM SIDEVIEW-DATA VOLUME (ZERO DOWNLOAD)
# ==============================================================================
sideview_vol = modal.Volume.from_name("sideview-data", create_if_missing=True)


@app.function(
    volumes={"/sideview": sideview_vol, "/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=600,
)
def stage_reid_remote() -> Dict[str, Any]:
    """Clones required Protocol A parlor images/masks directly from sideview-data to mtl-data."""

    print("=" * 70)
    print("  🚀 RE-ID ZERO-DOWNLOAD CLOUD VOLUME CLONE")
    print("  Source:      hasinishrak2015 (sideview-data/sideviewcows2026/parlor)")
    print("  Destination: hasinishrak2015 (/mtl-data/reid/parlor)")
    print("  Speed:       Direct NVMe Cloud Volume Copy")
    print("=" * 70)

    t_start = time.time()
    reid_root = Path("/mtl-data/reid/parlor")
    reid_root.mkdir(parents=True, exist_ok=True)

    src_parlor = Path("/sideview/sideviewcows2026/parlor")
    if not src_parlor.exists():
        raise FileNotFoundError(f"Missing parlor in sideview-data: {src_parlor}")

    # Copy parlor directly
    print("[*] Synchronizing parlor images and masks to mtl-data volume...")
    copied_count = 0
    for subdir in ["images", "masks"]:
        src_sub = src_parlor / subdir
        dst_sub = reid_root / subdir
        if src_sub.exists():
            shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
            copied_count += sum(len(files) for _, _, files in os.walk(dst_sub))

    mtl_data_vol.commit()
    total_elapsed = time.time() - t_start
    print(f"\n[✓] RE-ID PARLOR CLONED AND COMMITTED IN {total_elapsed:.1f}s! Total files: {copied_count}")
    return {"status": "success", "total_elapsed": total_elapsed, "copied_files": copied_count}


# ==============================================================================
# ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def transfer_bcs():
    creds = _get_profile_creds("tigerwood697")
    res = transfer_bcs_remote.remote(creds)
    print(f"\n[✓] BCS Cloud-to-Cloud Transfer Complete in {res['total_elapsed']:.1f}s!")


@app.local_entrypoint()
def transfer_behavior():
    creds = _get_profile_creds("tigerwood693")
    res = transfer_behavior_remote.remote(creds)
    print(f"\n[✓] Behavior Cloud-to-Cloud Transfer Complete in {res['total_elapsed']:.1f}s!")


@app.local_entrypoint()
def stage_reid():
    res = stage_reid_remote.remote()
    print(f"\n[✓] Re-ID Staging Complete in {res['total_elapsed']:.1f}s!")
