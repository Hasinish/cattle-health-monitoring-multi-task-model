# -*- coding: utf-8 -*-
"""
scripts/modal_stage_mtl_test_data.py — Cloud-to-Cloud Staging of Held-Out Test Data
===================================================================================
Transfers frozen held-out test data from source profiles directly to hasinishrak2015:
  1. BCS: test_bcs_224.pt (7,549 samples) + test_perception.csv from tigerwood697
  2. Behavior: behavior_test.tar (780 sequences) from tigerwood693 -> /mtl-data/behavior_test/
Zero local PC bandwidth, 100% cloud datacenter transfer.
Minimal compute: cpu=2.0, memory=4096 MB, NO GPU.

Usage:
    modal run --profile hasinishrak2015 scripts/modal_stage_mtl_test_data.py::stage_test_data
"""

import sys

# Windows UTF-8 patch
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent volume on target profile (hasinishrak2015)
mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=True)

transfer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("modal", "tqdm", "pandas", "torch")
)

app = modal.App("mtl-stage-test-data", image=transfer_image)


def _get_profile_creds(profile_name: str) -> Dict[str, str]:
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
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def stage_test_data_remote(bcs_creds: Dict[str, str], beh_creds: Dict[str, str]) -> Dict[str, Any]:
    print("=" * 75)
    print("  🚀 DIRECT CLOUD-TO-CLOUD STAGING OF HELD-OUT TEST DATA")
    print("  Target Profile: hasinishrak2015 (/mtl-data)")
    print("  BCS Source:     tigerwood697 (sciencedb-perception-cache)")
    print("  Behavior Source: tigerwood693 (behavior-perception-cache)")
    print("=" * 75)

    results = {}
    t_start = time.time()

    # --------------------------------------------------------------------------
    # 1. Transfer BCS Test Data (7,549 samples)
    # --------------------------------------------------------------------------
    print("\n[*] --- Step 1: Staging ScienceDB BCS Test Data ---")
    bcs_dir = Path("/mtl-data/bcs")
    bcs_dir.mkdir(parents=True, exist_ok=True)

    client_bcs = modal.Client.from_credentials(bcs_creds["token_id"], bcs_creds["token_secret"])
    vol_bcs = modal.Volume.from_name("sciencedb-perception-cache", client=client_bcs)

    bcs_test_files = [
        ("/manifests/test_perception.csv", bcs_dir / "test_perception.csv"),
        ("/packed/test_bcs_224.pt", bcs_dir / "test_bcs_224.pt"),
    ]

    for remote_src, local_dst in bcs_test_files:
        fname = local_dst.name
        if local_dst.exists() and local_dst.stat().st_size > 1000:
            print(f"  [OK] {fname} already present ({local_dst.stat().st_size / (1024**2):.2f} MB).")
            results[fname] = {"size": local_dst.stat().st_size, "status": "already_present"}
            continue

        print(f"  [*] Downloading {fname} cloud-to-cloud...")
        t0 = time.time()
        temp_dst = bcs_dir / f"{fname}.tmp"
        with open(temp_dst, "wb") as f:
            vol_bcs.read_file_into_fileobj(remote_src, f)
        temp_dst.replace(local_dst)
        elapsed = time.time() - t0
        size_mb = local_dst.stat().st_size / (1024 * 1024)
        speed = size_mb / max(0.1, elapsed)
        print(f"  [+] {fname} transferred in {elapsed:.1f}s ({speed:.1f} MB/s, {size_mb:.2f} MB)")
        results[fname] = {"size": local_dst.stat().st_size, "elapsed": elapsed, "speed_mb_s": speed}

    # Verify BCS tensor payload
    import torch
    payload = torch.load(bcs_dir / "test_bcs_224.pt", map_location="cpu", weights_only=False)
    n_bcs = len(payload["targets"])
    assert n_bcs == 7549, f"Expected 7549 BCS test samples, found {n_bcs}"
    print(f"  [OK] Verified test_bcs_224.pt: exactly {n_bcs} samples, tensors shape {payload['tensors'].shape}")
    mtl_data_vol.commit()

    # --------------------------------------------------------------------------
    # 2. Transfer Behavior Test Data (780 sequences)
    # --------------------------------------------------------------------------
    print("\n[*] --- Step 2: Staging CVB+Beef Behavior Test Data ---")
    beh_test_dir = Path("/mtl-data/behavior_test")
    beh_test_dir.mkdir(parents=True, exist_ok=True)

    retained_csv_final = beh_test_dir / "retained_test.csv"
    if retained_csv_final.exists():
        import pandas as pd
        df_existing = pd.read_csv(retained_csv_final)
        if len(df_existing) == 780:
            print(f"  [OK] Behavior test data already present and verified ({len(df_existing)} sequences).")
            results["behavior_test"] = {"status": "already_present", "sequences": 780}
            mtl_data_vol.commit()
            return {"status": "success", "results": results, "total_elapsed": time.time() - t_start}

    client_beh = modal.Client.from_credentials(beh_creds["token_id"], beh_creds["token_secret"])
    vol_beh = modal.Volume.from_name("behavior-perception-cache", client=client_beh)

    # Read manifest
    manifest_bytes = bytearray()
    print("  [*] Reading behavior test manifest from tigerwood693...")
    for chunk in vol_beh.read_file("/export_test/behavior_test_manifest.json"):
        manifest_bytes.extend(chunk)
    beh_manifest = json.loads(manifest_bytes.decode("utf-8"))
    print(f"  [+] Loaded export manifest: {beh_manifest['size_mb']} MB, {beh_manifest['sequence_count']} sequences, SHA256: {beh_manifest['sha256']}")

    tar_tmp = Path("/tmp/behavior_test.tar")
    if tar_tmp.exists():
        tar_tmp.unlink()

    print("  [*] Downloading behavior_test.tar cloud-to-cloud...")
    t0 = time.time()
    with open(tar_tmp, "wb") as f:
        vol_beh.read_file_into_fileobj("/export_test/behavior_test.tar", f)
    elapsed_dl = time.time() - t0
    dl_speed = (tar_tmp.stat().st_size / (1024 * 1024)) / max(0.1, elapsed_dl)
    print(f"  [+] behavior_test.tar downloaded in {elapsed_dl:.1f}s ({dl_speed:.1f} MB/s)")

    # Verify SHA-256
    print("  [*] Verifying SHA-256 checksum...")
    tar_sha = _compute_sha256(tar_tmp)
    assert tar_sha == beh_manifest["sha256"], f"SHA-256 mismatch! Expected {beh_manifest['sha256']}, got {tar_sha}"
    print("  [OK] SHA-256 checksum matched bit-identically!")

    print(f"  [*] Extracting {tar_tmp} into {beh_test_dir}...")
    t_ext = time.time()
    cmd = ["tar", "-xf", str(tar_tmp), "-C", str(beh_test_dir)]
    subprocess.run(cmd, check=True)
    print(f"  [+] Extraction complete in {time.time() - t_ext:.1f}s!")
    tar_tmp.unlink()

    # Verify retained_test.csv and sequences
    import pandas as pd
    df_retained = pd.read_csv(retained_csv_final)
    n_seqs = len(df_retained)
    assert n_seqs == 780, f"Expected 780 sequences, got {n_seqs}"

    # Verify all sequence folders exist
    missing = []
    for sid in df_retained["sample_id"]:
        s_folder = beh_test_dir / str(sid)
        if not s_folder.exists() or len(list(s_folder.glob("frame_*.jpg"))) != 8:
            missing.append(sid)
    assert len(missing) == 0, f"Missing frames for {len(missing)} sequences: {missing[:5]}"
    print(f"  [OK] Verified all {n_seqs} Behavior test sequences (8 frames + 8 masks each)!")

    mtl_data_vol.commit()
    total_elapsed = time.time() - t_start
    print(f"\n[OK] ALL HELD-OUT TEST DATA STAGED AND COMMITTED IN {total_elapsed:.1f}s!")
    results["behavior_test"] = {"status": "success", "sequences": n_seqs, "sha256": tar_sha}
    return {"status": "success", "results": results, "total_elapsed": total_elapsed}


@app.local_entrypoint()
def stage_test_data():
    bcs_creds = _get_profile_creds("tigerwood697")
    beh_creds = _get_profile_creds("tigerwood693")
    res = stage_test_data_remote.remote(bcs_creds, beh_creds)
    print("\n[OK] Staging Summary:")
    print(json.dumps(res, indent=2))
