# -*- coding: utf-8 -*-
"""
scripts/modal_export_behavior_test.py — Fast Parallel NVMe Behavior Test Packager
================================================================================
Packages the 780 frozen retained test sequences on 'tigerwood693' into an uncompressed
tar archive using 64-worker parallel NVMe pre-staging and native Linux tar.
Runtime: ~20-30 seconds.
Minimal compute: cpu=4.0, memory=8192 MB, NO GPU.

Usage:
    modal run --profile tigerwood693 scripts/modal_export_behavior_test.py::export_test_behavior
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent volume on tigerwood693
behavior_vol = modal.Volume.from_name("behavior-perception-cache", create_if_missing=False)

export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm", "pandas")
)

app = modal.App("mtl-export-behavior-test", image=export_image)


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@app.function(
    volumes={"/cache": behavior_vol},
    cpu=4.0,
    memory=8192,
    timeout=900,
)
def export_test_behavior_remote() -> Dict[str, Any]:
    print("=" * 70)
    print("  FAST PARALLEL PACKAGING OF BEHAVIOR TEST SEQUENCES (tigerwood693)")
    print("=" * 70)

    t0 = time.time()
    source_dir = Path("/cache/production_test")
    assert source_dir.exists(), f"Source directory missing: {source_dir}"

    retained_csv = source_dir / "retained_test.csv"
    assert retained_csv.exists(), f"Missing retained_test.csv at {retained_csv}"

    import pandas as pd
    df_retained = pd.read_csv(retained_csv)
    n_retained = len(df_retained)
    assert n_retained == 780, f"Expected exactly 780 retained test sequences, found {n_retained}"
    print(f"[*] Verified retained_test.csv: exactly {n_retained} sequences.")

    export_dir = Path("/cache/export_test")
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = export_dir / "behavior_test_manifest.json"
    tar_volume_path = export_dir / "behavior_test.tar"

    # Step 1: Pre-stage 780 sequences to fast local NVMe ephemeral disk
    stage_dir = Path("/tmp/stage_behavior_test")
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata files
    for m_file in ["retained_test.csv", "failed_test.csv", "perception_summary_test.json"]:
        p = source_dir / m_file
        if p.exists():
            shutil.copy2(p, stage_dir / m_file)

    seq_ids = [str(r["sample_id"]) for _, r in df_retained.iterrows()]
    print(f"[*] Pre-staging {len(seq_ids)} sequences to local NVMe via 64 workers...")
    t_stage_0 = time.time()

    def _copy_sequence(sid: str) -> bool:
        src = source_dir / sid
        dst = stage_dir / sid
        if not src.exists():
            return False
        shutil.copytree(src, dst)
        return True

    done_count = 0
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(_copy_sequence, sid): sid for sid in seq_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                ok = future.result()
                if not ok:
                    errors.append(f"Missing {sid}")
            except Exception as e:
                errors.append(f"{sid}: {e}")
            done_count += 1
            if done_count % 200 == 0 or done_count == len(seq_ids):
                el = time.time() - t_stage_0
                rate = done_count / max(0.1, el)
                print(f"    -> [{done_count:4d}/{len(seq_ids)}] staged in {el:4.1f}s ({rate:5.1f} seq/s)")

    if errors:
        raise RuntimeError(f"Errors copying test sequences: {errors[:5]}")

    stage_duration = time.time() - t_stage_0
    print(f"[+] All {n_retained} sequences pre-staged to local NVMe in {stage_duration:.1f}s ({n_retained/stage_duration:.1f} seq/s)!")

    # Step 2: Native Linux tar on local NVMe disk
    tar_tmp = Path("/tmp/behavior_test.tar")
    if tar_tmp.exists():
        tar_tmp.unlink()

    print("[*] Creating tar archive via native Linux tar...")
    t_tar = time.time()
    cmd = ["tar", "-cf", str(tar_tmp), "-C", str(stage_dir), "."]
    subprocess.run(cmd, check=True)
    tar_duration = time.time() - t_tar
    tar_size = tar_tmp.stat().st_size
    print(f"[+] Native Linux tar complete in {tar_duration:.2f}s ({tar_size / (1024**2):.2f} MB, {tar_size / (1024**2) / max(0.1, tar_duration):.1f} MB/s)!")

    # Step 3: Compute SHA-256 on local NVMe
    print("[*] Computing SHA-256 checksum...")
    sha256_hash = _compute_sha256(tar_tmp)
    print(f"[+] SHA-256: {sha256_hash}")

    # Step 4: Stream tar from local NVMe to persistent volume with 16MB stream buffer
    print(f"[*] Streaming {tar_size / (1024**2):.2f} MB to persistent volume {tar_volume_path}...")
    t_stream = time.time()
    with open(tar_tmp, "rb") as in_f, open(tar_volume_path, "wb") as out_f:
        shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)
    stream_duration = time.time() - t_stream
    print(f"[+] Volume write complete in {stream_duration:.1f}s ({(tar_size / (1024**2)) / max(0.1, stream_duration):.1f} MB/s)!")

    # Clean up ephemeral NVMe
    shutil.rmtree(stage_dir, ignore_errors=True)
    tar_tmp.unlink()

    metadata = {
        "sequence_count": n_retained,
        "archive_name": "behavior_test.tar",
        "size_bytes": tar_size,
        "size_mb": round(tar_size / (1024 * 1024), 2),
        "sha256": sha256_hash,
        "total_time_sec": round(time.time() - t0, 2),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    behavior_vol.commit()
    print(f"[OK] Total packaging completed and committed in {time.time() - t0:.1f}s!")
    return metadata


@app.local_entrypoint()
def export_test_behavior():
    res = export_test_behavior_remote.remote()
    print("\n[OK] Export complete:")
    print(f"    Sequences: {res['sequence_count']}")
    print(f"    Size     : {res['size_mb']} MB")
    print(f"    SHA-256  : {res['sha256']}")
    print(f"    Runtime  : {res['total_time_sec']}s")
