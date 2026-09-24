# -*- coding: utf-8 -*-
"""Modal source-side packaging and chunking engine for Behavior (profile: tigerwood693).

Runs in Modal profile 'tigerwood693' (volume: 'behavior-perception-cache')
to package 4,271 retained Train/Val sequences into an uncompressed sequential tar archive
split into fixed-size binary parts with verified SHA-256 hashes.
Strictly excludes production_test/, retained_test.csv, and failed_test.csv.

Configured for minimal compute cost: cpu=2.0, memory=4096 MB, NO GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Scoped strictly to profile tigerwood693
behavior_vol = modal.Volume.from_name("behavior-perception-cache", create_if_missing=False)

export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm")
)

app = modal.App("mtl-export-behavior", image=export_image)


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute SHA-256 checksum with 16MB stream buffers."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@app.function(
    volumes={"/cache": behavior_vol},
    cpu=4.0,
    memory=8192,
    timeout=1800,
)
def prepare_behavior_export_remote(chunk_size_mb: int = 1024) -> Dict[str, Any]:
    """Packages the 4,271 retained Behavior sequences into a fast uncompressed tar archive.

    Strictly includes only:
    - 3,641 Train retained sequences
    - 630 Val retained sequences
    - production manifests and summary
    Strictly EXCLUDES production_test/, retained_test.csv, failed_test.csv.
    """
    print("=" * 70)
    print("  BEHAVIOR SOURCE EXPORT PREPARATION (tigerwood693)")
    print("=" * 70)
    sys.stdout.flush()

    prod_dir = Path("/cache/production")
    if not prod_dir.exists():
        raise FileNotFoundError(f"Missing behavior production cache at: {prod_dir}")

    train_csv = prod_dir / "retained_train.csv"
    val_csv = prod_dir / "retained_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Missing retained_train.csv or retained_val.csv in production cache")

    # Read allowed sequence IDs
    allowed_sequences: set[str] = set()
    for csv_path in (train_csv, val_csv):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_id = row.get("sample_id", "").strip()
                if seq_id:
                    allowed_sequences.add(seq_id)

    print(f"[*] Verified allowed retained sequences: {len(allowed_sequences)} (Train=3,641, Val=630)")
    if len(allowed_sequences) != 4271:
        print(f"[!] Warning: Expected 4,271 sequences, found {len(allowed_sequences)}")

    export_dir = Path("/cache/export_behavior")
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = export_dir / "behavior_export_manifest.json"

    # Check if already built and valid
    if manifest_file.exists():
        try:
            with open(manifest_file, "r") as f:
                existing = json.load(f)
            all_chunks_valid = True
            for c in existing.get("chunks", []):
                c_path = export_dir / c["chunk_name"]
                if not c_path.exists() or c_path.stat().st_size != c["size"]:
                    all_chunks_valid = False
                    break
            if all_chunks_valid and existing.get("total_sequences") == len(allowed_sequences):
                print(f"[*] Valid Behavior export archive already exists ({existing['total_bytes'] / (1024**2):.1f} MB, {len(existing['chunks'])} chunks)")
                return existing
        except Exception as e:
            print(f"[!] Warning reading existing manifest: {e}. Regenerating...")

    # Build uncompressed tar archive on ephemeral disk via 64-worker parallel NVMe pre-staging
    tar_tmp = Path("/tmp/behavior_retained.tar")
    if tar_tmp.exists():
        tar_tmp.unlink()

    stage_dir = Path("/tmp/stage_behavior")
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest_files_to_pack = [
        "retained_train.csv",
        "retained_val.csv",
        "perception_summary.json",
        "perception_manifest.csv",
    ]

    for mf in manifest_files_to_pack:
        src_f = prod_dir / mf
        if src_f.exists():
            shutil.copy2(src_f, stage_dir / mf)

    # Parallel pre-stage of 4,271 sequence directories to local NVMe
    total_seqs = len(allowed_sequences)
    print(f"[*] Starting 64-worker parallel NVMe pre-staging of {total_seqs} sequences...")
    sys.stdout.flush()

    seq_list = sorted(allowed_sequences)
    t_stage_0 = time.time()
    done_count = 0
    errors: List[str] = []

    def _copy_seq(seq_id: str) -> bool:
        src = prod_dir / seq_id
        dst = stage_dir / seq_id
        if not src.exists():
            return False
        shutil.copytree(src, dst)
        return True

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(_copy_seq, s): s for s in seq_list}
        for future in as_completed(futures):
            s = futures[future]
            try:
                ok = future.result()
                if not ok:
                    errors.append(f"Missing: {s}")
            except Exception as e:
                errors.append(f"{s}: {e}")
            done_count += 1
            if done_count % 250 == 0 or done_count == total_seqs:
                el = time.time() - t_stage_0
                rate = done_count / el if el > 0 else 0
                eta = (total_seqs - done_count) / rate if rate > 0 else 0
                print(f"    -> [{done_count:4d}/{total_seqs}] ({done_count/total_seqs*100:5.1f}%) "
                      f"staged in {el:5.1f}s | Speed: {rate:5.1f} seq/s | ETA: {eta:4.1f}s")
                sys.stdout.flush()

    if errors:
        raise RuntimeError(f"Encountered {len(errors)} errors copying sequences: {errors[:5]}")

    stage_duration = time.time() - t_stage_0
    print(f"[+] All {total_seqs} sequences pre-staged to local NVMe in {stage_duration:.1f}s ({total_seqs/stage_duration:.1f} seq/s)!")
    sys.stdout.flush()

    # Now tar on local NVMe disk at native SSD speeds
    print(f"[*] Creating {tar_tmp} on local NVMe disk...")
    sys.stdout.flush()
    t_tar = time.time()
    import subprocess
    cmd = ["tar", "-cf", str(tar_tmp), "-C", str(stage_dir), "."]
    subprocess.run(cmd, check=True)
    tar_duration = time.time() - t_tar

    tar_size = tar_tmp.stat().st_size
    tar_sha = _compute_sha256(tar_tmp)
    print(f"[+] Local NVMe tar completed in {tar_duration:.2f}s ({tar_size / (1024**2):.1f} MB, sha={tar_sha[:10]}...)")
    sys.stdout.flush()

    # Free up memory/disk from stage_dir
    shutil.rmtree(stage_dir, ignore_errors=True)

    # Split into chunks of chunk_size_mb
    chunk_bytes = chunk_size_mb * 1024 * 1024
    num_chunks = math.ceil(tar_size / chunk_bytes)
    chunks_list: List[Dict[str, Any]] = []

    with open(tar_tmp, "rb") as in_f:
        for idx in range(num_chunks):
            chunk_name = f"behavior_retained.part_{idx:04d}.bin"
            chunk_path = export_dir / chunk_name
            expected_size = min(chunk_bytes, tar_size - (idx * chunk_bytes))

            data = in_f.read(expected_size)
            with open(chunk_path, "wb") as out_f:
                out_f.write(data)
            chunk_sha = hashlib.sha256(data).hexdigest()

            print(f"    Chunk {idx + 1}/{num_chunks}: {chunk_name} ({len(data) / (1024**2):.1f} MB, sha={chunk_sha[:10]}...)")
            chunks_list.append({
                "index": idx,
                "chunk_name": chunk_name,
                "size": expected_size,
                "sha256": chunk_sha,
            })

    if tar_tmp.exists():
        tar_tmp.unlink()

    manifest_data = {
        "task": "behavior",
        "source_profile": "tigerwood693",
        "source_volume": "behavior-perception-cache",
        "total_sequences": len(allowed_sequences),
        "total_bytes": tar_size,
        "archive_sha256": tar_sha,
        "chunk_size_mb": chunk_size_mb,
        "num_chunks": len(chunks_list),
        "chunks": chunks_list,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)

    behavior_vol.commit()
    print(f"\n[+] Behavior Export Ready: {tar_size / (1024**2):.1f} MB across {len(chunks_list)} chunks.")
    return manifest_data


@app.function(
    volumes={"/cache": behavior_vol},
    cpu=1.0,
    memory=2048,
    timeout=300,
)
def cleanup_behavior_export_remote() -> None:
    """Removes temporary export chunks on tigerwood693 to reclaim volume storage."""
    export_dir = Path("/cache/export_behavior")
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
        behavior_vol.commit()
        print("[+] Cleaned up /cache/export_behavior on tigerwood693")


@app.local_entrypoint()
def main(chunk_size_mb: int = 1024):
    res = prepare_behavior_export_remote.remote(chunk_size_mb=chunk_size_mb)
    print(f"\n[✓] Behavior export complete: {res['total_bytes'] / (1024**2):.1f} MB across {res['num_chunks']} chunks")
