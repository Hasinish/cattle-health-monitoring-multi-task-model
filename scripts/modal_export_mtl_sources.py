# -*- coding: utf-8 -*-
"""Modal source-side packaging and chunking engine for MTL dataset staging.

Runs in source Modal profiles to prepare large, sequential, resumable chunks:
- BCS (Profile 'tigerwood697', Volume 'sciencedb-perception-cache'):
    Chunks train_bcs_224.pt and val_bcs_224.pt into 512MB-1GB parts with SHA-256.
    Strictly excludes test_bcs_224.pt and test_perception.csv.
- Behavior (Profile 'tigerwood693', Volume 'behavior-perception-cache'):
    Packages ONLY the 4,271 retained Train/Val sequences into an uncompressed tar,
    splits into 512MB-1GB parts with SHA-256.
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

# Persistent volumes scoped to source environments
bcs_vol = modal.Volume.from_name("sciencedb-perception-cache", create_if_missing=False)
behavior_vol = modal.Volume.from_name("behavior-perception-cache", create_if_missing=False)

export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm")
)

app = modal.App("mtl-export-sources", image=export_image)


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute SHA-256 checksum with 16MB stream buffers."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# BCS SOURCE PREPARATION (Profile: tigerwood697)
# ==============================================================================
@app.function(
    volumes={"/cache": bcs_vol},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def prepare_bcs_export_remote(chunk_size_mb: int = 1024) -> Dict[str, Any]:
    """Prepares chunked, resumable export for BCS on tigerwood697.

    Splits /cache/packed/train_bcs_224.pt and /cache/packed/val_bcs_224.pt into
    fixed-size binary parts with verified SHA-256 hashes.
    """
    print("=" * 70)
    print("  BCS SOURCE EXPORT PREPARATION (tigerwood697)")
    print("=" * 70)
    sys.stdout.flush()

    chunk_bytes = chunk_size_mb * 1024 * 1024
    export_dir = Path("/cache/export_bcs")
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = export_dir / "bcs_export_manifest.json"

    # Targets to export (Strictly NO test data)
    source_files = {
        "train_bcs_224.pt": Path("/cache/packed/train_bcs_224.pt"),
        "val_bcs_224.pt": Path("/cache/packed/val_bcs_224.pt"),
    }
    manifest_sources = {
        "train_perception.csv": Path("/cache/manifests/train_perception.csv"),
        "val_perception.csv": Path("/cache/manifests/val_perception.csv"),
    }

    # Verify source files exist
    for name, path in {**source_files, **manifest_sources}.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required BCS source file: {path}")

    # Check if existing manifest is complete and valid
    if manifest_file.exists():
        try:
            with open(manifest_file, "r") as f:
                existing = json.load(f)
            all_chunks_valid = True
            for file_key, file_info in existing.get("files", {}).items():
                for c in file_info.get("chunks", []):
                    c_path = export_dir / c["chunk_name"]
                    if not c_path.exists() or c_path.stat().st_size != c["size"]:
                        all_chunks_valid = False
                        break
            if all_chunks_valid and existing.get("chunk_size_mb") == chunk_size_mb:
                print(f"[*] Valid BCS export already exists ({len(existing['files'])} files, {existing['total_bytes'] / (1024**3):.2f} GB)")
                return existing
        except Exception as e:
            print(f"[!] Warning reading existing manifest: {e}. Regenerating...")

    files_info: Dict[str, Any] = {}
    total_export_bytes = 0

    # Copy manifests directly to export_dir
    manifest_info_dict: Dict[str, Any] = {}
    for m_name, m_path in manifest_sources.items():
        dest_m = export_dir / m_name
        if not dest_m.exists() or dest_m.stat().st_size != m_path.stat().st_size:
            shutil.copy2(m_path, dest_m)
        sha = _compute_sha256(dest_m)
        manifest_info_dict[m_name] = {
            "size": dest_m.stat().st_size,
            "sha256": sha,
            "filename": m_name,
        }

    for file_key, src_path in source_files.items():
        src_size = src_path.stat().st_size
        total_export_bytes += src_size
        num_chunks = math.ceil(src_size / chunk_bytes)
        print(f"\n[*] Processing {file_key} ({src_size / (1024**3):.2f} GB, {num_chunks} chunks of {chunk_size_mb} MB)...")
        sys.stdout.flush()

        chunks_list: List[Dict[str, Any]] = []
        with open(src_path, "rb") as in_f:
            for idx in range(num_chunks):
                chunk_name = f"{src_path.stem}.part_{idx:04d}.bin"
                chunk_path = export_dir / chunk_name
                expected_size = min(chunk_bytes, src_size - (idx * chunk_bytes))

                if chunk_path.exists() and chunk_path.stat().st_size == expected_size:
                    chunk_sha = _compute_sha256(chunk_path)
                    print(f"    Chunk {idx + 1}/{num_chunks}: {chunk_name} (cached, sha={chunk_sha[:10]}...)")
                else:
                    in_f.seek(idx * chunk_bytes)
                    data = in_f.read(expected_size)
                    with open(chunk_path, "wb") as out_f:
                        out_f.write(data)
                    chunk_sha = hashlib.sha256(data).hexdigest()
                    print(f"    Chunk {idx + 1}/{num_chunks}: {chunk_name} ({len(data) / (1024**2):.1f} MB, sha={chunk_sha[:10]}...)")
                sys.stdout.flush()

                chunks_list.append({
                    "index": idx,
                    "chunk_name": chunk_name,
                    "size": expected_size,
                    "sha256": chunk_sha,
                })

        # Calculate whole-file SHA-256 for final target integrity check
        whole_sha = _compute_sha256(src_path)
        files_info[file_key] = {
            "source_path": str(src_path),
            "total_size": src_size,
            "sha256": whole_sha,
            "num_chunks": len(chunks_list),
            "chunks": chunks_list,
        }

    manifest_data = {
        "task": "bcs",
        "source_profile": "tigerwood697",
        "source_volume": "sciencedb-perception-cache",
        "chunk_size_mb": chunk_size_mb,
        "total_bytes": total_export_bytes,
        "files": files_info,
        "manifests": manifest_info_dict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)

    bcs_vol.commit()
    print(f"\n[+] BCS Export Ready: {total_export_bytes / (1024**3):.2f} GB across {sum(len(v['chunks']) for v in files_info.values())} chunks.")
    return manifest_data


@app.function(
    volumes={"/cache": bcs_vol},
    cpu=1.0,
    memory=2048,
    timeout=300,
)
def cleanup_bcs_export_remote() -> None:
    """Removes temporary export chunks on tigerwood697 to reclaim volume storage."""
    export_dir = Path("/cache/export_bcs")
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
        bcs_vol.commit()
        print("[+] Cleaned up /cache/export_bcs on tigerwood697")


# ==============================================================================
# BEHAVIOR SOURCE PREPARATION (Profile: tigerwood693)
# ==============================================================================
@app.function(
    volumes={"/cache": behavior_vol},
    cpu=2.0,
    memory=4096,
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

    # Build uncompressed tar archive on ephemeral disk
    tar_tmp = Path("/tmp/behavior_retained.tar")
    if tar_tmp.exists():
        tar_tmp.unlink()

    print(f"[*] Packaging {len(allowed_sequences)} sequence directories into {tar_tmp}...")
    sys.stdout.flush()
    t0 = time.time()

    manifest_files_to_pack = [
        "retained_train.csv",
        "retained_val.csv",
        "perception_summary.json",
        "perception_manifest.csv",
    ]

    with tarfile.open(tar_tmp, "w") as tar:
        # Add manifest files first
        for mf in manifest_files_to_pack:
            mf_path = prod_dir / mf
            if mf_path.exists():
                tar.add(mf_path, arcname=mf)

        # Add sequence directories
        packed_count = 0
        for seq_id in sorted(allowed_sequences):
            seq_dir = prod_dir / seq_id
            if seq_dir.exists() and seq_dir.is_dir():
                tar.add(seq_dir, arcname=seq_id)
                packed_count += 1
                if packed_count % 500 == 0 or packed_count == len(allowed_sequences):
                    print(f"    Packed {packed_count}/{len(allowed_sequences)} sequences ({packed_count / len(allowed_sequences) * 100:.1f}%)...")
                    sys.stdout.flush()
            else:
                raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")

    pack_duration = time.time() - t0
    tar_size = tar_tmp.stat().st_size
    tar_sha = _compute_sha256(tar_tmp)
    print(f"[+] Tar packaging completed in {pack_duration:.1f}s ({tar_size / (1024**2):.1f} MB, sha={tar_sha[:10]}...)")
    sys.stdout.flush()

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
