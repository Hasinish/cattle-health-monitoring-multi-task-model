# -*- coding: utf-8 -*-
"""Modal source-side packaging and chunking engine for BCS (profile: tigerwood697).

Runs in Modal profile 'tigerwood697' (volume: 'sciencedb-perception-cache')
to prepare monolithic binary tensors (train_bcs_224.pt, val_bcs_224.pt) into
resumable sequential chunks with SHA-256 verification.
Strictly excludes test_bcs_224.pt and test_perception.csv.

Configured for minimal compute cost: cpu=2.0, memory=4096 MB, NO GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Scoped strictly to profile tigerwood697
bcs_vol = modal.Volume.from_name("sciencedb-perception-cache", create_if_missing=False)

export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm")
)

app = modal.App("mtl-export-bcs", image=export_image)


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute SHA-256 checksum with 16MB stream buffers."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


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
