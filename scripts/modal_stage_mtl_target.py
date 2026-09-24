# -*- coding: utf-8 -*-
"""Modal destination-side reassembly, direct Re-ID hydration, and verification pipeline.

Runs in target Modal profile 'hasinishrak2015' attaching:
- 'mtl-data' mounted at /mtl-data
- 'mtl-checkpoints' mounted at /mtl-checkpoints

Target Structure:
/mtl-data/
    bcs/
        train_bcs_224.pt
        val_bcs_224.pt
        train_perception.csv
        val_perception.csv
    behavior/
        retained_train.csv
        retained_val.csv
        perception_summary.json
        perception_manifest.csv
        <sequence_id>/
            frame_000.jpg ... frame_007.jpg
            mask_000.png ... mask_007.png
            perception_metadata.json
    reid/
        parlor/
            images/<cow_id>/<img_file>
            masks/<cow_id>/<mask_file>
    staging_manifest.json

Configured for minimal compute cost: cpu=2.0, memory=4096 MB, NO GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Persistent volumes scoped to profile hasinishrak2015
mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=True)
mtl_checkpoints_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=True)

target_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "pandas", "pillow", "tqdm")
    .add_local_file(
        str(REPO_ROOT / "scripts" / "fast_download_sideviewcows.py"),
        remote_path="/root/scripts/fast_download_sideviewcows.py",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
)

app = modal.App("mtl-stage-target", image=target_image)


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute SHA-256 checksum with 16MB stream buffers."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# TASK A: BCS REASSEMBLY (Profile: hasinishrak2015)
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def reassemble_bcs_remote(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Reassembles BCS monolithic tensors from uploaded chunks in /mtl-data/bcs/staging."""
    import torch

    print("=" * 70)
    print("  BCS TARGET REASSEMBLY & INTEGRITY VERIFICATION")
    print("=" * 70)
    sys.stdout.flush()

    bcs_root = Path("/mtl-data/bcs")
    staging_dir = bcs_root / "staging"
    bcs_root.mkdir(parents=True, exist_ok=True)

    files_manifest = manifest.get("files", {})
    verified_files = {}

    for file_key, file_info in files_manifest.items():
        expected_sha = file_info["sha256"]
        expected_size = file_info["total_size"]
        dest_path = bcs_root / file_key

        if dest_path.exists() and dest_path.stat().st_size == expected_size:
            actual_sha = _compute_sha256(dest_path)
            if actual_sha == expected_sha:
                print(f"[*] {file_key} already fully assembled and verified (sha={actual_sha[:10]}...)")
                verified_files[file_key] = {"size": expected_size, "sha256": actual_sha}
                continue

        print(f"\n[*] Reassembling {file_key} ({expected_size / (1024**3):.2f} GB from {len(file_info['chunks'])} chunks)...")
        sys.stdout.flush()

        temp_dest = bcs_root / f"{file_key}.assembling"
        with open(temp_dest, "wb") as out_f:
            for c in file_info["chunks"]:
                c_name = c["chunk_name"]
                c_path = staging_dir / c_name
                if not c_path.exists():
                    raise FileNotFoundError(f"Missing chunk on volume: {c_path}")
                if c_path.stat().st_size != c["size"]:
                    raise ValueError(f"Chunk size mismatch for {c_name}: expected {c['size']}, got {c_path.stat().st_size}")

                with open(c_path, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)

        actual_sha = _compute_sha256(temp_dest)
        if actual_sha != expected_sha:
            temp_dest.unlink()
            raise ValueError(f"SHA-256 mismatch for {file_key}: expected {expected_sha}, got {actual_sha}")

        if dest_path.exists():
            dest_path.unlink()
        temp_dest.rename(dest_path)
        print(f"[+] Successfully assembled {file_key} ({dest_path.stat().st_size / (1024**3):.2f} GB, sha={actual_sha[:10]}...)")
        verified_files[file_key] = {"size": expected_size, "sha256": actual_sha}

        # Clean up chunk files for this tensor
        for c in file_info["chunks"]:
            c_path = staging_dir / c["chunk_name"]
            if c_path.exists():
                c_path.unlink()

    # Move manifests
    for m_name, m_info in manifest.get("manifests", {}).items():
        src_m = staging_dir / m_name
        dest_m = bcs_root / m_name
        if src_m.exists():
            shutil.copy2(src_m, dest_m)
            src_m.unlink()
        elif not dest_m.exists():
            raise FileNotFoundError(f"Missing BCS manifest file: {m_name}")
        verified_files[m_name] = {"size": dest_m.stat().st_size, "sha256": _compute_sha256(dest_m)}

    # Remove staging directory if empty
    if staging_dir.exists() and not list(staging_dir.iterdir()):
        staging_dir.rmdir()

    # Verify PyTorch loadability
    print("\n[*] Verifying PyTorch tensor loadability...")
    sys.stdout.flush()
    train_data = torch.load(bcs_root / "train_bcs_224.pt", weights_only=False)
    val_data = torch.load(bcs_root / "val_bcs_224.pt", weights_only=False)
    print(f"    train_bcs_224.pt keys: {list(train_data.keys())} (images: {train_data['images'].shape})")
    print(f"    val_bcs_224.pt keys:   {list(val_data.keys())} (images: {val_data['images'].shape})")
    del train_data, val_data

    # Assert test data absent
    assert not (bcs_root / "test_bcs_224.pt").exists(), "CRITICAL: test_bcs_224.pt found in MTL data volume!"
    assert not (bcs_root / "test_perception.csv").exists(), "CRITICAL: test_perception.csv found in MTL data volume!"

    mtl_data_vol.commit()
    print("[+] BCS Assembly Complete and Verified ✅")
    return {"status": "SUCCESS", "verified_files": verified_files}


# ==============================================================================
# TASK B: BEHAVIOR REASSEMBLY (Profile: hasinishrak2015)
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def reassemble_behavior_remote(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Reassembles Behavior uncompressed tar archive and extracts 4,271 sequences to /mtl-data/behavior."""
    print("=" * 70)
    print("  BEHAVIOR TARGET REASSEMBLY & EXTRACTION")
    print("=" * 70)
    sys.stdout.flush()

    beh_root = Path("/mtl-data/behavior")
    staging_dir = beh_root / "staging"
    beh_root.mkdir(parents=True, exist_ok=True)

    expected_sha = manifest["archive_sha256"]
    expected_size = manifest["total_bytes"]
    num_chunks = manifest["num_chunks"]

    tar_dest = Path("/tmp/behavior_retained.tar")
    if tar_dest.exists():
        tar_dest.unlink()

    print(f"[*] Reassembling behavior archive ({expected_size / (1024**2):.1f} MB from {num_chunks} chunks)...")
    sys.stdout.flush()

    with open(tar_dest, "wb") as out_f:
        for c in manifest["chunks"]:
            c_name = c["chunk_name"]
            c_path = staging_dir / c_name
            if not c_path.exists():
                raise FileNotFoundError(f"Missing Behavior chunk on volume: {c_path}")
            if c_path.stat().st_size != c["size"]:
                raise ValueError(f"Chunk size mismatch for {c_name}: expected {c['size']}, got {c_path.stat().st_size}")

            with open(c_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)

    actual_sha = _compute_sha256(tar_dest)
    if actual_sha != expected_sha:
        tar_dest.unlink()
        raise ValueError(f"SHA-256 mismatch for behavior archive: expected {expected_sha}, got {actual_sha}")

    print(f"[+] Archive reassembled successfully (sha={actual_sha[:10]}...). Extracting to {beh_root}...")
    sys.stdout.flush()

    t0 = time.time()
    with tarfile.open(tar_dest, "r") as tar:
        tar.extractall(path=beh_root)
    duration = time.time() - t0
    print(f"[+] Extracted behavior archive in {duration:.1f}s")

    if tar_dest.exists():
        tar_dest.unlink()

    # Clean up staging chunks
    for c in manifest["chunks"]:
        c_path = staging_dir / c["chunk_name"]
        if c_path.exists():
            c_path.unlink()
    if staging_dir.exists() and not list(staging_dir.iterdir()):
        staging_dir.rmdir()

    # Verification: Verify 4,271 retained sequences
    train_csv = beh_root / "retained_train.csv"
    val_csv = beh_root / "retained_val.csv"
    assert train_csv.exists(), "Missing retained_train.csv"
    assert val_csv.exists(), "Missing retained_val.csv"

    retained_ids: Set[str] = set()
    for p in (train_csv, val_csv):
        with open(p, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s_id = row.get("sample_id", "").strip()
                if s_id:
                    retained_ids.add(s_id)

    print(f"[*] Verifying {len(retained_ids)} retained sequence directories on volume...")
    sys.stdout.flush()
    missing_dirs = []
    for seq_id in retained_ids:
        s_dir = beh_root / seq_id
        if not s_dir.exists():
            missing_dirs.append(seq_id)
    if missing_dirs:
        raise FileNotFoundError(f"Missing {len(missing_dirs)} sequence directories (e.g. {missing_dirs[:5]})")

    assert len(retained_ids) == 4271, f"Expected 4,271 retained sequences, got {len(retained_ids)}"
    assert not (beh_root / "production_test").exists(), "CRITICAL: production_test found in MTL data volume!"
    assert not (beh_root / "retained_test.csv").exists(), "CRITICAL: retained_test.csv found in MTL data volume!"
    assert not (beh_root / "failed_test.csv").exists(), "CRITICAL: failed_test.csv found in MTL data volume!"

    mtl_data_vol.commit()
    print("[+] Behavior Assembly Complete and Verified ✅")
    return {"status": "SUCCESS", "total_sequences": len(retained_ids)}


# ==============================================================================
# TASK C: RE-ID DIRECT CLOUD HYDRATION & EXTRACTION (Profile: hasinishrak2015)
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol},
    cpu=2.0,
    memory=4096,
    timeout=3600,
)
def stage_reid_direct_remote(threads: int = 16) -> Dict[str, Any]:
    """Downloads parlor.zip directly from Zenodo into ephemeral /tmp/ inside hasinishrak2015,

    selectively extracts ONLY the 15,436 canonical Train/Val image and mask pairs
    for the 41 representation learning cows, asserts zero leakage, and unlinks zip.
    0 bytes relayed through the user's PC.
    """
    import pandas as pd
    sys.path.insert(0, "/root")
    from scripts.fast_download_sideviewcows import (
        FILES_CATALOG,
        CleanProgressBar,
        download_chunk_with_retry,
    )

    print("=" * 70)
    print("  RE-ID DIRECT CLOUD DOWNLOAD & SELECTIVE EXTRACTION")
    print("=" * 70)
    sys.stdout.flush()

    reid_root = Path("/mtl-data/reid")
    reid_root.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical protocols from container root
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    protocol_a_path = protocols_dir / "protocol_cross_setting.csv"
    protocol_d_path = protocols_dir / "protocol_closed_set.csv"

    if not protocol_a_path.exists() or not protocol_d_path.exists():
        raise FileNotFoundError(f"Missing SideView protocol CSVs at {protocols_dir}")

    df_a = pd.read_csv(protocol_a_path)
    train_cows = sorted(df_a.loc[df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())
    evaluation_cows = set(df_a.loc[~df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())

    assert len(train_cows) == 41, f"Expected 41 training cows, got {len(train_cows)}"
    assert len(evaluation_cows) == 69, f"Expected 69 evaluation cows, got {len(evaluation_cows)}"
    assert not set(train_cows).intersection(evaluation_cows), "Training and evaluation cow overlap detected!"

    df_d = pd.read_csv(protocol_d_path)
    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train = df_d_41[df_d_41["closed_set_split"].eq("train")].reset_index(drop=True)
    df_val = df_d_41[df_d_41["closed_set_split"].eq("val")].reset_index(drop=True)

    train_pairs_count = len(df_train)
    val_pairs_count = len(df_val)
    total_pairs_count = train_pairs_count + val_pairs_count

    print(f"[*] Canonical Re-ID Protocol Verified:")
    print(f"    Train pairs: {train_pairs_count} (expected 12,753)")
    print(f"    Val pairs:   {val_pairs_count} (expected 2,683)")
    print(f"    Total pairs: {total_pairs_count} (expected 15,436)")
    print(f"    Unique cows: {len(train_cows)} (expected 41)")
    sys.stdout.flush()

    assert train_pairs_count == 12753, f"Expected 12,753 train pairs, got {train_pairs_count}"
    assert val_pairs_count == 2683, f"Expected 2,683 val pairs, got {val_pairs_count}"
    assert total_pairs_count == 15436, f"Expected 15,436 total pairs, got {total_pairs_count}"

    # Collect exact relative file paths to extract
    required_archive_members: Set[str] = set()

    for df in (df_train, df_val):
        for _, row in df.iterrows():
            img_p = str(row["image_path"]).replace("\\", "/")
            mask_p = str(row["mask_path"]).replace("\\", "/")

            for p in (img_p, mask_p):
                # Normalize to zip archive relative member path (starts with 'parlor/...')
                parts = p.split("sideviewcows2026/")
                rel = parts[-1].lstrip("/") if len(parts) > 1 else p
                required_archive_members.add(rel)

    print(f"[*] Exact required archive members: {len(required_archive_members)} files (15,436 RGB + 15,436 masks)")
    sys.stdout.flush()
    assert len(required_archive_members) == 30872, f"Expected 30,872 files, got {len(required_archive_members)}"

    # 2. Check if already extracted and complete
    all_present = True
    for m in required_archive_members:
        target_f = reid_root / m
        if not target_f.exists() or target_f.stat().st_size == 0:
            all_present = False
            break

    if all_present:
        print("[+] All 15,436 Re-ID Train/Val pairs are already extracted and present on volume!")
        return {
            "status": "CACHED",
            "train_pairs": train_pairs_count,
            "val_pairs": val_pairs_count,
            "total_pairs": total_pairs_count,
            "unique_cows": len(train_cows),
        }

    # 3. Direct high-speed download of parlor.zip to ephemeral /tmp/
    tmp_dir = Path("/tmp/sideview_mtl")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parlor_zip_path = tmp_dir / "parlor.zip"
    parlor_info = FILES_CATALOG["parlor.zip"]
    expected_bytes = parlor_info["size"]

    print(f"\n[*] Downloading parlor.zip ({expected_bytes / (1024**3):.2f} GB) directly to ephemeral storage...")
    sys.stdout.flush()

    if not parlor_zip_path.exists() or parlor_zip_path.stat().st_size != expected_bytes:
        chunk_size = (expected_bytes + threads - 1) // threads
        pbar = CleanProgressBar(total_bytes=expected_bytes, desc="⚡ Downloading parlor.zip")
        part_paths = []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(threads):
                start_b = i * chunk_size
                end_b = min(expected_bytes - 1, start_b + chunk_size - 1)
                if start_b > end_b:
                    continue
                part_p = tmp_dir / f"parlor.part_{start_b}_{end_b}"
                part_paths.append(part_p)
                futures.append(
                    executor.submit(
                        download_chunk_with_retry,
                        parlor_info["url"],
                        part_p,
                        start_b,
                        end_b,
                        pbar,
                    )
                )
            for f in as_completed(futures):
                f.result()
        pbar.close()

        # Stitch into parlor.zip
        print("\n[*] Assembling parlor.zip parts...")
        sys.stdout.flush()
        with open(parlor_zip_path, "wb") as out_f:
            for part_p in part_paths:
                with open(part_p, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)
                part_p.unlink()

        assert parlor_zip_path.stat().st_size == expected_bytes, "parlor.zip download incomplete"
        print(f"[+] Download complete: {parlor_zip_path.stat().st_size / (1024**3):.2f} GB")

    # 4. Selective extraction: Extract ONLY the 15,436 Train/Val members
    print(f"\n[*] Selectively extracting {len(required_archive_members)} files to {reid_root}...")
    sys.stdout.flush()

    t0 = time.time()
    extracted_count = 0
    with zipfile.ZipFile(parlor_zip_path, "r") as z:
        for member_name in required_archive_members:
            z.extract(member_name, path=reid_root)
            extracted_count += 1
            if extracted_count % 5000 == 0 or extracted_count == len(required_archive_members):
                print(f"    Extracted {extracted_count}/{len(required_archive_members)} ({extracted_count / len(required_archive_members) * 100:.1f}%)...")
                sys.stdout.flush()

    extract_duration = time.time() - t0
    print(f"[+] Selective extraction complete in {extract_duration:.1f}s ({extracted_count} files)")

    # 5. Delete temporary parlor.zip immediately to free ephemeral container disk
    if parlor_zip_path.exists():
        parlor_zip_path.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[+] Cleaned up temporary parlor.zip archive")

    # 6. Strict Assertions
    # Check no barn or snapshots data exists in reid_root
    assert not (reid_root / "barn").exists(), "CRITICAL: barn evaluation subset found in MTL volume!"
    assert not (reid_root / "snapshots").exists(), "CRITICAL: snapshots evaluation subset found in MTL volume!"

    # Verify extracted cow directories are strictly subset of train_cows
    extracted_cows = set(p.name for p in (reid_root / "parlor" / "images").iterdir() if p.is_dir())
    assert extracted_cows == set(train_cows), f"Extracted cow set mismatch: {len(extracted_cows)} vs 41"
    assert not extracted_cows.intersection(evaluation_cows), "CRITICAL: Held-out cow overlap detected in MTL Re-ID!"

    mtl_data_vol.commit()
    print("[+] Re-ID Direct Hydration Complete and Verified ✅")
    return {
        "status": "SUCCESS",
        "train_pairs": train_pairs_count,
        "val_pairs": val_pairs_count,
        "total_pairs": total_pairs_count,
        "unique_cows": len(train_cows),
        "held_out_cow_overlap": 0,
    }


# ==============================================================================
# AUDIT & VERIFICATION GATEWAY (Profile: hasinishrak2015)
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol, "/mtl-checkpoints": mtl_checkpoints_vol},
    cpu=2.0,
    memory=4096,
    timeout=600,
)
def verify_mtl_workspace_remote() -> Dict[str, Any]:
    """Exhaustively verifies all 3 tasks on /mtl-data and creates staging_manifest.json."""
    import torch
    from PIL import Image

    print("=" * 70)
    print("  EXHAUSTIVE MTL WORKSPACE STAGING AUDIT & CERTIFICATION")
    print("=" * 70)
    sys.stdout.flush()

    data_root = Path("/mtl-data")
    checkpoints_root = Path("/mtl-checkpoints")

    # 1. Check volume mount & writeability of mtl-checkpoints
    assert checkpoints_root.exists(), "mtl-checkpoints volume is not mounted"
    test_probe = checkpoints_root / ".write_probe.tmp"
    with open(test_probe, "w") as f:
        f.write("OK")
    assert test_probe.exists() and test_probe.read_text() == "OK"
    test_probe.unlink()
    mtl_checkpoints_vol.commit()
    print("[*] mtl-checkpoints: Writable and verified ✅")

    # 2. Audit Task A: BCS
    bcs_root = data_root / "bcs"
    assert bcs_root.exists(), "Missing /mtl-data/bcs"
    bcs_train_pt = bcs_root / "train_bcs_224.pt"
    bcs_val_pt = bcs_root / "val_bcs_224.pt"
    bcs_train_csv = bcs_root / "train_perception.csv"
    bcs_val_csv = bcs_root / "val_perception.csv"

    for p in (bcs_train_pt, bcs_val_pt, bcs_train_csv, bcs_val_csv):
        assert p.exists() and p.stat().st_size > 0, f"BCS file missing or empty: {p}"

    assert not (bcs_root / "test_bcs_224.pt").exists(), "LEAKAGE: test_bcs_224.pt present in /mtl-data/bcs"
    assert not (bcs_root / "test_perception.csv").exists(), "LEAKAGE: test_perception.csv present in /mtl-data/bcs"

    train_tensors = torch.load(bcs_train_pt, weights_only=False)
    val_tensors = torch.load(bcs_val_pt, weights_only=False)
    bcs_audit = {
        "train_samples": int(train_tensors["images"].shape[0]),
        "val_samples": int(val_tensors["images"].shape[0]),
        "tensor_shape": list(train_tensors["images"].shape[1:]),
        "train_pt_bytes": bcs_train_pt.stat().st_size,
        "val_pt_bytes": bcs_val_pt.stat().st_size,
    }
    del train_tensors, val_tensors
    print(f"[*] BCS Audit PASS: Train={bcs_audit['train_samples']}, Val={bcs_audit['val_samples']}, Shape={bcs_audit['tensor_shape']} ✅")

    # 3. Audit Task B: Behavior
    beh_root = data_root / "behavior"
    assert beh_root.exists(), "Missing /mtl-data/behavior"
    beh_train_csv = beh_root / "retained_train.csv"
    beh_val_csv = beh_root / "retained_val.csv"
    assert beh_train_csv.exists() and beh_val_csv.exists()

    assert not (beh_root / "production_test").exists(), "LEAKAGE: production_test found in /mtl-data/behavior"
    assert not (beh_root / "retained_test.csv").exists(), "LEAKAGE: retained_test.csv found in /mtl-data/behavior"
    assert not (beh_root / "failed_test.csv").exists(), "LEAKAGE: failed_test.csv found in /mtl-data/behavior"

    beh_train_ids = set()
    with open(beh_train_csv, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            beh_train_ids.add(r["sample_id"].strip())
    beh_val_ids = set()
    with open(beh_val_csv, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            beh_val_ids.add(r["sample_id"].strip())

    assert len(beh_train_ids) == 3641, f"Expected 3,641 Behavior train sequences, got {len(beh_train_ids)}"
    assert len(beh_val_ids) == 630, f"Expected 630 Behavior val sequences, got {len(beh_val_ids)}"
    total_beh = len(beh_train_ids) + len(beh_val_ids)
    assert total_beh == 4271, f"Expected 4,271 Behavior sequences, got {total_beh}"

    # Sample check 10 random sequences for 8 frames + 8 masks + metadata
    all_beh_ids = list(beh_train_ids.union(beh_val_ids))
    for s_id in all_beh_ids[:10]:
        s_dir = beh_root / s_id
        assert s_dir.exists(), f"Missing sequence directory: {s_dir}"
        assert (s_dir / "perception_metadata.json").exists(), f"Missing metadata in {s_dir}"
        for t in range(8):
            f_p = s_dir / f"frame_{t:03d}.jpg"
            m_p = s_dir / f"mask_{t:03d}.png"
            assert f_p.exists() and f_p.stat().st_size > 0, f"Missing frame {f_p}"
            assert m_p.exists() and m_p.stat().st_size > 0, f"Missing mask {m_p}"

    beh_audit = {
        "train_sequences": len(beh_train_ids),
        "val_sequences": len(beh_val_ids),
        "total_sequences": total_beh,
    }
    print(f"[*] Behavior Audit PASS: Train={beh_audit['train_sequences']}, Val={beh_audit['val_sequences']}, Total={total_beh} ✅")

    # 4. Audit Task C: Re-ID
    reid_root = data_root / "reid"
    assert reid_root.exists(), "Missing /mtl-data/reid"
    assert not (reid_root / "barn").exists(), "LEAKAGE: barn present in /mtl-data/reid"
    assert not (reid_root / "snapshots").exists(), "LEAKAGE: snapshots present in /mtl-data/reid"

    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    df_a = pd.read_csv(protocols_dir / "protocol_cross_setting.csv")
    train_cows = sorted(df_a.loc[df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())
    eval_cows = set(df_a.loc[~df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())

    df_d = pd.read_csv(protocols_dir / "protocol_closed_set.csv")
    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train_reid = df_d_41[df_d_41["closed_set_split"].eq("train")].reset_index(drop=True)
    df_val_reid = df_d_41[df_d_41["closed_set_split"].eq("val")].reset_index(drop=True)

    # Check existence of all 15,436 pairs
    missing_reid = []
    for split_df in (df_train_reid, df_val_reid):
        for _, row in split_df.iterrows():
            img_rel = str(row["image_path"]).replace("\\", "/").split("sideviewcows2026/")[-1].lstrip("/")
            mask_rel = str(row["mask_path"]).replace("\\", "/").split("sideviewcows2026/")[-1].lstrip("/")
            if not (reid_root / img_rel).exists() or not (reid_root / mask_rel).exists():
                missing_reid.append((img_rel, mask_rel))

    assert not missing_reid, f"Missing {len(missing_reid)} Re-ID files (e.g. {missing_reid[:3]})"
    reid_cows_on_disk = set(p.name for p in (reid_root / "parlor" / "images").iterdir() if p.is_dir())
    assert reid_cows_on_disk == set(train_cows), "Re-ID cow identity mismatch"
    assert not reid_cows_on_disk.intersection(eval_cows), "LEAKAGE: Held-out cows found in Re-ID volume"

    reid_audit = {
        "train_pairs": len(df_train_reid),
        "val_pairs": len(df_val_reid),
        "total_pairs": len(df_train_reid) + len(df_val_reid),
        "unique_cows": len(reid_cows_on_disk),
        "held_out_cow_overlap": 0,
    }
    print(f"[*] Re-ID Audit PASS: Train={reid_audit['train_pairs']}, Val={reid_audit['val_pairs']}, Cows={reid_audit['unique_cows']} ✅")

    # 5. Build and write /mtl-data/staging_manifest.json
    manifest_data = {
        "workspace_profile": "hasinishrak2015",
        "volumes": {
            "data_volume": "mtl-data",
            "checkpoints_volume": "mtl-checkpoints",
        },
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": {
            "bcs": bcs_audit,
            "behavior": beh_audit,
            "reid": reid_audit,
        },
        "leakage_checks": {
            "bcs_test_absent": True,
            "behavior_test_absent": True,
            "reid_evaluation_subsets_absent": True,
            "reid_held_out_cow_overlap": 0,
        },
        "status": "CERTIFIED_READY_FOR_MTL",
    }

    manifest_path = data_root / "staging_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    mtl_data_vol.commit()
    print(f"\n[+] Staging Manifest Saved to /mtl-data/staging_manifest.json")
    print("=" * 70)
    print("  MTL WORKSPACE HASINISHRAK2015 100% CERTIFIED READY 🚀")
    print("=" * 70)
    return manifest_data
