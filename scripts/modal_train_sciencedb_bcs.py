# -*- coding: utf-8 -*-
"""
Modal Cloud Wrapper for Phase 3 ScienceDB RGB Single-Task BCS Baseline Training
================================================================================
Profile Target   : tigerwood697
Dataset Volume   : sciencedb-data (mounted at /data)
Checkpoint Volume: sciencedb-checkpoints (mounted at /checkpoints)
GPU Target       : NVIDIA L4 (24GB VRAM, Ada Lovelace tier)
Primary Script   : scripts/train_sciencedb_bcs_baseline.py

Usage:
  1. Readiness Verification (Readiness Checks Only):
     modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs.py::verify_readiness

  2. Full 30-Epoch Baseline Training (Manual execution by user):
     modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs.py::main --epochs 30 --head-type ordinal_bce --batch-size 32
"""

import sys
import os
from pathlib import Path
from typing import Optional

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Persistent Storage Volumes on tigerwood697
data_vol = modal.Volume.from_name("sciencedb-data")
checkpoint_vol = modal.Volume.from_name("sciencedb-checkpoints", create_if_missing=True)

# Container image with PyTorch, Vision, and project files
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy",
        "pandas",
        "pillow",
        "scikit-learn",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "bcs" / "sciencedb"),
        remote_path="/root/datasets/bcs/sciencedb",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sciencedb_bcs_baseline.py"),
        remote_path="/root/scripts/train_sciencedb_bcs_baseline.py",
    )
)

app = modal.App("sciencedb-bcs-baseline", image=train_image)


# ==============================================================================
# READINESS VERIFICATION FUNCTION (CHECKS ONLY)
# ==============================================================================
@app.function(
    gpu=os.environ.get("MODAL_GPU", "L4"),
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=300,
    cpu=2.0,
    memory=4096,
)
def verify_readiness_remote():
    """Performs exhaustive pre-flight verification without training."""
    import time
    from PIL import Image
    import pandas as pd
    import torch
    from tqdm import tqdm
    
    sys.path.insert(0, "/root")
    from scripts.train_sciencedb_bcs_baseline import resolve_image_path, compute_file_sha256

    print("\n" + "=" * 65)
    print("  MODAL PRE-FLIGHT READINESS VERIFICATION (PROFILE: tigerwood697)")
    print("=" * 65)

    report = {"all_passed": True, "checks": {}}

    # 1. CUDA & GPU Verification
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    print(f"[*] CUDA Available: {cuda_avail} | GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    report["checks"]["cuda"] = {
        "passed": cuda_avail and "L4" in gpu_name,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
    }

    # 2. Volume & Image Root Verification
    dataset_dir = Path("/data/dataset")
    expected_classes = ["3.25", "3.5", "3.75", "4.0", "4.25"]
    expected_counts = {"3.25": 7536, "3.5": 13256, "3.75": 14255, "4.0": 12556, "4.25": 5963}
    volume_ok = dataset_dir.exists() and all((dataset_dir / c).exists() for c in expected_classes)
    
    class_counts = {}
    total_images_found = 0
    zero_byte_count = 0
    if volume_ok:
        for c in expected_classes:
            c_dir = dataset_dir / c
            files = [f for f in c_dir.iterdir() if f.suffix.lower() == ".jpg"]
            n_imgs = len(files)
            class_counts[c] = n_imgs
            total_images_found += n_imgs
            for f in files:
                if f.stat().st_size == 0:
                    zero_byte_count += 1
    
    print(f"[*] ScienceDB Image Root: {dataset_dir} (Exists: {volume_ok})")
    print(f"[*] Verified Images on Volume: {total_images_found} images across 5 classes: {class_counts}")
    print(f"[*] Zero-byte corrupted files on volume: {zero_byte_count}")
    report["checks"]["dataset_volume"] = {
        "passed": volume_ok and total_images_found == 53566 and zero_byte_count == 0 and class_counts == expected_counts,
        "root_path": str(dataset_dir),
        "total_images": total_images_found,
        "zero_byte_count": zero_byte_count,
        "class_counts": class_counts,
    }

    # 3. Canonical Splits & Provenance Hashes
    splits_dir = Path("/root/datasets/bcs/sciencedb")
    split_status = {}
    for split_name in ["train", "val", "test"]:
        sp = splits_dir / f"{split_name}.csv"
        sp_exists = sp.exists()
        sha = compute_file_sha256(sp) if sp_exists else "MISSING"
        df = pd.read_csv(sp) if sp_exists else pd.DataFrame()
        split_status[split_name] = {
            "exists": sp_exists,
            "sha256": sha,
            "rows": len(df),
            "burst_groups": df["burst_group_id"].nunique() if sp_exists else 0,
        }
        print(f"[*] Split {split_name}.csv: rows={len(df)}, groups={split_status[split_name]['burst_groups']}, sha256={sha[:12]}...")

    expected_hashes = {
        "train": "9f6b0bc716e01a2ab22208daff1c49e49fd450a4d7cf0a57b2979275ed33497a",
        "val": "e223e3c4c081ca5c9f993f7156dc791df97b6ea6d011b8b4b6f068590b3d975d",
        "test": "eae459e031d06c4b1150ce2cbcdcb8259724b070b831341222b15c99e3626e5f",
    }
    splits_ok = all(
        split_status[s]["sha256"] == expected_hashes[s] and split_status[s]["rows"] > 0
        for s in ["train", "val", "test"]
    )
    report["checks"]["canonical_splits"] = {"passed": splits_ok, "details": split_status}

    # 4. Runtime Path Resolution Check (5 samples from each split)
    path_res_ok = True
    resolved_samples = []
    for split_name in ["train", "val", "test"]:
        sp = splits_dir / f"{split_name}.csv"
        df = pd.read_csv(sp)
        for _, row in df.head(5).iterrows():
            raw_path = str(row["image_path"])
            resolved = resolve_image_path(raw_path, data_root=Path("/data"))
            exists = resolved.exists()
            if exists:
                try:
                    with Image.open(resolved) as img:
                        w, h = img.size
                    resolved_samples.append({"split": split_name, "raw": raw_path[-30:], "resolved": str(resolved), "size": f"{w}x{h}"})
                except Exception as e:
                    exists = False
                    path_res_ok = False
            else:
                path_res_ok = False

    print(f"[*] Representative Path Resolution Check: {len(resolved_samples)}/15 samples opened successfully.")
    for s in resolved_samples[:3]:
        print(f"    - [{s['split']}] ...{s['raw']} -> {s['resolved']} ({s['size']})")
    report["checks"]["path_resolution"] = {"passed": path_res_ok and len(resolved_samples) == 15, "verified_samples": len(resolved_samples)}

    # 5. Persistent Checkpoint Storage Check
    ckpt_dir = Path("/checkpoints/bcs_baseline")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_file = ckpt_dir / "readiness_test.txt"
    test_file.write_text(f"Readiness check passed at {time.time()}\n")
    checkpoint_vol.commit()
    ckpt_writable = test_file.exists()
    print(f"[*] Persistent Checkpoint Directory: {ckpt_dir} (Writable & Committed: {ckpt_writable})")
    report["checks"]["persistent_checkpoints"] = {"passed": ckpt_writable, "path": str(ckpt_dir)}

    # 6. TQDM Progress Bar Visual Check
    print("[*] Testing tqdm progress streaming:")
    pbar = tqdm(total=5, desc="Readiness Test | TQDM", unit="batch", bar_format="{desc} | {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}{postfix}]", file=sys.stdout, leave=False)
    for b in range(5):
        time.sleep(0.05)
        pbar.set_postfix_str(f"loss={0.5 - b*0.02:.4f}")
        pbar.update(1)
    pbar.close()
    print("[OK] TQDM progress streaming verified cleanly.")
    report["checks"]["tqdm_streaming"] = {"passed": True}

    report["all_passed"] = all(c["passed"] for c in report["checks"].values())
    print("\n" + "=" * 65)
    print(f"  PRE-FLIGHT READINESS VERDICT: {'READY' if report['all_passed'] else 'NOT READY'}")
    print("=" * 65 + "\n")
    return report


# ==============================================================================
# FULL 30-EPOCH TRAINING REMOTE FUNCTION
# ==============================================================================
@app.function(
    gpu=os.environ.get("MODAL_GPU", "L4"),
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=3600 * 3,  # 3 hours max runtime
    cpu=4.0,
    memory=16384,     # 16 GB RAM
)
def train_sciencedb_bcs_remote(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    head_type: str = "ordinal_bce",
    seed: int = 42,
    smoke: bool = False,
    resume: Optional[str] = None,
    max_samples: int = 250,
    max_batches: int = 10,
    test_save_resume: bool = True,
    num_workers: int = 4,
):
    """
    Executes Phase 3 ScienceDB RGB single-task BCS baseline training on Modal.
    Mounts /data for dataset and /checkpoints for persistent state storage.
    Commits checkpoints to persistent cloud volume at the end of every epoch.
    """
    import argparse
    import torch

    print("\n" + "=" * 65)
    print("  MODAL CLOUD TRAINING: SCIENCEDB RGB BCS BASELINE")
    print("=" * 65)
    print(f"  GPU              : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM             : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "  VRAM: N/A")
    print(f"  Data Mount       : /data (ScienceDB dataset volume)")
    print(f"  Checkpoint Mount : /checkpoints (Persistent Volume)")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Head Type        : {head_type}")
    print(f"  Seed             : {seed}")
    print(f"  Execution Mode   : {'SMOKE TEST' if smoke else 'FULL 30-EPOCH TRAINING'}")
    print("=" * 65 + "\n")

    sys.path.insert(0, "/root")
    from scripts.train_sciencedb_bcs_baseline import run_bcs_pipeline

    # Epoch callback: commit checkpoint volume after each epoch for guaranteed persistence
    def on_epoch_end_callback(epoch, is_best, payload):
        checkpoint_vol.commit()
        status_tag = "[BEST CHECKPOINT SAVED & COMMITTED]" if is_best else "[LATEST CHECKPOINT SAVED & COMMITTED]"
        print(f"[*] {status_tag} -> Persistent Modal Volume 'sciencedb-checkpoints' synced for Epoch {epoch}")

    args = argparse.Namespace(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        image_size=224,
        seed=seed,
        head_type=head_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=num_workers,
        output_dir="/checkpoints/bcs_baseline",
        resume=resume,
        smoke=smoke,
        dry_run=False,
        max_samples=max_samples,
        max_batches=max_batches,
        test_save_resume=test_save_resume,
        data_root="/data",
        split_dir="/root/datasets/bcs/sciencedb",
    )

    exit_code = run_bcs_pipeline(args, on_epoch_end_callback=on_epoch_end_callback)
    checkpoint_vol.commit()
    print("[*] Final Volume Commit Complete! Training artifacts are permanently stored on 'sciencedb-checkpoints'.")
    return exit_code


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    head_type: str = "ordinal_bce",
    seed: int = 42,
    smoke: bool = False,
    resume: Optional[str] = None,
    num_workers: int = 4,
):
    """Default entrypoint to trigger training with CLI parameters."""
    train_sciencedb_bcs_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        head_type=head_type,
        seed=seed,
        smoke=smoke,
        resume=resume,
        num_workers=num_workers,
    )


@app.local_entrypoint()
def verify_readiness():
    """Standalone entrypoint to run readiness pre-flight checks."""
    res = verify_readiness_remote.remote()
    if not res.get("all_passed"):
        print("[!] Readiness checks failed!")
        sys.exit(1)
    else:
        print("[READY] All 6 readiness checks passed successfully! System is 100% READY for training.")
