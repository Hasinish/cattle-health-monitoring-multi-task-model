# -*- coding: utf-8 -*-
"""
scripts/modal_train_sciencedb_bcs_perception.py — Modal Cloud Wrapper for Run 4 BCS Perception
=============================================================================================
Execution Environment:
  - Target Modal Profile: tigerwood697
  - Volumes:
      * 'sciencedb-data' mounted at /data (53,566 source images)
      * 'sciencedb-perception-cache' mounted at /cache (frozen crops, masks, manifests)
      * 'sciencedb-checkpoints' mounted at /checkpoints (Run 4 model weights and metrics)
  - Preprocessing Target: NVIDIA L4 GPU (cost-effective, high VRAM)
  - Training Target: NVIDIA L40S GPU (48GB Ada Lovelace, maximum throughput)

Usage:
  # 1. Pre-flight verification (Zero-GPU, fast checks):
  modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs_perception.py::verify_readiness

  # 2. Build full perception cache (Run manually):
  modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs_perception.py::build_cache

  # 3. Launch full 30-epoch Run 4 training (Run manually):
  modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs_perception.py::main --epochs 30 --batch-size 64
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _orig_commonpath = os.path.commonpath

    def _safe_commonpath(paths):
        try:
            return _orig_commonpath(paths)
        except ValueError:
            return ""

    os.path.commonpath = _safe_commonpath

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Persistent Modal Volumes
data_volume = modal.Volume.from_name("sciencedb-data")
cache_volume = modal.Volume.from_name("sciencedb-perception-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("sciencedb-checkpoints", create_if_missing=True)

# Container Image with Pre-cached RT-DETR-L and SAM 2.1 weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "pandas",
        "scikit-learn",
        "pillow",
        "tqdm",
    )
    .run_commands(
        "python -c \"from ultralytics import RTDETR, SAM; RTDETR('rtdetr-l.pt'); SAM('sam2.1_s.pt')\""
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "build_sciencedb_perception_cache.py"),
        remote_path="/root/build_sciencedb_perception_cache.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sciencedb_bcs_perception.py"),
        remote_path="/root/train_sciencedb_bcs_perception.py",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "bcs" / "sciencedb"),
        remote_path="/root/sciencedb_splits",
    )
)

app = modal.App("sciencedb-bcs-perception-run4", image=image)


@app.function(
    volumes={
        "/data": data_volume,
        "/cache": cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    cpu=2.0,
    memory=4096,
    timeout=600,
)
def verify_readiness():
    """
    Zero-GPU Pre-Flight Verification on Modal.
    Verifies that sciencedb-data is populated, split CSVs exist, and cache/checkpoint volumes are writable.
    """
    import pandas as pd
    from PIL import Image

    print("\n" + "=" * 70, flush=True)
    print("  SCIENTEDB RUN 4 PRE-FLIGHT READINESS AUDIT", flush=True)
    print("=" * 70, flush=True)

    # 1. Verify ScienceDB Data Volume
    print("[1/4] Verifying source ScienceDB dataset on /data...", flush=True)
    data_dir = Path("/data/dataset")
    if not data_dir.exists():
        data_dir = Path("/data")
    assert data_dir.exists(), f"Source data directory missing: {data_dir}"

    for cls_name, expected in [("3.25", 7536), ("3.5", 13256), ("3.75", 14255), ("4.0", 12556), ("4.25", 5963)]:
        p = data_dir / cls_name
        assert p.exists(), f"Class dir missing: {p}"
        files = list(p.glob("*.jpg"))
        print(f"  ✓ Class '{cls_name:5s}': {len(files)} / {expected} images on volume", flush=True)
        assert len(files) == expected, f"Count mismatch for {cls_name}: {len(files)} != {expected}"

    # 2. Verify Canonical Split Manifests
    print("\n[2/4] Verifying canonical split CSVs (test set strictly frozen)...", flush=True)
    splits_dir = Path("/root/sciencedb_splits")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    assert train_csv.exists(), "train.csv missing"
    assert val_csv.exists(), "val.csv missing"
    assert test_csv.exists(), "test.csv missing"

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)
    print(f"  ✓ Train split: {len(df_tr)} images ({df_tr['burst_group_id'].nunique()} burst groups)", flush=True)
    print(f"  ✓ Val split:   {len(df_va)} images ({df_va['burst_group_id'].nunique()} burst groups)", flush=True)
    print(f"  ✓ Test split:  {len(df_te)} images ({df_te['burst_group_id'].nunique()} burst groups) [FROZEN]", flush=True)

    # 3. Verify Cache Volume Writable
    print("\n[3/4] Verifying persistent cache volume on /cache...", flush=True)
    test_write = Path("/cache/test_probe.tmp")
    with open(test_write, "w") as f:
        f.write("probe_ok")
    test_write.unlink()
    cache_volume.commit()
    print("  ✓ Volume 'sciencedb-perception-cache' is writable and committed.", flush=True)

    # 4. Verify Checkpoint Volume Writable
    print("\n[4/4] Verifying checkpoint volume on /checkpoints...", flush=True)
    out_dir = Path("/checkpoints/bcs_perception_run4")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ckpt = out_dir / "test_probe.tmp"
    with open(test_ckpt, "w") as f:
        f.write("probe_ok")
    test_ckpt.unlink()
    checkpoint_volume.commit()
    print("  ✓ Volume 'sciencedb-checkpoints' is writable and committed.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  ✓ 100% PRE-FLIGHT VERIFICATIONS PASSED — READY FOR RUN 4!", flush=True)
    print("=" * 70, flush=True)
    return {"status": "READY", "total_images": 53566}


@app.function(
    volumes={
        "/data": data_volume,
        "/cache": cache_volume,
    },
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=86400,  # 24h ceiling for cache generation
)
def build_cache(
    split: str = "all",
    smoke: bool = False,
    max_samples: int = None,
):
    """
    Executes full perception preprocessing (RT-DETR-L + SAM 2.1) on Modal.
    Saves crops, binary masks, and provenance manifests to persistent volume 'sciencedb-perception-cache'.
    """
    import sys
    sys.path.insert(0, "/root")
    from build_sciencedb_perception_cache import run_cache_generation

    data_dir = Path("/data/dataset")
    if not data_dir.exists():
        data_dir = Path("/data")
    splits_dir = Path("/root/sciencedb_splits")
    cache_dir = Path("/cache")

    stats = run_cache_generation(
        data_dir=data_dir,
        splits_dir=splits_dir,
        cache_dir=cache_dir,
        split_to_run=split,
        smoke=smoke,
        max_samples=max_samples,
        device="cuda",
    )

    cache_volume.commit()
    print("\n✓ Committed all crops, masks, and manifests to 'sciencedb-perception-cache'.", flush=True)
    return stats


CACHE_GPU = os.environ.get("MODAL_GPU", "L40S")


@app.function(
    volumes={
        "/data": data_volume,
        "/cache": cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    gpu=CACHE_GPU,
    cpu=8.0,
    memory=32768,
    timeout=7200,  # 2h ceiling for 30-epoch training
)
def main(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    mask_init: str = "mean",
    smoke: bool = False,
):
    """
    Main training entry point for Run 4 BCS Perception-Enhanced Model on Modal L40S.
    """
    import sys
    sys.path.insert(0, "/root")
    from train_sciencedb_bcs_perception import train_pipeline

    manifest_dir = Path("/cache/manifests")
    cache_dir = Path("/cache")
    output_dir = Path("/checkpoints/bcs_perception_run4")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_pipeline(
        manifest_dir=manifest_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        mask_init=mask_init,
        smoke=smoke,
    )

    checkpoint_volume.commit()
    print("\n✓ Committed Run 4 model checkpoints and metrics to volume 'sciencedb-checkpoints'.", flush=True)
    return metrics
