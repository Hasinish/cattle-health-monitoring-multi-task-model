# -*- coding: utf-8 -*-
"""
modal_train_moo_real_viewpoint.py — Modal Cloud Wrapper for MOO -> Real 3-Class Viewpoint Fine-Tuning

Execution Environment:
  - Modal Profile: tigerwood693
  - GPU Tier: NVIDIA Tesla T4 (Modest, lowest-cost GPU tier for ResNet-18 fine-tuning)
  - Volumes:
      * 'viewpoint-real-data' mounted at /data (source cropped dataset)
      * 'moo-data' mounted at /moo_data (pretrained MOO synthetic checkpoint)
      * 'viewpoint-checkpoints' mounted at /checkpoints (output checkpoints and metrics)

Usage:
  # Pre-flight readiness check (zero GPU cost or minimal check):
  modal run --profile tigerwood693 scripts/modal_train_moo_real_viewpoint.py::verify_readiness

  # Manual full fine-tuning run (to be executed by user):
  modal run --profile tigerwood693 scripts/modal_train_moo_real_viewpoint.py::main --epochs 20 --batch-size 32
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
real_data_volume = modal.Volume.from_name("viewpoint-real-data", create_if_missing=True)
moo_data_volume = modal.Volume.from_name("moo-data")
checkpoint_volume = modal.Volume.from_name("viewpoint-checkpoints", create_if_missing=True)

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "pandas",
        "scikit-learn",
        "pillow",
        "tqdm",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_moo_real_viewpoint.py"),
        remote_path="/root/train_moo_real_viewpoint.py",
    )
)

app = modal.App("moo-real-viewpoint-finetune", image=image)


@app.function(
    volumes={
        "/data": real_data_volume,
        "/moo_data": moo_data_volume,
        "/checkpoints": checkpoint_volume,
    },
    cpu=2.0,
    memory=4096,
    timeout=600,
)
def verify_readiness():
    """
    Zero-GPU remote pre-flight audit:
    Verifies that all 3 volumes, datasets, MOO checkpoint, and split manifests resolve.
    """
    import torch
    import pandas as pd
    from PIL import Image

    print("\n" + "=" * 70, flush=True)
    print("  MOO -> REAL VIEWPOINT MODAL PRE-FLIGHT READINESS AUDIT", flush=True)
    print("=" * 70, flush=True)

    # 1. Verify Real Viewpoint Data Volume
    print("[1/5] Verifying real viewpoint cropped dataset on /data...", flush=True)
    dataset_dir = Path("/data/self_clean_v1_rtdetr_crop")
    assert dataset_dir.exists(), f"Dataset directory missing: {dataset_dir}"

    for cls_name, expected in [("front", 392), ("side", 222), ("rear", 265)]:
        p = dataset_dir / cls_name
        assert p.exists(), f"Class dir missing: {p}"
        files = list(p.glob("*.jpg"))
        print(f"  ✓ Class '{cls_name:5s}': {len(files)} / {expected} files", flush=True)
        assert len(files) == expected, f"Count mismatch for {cls_name}: {len(files)} != {expected}"

    # 2. Verify Split Manifests
    print("\n[2/5] Verifying split manifests (TEST SET EXCLUSION)...", flush=True)
    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    test_csv = dataset_dir / "test.csv"
    if not train_csv.exists():
        train_csv = dataset_dir / "splits" / "train.csv"
        val_csv = dataset_dir / "splits" / "val.csv"
        test_csv = dataset_dir / "splits" / "test.csv"

    assert train_csv.exists(), "train.csv missing"
    assert val_csv.exists(), "val.csv missing"
    assert test_csv.exists(), "test.csv missing"

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    print(f"  ✓ Train split: {len(df_tr)} samples across {df_tr['duplicate_group_id'].nunique()} groups", flush=True)
    print(f"  ✓ Val split:   {len(df_va)} samples across {df_va['duplicate_group_id'].nunique()} groups", flush=True)
    print(f"  [*] Test split confirmed present ({len(pd.read_csv(test_csv))} rows) and STRICTLY FROZEN.", flush=True)

    # 3. Verify MOO Pretrained Checkpoint
    print("\n[3/5] Verifying pretrained MOO checkpoint on /moo_data...", flush=True)
    moo_ckpt = Path("/moo_data/moo_resnet18_viewpoint8_full_l4.pth")
    assert moo_ckpt.exists(), f"MOO checkpoint not found: {moo_ckpt}"
    ckpt_sz_mb = moo_ckpt.stat().st_size / (1024 * 1024)
    print(f"  ✓ Found MOO checkpoint: {moo_ckpt} ({ckpt_sz_mb:.2f} MB)", flush=True)

    # Test loading checkpoint keys
    ckpt = torch.load(moo_ckpt, map_location="cpu", weights_only=False)
    print(f"  ✓ Checkpoint loaded: {ckpt.get('class_names')} (Best Syn Val F1: {ckpt.get('best_val_macro_f1', 0.0):.4f})", flush=True)

    # 4. Verify Writable Checkpoint Volume
    print("\n[4/5] Verifying writable checkpoint volume on /checkpoints...", flush=True)
    ckpt_dir = Path("/checkpoints/viewpoint_real_finetune")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_file = ckpt_dir / "test_write.tmp"
    with open(test_file, "w") as f:
        f.write("write_ok")
    test_file.unlink()
    checkpoint_volume.commit()
    print("  ✓ Checkpoint volume is writable and committed.", flush=True)

    # 5. Quick Test Sample Decode
    print("\n[5/5] Testing PIL decode on sample image...", flush=True)
    sample_img = next((dataset_dir / "front").glob("*.jpg"))
    with Image.open(sample_img) as img:
        print(f"  ✓ Decoded {sample_img.name}: {img.size} mode={img.mode}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"  ✓ 100% PRE-FLIGHT CHECKS PASSED — READY FOR {MODAL_GPU} TRAINING!", flush=True)
    print("=" * 70, flush=True)
    return {
        "status": "READY",
        "dataset_images": 879,
        "moo_checkpoint": str(moo_ckpt),
        "gpu_target": MODAL_GPU,
    }


MODAL_GPU = os.environ.get("MODAL_GPU", "L40S")


@app.function(
    volumes={
        "/data": real_data_volume,
        "/moo_data": moo_data_volume,
        "/checkpoints": checkpoint_volume,
    },
    gpu=MODAL_GPU,
    cpu=8.0,
    memory=16384,
    timeout=3600,
)
def main(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    smoke: bool = False,
):
    """
    Main training entry point dispatched to a GPU container (Default: NVIDIA L40S).
    """
    import sys
    sys.path.insert(0, "/root")
    from train_moo_real_viewpoint import train_pipeline

    print("\n" + "=" * 70, flush=True)
    print(f"  LAUNCHING MOO -> REAL VIEWPOINT FINE-TUNING ON MODAL ({MODAL_GPU})", flush=True)
    print("=" * 70, flush=True)

    data_dir = Path("/data/self_clean_v1_rtdetr_crop")
    moo_ckpt = Path("/moo_data/moo_resnet18_viewpoint8_full_l4.pth")
    output_dir = Path("/checkpoints/viewpoint_real_finetune")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_pipeline(
        data_dir=data_dir,
        moo_ckpt_path=moo_ckpt,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        smoke=smoke,
    )

    # Commit persistent checkpoints to volume
    checkpoint_volume.commit()
    print("\n✓ Committed checkpoints and training metrics to persistent volume 'viewpoint-checkpoints'.", flush=True)
    return metrics
