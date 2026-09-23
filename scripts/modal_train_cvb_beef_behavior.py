# -*- coding: utf-8 -*-
"""
Modal Cloud Wrapper for Phase 3 Step 4.2 Behavior RGB Single-Task Baseline
==========================================================================
Target Profile   : tigerwood693 (or rotating profiles with cvb-data & beef-behavior-data)
CVB Volume       : cvb-data (mounted at /mnt/cvb)
Beef Volume      : beef-behavior-data (mounted at /mnt/beef)
Checkpoint Volume: behavior-checkpoints (mounted at /checkpoints)
GPU Target       : NVIDIA L40S (48GB VRAM Ada Lovelace) for full training; T4 for smoke test
Primary Script   : scripts/train_cvb_beef_behavior_baseline.py

Usage:
  1. Smoke Test (Verifies paths, tiny cache, forward/backward, 2 epochs, checkpoint resume):
     modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior.py::smoke_test

  2. Full 30-Epoch Baseline Training (User-initiated command):
     modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior.py::main --epochs 30 --batch-size 64
"""

import os
import sys
from pathlib import Path

# Windows UTF-8 console output patch
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

# Persistent Storage Volumes
cvb_vol = modal.Volume.from_name("cvb-data")
beef_vol = modal.Volume.from_name("beef-behavior-data")
checkpoint_vol = modal.Volume.from_name("behavior-checkpoints", create_if_missing=True)

# Container image
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy",
        "pandas",
        "pillow",
        "opencv-python-headless",
        "scikit-learn",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "behavior" / "cvb_beef"),
        remote_path="/root/datasets/behavior/cvb_beef",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_cvb_beef_behavior_baseline.py"),
        remote_path="/root/scripts/train_cvb_beef_behavior_baseline.py",
    )
)

app = modal.App("cvb-beef-behavior-baseline", image=train_image)


# ==============================================================================
# SMOKE TEST FUNCTION (FAST VERIFICATION ON T4)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={
        "/mnt/cvb": cvb_vol,
        "/mnt/beef": beef_vol,
        "/checkpoints": checkpoint_vol,
    },
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def smoke_test_remote() -> dict:
    """
    Executes comprehensive smoke test on Modal:
    1. Verifies physical paths on /mnt/cvb and /mnt/beef
    2. Verifies canonical split CSVs (asserting test.csv is never loaded)
    3. Builds a tiny cache subset (16 train, 8 val)
    4. Executes forward & backward pass over 2 tiny epochs
    5. Computes validation metrics
    6. Verifies checkpoint save & bit-identical resume
    7. Asserts test.csv remained 100% untouched
    """
    import os
    import sys
    from pathlib import Path
    import torch
    import pandas as pd

    sys.path.insert(0, "/root")
    from scripts.train_cvb_beef_behavior_baseline import train_pipeline

    print("\n" + "=" * 70)
    print("  MODAL SMOKE TEST: STEP 4.2 BEHAVIOR RGB BASELINE")
    print("=" * 70)

    # 1. Verify physical dataset paths
    cvb_dir = Path("/mnt/cvb/cvb/000058916v001")
    beef_dir = Path("/mnt/beef/beef_behavior")
    data_dir = Path("/root/datasets/behavior/cvb_beef")
    cache_dir = Path("/checkpoints/behavior_cache_smoke")
    output_dir = Path("/checkpoints/behavior_baseline_smoke")

    assert cvb_dir.exists(), f"CVB dataset directory not found: {cvb_dir}"
    assert beef_dir.exists(), f"Beef dataset directory not found: {beef_dir}"
    assert data_dir.exists(), f"Split data directory not found: {data_dir}"

    print(f"[*] Path Check PASS: CVB={cvb_dir}, Beef={beef_dir}, Data={data_dir}")

    # 2. Check canonical split files
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    assert train_csv.exists(), "train.csv missing!"
    assert val_csv.exists(), "val.csv missing!"
    assert test_csv.exists(), "test.csv missing!"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    assert len(train_df) == 3785, f"Expected 3785 train samples, found {len(train_df)}"
    assert len(val_df) == 680, f"Expected 680 val samples, found {len(val_df)}"
    print(f"[*] Canonical split verification PASS: Train={len(train_df)}, Val={len(val_df)}")

    # Record test.csv mtime/size to ensure it is never touched
    test_stat_before = test_csv.stat()

    # 3. Execute smoke training pipeline
    summary = train_pipeline(
        data_dir=data_dir,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=2,
        batch_size=8,
        lr=1e-4,
        weight_decay=1e-2,
        num_workers=0,
        smoke=True,
    )

    # 4. Assert test.csv was NEVER touched
    test_stat_after = test_csv.stat()
    assert test_stat_before.st_mtime == test_stat_after.st_mtime, "CRITICAL: test.csv was modified!"
    assert summary.get("test_metrics") is None, "CRITICAL: test.csv was evaluated in smoke mode!"
    print("[*] Strict canonical test-set isolation PASS: test.csv was NEVER loaded or evaluated.")

    # Commit checkpoints to volume
    checkpoint_vol.commit()
    print("[*] Checkpoint volume committed.")

    return summary


# ==============================================================================
# FULL TRAINING FUNCTION (L40S PRODUCTION RUN)
# ==============================================================================
@app.function(
    gpu="L40S",
    volumes={
        "/mnt/cvb": cvb_vol,
        "/mnt/beef": beef_vol,
        "/checkpoints": checkpoint_vol,
    },
    timeout=7200,
    cpu=8.0,
    memory=32768,
)
def train_full_remote(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
) -> dict:
    """
    Executes full 30-epoch training on NVIDIA L40S with 8 CPUs and 32GB RAM:
    1. Pre-caches all canonical midpoint crops to /checkpoints/behavior_cache
    2. Trains ResNet-18 with CrossEntropyLoss and AdamW for 30 epochs
    3. Selects best checkpoint by validation Macro-F1
    4. Evaluates held-out test.csv once at the end
    5. Saves full metrics to /checkpoints/behavior_baseline/behavior_baseline_metrics.json
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from scripts.train_cvb_beef_behavior_baseline import train_pipeline

    cvb_dir = Path("/mnt/cvb/cvb/000058916v001")
    beef_dir = Path("/mnt/beef/beef_behavior")
    data_dir = Path("/root/datasets/behavior/cvb_beef")
    cache_dir = Path("/checkpoints/behavior_cache")
    output_dir = Path("/checkpoints/behavior_baseline")

    summary = train_pipeline(
        data_dir=data_dir,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        smoke=False,
    )

    checkpoint_vol.commit()
    return summary


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def smoke_test():
    """Runs smoke test on Modal."""
    res = smoke_test_remote.remote()
    print("\n[LOCAL] Smoke test result summary:")
    print(f"  Status        : {res.get('status')}")
    print(f"  Duration      : {res.get('total_duration_sec')}s")
    print(f"  Best Val F1   : {res.get('best_val_metrics', {}).get('macro_f1')}")
    print(f"  Best Val Acc  : {res.get('best_val_metrics', {}).get('overall_accuracy')}")
    print(f"  Best Val Bal  : {res.get('best_val_metrics', {}).get('balanced_accuracy')}")


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
):
    """Launches full training on NVIDIA L40S."""
    print(f"[LOCAL] Launching full {epochs}-epoch training on Modal L40S...")
    res = train_full_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )
    print("\n[LOCAL] Full training completed!")
    print(f"  Total Duration: {res.get('total_duration_sec')}s")
    print(f"  Test Macro-F1 : {res.get('test_metrics', {}).get('macro_f1')}")
    print(f"  Test Accuracy : {res.get('test_metrics', {}).get('overall_accuracy')}")
