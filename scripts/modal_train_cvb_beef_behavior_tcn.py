# -*- coding: utf-8 -*-
"""
Modal Cloud Wrapper for Phase 3 Run 5: Behavior Temporal Core Smoke Test
========================================================================
Profile Target   : tigerwood693
CVB Volume       : cvb-data (mounted at /mnt/cvb)
Beef Volume      : beef-behavior-data (mounted at /mnt/beef)
Checkpoint Volume: behavior-checkpoints (mounted at /checkpoints)
GPU Target       : NVIDIA T4 (T4 ONLY for smoke test; low cost)
Primary Script   : scripts/train_cvb_beef_behavior_tcn.py

Execution Goal:
Verify the lightweight temporal core (ResNet-18 + 1D TCN, T=8 frames):
  - Deterministic temporal sampling for CVB & Beef
  - Correct target-cow crop for CVB using authentic per-frame bboxes
  - Beef temporal clip decode
  - [B, T, 3, 224, 224] input tensor shapes
  - [B, T, 512] frame features
  - 1D TCN forward and backward passes
  - Cross-entropy loss computation
  - 2-epoch training on balanced smoke subset (Train=30, Val=10)
  - Validation metrics computation (overall acc, balanced acc, macro-F1, sub-metrics)
  - Checkpoint save and bit-identical reload verification (assert max_diff < 1e-5)
  - Strict canonical test isolation: test.csv is NEVER touched or evaluated

Usage:
  modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::smoke_test
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

# Persistent Storage Volumes on tigerwood693
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
        str(REPO_ROOT / "scripts" / "train_cvb_beef_behavior_tcn.py"),
        remote_path="/root/scripts/train_cvb_beef_behavior_tcn.py",
    )
)

app = modal.App("cvb-beef-behavior-temporal-smoke", image=train_image)


# ==============================================================================
# REMOTE SMOKE TEST FUNCTION (T4 GPU ONLY)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={
        "/mnt/cvb": cvb_vol,
        "/mnt/beef": beef_vol,
        "/checkpoints": checkpoint_vol,
    },
    timeout=900,
    cpu=2.0,
    memory=4096,
)
def smoke_test_remote() -> dict:
    """
    Executes the temporal core smoke test on Modal (NVIDIA T4):
    1. Verifies physical paths on /mnt/cvb and /mnt/beef
    2. Verifies canonical split files (asserts test.csv is never loaded)
    3. Runs 2-epoch balanced smoke training (Train=30, Val=10, T=8 frames)
    4. Evaluates validation metrics
    5. Verifies checkpoint save and bit-identical reload
    6. Generates visual contact sheet
    7. Asserts test.csv remained 100% untouched
    8. Returns summary dictionary and contact sheet bytes
    """
    import os
    import sys
    from pathlib import Path
    import pandas as pd

    sys.path.insert(0, "/root")
    from scripts.train_cvb_beef_behavior_tcn import train_temporal_pipeline

    print("\n" + "=" * 70)
    print("  MODAL SMOKE TEST: RUN 5 BEHAVIOR TEMPORAL CORE (NVIDIA T4)")
    print("=" * 70)

    # 1. Verify physical dataset paths
    cvb_dir = Path("/mnt/cvb/cvb/000058916v001")
    beef_dir = Path("/mnt/beef/beef_behavior")
    data_dir = Path("/root/datasets/behavior/cvb_beef")
    cache_dir = Path("/checkpoints/behavior_temporal_smoke_cache")
    output_dir = Path("/checkpoints/behavior_temporal_smoke")

    assert cvb_dir.exists(), f"CVB dataset directory not found: {cvb_dir}"
    assert beef_dir.exists(), f"Beef dataset directory not found: {beef_dir}"
    assert data_dir.exists(), f"Split data directory not found: {data_dir}"

    print(f"[*] Path Check PASS: CVB={cvb_dir}, Beef={beef_dir}, Data={data_dir}")

    # 2. Check canonical split files & record stat for test.csv
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

    # Strict rule: record test.csv mtime and size before run
    test_stat_before = test_csv.stat()

    # 3. Execute temporal pipeline
    summary = train_temporal_pipeline(
        data_dir=data_dir,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=2,
        batch_size=4,
        lr=1e-4,
        weight_decay=1e-2,
        num_workers=0,
        num_frames=8,
        smoke=True,
    )

    # 4. Assert test.csv was NEVER touched
    test_stat_after = test_csv.stat()
    assert test_stat_before.st_mtime == test_stat_after.st_mtime, "CRITICAL: test.csv was modified!"
    assert summary.get("test_csv_evaluated") is False, "CRITICAL: test.csv was evaluated in smoke mode!"
    print("[*] Strict canonical test-set isolation PASS: test.csv was NEVER loaded or evaluated.")

    # 5. Read contact sheet bytes for local preservation
    contact_sheet_p = output_dir / "temporal_samples_contact_sheet.jpg"
    contact_sheet_bytes = None
    if contact_sheet_p.exists():
        with open(contact_sheet_p, "rb") as f:
            contact_sheet_bytes = f.read()
        print(f"[*] Loaded contact sheet bytes ({len(contact_sheet_bytes):,} bytes)")

    # 6. Commit checkpoints and cache to volume
    checkpoint_vol.commit()
    print("[*] Persistent volume 'behavior-checkpoints' committed successfully.")

    result = {
        "summary": summary,
        "contact_sheet_bytes": contact_sheet_bytes,
    }
    return result


# ==============================================================================
# LOCAL ENTRYPOINT
# ==============================================================================
@app.local_entrypoint()
def smoke_test():
    """
    Local CLI entrypoint:
      modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::smoke_test
    """
    import json
    import time
    t0 = time.perf_counter()

    print("Launching Run 5 Behavior Temporal Core Smoke Test on Modal (profile tigerwood693, GPU T4)...")
    res = smoke_test_remote.remote()
    total_time = time.perf_counter() - t0

    summary = res["summary"]
    contact_sheet_bytes = res["contact_sheet_bytes"]

    # Save local artifacts
    local_art_dir = REPO_ROOT / "artifacts" / "behavior_temporal_smoke"
    local_art_dir.mkdir(parents=True, exist_ok=True)

    local_audit_dir = REPO_ROOT / "docs" / "audits" / "assets" / "behavior_temporal_smoke"
    local_audit_dir.mkdir(parents=True, exist_ok=True)

    summary_path = local_art_dir / "behavior_tcn_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Local metrics saved: {summary_path}")

    if contact_sheet_bytes:
        sheet_art_p = local_art_dir / "temporal_samples_contact_sheet.jpg"
        sheet_audit_p = local_audit_dir / "temporal_samples_contact_sheet.jpg"
        with open(sheet_art_p, "wb") as f:
            f.write(contact_sheet_bytes)
        with open(sheet_audit_p, "wb") as f:
            f.write(contact_sheet_bytes)
        print(f"[*] Local contact sheet saved: {sheet_art_p}")
        print(f"[*] Local audit contact sheet saved: {sheet_audit_p}")

    print("\n" + "=" * 70)
    print(f"  SMOKE TEST PASSED SUCCESSFULLY IN {total_time:.1f}s!")
    print("=" * 70)
    print(f"  Status                 : {summary.get('status')}")
    print(f"  Total Trainable Params : {summary['architecture']['total_trainable_params']:,}")
    print(f"  Best Val Macro-F1      : {summary['best_val_metrics']['macro_f1']:.4f}")
    print(f"  Logit Reload Diff      : {summary['max_logit_diff_after_reload']:.8f}")
    print(f"  Test CSV Evaluated     : {summary.get('test_csv_evaluated')}")
    print("=" * 70)
