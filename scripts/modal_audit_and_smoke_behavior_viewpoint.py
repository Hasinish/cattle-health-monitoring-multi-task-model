# -*- coding: utf-8 -*-
"""Modal Cloud Runner for CVB + Kaggle Beef Behavior Viewpoint Audit & Smoke Certification.

Target Profile   : hasinishrak2015
Mounted Volumes  :
  - mtl-data        -> /mtl-data        (Staged Behavior sequences & manifests)
  - mtl-checkpoints -> /mtl-checkpoints (Stores viewpoint checkpoint, audit outputs, and smoke metrics)

Hardware Tier    :
  - audit_remote     : NVIDIA Tesla T4 (cpu=4.0, memory=16384 MB)
  - smoke_test_remote: NVIDIA Tesla T4 (cpu=4.0, memory=16384 MB)

Usage:
  1. Feasibility Audit Only:
     modal run --profile hasinishrak2015 scripts/modal_audit_and_smoke_behavior_viewpoint.py::run_audit

  2. Smoke Test Only:
     modal run --profile hasinishrak2015 scripts/modal_audit_and_smoke_behavior_viewpoint.py::run_smoke

  3. Full Pipeline (Audit -> Usability Evaluation -> Smoke Test -> Local Sync):
     modal run --profile hasinishrak2015 scripts/modal_audit_and_smoke_behavior_viewpoint.py::main
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
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
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=False)
mtl_checkpoints_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=True)

viewpoint_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "opencv-python-headless",
        "pillow",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_cvb_beef_behavior_tcn.py"),
        remote_path="/root/scripts/train_cvb_beef_behavior_tcn.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "audit_cvb_beef_viewpoint_transfer.py"),
        remote_path="/root/scripts/audit_cvb_beef_viewpoint_transfer.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_cvb_beef_behavior_viewpoint.py"),
        remote_path="/root/scripts/train_cvb_beef_behavior_viewpoint.py",
    )
)

app = modal.App("cvb-beef-behavior-viewpoint-suite", image=viewpoint_image)


# ==============================================================================
# 1. VIEWPOINT AUDIT REMOTE FUNCTION
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/mtl-data": mtl_data_vol, "/mtl-checkpoints": mtl_checkpoints_vol},
    timeout=900,
    cpu=4.0,
    memory=16384,
)
def run_audit(seqs_per_cell: int = 6, seed: int = 2026) -> Dict[str, Any]:
    """Executes the transfer sanity audit of frozen viewpoint model on Tesla T4."""
    from scripts.audit_cvb_beef_viewpoint_transfer import audit_cvb_beef_viewpoint

    print("=" * 80)
    print("  STEP 1: REMOTE VIEWPOINT TRANSFER SANITY AUDIT (TESLA T4)")
    print("=" * 80)

    behavior_dir = Path("/mtl-data/behavior")
    checkpoint_path = Path("/mtl-checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth")
    output_dir = Path("/mtl-checkpoints/behavior_viewpoint_audit")

    metrics = audit_cvb_beef_viewpoint(
        behavior_data_dir=behavior_dir,
        viewpoint_checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        sample_seed=seed,
        seqs_per_cell=seqs_per_cell,
        device_str="cuda",
    )

    mtl_checkpoints_vol.commit()
    print("[OK] Committed viewpoint audit outputs to mtl-checkpoints volume.")

    # Read back bytes for local sync
    frame_csv_bytes = (output_dir / "behavior_viewpoint_frame_results.csv").read_bytes()
    seq_csv_bytes = (output_dir / "behavior_viewpoint_sequence_summary.csv").read_bytes()
    metrics_json_bytes = (output_dir / "behavior_viewpoint_transfer_sanity_metrics.json").read_bytes()

    contact_sheet_bytes = None
    cs_path = output_dir / "behavior_viewpoint_all_classes_contact_sheet.jpg"
    if cs_path.exists():
        contact_sheet_bytes = cs_path.read_bytes()

    return {
        "metrics": metrics,
        "files": {
            "behavior_viewpoint_frame_results.csv": frame_csv_bytes,
            "behavior_viewpoint_sequence_summary.csv": seq_csv_bytes,
            "behavior_viewpoint_transfer_sanity_metrics.json": metrics_json_bytes,
            "behavior_viewpoint_all_classes_contact_sheet.jpg": contact_sheet_bytes,
        },
    }


# ==============================================================================
# 2. RUN 5 + VIEWPOINT SMOKE TEST REMOTE FUNCTION
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/mtl-data": mtl_data_vol, "/mtl-checkpoints": mtl_checkpoints_vol},
    timeout=900,
    cpu=4.0,
    memory=16384,
)
def run_smoke(epochs: int = 2, batch_size: int = 4) -> Dict[str, Any]:
    """Executes the 2-epoch controlled Run 5 + Viewpoint smoke test on Tesla T4."""
    from scripts.train_cvb_beef_behavior_viewpoint import run_behavior_viewpoint_smoke_test

    print("=" * 80)
    print("  STEP 2: REMOTE RUN 5 + VIEWPOINT SMOKE CERTIFICATION (TESLA T4)")
    print("=" * 80)

    behavior_dir = Path("/mtl-data/behavior")
    checkpoint_path = Path("/mtl-checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth")
    output_dir = Path("/mtl-checkpoints/behavior_viewpoint_smoke")

    smoke_metrics = run_behavior_viewpoint_smoke_test(
        behavior_data_dir=behavior_dir,
        viewpoint_checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        device_str="cuda",
    )

    mtl_checkpoints_vol.commit()
    print("[OK] Committed smoke checkpoint and metrics to mtl-checkpoints volume.")

    metrics_json_bytes = (output_dir / "behavior_viewpoint_smoke_metrics.json").read_bytes()
    return {
        "metrics": smoke_metrics,
        "files": {
            "behavior_viewpoint_smoke_metrics.json": metrics_json_bytes,
        },
    }


# ==============================================================================
# 3. LOCAL ORCHESTRATION ENTRYPOINT
# ==============================================================================
@app.local_entrypoint()
def main(seqs_per_cell: int = 6, seed: int = 2026, epochs: int = 2):
    """Full pipeline: runs audit -> checks usability -> executes smoke -> syncs artifacts."""
    print("=" * 80)
    print(">>> 1. LAUNCHING VIEWPOINT TRANSFER SANITY AUDIT...")
    print("=" * 80)
    audit_res = run_audit.remote(seqs_per_cell=seqs_per_cell, seed=seed)

    audit_local_dir = REPO_ROOT / "artifacts" / "behavior_viewpoint_audit"
    audit_local_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in audit_res["files"].items():
        if content is not None:
            (audit_local_dir / fname).write_bytes(content)
            print(f"     [Saved] {audit_local_dir / fname} ({len(content) / 1024:.1f} KB)")

    audit_metrics = audit_res["metrics"]
    is_usable = audit_metrics["overall"]["is_non_degenerate"]
    print(f"\n[EVALUATION] Viewpoint Transfer Technical Usability: {'USABLE' if is_usable else 'DEGENERATE'}")

    if not is_usable:
        print("[WARN] Viewpoint audit detected degenerate distributions. Skipping smoke test.")
        return

    print("\n" + "=" * 80)
    print(f">>> 2. LAUNCHING CONTROLLED RUN 5 + VIEWPOINT SMOKE TEST ({epochs} EPOCHS)...")
    print("=" * 80)
    smoke_res = run_smoke.remote(epochs=epochs, batch_size=4)

    smoke_local_dir = REPO_ROOT / "artifacts" / "behavior_viewpoint_smoke"
    smoke_local_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in smoke_res["files"].items():
        if content is not None:
            (smoke_local_dir / fname).write_bytes(content)
            print(f"     [Saved] {smoke_local_dir / fname} ({len(content) / 1024:.1f} KB)")

    print("\n[SUCCESS] Viewpoint audit and smoke certification completed and synced locally!")


if __name__ == "__main__":
    pass
