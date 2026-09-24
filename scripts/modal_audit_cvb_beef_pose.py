# -*- coding: utf-8 -*-
"""Modal Cloud Wrapper for CVB + Kaggle Beef SuperAnimal Pose Feasibility Audit.

Target Profile   : hasinishrak2015
Mounted Volumes  :
  - mtl-data        -> /mtl-data        (Contains /mtl-data/behavior/ with 4,271 sequence folders & manifests)
  - mtl-checkpoints -> /mtl-checkpoints (Stores audit metrics, CSVs, and contact sheets)

Hardware Tier    :
  - verify_readiness: CPU only (cpu=2.0, memory=4096 MB, NO GPU)
  - run_audit       : NVIDIA Tesla T4 (Low-cost cloud GPU tier, cpu=4.0, memory=16384 MB)

Usage:
  1. Zero-GPU Readiness Verification:
     modal run --profile hasinishrak2015 scripts/modal_audit_cvb_beef_pose.py::verify_readiness

  2. Cloud GPU Pose Feasibility Audit (Tesla T4, 54 Sequences x 8 Frames = 432 Frames):
     modal run --profile hasinishrak2015 scripts/modal_audit_cvb_beef_pose.py::run_audit

  3. Full Pipeline with Local Artifact Download:
     modal run --profile hasinishrak2015 scripts/modal_audit_cvb_beef_pose.py::main
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Windows path patch for Modal client
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

# Persistent Storage Volumes on profile hasinishrak2015
mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=False)
mtl_checkpoints_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=True)

# Container image with official SuperAnimal / DeepLabCut dependencies
pose_audit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "deeplabcut>=3.0.0",
        "dlclibrary",
        "opencv-python-headless",
        "pillow",
        "pandas",
        "numpy<2.0.0",
        "scikit-learn",
        "tqdm",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "audit_cvb_beef_pose_feasibility.py"),
        remote_path="/root/scripts/audit_cvb_beef_pose_feasibility.py",
    )
)

app = modal.App("cvb-beef-behavior-pose-audit", image=pose_audit_image)


# ==============================================================================
# 1. READINESS VERIFICATION (ZERO GPU COST)
# ==============================================================================
@app.function(
    volumes={"/mtl-data": mtl_data_vol, "/mtl-checkpoints": mtl_checkpoints_vol},
    timeout=300,
    cpu=2.0,
    memory=4096,
)
def verify_readiness() -> Dict[str, Any]:
    """Verifies volume mounts, Behavior manifests, sequence directories, and frame formats."""
    import pandas as pd

    print("=" * 80)
    print("  STEP 1: PRE-FLIGHT READINESS VERIFICATION (PROFILE: hasinishrak2015)")
    print("=" * 80)

    behavior_dir = Path("/mtl-data/behavior")
    train_csv = behavior_dir / "retained_train.csv"
    val_csv = behavior_dir / "retained_val.csv"

    if not behavior_dir.exists():
        raise FileNotFoundError(f"/mtl-data/behavior does not exist on mtl-data volume!")
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing retained manifests in {behavior_dir}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    print(f"[OK] Manifests verified: {len(df_train)} train, {len(df_val)} val sequences.")

    # Verify all 9 source x class cells
    expected_cells = [
        ("cvb", "Standing"),
        ("cvb", "Lying"),
        ("cvb", "Feeding"),
        ("cvb", "Drinking"),
        ("cvb", "Walking"),
        ("beef_cattle_behavior", "Standing"),
        ("beef_cattle_behavior", "Lying"),
        ("beef_cattle_behavior", "Feeding"),
        ("beef_cattle_behavior", "Drinking"),
    ]

    cell_counts = {}
    for src, cls in expected_cells:
        t_cnt = len(df_train[(df_train["dataset"] == src) & (df_train["behavior_canonical"] == cls)])
        v_cnt = len(df_val[(df_val["dataset"] == src) & (df_val["behavior_canonical"] == cls)])
        cell_counts[f"{src}_{cls}"] = {"train": t_cnt, "val": v_cnt}
        print(f"     - [{src:20s} x {cls:8s}]: {t_cnt:4d} train, {v_cnt:4d} val")
        if t_cnt == 0 or v_cnt == 0:
            raise AssertionError(f"Cell {src} x {cls} has 0 sequences!")

    # Check a few random sequence folders for frame_00..07 and mask_00..07
    checked_seqs = 0
    for sample_id in df_train["sample_id"].head(5):
        s_dir = behavior_dir / str(sample_id)
        if not s_dir.exists():
            raise FileNotFoundError(f"Sample folder {s_dir} not found")
        for t in range(8):
            f_p = s_dir / f"frame_{t:02d}.jpg"
            m_p = s_dir / f"mask_{t:02d}.png"
            if not f_p.exists() or f_p.stat().st_size == 0:
                raise FileNotFoundError(f"Missing frame {f_p}")
            if not m_p.exists() or m_p.stat().st_size == 0:
                raise FileNotFoundError(f"Missing mask {m_p}")
        checked_seqs += 1

    print(f"[OK] Verified physical frame and mask integrity on spot-checked sequences ({checked_seqs} seqs).")

    # Writable probe on /mtl-checkpoints
    chk_dir = Path("/mtl-checkpoints/behavior_pose_audit")
    chk_dir.mkdir(parents=True, exist_ok=True)
    probe_file = chk_dir / ".probe.tmp"
    probe_file.write_text("ok", encoding="utf-8")
    probe_file.unlink()
    print("[OK] Verified writable volume /mtl-checkpoints/behavior_pose_audit.")

    return {
        "status": "CERTIFIED_READY_FOR_AUDIT",
        "train_sequences": len(df_train),
        "val_sequences": len(df_val),
        "cell_counts": cell_counts,
    }


# ==============================================================================
# 2. RUN AUDIT (LOW-COST TESLA T4 GPU)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/mtl-data": mtl_data_vol, "/mtl-checkpoints": mtl_checkpoints_vol},
    timeout=1800,
    cpu=4.0,
    memory=16384,
)
def run_audit(seqs_per_cell: int = 6, seed: int = 2026) -> Dict[str, Any]:
    """Executes the full SuperAnimal pose feasibility audit on Tesla T4."""
    from scripts.audit_cvb_beef_pose_feasibility import audit_cvb_beef_pose

    print("=" * 80)
    print("  STEP 2: LAUNCHING REMOTE POSE FEASIBILITY AUDIT ON TESLA T4")
    print("=" * 80)

    output_dir = Path("/mtl-checkpoints/behavior_pose_audit")
    metrics = audit_cvb_beef_pose(
        behavior_data_dir=Path("/mtl-data/behavior"),
        output_dir=output_dir,
        sample_seed=seed,
        seqs_per_cell=seqs_per_cell,
        device_str="cuda",
    )

    # Commit persistent volume
    mtl_checkpoints_vol.commit()
    print("[OK] Committed audit outputs to persistent volume mtl-checkpoints.")

    # Read back file bytes for local synchronization
    frame_csv_bytes = (output_dir / "behavior_pose_frame_results.csv").read_bytes()
    seq_csv_bytes = (output_dir / "behavior_pose_sequence_summary.csv").read_bytes()
    metrics_json_bytes = (output_dir / "behavior_pose_aggregate_metrics.json").read_bytes()
    schema_json_bytes = (output_dir / "superanimal_39_keypoints_schema.json").read_bytes()

    contact_sheet_bytes = None
    contact_sheet_path = output_dir / "behavior_pose_all_classes_contact_sheet.jpg"
    if contact_sheet_path.exists():
        contact_sheet_bytes = contact_sheet_path.read_bytes()

    return {
        "metrics": metrics,
        "files": {
            "behavior_pose_frame_results.csv": frame_csv_bytes,
            "behavior_pose_sequence_summary.csv": seq_csv_bytes,
            "behavior_pose_aggregate_metrics.json": metrics_json_bytes,
            "superanimal_39_keypoints_schema.json": schema_json_bytes,
            "behavior_pose_all_classes_contact_sheet.jpg": contact_sheet_bytes,
        },
    }


# ==============================================================================
# 3. LOCAL ORCHESTRATION ENTRYPOINT
# ==============================================================================
@app.local_entrypoint()
def main(seqs_per_cell: int = 6, seed: int = 2026):
    """Local orchestration: verifies readiness, executes audit, downloads all artifacts."""
    print(">>> 1. Verifying readiness...")
    readiness = verify_readiness.remote()
    print(f"[OK] Readiness status: {readiness['status']}")

    print(f"\n>>> 2. Running remote pose feasibility audit (seqs_per_cell={seqs_per_cell}, seed={seed})...")
    result = run_audit.remote(seqs_per_cell=seqs_per_cell, seed=seed)

    print("\n>>> 3. Syncing audit artifacts to local workspace...")
    local_output_dir = REPO_ROOT / "artifacts" / "behavior_pose_audit"
    local_output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in result["files"].items():
        if content is not None:
            out_file = local_output_dir / filename
            out_file.write_bytes(content)
            print(f"     [Saved] {out_file} ({len(content) / 1024:.1f} KB)")

    print(f"\n[SUCCESS] Pose audit completed and 100% synchronized to {local_output_dir}!")


if __name__ == "__main__":
    pass
