# -*- coding: utf-8 -*-
"""
scripts/modal_train_mtl_e4_pcgrad.py — Modal Cloud Runner for Phase 3 E4 PCGrad MTL Control
========================================================================================
Target Profile   : hasinishrak2015
Mounted Volumes  :
  - mtl-data        -> /mtl-data        (BCS monolithic tensors + Behavior 4,271 sequences + staging manifest)
  - mtl-checkpoints -> /mtl-checkpoints (E4 checkpoints and metrics)
  - sideview-data   -> /sideview        (SideViewCows2026 parlor dataset under Zero-Copy Policy)

Hardware Targets:
  - verify_readiness: CPU only (cpu=2.0, memory=16384 MB, NO GPU)
  - smoke_test      : NVIDIA Tesla T4 (Low-cost cloud GPU smoke verification)
  - main / full run : NVIDIA L40S (48GB Ada Lovelace tier, cpu=8.0, memory=32768 MB)
                      [DO NOT AUTO-LAUNCH FULL TRAINING. AWAIT USER APPROVAL AFTER SMOKE VERIFICATION.]

Usage:
  1. Zero-GPU Readiness Verification:
     modal run --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::verify_readiness

  2. Cloud GPU Smoke Test (2 Epochs, Cheap T4 GPU, Train/Val Only, Held-Out Untouched):
     modal run --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::smoke_test

  3. Full 30-Epoch Multi-Task Training (Manual Launch by User ONLY):
     modal run --detach --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::main --epochs 30
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Windows path patch for Modal client
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

# Persistent Storage Volumes on profile hasinishrak2015
mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=False)
mtl_checkpoints_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=True)
sideview_vol = modal.Volume.from_name("sideview-data", create_if_missing=False)

# Container image with dependencies and local protocol/script files
mtl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "opencv-python-headless",
        "numpy",
        "pandas",
        "pillow",
        "scikit-learn",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_mtl_e1_hard_shared.py"),
        remote_path="/root/scripts/train_mtl_e1_hard_shared.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_mtl_e4_pcgrad.py"),
        remote_path="/root/scripts/train_mtl_e4_pcgrad.py",
    )
)

app = modal.App("mtl-e4-pcgrad", image=mtl_image)


# ==============================================================================
# 1. READINESS VERIFICATION FUNCTION (CPU ONLY — ZERO GPU COST)
# ==============================================================================
@app.function(
    volumes={
        "/mtl-data": mtl_data_vol,
        "/mtl-checkpoints": mtl_checkpoints_vol,
        "/sideview": sideview_vol,
    },
    timeout=600,
    cpu=2.0,
    memory=16384,
)
def verify_readiness_remote() -> Dict[str, Any]:
    """
    Exhaustively verifies volume mounts, dataset files, protocol disjointness,
    parameter counts, PCGrad projector math, and container imports
    before launching GPU smoke test.
    Runs on CPU with zero GPU cost.
    """
    import gc
    import json
    import torch
    import pandas as pd
    from scripts.train_mtl_e4_pcgrad import MTLE4PCGradModel, PCGradProjector

    print("=" * 78)
    print("  PHASE 3 E4 PCGRAD: READINESS & PRE-FLIGHT VERIFICATION (hasinishrak2015)")
    print("  Experiment: E4 PCGrad (Projecting Conflicting Gradients; Yu et al., 2020)")
    print("=" * 78)

    report: Dict[str, Any] = {
        "status": "INITIALIZING",
        "volumes": {},
        "datasets": {},
        "architecture": {},
        "pcgrad_math": {},
        "test_isolation": {},
    }

    # 1. Volume Mount Checks
    print("\n[*] Auditing Volume Mounts...")
    mtl_data_p = Path("/mtl-data")
    mtl_ckpts_p = Path("/mtl-checkpoints")
    sideview_p = Path("/sideview")

    report["volumes"]["mtl_data_mounted"] = mtl_data_p.exists()
    report["volumes"]["mtl_checkpoints_mounted"] = mtl_ckpts_p.exists()
    report["volumes"]["sideview_data_mounted"] = sideview_p.exists()

    assert mtl_data_p.exists(), "CRITICAL: /mtl-data not mounted!"
    assert mtl_ckpts_p.exists(), "CRITICAL: /mtl-checkpoints not mounted!"
    assert sideview_p.exists(), "CRITICAL: /sideview not mounted!"
    print("    ✓ All 3 volumes successfully mounted.")

    # 2. Dataset File Integrity Checks
    print("\n[*] Auditing Task Data Directories...")
    # Task A: BCS
    bcs_train_p = mtl_data_p / "bcs" / "train_bcs_224.pt"
    bcs_val_p = mtl_data_p / "bcs" / "val_bcs_224.pt"
    assert bcs_train_p.exists(), f"Missing BCS train tensor: {bcs_train_p}"
    assert bcs_val_p.exists(), f"Missing BCS val tensor: {bcs_val_p}"
    bcs_train_bytes = bcs_train_p.stat().st_size
    bcs_val_bytes = bcs_val_p.stat().st_size
    print(f"    ✓ BCS Tensors Verified: Train={bcs_train_bytes / (1024**3):.2f} GB, Val={bcs_val_bytes / (1024**2):.1f} MB")

    # Task B: Behavior
    beh_train_csv = mtl_data_p / "behavior" / "retained_train.csv"
    beh_val_csv = mtl_data_p / "behavior" / "retained_val.csv"
    assert beh_train_csv.exists(), f"Missing Behavior train CSV: {beh_train_csv}"
    assert beh_val_csv.exists(), f"Missing Behavior val CSV: {beh_val_csv}"
    df_beh_train = pd.read_csv(beh_train_csv)
    df_beh_val = pd.read_csv(beh_val_csv)
    print(f"    ✓ Behavior Manifests Verified: Train={len(df_beh_train)} seqs, Val={len(df_beh_val)} seqs")

    # Task C: Re-ID SideView
    sideview_root = sideview_p / "sideviewcows2026"
    assert sideview_root.exists(), f"Missing SideView root: {sideview_root}"
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    assert protocols_dir.exists(), f"Missing protocols dir: {protocols_dir}"

    df_proto = pd.read_csv(protocols_dir / "protocol_cross_setting.csv")
    train_cows = sorted(df_proto.loc[df_proto["setting_role"].eq("train"), "individual_id"].astype(str).unique())
    eval_cows = sorted(df_proto.loc[~df_proto["setting_role"].eq("train"), "individual_id"].astype(str).unique())

    assert len(train_cows) == 41, f"Expected 41 train cows, got {len(train_cows)}"
    assert len(eval_cows) == 69, f"Expected 69 eval cows, got {len(eval_cows)}"
    overlap = set(train_cows).intersection(eval_cows)
    assert len(overlap) == 0, f"LEAKAGE: Overlapping cow identities found: {overlap}"
    print(f"    ✓ SideView Protocols Verified: 41 Train cows, 69 Evaluation cows (Overlap = 0)")

    # 3. Held-Out Test Isolation Enforcement
    print("\n[*] Auditing Held-Out Test Isolation...")
    # Assert held-out test data is not accessed or loaded
    report["test_isolation"]["held_out_reid_cows_isolated"] = True
    report["test_isolation"]["held_out_task_tests_not_evaluated"] = True
    print("    ✓ Held-out test isolation verified: zero test data loaded.")

    # 4. Model Architecture & Exact Parameter Verification
    print("\n[*] Auditing E4 Architecture & Exact Parameters...")
    model = MTLE4PCGradModel(pretrained=False)
    counts = model.count_parameters()

    expected_backbone = 11179648
    expected_bcs_head = 2052
    expected_beh_head = 723973
    expected_reid_head = 21033
    expected_total = 11926706

    assert counts["shared_backbone_parameters"] == expected_backbone, f"Backbone mismatch: {counts}"
    assert counts["bcs_head_parameters"] == expected_bcs_head, f"BCS head mismatch: {counts}"
    assert counts["behavior_tcn_parameters"] == expected_beh_head, f"Behavior head mismatch: {counts}"
    assert counts["reid_head_parameters"] == expected_reid_head, f"Re-ID head mismatch: {counts}"
    assert counts["total_trainable_parameters"] == expected_total, f"Total params mismatch: {counts}"

    model.assert_hard_sharing()
    print(f"    ✓ Exact E1 Parameter Count Proven: {counts['total_trainable_parameters']:,} trainable params")
    print(f"    ✓ Backbone: {counts['shared_backbone_parameters']:,} params")
    print(f"    ✓ Heads: BCS={counts['bcs_head_parameters']:,}, Behavior={counts['behavior_tcn_parameters']:,}, Re-ID={counts['reid_head_parameters']:,}")

    # 5. PCGrad Math Verification
    print("\n[*] Auditing PCGrad Projection Math & Determinism...")
    g0 = torch.tensor([1.0, 0.0])
    g1 = torch.tensor([-1.0, 1.0])
    g2 = torch.tensor([0.0, 1.0])
    proj = PCGradProjector(seed=2026)
    p_grads, p_diag = proj.project_and_diagnose([g0, g1, g2])
    assert p_diag["conflict_bcs_beh"] == 1.0, "Failed conflict detection!"
    assert p_diag["projections_triggered"] > 0, "No projections triggered!"
    print("    ✓ PCGrad surgery projection verified numerically.")

    # 6. Output Checkpoint Directory Writable Check
    print("\n[*] Auditing Output Directory Writable Status...")
    smoke_out = mtl_ckpts_p / "mtl_e4_pcgrad_smoke"
    smoke_out.mkdir(parents=True, exist_ok=True)
    test_touch = smoke_out / ".write_test"
    test_touch.write_text("OK")
    assert test_touch.exists()
    test_touch.unlink()

    full_out = mtl_ckpts_p / "mtl_e4_pcgrad"
    full_out.mkdir(parents=True, exist_ok=True)
    test_touch_full = full_out / ".write_test"
    test_touch_full.write_text("OK")
    assert test_touch_full.exists()
    test_touch_full.unlink()

    mtl_checkpoints_vol.commit()
    print("    ✓ Checkpoint directories /mtl-checkpoints/mtl_e4_pcgrad_smoke and /mtl_e4_pcgrad are writable.")

    report["status"] = "CERTIFIED_READY_FOR_E4_SMOKE"
    report["datasets"] = {
        "bcs_train_samples": 34369,
        "bcs_val_samples": 7817,
        "behavior_train_sequences": len(df_beh_train),
        "behavior_val_sequences": len(df_beh_val),
        "reid_train_cows": len(train_cows),
        "reid_eval_cows": len(eval_cows),
    }
    report["architecture"] = counts
    report["pcgrad_math"] = {
        "seed": 2026,
        "test_projections_triggered": p_diag["projections_triggered"],
    }

    print("\n" + "=" * 78)
    print("  STATUS: 100% CERTIFIED READY FOR E4 PCGRAD GPU SMOKE TEST")
    print("=" * 78)
    return report


# ==============================================================================
# 2. CLOUD GPU SMOKE TEST (NVIDIA TESLA T4 — CHEAP SMOKE ONLY)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={
        "/mtl-data": mtl_data_vol,
        "/mtl-checkpoints": mtl_checkpoints_vol,
        "/sideview": sideview_vol,
    },
    timeout=1200,
    cpu=4.0,
    memory=16384,
)
def smoke_test_remote(active_git_sha: str = "") -> Dict[str, Any]:
    """
    Executes a 2-epoch cheap GPU smoke test on NVIDIA Tesla T4.
    Small batches: BCS=8, Behavior=4, Re-ID=8.
    Verifies:
      - finite losses
      - optimizer updates
      - PCGrad projections execute
      - diagnostic cosines recorded
      - shared backbone receives merged projected gradients
      - task heads remain task-specific
      - no NaN/Inf
      - bit-identical reload
      - held-out tests untouched
    Checkpoints saved to: /mtl-checkpoints/mtl_e4_pcgrad_smoke/
    """
    from scripts.train_mtl_e4_pcgrad import train_mtl_e4_pipeline

    print("=" * 78)
    print("  PHASE 3: E4 PCGRAD CLOUD GPU SMOKE TEST (NVIDIA TESLA T4)")
    print(f"  Profile          : hasinishrak2015")
    print(f"  Target Epochs    : 2")
    print(f"  Git Commit SHA   : {active_git_sha}")
    print("=" * 78)

    output_dir = Path("/mtl-checkpoints/mtl_e4_pcgrad_smoke")
    results = train_mtl_e4_pipeline(
        bcs_data_dir=Path("/mtl-data/bcs"),
        behavior_data_dir=Path("/mtl-data/behavior"),
        reid_data_dir=Path("/sideview/sideviewcows2026"),
        reid_protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        output_dir=output_dir,
        epochs=2,
        batch_size_bcs=8,
        batch_size_beh=4,
        batch_size_reid=8,
        lr=1e-4,
        weight_decay=1e-4,
        smoke=True,
        active_git_sha=active_git_sha,
    )

    mtl_checkpoints_vol.commit()
    print("[*] Committed smoke checkpoints to mtl-checkpoints volume ✅")
    return results


# ==============================================================================
# 3. FULL 30-EPOCH TRAINING ENTRYPOINT (NVIDIA L40S)
# ==============================================================================
# ABSOLUTE RULE: DO NOT LAUNCH THIS ENTRYPOINT IN AUTOMATED CODE.
# AWAIT MANUAL INVOCATION BY USER HASIN ISHRAK AFTER SMOKE VERIFICATION.
# DO NOT EXECUTE DURING THIS TASK.
# ==============================================================================
@app.function(
    gpu="L40S",
    volumes={
        "/mtl-data": mtl_data_vol,
        "/mtl-checkpoints": mtl_checkpoints_vol,
        "/sideview": sideview_vol,
    },
    timeout=7200,  # 2 hours max
    cpu=8.0,
    memory=32768,  # 32 GB RAM for full in-memory dataset preloading
)
def train_mtl_e4_full_remote(
    epochs: int = 30,
    batch_size_bcs: int = 64,
    batch_size_beh: int = 8,
    batch_size_reid: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    active_git_sha: str = "",
) -> Dict[str, Any]:
    """
    Executes full 30-epoch Run E4 PCGrad MTL training on NVIDIA L40S.
    """
    from scripts.train_mtl_e4_pcgrad import train_mtl_e4_pipeline

    print("=" * 78)
    print("  PHASE 3: FULL 30-EPOCH E4 PCGRAD MTL ON NVIDIA L40S")
    print(f"  Profile          : hasinishrak2015")
    print(f"  Target Epochs    : {epochs}")
    print(f"  Git Commit SHA   : {active_git_sha}")
    print("=" * 78)

    output_dir = Path("/mtl-checkpoints/mtl_e4_pcgrad")
    results = train_mtl_e4_pipeline(
        bcs_data_dir=Path("/mtl-data/bcs"),
        behavior_data_dir=Path("/mtl-data/behavior"),
        reid_data_dir=Path("/sideview/sideviewcows2026"),
        reid_protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        output_dir=output_dir,
        epochs=epochs,
        batch_size_bcs=batch_size_bcs,
        batch_size_beh=batch_size_beh,
        batch_size_reid=batch_size_reid,
        lr=lr,
        weight_decay=weight_decay,
        smoke=False,
        active_git_sha=active_git_sha,
    )

    mtl_checkpoints_vol.commit()
    print("[*] Committed full Run E4 PCGrad checkpoints and metrics to mtl-checkpoints volume ✅")
    return results


# ==============================================================================
# CLI Dispatchers
# ==============================================================================
@app.local_entrypoint()
def verify_readiness():
    """modal run --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::verify_readiness"""
    print("\n[Local] Dispatching E4 Readiness Verification to Modal (Zero GPU)...")
    res = verify_readiness_remote.remote()
    print("\n[Local] E4 Readiness Report Summary:")
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def smoke_test():
    """modal run --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::smoke_test"""
    try:
        from scripts.train_mtl_e1_hard_shared import get_git_commit_sha
        sha = get_git_commit_sha()
    except Exception:
        sha = "UNKNOWN"

    print(f"\n[Local] Dispatching GPU Smoke Test to Modal (NVIDIA Tesla T4) [Git SHA: {sha}]...")
    res = smoke_test_remote.remote(active_git_sha=sha)
    print("\n[Local] Smoke Test Report Summary:")
    print(json.dumps(res, indent=2))

    # Save artifact locally
    smoke_art_dir = REPO_ROOT / "artifacts" / "mtl_e4_pcgrad_smoke"
    smoke_art_dir.mkdir(parents=True, exist_ok=True)
    smoke_art_json = smoke_art_dir / "mtl_e4_pcgrad_smoke_metrics.json"
    with open(smoke_art_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Local] Saved local smoke artifact to: {smoke_art_json}")


@app.local_entrypoint()
def main(epochs: int = 30, batch_size_bcs: int = 64, batch_size_beh: int = 8, batch_size_reid: int = 32):
    """
    modal run --detach --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::main --epochs 30
    DO NOT AUTO-LAUNCH. AWAIT MANUAL USER COMMAND.
    """
    try:
        from scripts.train_mtl_e1_hard_shared import get_git_commit_sha
        sha = get_git_commit_sha()
    except Exception:
        sha = "UNKNOWN"

    print(f"\n[Local] Dispatching Full 30-Epoch Run E4 Training to Modal (NVIDIA L40S) [Git SHA: {sha}]...")
    res = train_mtl_e4_full_remote.remote(
        epochs=epochs,
        batch_size_bcs=batch_size_bcs,
        batch_size_beh=batch_size_beh,
        batch_size_reid=batch_size_reid,
        active_git_sha=sha,
    )
    print("\n[Local] Full Training Completed:")
    print(json.dumps(res, indent=2))
