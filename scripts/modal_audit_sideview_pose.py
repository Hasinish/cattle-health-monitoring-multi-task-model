# -*- coding: utf-8 -*-
"""Modal Cloud Wrapper for SideViewCows2026 Pose Feasibility Audit.

Target Profile   : tigerwood697
Dataset Volume   : sideview-data (mounted at /data)
Checkpoint Volume: reid-checkpoints (mounted at /checkpoints)
GPU              : Tesla T4 (low-cost evaluation)

Usage:
    modal run --profile tigerwood697 scripts/modal_audit_sideview_pose.py
"""

import json
import os
import sys
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
    print("[ERROR] modal package not found")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

data_vol = modal.Volume.from_name("sideview-data")
checkpoint_vol = modal.Volume.from_name("reid-checkpoints", create_if_missing=True)

# Image with DeepLabCut and SuperAnimal
pose_image = (
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
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_baseline.py"),
        remote_path="/root/scripts/train_sideview_reid_baseline.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_perception.py"),
        remote_path="/root/scripts/train_sideview_reid_perception.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "audit_sideview_pose_feasibility.py"),
        remote_path="/root/scripts/audit_sideview_pose_feasibility.py",
    )
)

app = modal.App("sideview-pose-feasibility-audit", image=pose_image)


@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=1200,
    cpu=4.0,
    memory=16384,
)
def run_pose_audit_remote(sample_size: int = 50, seed: int = 2026) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from scripts.audit_sideview_pose_feasibility import audit_pose_feasibility

    output_dir = Path("/checkpoints/sideview_pose_feasibility")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = audit_pose_feasibility(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
        device_str="cuda",
    )

    checkpoint_vol.commit()

    # Read contact sheet and csv bytes to return locally
    contact_sheet_path = output_dir / "sideview_pose_contact_sheet.jpg"
    contact_sheet_bytes = contact_sheet_path.read_bytes() if contact_sheet_path.exists() else b""

    keypoints_csv_path = output_dir / "pose_keypoints_summary.csv"
    keypoints_csv_str = keypoints_csv_path.read_text(encoding="utf-8") if keypoints_csv_path.exists() else ""

    return {
        "summary": summary,
        "contact_sheet_bytes": contact_sheet_bytes,
        "keypoints_csv_str": keypoints_csv_str,
    }


@app.local_entrypoint()
def main(sample_size: int = 50, seed: int = 2026):
    print(f"\n[LOCAL] Dispatching SideView Pose Feasibility Audit to Modal (tigerwood697)...")
    res = run_pose_audit_remote.remote(sample_size=sample_size, seed=seed)
    summary = res["summary"]

    # Save to local artifacts
    local_out = REPO_ROOT / "artifacts" / "reid_pose_ablation"
    local_out.mkdir(parents=True, exist_ok=True)

    json_path = local_out / "pose_feasibility_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[LOCAL] Saved local metrics JSON: {json_path}")

    if res.get("keypoints_csv_str"):
        csv_path = local_out / "pose_keypoints_summary.csv"
        csv_path.write_text(res["keypoints_csv_str"], encoding="utf-8")
        print(f"[LOCAL] Saved local keypoints CSV: {csv_path}")

    if res.get("contact_sheet_bytes"):
        sheet_path = local_out / "sideview_pose_contact_sheet.jpg"
        sheet_path.write_bytes(res["contact_sheet_bytes"])
        print(f"[LOCAL] Saved local visual contact sheet: {sheet_path}")

    print("\n[LOCAL] Step B Pose Feasibility Audit Complete! Summary:")
    print(f"  - Evaluated: {summary['total_evaluated']} crops")
    print(f"  - Output Return Rate: {summary['pose_output_returned_rate']}%")
    print(f"  - Detector Failure Rate: {summary['pose_detector_failure_rate']}%")
    print(f"  - Mean Confidence: {summary['mean_raw_confidence']}")
    print(f"  - Keypoints Inside Mask (Sanity): {summary['mean_keypoints_inside_mask_rate']}%")
