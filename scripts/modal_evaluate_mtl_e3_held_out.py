# -*- coding: utf-8 -*-
"""
scripts/modal_evaluate_mtl_e3_held_out.py — Modal Cloud Runner for MTL E3 Held-Out Evaluation
=============================================================================================
Dispatches the unified held-out evaluation of Run 8 MTL E3 Best Checkpoint
(`/mtl-checkpoints/mtl_e3_modular/mtl_e3_best.pth`) on Modal profile 'hasinishrak2015'.

Evaluates:
  1. BCS on 7,549 ScienceDB test images (exact Run 4 matched identities)
  2. Behavior on 780 sequences (exact Run 5 matched retained population)
  3. Re-ID on SideViewCows2026 Protocol A (69 held-out cows; 36,811 Parlor gallery, 25,260 Barn queries, 607 Snapshots queries)

Usage:
    modal run --profile hasinishrak2015 scripts/modal_evaluate_mtl_e3_held_out.py::run_evaluation
"""

import sys

# Windows UTF-8 patch
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Modal persistent volumes on hasinishrak2015
mtl_checkpoints_vol = modal.Volume.from_name("mtl-checkpoints", create_if_missing=False)
mtl_data_vol = modal.Volume.from_name("mtl-data", create_if_missing=False)
sideview_vol = modal.Volume.from_name("sideview-data", create_if_missing=False)

eval_image = (
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
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_mtl_e3_modular.py"),
        remote_path="/root/scripts/train_mtl_e3_modular.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_perception.py"),
        remote_path="/root/scripts/train_sideview_reid_perception.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_baseline.py"),
        remote_path="/root/scripts/train_sideview_reid_baseline.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "evaluate_mtl_e3_held_out.py"),
        remote_path="/root/scripts/evaluate_mtl_e3_held_out.py",
    )
)

app = modal.App("mtl-e3-held-out-evaluation", image=eval_image)


@app.function(
    gpu="L40S",
    volumes={
        "/mtl-checkpoints": mtl_checkpoints_vol,
        "/mtl-data": mtl_data_vol,
        "/sideview": sideview_vol,
    },
    cpu=8.0,
    memory=32768,
    timeout=14400,
)
def evaluate_held_out_remote() -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")

    import torch
    from scripts.evaluate_mtl_e3_held_out import evaluate_run8_held_out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("/mtl-checkpoints/mtl_e3_modular/mtl_e3_best.pth")
    bcs_tensor_path = Path("/mtl-data/bcs/test_bcs_224.pt")
    behavior_dir = Path("/mtl-data/behavior_test")
    reid_root = Path("/sideview/sideviewcows2026")
    reid_protocol_csv = Path("/root/datasets/id/sideviewcows2026/protocol_cross_setting.csv")
    output_dir = Path("/mtl-checkpoints/mtl_e3_evaluation")

    results = evaluate_run8_held_out(
        checkpoint_path=checkpoint_path,
        bcs_tensor_path=bcs_tensor_path,
        behavior_dir=behavior_dir,
        reid_root=reid_root,
        reid_protocol_csv=reid_protocol_csv,
        output_dir=output_dir,
        device=device,
    )

    mtl_checkpoints_vol.commit()
    mtl_data_vol.commit()
    print("[+] Evaluation results and cache successfully committed to persistent volumes.")
    return results


@app.local_entrypoint()
def run_evaluation():
    print("=" * 75)
    print("  DISPATCHING RUN 8 E3 HELD-OUT EVALUATION TO MODAL CLOUD")
    print("=" * 75)
    results = evaluate_held_out_remote.remote()

    local_out_dir = REPO_ROOT / "artifacts" / "mtl_e3_evaluation"
    local_out_dir.mkdir(parents=True, exist_ok=True)
    local_json = local_out_dir / "mtl_e3_test_evaluation_metrics.json"

    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Evaluation metrics synced locally to {local_json}")
