# -*- coding: utf-8 -*-
"""Modal Cloud Wrapper for SideViewCows2026 Re-ID + SuperAnimal Pose Ablation.

Target Profile   : tigerwood697
Dataset Volume   : sideview-data (mounted at /data)
Checkpoint Volume: reid-checkpoints (mounted at /checkpoints)
Output Directory : /checkpoints/sideview_reid_pose_ablation
Smoke Directory  : /checkpoints/sideview_reid_pose_smoke
Smoke GPU        : NVIDIA Tesla T4 (low cost)
Full GPU         : NVIDIA L40S / Tesla T4

Usage:
    # 1. Readiness verification (no training):
    modal run --profile tigerwood697 scripts/modal_train_sideview_reid_pose.py::verify_readiness

    # 2. Smoke test (2 epochs on T4, train/val only, held-out cows untouched):
    modal run --profile tigerwood697 scripts/modal_train_sideview_reid_pose.py::smoke_test

    # 3. Full 30-epoch training (manual user launch):
    modal run --detach --profile tigerwood697 scripts/modal_train_sideview_reid_pose.py::main --epochs 30
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

# Image matches test_modal_superanimal cached layers
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
        str(REPO_ROOT / "scripts" / "train_sideview_reid_pose.py"),
        remote_path="/root/scripts/train_sideview_reid_pose.py",
    )
)

app = modal.App("sideview-reid-pose-ablation", image=pose_image)


# ==============================================================================
# 1. READINESS VERIFICATION
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=300,
    cpu=2.0,
    memory=4096,
)
def verify_readiness_remote() -> Dict[str, Any]:
    import torch
    import numpy as np
    import pandas as pd
    from PIL import Image

    print("\n" + "=" * 76)
    print("  MODAL RE-ID + POSE ABLATION READINESS VERIFICATION (PROFILE: tigerwood697)")
    print("=" * 76)

    report: Dict[str, Any] = {
        "all_passed": True,
        "profile": "tigerwood697",
        "checks": {},
    }

    # 1. GPU Check
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    print(f"[*] CUDA Available: {cuda_avail} | GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    report["checks"]["cuda"] = {"passed": cuda_avail, "gpu": gpu_name, "vram_gb": round(vram_gb, 2)}

    # 2. Dataset Root on Volume
    dataset_dir = Path("/data/sideviewcows2026")
    dataset_exists = dataset_dir.exists()
    parlor_imgs = list((dataset_dir / "parlor" / "images").glob("*/*.jpg")) if dataset_exists else []
    barn_imgs = list((dataset_dir / "barn" / "images").glob("*/*.jpg")) if dataset_exists else []
    snap_imgs = list((dataset_dir / "snapshots" / "images").glob("*/*.jpg")) if dataset_exists else []
    total_imgs = len(parlor_imgs) + len(barn_imgs) + len(snap_imgs)

    parlor_masks = list((dataset_dir / "parlor" / "masks").glob("*/*.png")) if dataset_exists else []
    barn_masks = list((dataset_dir / "barn" / "masks").glob("*/*.png")) if dataset_exists else []
    snap_masks = list((dataset_dir / "snapshots" / "masks").glob("*/*.png")) if dataset_exists else []
    total_masks = len(parlor_masks) + len(barn_masks) + len(snap_masks)

    print(f"[*] Volume Dataset Root: {dataset_dir} (Exists: {dataset_exists})")
    print(f"[*] RGB Images: {total_imgs}/80,260 | GT Masks: {total_masks}/80,260")
    report["checks"]["dataset"] = {
        "passed": dataset_exists and total_imgs == 80260 and total_masks == 80260,
        "total_images": total_imgs,
        "total_masks": total_masks,
    }

    # 3. Protocol Files & Cow Disjointness
    proto_dir = Path("/root/datasets/id/sideviewcows2026")
    df_a = pd.read_csv(proto_dir / "protocol_cross_setting.csv")
    df_d = pd.read_csv(proto_dir / "protocol_closed_set.csv")

    train_cows = sorted(df_a[df_a["setting_role"] == "train"]["individual_id"].astype(str).unique())
    eval_cows = sorted(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique())
    overlap = set(train_cows).intersection(set(eval_cows))

    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)]
    train_d = len(df_d_41[df_d_41["closed_set_split"] == "train"])
    val_d = len(df_d_41[df_d_41["closed_set_split"] == "val"])

    proto_ok = (len(train_cows) == 41 and len(eval_cows) == 69 and len(overlap) == 0 and train_d == 12753 and val_d == 2683)
    print(f"[*] Protocols Integrity: {'PASS ✅' if proto_ok else 'FAIL ❌'} (41 Train, 69 Eval, 0 Overlap)")
    report["checks"]["protocols"] = {
        "passed": proto_ok,
        "train_cows": len(train_cows),
        "held_out_cows": len(eval_cows),
        "overlap": len(overlap),
        "train_samples": train_d,
        "val_samples": val_d,
    }

    # 4. Checkpoint Volume Writable
    ckpt_dir = Path("/checkpoints/sideview_reid_pose_smoke")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    probe_f = ckpt_dir / "_probe.txt"
    probe_f.write_text("ok")
    probe_ok = probe_f.exists() and probe_f.read_text() == "ok"
    probe_f.unlink(missing_ok=True)
    checkpoint_vol.commit()
    print(f"[*] Checkpoint Volume Writable: {'PASS ✅' if probe_ok else 'FAIL ❌'}")
    report["checks"]["checkpoint_volume"] = {"passed": probe_ok}

    # 5. Trainer Script & Tensor Shape Check
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_pose import ResNet18ReIDPoseAblation, get_parameter_counts

    dev_str = "cuda" if cuda_avail else "cpu"
    model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False).to(dev_str)
    dummy_vis = torch.randn(2, 4, 224, 224, device=dev_str)
    dummy_pose = torch.randn(2, 156, device=dev_str)
    logits, embs = model(dummy_vis, dummy_pose)

    shapes_ok = (logits.shape == (2, 41) and embs.shape == (2, 576))
    norms = torch.norm(embs, p=2, dim=1).cpu().detach().numpy()
    norm_ok = np.allclose(norms, 1.0, atol=1e-5)
    params = get_parameter_counts()
    print(f"[*] Model Shapes Check: Logits={logits.shape}, Embs={embs.shape} | L2 Norm: {norms} | Pass: {shapes_ok and norm_ok}")
    print(f"[*] Exact Parameters: Total = {params['total_trainable_parameters']:,} (Visual: {params['visual_backbone_trainable_parameters']:,}, Pose: {params['pose_mlp_trainable_parameters']:,}, Cls: {params['classifier_trainable_parameters']:,})")

    report["checks"]["model_tensor_shapes"] = {
        "passed": shapes_ok and norm_ok,
        "logits_shape": list(logits.shape),
        "embedding_shape": list(embs.shape),
        "unit_l2_norm_verified": bool(norm_ok),
        "trainable_parameters": params,
    }

    all_passed = all(c["passed"] for c in report["checks"].values())
    report["all_passed"] = all_passed
    print(f"\n[*] Overall Readiness Result: {'ALL CHECKS PASSED ✅' if all_passed else 'FAILED ❌'}\n")
    return report


# ==============================================================================
# 2. CLOUD SMOKE TEST
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=600,
    cpu=4.0,
    memory=16384,
)
def smoke_test_remote() -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from scripts.train_sideview_reid_pose import train_sideview_reid_pose

    out_dir = Path("/checkpoints/sideview_reid_pose_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_sideview_reid_pose(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        output_dir=out_dir,
        epochs=2,
        batch_size=16,
        lr=1e-4,
        weight_decay=1e-4,
        smoke=True,
        smoke_samples=64,
    )

    checkpoint_vol.commit()
    return metrics


# ==============================================================================
# 3. FULL 30-EPOCH TRAINING ENTRYPOINT (DEFERRED FOR MANUAL LAUNCH)
# ==============================================================================
@app.function(
    gpu="T4",  # Configurable
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=7200,
    cpu=4.0,
    memory=16384,
)
def train_full_remote(epochs: int = 30, batch_size: int = 64) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from scripts.train_sideview_reid_pose import train_sideview_reid_pose

    out_dir = Path("/checkpoints/sideview_reid_pose_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_sideview_reid_pose(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        output_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        weight_decay=1e-4,
        smoke=False,
    )

    checkpoint_vol.commit()
    return metrics


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def verify_readiness():
    print("[LOCAL] Running readiness verification on profile tigerwood697...")
    res = verify_readiness_remote.remote()
    print("Result:", json.dumps(res, indent=2))


@app.local_entrypoint()
def smoke_test():
    print("[LOCAL] Running Re-ID + Pose cloud smoke test on profile tigerwood697 (T4 GPU)...")
    res = smoke_test_remote.remote()

    local_out = REPO_ROOT / "artifacts" / "reid_pose_ablation"
    local_out.mkdir(parents=True, exist_ok=True)
    metrics_path = local_out / "reid_pose_smoke_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[LOCAL] Smoke Test Succeeded! Saved metrics to: {metrics_path}")
    print(f"  - Reload Bit-Identical Max Logit Diff: {res['checkpoint_reload_max_logit_diff']:.8f}")
    print(f"  - Total Trainable Parameters: {res['trainable_parameters']['total_trainable_parameters']:,}")


@app.local_entrypoint()
def main(epochs: int = 30, batch_size: int = 64):
    print(f"[LOCAL] Full training requested: {epochs} epochs, batch size {batch_size}")
    res = train_full_remote.remote(epochs=epochs, batch_size=batch_size)
    print("Full training finished:", res)
