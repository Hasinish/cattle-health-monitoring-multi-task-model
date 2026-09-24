# -*- coding: utf-8 -*-
"""Modal Cloud Wrapper for SideViewCows2026 Re-ID + Viewpoint Ablation.

Target Profile   : dryousufmozumder
Dataset Volume   : sideview-data (mounted at /data)
Dataset Root     : /data/sideviewcows2026 (80,260 RGB, 80,260 masks, 110 biological cows)
Checkpoint Volume: reid-checkpoints (mounted at /checkpoints)
Transferred Model: /checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth
Output Directory : /checkpoints/sideview_reid_viewpoint_ablation
Smoke Directory  : /checkpoints/sideview_reid_viewpoint_smoke
Sanity Directory : /checkpoints/sideview_viewpoint_sanity
Default Smoke GPU: NVIDIA T4 (Low-cost verification)
Default Full GPU : NVIDIA L40S (48GB Ada Lovelace tier; configurable)

Scientific Ablation:
  Controlled test of whether frozen certified real-cattle viewpoint priors
  complement the Run 6 GT-mask visual representation for cattle Re-ID.
  - Run 6: 4-channel ResNet-18 [R, G, B, Mask] (512-D) -> Linear(512, 41)
  - New: exact same Run 6 visual trunk (512-D) + Viewpoint MLP (16-D) -> Fused(528-D) -> Linear(528, 41)

Strict Isolation:
  - 41 training cows (Protocol D: 12,753 train, 2,683 val).
  - 69 held-out evaluation cows strictly untouched during readiness, sanity audit, and smoke testing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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

data_vol = modal.Volume.from_name("sideview-data")
checkpoint_vol = modal.Volume.from_name("reid-checkpoints", create_if_missing=True)

viewpoint_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
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
        str(REPO_ROOT / "scripts" / "audit_sideview_viewpoint_transfer.py"),
        remote_path="/root/scripts/audit_sideview_viewpoint_transfer.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_viewpoint.py"),
        remote_path="/root/scripts/train_sideview_reid_viewpoint.py",
    )
)

app = modal.App("sideview-reid-viewpoint-ablation", image=viewpoint_image)

EXPECTED_VIEWPOINT_HASH = "a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a"
EXPECTED_VIEWPOINT_SIZE = 134275929


# ==============================================================================
# 1. READINESS & PHYSICAL AUDIT (STEP B)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=600,
    cpu=2.0,
    memory=8192,
)
def verify_readiness_remote() -> Dict[str, Any]:
    import torch
    import torchvision.models as models
    import numpy as np
    import pandas as pd

    print("\n" + "=" * 78)
    print("  STEP B: PHYSICAL AUDIT & READINESS ON dryousufmozumder")
    print("=" * 78)

    report: Dict[str, Any] = {
        "all_passed": True,
        "profile": "dryousufmozumder",
        "checks": {},
    }

    # 1. GPU Check
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    print(f"[*] CUDA Available: {cuda_avail} | GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    report["checks"]["cuda"] = {"passed": cuda_avail, "gpu": gpu_name, "vram_gb": round(vram_gb, 2)}

    # 2. Dataset Physical Verification (sideview-data)
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

    all_cow_dirs = set()
    for subset in ("parlor", "barn", "snapshots"):
        sub_d = dataset_dir / subset / "images"
        if sub_d.exists():
            all_cow_dirs.update(d.name for d in sub_d.iterdir() if d.is_dir())
    total_cows = len(all_cow_dirs)

    print(f"[*] Volume Dataset Root: {dataset_dir} (Exists: {dataset_exists})")
    print(f"[*] RGB Images: {total_imgs}/80,260 (Parlor: {len(parlor_imgs)}, Barn: {len(barn_imgs)}, Snapshots: {len(snap_imgs)})")
    print(f"[*] GT Masks  : {total_masks}/80,260 (Parlor: {len(parlor_masks)}, Barn: {len(barn_masks)}, Snapshots: {len(snap_masks)})")
    print(f"[*] Unique Cow Identities: {total_cows}/110")

    vol_passed = (
        dataset_exists
        and total_imgs == 80260
        and total_masks == 80260
        and total_cows == 110
        and len(parlor_imgs) == len(parlor_masks) == 54393
        and len(barn_imgs) == len(barn_masks) == 25260
        and len(snap_imgs) == len(snap_masks) == 607
    )
    report["checks"]["dataset"] = {
        "passed": vol_passed,
        "total_images": total_imgs,
        "total_masks": total_masks,
        "unique_cow_identities": total_cows,
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
    print(f"[*] Protocols Integrity: {'PASS ✅' if proto_ok else 'FAIL ❌'} (41 Train Cows, 69 Eval Cows, 0 Overlap)")
    print(f"[*] Protocol D Split: Train={train_d}/12,753 | Val={val_d}/2,683")
    report["checks"]["protocols"] = {
        "passed": proto_ok,
        "train_cows": len(train_cows),
        "held_out_cows": len(eval_cows),
        "overlap": len(overlap),
        "train_d_samples": train_d,
        "val_d_samples": val_d,
    }

    # 4. Checkpoint Volume Writable Check
    probe_dir = Path("/checkpoints/sideview_reid_viewpoint_smoke")
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_file = probe_dir / "_probe.txt"
    probe_file.write_text("ok")
    probe_ok = probe_file.exists() and probe_file.read_text() == "ok"
    probe_file.unlink(missing_ok=True)
    checkpoint_vol.commit()
    print(f"[*] Checkpoint Volume Writable: {'PASS ✅' if probe_ok else 'FAIL ❌'}")
    report["checks"]["checkpoint_volume_writable"] = {"passed": probe_ok}

    # 5. Transferred Viewpoint Checkpoint Integrity & PyTorch Load
    vp_path = Path("/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth")
    vp_exists = vp_path.exists()
    vp_size = vp_path.stat().st_size if vp_exists else 0
    with open(vp_path, "rb") as f:
        vp_hash = hashlib.sha256(f.read()).hexdigest() if vp_exists else ""

    hash_match = vp_hash == EXPECTED_VIEWPOINT_HASH
    size_match = vp_size == EXPECTED_VIEWPOINT_SIZE

    print(f"[*] Viewpoint Checkpoint Path: {vp_path} (Exists: {vp_exists})")
    print(f"[*] Size: {vp_size:,} bytes (Matches expected: {size_match})")
    print(f"[*] SHA-256: {vp_hash}")
    print(f"[*] SHA-256 Matches Certified Hash: {'PASS ✅' if hash_match else 'FAIL ❌'}")

    # Load into PyTorch
    ckpt_load_ok = False
    try:
        ckpt = torch.load(vp_path, map_location="cpu", weights_only=False)
        vp_model = models.resnet18()
        vp_model.fc = torch.nn.Linear(512, 3)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = vp_model.load_state_dict(state_dict, strict=True)
        ckpt_load_ok = (len(missing) == 0 and len(unexpected) == 0)
        print(f"[*] Viewpoint Model PyTorch Load: {'PASS ✅' if ckpt_load_ok else 'FAIL ❌'}")
    except Exception as e:
        print(f"[!] PyTorch load failed: {e}")

    report["checks"]["viewpoint_checkpoint"] = {
        "passed": vp_exists and size_match and hash_match and ckpt_load_ok,
        "path": str(vp_path),
        "size_bytes": vp_size,
        "sha256": vp_hash,
        "hash_verified": hash_match,
        "pytorch_load_verified": ckpt_load_ok,
    }

    # 6. Re-ID + Viewpoint Ablation Architecture & Forward/Shape Check
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_viewpoint import ResNet18ReIDViewpointAblation, get_parameter_counts

    dev_str = "cuda" if cuda_avail else "cpu"
    ablation_model = ResNet18ReIDViewpointAblation(
        num_classes=41,
        viewpoint_checkpoint_path=str(vp_path),
        pretrained=False,
    ).to(dev_str)

    dummy_input = torch.randn(2, 4, 224, 224, device=dev_str)
    logits, embs = ablation_model(dummy_input)

    shapes_ok = (logits.shape == (2, 41) and embs.shape == (2, 528))
    norms = torch.norm(embs, p=2, dim=1).cpu().detach().numpy()
    norm_ok = np.allclose(norms, 1.0, atol=1e-5)

    # Check zero gradients assertion
    loss = logits.sum()
    loss.backward()
    ablation_model.assert_frozen_viewpoint()
    grad_assert_ok = True
    print(f"[*] Ablation Shapes: Logits={logits.shape}, Embs={embs.shape} | L2 Norm: {norms} | Pass: {shapes_ok and norm_ok}")
    print(f"[*] Frozen Viewpoint Gradients Check: PASS ✅ (zero gradients verified)")

    params = get_parameter_counts()
    report["checks"]["model_shapes_and_gradients"] = {
        "passed": shapes_ok and norm_ok and grad_assert_ok,
        "logits_shape": list(logits.shape),
        "embedding_shape": list(embs.shape),
        "unit_l2_norm_verified": bool(norm_ok),
        "frozen_viewpoint_zero_grad_verified": grad_assert_ok,
        "parameter_counts": params,
    }

    all_passed = all(c["passed"] for c in report["checks"].values())
    report["all_passed"] = all_passed
    print(f"\n[*] Overall Step B Audit Result: {'ALL CHECKS PASSED ✅' if all_passed else 'FAILED ❌'}\n")
    return report


# ==============================================================================
# 2. VIEWPOINT TRANSFER SANITY AUDIT (STEP C)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=600,
    cpu=4.0,
    memory=8192,
)
def audit_viewpoint_sanity_remote(sample_size: int = 50, seed: int = 2026) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.audit_sideview_viewpoint_transfer import audit_viewpoint_transfer

    out_dir = Path("/checkpoints/sideview_viewpoint_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = audit_viewpoint_transfer(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        checkpoint_path=Path("/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth"),
        output_dir=out_dir,
        sample_size=sample_size,
        seed=seed,
        device_str="cuda",
    )

    checkpoint_vol.commit()

    # Read artifact files to return locally
    csv_bytes = (out_dir / "viewpoint_transfer_samples.csv").read_bytes()
    json_bytes = (out_dir / "viewpoint_transfer_sanity_metrics.json").read_bytes()
    sheet_bytes = (out_dir / "viewpoint_transfer_contact_sheet.jpg").read_bytes()

    return {
        "summary": summary,
        "csv_bytes": csv_bytes,
        "json_bytes": json_bytes,
        "sheet_bytes": sheet_bytes,
    }


# ==============================================================================
# 3. SMOKE TEST (STEP F)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=1200,
    cpu=4.0,
    memory=16384,
)
def smoke_test_remote(sample_size: int = 64, epochs: int = 2, batch_size: int = 64, seed: int = 2026) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_viewpoint import train_sideview_reid_viewpoint

    smoke_dir = Path("/checkpoints/sideview_reid_viewpoint_smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_sideview_reid_viewpoint(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        viewpoint_checkpoint_path=Path("/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth"),
        output_dir=smoke_dir,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        smoke=True,
        smoke_samples=sample_size,
        device_str="cuda",
        num_workers=2,
    )

    checkpoint_vol.commit()

    # Read metrics bytes to return
    metrics_file = smoke_dir / "reid_viewpoint_smoke_metrics.json"
    metrics_bytes = metrics_file.read_bytes() if metrics_file.exists() else b"{}"

    return {
        "metrics": metrics,
        "metrics_bytes": metrics_bytes,
    }


# ==============================================================================
# 4. FULL TRAINING (USER MANUAL LAUNCH ONLY)
# ==============================================================================
@app.function(
    gpu="L40S",  # or T4 / configurable
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=14400,
    cpu=8.0,
    memory=32768,
)
def train_full_remote(epochs: int = 30, batch_size: int = 64, lr: float = 1e-4, seed: int = 2026) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_viewpoint import train_sideview_reid_viewpoint

    out_dir = Path("/checkpoints/sideview_reid_viewpoint_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_sideview_reid_viewpoint(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        viewpoint_checkpoint_path=Path("/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth"),
        output_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        smoke=False,
        device_str="cuda",
        num_workers=8,
    )
    checkpoint_vol.commit()
    return metrics


# ==============================================================================
# 5. DEDICATED PROTOCOL A EVALUATION (LOADS SAVED CHECKPOINT)
# ==============================================================================
@app.function(
    gpu="L40S",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=14400,
    cpu=8.0,
    memory=32768,
)
def evaluate_protocol_a_remote(batch_size: int = 128, num_workers: int = 8) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_viewpoint import evaluate_protocol_a_from_checkpoint

    out_dir = Path("/checkpoints/sideview_reid_viewpoint_ablation")
    metrics = evaluate_protocol_a_from_checkpoint(
        data_root=Path("/data/sideviewcows2026"),
        protocols_dir=Path("/root/datasets/id/sideviewcows2026"),
        viewpoint_checkpoint_path=Path("/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth"),
        checkpoint_dir=out_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device_str="cuda",
    )
    checkpoint_vol.commit()

    metrics_file = out_dir / "reid_viewpoint_metrics.json"
    metrics_bytes = metrics_file.read_bytes() if metrics_file.exists() else b"{}"

    return {
        "metrics": metrics,
        "metrics_bytes": metrics_bytes,
    }


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def verify_readiness():
    print("[LOCAL] Executing Step B physical readiness audit on dryousufmozumder...")
    report = verify_readiness_remote.remote()
    print("\n" + "=" * 78)
    print("STEP B PHYSICAL AUDIT RECEIPT:")
    print(json.dumps(report, indent=2))
    print("=" * 78)
    assert report["all_passed"], "Readiness checks failed!"


@app.local_entrypoint()
def audit_sanity(sample_size: int = 50, seed: int = 2026):
    print(f"[LOCAL] Executing Step C viewpoint sanity audit on dryousufmozumder (sample_size={sample_size}, seed={seed})...")
    res = audit_viewpoint_sanity_remote.remote(sample_size=sample_size, seed=seed)
    summary = res["summary"]

    # Save artifacts locally
    local_dir = REPO_ROOT / "artifacts" / "reid_viewpoint_ablation"
    local_dir.mkdir(parents=True, exist_ok=True)

    (local_dir / "viewpoint_transfer_samples.csv").write_bytes(res["csv_bytes"])
    (local_dir / "viewpoint_transfer_sanity_metrics.json").write_bytes(res["json_bytes"])
    (local_dir / "viewpoint_transfer_contact_sheet.jpg").write_bytes(res["sheet_bytes"])

    print(f"\n[OK] Downloaded Step C artifacts to: {local_dir}")
    print("\n" + "=" * 78)
    print("STEP C SANITY AUDIT SUMMARY:")
    print(f"  Distribution: Front: {summary['front_pct']}% | Side: {summary['side_pct']}% | Rear: {summary['rear_pct']}%")
    print(f"  Mean Confidence: {summary['mean_confidence']:.4f} (median: {summary['confidence_statistics']['median']:.4f})")
    print(f"  Mean Entropy: {summary['entropy_statistics']['mean']:.4f}")
    print(f"  Low Confidence (<50%): {summary['low_confidence_pct_50']}% ({summary['low_confidence_count_50']}/{summary['sample_size']})")
    print("=" * 78)


@app.local_entrypoint()
def smoke_test(sample_size: int = 64, epochs: int = 2, batch_size: int = 64, seed: int = 2026):
    print(f"[LOCAL] Executing Step F smoke certification on dryousufmozumder...")
    res = smoke_test_remote.remote(sample_size=sample_size, epochs=epochs, batch_size=batch_size, seed=seed)
    metrics = res["metrics"]

    local_dir = REPO_ROOT / "artifacts" / "reid_viewpoint_ablation"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "reid_viewpoint_smoke_metrics.json").write_bytes(res["metrics_bytes"])

    print(f"\n[OK] Smoke test completed and metrics saved to: {local_dir / 'reid_viewpoint_smoke_metrics.json'}")
    print("\n" + "=" * 78)
    print("STEP F SMOKE CERTIFICATION RECEIPT:")
    print(f"  Train Loss Drops: {metrics['history'][0]['train_loss']} -> {metrics['history'][-1]['train_loss']}")
    print(f"  Bit-Identical Reload: {metrics['checkpoint_reload']['bit_identical']} (max logit diff: {metrics['checkpoint_reload']['max_logit_difference']})")
    print(f"  Held-Out Protocol A Evaluated: {metrics['retrieval_results']['protocol_a_held_out_evaluated']} (0 images)")
    print(f"  Added Parameters vs Run 6: +{metrics['parameter_counts']['added_trainable_parameters_vs_run6']:,} ({metrics['parameter_counts']['added_trainable_percentage']}%)")
    print("=" * 78)


@app.local_entrypoint()
def evaluate_protocol_a(batch_size: int = 128, num_workers: int = 8):
    print(f"[LOCAL] Launching Protocol A retrieval evaluation on dryousufmozumder (batch_size={batch_size}, workers={num_workers})...")
    res = evaluate_protocol_a_remote.remote(batch_size=batch_size, num_workers=num_workers)
    metrics = res["metrics"]

    local_dir = REPO_ROOT / "artifacts" / "reid_viewpoint_ablation"
    local_dir.mkdir(parents=True, exist_ok=True)
    if "metrics_bytes" in res and res["metrics_bytes"]:
        (local_dir / "reid_viewpoint_metrics.json").write_bytes(res["metrics_bytes"])
    else:
        with open(local_dir / "reid_viewpoint_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("\n" + "=" * 78)
    print("PROTOCOL A RETRIEVAL RESULTS (VIEWPOINT ABLATION):")
    barn = metrics["retrieval_results"]["query_barn"]
    snap = metrics["retrieval_results"]["query_snapshots"]
    print(f"  Barn -> Parlor:      Rank-1: {barn['rank_1']:.2f}% | Rank-5: {barn['rank_5']:.2f}% | Rank-10: {barn['rank_10']:.2f}% | mAP: {barn['mAP']:.2f}%")
    print(f"  Snapshots -> Parlor: Rank-1: {snap['rank_1']:.2f}% | Rank-5: {snap['rank_5']:.2f}% | Rank-10: {snap['rank_10']:.2f}% | mAP: {snap['mAP']:.2f}%")
    print("=" * 78)


@app.local_entrypoint()
def main(epochs: int = 30, batch_size: int = 64, lr: float = 1e-4, seed: int = 2026):
    print("[LOCAL] Launching full 30-epoch training and Protocol A evaluation...")
    metrics = train_full_remote.remote(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    print("\n[OK] Full training finished!")
    print(json.dumps(metrics, indent=2))

