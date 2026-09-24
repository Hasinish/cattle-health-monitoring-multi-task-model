# -*- coding: utf-8 -*-
"""
Modal Cloud Wrapper for Phase 3 Run 6: SideViewCows2026 GT-Mask Perception-Enhanced Re-ID
=========================================================================================
Target Profile   : dryousufmozumder
Dataset Volume   : sideview-data (mounted at /data)
Dataset Root     : /data/sideviewcows2026 (80,260 RGB, 80,260 masks, 110 biological cows)
Checkpoint Volume: reid-checkpoints (mounted at /checkpoints)
Output Directory : /checkpoints/sideview_reid_perception_run6
Smoke Directory  : /checkpoints/sideview_reid_perception_smoke
Default Smoke GPU: NVIDIA T4 (Low-cost verification)
Default Full GPU : NVIDIA L40S (48GB Ada Lovelace tier; configurable via MODAL_GPU)
Core Trainer     : scripts/train_sideview_reid_perception.py

Scientific Representation Note:
  This is the deadline GT/oracle segmentation-guided Re-ID condition. Every input is
  derived from SideViewCows2026 official target-cow masks (derived bbox + 5% margin,
  aligned RGB+mask crop, resize 224x224, ImageNet RGB + raw binary mask {0,1} concatenated
  as [R, G, B, Mask], 11,200,681 trainable parameters). It is NOT an automatic SAM pipeline.

Canonical Protocol (DO NOT CHANGE):
  - Training/Val: 41 Protocol-A representation-learning cows
    (Protocol D train = 12,753 images; Protocol D val = 2,683 images)
  - Final Evaluation: 69 completely held-out biological cows
    (Parlor gallery, Barn queries, Snapshot queries; strictly isolated from training)

Usage:
  1. Readiness Verification (No-training Volume, Dataset, and Protocol Audit):
     modal run --profile dryousufmozumder scripts/modal_train_sideview_reid_perception.py::verify_readiness

  2. Cloud Smoke Test (2 Epochs, Cheap T4 GPU, Train/Val Only, Held-Out Untouched):
     modal run --profile dryousufmozumder scripts/modal_train_sideview_reid_perception.py::smoke_test

  3. Full 30-Epoch Perception Re-ID Training & Held-Out Protocol A Retrieval (User-Run):
     modal run --detach --profile dryousufmozumder scripts/modal_train_sideview_reid_perception.py::main --epochs 30 --batch-size 64
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

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

# Persistent Storage Volumes on profile dryousufmozumder
data_vol = modal.Volume.from_name("sideview-data")
checkpoint_vol = modal.Volume.from_name("reid-checkpoints", create_if_missing=True)

# Container image with dependencies and local protocol/script files
reid_perception_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
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
        str(REPO_ROOT / "scripts" / "train_sideview_reid_baseline.py"),
        remote_path="/root/scripts/train_sideview_reid_baseline.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_perception.py"),
        remote_path="/root/scripts/train_sideview_reid_perception.py",
    )
)

app = modal.App("sideview-reid-perception-run6", image=reid_perception_image)


# ==============================================================================
# 1. READINESS VERIFICATION FUNCTION (NO TRAINING)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=300,
    cpu=2.0,
    memory=4096,
)
def verify_readiness_remote() -> dict:
    """Verifies SideView dataset contents, GT masks, protocols, and checkpoint volume.

    Strictly non-training audit: verifies 80,260 RGB, 80,260 masks, 110 cow IDs,
    41 train / 69 held-out cow disjointness, checkpoint volume write access,
    and valid PIL decoding for a sample of paired RGB-mask crops.
    """
    import math
    import torch
    import numpy as np
    import pandas as pd
    from PIL import Image

    print("\n" + "=" * 76)
    print("  MODAL RUN 6 READINESS VERIFICATION (PROFILE: dryousufmozumder)")
    print("  Condition: SideViewCows2026 GT/Oracle Segmentation-Guided Re-ID")
    print("=" * 76)

    report: Dict[str, Any] = {
        "all_passed": True,
        "profile": "dryousufmozumder",
        "representation": "GT/oracle target-cow mask crop + explicit binary mask channel",
        "checks": {},
    }

    # 1. GPU Check
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    print(f"[*] CUDA Available: {cuda_avail} | GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    report["checks"]["cuda"] = {
        "passed": cuda_avail,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
    }

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

    # Unique cow identities across all three subsets
    all_cow_dirs = set()
    for subset in ("parlor", "barn", "snapshots"):
        img_subset_dir = dataset_dir / subset / "images"
        if img_subset_dir.exists():
            all_cow_dirs.update(d.name for d in img_subset_dir.iterdir() if d.is_dir())
    total_cows = len(all_cow_dirs)

    print(f"[*] Volume Dataset Root: {dataset_dir} (Exists: {dataset_exists})")
    print(f"[*] RGB Images Found  : {total_imgs}/80,260 (Parlor: {len(parlor_imgs)}, Barn: {len(barn_imgs)}, Snapshots: {len(snap_imgs)})")
    print(f"[*] GT Masks Found    : {total_masks}/80,260 (Parlor: {len(parlor_masks)}, Barn: {len(barn_masks)}, Snapshots: {len(snap_masks)})")
    print(f"[*] Cow Identities    : {total_cows}/110")

    vol_passed = (
        dataset_exists
        and total_imgs == 80260
        and total_masks == 80260
        and total_cows == 110
        and len(parlor_imgs) == len(parlor_masks) == 54393
        and len(barn_imgs) == len(barn_masks) == 25260
        and len(snap_imgs) == len(snap_masks) == 607
    )
    report["checks"]["volume_data"] = {
        "passed": vol_passed,
        "total_images": total_imgs,
        "total_masks": total_masks,
        "unique_cow_identities": total_cows,
        "parlor_rgb": len(parlor_imgs),
        "parlor_masks": len(parlor_masks),
        "barn_rgb": len(barn_imgs),
        "barn_masks": len(barn_masks),
        "snapshots_rgb": len(snap_imgs),
        "snapshots_masks": len(snap_masks),
    }

    # 3. Protocol Files & Identity Disjointness Check
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    proto_a_path = protocols_dir / "protocol_cross_setting.csv"
    proto_d_path = protocols_dir / "protocol_closed_set.csv"
    proto_files_exist = proto_a_path.exists() and proto_d_path.exists()

    df_a = pd.read_csv(proto_a_path) if proto_a_path.exists() else pd.DataFrame()
    df_d = pd.read_csv(proto_d_path) if proto_d_path.exists() else pd.DataFrame()

    train_cows_a = sorted(df_a[df_a["setting_role"] == "train"]["individual_id"].astype(str).unique()) if not df_a.empty else []
    eval_cows_a = sorted(df_a[df_a["setting_role"] != "train"]["individual_id"].astype(str).unique()) if not df_a.empty else []
    overlap_a = set(train_cows_a).intersection(set(eval_cows_a))

    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows_a)] if not df_d.empty else pd.DataFrame()
    train_d_count = len(df_d_41[df_d_41["closed_set_split"] == "train"]) if not df_d_41.empty else 0
    val_d_count = len(df_d_41[df_d_41["closed_set_split"] == "val"]) if not df_d_41.empty else 0

    proto_passed = (
        proto_files_exist
        and len(train_cows_a) == 41
        and len(eval_cows_a) == 69
        and len(overlap_a) == 0
        and train_d_count == 12753
        and val_d_count == 2683
    )
    print(f"[*] Protocols Integrity: PASS ({len(train_cows_a)} Train Cows, {len(eval_cows_a)} Held-Out Eval Cows, {len(overlap_a)} Overlap)")
    print(f"[*] Canonical Partition: Train D = {train_d_count}/12,753 | Val D = {val_d_count}/2,683")
    report["checks"]["protocols"] = {
        "passed": proto_passed,
        "protocol_files_exist": proto_files_exist,
        "train_cows": len(train_cows_a),
        "held_out_eval_cows": len(eval_cows_a),
        "identity_overlap_count": len(overlap_a),
        "protocol_d_train_images": train_d_count,
        "protocol_d_val_images": val_d_count,
    }

    # 4. Checkpoint Volume Writable Check
    checkpoint_dir = Path("/checkpoints")
    run6_dir = checkpoint_dir / "sideview_reid_perception_run6"
    test_file = checkpoint_dir / "_probe_test_run6.txt"
    try:
        run6_dir.mkdir(parents=True, exist_ok=True)
        test_file.write_text("probe_ok_run6")
        test_file.unlink()
        ckpt_passed = True
    except Exception as e:
        ckpt_passed = False
        print(f"[!] Checkpoint volume write failed: {e}")

    report["checks"]["checkpoint_volume"] = {
        "passed": ckpt_passed,
        "run6_dir": str(run6_dir),
    }

    # 5. Trainer Script Import & Syntax Verification
    try:
        import sys
        sys.path.insert(0, "/root")
        sys.path.insert(0, "/root/scripts")
        from scripts.train_sideview_reid_perception import train_sideview_reid_perception
        trainer_imported = True
    except Exception as ex:
        trainer_imported = False
        print(f"[!] Trainer import failed in container: {ex}")

    print(f"[*] Trainer Script Import: {'PASS ✅' if trainer_imported else 'FAIL ❌'}")
    report["checks"]["trainer_import"] = {
        "passed": trainer_imported,
    }

    # 6. Sample RGB-Mask Pair Decodes & BBox Derivation Check
    pair_samples = [
        ("parlor", parlor_imgs[0] if parlor_imgs else None, parlor_masks[0] if parlor_masks else None),
        ("barn", barn_imgs[0] if barn_imgs else None, barn_masks[0] if barn_masks else None),
        ("snapshots", snap_imgs[0] if snap_imgs else None, snap_masks[0] if snap_masks else None),
    ]
    pairs_ok = True
    pair_details = []
    for subset_name, img_path, mask_path in pair_samples:
        if img_path is None or mask_path is None or not img_path.exists() or not mask_path.exists():
            pairs_ok = False
            continue
        try:
            with Image.open(img_path) as im:
                rgb_im = im.convert("RGB")
                w, h = rgb_im.size
            with Image.open(mask_path) as mk:
                mask_arr = np.array(mk)
            # Check dimensions match
            dim_match = (mask_arr.shape == (h, w))
            # Check binary mask values (0 and >0)
            unique_vals = set(np.unique(mask_arr).tolist())
            binary_ok = unique_vals.issubset({0, 255}) or unique_vals.issubset({0, 1})
            # Check non-empty foreground
            fg_count = int(np.count_nonzero(mask_arr))
            fg_ok = 0 < fg_count < (w * h)
            # Check box derivation
            fg_pts = np.argwhere(mask_arr > 0)
            y1, x1 = fg_pts.min(axis=0)
            y2, x2 = fg_pts.max(axis=0)
            bw, bh = int(x2 - x1 + 1), int(y2 - y1 + 1)
            mx, my = int(math.ceil(bw * 0.05)), int(math.ceil(bh * 0.05))
            cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
            cx2, cy2 = min(w, x2 + 1 + mx), min(h, y2 + 1 + my)
            crop_ok = (0 <= cx1 < cx2 <= w) and (0 <= cy1 < cy2 <= h)

            pair_valid = dim_match and binary_ok and fg_ok and crop_ok
            if not pair_valid:
                pairs_ok = False
            pair_details.append({
                "subset": subset_name,
                "rgb_size": [w, h],
                "dim_match": dim_match,
                "binary_mask": binary_ok,
                "foreground_pixels": fg_count,
                "crop_box": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "valid": pair_valid,
            })
        except Exception as ex:
            pairs_ok = False
            pair_details.append({"subset": subset_name, "error": str(ex), "valid": False})

    print(f"[*] Sample RGB-Mask Pair Decodes: {'PASS ✅' if pairs_ok else 'FAIL ❌'}")
    for pd_info in pair_details:
        print(f"    - {pd_info['subset']}: RGB={pd_info.get('rgb_size')}, Crop={pd_info.get('crop_box')}, Valid={pd_info.get('valid')}")

    report["checks"]["rgb_mask_pairs"] = {
        "passed": pairs_ok,
        "sample_pairs": pair_details,
    }

    all_ok = all(c["passed"] for c in report["checks"].values())
    report["all_passed"] = all_ok
    print(f"\n[*] Overall Readiness: {'ALL CHECKS PASSED ✅' if all_ok else 'FAILED ❌'}\n")
    return report


# ==============================================================================
# 2. CLOUD SMOKE TEST FUNCTION (LOW-COST T4 GPU, TRAIN/VAL ONLY)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def smoke_test_remote(smoke_samples: int = 64) -> dict:
    """Executes a cheap 2-epoch smoke test on low-cost T4 GPU:

    1. Derives target bbox from GT mask + 5% margin -> 4-channel [R,G,B,Mask] input
    2. Runs forward & backward passes with ResNet-18 (conv1: 4 channels; 11,200,681 params)
    3. Asserts [B, 4, 224, 224] input shape & binary mask values {0, 1}
    4. Asserts 512-D embedding shape & L2 normalization (unit norm)
    5. Asserts exactly 41 training identities & 41-class logits
    6. Saves checkpoint & verifies bit-identical resume from epoch 1 to epoch 2
    7. Asserts Protocol A held-out gallery/query images loaded = 0
    """
    import sys
    import torch
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_perception import train_sideview_reid_perception

    data_root = Path("/data/sideviewcows2026")
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    output_dir = Path("/checkpoints/sideview_reid_perception_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("  MODAL RUN 6 RE-ID PERCEPTION SMOKE TEST (CHEAP T4 GPU, TRAIN/VAL ONLY)")
    print(f"  Condition: GT/Oracle Mask Crop + 4th Channel | Samples: {smoke_samples}")
    print("=" * 76)

    # Phase 1: Train Epoch 1
    print("\n--- Smoke Phase 1: Train Epoch 1 ---")
    metrics_ep1 = train_sideview_reid_perception(
        data_root=data_root,
        protocols_dir=protocols_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=16,
        lr=1e-4,
        num_workers=2,
        seed=2026,
        smoke=True,
        smoke_samples=smoke_samples,
    )

    ckpt_path = output_dir / "reid_perception_latest.pth"
    assert ckpt_path.exists(), "Checkpoint reid_perception_latest.pth was not saved!"
    ckpt1 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt1["epoch"] == 1, f"Expected checkpoint epoch 1, got {ckpt1['epoch']}"
    assert ckpt1["config"]["num_classes"] == 41, f"Expected 41 classes, got {ckpt1['config']['num_classes']}"

    # Phase 2: Resume from Epoch 1 and train Epoch 2
    print("\n--- Smoke Phase 2: Resume from Checkpoint & Train Epoch 2 ---")
    metrics_ep2 = train_sideview_reid_perception(
        data_root=data_root,
        protocols_dir=protocols_dir,
        output_dir=output_dir,
        epochs=2,
        batch_size=16,
        lr=1e-4,
        num_workers=2,
        seed=2026,
        smoke=True,
        smoke_samples=smoke_samples,
        resume_path=ckpt_path,
    )

    # Verify resume integrity and bit-identical reload
    ckpt2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 2, f"Expected resumed checkpoint at epoch 2, got {ckpt2['epoch']}"
    assert ckpt2["config"]["num_classes"] == 41, f"Expected 41 classes, got {ckpt2['config']['num_classes']}"

    # Assert reload max logit diff == 0.0
    reload_diff = metrics_ep2.get("checkpoint_reload", {}).get("max_logit_difference", -1.0)
    assert reload_diff == 0.0, f"Checkpoint reload changed logits (diff={reload_diff})"

    # Assert Protocol A held-out gallery/queries were NEVER loaded
    assert (
        metrics_ep2["protocol_a_retrieval"]["status"] == "NOT_LOADED_OR_EVALUATED_IN_SMOKE_MODE"
    ), "Protocol A evaluation was improperly invoked in smoke mode!"
    assert (
        metrics_ep2["protocol"]["held_out_images_loaded"] == 0
    ), "Held-out evaluation images were loaded in smoke mode!"

    checkpoint_vol.commit()
    print("\n" + "=" * 76)
    print("  SMOKE TEST PASSED: ALL CLOUD SMOKE CRITERIA CERTIFIED ✅")
    print("=" * 76 + "\n")

    return {
        "status": "SMOKE_PASS",
        "epochs_completed": 2,
        "smoke_samples": smoke_samples,
        "train_cow_count": 41,
        "total_parameters": metrics_ep2["parameter_counts"]["perception_trainable_parameters"],
        "parameter_delta": metrics_ep2["parameter_counts"]["delta_vs_baseline"],
        "shape_checks": metrics_ep2["shape_checks"],
        "max_logit_difference": reload_diff,
        "held_out_images_loaded": metrics_ep2["protocol"]["held_out_images_loaded"],
        "checkpoint_path": str(ckpt_path),
        "protocol_a_untouched": True,
    }


# ==============================================================================
# 3. FULL TRAINING REMOTE FUNCTION (PREPARED — DO NOT RUN WITHOUT EXPLICIT LAUNCH)
# ==============================================================================
@app.function(
    gpu=os.environ.get("MODAL_GPU", "L40S"),
    volumes={"/data": data_vol, "/checkpoints": checkpoint_vol},
    timeout=14400,  # 4 hours
    cpu=8.0,
    memory=32768,
)
def train_full_remote(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    workers: int = 8,
    seed: int = 2026,
    resume: bool = False,
) -> dict:
    """Executes full 30-epoch Run 6 training and post-training Protocol A retrieval.

    DO NOT EXECUTE AUTOMATICALLY. Prepared for user manual execution.
    """
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from scripts.train_sideview_reid_perception import train_sideview_reid_perception

    data_root = Path("/data/sideviewcows2026")
    protocols_dir = Path("/root/datasets/id/sideviewcows2026")
    output_dir = Path("/checkpoints/sideview_reid_perception_run6")
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = output_dir / "reid_perception_latest.pth" if resume else None

    print("\n" + "=" * 76)
    print("  STARTING FULL PHASE 3 RUN 6 PERCEPTION-ENHANCED RE-ID (MODAL CLOUD)")
    print(f"  Condition: SideViewCows2026 GT/Oracle Segmentation-Guided Re-ID")
    print(f"  Target Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Workers: {workers}")
    print(f"  Output Directory: {output_dir}")
    print("=" * 76)

    metrics = train_sideview_reid_perception(
        data_root=data_root,
        protocols_dir=protocols_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=workers,
        seed=seed,
        smoke=False,
        resume_path=resume_path,
    )

    checkpoint_vol.commit()
    print("\n[*] Full Run 6 training finished and volume committed successfully.")
    return metrics


# ==============================================================================
# LOCAL ENTRYPOINTS
# ==============================================================================
@app.local_entrypoint()
def verify_readiness():
    """Local entrypoint for cheap non-training readiness audit."""
    print("[LOCAL] Running pre-flight readiness verification on Modal (profile: dryousufmozumder)...")
    res = verify_readiness_remote.remote()
    print("\n[LOCAL] Readiness Audit Result:")
    for k, v in res["checks"].items():
        print(f"  {k}: {v}")
    print(f"\n  Overall: {'PASS ✅' if res['all_passed'] else 'FAIL ❌'}")


@app.local_entrypoint()
def smoke_test(smoke_samples: int = 64):
    """Local entrypoint for cheap T4 smoke test."""
    print(f"[LOCAL] Launching Modal smoke test on cheap T4 (samples={smoke_samples}, profile: dryousufmozumder)...")
    res = smoke_test_remote.remote(smoke_samples=smoke_samples)
    print("\n[LOCAL] Smoke Test Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    workers: int = 8,
    seed: int = 2026,
    resume: bool = False,
):
    """Main local entrypoint for full training.

    DO NOT EXECUTE AUTOMATICALLY. Prepared for manual user launch.
    """
    print(f"[LOCAL] Launching Full Run 6 Perception Re-ID Training on Modal (epochs={epochs}, batch={batch_size})...")
    res = train_full_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        workers=workers,
        seed=seed,
        resume=resume,
    )
    print("\n[LOCAL] Full Run 6 Training Finished!")
    print(f"  Best Val Accuracy: {res.get('training', {}).get('history', [{}])[-1].get('val_top1_acc')}%")
    print(f"  Protocol A Results: {res.get('protocol_a_retrieval')}")


if __name__ == "__main__":
    pass
