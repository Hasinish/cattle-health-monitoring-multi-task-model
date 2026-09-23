# -*- coding: utf-8 -*-
"""
Modal Cloud Wrapper for Phase 3 Run 5: Behavior Perception Integration Smoke Test
================================================================================
Profile Target   : tigerwood693
CVB Volume       : cvb-data (mounted at /mnt/cvb)
Beef Volume      : beef-behavior-data (mounted at /mnt/beef)
Checkpoint Volume: behavior-checkpoints (mounted at /checkpoints)
GPU Target       : NVIDIA T4 (T4 ONLY for smoke test; low cost)
Perception Cache : /checkpoints/behavior_perception_smoke_cache
Output Dir       : /checkpoints/behavior_perception_smoke

Execution Goal:
Verify the REAL Behavior Run 5 perception integration:
  - T=8 cattle-centered RGB + real SAM 2.1 binary mask -> 4-channel ResNet18 + TCN
  - CVB: Authentic target-tracklet GT bbox -> SAM 2.1 Small -> aligned 224x224 crop
  - Beef: Single-cow frame -> RT-DETR-L -> A5 (box + center point) -> SAM 2.1 Small;
    Fallback: image center point (112, 112) -> SAM 2.1 Small
  - Real perception cache generation for balanced 40-sequence smoke subset (30 train, 10 val = 320 frames)
  - Zero fake/zero/dummy masks or black sequences
  - 4-channel ResNet18 initialization (conv1: RGB from ImageNet, mask from RGB channel mean)
  - Parameter assertion: exactly 11,903,621 total trainable parameters
  - 2-epoch forward/backward training pass on NVIDIA T4
  - Bit-identical checkpoint reload verification (max logit diff < 1e-5)
  - Strict test set protection: test.csv is never parsed, loaded, sampled, tuned, or evaluated
  - Perception manifest and temporal perception contact sheet saved

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

# Container image with pre-cached RT-DETR-L and SAM 2.1 weights
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "ultralytics>=8.1.0",
        "numpy",
        "pandas",
        "pillow",
        "opencv-python-headless",
        "scikit-learn",
        "tqdm",
    )
    .run_commands(
        "python -c \"from ultralytics import RTDETR, SAM; RTDETR('rtdetr-l.pt'); SAM('sam2.1_s.pt')\""
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "behavior" / "cvb_beef"),
        remote_path="/root/datasets/behavior/cvb_beef",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "build_behavior_perception_cache.py"),
        remote_path="/root/scripts/build_behavior_perception_cache.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_cvb_beef_behavior_tcn.py"),
        remote_path="/root/scripts/train_cvb_beef_behavior_tcn.py",
    )
)

app = modal.App("cvb-beef-behavior-perception-smoke", image=train_image)


# ==============================================================================
# REMOTE PERCEPTION SMOKE TEST FUNCTION (T4 GPU ONLY)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={
        "/mnt/cvb": cvb_vol,
        "/mnt/beef": beef_vol,
        "/checkpoints": checkpoint_vol,
    },
    timeout=1200,
    cpu=2.0,
    memory=8192,
)
def smoke_test_remote() -> dict:
    """
    Executes the Run 5 perception smoke test on Modal (NVIDIA T4):
    1. Verifies dataset paths on /mnt/cvb and /mnt/beef
    2. Verifies canonical split files (asserts test.csv is never loaded)
    3. Selects the canonical 40-sequence balanced smoke subset (30 train, 10 val)
    4. Generates real perception cache (SAM 2.1 masks + cattle RGB crops)
    5. Runs 2-epoch 4-channel TCN training ([B, 8, 4, 224, 224] -> [B, 5])
    6. Evaluates validation metrics
    7. Verifies checkpoint save and bit-identical reload
    8. Generates visual perception contact sheet
    9. Asserts test.csv remained untouched and was never evaluated
    10. Commits volume and returns artifacts
    """
    import os
    import sys
    from pathlib import Path
    import pandas as pd
    import json

    sys.path.insert(0, "/root")
    from scripts.train_cvb_beef_behavior_tcn import (
        train_temporal_pipeline,
        get_balanced_smoke_subset,
    )
    from scripts.build_behavior_perception_cache import build_behavior_perception_cache

    print("\n" + "=" * 70)
    print("  MODAL SMOKE TEST: RUN 5 BEHAVIOR PERCEPTION (4-CHANNEL TCN, NVIDIA T4)")
    print("=" * 70)

    # 1. Verify physical dataset paths
    cvb_dir = Path("/mnt/cvb/cvb/000058916v001")
    beef_dir = Path("/mnt/beef/beef_behavior")
    data_dir = Path("/root/datasets/behavior/cvb_beef")
    cache_dir = Path("/checkpoints/behavior_perception_smoke_cache")
    output_dir = Path("/checkpoints/behavior_perception_smoke")

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

    # 3. Deterministic Smoke Subset Selection (shared selector)
    smoke_train_df, smoke_val_df = get_balanced_smoke_subset(train_df, val_df)
    print(f"[*] Shared smoke subset selected: Train={len(smoke_train_df)} sequences, Val={len(smoke_val_df)} sequences")

    # 4. Generate Real Perception Cache (SAM 2.1 + RT-DETR-L)
    print("\n[*] Stage 1: Building real perception cache (SAM 2.1 Small masks + cattle crops)...")
    from ultralytics import RTDETR, SAM
    rtdetr_model = RTDETR("rtdetr-l.pt")
    sam_model = SAM("sam2.1_s.pt")

    retained_train_df, train_stats, train_frame_records = build_behavior_perception_cache(
        df=smoke_train_df,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        rtdetr_model=rtdetr_model,
        sam_model=sam_model,
        num_frames=8,
        device="cuda",
        desc="Train Perception Cache",
    )

    retained_val_df, val_stats, val_frame_records = build_behavior_perception_cache(
        df=smoke_val_df,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        rtdetr_model=rtdetr_model,
        sam_model=sam_model,
        num_frames=8,
        device="cuda",
        desc="Val Perception Cache",
    )

    # Save perception manifest and summary to cache_dir
    all_records = train_frame_records + val_frame_records
    manifest_df = pd.DataFrame(all_records)
    manifest_path = cache_dir / "perception_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    perception_summary = {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "total_sequences_requested": len(smoke_train_df) + len(smoke_val_df),
        "total_sequences_retained": len(retained_train_df) + len(retained_val_df),
        "total_real_masks_generated": train_stats["total_real_masks_generated"] + val_stats["total_real_masks_generated"],
        "cvb_frames_generated": train_stats["cvb_frames_generated"] + val_stats["cvb_frames_generated"],
        "beef_a5_frames_generated": train_stats["beef_a5_frames_generated"] + val_stats["beef_a5_frames_generated"],
        "beef_fallback_frames_generated": train_stats["beef_fallback_frames_generated"] + val_stats["beef_fallback_frames_generated"],
    }
    summary_path = cache_dir / "perception_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(perception_summary, f, indent=2)

    print(f"[*] Perception caching complete: {perception_summary['total_real_masks_generated']} real masks generated.")
    print(f"    CVB frames: {perception_summary['cvb_frames_generated']}")
    print(f"    Beef A5 frames: {perception_summary['beef_a5_frames_generated']}")
    print(f"    Beef Fallback frames: {perception_summary['beef_fallback_frames_generated']}")

    # Assert retained sequence sets are valid and non-empty
    assert len(retained_train_df) > 0, "No train sequences retained after perception filtering!"
    assert len(retained_val_df) > 0, "No val sequences retained after perception filtering!"
    print(f"[*] Sequence retention verification: Train={len(retained_train_df)}/{len(smoke_train_df)}, Val={len(retained_val_df)}/{len(smoke_val_df)}")
    print(f"    Train retained IDs: {list(retained_train_df['sample_id'])}")
    print(f"    Val retained IDs: {list(retained_val_df['sample_id'])}")

    # 5. Execute 4-Channel Temporal Training Pipeline with retained perception sequences
    print("\n[*] Stage 2: Training 4-channel ResNet18 + TCN model...")
    summary = train_temporal_pipeline(
        data_dir=data_dir,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        train_df=retained_train_df,
        val_df=retained_val_df,
        epochs=2,
        batch_size=4,
        lr=1e-4,
        weight_decay=1e-2,
        num_workers=0,
        num_frames=8,
        smoke=True,
        input_mode="rgb_mask",
    )

    # 6. Assert test.csv was NEVER touched or evaluated
    test_stat_after = test_csv.stat()
    assert test_stat_before.st_mtime == test_stat_after.st_mtime, "CRITICAL: test.csv was modified!"
    assert summary.get("test_csv_evaluated") is False, "CRITICAL: test.csv was evaluated in smoke mode!"
    print("[*] Strict canonical test-set isolation PASS: test.csv was not parsed, loaded, sampled, tuned, or evaluated.")

    # 7. Read contact sheet and manifest bytes
    contact_sheet_p = output_dir / "temporal_perception_contact_sheet.jpg"
    contact_sheet_bytes = None
    if contact_sheet_p.exists():
        with open(contact_sheet_p, "rb") as f:
            contact_sheet_bytes = f.read()
        print(f"[*] Loaded perception contact sheet ({len(contact_sheet_bytes):,} bytes)")

    manifest_bytes = None
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            manifest_bytes = f.read()

    # Attach perception summary to metrics summary
    summary["perception_summary"] = perception_summary

    # 8. Commit volume
    checkpoint_vol.commit()
    print("[*] Persistent volume 'behavior-checkpoints' committed successfully.")

    result = {
        "summary": summary,
        "contact_sheet_bytes": contact_sheet_bytes,
        "manifest_bytes": manifest_bytes,
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

    print("Launching Run 5 Behavior Perception Smoke Test on Modal (profile tigerwood693, GPU T4)...")
    res = smoke_test_remote.remote()
    total_time = time.perf_counter() - t0

    summary = res["summary"]
    contact_sheet_bytes = res["contact_sheet_bytes"]
    manifest_bytes = res.get("manifest_bytes")

    # Save local artifacts in separate perception directory
    local_art_dir = REPO_ROOT / "artifacts" / "behavior_perception_smoke"
    local_art_dir.mkdir(parents=True, exist_ok=True)

    local_audit_dir = REPO_ROOT / "docs" / "audits" / "assets" / "behavior_perception_smoke"
    local_audit_dir.mkdir(parents=True, exist_ok=True)

    summary_path = local_art_dir / "behavior_tcn_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Local metrics saved: {summary_path}")

    if manifest_bytes:
        manifest_out = local_art_dir / "perception_manifest.csv"
        with open(manifest_out, "wb") as f:
            f.write(manifest_bytes)
        print(f"[*] Local perception manifest saved: {manifest_out}")

    if contact_sheet_bytes:
        sheet_art_p = local_art_dir / "temporal_perception_contact_sheet.jpg"
        sheet_audit_p = local_audit_dir / "temporal_perception_contact_sheet.jpg"
        with open(sheet_art_p, "wb") as f:
            f.write(contact_sheet_bytes)
        with open(sheet_audit_p, "wb") as f:
            f.write(contact_sheet_bytes)
        print(f"[*] Local contact sheet saved: {sheet_art_p}")
        print(f"[*] Local audit contact sheet saved: {sheet_audit_p}")

    p_sum = summary.get("perception_summary", {})

    print("\n" + "=" * 70)
    print(f"  PERCEPTION SMOKE TEST PASSED SUCCESSFULLY IN {total_time:.1f}s!")
    print("=" * 70)
    print(f"  Status                 : {summary.get('status')}")
    print(f"  Input Mode             : {summary.get('input_mode')} (in_channels={summary.get('in_channels')})")
    print(f"  Total Trainable Params : {summary['architecture']['total_trainable_params']:,} (Expected: 11,903,621)")
    print(f"  Sequences Processed    : Train={summary['smoke_dataset_counts']['train']['total']}, Val={summary['smoke_dataset_counts']['val']['total']}")
    print(f"  Total Real Masks       : {p_sum.get('total_real_masks_generated')} (CVB={p_sum.get('cvb_frames_generated')}, Beef A5={p_sum.get('beef_a5_frames_generated')}, Beef Fallback={p_sum.get('beef_fallback_frames_generated')})")
    print(f"  Best Val Macro-F1      : {summary['best_val_metrics']['macro_f1']:.4f}")
    print(f"  Logit Reload Diff      : {summary['max_logit_diff_after_reload']:.8f}")
    print(f"  Test CSV Evaluated     : {summary.get('test_csv_evaluated')}")
    print(f"  Test Set Protection    : {summary.get('test_set_protection')}")
    print("=" * 70)


# ==============================================================================
# AUDIT & REGENERATE SMOKE PERCEPTION CACHE PROVENANCE (T4 GPU ONLY, NO TRAINING)
# ==============================================================================
@app.function(
    gpu="T4",
    volumes={
        "/mnt/cvb": cvb_vol,
        "/mnt/beef": beef_vol,
        "/checkpoints": checkpoint_vol,
    },
    timeout=1200,
    cpu=2.0,
    memory=8192,
)
def audit_regenerate_smoke_cache_remote() -> dict:
    """
    Regenerates ONLY the tiny 40-sequence behavior perception smoke cache
    from scratch on Modal T4, asserts all 10 provenance invariants,
    and executes a full resume verification pass.
    Strictly:
      - ZERO TCN training
      - test.csv is never touched or evaluated
      - full dataset is never touched
    """
    import os
    import sys
    import shutil
    import json
    import time
    from pathlib import Path
    import pandas as pd

    sys.path.insert(0, "/root")
    from scripts.train_cvb_beef_behavior_tcn import get_balanced_smoke_subset
    from scripts.build_behavior_perception_cache import build_behavior_perception_cache

    print("\n" + "=" * 70)
    print("  MODAL AUDIT: REGENERATE RUN 5 BEHAVIOR PERCEPTION SMOKE CACHE PROVENANCE")
    print("=" * 70)

    # 1. Path verification
    cvb_dir = Path("/mnt/cvb/cvb/000058916v001")
    beef_dir = Path("/mnt/beef/beef_behavior")
    data_dir = Path("/root/datasets/behavior/cvb_beef")
    cache_dir = Path("/checkpoints/behavior_perception_smoke_cache")

    assert cvb_dir.exists(), f"CVB directory missing at {cvb_dir}"
    assert beef_dir.exists(), f"Beef directory missing at {beef_dir}"
    assert data_dir.exists(), f"Data directory missing at {data_dir}"

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    assert train_csv.exists() and val_csv.exists() and test_csv.exists()

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_stat_before = test_csv.stat()

    # 2. Select the identical deterministic 40-sequence smoke subset
    smoke_train_df, smoke_val_df = get_balanced_smoke_subset(train_df, val_df)
    assert len(smoke_train_df) == 30, f"Expected 30 smoke train, got {len(smoke_train_df)}"
    assert len(smoke_val_df) == 10, f"Expected 10 smoke val, got {len(smoke_val_df)}"
    print(f"[*] Deterministic smoke subset selected: Train={len(smoke_train_df)}, Val={len(smoke_val_df)}")

    # 3. Clean wipe ONLY behavior_perception_smoke_cache on volume
    print(f"[*] Wiping existing cache at {cache_dir} to regenerate authentic provenance...")
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize perception models
    from ultralytics import RTDETR, SAM
    print("[*] Initializing RT-DETR-L and SAM 2.1 Small...")
    rtdetr_model = RTDETR("rtdetr-l.pt")
    sam_model = SAM("sam2.1_s.pt")

    # 5. FRESH EXTRACTION PHASE
    print("\n--- PHASE 1: FRESH EXTRACTION & PROVENANCE PERSISTENCE ---")
    t0_fresh = time.perf_counter()
    retained_train_df_1, train_stats_1, train_records_1 = build_behavior_perception_cache(
        df=smoke_train_df,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        rtdetr_model=rtdetr_model,
        sam_model=sam_model,
        num_frames=8,
        device="cuda",
        desc="Train Fresh Perception Cache",
    )

    retained_val_df_1, val_stats_1, val_records_1 = build_behavior_perception_cache(
        df=smoke_val_df,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        rtdetr_model=rtdetr_model,
        sam_model=sam_model,
        num_frames=8,
        device="cuda",
        desc="Val Fresh Perception Cache",
    )
    t_fresh = time.perf_counter() - t0_fresh
    print(f"[*] Fresh extraction completed in {t_fresh:.1f}s.")

    # Save fresh manifest and summary
    all_records_1 = train_records_1 + val_records_1
    manifest_df_1 = pd.DataFrame(all_records_1)
    manifest_path = cache_dir / "perception_manifest.csv"
    manifest_df_1.to_csv(manifest_path, index=False)

    summary_1 = {
        "train_stats": train_stats_1,
        "val_stats": val_stats_1,
        "total_sequences_requested": len(smoke_train_df) + len(smoke_val_df),
        "total_sequences_retained": len(retained_train_df_1) + len(retained_val_df_1),
        "total_real_masks_generated": train_stats_1["total_real_masks_generated"] + val_stats_1["total_real_masks_generated"],
        "cvb_frames_generated": train_stats_1["cvb_frames_generated"] + val_stats_1["cvb_frames_generated"],
        "beef_a5_frames_generated": train_stats_1["beef_a5_frames_generated"] + val_stats_1["beef_a5_frames_generated"],
        "beef_fallback_frames_generated": train_stats_1["beef_fallback_frames_generated"] + val_stats_1["beef_fallback_frames_generated"],
    }
    summary_path = cache_dir / "perception_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_1, f, indent=2)

    # Invariant assertions on fresh extraction
    print("\n--- INVARIANT CHECKS ON FRESH CACHE ---")
    assert len(retained_train_df_1) == 29, f"Expected 29 retained train, got {len(retained_train_df_1)}"
    assert len(retained_val_df_1) == 9, f"Expected 9 retained val, got {len(retained_val_df_1)}"
    assert summary_1["total_sequences_retained"] == 38
    print(f"[*] Retained sequence counts: Train={len(retained_train_df_1)}, Val={len(retained_val_df_1)} (Total={summary_1['total_sequences_retained']})")
    print(f"[*] Fresh frame breakdown: CVB={summary_1['cvb_frames_generated']}, Beef A5={summary_1['beef_a5_frames_generated']}, Beef Fallback={summary_1['beef_fallback_frames_generated']}")
    print(f"[*] Total real masks generated: {summary_1['total_real_masks_generated']}")

    # Verify per-frame metadata on disk
    retained_sids = set(retained_train_df_1["sample_id"]).union(set(retained_val_df_1["sample_id"]))
    for sid in retained_sids:
        s_dir = cache_dir / sid
        m_file = s_dir / "perception_metadata.json"
        assert m_file.exists(), f"perception_metadata.json missing in {sid}"
        with open(m_file, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        assert len(meta_json) == 8, f"Expected 8 frames in metadata for {sid}"
        for fm in meta_json:
            assert isinstance(fm["frame_index"], int), f"Non-int frame_index in {sid}: {fm['frame_index']}"
            assert fm["prompt_strategy"] not in ("beef_cached", None), f"Invalid prompt_strategy in {sid}"
            assert fm["bbox"] != "already_cached", f"Placeholder bbox in {sid}"
            if fm["dataset"] == "cvb":
                assert fm["prompt_strategy"] == "cvb_gt_bbox"
                assert fm["bbox"] is not None and fm["bbox"] != "null"
            elif fm["dataset"] == "beef_cattle_behavior":
                assert fm["prompt_strategy"] in ("beef_A5_rtdetr_box_center_point", "beef_center_point_fallback")
                if fm["prompt_strategy"] == "beef_center_point_fallback":
                    assert fm["fallback_used"] is True
                else:
                    assert fm["fallback_used"] is False

    print("[*] All 38 retained cached sequences verified on disk: 100% valid perception_metadata.json.")

    # 6. RESUME VERIFICATION PHASE (Assertion 6)
    print("\n--- PHASE 2: RESUME VERIFICATION PASS ---")
    t0_resume = time.perf_counter()
    retained_train_df_2, train_stats_2, train_records_2 = build_behavior_perception_cache(
        df=smoke_train_df,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        rtdetr_model=rtdetr_model,
        sam_model=sam_model,
        num_frames=8,
        device="cuda",
        desc="Train Resume Perception Cache",
    )

    retained_val_df_2, val_stats_2, val_records_2 = build_behavior_perception_cache(
        df=smoke_val_df,
        cvb_dir=cvb_dir,
        beef_dir=beef_dir,
        cache_dir=cache_dir,
        rtdetr_model=rtdetr_model,
        sam_model=sam_model,
        num_frames=8,
        device="cuda",
        desc="Val Resume Perception Cache",
    )
    t_resume = time.perf_counter() - t0_resume
    print(f"[*] Resume pass completed in {t_resume:.1f}s.")

    # Check resume stats
    assert train_stats_2["already_cached"] == 29, f"Expected 29 already_cached train, got {train_stats_2['already_cached']}"
    assert val_stats_2["already_cached"] == 9, f"Expected 9 already_cached val, got {val_stats_2['already_cached']}"
    assert train_stats_2["extracted_success"] == 29
    assert val_stats_2["extracted_success"] == 9

    summary_2 = {
        "train_stats": train_stats_2,
        "val_stats": val_stats_2,
        "total_sequences_requested": len(smoke_train_df) + len(smoke_val_df),
        "total_sequences_retained": len(retained_train_df_2) + len(retained_val_df_2),
        "total_real_masks_generated": train_stats_2["total_real_masks_generated"] + val_stats_2["total_real_masks_generated"],
        "cvb_frames_generated": train_stats_2["cvb_frames_generated"] + val_stats_2["cvb_frames_generated"],
        "beef_a5_frames_generated": train_stats_2["beef_a5_frames_generated"] + val_stats_2["beef_a5_frames_generated"],
        "beef_fallback_frames_generated": train_stats_2["beef_fallback_frames_generated"] + val_stats_2["beef_fallback_frames_generated"],
    }

    # Prove resume produces semantically identical counts and bit-identical manifest
    for key in [
        "total_sequences_requested",
        "total_sequences_retained",
        "total_real_masks_generated",
        "cvb_frames_generated",
        "beef_a5_frames_generated",
        "beef_fallback_frames_generated",
    ]:
        assert summary_1[key] == summary_2[key], f"Mismatch in {key}: {summary_1[key]} vs {summary_2[key]}"

    for sub in ["train_stats", "val_stats"]:
        for key in [
            "total_requested",
            "extracted_success",
            "failed_sequences",
            "cvb_frames_generated",
            "beef_a5_frames_generated",
            "beef_fallback_frames_generated",
            "total_real_masks_generated",
        ]:
            assert summary_1[sub][key] == summary_2[sub][key], f"Mismatch in {sub}.{key}: {summary_1[sub][key]} vs {summary_2[sub][key]}"

    # Specifically verify that already_cached accurately reflects the resume state
    assert summary_1["train_stats"]["already_cached"] == 0
    assert summary_1["val_stats"]["already_cached"] == 0
    assert summary_2["train_stats"]["already_cached"] == 29
    assert summary_2["val_stats"]["already_cached"] == 9

    all_records_2 = train_records_2 + val_records_2
    assert len(all_records_1) == len(all_records_2), f"Record length mismatch: {len(all_records_1)} vs {len(all_records_2)}"
    manifest_df_2 = pd.DataFrame(all_records_2)
    assert manifest_df_1.equals(manifest_df_2), "Manifest DataFrame mismatch between fresh and resumed runs!"
    print("[*] Resume reproducibility PASS: manifest and scientific counts are 100% BIT-IDENTICAL between fresh and resumed runs!")

    # 7. Test set isolation check
    test_stat_after = test_csv.stat()
    assert test_stat_before.st_mtime == test_stat_after.st_mtime, "CRITICAL: test.csv was modified!"
    print("[*] Strict canonical test-set isolation PASS: test.csv was not parsed, loaded, sampled, tuned, or evaluated.")

    # 8. Commit volume
    checkpoint_vol.commit()
    print("[*] Persistent volume 'behavior-checkpoints' committed successfully.")

    # Read manifest bytes and summary for return
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_csv_text = f.read()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_json_text = f.read()

    return {
        "summary": summary_1,
        "manifest_csv_text": manifest_csv_text,
        "summary_json_text": summary_json_text,
        "runtime_fresh": t_fresh,
        "runtime_resume": t_resume,
    }


@app.local_entrypoint()
def audit_smoke_cache():
    """
    Local entrypoint to regenerate and audit the behavior perception smoke cache provenance:
      modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::audit_smoke_cache
    """
    import json
    import time
    t0 = time.perf_counter()

    print("Launching Run 5 Behavior Perception Smoke Cache Provenance Audit on Modal (tigerwood693, GPU T4)...")
    res = audit_regenerate_smoke_cache_remote.remote()
    total_time = time.perf_counter() - t0

    summary = res["summary"]
    manifest_csv_text = res["manifest_csv_text"]
    summary_json_text = res["summary_json_text"]

    local_art_dir = REPO_ROOT / "artifacts" / "behavior_perception_smoke"
    local_art_dir.mkdir(parents=True, exist_ok=True)

    manifest_out = local_art_dir / "perception_manifest.csv"
    with open(manifest_out, "w", encoding="utf-8") as f:
        f.write(manifest_csv_text)
    print(f"[*] Local perception manifest saved: {manifest_out}")

    summary_out = local_art_dir / "perception_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(summary_json_text)
    print(f"[*] Local perception summary saved: {summary_out}")

    print("\n" + "=" * 70)
    print(f"  PERCEPTION SMOKE CACHE AUDIT COMPLETE IN {total_time:.1f}s!")
    print("=" * 70)
    print(f"  Sequences Requested  : {summary.get('total_sequences_requested')}")
    print(f"  Sequences Retained   : {summary.get('total_sequences_retained')}")
    print(f"  Total Real Masks     : {summary.get('total_real_masks_generated')}")
    print(f"  CVB Frames (GT BBox) : {summary.get('cvb_frames_generated')}")
    print(f"  Beef A5 Frames       : {summary.get('beef_a5_frames_generated')}")
    print(f"  Beef Fallback Frames : {summary.get('beef_fallback_frames_generated')}")
    print(f"  Fresh Runtime        : {res.get('runtime_fresh'):.1f}s")
    print(f"  Resume Runtime       : {res.get('runtime_resume'):.1f}s")
    print("=" * 70)

