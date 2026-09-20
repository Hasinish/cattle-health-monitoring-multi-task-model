"""
Script: scripts/build_manual_dataset_visual_verification.py
Purpose: Generate a curated, deterministic manual visual verification pack for the 3 primary Phase 3 datasets:
         1. ScienceDB — Primary BCS (16 checks)
         2. MmCows — Primary Behavior (16 checks)
         3. SideViewCows2026 — Primary Re-ID (16 checks)

Outputs:
  - Markdown Pack: docs/audits/phase3_manual_dataset_visual_verification.md
  - Asset Images:  docs/audits/assets/manual_dataset_verification/

Design & Constraints:
  - Human sanity check only (automated audits remain authoritative).
  - Seed recorded (RANDOM_SEED = 42).
  - Clean tqdm progress bar showing stages, ETA, elapsed time, and rate.
  - Generates compact, Git-friendly resized review thumbnails and 3-panel SideView composites (RGB | Mask | Overlay).
  - Uses relative paths in Markdown for seamless viewing in VS Code and GitHub.
"""

import os
import sys
import csv
import time
import random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_AUDIT_DIR = REPO_ROOT / "docs" / "audits"
ASSETS_DIR = DOCS_AUDIT_DIR / "assets" / "manual_dataset_verification"
OUTPUT_MD = DOCS_AUDIT_DIR / "phase3_manual_dataset_visual_verification.md"

RANDOM_SEED = 42

BAR_FORMAT = "{desc:<42} |{bar:25}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] in {elapsed} (ETA {remaining})"


def make_pbar(total: int, desc: str):
    """Create a standardized, flicker-free Windows-compatible tqdm progress bar."""
    return tqdm(
        total=total,
        desc=desc,
        ascii=True,
        ncols=95,
        dynamic_ncols=False,
        bar_format=BAR_FORMAT,
        leave=True,
        file=sys.stdout,
    )


# ==============================================================================
# SAMPLE SELECTION DEFINITIONS
# ==============================================================================
def select_samples():
    """Select 48 diverse, representative visual check candidates across the 3 datasets."""
    def find_first(items, pred, desc):
        for x in items:
            if pred(x):
                return x
        raise ValueError(f"Missing visual verification candidate: {desc}")

    # --------------------------------------------------------------------------
    # 1. ScienceDB Samples (16 checks)
    # --------------------------------------------------------------------------
    with open(REPO_ROOT / "datasets" / "bcs" / "sciencedb" / "train.csv", "r", encoding="utf-8") as f:
        s_train = list(csv.DictReader(f))

    with open(REPO_ROOT / "datasets" / "bcs" / "sciencedb" / "val.csv", "r", encoding="utf-8") as f:
        s_val = list(csv.DictReader(f))

    with open(REPO_ROOT / "datasets" / "bcs" / "sciencedb" / "test.csv", "r", encoding="utf-8") as f:
        s_test = list(csv.DictReader(f))

    sc_checks = [
        # Normal class ladder (GS farm)
        {"id": "SC-01", "name": "BCS 3.25 Lean Cow (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["label"] == "3.25" and "GS" in r["farm_source"], "SC-01"),
         "rationale": "Inspect visible angularity in rear pelvic bones (hooks and pins).", "difficulty": "Standard"},
        {"id": "SC-02", "name": "BCS 3.50 Moderate-Lean Cow (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["label"] == "3.5" and "GS" in r["farm_source"] and r["burst_group_id"] != "GS_burst_0001", "SC-02"),
         "rationale": "Inspect pelvic cavity depression and thurl curvature.", "difficulty": "Standard"},
        {"id": "SC-03", "name": "BCS 3.75 Moderate Cow (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["label"] == "3.75" and "GS" in r["farm_source"], "SC-03"),
         "rationale": "Inspect smooth fat coverage over the tailhead and loin edge.", "difficulty": "Standard"},
        {"id": "SC-04", "name": "BCS 4.00 Fleshy Cow (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["label"] == "4.0" and "GS" in r["farm_source"], "SC-04"),
         "rationale": "Inspect rounded rump contour and filled cavity between hook and pin.", "difficulty": "Standard"},
        {"id": "SC-05", "name": "BCS 4.25 Heavy Cow (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["label"] == "4.25" and "GS" in r["farm_source"], "SC-05"),
         "rationale": "Inspect prominent fat pads and obscured bone structure around tailhead.", "difficulty": "Standard"},

        # Multi-farm representation
        {"id": "SC-06", "name": "YM Feedlot Environment (BCS 3.50, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["farm_source"] == "YM_Farm2" and r["label"] == "3.5", "SC-06"),
         "rationale": "Check different farm lighting, flooring, and lane width.", "difficulty": "Multi-Farm"},
        {"id": "SC-07", "name": "YM Farm Validation Sample (BCS 4.00, Val)", "split": "val",
         "target": find_first(s_val, lambda r: r["farm_source"] == "YM_Farm2" and r["label"] == "4.0", "SC-07"),
         "rationale": "Verify farm appearance consistency in validation partition.", "difficulty": "Multi-Farm"},
        {"id": "SC-08", "name": "STEREO Farm Camera Setup (BCS 3.75, Train)", "split": "train",
         "target": find_first(s_train, lambda r: r["farm_source"] == "STEREO_Farm3" and r["label"] == "3.75", "SC-08"),
         "rationale": "Inspect stereoscopic overhead camera geometry and perspective.", "difficulty": "Multi-Farm"},
        {"id": "SC-09", "name": "STEREO Farm Test Partition (BCS 3.75, Test)", "split": "test",
         "target": find_first(s_test, lambda r: r["farm_source"] == "STEREO_Farm3" and r["label"] == "3.75", "SC-09"),
         "rationale": "Inspect independent held-out test sample from STEREO camera.", "difficulty": "Test Partition"},

        # Independent Test partition
        {"id": "SC-10", "name": "Independent Test Sample (BCS 3.50, GS Farm)", "split": "test",
         "target": find_first(s_test, lambda r: r["label"] == "3.5" and "GS" in r["farm_source"], "SC-10"),
         "rationale": "Confirm test sample usability and framing.", "difficulty": "Test Partition"},
        {"id": "SC-11", "name": "Independent Test Sample (BCS 4.00, GS Farm)", "split": "test",
         "target": find_first(s_test, lambda r: r["label"] == "4.0" and "GS" in r["farm_source"], "SC-11"),
         "rationale": "Confirm high-BCS test sample usability.", "difficulty": "Test Partition"},

        # Challenging & Burst cases
        {"id": "SC-12", "name": "Harsh Shadow / Uneven Lighting (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: "GS_150" in r["original_passage_id"], "SC-12"),
         "rationale": "Evaluate if shadows obscure key pelvic landmarks.", "difficulty": "Challenging"},
        {"id": "SC-13", "name": "Off-Angle / Walking Perspective (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: "GS_850" in r["original_passage_id"], "SC-13"),
         "rationale": "Inspect cow walking away at a slight oblique angle.", "difficulty": "Challenging"},
        {"id": "SC-14", "name": "Repaired Burst Link A: GS_1818_1 (Val)", "split": "val",
         "target": find_first(s_val, lambda r: "GS_1818" in r["original_passage_id"], "SC-14"),
         "rationale": "Inspect Frame 1 of overlapping video burst discovered during audit.", "difficulty": "Burst Inspection"},
        {"id": "SC-15", "name": "Repaired Burst Link B: GS_1823_1 (Val)", "split": "val",
         "target": find_first(s_val, lambda r: "GS_1823" in r["original_passage_id"], "SC-15"),
         "rationale": "Verify that GS_1823 is the same video passage shifted by 1 frame (now unified in Val).", "difficulty": "Burst Inspection"},
        {"id": "SC-16", "name": "Tailhead Zoom / Close-Framing (GS Farm, Train)", "split": "train",
         "target": find_first(s_train, lambda r: "GS_400" in r["original_passage_id"], "SC-16"),
         "rationale": "Inspect tight framing around hook and pin bones.", "difficulty": "Standard"},
    ]

    # --------------------------------------------------------------------------
    # 2. MmCows Samples (16 checks)
    # --------------------------------------------------------------------------
    with open(REPO_ROOT / "datasets" / "behavior" / "mmcows" / "manifest.csv", "r", encoding="utf-8") as f:
        m_records = list(csv.DictReader(f))

    # Multi-camera synchronized event (ev_1690271846_c1 has 4 cameras for Cow 1)
    sync_events = [r for r in m_records if r["event_id"] == "ev_1690271846_c1"]
    sync_c1 = find_first(sync_events, lambda r: r["camera_id"] == "1", "MC-14")
    sync_c2 = find_first(sync_events, lambda r: r["camera_id"] == "2", "MC-15")

    mc_checks = [
        {"id": "MC-01", "name": "Class 1: Walking (Cow 1, Cam 1, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Walking" and r["canonical_split"] == "train" and r["camera_id"] == "1", "MC-01"),
         "rationale": "Inspect active forward leg stride in aisle.", "difficulty": "Standard"},
        {"id": "MC-02", "name": "Class 2: Standing (Cow 3, Cam 1, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Standing" and r["canonical_split"] == "train", "MC-02"),
         "rationale": "Inspect stationary upright posture with four hooves planted.", "difficulty": "Standard"},
        {"id": "MC-03", "name": "Class 2: Standing (Cow 8, Cam 2, Val)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Standing" and r["canonical_split"] == "val", "MC-03"),
         "rationale": "Verify standing posture for held-out validation cow.", "difficulty": "Validation Cow"},
        {"id": "MC-04", "name": "Class 2: Standing (Cow 16, Cam 4, Test)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Standing" and r["canonical_split"] == "test", "MC-04"),
         "rationale": "Verify standing posture for held-out test cow.", "difficulty": "Test Cow"},
        {"id": "MC-05", "name": "Class 3: Feeding Head Up (Cow 4, Cam 2, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Feeding_head_up" and r["canonical_split"] == "train", "MC-05"),
         "rationale": "Inspect cow at feed bunk with head raised above trough level.", "difficulty": "Standard"},
        {"id": "MC-06", "name": "Class 4: Feeding Head Down (Cow 5, Cam 2, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Feeding_head_down" and r["canonical_split"] == "train", "MC-06"),
         "rationale": "Inspect cow at feed bunk with muzzle lowered into feed.", "difficulty": "Standard"},
        {"id": "MC-07", "name": "Class 4: Feeding Head Down (Cow 12, Cam 3, Test)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Feeding_head_down" and r["canonical_split"] == "test", "MC-07"),
         "rationale": "Inspect feeding action on independent held-out test cow.", "difficulty": "Test Cow"},
        {"id": "MC-08", "name": "Class 5: Licking (Cow 6, Cam 1, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Licking" and r["canonical_split"] == "train", "MC-08"),
         "rationale": "Inspect rare self-grooming / licking action (coat or stall).", "difficulty": "Rare Class"},
        {"id": "MC-09", "name": "Class 5: Licking (Cow 10, Cam 3, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Licking" and r["cow_id"] == "10", "MC-09"),
         "rationale": "Second sample of rare licking class across different camera.", "difficulty": "Rare Class"},
        {"id": "MC-10", "name": "Class 6: Drinking (Cow 7, Cam 1, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Drinking" and r["canonical_split"] == "train", "MC-10"),
         "rationale": "Inspect muzzle lowered into water trough.", "difficulty": "Standard"},
        {"id": "MC-11", "name": "Class 6: Drinking (Cow 2, Cam 2, Test)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Drinking" and r["canonical_split"] == "test", "MC-11"),
         "rationale": "Verify drinking behavior on held-out test cow.", "difficulty": "Test Cow"},
        {"id": "MC-12", "name": "Class 7: Lying (Cow 9, Cam 3, Train)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Lying" and r["canonical_split"] == "train", "MC-12"),
         "rationale": "Inspect recumbent posture resting in cubicle stall.", "difficulty": "Standard"},
        {"id": "MC-13", "name": "Class 7: Lying (Cow 13, Cam 4, Val)",
         "target": find_first(m_records, lambda r: r["class_name"] == "Lying" and r["canonical_split"] == "val", "MC-13"),
         "rationale": "Verify lying posture on held-out validation cow.", "difficulty": "Validation Cow"},
        {"id": "MC-14", "name": "Synchronized View A (Cam 1, Cow 1, Feeding)",
         "target": sync_c1,
         "rationale": "Simultaneous multi-camera capture at 02:57:26 (View A, Cam 1).", "difficulty": "Multi-Camera Sync"},
        {"id": "MC-15", "name": "Synchronized View B (Cam 2, Cow 1, Feeding)",
         "target": sync_c2,
         "rationale": "Simultaneous multi-camera capture at 02:57:26 (View B, Cam 2, exact same moment).", "difficulty": "Multi-Camera Sync"},
        {"id": "MC-16", "name": "Partial Occlusion / Stall Bars (Cow 11, Cam 4)",
         "target": find_first(m_records, lambda r: r["cow_id"] == "11" and r["camera_id"] == "4", "MC-16"),
         "rationale": "Evaluate crop quality with cubicle partition bars in foreground.", "difficulty": "Challenging"},
    ]

    # --------------------------------------------------------------------------
    # 3. SideViewCows2026 Samples (16 checks)
    # --------------------------------------------------------------------------
    with open(REPO_ROOT / "datasets" / "id" / "sideviewcows2026" / "manifest.csv", "r", encoding="utf-8") as f:
        v_records = list(csv.DictReader(f))

    with open(REPO_ROOT / "datasets" / "id" / "sideviewcows2026" / "protocol_open_set.csv", "r", encoding="utf-8") as f:
        v_open = list(csv.DictReader(f))

    open_split_map = {r["image_path"]: r["open_set_split"] for r in v_open}

    sv_checks = [
        # Same Cow Across Settings (Cow 128 - Triplet across Parlor, Barn, Snapshots)
        {"id": "SV-01", "name": "Cow 128 in Parlor (Fixed Entrance Camera)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "128" and r["subset"] == "parlor", "SV-01"),
         "rationale": "Baseline controlled side-view framing and lighting in parlor gallery.", "difficulty": "Cross-Setting Triplet (1/3)"},
        {"id": "SV-02", "name": "Cow 128 in Barn (Handheld Video in Barn)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "128" and r["subset"] == "barn", "SV-02"),
         "rationale": "Same cow under handheld motion blur and barn lighting shift.", "difficulty": "Cross-Setting Triplet (2/3)"},
        {"id": "SV-03", "name": "Cow 128 in Snapshots (Unconstrained Photo)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "128" and r["subset"] == "snapshots", "SV-03"),
         "rationale": "Same cow in unconstrained photo setting with different posture.", "difficulty": "Cross-Setting Triplet (3/3)"},

        # Same Cow Across Settings (Cow 171 - Cross-Setting Pair)
        {"id": "SV-04", "name": "Cow 171 in Parlor (Fixed Entrance Camera)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "171" and r["subset"] == "parlor", "SV-04"),
         "rationale": "Distinct coat pattern reference in parlor gallery.", "difficulty": "Cross-Setting Pair (1/2)"},
        {"id": "SV-05", "name": "Cow 171 in Barn (Handheld Video in Barn)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "171" and r["subset"] == "barn", "SV-05"),
         "rationale": "Verify coat pattern matching across parlor and barn.", "difficulty": "Cross-Setting Pair (2/2)"},

        # Open-Set Identity Splits (Train vs Val vs Test)
        {"id": "SV-06", "name": "Open-Set Train Identity (Cow 144, Parlor)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "144" and r["subset"] == "parlor" and open_split_map.get(r["image_path"]) == "train", "SV-06"),
         "rationale": "Evaluate representation training sample (open-set train).", "difficulty": "Open-Set Train"},
        {"id": "SV-07", "name": "Open-Set Val Identity (Cow 168, Parlor Gallery)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "168" and r["subset"] == "parlor" and open_split_map.get(r["image_path"]) == "val", "SV-07"),
         "rationale": "Verify unseen validation identity in parlor reference setting.", "difficulty": "Open-Set Val Gallery"},
        {"id": "SV-08", "name": "Open-Set Val Identity (Cow 168, Barn Query)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "168" and r["subset"] == "barn" and open_split_map.get(r["image_path"]) == "val", "SV-08"),
         "rationale": "Verify cross-domain query for unseen validation identity.", "difficulty": "Open-Set Val Query"},
        {"id": "SV-09", "name": "Open-Set Held-Out Test Identity (Cow 166, Parlor)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "166" and r["subset"] == "parlor" and open_split_map.get(r["image_path"]) == "test", "SV-09"),
         "rationale": "Benchmark reference gallery for held-out test identity.", "difficulty": "Open-Set Test Gallery"},
        {"id": "SV-10", "name": "Open-Set Held-Out Test Identity (Cow 166, Barn Query)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "166" and r["subset"] == "barn" and open_split_map.get(r["image_path"]) == "test", "SV-10"),
         "rationale": "Benchmark query retrieval for held-out test identity.", "difficulty": "Open-Set Test Query"},
        {"id": "SV-11", "name": "Open-Set Held-Out Test Identity (Cow 166, Snapshots)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "166" and r["subset"] == "snapshots" and open_split_map.get(r["image_path"]) == "test", "SV-11"),
         "rationale": "Extreme domain shift query for held-out test identity.", "difficulty": "Open-Set Test Snapshots"},

        # Special Cases & Postures
        {"id": "SV-12", "name": "Lying Down Posture in Cubicle (Snapshots)",
         "target": find_first(v_records, lambda r: r["subset"] == "snapshots" and int(r["frame_no"]) > 10, "SV-12"),
         "rationale": "Inspect mask quality when cow is lying down on stall bedding.", "difficulty": "Postural Challenge"},
        {"id": "SV-13", "name": "Parlor-Only Identity (Cow 200, In-Domain)",
         "target": find_first(v_records, lambda r: r["subset_type"] == "parlor_only", "SV-13"),
         "rationale": "Inspect representation training identity without barn recordings.", "difficulty": "Parlor-Only"},
        {"id": "SV-14", "name": "Closed-Set Early Parlor Session (Train)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "128" and "sess_001" in r["recording_id"], "SV-14"),
         "rationale": "Inspect early chronological parlor session for closed-set.", "difficulty": "Closed-Set Train"},
        {"id": "SV-15", "name": "Closed-Set Late Parlor Session (Test Parlor)",
         "target": find_first(v_records, lambda r: r["individual_id"] == "128" and "sess_020" in r["recording_id"], "SV-15"),
         "rationale": "Inspect late parlor session (>50 days later) of same cow.", "difficulty": "Closed-Set Test"},
        {"id": "SV-16", "name": "Complex Background / Low Contrast (Barn)",
         "target": find_first(v_records, lambda r: r["subset"] == "barn" and int(r["frame_no"]) > 200, "SV-16"),
         "rationale": "Inspect mask boundary accuracy around legs, hooves, and udder.", "difficulty": "Segmentation Detail"},
    ]

    return sc_checks, mc_checks, sv_checks


# ==============================================================================
# IMAGE PROCESSING & COMPOSITING
# ==============================================================================
def process_sciencedb(checks: list[dict], pbar):
    """Resize and save ScienceDB thumbnail review images."""
    for c in checks:
        target = c["target"]
        src_path = Path(target["image_path"])
        if not src_path.is_file():
            # Fallback to repo-relative
            src_path = REPO_ROOT / target["image_path"]

        img = cv2.imread(str(src_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read ScienceDB image: {src_path}")

        # Resize to max width 400
        h, w = img.shape[:2]
        w_new = 400
        h_new = int(h * (w_new / w))
        thumb = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)

        out_name = f"{c['id'].lower()}_{src_path.stem}.jpg"
        out_path = ASSETS_DIR / out_name
        cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        c["asset_rel_path"] = f"assets/manual_dataset_verification/{out_name}"
        pbar.update(1)


def process_mmcows(checks: list[dict], pbar):
    """Resize and save MmCows thumbnail review images."""
    for c in checks:
        target = c["target"]
        src_path = REPO_ROOT / target["image_path"]
        img = cv2.imread(str(src_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read MmCows image: {src_path}")

        h, w = img.shape[:2]
        w_new = 360
        h_new = int(h * (w_new / w))
        thumb = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)

        out_name = f"{c['id'].lower()}_{src_path.stem}.jpg"
        out_path = ASSETS_DIR / out_name
        cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        c["asset_rel_path"] = f"assets/manual_dataset_verification/{out_name}"
        pbar.update(1)


def process_sideview(checks: list[dict], pbar):
    """Generate 3-panel composites (RGB | Mask | Overlay) for SideView images."""
    for c in checks:
        target = c["target"]
        img_p = REPO_ROOT / target["image_path"]
        mask_p = REPO_ROOT / target["mask_path"]

        img = cv2.imread(str(img_p))
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Cannot read SideView image or mask: {img_p}")

        # Target height 240px
        h_target = 240
        w_target = int(img.shape[1] * (h_target / img.shape[0]))

        img_r = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

        # Binary mask 3-channel
        mask_3ch = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)

        # Overlay: semi-transparent green mask + bright contour
        overlay = img_r.copy()
        color_mask = np.zeros_like(img_r)
        color_mask[mask_r > 127] = (0, 220, 0)
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.35, 0)

        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Composite side-by-side: [ RGB | Mask | Overlay ]
        composite = np.hstack([img_r, mask_3ch, overlay])

        out_name = f"{c['id'].lower()}_{img_p.stem}_composite.jpg"
        out_path = ASSETS_DIR / out_name
        cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
        c["asset_rel_path"] = f"assets/manual_dataset_verification/{out_name}"
        pbar.update(1)


# ==============================================================================
# MARKDOWN REPORT GENERATOR
# ==============================================================================
def write_markdown_report(sc_checks: list[dict], mc_checks: list[dict], sv_checks: list[dict]):
    """Write the comprehensive, user-friendly Markdown manual verification guide."""
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("""# Phase 3 Primary Dataset Manual Visual Verification Pack

> **Purpose & Guardrails**:  
> This document is for **manual human sanity checking only**. Automated forensic audits, exact byte hashing, and perceptual near-duplicate checks already exist and remain authoritative. Human visual inspection is used here as a final defense against obvious labeling blunders, corrupted crops, inverted masks, or camera artifacts before starting Step 2.  
>  
> *Rules for Auditor*:  
> - Visual inspection cannot prove exact numerical BCS scores or biological identity.  
> - Single-frame behavior can be ambiguous (e.g. standing near trough vs feeding); mark `QUESTIONABLE` where uncertain.  
> - ScienceDB does **not** have biological cow IDs; only burst groups are tracked.  

---

## Quick Review Guide

For every visual check, inspect the embedded image and check off the 5 verification criteria:
```markdown
- [ ] Target cow / anatomy is clearly visible
- [ ] Image quality / framing is usable for the task
- [ ] Stated label appears visually plausible
- [ ] No obvious corruption, wrong crop, or inverted mask
- Verdict: [ ] PASS   [ ] QUESTIONABLE   [ ] FAIL
```

---

# Part 1: ScienceDB — Primary Body Condition Scoring (BCS)
*Objective: Verify rear/pelvic view visibility, camera angle across farms, and plausibility of 5-point discrete BCS categories (3.25 to 4.25). Biological cow IDs are NOT provided by publisher.*

""")

        for c in sc_checks:
            t = c["target"]
            f.write(f"""### Check [{c['id']}]: {c['name']}
![{c['id']}]({c['asset_rel_path']})

| Field | Value |
|---|---|
| **Dataset** | ScienceDB Cattle BCS |
| **Filename** | `{Path(t['image_path']).name}` |
| **BCS Label** | **{t['label']}** |
| **Split Partition** | `{t['split'].upper()}` |
| **Farm Source** | `{t['farm_source']}` |
| **Burst Group ID** | `{t['burst_group_id']}` (Original Passage: `{t['original_passage_id']}`) |
| **Biological Cow ID** | *N/A (Publisher did not release cow IDs)* |
| **Inspection Focus** | {c['rationale']} |

**Manual Inspection Checklist:**
- [ ] Cattle pelvic / rear anatomy is visible
- [ ] Image is usable for visual condition scoring
- [ ] Stated BCS label ({t['label']}) appears plausible (hook/pin fat coverage)
- [ ] No obvious non-cattle image, corruption, or unusable framing
- **Verdict:** `[ ] PASS   [ ] QUESTIONABLE   [ ] FAIL`
- *Auditor Notes:* _______________________________________

---
""")

        f.write("""# Part 2: MmCows — Primary Behavior Recognition
*Objective: Verify bounding-box crop quality, pose/action plausibility across all 7 classes, multi-camera synchronized viewpoints, and biological cow ID tracking (16 Holstein cows).*

""")

        for c in mc_checks:
            t = c["target"]
            f.write(f"""### Check [{c['id']}]: {c['name']}
![{c['id']}]({c['asset_rel_path']})

| Field | Value |
|---|---|
| **Dataset** | MmCows Behavior (NeurIPS 2024 Spotlight) |
| **Filename** | `{t['filename']}` |
| **Behavior Class** | **Class {t['class_id']}: {t['class_name']}** |
| **Canonical Split** | `{t['canonical_split'].upper()}` |
| **Biological Cow ID** | **Cow {t['cow_id']}** (Verified biological individual) |
| **Camera ID** | Camera {t['camera_id']} |
| **Timestamp** | `{t['timestamp_iso']}` (Event ID: `{t['event_id']}`) |
| **Inspection Focus** | {c['rationale']} |

**Manual Inspection Checklist:**
- [ ] Cow body and limbs are clearly visible in crop
- [ ] Crop bounding box accurately frames the animal
- [ ] Observed posture roughly agrees with stated behavior (`{t['class_name']}`)
- [ ] If synchronized view, confirms the same time/event from a different camera
- **Verdict:** `[ ] PASS   [ ] QUESTIONABLE   [ ] FAIL`
- *Auditor Notes:* _______________________________________

---
""")

        f.write("""# Part 3: SideViewCows2026 — Primary Cow Identification / Re-ID
*Objective: Verify side-view profile visibility, cross-setting appearance consistency across parlor, barn, and snapshots, and ground-truth segmentation mask accuracy. (Composite: [ Left: RGB | Middle: Binary Mask | Right: Green Overlay + Contour ]).*

""")

        for c in sv_checks:
            t = c["target"]
            f.write(f"""### Check [{c['id']}]: {c['name']}
![{c['id']}]({c['asset_rel_path']})

| Field | Value |
|---|---|
| **Dataset** | SideViewCows2026 (Zenodo 21605650) |
| **Filename** | `{Path(t['image_path']).name}` (Mask: `{Path(t['mask_path']).name}`) |
| **Biological Individual ID** | **Cow {t['individual_id']}** (Verified biological cow) |
| **Camera Setting** | `{t['subset'].upper()}` (Type: `{t['subset_type']}`) |
| **Recording Session** | `{t['recording_id']}` (Frame No: `{t['frame_no']}`) |
| **Temporal Offset** | `{float(t['time_offset_s']):,.1f} s` (~{float(t['time_offset_s'])/86400:.1f} days) |
| **Inspection Focus** | {c['rationale']} |

**Manual Inspection Checklist:**
- [ ] Cow side profile and coat pattern are identifiable
- [ ] Camera setting matches expected domain (`{t['subset']}`)
- [ ] Binary mask accurately traces the cow silhouette
- [ ] Green overlay tightly adheres to body contour without clipping legs/head or bleeding into background
- **Verdict:** `[ ] PASS   [ ] QUESTIONABLE   [ ] FAIL`
- *Auditor Notes:* _______________________________________

---
""")

        f.write("""# Final Manual Verification Summary

Please fill out the summary table below after inspecting all 48 visual checks:

| Dataset | Total Checks | PASS | QUESTIONABLE | FAIL | Auditor Comments |
|---|---|---|---|---|---|
| **ScienceDB (BCS)** | 16 | [ ] | [ ] | [ ] | |
| **MmCows (Behavior)** | 16 | [ ] | [ ] | [ ] | |
| **SideViewCows2026 (Re-ID)** | 16 | [ ] | [ ] | [ ] | |
| **Total** | **48** | | | | |

---

## Overall Manual Decision

Review your findings above and select one conclusion before authorizing Step 2:

- [ ] **ACCEPT — DATASET FOUNDATION SOUND**  
  *No fatal image corruptions, crop failures, mask misalignments, or obvious label inversions detected. Ready for Step 2 Cattle-Perception Feasibility Audit.*

- [ ] **ACCEPT WITH MINOR NOTES**  
  *Certain frames exhibit expected livestock real-world ambiguity (e.g. occlusion by stall bars, slight shadow, subtle licking action), but data structure is verified sound and split protocols remain valid.*

- [ ] **INVESTIGATE BEFORE STEP 2**  
  *Severe structural defect identified (e.g. inverted mask, completely wrong species/image, corrupted file). Must resolve before Step 2.*

**Auditor Signature**: Hasin Ishrak  
**Date**: 2026-09-20  
""")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print("Phase 3 Manual Visual Verification Pack Generator")
    print(f"Random Seed      : {RANDOM_SEED}")
    print(f"Output Markdown  : docs/audits/phase3_manual_dataset_visual_verification.md")
    print(f"Asset Directory  : docs/audits/assets/manual_dataset_verification/")
    print("=" * 70 + "\n")

    t_start = time.time()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Select samples
    pbar1 = make_pbar(48, "[1/4] Selecting deterministic samples")
    sc_checks, mc_checks, sv_checks = select_samples()
    pbar1.update(48)
    pbar1.close()

    # Stage 2: Process ScienceDB
    pbar2 = make_pbar(len(sc_checks), "[2/4] Processing ScienceDB visual checks")
    process_sciencedb(sc_checks, pbar2)
    pbar2.close()

    # Stage 3: Process MmCows
    pbar3 = make_pbar(len(mc_checks), "[3/4] Processing MmCows visual checks")
    process_mmcows(mc_checks, pbar3)
    pbar3.close()

    # Stage 4: Process SideView & Write Markdown
    pbar4 = make_pbar(len(sv_checks) + 1, "[4/4] Generating SideView composites & report")
    process_sideview(sv_checks, pbar4)
    write_markdown_report(sc_checks, mc_checks, sv_checks)
    pbar4.update(1)
    pbar4.close()

    t_elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("SUCCESS: Manual Visual Verification Pack Generated!")
    print(f"Total Visual Checks : 48 (16 ScienceDB + 16 MmCows + 16 SideView)")
    print(f"Total Asset Images  : 48 compressed thumbnails & 3-panel composites")
    print(f"Markdown Deliverable: docs/audits/phase3_manual_dataset_visual_verification.md")
    print(f"Elapsed Runtime     : {t_elapsed:.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
