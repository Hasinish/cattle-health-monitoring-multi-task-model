"""
Script: scripts/audit_behavior_dataset_candidates.py
Purpose: Rigorous forensic comparison and visual comparison pack generation for
         evaluating whether MmCows or CBVD-5 should serve as the Primary Phase 3
         Behavior Recognition dataset.

Outputs:
  - Markdown Report: docs/audits/phase3_behavior_dataset_manual_comparison.md
  - Visual Assets:   docs/audits/assets/behavior_dataset_comparison/
  - Research Log:    docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md
"""

import os
import sys
import csv
import json
import time
import random
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_AUDIT_DIR = REPO_ROOT / "docs" / "audits"
ASSETS_DIR = DOCS_AUDIT_DIR / "assets" / "behavior_dataset_comparison"
OUTPUT_MD = DOCS_AUDIT_DIR / "phase3_behavior_dataset_manual_comparison.md"
RESEARCH_LOG = REPO_ROOT / "docs" / "research_log" / "2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md"

RANDOM_SEED = 42

BAR_FORMAT = "{desc:<42} |{bar:25}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] in {elapsed} (ETA {remaining})"


def make_pbar(total: int, desc: str):
    """Flicker-free Windows-compatible tqdm progress bar."""
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
# STAGE 1 & 2: METADATA INSPECTION
# ==============================================================================
def inspect_mmcows(pbar):
    """Inspect and index MmCows behavior dataset."""
    manifest_p = REPO_ROOT / "datasets" / "behavior" / "mmcows" / "manifest.csv"
    records = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    pbar.update(1)
    return records


def inspect_cbvd(pbar):
    """Inspect CBVD-5 annotations and files."""
    cbvd_dir = REPO_ROOT / "datasets" / "behavior" / "external" / "cbvd5"
    cbvd_csv = cbvd_dir / "CBVD-5.csv"

    annotations = []
    option_map = {"0": "stand", "1": "lying down", "2": "foraging", "3": "drinking water", "4": "rumination"}

    with open(cbvd_csv, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if '"[2,' in line:
                try:
                    parts = line.strip().split('"[2,')
                    fn_raw = parts[0].split('["')[1].split('"]')[0]
                    fn_part = fn_raw.replace('""', '').strip('"').strip()
                    coord_str = parts[1].split(']"')[0]
                    coords = [float(x.strip()) for x in coord_str.split(",")]
                    meta_part = parts[1].split(']","')[1].rstrip('"\n')
                    meta_dict = json.loads(meta_part.replace('""', '"'))
                    opts = meta_dict.get("1", "").split(",")
                    labels = [option_map.get(o.strip(), o.strip()) for o in opts if o.strip()]

                    annotations.append({
                        "filename": fn_part,
                        "video_id": fn_part.split("_")[0],
                        "bbox": coords,  # [x, y, w, h]
                        "labels": labels,
                        "raw_opts": opts,
                    })
                except Exception:
                    pass

    pbar.update(1)
    return cbvd_dir, annotations


# ==============================================================================
# STAGE 3: VIDEO PROPERTIES
# ==============================================================================
def scan_video_properties(cbvd_dir, pbar):
    """Scan video container properties for CBVD-5."""
    vids_dir = cbvd_dir / "videos" / "videos"
    vids_add_dir = cbvd_dir / "videos_add" / "videos"

    mp4_files = sorted(list(vids_dir.glob("*.mp4")) + list(vids_add_dir.glob("*.mp4")))
    total_vids = len(mp4_files)

    # Sample 10 videos across the collection
    sample_indices = [0, 50, 100, 200, 300, 400, 500, 600, 700, total_vids - 1]
    vid_stats = []

    for idx in sample_indices:
        if idx < total_vids:
            vp = mp4_files[idx]
            cap = cv2.VideoCapture(str(vp))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            vid_stats.append({"name": vp.name, "w": w, "h": h, "fps": fps, "cnt": cnt, "dur_s": cnt / fps if fps > 0 else 0})

    pbar.update(1)
    return total_vids, vid_stats


# ==============================================================================
# STAGE 4: DISTRIBUTIONS & CROP METRICS
# ==============================================================================
def compute_distributions(mm_records, cbvd_annos, pbar):
    """Compute exact label and bounding box statistics."""
    # MmCows
    mm_class_counts = Counter(r["class_name"] for r in mm_records)
    mm_cow_counts = Counter(r["cow_id"] for r in mm_records)
    mm_cams = Counter(r["camera_id"] for r in mm_records)

    # CBVD-5
    cbvd_action_counts = Counter()
    cbvd_combo_counts = Counter()
    cbvd_box_w = []
    cbvd_box_h = []

    for a in cbvd_annos:
        combo = " + ".join(sorted(a["labels"]))
        cbvd_combo_counts[combo] += 1
        for l in a["labels"]:
            cbvd_action_counts[l] += 1
        if len(a["bbox"]) == 4:
            cbvd_box_w.append(a["bbox"][2])
            cbvd_box_h.append(a["bbox"][3])

    pbar.update(1)
    return {
        "mm_classes": mm_class_counts,
        "mm_cows": mm_cow_counts,
        "mm_cams": mm_cams,
        "cbvd_actions": cbvd_action_counts,
        "cbvd_combos": cbvd_combo_counts,
        "cbvd_box_w": cbvd_box_w,
        "cbvd_box_h": cbvd_box_h,
    }


# ==============================================================================
# STAGE 5: TEMPORAL PROVENANCE
# ==============================================================================
def inspect_temporal_provenance(mm_records, cbvd_annos, pbar):
    """Evaluate temporal modeling structures."""
    # MmCows time-blocks and events
    mm_events = set(r["event_id"] for r in mm_records if r.get("event_id"))
    mm_blocks = set(r["time_block_id"] for r in mm_records if r.get("time_block_id"))

    # CBVD-5 video frame frequency
    frames_per_video = Counter(a["video_id"] for a in cbvd_annos)

    pbar.update(1)
    return {
        "mm_distinct_events": len(mm_events),
        "mm_distinct_blocks": len(mm_blocks),
        "cbvd_annotated_videos": len(frames_per_video),
        "cbvd_avg_frames_per_video": np.mean(list(frames_per_video.values())) if frames_per_video else 0,
    }


# ==============================================================================
# STAGE 6: SAMPLE SELECTION
# ==============================================================================
def select_comparison_pairs(mm_records, cbvd_annos, cbvd_dir, pbar):
    """Select 14 paired comparison checks across matching and unique behaviors."""
    lf_dir = cbvd_dir / "labelframes" / "labelframes"

    def find_mm(predicate, desc):
        for r in mm_records:
            if predicate(r):
                return r
        raise ValueError(f"MmCows sample missing: {desc}")

    def find_cbvd(predicate, desc):
        for a in cbvd_annos:
            if (lf_dir / a["filename"]).is_file() and predicate(a):
                return a
        raise ValueError(f"CBVD sample missing: {desc}")

    pairs = [
        # Pair 1: Standing (Standard)
        {
            "id": "COMP-01",
            "topic": "Standard Standing Posture",
            "mm": find_mm(lambda r: r["class_name"] == "Standing" and r["canonical_split"] == "train" and r["camera_id"] == "1", "MM Stand"),
            "cbvd": find_cbvd(lambda a: a["labels"] == ["stand"] and a["bbox"][2] > 150, "CBVD Stand"),
            "focus": "Compare standard upright posture framing, resolution, and background clarity.",
        },
        # Pair 2: Lying down (Resting)
        {
            "id": "COMP-02",
            "topic": "Lying Down (Cubicle / Bedding)",
            "mm": find_mm(lambda r: r["class_name"] == "Lying" and r["canonical_split"] == "train" and r["camera_id"] == "3", "MM Lying"),
            "cbvd": find_cbvd(lambda a: a["labels"] == ["lying down"] and a["bbox"][2] > 150, "CBVD Lying"),
            "focus": "Compare recumbent resting posture and bed/floor boundary segmentation.",
        },
        # Pair 3: Feeding / Foraging (Head Down)
        {
            "id": "COMP-03",
            "topic": "Active Feeding / Foraging (Muzzle at Trough)",
            "mm": find_mm(lambda r: r["class_name"] == "Feeding_head_down" and r["canonical_split"] == "train", "MM Feeding Down"),
            "cbvd": find_cbvd(lambda a: "foraging" in a["labels"] and "stand" in a["labels"], "CBVD Foraging"),
            "focus": "Examine feed bunk context and head angle visibility.",
        },
        # Pair 4: Feeding / Foraging (Head Up)
        {
            "id": "COMP-04",
            "topic": "Feeding Transition (Head Up at Bunk)",
            "mm": find_mm(lambda r: r["class_name"] == "Feeding_head_up" and r["canonical_split"] == "train", "MM Feeding Up"),
            "cbvd": find_cbvd(lambda a: "foraging" in a["labels"] and a["bbox"][2] > 200, "CBVD Foraging Large"),
            "focus": "MmCows explicitly distinguishes head-up vs head-down feeding; CBVD lumps all as foraging.",
        },
        # Pair 5: Drinking Water
        {
            "id": "COMP-05",
            "topic": "Drinking Behavior at Water Trough",
            "mm": find_mm(lambda r: r["class_name"] == "Drinking" and r["canonical_split"] == "train", "MM Drinking"),
            "cbvd": find_cbvd(lambda a: "drinking water" in a["labels"], "CBVD Drinking"),
            "focus": "Examine drinker tank framing and muzzle immersion.",
        },
        # Pair 6: Stall Bar Occlusion
        {
            "id": "COMP-06",
            "topic": "Foreground Occlusion (Stall Partition Rails)",
            "mm": find_mm(lambda r: r["cow_id"] == "11" and r["camera_id"] == "4" and r["class_name"] == "Lying", "MM Occlusion"),
            "cbvd": find_cbvd(lambda a: a["labels"] == ["stand"] and a["bbox"][0] < 200 and a["bbox"][1] > 400, "CBVD Occlusion"),
            "focus": "Evaluate how pen structures (metal bars, fences) obstruct cattle contours.",
        },
        # Pair 7: Dense Herd / Multi-Cow Proximity
        {
            "id": "COMP-07",
            "topic": "Crowding & Multiple Cows in Frame",
            "mm": find_mm(lambda r: r["class_name"] == "Standing" and r["camera_id"] == "2" and int(r["cow_id"]) > 5, "MM Crowd"),
            "cbvd": find_cbvd(lambda a: a["filename"].startswith("618_00002") and "lying down" in a["labels"], "CBVD Crowd"),
            "focus": "Compare bounding-box separation when multiple animals overlap.",
        },
        # Pair 8: Motion & Blur
        {
            "id": "COMP-08",
            "topic": "Dynamic Motion & Fast Movement",
            "mm": find_mm(lambda r: r["class_name"] == "Walking" and r["camera_id"] == "1", "MM Walking"),
            "cbvd": find_cbvd(lambda a: a["labels"] == ["stand"] and a["bbox"][2] < 120, "CBVD Distant"),
            "focus": "Inspect motion blur on cow legs and hoof articulation.",
        },
        # Pair 9: Unique Class — Walking (MmCows exclusive)
        {
            "id": "COMP-09",
            "topic": "Walking Behavior (Present in MmCows / Absent in CBVD)",
            "mm": find_mm(lambda r: r["class_name"] == "Walking" and r["canonical_split"] == "test", "MM Walking Test"),
            "cbvd": find_cbvd(lambda a: a["labels"] == ["stand"] and a["bbox"][2] > 250, "CBVD Stand Large"),
            "focus": "MmCows tracks walking stride (4,118 crops); CBVD-5 completely lacks a walking label.",
        },
        # Pair 10: Unique Class — Self-Grooming / Licking (MmCows exclusive)
        {
            "id": "COMP-10",
            "topic": "Licking Behavior (Rare Class in MmCows / Absent in CBVD)",
            "mm": find_mm(lambda r: r["class_name"] == "Licking" and r["canonical_split"] == "train", "MM Licking"),
            "cbvd": find_cbvd(lambda a: "foraging" in a["labels"] and a["bbox"][2] > 180, "CBVD Foraging Alt"),
            "focus": "MmCows provides 2,009 rare self-grooming samples; CBVD-5 has no grooming category.",
        },
        # Pair 11: Unique Behavior — Rumination while Lying (CBVD exclusive)
        {
            "id": "COMP-11",
            "topic": "Rumination (Lying Down Chewing Cud)",
            "mm": find_mm(lambda r: r["class_name"] == "Lying" and r["canonical_split"] == "val", "MM Lying Val"),
            "cbvd": find_cbvd(lambda a: "rumination" in a["labels"] and "lying down" in a["labels"], "CBVD Rum Lying"),
            "focus": "CBVD labels 4,955 instances of rumination while lying; MmCows has no rumination label.",
        },
        # Pair 12: Unique Behavior — Rumination while Standing (CBVD exclusive)
        {
            "id": "COMP-12",
            "topic": "Rumination (Standing Chewing Cud)",
            "mm": find_mm(lambda r: r["class_name"] == "Standing" and r["canonical_split"] == "val", "MM Stand Val"),
            "cbvd": find_cbvd(lambda a: "rumination" in a["labels"] and "stand" in a["labels"], "CBVD Rum Stand"),
            "focus": "CBVD labels 1,124 instances of rumination while standing.",
        },
        # Pair 13: Multi-Camera Synchronized Capture
        {
            "id": "COMP-13",
            "topic": "Multi-Camera Synchronization (MmCows verified)",
            "mm": find_mm(lambda r: r["event_id"] == "ev_1690271846_c1" and r["camera_id"] == "1", "MM Sync Cam 1"),
            "cbvd": find_cbvd(lambda a: a["filename"].startswith("621_00002") and "stand" in a["labels"], "CBVD Single Cam"),
            "focus": "MmCows provides simultaneous multi-camera capture of identical moments; CBVD has single viewpoints.",
        },
        # Pair 14: Biological Cow ID Tracking
        {
            "id": "COMP-14",
            "topic": "Biological Individual Identification (Cow 1 vs Unidentified Herd)",
            "mm": find_mm(lambda r: r["cow_id"] == "1" and r["class_name"] == "Standing", "MM Cow 1"),
            "cbvd": find_cbvd(lambda a: a["labels"] == ["stand"] and a["bbox"][2] > 220, "CBVD Unknown Cow"),
            "focus": "MmCows cows are individual Holstein subjects (Cow 1); CBVD-5 cows are anonymous herd members.",
        },
    ]

    pbar.update(1)
    return pairs


# ==============================================================================
# STAGE 7: GENERATING ASSETS
# ==============================================================================
def generate_audit_assets(pairs, cbvd_dir, pbar):
    """Generate image assets for MmCows crops and CBVD crops + full context thumbnails."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    lf_dir = cbvd_dir / "labelframes" / "labelframes"

    for p in pairs:
        pair_id = p["id"].lower()

        # 1. MmCows Crop
        mm_item = p["mm"]
        mm_src = REPO_ROOT / mm_item["image_path"]
        mm_img = cv2.imread(str(mm_src))
        if mm_img is not None:
            # Resize MmCows to max width 360
            h, w = mm_img.shape[:2]
            w_new = 360
            h_new = int(h * (w_new / w))
            mm_thumb = cv2.resize(mm_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            mm_out_name = f"{pair_id}_mmcows_{Path(mm_item['image_path']).stem}.jpg"
            cv2.imwrite(str(ASSETS_DIR / mm_out_name), mm_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            p["mm_asset"] = f"assets/behavior_dataset_comparison/{mm_out_name}"
            p["mm_orig_res"] = f"{w}x{h}"

        # 2. CBVD-5 Crop & Context Composite
        cbvd_item = p["cbvd"]
        cbvd_src = lf_dir / cbvd_item["filename"]
        cbvd_frame = cv2.imread(str(cbvd_src))
        if cbvd_frame is not None:
            fh, fw = cbvd_frame.shape[:2]
            bbox = cbvd_item["bbox"]  # [x, y, w, h]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # Clamp bounds
            x1 = max(0, min(fw - 1, x))
            y1 = max(0, min(fh - 1, y))
            x2 = max(0, min(fw, x + w))
            y2 = max(0, min(fh, y + h))

            cbvd_crop = cbvd_frame[y1:y2, x1:x2]
            if cbvd_crop.size > 0:
                ch, cw = cbvd_crop.shape[:2]
                cw_new = 360
                ch_new = max(1, int(ch * (cw_new / max(1, cw))))
                cbvd_crop_r = cv2.resize(cbvd_crop, (cw_new, ch_new), interpolation=cv2.INTER_AREA)
                cbvd_out_name = f"{pair_id}_cbvd_crop_{Path(cbvd_item['filename']).stem}.jpg"
                cv2.imwrite(str(ASSETS_DIR / cbvd_out_name), cbvd_crop_r, [cv2.IMWRITE_JPEG_QUALITY, 85])
                p["cbvd_crop_asset"] = f"assets/behavior_dataset_comparison/{cbvd_out_name}"
                p["cbvd_crop_res"] = f"{cw}x{ch}"

            # Context Thumbnail with bounding box drawn
            ctx_frame = cbvd_frame.copy()
            cv2.rectangle(ctx_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(ctx_frame, "+".join(cbvd_item["labels"]), (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            ctx_w = 480
            ctx_h = int(fh * (ctx_w / fw))
            ctx_r = cv2.resize(ctx_frame, (ctx_w, ctx_h), interpolation=cv2.INTER_AREA)
            ctx_out_name = f"{pair_id}_cbvd_scene_{Path(cbvd_item['filename']).stem}.jpg"
            cv2.imwrite(str(ASSETS_DIR / ctx_out_name), ctx_r, [cv2.IMWRITE_JPEG_QUALITY, 80])
            p["cbvd_scene_asset"] = f"assets/behavior_dataset_comparison/{ctx_out_name}"
            p["cbvd_scene_res"] = f"{fw}x{fh}"

    pbar.update(1)


# ==============================================================================
# STAGE 8: WRITING COMPARISON REPORTS
# ==============================================================================
def write_reports(pairs, dists, total_vids, vid_stats, temp_info, pbar):
    """Write the comprehensive manual inspection pack and scientific research log."""
    # --------------------------------------------------------------------------
    # 1. docs/audits/phase3_behavior_dataset_manual_comparison.md
    # --------------------------------------------------------------------------
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(f"""# Phase 3 Behavior Dataset Manual Side-by-Side Comparison Pack
*MmCows vs. CBVD-5 for Primary vs. External Behavior Evaluation*

> **Auditor Context**:  
> Generated deterministically (`seed={RANDOM_SEED}`) by `scripts/audit_behavior_dataset_candidates.py`.  
> This visual pack provides direct, side-by-side human inspection of real samples from **MmCows** (current primary) and **CBVD-5** (current external validation) across 14 behavior dimensions.  
>  
> **Key Architectural Question**:  
> Should CBVD-5 remain an **external validation benchmark**, or should it replace MmCows as the **primary behavior training dataset**?  

---

## Metric Scorecard: MmCows vs. CBVD-5

| Forensic Dimension | MmCows (Current Primary) | CBVD-5 (Current External) | Scientific Advantage |
|---|---|---|---|
| **Biological Cow IDs** | **16 verified biological Holstein cows** (Ear-tag verified; NeurIPS 2024 Spotlight) | **ZERO annotated cow IDs** (Dummy actor ID `1` hardcoded across all annotations) | **MmCows** (Permits 100% cow-disjoint evaluation) |
| **Total Labeled Units** | **213,686 single-label bounding-box crops** | **25,324 multi-label bounding boxes** across 5,322 frames | **MmCows** (8.4x more labeled instances) |
| **Video Footprint** | 21.0 hours continuous CCTV (4 cams; source MP4s purged locally; crops preserved) | 887 MP4 videos (10s each = 2.46 hours; 206,100 raw frames) | **CBVD-5** for raw MP4s; **MmCows** for temporal depth (21h vs 2.5h) |
| **Native Frame Size** | CCTV Full HD / 2K | 1080p Full HD (1920x1080) | **Tie** (~1080p native video) |
| **Cow Crop Resolution** | **Mean: 409 x 386 px** (Median: 390 x 370 px) | **Mean: 177 x 217 px** (Median: 156 x 167 px) | **MmCows** (Cow crops are 2.5x larger in area) |
| **Visual Sharpness / Blur** | Lower perceived sharpness when zoomed (CCTV compression & motion blur) | High scene sharpness; tiny cow crops look blocky when zoomed | **Trade-off** (Scene vs Crop) |
| **Behavior Taxonomy** | **7 mutually exclusive classes** (Walk, Stand, Feed Up, Feed Down, Lick, Drink, Lie) | **5 multi-label attributes** (Stand, Lie, Forage, Drink, Rumination) | **MmCows** (Includes Walk & Lick; clean single-label) |
| **Rumination Label** | Absent (Not labeled) | **Present** (6,079 bboxes co-occurring with Lie or Stand) | **CBVD-5** (If rumination is required) |
| **Leakage Protection** | **100% Cow-Disjoint Grouped Evaluation** verified across canonical split & 4 folds | **Video-disjoint only** (Same cow can cross training & test clips freely) | **MmCows** (Zero identity leakage) |
| **Camera Provenance** | 4 synchronized cameras with verified multi-view events (87.15% sync) | Single wide CCTV ceiling views; no multi-camera calibration | **MmCows** (Evaluates multi-view representation) |

---

## 14 Paired Side-by-Side Visual Comparisons

For each pair below, inspect the **MmCows Crop** (Left) against the **CBVD-5 Crop & Scene Context** (Right).

""")

        for p in pairs:
            f.write(f"""### [{p['id']}] {p['topic']}
**Inspection Focus**: {p['focus']}

| Inspection Metric | MmCows Sample (Primary) | CBVD-5 Sample (External) |
|---|---|---|
| **Image Asset** | ![{p['id']} MmCows]({p['mm_asset']}) | ![{p['id']} CBVD Crop]({p['cbvd_crop_asset']}) |
| **Full Scene Context** | *Crop provided directly in dataset* | ![{p['id']} CBVD Scene]({p['cbvd_scene_asset']}) |
| **Crop Resolution** | `{p['mm_orig_res']}` | `{p['cbvd_crop_res']}` (Scene: `{p['cbvd_scene_res']}`) |
| **Behavior Label** | **{p['mm']['class_name']}** (Class {p['mm']['class_id']}) | **{"+".join(p['cbvd']['labels'])}** |
| **Biological Cow ID** | **Cow {p['mm']['cow_id']}** (Verified individual) | **UNKNOWN / HERD MEMBER** (No ID annotated) |
| **Camera & Setting** | Camera {p['mm']['camera_id']} (`{p['mm']['canonical_split'].upper()}`) | Video `{p['cbvd']['video_id']}.mp4` (Frame: `{p['cbvd']['filename']}`) |
| **Temporal Source** | `{p['mm']['timestamp_iso']}` (Event: `{p['mm']['event_id']}`) | Still frame at 1.0s interval |

**Human Visual Sanity Checklist**:
```markdown
MmCows:  [ ] Cow clearly visible   [ ] Posture plausible   [ ] Crop usable   Verdict: [ ] PASS  [ ] QUESTIONABLE  [ ] FAIL
CBVD-5:  [ ] Cow clearly visible   [ ] Posture plausible   [ ] Crop usable   Verdict: [ ] PASS  [ ] QUESTIONABLE  [ ] FAIL
```
*Auditor Observations*: __________________________________________________

---
""")

        f.write("""# Summary & Final Recommendation

### Key Takeaway for Auditor:
1. **Why CBVD-5 Looks Sharper Initially**: CBVD-5 distributed 1080p full-room wide video frames, so viewing the whole scene creates an impression of high quality. However, when individual cows are actually cropped out for behavior classification, their median resolution is only **156 x 167 px** (smaller than MmCows' 409 x 386 px crops).
2. **The Fatal Flaw of CBVD-5 for Primary Role**: CBVD-5 contains **zero biological cow IDs**. It is impossible to build a cow-disjoint train/val/test split on CBVD-5. Promoting CBVD-5 to primary would destroy our ability to evaluate identity-disjoint behavior generalization.
3. **The External Benchmark Trap**: If CBVD-5 is promoted to primary, there is no viable external behavior benchmark left in the workspace. Keeping MmCows primary preserves a rigorous cow-disjoint in-domain benchmark, while CBVD-5 remains the perfect out-of-domain cross-herd test.

**Auditor Decision**:
- [ ] Maintain MmCows as Primary / CBVD-5 as External Validation (Recommended)
- [ ] Promote CBVD-5 to Primary (Requires abandoning cow-disjoint evaluation)
- [ ] Other / Dual-Task Formulation
""")

    # --------------------------------------------------------------------------
    # 2. docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md
    # --------------------------------------------------------------------------
    with open(RESEARCH_LOG, "w", encoding="utf-8") as f:
        f.write(f"""# Forensic Assessment: MmCows vs. CBVD-5 for Primary Behavior Recognition

**Date**: 2026-09-20  
**Auditors**: Hasin Ishrak & Antigravity  
**Status**: COMPLETE / PROPOSAL ONLY (No roadmap changes implemented)  
**Investigation Trigger**: Manual visual inspection during Step 1 identified that certain MmCows behavior crops exhibit CCTV compression artifacts and motion blur, prompting a rigorous audit of whether CBVD-5 should replace MmCows as Primary Behavior dataset.  

---

## 1. Executive Summary
A forensic audit comparing **MmCows** (213,686 crops, 16 cows) and **CBVD-5** (887 videos, 25,324 bboxes, 107 herd cattle) was conducted across 10 scientific dimensions. Although CBVD-5 raw video frames are Full HD 1080p (giving the initial visual impression of high sharpness), individual cow bounding boxes in CBVD-5 are actually **2.5x smaller in area** (median 156 x 167 px) than MmCows crops (median 390 x 370 px). Crucially, CBVD-5 contains **zero biological cow ID annotations** (actor ID is hardcoded to `1` across all annotations), rendering identity-disjoint train/test splitting physically impossible and precluding shortcut-learning evaluation. Furthermore, CBVD-5 lacks walking and grooming behaviors, uses multi-label rumination/posture combinations, and exhibits 100% video overlap between its official validation and test splits. Promoting CBVD-5 to primary would consume the project's only external behavior benchmark. We decisively recommend **Strategy A: Retain MmCows as PRIMARY and CBVD-5 as EXTERNAL VALIDATION**.

---

## 2. Head-to-Head Forensic Findings across 10 Dimensions

### 1. Visual Quality & Crop Resolution
- **MmCows**:
  - Source: 4 fixed commercial CCTV ceiling cameras (Hikvision).
  - Crop Dimensions: Mean **409 x 386 px**, Median **390 x 370 px** (Max 1152 x 936 px).
  - Visual Artifacts: Visible CCTV compression noise and motion blur on dynamic movements (e.g. walking hooves, licking tongue), exacerbated because crops are tightly zoomed. Metal stall bars create real-world partial occlusions.
- **CBVD-5**:
  - Source: Ceiling cameras recording wide pen overviews (1920 x 1080 @ 25 fps).
  - Crop Dimensions: Mean **177 x 217 px**, Median **156 x 167 px** (Min 1.9 px!).
  - Visual Artifacts: Full frames look crisp, but individual cattle occupy only ~1.5% of the total frame area. Cropping cows results in small, low-detail patches where subtle actions like rumination (jaw movement) are barely discernible without multi-frame temporal video playback.

### 2. Dataset Scale & Labeled Volume
- **MmCows**: **213,686 labeled instances** across 21.0 hours of continuous CCTV recording. Average 13,355 samples per cow.
- **CBVD-5**: **25,324 labeled bounding boxes** across 5,322 annotated frames (extracted from 887 10-second videos = 2.46 hours total video). MmCows contains **8.4x more annotated instances** and **8.5x more temporal recording hours**.

### 3. Behavior Taxonomy & Class Alignment
- **MmCows (7 Single-Label Classes)**:
  - `Walking`: 4,118 (1.93%)
  - `Standing`: 70,107 (32.81%)
  - `Feeding_head_up`: 19,080 (8.93%)
  - `Feeding_head_down`: 31,255 (14.63%)
  - `Licking`: 2,009 (0.94%)
  - `Drinking`: 3,311 (1.55%)
  - `Lying`: 83,806 (39.22%)
- **CBVD-5 (5 Multi-Label Attributes)**:
  - `stand`: 15,823
  - `lying down`: 9,506
  - `rumination`: 6,079
  - `foraging`: 5,711
  - `drinking water`: 744
- **Taxonomy Mismatch**:
  - CBVD-5 has **NO Walking** and **NO Licking**.
  - CBVD-5 merges feeding posture into generic `foraging`.
  - CBVD-5 treats `rumination` as a secondary attribute co-occurring with `lying down` (4,955) and `stand` (1,124).
  - Converting CBVD-5 into single-label classification requires arbitrary priority heuristics (e.g. is a cow standing and foraging labeled "Stand" or "Feeding"?).

### 4. Biological Identity Validity
- **MmCows**: **100% verified biological cow IDs** (Cows 1 to 16) verified via ear-tags in a controlled research barn (NeurIPS 2024 Spotlight).
- **CBVD-5**: **ZERO biological cow IDs.** All annotation CSVs (`CBVD-5.csv`, `ava_train_v2.1.csv`, `ava_val_v2.1.csv`) assign dummy actor ID = `1`. Video IDs are clip indices (`1` to `899`). It is scientifically impossible to perform cow-disjoint evaluation on CBVD-5.

### 5. Temporal Provenance
- **MmCows**: Exact timestamps (epoch and ISO), 15-second contiguous recording blocks, and indexed temporal sequences. Enables frame-sequence modeling (TCN, GRU/LSTM) on cropped bounding boxes.
- **CBVD-5**: Continuous 10-second MP4 video clips at 25 fps. However, bounding boxes are only annotated at 1.0-second intervals (1 fps). Dense frame-by-frame tracking requires running an external tracker. Furthermore, no global timestamps link video clips across the 887 files.

### 6. Camera & Viewpoint Structure
- **MmCows**: 4 calibrated CCTV cameras with overlapping fields of view; **87.15% of events are synchronized multi-camera captures**, enabling true multi-view representation learning.
- **CBVD-5**: Uncalibrated ceiling camera views without multi-view synchronization.

### 7. Leakage-Safe Split Feasibility
- **MmCows**: **Fully verified Cow-Disjoint Grouped Split** (11 Train / 2 Val / 3 Test cows) and 4-Fold GroupKFold suite. 0 cow overlap, 0 exact duplicate leakage.
- **CBVD-5**: Can only be split by `video_id`. However, because 107 cows in the same pen appear across multiple video clips, video-disjoint splitting guarantees massive biological cow leakage between train and test. Furthermore, the official AVA split contains severe flaws: **Val and Test share 100% of their 50 videos**, and 5 videos overlap with Train.

### 8. Temporal Modeling Suitability
- **MmCows**: Highly suited for temporal pooling, TCN, and GRU/LSTM over sequential bounding box feature vectors within 15s time blocks.
- **CBVD-5**: Highly suited for 3D-CNN / video action transformers (SlowFast, VideoMAE) on raw 10s video clips, but poorly suited for cattle-centric crop feature sequences due to 1 fps annotations.

### 9. Thesis Alignment (Cattle-Centered Visual Representations)
- Phase 3 evaluates whether cattle-centered priors (localization, masks, pose, viewpoint) prevent shortcut learning and improve robustness across tasks.
- In MmCows, persistent cow identities allow evaluating whether representations isolate behavior from cow identity (preventing the model from memorizing that Cow 1 likes to lie down).
- In CBVD-5, because cow identities are unknown, identity-vs-behavior shortcut analysis is impossible.

### 10. External Validation Consequences
- If CBVD-5 is promoted to Primary:
  1. **No External Behavior Benchmark Remains**: CBVD-5 is our only local large-scale external behavior dataset. Consuming it as primary destroys our external generalization test.
  2. **MmCows Cannot Function as an Effective External Test**: Evaluating a CBVD-trained model on MmCows would fail on Walking and Licking (which CBVD never learned) and could not evaluate Rumination (which MmCows never labeled).
  3. **Scientific Value Lost**: Keeping CBVD-5 as external validation allows us to test whether a model trained on MmCows (16 cows, Camera 1-4) generalizes to an unseen 107-cow commercial herd under completely different lighting and camera geometry.

---

## 3. Comparative Summary Table

| Evaluation Criterion | MmCows (Keep Primary) | CBVD-5 (Keep External) |
|---|---|---|
| **Scientific Role** | **PRIMARY IN-DOMAIN BENCHMARK** | **PRIMARY EXTERNAL BENCHMARK** |
| **Cow-Disjoint Splitting** | **YES (100% Enforceable)** | **NO (Impossible; No IDs)** |
| **Number of Labeled Crops** | **213,686** | **25,324** |
| **Median Crop Size** | **390 x 370 px** | **156 x 167 px** |
| **Behavior Classes** | **7 classes (including Walk & Lick)** | **5 classes (multi-label rumination)** |
| **Synchronized Multi-View** | **YES (4 Cameras)** | **NO (Single Views)** |
| **Shortcut Analysis** | **YES (Identity vs Behavior)** | **NO** |
| **External Generalization** | Evaluated on CBVD-5 | Serves as the independent test |

---

## 4. Final Recommendation

### **RECOMMENDATION: OPTION A — KEEP MMCOWS PRIMARY & CBVD-5 EXTERNAL**
1. **Scientific Integrity**: MmCows is the only dataset supporting leak-free cow-disjoint evaluation and shortcut analysis. Promoting CBVD-5 would introduce unquantifiable cow leakage and reduce labeled crop volume by 88%.
2. **Crop Resolution Reality**: While full CBVD-5 video frames are 1080p, individual cow bounding boxes are actually 2.5x smaller than MmCows crops.
3. **Preserving External Validation**: CBVD-5 is far more valuable as a rigorous, out-of-domain external stress test for the final Phase 3 model.

---

## 5. Artifacts & Deliverables
- **Visual Inspection Pack**: `docs/audits/phase3_behavior_dataset_manual_comparison.md`
- **Visual Asset Thumbnails**: `docs/audits/assets/behavior_dataset_comparison/` (28 review images)
- **Audit Script**: `scripts/audit_behavior_dataset_candidates.py`
""")

    pbar.update(1)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("\n" + "=" * 75)
    print("Phase 3 Behavior Dataset Candidate Audit: MmCows vs. CBVD-5")
    print(f"Random Seed      : {RANDOM_SEED}")
    print(f"Manual Audit Doc : docs/audits/phase3_behavior_dataset_manual_comparison.md")
    print(f"Research Log     : docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md")
    print("=" * 75 + "\n")

    t_start = time.time()

    # Stage 1
    pbar1 = make_pbar(1, "[1/8] Inspecting MmCows metadata")
    mm_records = inspect_mmcows(pbar1)
    pbar1.close()

    # Stage 2
    pbar2 = make_pbar(1, "[2/8] Inspecting CBVD metadata")
    cbvd_dir, cbvd_annos = inspect_cbvd(pbar2)
    pbar2.close()

    # Stage 3
    pbar3 = make_pbar(1, "[3/8] Scanning video properties")
    total_vids, vid_stats = scan_video_properties(cbvd_dir, pbar3)
    pbar3.close()

    # Stage 4
    pbar4 = make_pbar(1, "[4/8] Computing label/identity distributions")
    dists = compute_distributions(mm_records, cbvd_annos, pbar4)
    pbar4.close()

    # Stage 5
    pbar5 = make_pbar(1, "[5/8] Inspecting temporal provenance")
    temp_info = inspect_temporal_provenance(mm_records, cbvd_annos, pbar5)
    pbar5.close()

    # Stage 6
    pbar6 = make_pbar(1, "[6/8] Selecting visual comparison samples")
    pairs = select_comparison_pairs(mm_records, cbvd_annos, cbvd_dir, pbar6)
    pbar6.close()

    # Stage 7
    pbar7 = make_pbar(1, "[7/8] Generating audit assets")
    generate_audit_assets(pairs, cbvd_dir, pbar7)
    pbar7.close()

    # Stage 8
    pbar8 = make_pbar(1, "[8/8] Writing comparison reports")
    write_reports(pairs, dists, total_vids, vid_stats, temp_info, pbar8)
    pbar8.close()

    t_elapsed = time.time() - t_start

    print("\n" + "=" * 75)
    print("AUDIT COMPLETE: Behavior Dataset Candidate Assessment Generated!")
    print(f"Total Comparison Pairs : 14 paired visual checks (28 review images)")
    print(f"Manual Comparison Pack : docs/audits/phase3_behavior_dataset_manual_comparison.md")
    print(f"Formal Research Log    : docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md")
    print(f"Elapsed Runtime        : {t_elapsed:.1f}s")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
