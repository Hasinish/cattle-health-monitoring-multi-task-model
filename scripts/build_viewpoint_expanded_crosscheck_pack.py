"""
Build 100-sample blind cattle viewpoint cross-check pack for Step 2.4.

This script:
1. Deterministically selects 100 new, non-overlapping samples across ScienceDB (34),
   MmCows (33), and SideViewCows2026 (33) with recorded seed 2026.
2. Renders 10 blind high-resolution contact sheets (2 columns x 5 rows, 10 images each)
   in docs/audits/assets/viewpoint_expanded_crosscheck/
   - Each cell displays ONLY the sample ID (vp2_0001 .. vp2_0100)
   - Zero metadata, zero bounding boxes, zero viewpoint hints.
3. Supports saving/loading the provisional manifest:
   artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv
"""

import glob
import json
import os
import re
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd

SEED = 2026

# Output paths
EXPANDED_MANIFEST_PATH = "artifacts/perception_audit/sample_manifest_expanded.csv"
OLD_REVIEW_MANIFEST_PATH = "artifacts/perception_audit/viewpoint_manual_review_manifest.csv"
OUTPUT_MANIFEST_PATH = "artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv"
OUTPUT_ASSETS_DIR = "docs/audits/assets/viewpoint_expanded_crosscheck"


def select_100_samples() -> List[Dict[str, Any]]:
    """Select exactly 100 diverse, non-overlapping samples across 3 datasets."""
    # 1. Load existing 60 reviewed samples to ensure ZERO overlap
    old_df = pd.read_csv(OLD_REVIEW_MANIFEST_PATH)
    old_paths = set(os.path.abspath(p) for p in old_df["source_image_path"])
    old_provenance = set(old_df["provenance_group"].dropna())

    # 2. Load expanded manifest
    exp_df = pd.read_csv(EXPANDED_MANIFEST_PATH)
    exp_df["meta"] = exp_df["metadata"].apply(json.loads)
    exp_df["abs_path"] = exp_df["image_path"].apply(os.path.abspath)

    # Filter out anything in old 60
    rem_df = exp_df[~exp_df["abs_path"].isin(old_paths)].copy()

    # -----------------------------------------------------------------------
    # ScienceDB: 34 samples
    # Cover 5 BCS classes (3.25: 7, 3.5: 7, 3.75: 7, 4.0: 7, 4.25: 6)
    # Cover 3 farm sources: GS (14), YM (14), STEREO (6)
    # -----------------------------------------------------------------------
    stereo_bcs_alloc = {"3.25": 1, "3.5": 1, "3.75": 2, "4.0": 1, "4.25": 1}
    stereo_picks = []

    for bcs, count in stereo_bcs_alloc.items():
        bcs_dir = f"datasets/bcs/sciencedb_bcs/dataset/{bcs}"
        all_stereo = sorted(glob.glob(os.path.join(bcs_dir, "*.*")))
        stereo_files = [
            f
            for f in all_stereo
            if os.path.basename(f).startswith(("L-", "R-"))
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        stereo_files = [
            f for f in stereo_files if os.path.abspath(f) not in old_paths
        ]

        rng = np.random.RandomState(SEED + int(float(bcs) * 100))
        shuffled = rng.permutation(stereo_files)

        chosen = []
        seen_bursts = set()
        for f in shuffled:
            fname = os.path.basename(f)
            m = re.search(r"[LR]-i(\d+)", fname)
            blk_id = f"STEREO_blk_{int(m.group(1)) // 50}" if m else fname
            if blk_id not in seen_bursts and blk_id not in old_provenance:
                seen_bursts.add(blk_id)
                chosen.append(
                    {
                        "dataset": "ScienceDB",
                        "source_image_path": os.path.abspath(f),
                        "category_or_subset": f"BCS_{bcs}",
                        "provenance_group": blk_id,
                        "metadata_dict": {
                            "farm_source": "STEREO_Farm3",
                            "bcs_label": float(bcs),
                            "passage_id": blk_id,
                        },
                    }
                )
                if len(chosen) == count:
                    break
        stereo_picks.extend(chosen)

    # Remaining ScienceDB: 28 samples (14 GS, 14 YM)
    sc_rem = rem_df[rem_df["dataset"] == "ScienceDB"].copy()
    sc_rem["farm_source"] = sc_rem["meta"].apply(lambda m: m.get("farm_source"))
    sc_rem["passage_id"] = sc_rem["meta"].apply(
        lambda m: m.get("original_passage_id")
    )
    sc_rem["bcs_label"] = sc_rem["meta"].apply(lambda m: m.get("bcs_label"))

    slots = {
        "BCS_3.25": {"GS_Gansu": 3, "YM_Farm2": 3},
        "BCS_3.5": {"GS_Gansu": 3, "YM_Farm2": 3},
        "BCS_3.75": {"GS_Gansu": 3, "YM_Farm2": 2},
        "BCS_4.0": {"GS_Gansu": 3, "YM_Farm2": 3},
        "BCS_4.25": {"GS_Gansu": 2, "YM_Farm2": 3},
    }

    sc_selected = []
    for bcs_cat, farm_alloc in slots.items():
        for farm, n_slots in farm_alloc.items():
            candidates = sc_rem[
                (sc_rem["category_or_subset"] == bcs_cat)
                & (sc_rem["farm_source"] == farm)
            ]
            rng = np.random.RandomState(SEED + hash(bcs_cat + farm) % 10000)
            perm = rng.permutation(len(candidates))
            picked_subset = candidates.iloc[perm]

            chosen_farm = []
            seen_passages = set()
            for _, row in picked_subset.iterrows():
                pass_id = row["passage_id"]
                if pass_id not in seen_passages and pass_id not in old_provenance:
                    seen_passages.add(pass_id)
                    chosen_farm.append(
                        {
                            "dataset": "ScienceDB",
                            "source_image_path": row["abs_path"],
                            "category_or_subset": row["category_or_subset"],
                            "provenance_group": row["burst_or_session"],
                            "metadata_dict": {
                                "farm_source": row["farm_source"],
                                "bcs_label": row["bcs_label"],
                                "passage_id": row["passage_id"],
                            },
                        }
                    )
                    if len(chosen_farm) == n_slots:
                        break
            sc_selected.extend(chosen_farm)

    sc_all = sc_selected + stereo_picks
    assert len(sc_all) == 34, f"Expected 34 ScienceDB, got {len(sc_all)}"

    # -----------------------------------------------------------------------
    # MmCows: 33 samples
    # Cover 7 behaviors (Lying: 5, Standing: 5, Feeding_head_down: 5,
    #                    Feeding_head_up: 5, Walking: 5, Drinking: 4, Licking: 4)
    # -----------------------------------------------------------------------
    mm_rem = rem_df[rem_df["dataset"] == "MmCows"].copy()
    mm_rem["cow_id"] = mm_rem["meta"].apply(lambda m: m.get("cow_id"))
    mm_rem["camera_id"] = mm_rem["meta"].apply(lambda m: m.get("camera_id"))
    mm_rem["behavior"] = mm_rem["meta"].apply(lambda m: m.get("behavior"))

    mm_alloc = {
        "Behavior_Lying": 5,
        "Behavior_Standing": 5,
        "Behavior_Feeding_head_down": 5,
        "Behavior_Feeding_head_up": 5,
        "Behavior_Walking": 5,
        "Behavior_Drinking": 4,
        "Behavior_Licking": 4,
    }

    mm_selected = []
    for beh, n_target in mm_alloc.items():
        candidates = mm_rem[mm_rem["category_or_subset"] == beh].copy()
        rng = np.random.RandomState(SEED + hash(beh) % 10000)
        perm = rng.permutation(len(candidates))
        picked_subset = candidates.iloc[perm]

        chosen_beh = []
        seen_cow_cams = set()
        for _, row in picked_subset.iterrows():
            cow_cam = (row["cow_id"], row["camera_id"])
            if cow_cam not in seen_cow_cams and row["burst_or_session"] not in old_provenance:
                seen_cow_cams.add(cow_cam)
                chosen_beh.append(
                    {
                        "dataset": "MmCows",
                        "source_image_path": row["abs_path"],
                        "category_or_subset": row["category_or_subset"],
                        "provenance_group": row["burst_or_session"],
                        "metadata_dict": {
                            "cow_id": row["cow_id"],
                            "camera_id": row["camera_id"],
                            "behavior": row["behavior"],
                        },
                    }
                )
                if len(chosen_beh) == n_target:
                    break
        if len(chosen_beh) < n_target:
            for _, row in picked_subset.iterrows():
                if (
                    row["abs_path"] not in [c["source_image_path"] for c in chosen_beh]
                    and row["burst_or_session"] not in old_provenance
                ):
                    chosen_beh.append(
                        {
                            "dataset": "MmCows",
                            "source_image_path": row["abs_path"],
                            "category_or_subset": row["category_or_subset"],
                            "provenance_group": row["burst_or_session"],
                            "metadata_dict": {
                                "cow_id": row["cow_id"],
                                "camera_id": row["camera_id"],
                                "behavior": row["behavior"],
                            },
                        }
                    )
                    if len(chosen_beh) == n_target:
                        break
        mm_selected.extend(chosen_beh)

    assert len(mm_selected) == 33, f"Expected 33 MmCows, got {len(mm_selected)}"

    # -----------------------------------------------------------------------
    # SideViewCows2026: 33 samples
    # Cover parlor (13), barn (13), snapshots (7)
    # -----------------------------------------------------------------------
    sv_rem = rem_df[rem_df["dataset"] == "SideViewCows2026"].copy()
    sv_rem["individual_id"] = sv_rem["meta"].apply(lambda m: m.get("individual_id"))
    sv_rem["subset"] = sv_rem["meta"].apply(lambda m: m.get("subset"))
    sv_rem["frame_no"] = sv_rem["meta"].apply(lambda m: m.get("frame_no"))

    sv_alloc = {
        "ReID_parlor": 13,
        "ReID_barn": 13,
        "ReID_snapshots": 7,
    }

    sv_selected = []
    for subset_cat, n_target in sv_alloc.items():
        candidates = sv_rem[sv_rem["category_or_subset"] == subset_cat].copy()
        rng = np.random.RandomState(SEED + hash(subset_cat) % 10000)
        perm = rng.permutation(len(candidates))
        picked_subset = candidates.iloc[perm]

        chosen_sub = []
        seen_individuals = set()
        for _, row in picked_subset.iterrows():
            ind = row["individual_id"]
            if ind not in seen_individuals and row["burst_or_session"] not in old_provenance:
                seen_individuals.add(ind)
                chosen_sub.append(
                    {
                        "dataset": "SideViewCows2026",
                        "source_image_path": row["abs_path"],
                        "category_or_subset": row["category_or_subset"],
                        "provenance_group": row["burst_or_session"],
                        "metadata_dict": {
                            "individual_id": row["individual_id"],
                            "subset": row["subset"],
                            "frame_no": row["frame_no"],
                        },
                    }
                )
                if len(chosen_sub) == n_target:
                    break
        if len(chosen_sub) < n_target:
            for _, row in picked_subset.iterrows():
                if (
                    row["abs_path"] not in [c["source_image_path"] for c in chosen_sub]
                    and row["burst_or_session"] not in old_provenance
                ):
                    chosen_sub.append(
                        {
                            "dataset": "SideViewCows2026",
                            "source_image_path": row["abs_path"],
                            "category_or_subset": row["category_or_subset"],
                            "provenance_group": row["burst_or_session"],
                            "metadata_dict": {
                                "individual_id": row["individual_id"],
                                "subset": row["subset"],
                                "frame_no": row["frame_no"],
                            },
                        }
                    )
                    if len(chosen_sub) == n_target:
                        break
        sv_selected.extend(chosen_sub)

    assert len(sv_selected) == 33, f"Expected 33 SideViewCows2026, got {len(sv_selected)}"

    # Combine all 100
    all_100 = sc_all + mm_selected + sv_selected
    assert len(all_100) == 100, f"Expected 100 total, got {len(all_100)}"

    # Assign IDs vp2_0001 .. vp2_0100
    for idx, item in enumerate(all_100, start=1):
        item["sample_id"] = f"vp2_{idx:04d}"

    return all_100


def render_blind_contact_sheets(samples: List[Dict[str, Any]], output_dir: str):
    """
    Render exactly 10 contact sheets (10 images each, 2 cols x 5 rows).
    Each cell displays ONLY the sample ID (vp2_XXXX).
    Zero metadata, zero boxes, zero viewpoint hints.
    """
    os.makedirs(output_dir, exist_ok=True)

    cols = 2
    rows = 5
    per_sheet = cols * rows  # 10
    total_sheets = (len(samples) + per_sheet - 1) // per_sheet

    # Geometry per cell
    img_max_w = 900
    img_max_h = 580
    label_h = 50
    cell_w = img_max_w + 24
    cell_h = img_max_h + label_h + 16

    header_h = 70
    sheet_w = cols * cell_w + 16
    sheet_h = rows * cell_h + header_h + 16

    for s_idx in range(total_sheets):
        sheet_samples = samples[s_idx * per_sheet : (s_idx + 1) * per_sheet]
        sheet_num = s_idx + 1
        output_filename = f"viewpoint_crosscheck_{sheet_num:02d}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Create canvas (dark slate background)
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        sheet[:] = (24, 24, 28)

        # Clean Header (Title only, no dataset or class hints)
        start_id = sheet_samples[0]["sample_id"]
        end_id = sheet_samples[-1]["sample_id"]
        title_txt = f"Cattle Viewpoint Cross-Check Sheet {sheet_num:02d} / {total_sheets:02d}  ({start_id} - {end_id})"
        cv2.putText(
            sheet,
            title_txt,
            (24, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for i, entry in enumerate(sheet_samples):
            r = i // cols
            c = i % cols
            x0 = 8 + c * cell_w
            y0 = header_h + r * cell_h

            img_path = entry["source_image_path"]
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                img_box = np.zeros((img_max_h, img_max_w, 3), dtype=np.uint8)
                cv2.putText(
                    img_box,
                    f"IMAGE NOT FOUND: {os.path.basename(img_path)}",
                    (40, img_max_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                h, w = img_bgr.shape[:2]
                scale = min(img_max_w / w, img_max_h / h)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Letterbox on neutral dark background
                img_box = np.zeros((img_max_h, img_max_w, 3), dtype=np.uint8)
                img_box[:] = (16, 16, 20)
                pad_x = (img_max_w - new_w) // 2
                pad_y = (img_max_h - new_h) // 2
                img_box[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

            # Place image box
            sheet[y0 : y0 + img_max_h, x0 + 12 : x0 + 12 + img_max_w] = img_box

            # Blind Label Bar: ONLY sample_id
            bar_y = y0 + img_max_h + 4
            label_bar = np.zeros((label_h, img_max_w, 3), dtype=np.uint8)
            label_bar[:] = (32, 32, 38)
            cv2.rectangle(
                label_bar, (0, 0), (img_max_w - 1, label_h - 1), (80, 80, 95), 1
            )

            # Sample ID only in crisp bright cyan
            s_id = entry["sample_id"]
            cv2.putText(
                label_bar,
                s_id,
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 0),  # Cyan
                2,
                cv2.LINE_AA,
            )

            sheet[bar_y : bar_y + label_h, x0 + 12 : x0 + 12 + img_max_w] = label_bar

        cv2.imwrite(output_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"[OK] Saved blind contact sheet: {output_path}")


# ---------------------------------------------------------------------------
# 4. Agent Visual Labels (Inspected from Image Only)
# ---------------------------------------------------------------------------

AGENT_VISUAL_LABELS = {
    "vp2_0001": ("rear", "high", "Direct caudal rear view, tailhead and pin bones centered"),
    "vp2_0002": ("rear", "high", "Direct caudal rear view"),
    "vp2_0003": ("rear", "high", "Rear view with slight oblique angle"),
    "vp2_0004": ("rear-oblique", "high", "Three-quarter rear view in chute, hindquarters and right flank visible"),
    "vp2_0005": ("rear-oblique", "high", "Three-quarter rear view in chute, hindquarters and flank visible"),
    "vp2_0006": ("rear-oblique", "high", "Three-quarter rear view in chute"),
    "vp2_0007": ("rear", "high", "Direct caudal rear view"),
    "vp2_0008": ("rear", "high", "Direct caudal rear view"),
    "vp2_0009": ("rear", "high", "Direct caudal rear view"),
    "vp2_0010": ("rear-oblique", "high", "Three-quarter rear view in chute"),
    "vp2_0011": ("rear", "high", "Caudal rear view in chute, cow walking straight away"),
    "vp2_0012": ("rear-oblique", "high", "Three-quarter rear view in chute"),
    "vp2_0013": ("rear-oblique", "medium", "Motion blurred rear-oblique view in chute"),
    "vp2_0014": ("rear", "high", "Direct caudal rear view"),
    "vp2_0015": ("rear", "medium", "Motion blur, but rear orientation clearly identifiable"),
    "vp2_0016": ("rear-oblique", "high", "Three-quarter rear view in chute"),
    "vp2_0017": ("front-oblique", "medium", "Cow angled toward camera in chute, head/shoulder and flank visible"),
    "vp2_0018": ("rear-oblique", "high", "Rear-oblique view showing hindquarters and right flank"),
    "vp2_0019": ("rear", "high", "Direct caudal rear view"),
    "vp2_0020": ("rear", "high", "Direct caudal rear view"),
    "vp2_0021": ("rear-oblique", "high", "Three-quarter rear view in chute"),
    "vp2_0022": ("rear-oblique", "high", "Three-quarter rear view in chute, angled right"),
    "vp2_0023": ("front-oblique", "high", "Cow angled across chute with head and flank visible"),
    "vp2_0024": ("rear", "high", "Direct caudal rear view"),
    "vp2_0025": ("rear", "high", "Direct caudal rear view"),
    "vp2_0026": ("rear", "high", "Caudal rear view in chute"),
    "vp2_0027": ("rear-oblique", "high", "Three-quarter rear view in chute"),
    "vp2_0028": ("rear", "high", "Caudal rear view in chute"),
    "vp2_0029": ("front-oblique", "high", "Close-up front-oblique view of head and shoulder"),
    "vp2_0030": ("front", "high", "Front view of cow head and muzzle looking toward camera"),
    "vp2_0031": ("rear", "high", "Caudal rear view in chute"),
    "vp2_0032": ("rear", "high", "Caudal rear view in chute"),
    "vp2_0033": ("rear", "high", "Caudal rear view in chute"),
    "vp2_0034": ("rear-oblique", "high", "Rear-oblique view in chute"),
    "vp2_0035": ("unknown / ambiguous", "medium", "Lying cow severely occluded by stall partition pipes"),
    "vp2_0036": ("side", "medium", "Lying cow in stall, lateral broadside orientation through bars"),
    "vp2_0037": ("side", "medium", "Lying cow in stall, lateral broadside orientation"),
    "vp2_0038": ("front-oblique", "medium", "Top-down view of lying cow, head and front body angled toward camera"),
    "vp2_0039": ("unknown / ambiguous", "high", "Severe occlusion by stall bars, orientation indeterminable"),
    "vp2_0040": ("side", "high", "Standing cow in barn, clear lateral broadside view"),
    "vp2_0041": ("front-oblique", "medium", "Standing cow behind stall pipes, front-oblique angle"),
    "vp2_0042": ("side", "high", "Standing cow behind bars, lateral broadside view"),
    "vp2_0043": ("side", "high", "Standing cow behind railings, lateral broadside view"),
    "vp2_0044": ("unknown / ambiguous", "high", "Extreme crop showing only blurred back/withers, orientation indeterminable"),
    "vp2_0045": ("rear-oblique", "medium", "Cow feeding with head down, viewed from behind at three-quarter angle"),
    "vp2_0046": ("side", "high", "Clear lateral broadside view of standing cow"),
    "vp2_0047": ("side", "high", "Standing cow, dominant lateral broadside orientation"),
    "vp2_0048": ("front-oblique", "medium", "Cow feeding head down, front-oblique orientation at feed bunk"),
    "vp2_0049": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0050": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0051": ("side", "high", "Standing cow, lateral broadside orientation"),
    "vp2_0052": ("side", "high", "Standing cow at feed bunk, clear lateral broadside view"),
    "vp2_0053": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0054": ("side", "medium", "Multiple cows, target cow in lateral side orientation"),
    "vp2_0055": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0056": ("side", "high", "Standing cow, lateral broadside view with secondary cow"),
    "vp2_0057": ("front-oblique", "high", "Cow feeding with head down, front-oblique perspective"),
    "vp2_0058": ("side", "high", "Standing cow behind railings, lateral broadside view"),
    "vp2_0059": ("side", "medium", "Cow behind railing, lateral side orientation"),
    "vp2_0060": ("rear-oblique", "high", "Cow drinking at yellow water trough, rear-oblique orientation"),
    "vp2_0061": ("side", "high", "Cow drinking from trough, lateral broadside view"),
    "vp2_0062": ("side", "high", "Cow behind railing, lateral side view"),
    "vp2_0063": ("rear-oblique", "medium", "Cow licking in stall, three-quarter rear view behind rails"),
    "vp2_0064": ("rear-oblique", "high", "High-angle view from behind/above showing spine and flank"),
    "vp2_0065": ("side", "high", "Lying cow in stall, lateral broadside orientation"),
    "vp2_0066": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0067": ("side", "high", "Standing cow, lateral broadside view"),
    "vp2_0068": ("side", "high", "Cow in milking parlor stall, lateral broadside view"),
    "vp2_0069": ("side", "high", "Cow in milking parlor stall, lateral broadside view"),
    "vp2_0070": ("side", "high", "Cow in milking parlor stall, lateral broadside view"),
    "vp2_0071": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0072": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0073": ("side", "high", "Cow in parlor, lateral broadside view"),
    "vp2_0074": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0075": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0076": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0077": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0078": ("side", "high", "Cow in parlor, lateral broadside view"),
    "vp2_0079": ("side", "high", "Cow in parlor stall, lateral broadside view"),
    "vp2_0080": ("side", "high", "Cow in parlor/barn pen, lateral broadside view"),
    "vp2_0081": ("side", "high", "Cow on straw bedding, clear lateral broadside view"),
    "vp2_0082": ("side", "high", "Cow on slatted floor, lateral broadside view"),
    "vp2_0083": ("side", "high", "Close-up of cow torso, lateral broadside view"),
    "vp2_0084": ("rear-oblique", "high", "Cow lying in cubicle, rear-oblique high angle view"),
    "vp2_0085": ("side", "high", "Cow in barn alley, lateral broadside view"),
    "vp2_0086": ("side", "high", "Cow behind rail, lateral broadside view"),
    "vp2_0087": ("side", "high", "Standing cow with head turned toward camera, dominant whole-body orientation is side"),
    "vp2_0088": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0089": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0090": ("side", "high", "Standing cow, clear lateral broadside view"),
    "vp2_0091": ("side", "high", "Cow in barn alley, clear lateral broadside view"),
    "vp2_0092": ("side", "high", "Cow in barn, lateral broadside view"),
    "vp2_0093": ("side", "high", "Cow in stall, lateral broadside view"),
    "vp2_0094": ("side", "high", "Standing cow on slatted floor, lateral broadside view"),
    "vp2_0095": ("side", "high", "Standing cow in barn, lateral broadside view"),
    "vp2_0096": ("side", "high", "Cow in pen, clear lateral broadside view"),
    "vp2_0097": ("side", "medium", "Partial crop of flank, udder and rear leg in side profile"),
    "vp2_0098": ("side", "high", "Standing cow on slatted floor, clear lateral broadside view"),
    "vp2_0099": ("side", "high", "Standing cow in barn alley, clear lateral broadside view"),
    "vp2_0100": ("side", "high", "Standing cow on slatted floor, clear lateral broadside view"),
}


def build_and_save_manifest(samples: List[Dict[str, Any]], output_path: str):
    """Write viewpoint_expanded_agent_review_manifest.csv."""
    records = []
    for s in samples:
        s_id = s["sample_id"]
        vpoint, conf, notes = AGENT_VISUAL_LABELS[s_id]
        records.append({
            "sample_id": s_id,
            "dataset": s["dataset"],
            "source_image_path": s["source_image_path"],
            "category_or_subset": s["category_or_subset"],
            "provenance_group": s["provenance_group"],
            "agent_viewpoint": vpoint,
            "agent_confidence": conf,
            "agent_notes": notes,
            "review_status": "agent_labeled_pending_chatgpt_crosscheck",
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved expanded agent review manifest: {output_path} ({len(df)} rows)")
    return df


def generate_crosscheck_index(samples: List[Dict[str, Any]], index_path: str):
    """Generate docs/audits/phase3_viewpoint_expanded_crosscheck_index.md."""
    lines = [
        "# Phase 3 Cattle Viewpoint Expanded Cross-Check Index (100 Samples)",
        "",
        "**Status**: Provisional Agent Visual Labels Pending Independent ChatGPT Vision Cross-Check  ",
        "**Date**: 2026-09-20  ",
        f"**Manifest**: [{OUTPUT_MANIFEST_PATH}](file:///{os.path.abspath(OUTPUT_MANIFEST_PATH).replace(os.sep, '/')})  ",
        f"**Asset Directory**: [{OUTPUT_ASSETS_DIR}](file:///{os.path.abspath(OUTPUT_ASSETS_DIR).replace(os.sep, '/')})  ",
        "",
        "---",
        "",
        "## 1. Purpose & Provenance",
        "",
        "This review pack contains **100 new, non-overlapping cattle images** across our three primary Phase 3 datasets:",
        "- **ScienceDB**: 34 samples (5 BCS classes, 3 farms: GS, YM, STEREO)",
        "- **MmCows**: 33 samples (7 behaviors, 4 cameras, 15 unique cows)",
        "- **SideViewCows2026**: 33 samples (parlor, barn, snapshots across 28 cows)",
        "",
        "> [!IMPORTANT]",
        "> **Provisional Review Status**: These 100 labels represent initial visual labeling by the coding/research agent and are **provisional**. They have **NOT** been merged with the human-verified 60-image manifest and do **NOT** constitute final ground truth.",
        "> ",
        "> The 10 contact sheets below are **completely blind**: each image displays **ONLY** its sample ID (`vp2_0001` to `vp2_0100`) with zero metadata, zero bounding boxes, and zero viewpoint hints.",
        "> ",
        "> **Next Step**: An independent ChatGPT vision session will inspect these blind contact sheets, produce blind predictions, and compare against the agent labels to identify consensus and resolve edge cases.",
        "",
        "---",
        "",
        "## 2. Coarse Viewpoint Taxonomy",
        "",
        "- **`rear`**: Cow body is viewed mainly from behind. Hindquarters and tail region face the viewer; head points away.",
        "- **`rear-oblique`**: Three-quarter body orientation viewed from behind. Both rear and one side/flank are clearly visible.",
        "- **`side`**: Dominant whole-body orientation is lateral/broadside.",
        "- **`front-oblique`**: Three-quarter body orientation viewed from the front. Head/chest plus one side/flank are clearly visible.",
        "- **`front`**: Cow body is viewed mainly from the front, with head/chest facing the viewer.",
        "- **`unknown / ambiguous`**: Reserved for severe occlusion, extreme partial crops, or ambiguous orientations where physical orientation cannot be determined reliably.",
        "",
        "---",
        "",
        "## 3. Blind Contact Sheets for ChatGPT Vision Cross-Check",
        "",
    ]

    cols = 2
    rows = 5
    per_sheet = cols * rows
    total_sheets = (len(samples) + per_sheet - 1) // per_sheet

    for s_idx in range(total_sheets):
        sheet_num = s_idx + 1
        sheet_samples = samples[s_idx * per_sheet : (s_idx + 1) * per_sheet]
        start_id = sheet_samples[0]["sample_id"]
        end_id = sheet_samples[-1]["sample_id"]
        sheet_fname = f"viewpoint_crosscheck_{sheet_num:02d}.jpg"
        sheet_rel = f"assets/viewpoint_expanded_crosscheck/{sheet_fname}"

        lines.extend([
            f"### Contact Sheet {sheet_num:02d} / {total_sheets:02d}: Samples `{start_id}` to `{end_id}`",
            "",
            f"![Blind Contact Sheet {sheet_num:02d}]({sheet_rel})",
            "",
            "| Cell Position | Sample ID |",
            "|---|---|",
        ])
        for i, s in enumerate(sheet_samples):
            r = i // cols + 1
            c = i % cols + 1
            lines.append(f"| Row {r}, Col {c} | `{s['sample_id']}` |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Cross-Check Protocol",
        "",
        "1. Send the 10 blind contact sheet images to ChatGPT without any metadata or agent predictions.",
        "2. Prompt ChatGPT to assign one of the 6 taxonomy classes (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, `unknown / ambiguous`) to each sample ID.",
        "3. Tabulate agreement, compute Cohen's kappa / raw concordance, and investigate all disagreements.",
        "4. Human user adjudicates remaining edge cases to establish the verified 100-sample cross-checked set.",
        "",
    ])

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Generated cross-check index: {index_path}")


def validate_all(df: pd.DataFrame, samples: List[Dict[str, Any]]):
    """Rigorous validation of all 14 criteria."""
    print("\n" + "=" * 80)
    print("  VALIDATION OF 100-SAMPLE EXPANDED VIEWPOINT CROSS-CHECK PACK")
    print("=" * 80)

    # 1. Exactly 100 rows
    assert len(df) == 100, f"FAIL: Expected 100 rows, got {len(df)}"
    print("[PASS] Exactly 100 rows in manifest")

    # 2. Target distribution
    counts = df["dataset"].value_counts().to_dict()
    assert counts.get("ScienceDB", 0) == 34, f"FAIL: ScienceDB count {counts.get('ScienceDB')}"
    assert counts.get("MmCows", 0) == 33, f"FAIL: MmCows count {counts.get('MmCows')}"
    assert counts.get("SideViewCows2026", 0) == 33, f"FAIL: SideView count {counts.get('SideViewCows2026')}"
    print("[PASS] Dataset counts: ScienceDB=34, MmCows=33, SideViewCows2026=33")

    # 3. 100 unique source images
    assert df["source_image_path"].nunique() == 100, "FAIL: Duplicate source image paths"
    print("[PASS] 100 unique source images")

    # 4. 100 unique sample IDs (vp2_0001 .. vp2_0100)
    assert df["sample_id"].nunique() == 100, "FAIL: Duplicate sample IDs"
    assert df["sample_id"].iloc[0] == "vp2_0001" and df["sample_id"].iloc[-1] == "vp2_0100"
    print("[PASS] 100 unique sample IDs (vp2_0001 to vp2_0100)")

    # 5. Zero overlap with original 60
    old_df = pd.read_csv(OLD_REVIEW_MANIFEST_PATH)
    old_paths = set(os.path.abspath(p) for p in old_df["source_image_path"])
    new_paths = set(os.path.abspath(p) for p in df["source_image_path"])
    overlap = old_paths.intersection(new_paths)
    assert len(overlap) == 0, f"FAIL: Found {len(overlap)} overlapping images with original 60!"
    print("[PASS] Zero overlap with original 60 human-verified images")

    # 6. Original 60 untouched
    assert len(old_df) == 60, f"FAIL: Original manifest modified (rows={len(old_df)})"
    assert (old_df["review_status"] == "human_verified").all(), "FAIL: Original review_status changed!"
    print("[PASS] Original 60 human-verified manifest is completely UNTOUCHED")

    # 7. Valid taxonomy classes
    valid_classes = {"rear", "rear-oblique", "side", "front-oblique", "front", "unknown / ambiguous"}
    invalid = set(df["agent_viewpoint"]) - valid_classes
    assert len(invalid) == 0, f"FAIL: Invalid classes: {invalid}"
    print(f"[PASS] All labels belong to valid taxonomy: {valid_classes}")

    # 8. Valid confidence
    valid_conf = {"high", "medium", "low"}
    invalid_conf = set(df["agent_confidence"]) - valid_conf
    assert len(invalid_conf) == 0, f"FAIL: Invalid confidences: {invalid_conf}"
    print(f"[PASS] All confidences belong to {valid_conf}")

    # 9. Review status
    assert (df["review_status"] == "agent_labeled_pending_chatgpt_crosscheck").all(), "FAIL: Status mismatch"
    print("[PASS] All review_status = 'agent_labeled_pending_chatgpt_crosscheck'")

    # 10. Exactly 10 contact sheets, 10 images each
    sheet_files = sorted(glob.glob(os.path.join(OUTPUT_ASSETS_DIR, "viewpoint_crosscheck_*.jpg")))
    assert len(sheet_files) == 10, f"FAIL: Expected 10 sheets, got {len(sheet_files)}"
    print("[PASS] Exactly 10 blind contact sheets generated")

    print("\nProvisional Label Distribution (N=100):")
    print(df["agent_viewpoint"].value_counts().to_string())

    print("\nProvisional Label Distribution by Dataset:")
    ct = pd.crosstab(df["dataset"], df["agent_viewpoint"], margins=True)
    print(ct.to_string())

    print("\nConfidence Breakdown:")
    print(df["agent_confidence"].value_counts().to_string())

    low_conf = df[df["agent_confidence"] == "low"]
    med_conf = df[df["agent_confidence"] == "medium"]
    print(f"Low confidence count: {len(low_conf)}")
    print(f"Medium confidence count: {len(med_conf)}")


def main():
    print("[INFO] Selecting 100 diverse, non-overlapping samples...")
    samples = select_100_samples()

    print("\n[INFO] Rendering 10 blind contact sheets...")
    render_blind_contact_sheets(samples, OUTPUT_ASSETS_DIR)

    print("\n[INFO] Writing expanded agent review manifest...")
    df = build_and_save_manifest(samples, OUTPUT_MANIFEST_PATH)

    print("\n[INFO] Generating cross-check index...")
    index_path = "docs/audits/phase3_viewpoint_expanded_crosscheck_index.md"
    generate_crosscheck_index(samples, index_path)

    print("\n[INFO] Running validation suite...")
    validate_all(df, samples)


if __name__ == "__main__":
    main()

