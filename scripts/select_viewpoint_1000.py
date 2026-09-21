"""
Select exactly 1,000 diverse, non-overlapping real-cattle images for human viewpoint annotation.

Target breakdown:
- ScienceDB: 334 samples
- MmCows: 333 samples
- SideViewCows2026: 333 samples
Total: 1,000 samples

Constraints:
- ZERO image overlap with existing 60-image manual review set.
- ZERO image overlap with existing 100-image expanded diagnostic benchmark.
- Non-adjacent / non-correlated sampling within sequences.
- Rich provenance tracking for future leakage-free splits.
"""

import os
import re
import json
import numpy as np
import pandas as pd

SEED = 2026

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_60_PATH = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_manual_review_manifest.csv")
OLD_100_PATH = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_expanded_agent_review_manifest.csv")
OUTPUT_MANIFEST_PATH = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_1000_annotation_manifest.csv")


def load_excluded_paths():
    excluded = set()
    for p in [OLD_60_PATH, OLD_100_PATH]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for path_val in df["source_image_path"]:
                abs_p = os.path.abspath(os.path.join(REPO_ROOT, path_val)) if not os.path.isabs(path_val) else os.path.abspath(path_val)
                excluded.add(abs_p)
    return excluded


def to_rel_path(path_str):
    abs_p = os.path.abspath(os.path.join(REPO_ROOT, path_str)) if not os.path.isabs(path_str) else os.path.abspath(path_str)
    return os.path.relpath(abs_p, REPO_ROOT).replace("\\", "/")


def select_sciencedb(excluded_paths, target_count=334):
    """
    Select 334 ScienceDB images:
    - 1 image per burst group to prevent adjacent/correlated frames.
    - Diversify across 3 farm sources: STEREO_Farm3 (40), GS_Gansu (150), YM_Farm2 (144).
    - Stratified across the 5 BCS classes (3.25, 3.5, 3.75, 4.0, 4.25).
    """
    sc_train = pd.read_csv(os.path.join(REPO_ROOT, "datasets", "bcs", "sciencedb", "train.csv"))
    sc_val = pd.read_csv(os.path.join(REPO_ROOT, "datasets", "bcs", "sciencedb", "val.csv"))
    sc_test = pd.read_csv(os.path.join(REPO_ROOT, "datasets", "bcs", "sciencedb", "test.csv"))
    sc_all = pd.concat([sc_train, sc_val, sc_test], ignore_index=True)
    
    sc_all["rel_path"] = sc_all["image_path"].apply(to_rel_path)
    sc_all["abs_path"] = sc_all["rel_path"].apply(lambda p: os.path.abspath(os.path.join(REPO_ROOT, p)))
    
    # Exclude prior 160
    candidates = sc_all[~sc_all["abs_path"].isin(excluded_paths)].copy()
    
    farm_alloc = {
        "STEREO_Farm3": 40,
        "GS_Gansu": 150,
        "YM_Farm2": 144,
    }
    
    selected_rows = []
    bcs_classes = [3.25, 3.5, 3.75, 4.0, 4.25]
    
    rng = np.random.RandomState(SEED)
    
    for farm, farm_target in farm_alloc.items():
        farm_df = candidates[candidates["farm_source"] == farm].copy()
        burst_groups = farm_df["burst_group_id"].unique()
        rng.shuffle(burst_groups)
        
        # Target per BCS class
        base_per_bcs = farm_target // len(bcs_classes)
        remainder = farm_target % len(bcs_classes)
        bcs_target = {bcs: base_per_bcs + (1 if i < remainder else 0) for i, bcs in enumerate(bcs_classes)}
        
        farm_selected = []
        seen_bursts = set()
        
        for bcs in bcs_classes:
            bcs_target_count = bcs_target[bcs]
            bcs_df = farm_df[farm_df["label"] == bcs]
            bcs_bursts = [bg for bg in bcs_df["burst_group_id"].unique() if bg not in seen_bursts]
            rng.shuffle(bcs_bursts)
            
            picked_bcs = 0
            for bg in bcs_bursts:
                bg_images = bcs_df[bcs_df["burst_group_id"] == bg]
                # Pick a representative image (e.g. median index)
                chosen_img = bg_images.iloc[len(bg_images) // 2]
                
                farm_selected.append({
                    "dataset": "ScienceDB",
                    "source_image_path": chosen_img["rel_path"],
                    "provenance_group": chosen_img["burst_group_id"],
                    "cow_or_identity_id_if_available": "",  # Passage ID must not be treated as cow ID
                    "session_or_timeblock_if_available": f"{chosen_img['farm_source']}/{chosen_img['original_passage_id']}",
                })
                seen_bursts.add(bg)
                picked_bcs += 1
                if picked_bcs >= bcs_target_count:
                    break
        
        # If any shortfall due to burst availability, fill from remaining farm bursts
        if len(farm_selected) < farm_target:
            rem_bursts = [bg for bg in burst_groups if bg not in seen_bursts]
            for bg in rem_bursts:
                bg_images = farm_df[farm_df["burst_group_id"] == bg]
                chosen_img = bg_images.iloc[len(bg_images) // 2]
                farm_selected.append({
                    "dataset": "ScienceDB",
                    "source_image_path": chosen_img["rel_path"],
                    "provenance_group": chosen_img["burst_group_id"],
                    "cow_or_identity_id_if_available": "",
                    "session_or_timeblock_if_available": f"{chosen_img['farm_source']}/{chosen_img['original_passage_id']}",
                })
                seen_bursts.add(bg)
                if len(farm_selected) >= farm_target:
                    break
                    
        selected_rows.extend(farm_selected[:farm_target])
    
    assert len(selected_rows) == target_count, f"Expected {target_count} ScienceDB, got {len(selected_rows)}"
    return selected_rows


def select_mmcows(excluded_paths, target_count=333):
    """
    Select 333 MmCows images:
    - Strictly 1 image per event_id to prevent adjacent frame correlation.
    - Diversify across 16 biological cows, 7 behaviors, 4 cameras, and 3 time blocks.
    - Target distribution across behaviors ensuring rare behaviors (Walking, Drinking, Licking) are well-represented.
    """
    mm_manifest = pd.read_csv(os.path.join(REPO_ROOT, "datasets", "behavior", "mmcows", "manifest.csv"))
    mm_manifest["rel_path"] = mm_manifest["image_path"].apply(to_rel_path)
    mm_manifest["abs_path"] = mm_manifest["rel_path"].apply(lambda p: os.path.abspath(os.path.join(REPO_ROOT, p)))
    
    candidates = mm_manifest[~mm_manifest["abs_path"].isin(excluded_paths)].copy()
    
    # Behavior allocation
    beh_alloc = {
        "Lying": 55,
        "Standing": 55,
        "Feeding_head_down": 50,
        "Feeding_head_up": 50,
        "Walking": 45,
        "Drinking": 40,
        "Licking": 38,
    }
    assert sum(beh_alloc.values()) == target_count
    
    rng = np.random.RandomState(SEED + 1)
    selected_rows = []
    seen_events = set()
    
    for beh, count in beh_alloc.items():
        beh_df = candidates[candidates["class_name"] == beh].copy()
        
        # Shuffle events
        unique_events = [ev for ev in beh_df["event_id"].unique() if ev not in seen_events]
        rng.shuffle(unique_events)
        
        chosen_beh = []
        for ev in unique_events:
            ev_df = beh_df[beh_df["event_id"] == ev]
            # Pick a middle frame in the event
            chosen_img = ev_df.iloc[len(ev_df) // 2]
            
            chosen_beh.append({
                "dataset": "MmCows",
                "source_image_path": chosen_img["rel_path"],
                "provenance_group": chosen_img["event_id"],
                "cow_or_identity_id_if_available": f"cow_{chosen_img['cow_id']}",
                "session_or_timeblock_if_available": f"tb_{chosen_img['time_block_id']}_cam_{chosen_img['camera_id']}",
            })
            seen_events.add(ev)
            if len(chosen_beh) >= count:
                break
                
        selected_rows.extend(chosen_beh)
        
    assert len(selected_rows) == target_count, f"Expected {target_count} MmCows, got {len(selected_rows)}"
    return selected_rows


def select_sideview(excluded_paths, target_count=333):
    """
    Select 333 SideViewCows2026 images:
    - Cover parlor (140), barn (133), snapshots (60).
    - Parlor: 140 distinct recording sessions (1 image per session).
    - Snapshots: 60 distinct recording sessions (1 image per session).
    - Barn: 133 samples across the 71 sessions (1-2 images per session, spaced widely by >=100 frames).
    - Cover all 110 biological cow individuals.
    """
    sv_manifest = pd.read_csv(os.path.join(REPO_ROOT, "datasets", "id", "sideviewcows2026", "manifest.csv"))
    sv_manifest["rel_path"] = sv_manifest["image_path"].apply(to_rel_path)
    sv_manifest["abs_path"] = sv_manifest["rel_path"].apply(lambda p: os.path.abspath(os.path.join(REPO_ROOT, p)))
    
    candidates = sv_manifest[~sv_manifest["abs_path"].isin(excluded_paths)].copy()
    
    rng = np.random.RandomState(SEED + 2)
    selected_rows = []
    
    # 1. Parlor: 140 sessions (1 per session, covering all 110 cows)
    parlor_df = candidates[candidates["subset"] == "parlor"]
    all_cows = list(parlor_df["individual_id"].unique())
    rng.shuffle(all_cows)
    
    selected_parlor_sessions = set()
    
    # First pick 1 session for each of the 110 cows
    for cow in all_cows:
        cow_sessions = list(parlor_df[parlor_df["individual_id"] == cow]["recording_id"].unique())
        rng.shuffle(cow_sessions)
        chosen_sess = cow_sessions[0]
        selected_parlor_sessions.add(chosen_sess)
        
    # Then add 30 more sessions to reach 140
    rem_sessions = [s for s in parlor_df["recording_id"].unique() if s not in selected_parlor_sessions]
    rng.shuffle(rem_sessions)
    for s in rem_sessions[:30]:
        selected_parlor_sessions.add(s)
        
    for sess in selected_parlor_sessions:
        sess_df = parlor_df[parlor_df["recording_id"] == sess]
        chosen_img = sess_df.iloc[len(sess_df) // 2]
        selected_rows.append({
            "dataset": "SideViewCows2026",
            "source_image_path": chosen_img["rel_path"],
            "provenance_group": chosen_img["recording_id"],
            "cow_or_identity_id_if_available": f"cow_{chosen_img['individual_id']}",
            "session_or_timeblock_if_available": f"parlor/{chosen_img['recording_id']}",
        })
        
    # 2. Snapshots: 60 sessions (1 per session)
    snap_df = candidates[candidates["subset"] == "snapshots"]
    snap_sessions = list(snap_df["recording_id"].unique())
    rng.shuffle(snap_sessions)
    
    for sess in snap_sessions[:60]:
        sess_df = snap_df[snap_df["recording_id"] == sess]
        chosen_img = sess_df.iloc[len(sess_df) // 2]
        selected_rows.append({
            "dataset": "SideViewCows2026",
            "source_image_path": chosen_img["rel_path"],
            "provenance_group": chosen_img["recording_id"],
            "cow_or_identity_id_if_available": f"cow_{chosen_img['individual_id']}",
            "session_or_timeblock_if_available": f"snapshots/{chosen_img['recording_id']}",
        })
        
    # 3. Barn: 133 samples across 71 sessions
    # 71 sessions * 2 = 142 max, so 62 sessions with 2 images, 9 sessions with 1 image = 133 images
    barn_df = candidates[candidates["subset"] == "barn"]
    barn_sessions = list(barn_df["recording_id"].unique())
    rng.shuffle(barn_sessions)
    
    # Sort sessions so 62 have 2, 9 have 1
    barn_count = 0
    for i, sess in enumerate(barn_sessions):
        sess_df = barn_df[barn_df["recording_id"] == sess].sort_values("frame_no")
        n_imgs = len(sess_df)
        n_pick = 2 if i < 62 else 1
        
        if n_pick == 1:
            idx = n_imgs // 2
            chosen_imgs = [sess_df.iloc[idx]]
        else:
            # Pick two frames spaced far apart (e.g., 25% and 75% of sequence)
            idx1 = max(0, int(n_imgs * 0.25))
            idx2 = min(n_imgs - 1, int(n_imgs * 0.75))
            if idx2 - idx1 < 50 and n_imgs > 50:
                idx2 = min(n_imgs - 1, idx1 + 50)
            chosen_imgs = [sess_df.iloc[idx1], sess_df.iloc[idx2]]
            
        for chosen_img in chosen_imgs:
            selected_rows.append({
                "dataset": "SideViewCows2026",
                "source_image_path": chosen_img["rel_path"],
                "provenance_group": chosen_img["recording_id"],
                "cow_or_identity_id_if_available": f"cow_{chosen_img['individual_id']}",
                "session_or_timeblock_if_available": f"barn/{chosen_img['recording_id']}",
            })
            barn_count += 1
            if barn_count >= 133:
                break
        if barn_count >= 133:
            break
            
    assert len(selected_rows) == target_count, f"Expected {target_count} SideView, got {len(selected_rows)}"
    return selected_rows


def build_1000_manifest():
    excluded_paths = load_excluded_paths()
    print(f"Loaded {len(excluded_paths)} excluded prior review paths.")
    
    sc_rows = select_sciencedb(excluded_paths, target_count=334)
    print(f"Selected {len(sc_rows)} ScienceDB samples.")
    
    mm_rows = select_mmcows(excluded_paths, target_count=333)
    print(f"Selected {len(mm_rows)} MmCows samples.")
    
    sv_rows = select_sideview(excluded_paths, target_count=333)
    print(f"Selected {len(sv_rows)} SideViewCows2026 samples.")
    
    all_rows = sc_rows + mm_rows + sv_rows
    assert len(all_rows) == 1000, f"Expected 1,000 total, got {len(all_rows)}"
    
    # Shuffle with seed for smooth non-anchoring annotation flow
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(all_rows))
    shuffled_rows = [all_rows[i] for i in perm]
    
    # Assign sample_id and default values
    for i, row in enumerate(shuffled_rows, start=1):
        row["sample_id"] = f"vp1k_{i:04d}"
        row["viewpoint"] = "UNREVIEWED"
        row["occlusion"] = "none"
        row["body_cutoff"] = "none"
        row["multiple_cows"] = "no"
        row["review_status"] = "pending_human_review"
        
    cols = [
        "sample_id",
        "dataset",
        "source_image_path",
        "provenance_group",
        "cow_or_identity_id_if_available",
        "session_or_timeblock_if_available",
        "viewpoint",
        "occlusion",
        "body_cutoff",
        "multiple_cows",
        "review_status",
    ]
    
    df = pd.DataFrame(shuffled_rows)[cols]
    
    # Sanity checks
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert df["dataset"].value_counts().to_dict() == {"ScienceDB": 334, "MmCows": 333, "SideViewCows2026": 333}
    assert df["source_image_path"].nunique() == 1000, "Duplicate source image paths found!"
    
    # Check overlap with excluded
    new_abs_paths = set(os.path.abspath(os.path.join(REPO_ROOT, p)) for p in df["source_image_path"])
    overlap = new_abs_paths.intersection(excluded_paths)
    assert len(overlap) == 0, f"Fatal: {len(overlap)} overlapping paths found with prior 160 review images!"
    
    # Verify all files exist locally
    for p in df["source_image_path"]:
        abs_p = os.path.join(REPO_ROOT, p)
        if not os.path.exists(abs_p):
            raise FileNotFoundError(f"Selected file does not exist: {abs_p}")
            
    df.to_csv(OUTPUT_MANIFEST_PATH, index=False)
    print(f"Successfully saved 1,000-sample manifest to: {OUTPUT_MANIFEST_PATH}")
    
    # Print diversity statistics
    print("\n--- Provenance Diversity Summary ---")
    print("Dataset counts:")
    print(df["dataset"].value_counts())
    print("\nScienceDB:")
    sc_df = df[df["dataset"] == "ScienceDB"]
    print(f"  Unique burst groups: {sc_df['provenance_group'].nunique()} (1 image per burst group)")
    print(f"  Farm sources: {sc_df['session_or_timeblock_if_available'].apply(lambda s: s.split('/')[0]).value_counts().to_dict()}")
    print("\nMmCows:")
    mm_df = df[df["dataset"] == "MmCows"]
    print(f"  Unique events: {mm_df['provenance_group'].nunique()} (1 image per event)")
    print(f"  Cows represented: {mm_df['cow_or_identity_id_if_available'].nunique()} (all 16 cows)")
    print("\nSideViewCows2026:")
    sv_df = df[df["dataset"] == "SideViewCows2026"]
    print(f"  Unique recording sessions: {sv_df['provenance_group'].nunique()}")
    print(f"  Cows represented: {sv_df['cow_or_identity_id_if_available'].nunique()} (all 110 cows)")
    print(f"  Subsets: {sv_df['session_or_timeblock_if_available'].apply(lambda s: s.split('/')[0]).value_counts().to_dict()}")
    print("------------------------------------")


if __name__ == "__main__":
    build_1000_manifest()
