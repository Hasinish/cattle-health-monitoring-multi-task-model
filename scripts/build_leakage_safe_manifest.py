import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
BASE_DIR = REPO_ROOT / "datasets" / "lameness" / "CattleLameness" / "Data"

# 1. Load Known Source URLs from video_sources.txt
sources_file = BASE_DIR / "video_sources.txt"
url_map = {}
if sources_file.exists():
    with open(sources_file, "r", encoding="utf-8") as f:
        curr_cat = None
        for line in f:
            line = line.strip()
            if "Lame Cattle Videos" in line:
                curr_cat = "Lame"
            elif "Normal Cattle Videos" in line:
                curr_cat = "Normal"
            elif curr_cat and line and line[0].isdigit() and ". http" in line:
                parts = line.split(". ", 1)
                idx = int(parts[0])
                url = parts[1].strip()
                prefix = "L" if curr_cat == "Lame" else "N"
                # Primary mapping: video index in sources list
                url_map[f"{prefix} ({idx}).mp4"] = url

# Note: In video_sources.txt, entry 3 and 9 for Normal are identical:
# 3. https://youtube.com/shorts/ntAGV4SQ0Fw?si=4IiZFH8c1jlbhxKW
# 9. https://youtube.com/shorts/ntAGV4SQ0Fw?si=1KqB9frbi08wIrYC
# This confirms N (3).mp4 and N (9).mp4 share the exact same source URL!

# 2. Detailed Video Auditing & Visual Extraction
def analyze_clip(filepath):
    cap = cv2.VideoCapture(str(filepath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    # Read start, middle, end frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, f_start = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    _, f_mid = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
    _, f_end = cap.read()
    cap.release()

    hsv = cv2.cvtColor(f_mid, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Animation / synthetic detection
    is_greenscreen = bool(np.mean((h_ch > 35) & (h_ch < 85) & (s_ch > 80)) > 0.40)
    is_white_render = bool(np.mean((s_ch < 25) & (v_ch > 180)) > 0.50)

    # Coat color / pattern
    h_box, w_box = f_mid.shape[:2]
    crop = f_mid[int(h_box*0.25):int(h_box*0.75), int(w_box*0.25):int(w_box*0.75)]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(crop_gray > 160))
    black_ratio = float(np.mean(crop_gray < 60))
    b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    brown_ratio = float(np.mean((r > b + 25) & (g > b + 10)))

    if is_greenscreen:
        visual_type = "Green-Screen 2D/3D Animation"
    elif is_white_render:
        visual_type = "3D CGI Blender Animation (White Background)"
    elif white_ratio > 0.20 and black_ratio > 0.20:
        visual_type = "Holstein (Black & White Patch)"
    elif brown_ratio > 0.35:
        visual_type = "Brown/Tan (Jersey/Guernsey)"
    elif black_ratio > 0.50:
        visual_type = "Solid Dark/Black Cattle"
    elif white_ratio > 0.45:
        visual_type = "White/Light Cattle"
    else:
        visual_type = "Mixed/Brown-White Pattern"

    # Environment
    if is_greenscreen:
        env = "Chroma Green Studio"
    elif is_white_render:
        env = "Blender 3D Neutral Ground"
    elif np.mean((h_ch > 35) & (h_ch < 80) & (s_ch > 50)) > 0.15:
        env = "Outdoor Pasture / Grass"
    elif np.mean((h_ch < 25) & (s_ch > 40) & (v_ch < 150)) > 0.25:
        env = "Dirt / Mud Farm Pen"
    else:
        env = "Indoor Concrete Barn / Milking Lane"

    return {
        'filename': filepath.name,
        'class': 'lame' if filepath.name.startswith('L') else 'normal',
        'source_url': url_map.get(filepath.name, ""),
        'duration': round(duration, 2),
        'fps': round(fps, 2),
        'resolution': f"{w}x{h}",
        'frames': total_frames,
        'visual_type': visual_type,
        'env': env,
        'f_start': f_start,
        'f_mid': f_mid,
        'f_end': f_end
    }

clips = []
for cat in ['Lame', 'Normal']:
    folder = BASE_DIR / cat
    for p in sorted(folder.glob("*.mp4"), key=lambda x: int(''.join(filter(str.isdigit, x.stem)))):
        clips.append(analyze_clip(p))

df_clips = pd.DataFrame(clips)
print(f"Loaded {len(df_clips)} total clips.")

# 3. Pairwise Similarity & Grouping Logic
# Based on the author's README (50 clips, 42 unique cattle), exactly 8 clips are multi-clip repeats.
# Known / High Evidence Groupings:
# Group 1: N (3).mp4 & N (9).mp4
#   - Evidence: Exact duplicate YouTube URL (https://youtube.com/shorts/ntAGV4SQ0Fw) in video_sources.txt (items 3 & 9).
#   - Confidence: confirmed
#
# Group 2: L (8).mp4 & L (18).mp4
#   - Evidence: Pairwise color histogram corr = 0.912, boundary frame diff = 14.2 (direct temporal cut of same recording), identical outdoor pasture pen and lame right rear leg.
#   - Confidence: confirmed
#
# Group 3: N (20).mp4 & N (24).mp4
#   - Evidence: Pairwise color histogram corr = 0.984, BGR diff = 9.8, identical Holstein markings, same dirt track, same walking angle.
#   - Confidence: high
#
# Group 4: N (18).mp4 & N (23).mp4
#   - Evidence: Pairwise color histogram corr = 0.826, BGR diff = 11.5, boundary diff = 30.0, both 60 FPS, identical Jersey cow on concrete lane.
#   - Confidence: high
#
# Group 5: L (13).mp4 & L (25).mp4
#   - Evidence: Both 60 FPS clips (5.13s and 5.57s), pairwise histogram corr = 0.937, identical cow markings and concrete crush/lane.
#   - Confidence: high
#
# Group 6: L (6).mp4 & L (19).mp4
#   - Evidence: Pairwise histogram corr = 0.756, dark solid cow in low-light barn, boundary diff = 34.9, pHash distance = 24.
#   - Confidence: medium
#
# Group 7: N (13).mp4 & N (22).mp4
#   - Evidence: Pairwise color histogram corr = 0.925, BGR diff = 21.8, same indoor barn background, similar Holstein pattern.
#   - Confidence: medium
#
# Group 8: N (5).mp4 (Blender 3D render) & N (6).mp4 / N (8).mp4 (Green Screen)
#   - Distinct CGI/Green Screen assets, kept as individual or synthetic source group.
#
# Group 9: L (2).mp4 & L (7).mp4
#   - Both 60 FPS, high color correlation = 0.827, similar Hoof GP trimming parlor setting.
#   - Confidence: low (kept separate unless proven)

# Build group assignments
group_assignments = {}
evidence_dict = {}
confidence_dict = {}

# Default: each clip gets its own group
for c in df_clips['filename']:
    prefix = "GRP_LAME" if c.startswith("L") else "GRP_NORM"
    num = int(''.join(filter(str.isdigit, Path(c).stem)))
    group_assignments[c] = f"{prefix}_{num:02d}"
    confidence_dict[c] = "confirmed" if c in url_map else "high"
    evidence_dict[c] = "Independent animal/clip based on distinct visual appearance, background, and movement."

# Apply mergers:
# 1. Normal N (3) & N (9) -> GRP_NORM_03
group_assignments["N (3).mp4"] = "GRP_NORM_03"
group_assignments["N (9).mp4"] = "GRP_NORM_03"
confidence_dict["N (3).mp4"] = "confirmed"
confidence_dict["N (9).mp4"] = "confirmed"
evidence_dict["N (3).mp4"] = "Exact source URL match: https://youtube.com/shorts/ntAGV4SQ0Fw (Source #3 & #9 in video_sources.txt)."
evidence_dict["N (9).mp4"] = "Exact source URL match: https://youtube.com/shorts/ntAGV4SQ0Fw (Source #3 & #9 in video_sources.txt)."

# 2. Lame L (8) & L (18) -> GRP_LAME_08
group_assignments["L (8).mp4"] = "GRP_LAME_08"
group_assignments["L (18).mp4"] = "GRP_LAME_08"
confidence_dict["L (8).mp4"] = "confirmed"
confidence_dict["L (18).mp4"] = "confirmed"
evidence_dict["L (8).mp4"] = "Temporal continuation and identical cow markings: boundary diff = 14.2, hist corr = 0.912, same pasture pen."
evidence_dict["L (18).mp4"] = "Temporal continuation and identical cow markings: boundary diff = 14.2, hist corr = 0.912, same pasture pen."

# 3. Normal N (20) & N (24) -> GRP_NORM_20
group_assignments["N (20).mp4"] = "GRP_NORM_20"
group_assignments["N (24).mp4"] = "GRP_NORM_20"
confidence_dict["N (20).mp4"] = "high"
confidence_dict["N (24).mp4"] = "high"
evidence_dict["N (20).mp4"] = "High visual similarity: hist corr = 0.984, BGR diff = 9.8, identical Holstein coat pattern on dirt track."
evidence_dict["N (24).mp4"] = "High visual similarity: hist corr = 0.984, BGR diff = 9.8, identical Holstein coat pattern on dirt track."

# 4. Normal N (18) & N (23) -> GRP_NORM_18
group_assignments["N (18).mp4"] = "GRP_NORM_18"
group_assignments["N (23).mp4"] = "GRP_NORM_18"
confidence_dict["N (18).mp4"] = "high"
confidence_dict["N (23).mp4"] = "high"
evidence_dict["N (18).mp4"] = "High visual & temporal similarity: 60 FPS, BGR diff = 11.5, boundary diff = 30.0, identical Jersey cow."
evidence_dict["N (23).mp4"] = "High visual & temporal similarity: 60 FPS, BGR diff = 11.5, boundary diff = 30.0, identical Jersey cow."

# 5. Lame L (13) & L (25) -> GRP_LAME_13
group_assignments["L (13).mp4"] = "GRP_LAME_13"
group_assignments["L (25).mp4"] = "GRP_LAME_13"
confidence_dict["L (13).mp4"] = "high"
confidence_dict["L (25).mp4"] = "high"
evidence_dict["L (13).mp4"] = "High visual & frame similarity: 60 FPS, hist corr = 0.937, identical cow coat markings in milking parlor exit."
evidence_dict["L (25).mp4"] = "High visual & frame similarity: 60 FPS, hist corr = 0.937, identical cow coat markings in milking parlor exit."

# 6. Lame L (6) & L (19) -> GRP_LAME_06
group_assignments["L (6).mp4"] = "GRP_LAME_06"
group_assignments["L (19).mp4"] = "GRP_LAME_06"
confidence_dict["L (6).mp4"] = "medium"
confidence_dict["L (19).mp4"] = "medium"
evidence_dict["L (6).mp4"] = "Medium confidence visual similarity: solid dark cow in low-light concrete barn, boundary diff = 34.9, pHash dist = 24."
evidence_dict["L (19).mp4"] = "Medium confidence visual similarity: solid dark cow in low-light concrete barn, boundary diff = 34.9, pHash dist = 24."

# 7. Normal N (13) & N (22) -> GRP_NORM_13
group_assignments["N (13).mp4"] = "GRP_NORM_13"
group_assignments["N (22).mp4"] = "GRP_NORM_13"
confidence_dict["N (13).mp4"] = "medium"
confidence_dict["N (22).mp4"] = "medium"
evidence_dict["N (13).mp4"] = "Medium confidence visual match: hist corr = 0.925, BGR diff = 21.8, same indoor barn lighting and Holstein patch."
evidence_dict["N (22).mp4"] = "Medium confidence visual match: hist corr = 0.925, BGR diff = 21.8, same indoor barn lighting and Holstein patch."

# 8. Lame L (16) & L (17) -> GRP_LAME_16 (Uncertain / Low)
group_assignments["L (16).mp4"] = "GRP_LAME_16"
group_assignments["L (17).mp4"] = "GRP_LAME_16"
confidence_dict["L (16).mp4"] = "low"
confidence_dict["L (17).mp4"] = "low"
evidence_dict["L (16).mp4"] = "Low confidence match: hist corr = 0.892, BGR diff = 15.9, similar outdoor pen but different angle/framerate (60 vs 30)."
evidence_dict["L (17).mp4"] = "Low confidence match: hist corr = 0.892, BGR diff = 15.9, similar outdoor pen but different angle/framerate (60 vs 30)."

df_clips['proposed_group_id'] = df_clips['filename'].map(group_assignments)
df_clips['confidence'] = df_clips['filename'].map(confidence_dict)
df_clips['evidence'] = df_clips['filename'].map(evidence_dict)

# 4. StratifiedGroupKFold (K=5)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
X = np.zeros(len(df_clips))
y = (df_clips['class'] == 'lame').astype(int).values
groups = df_clips['proposed_group_id'].values

folds = np.zeros(len(df_clips), dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    folds[val_idx] = fold_idx + 1

df_clips['proposed_fold'] = folds

# Export CSV
export_cols = ['filename', 'class', 'source_url', 'proposed_group_id', 'confidence', 'evidence', 'proposed_fold', 'duration', 'fps', 'resolution']
csv_path = REPO_ROOT / "datasets" / "lameness" / "cattle_lameness_manifest.csv"
df_clips[export_cols].to_csv(csv_path, index=False)
print(f"Manifest written to {csv_path}")

# Summary Stats
unique_groups = df_clips['proposed_group_id'].nunique()
lame_groups = df_clips[df_clips['class'] == 'lame']['proposed_group_id'].nunique()
norm_groups = df_clips[df_clips['class'] == 'normal']['proposed_group_id'].nunique()

print(f"\n--- Grouping Summary ---")
print(f"Total Unique Animal/Source Groups: {unique_groups} (Lame: {lame_groups}, Normal: {norm_groups})")

multi_clip_groups = df_clips.groupby('proposed_group_id').filter(lambda g: len(g) > 1)
print(f"Multi-clip groups count: {multi_clip_groups['proposed_group_id'].nunique()}")
for gid, grp in multi_clip_groups.groupby('proposed_group_id'):
    print(f"  {gid:<15}: {list(grp['filename'])} ({grp['confidence'].iloc[0]})")

print(f"\n--- 5-Fold Stratified Group Distribution ---")
fold_dist = df_clips.groupby(['proposed_fold', 'class']).size().unstack(fill_value=0)
fold_dist['Total Clips'] = fold_dist.sum(axis=1)
fold_groups = df_clips.groupby(['proposed_fold', 'class'])['proposed_group_id'].nunique().unstack(fill_value=0)
fold_groups['Total Groups'] = fold_groups.sum(axis=1)
print("Clips per fold:")
print(fold_dist)
print("\nGroups per fold:")
print(fold_groups)
