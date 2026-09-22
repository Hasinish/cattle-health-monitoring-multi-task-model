# -*- coding: utf-8 -*-
"""
Extract annotation-level track segments from all 502 CVB instances_default.json files
on Modal volume cvb-data under profile tigerwood693.

Saves: datasets/behavior/cvb/cvb_tracks_manifest.csv
"""

import os
import sys
from pathlib import Path

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
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

volume = modal.Volume.from_name("cvb-data", create_if_missing=True)
VOLUME_DIR = "/data"
CVB_DIR = "/data/cvb/000058916v001"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "tqdm")
)
app = modal.App("cvb-extract-tracks", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def extract_cvb_track_segments_remote() -> str:
    """Parse all 502 JSONs and return CSV string of all track segments."""
    import glob
    import json
    import io
    import pandas as pd
    from collections import defaultdict

    data_dir = os.path.join(CVB_DIR, "data")
    ann_dir = os.path.join(data_dir, "annotations")
    json_paths = sorted(glob.glob(os.path.join(ann_dir, "*", "annotations", "instances_default.json")))
    print(f"[MODAL] Found {len(json_paths)} instances_default.json files.")

    records = []

    for jp in json_paths:
        cut_name = os.path.basename(os.path.dirname(os.path.dirname(jp)))
        parts = cut_name.split("_")
        
        # Parse source video id: arm01_{camera}_{date}_{time}
        camera = parts[2] if len(parts) > 2 else "unknown"
        date = parts[3] if len(parts) > 3 else "unknown"
        time_str = parts[4] if len(parts) > 4 else "unknown"
        source_video_id = f"arm01_{camera}_{date}_{time_str}"

        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])

        # Map image_id to frame index
        img_id_to_frame = {}
        for img in images:
            fname = img.get("file_name", "")
            # Pattern: .../img_00123.jpg
            if "img_" in fname:
                try:
                    fnum = int(fname.split("img_")[-1].split(".jpg")[0])
                except ValueError:
                    fnum = img.get("id", 0)
            else:
                fnum = img.get("id", 0)
            img_id_to_frame[img["id"]] = fnum

        # Group annotations by track_id
        tracks_anns = defaultdict(list)
        for ann in annotations:
            attrs = ann.get("attributes", {})
            tr_id = attrs.get("track_id", 0)
            beh = attrs.get("behavior", "missing")
            occ = bool(attrs.get("occluded", False))
            bbox = ann.get("bbox", [0, 0, 0, 0])
            img_id = ann.get("image_id", 0)
            frame_idx = img_id_to_frame.get(img_id, img_id)
            tracks_anns[tr_id].append({
                "frame": frame_idx,
                "behavior": beh,
                "occluded": occ,
                "bbox": bbox
            })

        # Process each track into contiguous segments of identical behavior
        for tr_id, anns in sorted(tracks_anns.items()):
            anns.sort(key=lambda x: x["frame"])
            if not anns:
                continue

            current_segment = [anns[0]]
            current_beh = anns[0]["behavior"]
            seg_idx = 0

            def finalize_segment(seg_items, s_idx, beh):
                start_f = seg_items[0]["frame"]
                end_f = seg_items[-1]["frame"]
                n_f = len(seg_items)
                occ_f = sum(1 for it in seg_items if it["occluded"]) / n_f
                mean_w = sum(it["bbox"][2] for it in seg_items if len(it["bbox"]) >= 4) / n_f
                mean_h = sum(it["bbox"][3] for it in seg_items if len(it["bbox"]) >= 4) / n_f
                records.append({
                    "cut_name": cut_name,
                    "source_video_id": source_video_id,
                    "camera_id": camera,
                    "date": date,
                    "time": time_str,
                    "track_id": tr_id,
                    "segment_idx": s_idx,
                    "sample_id": f"cvb_{cut_name}_tr{tr_id}_seg{s_idx}",
                    "behavior_original": beh,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "n_frames": n_f,
                    "fps": 30.0,
                    "mean_bbox_w": round(mean_w, 2),
                    "mean_bbox_h": round(mean_h, 2),
                    "occluded_fraction": round(occ_f, 4),
                })

            for ann in anns[1:]:
                # Contiguous frame and same behavior?
                if ann["behavior"] == current_beh and ann["frame"] == current_segment[-1]["frame"] + 1:
                    current_segment.append(ann)
                else:
                    finalize_segment(current_segment, seg_idx, current_beh)
                    seg_idx += 1
                    current_segment = [ann]
                    current_beh = ann["behavior"]

            finalize_segment(current_segment, seg_idx, current_beh)

    df = pd.DataFrame(records)
    print(f"[MODAL] Extracted {len(df)} track segments across {df['cut_name'].nunique()} cuts and {df['source_video_id'].nunique()} source videos.")
    print("[MODAL] Behavior breakdown:\n", df["behavior_original"].value_counts())
    
    out_buf = io.StringIO()
    df.to_csv(out_buf, index=False)
    return out_buf.getvalue()


@app.local_entrypoint()
def main():
    print("Launching CVB track segment extraction on Modal (profile tigerwood693)...")
    csv_data = extract_cvb_track_segments_remote.remote()
    out_path = Path("datasets/behavior/cvb/cvb_tracks_manifest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(csv_data)
    print(f"[LOCAL] Successfully saved {len(csv_data)} bytes to {out_path}")
