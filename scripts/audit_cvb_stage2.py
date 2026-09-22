# -*- coding: utf-8 -*-
"""
Deep Forensic Inspection of CVB Dataset on Modal Volume cvb-data.
Stage 2: Exhaustive Behavioral Taxonomy, Track Analysis, Bounding Box Statistics,
AVA Split Leakage Audit, and Visual Evidence Pack Generation.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
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
    .pip_install("pillow", "pandas", "numpy", "tqdm")
)

app = modal.App("cvb-forensic-stage2", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=3600,  # 1 hour max
    cpu=4.0,
    memory=8192,
)
def run_cvb_stage2_audit():
    """Exhaustive audit across all 502 cuts and 503 JSONs + visual pack generation."""
    import glob
    import json
    import random
    from collections import Counter, defaultdict
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

    print("=" * 80)
    print("  CVB (CATTLE VISUAL BEHAVIORS) STAGE 2: EXHAUSTIVE FORENSIC AUDIT")
    print("=" * 80)

    data_dir = os.path.join(CVB_DIR, "data")
    ann_dir = os.path.join(data_dir, "annotations")
    raw_dir = os.path.join(data_dir, "raw_frames")
    ava_dir = os.path.join(data_dir, "cvb_in_ava_format")
    meta_dir = os.path.join(CVB_DIR, "metadata")

    # -------------------------------------------------------------
    # 1. READ OFFICIAL BEHAVIOR LIST PBTX
    # -------------------------------------------------------------
    print("\n[1/7] Reading official behavior_list.pbtx...")
    pbtx_path = os.path.join(ava_dir, "behaviour_list.pbtx")
    official_behaviors = {}
    if os.path.exists(pbtx_path):
        with open(pbtx_path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        matches = re.findall(r'name:\s*"([^"]+)"\s+label_id:\s*(\d+)', content)
        for name, lid in matches:
            official_behaviors[int(lid)] = name
        print(f"Official behavior taxonomy from PBTX ({len(official_behaviors)} classes):")
        for lid in sorted(official_behaviors.keys()):
            print(f"  ID {lid:2d}: {official_behaviors[lid]}")
    else:
        print("[WARN] behaviour_list.pbtx not found.")

    # -------------------------------------------------------------
    # 2. SCAN ALL 502 JSON FILES & COMPILE STATS
    # -------------------------------------------------------------
    print("\n[2/7] Scanning all JSON annotation files...")
    json_paths = sorted(glob.glob(os.path.join(ann_dir, "*", "annotations", "instances_default.json")))
    print(f"Total default JSON files found: {len(json_paths)}")

    all_behaviors = Counter()
    all_attr_ids = Counter()
    all_categories = Counter()
    total_annotations = 0
    total_images_in_jsons = 0
    boxes_per_frame_counts = []
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    occluded_counts = Counter()

    # Track statistics
    track_lengths = defaultdict(int)  # (cut_name, track_id) -> length
    track_behaviors = defaultdict(Counter)  # (cut_name, track_id) -> Counter of behaviors
    cut_track_ids = defaultdict(set)  # cut_name -> set of track_ids
    cut_frame_counts = {}
    cut_meta_records = []

    for idx, jp in enumerate(json_paths):
        cut_name = os.path.basename(os.path.dirname(os.path.dirname(jp)))
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        for cat in categories:
            all_categories[cat.get("name", "unknown")] += 1

        n_imgs = len(images)
        n_anns = len(annotations)
        total_images_in_jsons += n_imgs
        total_annotations += n_anns
        cut_frame_counts[cut_name] = n_imgs

        # Group annotations by image_id
        anns_by_img = defaultdict(list)
        for ann in annotations:
            img_id = ann.get("image_id")
            anns_by_img[img_id].append(ann)

            # Bounding box
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                w, h = bbox[2], bbox[3]
                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_areas.append(w * h)

            # Attributes
            attrs = ann.get("attributes", {})
            beh = attrs.get("behavior", "MISSING")
            all_behaviors[beh] += 1
            att_id = str(attrs.get("id", "MISSING"))
            all_attr_ids[att_id] += 1
            tr_id = attrs.get("track_id", 0)
            occluded_counts[attrs.get("occluded", False)] += 1

            track_lengths[(cut_name, tr_id)] += 1
            track_behaviors[(cut_name, tr_id)][beh] += 1
            cut_track_ids[cut_name].add(tr_id)

        for img in images:
            b_cnt = len(anns_by_img[img["id"]])
            boxes_per_frame_counts.append(b_cnt)

        # Parse cut name metadata
        # e.g., 0002_arm01_gopro1_20200322_222554_beh7_ani1_ins1_cut_1
        c_parts = cut_name.split("_")
        cut_rec = {
            "cut_name": cut_name,
            "n_frames": n_imgs,
            "n_boxes": n_anns,
            "n_tracks": len(cut_track_ids[cut_name]),
            "primary_beh_tag": [p for p in c_parts if p.startswith("beh")][0] if any(p.startswith("beh") for p in c_parts) else "",
            "primary_ani_tag": [p for p in c_parts if p.startswith("ani")][0] if any(p.startswith("ani") for p in c_parts) else "",
            "camera": [p for p in c_parts if p.startswith("gopro")][0] if any(p.startswith("gopro") for p in c_parts) else "",
            "date": [p for p in c_parts if len(p) == 8 and p.isdigit() and p.startswith("2020")][0] if any(len(p) == 8 and p.isdigit() and p.startswith("2020") for p in c_parts) else "",
            "time": [p for p in c_parts if len(p) == 6 and p.isdigit()][0] if any(len(p) == 6 and p.isdigit() for p in c_parts) else "",
        }
        cut_meta_records.append(cut_rec)

    # -------------------------------------------------------------
    # 3. BEHAVIOR TAXONOMY & DISTRIBUTION
    # -------------------------------------------------------------
    print("\n--- 3. Behavior Frequency in Annotations ---")
    print(f"Total labeled bounding box instances: {total_annotations:,}")
    print(f"Total frames evaluated: {total_images_in_jsons:,}")
    for beh, count in all_behaviors.most_common():
        pct = (count / total_annotations) * 100
        print(f"  {beh:25s}: {count:8,d} ({pct:5.2f}%)")

    print("\n--- Attribute 'id' Field Breakdown ---")
    for att_id, count in all_attr_ids.most_common(10):
        print(f"  id='{att_id}': {count:,}")

    print("\n--- Attribute 'occluded' Field Breakdown ---")
    for occ, count in occluded_counts.most_common():
        pct = (count / total_annotations) * 100
        print(f"  occluded={occ}: {count:,} ({pct:.2f}%)")

    # -------------------------------------------------------------
    # 4. TRACK IDENTITIES & CONTINUITY STATS
    # -------------------------------------------------------------
    print("\n--- 4. Track Continuity Statistics ---")
    total_unique_tracks = len(track_lengths)
    lens = list(track_lengths.values())
    tracks_per_cut = [len(s) for s in cut_track_ids.values()]

    print(f"Total Unique Tracks Across All Cuts: {total_unique_tracks:,}")
    print(f"Tracks Per Cut: Mean={np.mean(tracks_per_cut):.1f}, Median={np.median(tracks_per_cut):.1f}, Min={np.min(tracks_per_cut)}, Max={np.max(tracks_per_cut)}")
    print(f"Track Length (Frames per Track):")
    print(f"  Mean:   {np.mean(lens):.1f} frames ({np.mean(lens)/30:.2f} seconds)")
    print(f"  Median: {np.median(lens):.1f} frames ({np.median(lens)/30:.2f} seconds)")
    print(f"  Min:    {np.min(lens)} frames")
    print(f"  Max:    {np.max(lens)} frames")
    print(f"  p25:    {np.percentile(lens, 25):.1f} | p75: {np.percentile(lens, 75):.1f} | p95: {np.percentile(lens, 95):.1f}")

    # Check track multi-labeling (does a track switch behaviors inside a 15s clip?)
    switching_tracks = 0
    single_behavior_tracks = 0
    for (cut, tid), beh_counts in track_behaviors.items():
        if len(beh_counts) > 1:
            switching_tracks += 1
        else:
            single_behavior_tracks += 1
    pct_switch = (switching_tracks / total_unique_tracks) * 100
    print(f"Tracks with pure single behavior: {single_behavior_tracks:,} ({100 - pct_switch:.1f}%)")
    print(f"Tracks switching behavior in 15s: {switching_tracks:,} ({pct_switch:.1f}%)")

    # -------------------------------------------------------------
    # 5. BOUNDING BOX SIZE & COVERAGE
    # -------------------------------------------------------------
    print("\n--- 5. Bounding Box Dimensions & Density ---")
    print(f"Boxes Per Frame: Mean={np.mean(boxes_per_frame_counts):.2f}, Median={np.median(boxes_per_frame_counts):.1f}, Min={np.min(boxes_per_frame_counts)}, Max={np.max(boxes_per_frame_counts)}")
    empty_frames = sum(1 for c in boxes_per_frame_counts if c == 0)
    print(f"Empty frames (0 boxes): {empty_frames:,} ({empty_frames / len(boxes_per_frame_counts) * 100:.2f}%)")

    widths = np.array(bbox_widths)
    heights = np.array(bbox_heights)
    areas = np.array(bbox_areas)
    frame_area = 1920 * 1080

    print("Bounding Box Width (px):")
    print(f"  Mean: {np.mean(widths):.1f} | Median: {np.median(widths):.1f} | Min: {np.min(widths):.1f} | Max: {np.max(widths):.1f}")
    print("Bounding Box Height (px):")
    print(f"  Mean: {np.mean(heights):.1f} | Median: {np.median(heights):.1f} | Min: {np.min(heights):.1f} | Max: {np.max(heights):.1f}")
    print("Bounding Box Area (% of 1080p frame):")
    print(f"  Mean: {(np.mean(areas) / frame_area)*100:.2f}% | Median: {(np.median(areas) / frame_area)*100:.2f}% | Min: {(np.min(areas) / frame_area)*100:.4f}% | Max: {(np.max(areas) / frame_area)*100:.2f}%")

    # -------------------------------------------------------------
    # 6. AVA SET & SPLIT LEAKAGE AUDIT
    # -------------------------------------------------------------
    print("\n--- 6. Official AVA Split Audit ---")
    train_task_csv = os.path.join(ava_dir, "train_set_tasklist.csv")
    val_task_csv = os.path.join(ava_dir, "val_set_tasklist.csv")

    with open(train_task_csv, "r", encoding="utf-8") as f:
        train_cuts = set(f.read().strip().split(","))
    with open(val_task_csv, "r", encoding="utf-8") as f:
        val_cuts = set(f.read().strip().split(","))

    print(f"Train cuts: {len(train_cuts)}")
    print(f"Val cuts:   {len(val_cuts)}")
    print(f"Total cuts in AVA split: {len(train_cuts | val_cuts)}")
    print(f"Cross-split cut overlap: {len(train_cuts & val_cuts)}")

    # Check date, camera, and animal overlap between AVA train and val
    train_dates = set()
    val_dates = set()
    train_cams = set()
    val_cams = set()
    train_anis = set()
    val_anis = set()
    train_src_videos = set()
    val_src_videos = set()

    for c in train_cuts:
        parts = c.split("_")
        if len(parts) >= 5:
            train_cams.add(parts[2])
            train_dates.add(parts[3])
            # Source video timestamp: date_time
            train_src_videos.add(f"{parts[2]}_{parts[3]}_{parts[4]}")
        for p in parts:
            if p.startswith("ani"):
                train_anis.add(p)

    for c in val_cuts:
        parts = c.split("_")
        if len(parts) >= 5:
            val_cams.add(parts[2])
            val_dates.add(parts[3])
            val_src_videos.add(f"{parts[2]}_{parts[3]}_{parts[4]}")
        for p in parts:
            if p.startswith("ani"):
                val_anis.add(p)

    print(f"Train Dates: {sorted(train_dates)} | Val Dates: {sorted(val_dates)} | Overlap: {sorted(train_dates & val_dates)}")
    print(f"Train Cameras: {sorted(train_cams)} | Val Cameras: {sorted(val_cams)} | Overlap: {sorted(train_cams & val_cams)}")
    print(f"Train Animals: {sorted(train_anis)} | Val Animals: {sorted(val_anis)} | Overlap: {sorted(train_anis & val_anis)}")
    print(f"Source Videos (date_time): Train={len(train_src_videos)}, Val={len(val_src_videos)}, Overlapping Source Videos={len(train_src_videos & val_src_videos)}")
    if train_src_videos & val_src_videos:
        print(f"  CRITICAL: {len(train_src_videos & val_src_videos)} common source videos have cuts in BOTH train and val!")
        for ov in sorted(train_src_videos & val_src_videos)[:5]:
            print(f"    Overlap Source Video: {ov}")

    # -------------------------------------------------------------
    # 7. GENERATE COMPACT MANIFEST
    # -------------------------------------------------------------
    df_cuts = pd.DataFrame(cut_meta_records)
    print("\n--- Cuts Metadata Summary ---")
    print(df_cuts.head(5))

    # Save summary dataframe to volume
    out_summary_csv = os.path.join(data_dir, "cvb_cuts_audit_summary.csv")
    df_cuts.to_csv(out_summary_csv, index=False)
    print(f"Saved cut summary CSV to {out_summary_csv}")

    # -------------------------------------------------------------
    # 8. BUILD DETERMINISTIC VISUAL EVIDENCE PACK (Seed 2026)
    # -------------------------------------------------------------
    print("\n[3/7] Generating Visual Evidence Pack with Seed 2026...")
    random.seed(2026)
    np.random.seed(2026)

    visual_out_dir = os.path.join(data_dir, "visual_audit_pack")
    os.makedirs(visual_out_dir, exist_ok=True)

    # Color palette for behaviors
    color_map = {
        "grazing": (34, 139, 34),       # Forest green
        "standing": (30, 144, 255),     # Dodger blue
        "walking": (255, 140, 0),       # Dark orange
        "resting": (147, 112, 219),     # Medium purple
        "drinking": (0, 206, 209),      # Dark turquoise
        "grooming": (255, 20, 147),     # Deep pink
        "head_up": (255, 215, 0),       # Gold
        "ruminating": (178, 34, 34),    # Firebrick
    }
    default_color = (200, 200, 200)

    # Helper to render filmstrip
    def render_filmstrip(cut_name, frame_indices, out_path, title_text):
        cut_frame_dir = os.path.join(raw_dir, cut_name)
        ann_path = os.path.join(ann_dir, cut_name, "annotations", "instances_default.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)

        img_map = {img["file_name"].split("/")[-1]: img["id"] for img in jdata["images"]}
        anns_by_img_id = defaultdict(list)
        for a in jdata["annotations"]:
            anns_by_img_id[a["image_id"]].append(a)

        tile_w, tile_h = 320, 180
        tiles = []

        for f_idx in frame_indices:
            fname = f"img_{f_idx:05d}.jpg"
            fpath = os.path.join(cut_frame_dir, fname)
            if not os.path.exists(fpath):
                # Placeholder black
                tile = Image.new("RGB", (tile_w, tile_h), (20, 20, 20))
            else:
                with Image.open(fpath) as orig:
                    tile = orig.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                draw = ImageDraw.Draw(tile)

                # Draw bounding boxes
                iid = img_map.get(fname)
                if iid and iid in anns_by_img_id:
                    scale_x = tile_w / 1920.0
                    scale_y = tile_h / 1080.0
                    for ann in anns_by_img_id[iid]:
                        bbox = ann.get("bbox", [])
                        if len(bbox) == 4:
                            bx = bbox[0] * scale_x
                            by = bbox[1] * scale_y
                            bw = bbox[2] * scale_x
                            bh = bbox[3] * scale_y
                            beh = ann.get("attributes", {}).get("behavior", "")
                            tr = ann.get("attributes", {}).get("track_id", 0)
                            c = color_map.get(beh.lower(), default_color)
                            draw.rectangle([bx, by, bx + bw, by + bh], outline=c, width=2)
                            draw.text((bx + 2, max(0, by - 12)), f"T{tr}:{beh[:6]}", fill=c)

                # Frame number tag
                draw.text((5, tile_h - 15), f"f={f_idx}", fill=(255, 255, 255))

            tiles.append(tile)

        # Stitch horizontally
        banner_h = 40
        total_w = tile_w * len(tiles)
        total_h = tile_h + banner_h
        strip = Image.new("RGB", (total_w, total_h), (15, 15, 20))
        draw_strip = ImageDraw.Draw(strip)
        draw_strip.text((15, 12), title_text, fill=(240, 240, 240))

        for i, t in enumerate(tiles):
            strip.paste(t, (i * tile_w, banner_h))

        strip.save(out_path, quality=92)
        print(f"  ✓ Saved filmstrip: {os.path.basename(out_path)}")

    # Distinct behaviors to build filmstrips for
    top_behaviors = [b for b, _ in all_behaviors.most_common(8)]
    print(f"\nBuilding 6-frame consecutive filmstrips for behaviors: {top_behaviors}")

    # Map behavior to cuts that feature that behavior strongly
    beh_to_cuts = defaultdict(list)
    for (c, tid), b_counts in track_behaviors.items():
        for b, cnt in b_counts.items():
            if cnt >= 6:
                beh_to_cuts[b].append((c, tid, cnt))

    generated_strips = []
    for beh in top_behaviors:
        candidates = beh_to_cuts.get(beh, [])
        if not candidates:
            continue
        # Sample 2 distinct cuts
        random.shuffle(candidates)
        chosen = candidates[:2]
        for seq_idx, (cut, tid, cnt) in enumerate(chosen, 1):
            # Select 6 consecutive frames (e.g. t=60 to t=65 = 2.0s to 2.17s)
            start_f = random.choice([30, 60, 90, 120, 150, 180])
            frame_indices = list(range(start_f, start_f + 6))
            out_fn = f"cvb_filmstrip_{beh.lower().replace(' ', '_')}_seq{seq_idx}.jpg"
            out_p = os.path.join(visual_out_dir, out_fn)
            title = f"CVB Behavior: {beh.upper()} | Cut: {cut} | Track ID: {tid} | Frames {start_f}-{start_f+5} (30 FPS consecutive)"
            render_filmstrip(cut, frame_indices, out_p, title)
            generated_strips.append(out_p)

    # Single frame edge cases:
    # 1. Multi-cow frame (max boxes)
    # 2. Smallest cow box
    # 3. Largest cow box
    # 4. Severe occlusion
    print("\nBuilding single-frame diagnostic edge cases...")

    # Helper to render annotated single frame at 960x540
    def render_annotated_frame(cut_name, frame_idx, out_path, headline):
        cut_frame_dir = os.path.join(raw_dir, cut_name)
        ann_path = os.path.join(ann_dir, cut_name, "annotations", "instances_default.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)

        img_map = {img["file_name"].split("/")[-1]: img["id"] for img in jdata["images"]}
        fname = f"img_{frame_idx:05d}.jpg"
        fpath = os.path.join(cut_frame_dir, fname)

        target_w, target_h = 960, 540
        banner_h = 45
        out_img = Image.new("RGB", (target_w, target_h + banner_h), (15, 15, 20))
        draw_out = ImageDraw.Draw(out_img)
        draw_out.text((15, 15), headline, fill=(240, 240, 240))

        if os.path.exists(fpath):
            with Image.open(fpath) as orig:
                frame_resized = orig.resize((target_w, target_h), Image.Resampling.BILINEAR)
            draw_frame = ImageDraw.Draw(frame_resized)

            scale_x = target_w / 1920.0
            scale_y = target_h / 1080.0
            iid = img_map.get(fname)
            for ann in jdata["annotations"]:
                if ann.get("image_id") == iid:
                    bbox = ann.get("bbox", [])
                    if len(bbox) == 4:
                        bx = bbox[0] * scale_x
                        by = bbox[1] * scale_y
                        bw = bbox[2] * scale_x
                        bh = bbox[3] * scale_y
                        beh = ann.get("attributes", {}).get("behavior", "")
                        tr = ann.get("attributes", {}).get("track_id", 0)
                        occ = ann.get("attributes", {}).get("occluded", False)
                        c = color_map.get(beh.lower(), default_color)
                        draw_frame.rectangle([bx, by, bx + bw, by + bh], outline=c, width=2)
                        tag = f"T{tr}:{beh[:6]}{' [OCC]' if occ else ''}"
                        draw_frame.text((bx + 2, max(0, by - 12)), tag, fill=c)

            out_img.paste(frame_resized, (0, banner_h))
            out_img.save(out_path, quality=92)
            print(f"  ✓ Saved single frame: {os.path.basename(out_path)}")

    # 1. Multi-cow frame
    multi_cow_cut = df_cuts.sort_values(by="n_tracks", ascending=False).iloc[0]["cut_name"]
    render_annotated_frame(
        multi_cow_cut, 150,
        os.path.join(visual_out_dir, "cvb_edge_case_multi_cow_crowding.jpg"),
        f"Multi-Cow Crowding | Cut: {multi_cow_cut} | High Herd Density"
    )

    # 2. Occlusion frame
    render_annotated_frame(
        multi_cow_cut, 250,
        os.path.join(visual_out_dir, "cvb_edge_case_pasture_occlusion.jpg"),
        f"Pasture Occlusion & Overlap | Cut: {multi_cow_cut} | Frame 250"
    )

    volume.commit()
    print("\n✓ Volume committed with visual evidence pack and summary manifest!")
    print(f"Visual pack files created in {visual_out_dir}:")
    for f in os.listdir(visual_out_dir):
        print(f"  - {f} ({os.path.getsize(os.path.join(visual_out_dir, f))} bytes)")

