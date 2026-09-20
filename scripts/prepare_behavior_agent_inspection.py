"""
Script: scripts/prepare_behavior_agent_inspection.py
Purpose: Curates and stages 20 MmCows samples and 20 CBVD-5 samples for direct
         multimodal agent visual inspection across all classes, cameras, cows,
         and challenge conditions (occlusion, blur, lighting, scale, rumination).
"""

import os
import csv
import json
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_AUDIT_DIR = REPO_ROOT / "docs" / "audits"
STAGING_DIR = DOCS_AUDIT_DIR / "assets" / "agent_behavior_inspection"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. MMCOWS SAMPLE SELECTION (20 diverse samples)
# ------------------------------------------------------------------------------
mm_manifest_p = REPO_ROOT / "datasets" / "behavior" / "mmcows" / "manifest.csv"
with open(mm_manifest_p, "r", encoding="utf-8") as f:
    mm_records = list(csv.DictReader(f))

def find_mm(predicate, desc):
    for r in mm_records:
        if predicate(r):
            return r
    raise ValueError(f"Missing MmCows sample: {desc}")

mm_picks = [
    # Walking (Class 1) - dynamic stride, motion blur check
    {"id": "MM-01", "name": "Walking (Sharp / Aisle)", "pred": lambda r: r["class_name"] == "Walking" and r["camera_id"] == "1" and r["cow_id"] == "1", "focus": "Inspect active forward limb stride and floor boundary clarity."},
    {"id": "MM-02", "name": "Walking (Distant / Motion Blur)", "pred": lambda r: r["class_name"] == "Walking" and r["camera_id"] == "1" and r["cow_id"] == "5", "focus": "Evaluate motion blur on leg articulation and hoof boundary."},

    # Standing (Class 2) - upright posture across cameras and cows
    {"id": "MM-03", "name": "Standing (Clean Torso / Cam 1)", "pred": lambda r: r["class_name"] == "Standing" and r["camera_id"] == "1" and r["cow_id"] == "3", "focus": "Standard four-leg standing posture with full body profile."},
    {"id": "MM-04", "name": "Standing (Cam 2 / Feed Alley)", "pred": lambda r: r["class_name"] == "Standing" and r["camera_id"] == "2" and r["cow_id"] == "8", "focus": "Standing in alleyway from side-angle CCTV."},
    {"id": "MM-05", "name": "Standing (Cam 4 / Stall)", "pred": lambda r: r["class_name"] == "Standing" and r["camera_id"] == "4" and r["cow_id"] == "16", "focus": "Standing in cubicle stall; check partition obstruction."},

    # Feeding Head Up (Class 3) - head raised at feed bunk
    {"id": "MM-06", "name": "Feeding Head Up (Cow 4)", "pred": lambda r: r["class_name"] == "Feeding_head_up" and r["cow_id"] == "4", "focus": "Head lifted above trough; verify muzzle and neck angle."},
    {"id": "MM-07", "name": "Feeding Head Up (Cow 1)", "pred": lambda r: r["class_name"] == "Feeding_head_up" and r["cow_id"] == "1" and r["camera_id"] == "2", "focus": "Secondary camera angle on head-up transition."},

    # Feeding Head Down (Class 4) - muzzle inside feed bunk
    {"id": "MM-08", "name": "Feeding Head Down (Cow 5)", "pred": lambda r: r["class_name"] == "Feeding_head_down" and r["cow_id"] == "5", "focus": "Muzzle deep in feed bunk; check head visibility vs feed."},
    {"id": "MM-09", "name": "Feeding Head Down (Cow 12, Test)", "pred": lambda r: r["class_name"] == "Feeding_head_down" and r["cow_id"] == "12", "focus": "Independent held-out test cow feeding posture."},

    # Licking (Class 5) - rare self-grooming behavior
    {"id": "MM-10", "name": "Licking (Cow 6 / Coat)", "pred": lambda r: r["class_name"] == "Licking" and r["cow_id"] == "6", "focus": "Rare self-grooming; inspect head turned back licking side coat."},
    {"id": "MM-11", "name": "Licking (Cow 10 / Stall Rail)", "pred": lambda r: r["class_name"] == "Licking" and r["cow_id"] == "10", "focus": "Muzzle licking stall hardware/rail; subtle contact action."},

    # Drinking (Class 6) - water trough interaction
    {"id": "MM-12", "name": "Drinking (Cow 7)", "pred": lambda r: r["class_name"] == "Drinking" and r["cow_id"] == "7", "focus": "Muzzle lowered into water drinker."},
    {"id": "MM-13", "name": "Drinking (Cow 2, Test)", "pred": lambda r: r["class_name"] == "Drinking" and r["cow_id"] == "2", "focus": "Drinking on held-out test cow."},

    # Lying (Class 7) - recumbent resting in stalls
    {"id": "MM-14", "name": "Lying (Clean Cubicle / Cow 9)", "pred": lambda r: r["class_name"] == "Lying" and r["cow_id"] == "9" and r["camera_id"] == "3", "focus": "Resting recumbent posture in cubicle bed."},
    {"id": "MM-15", "name": "Lying (Deep Stall / Cow 13, Val)", "pred": lambda r: r["class_name"] == "Lying" and r["cow_id"] == "13" and r["camera_id"] == "4", "focus": "Validation cow resting under lower barn lighting."},

    # Challenging conditions: Occlusions, Shadows, Synchronized Views
    {"id": "MM-16", "name": "Occlusion: Stall Metal Bars", "pred": lambda r: r["cow_id"] == "11" and r["camera_id"] == "4" and r["class_name"] == "Lying", "focus": "Severe foreground obstruction by stall metal divider pipes."},
    {"id": "MM-17", "name": "Night / Dim Lighting", "pred": lambda r: "T23:" in r["timestamp_iso"], "focus": "Low-light barn illumination; check camera ISO noise and contrast."},
    {"id": "MM-18", "name": "Multi-Camera Sync View A (Cam 1)", "pred": lambda r: r["event_id"] == "ev_1690271846_c1" and r["camera_id"] == "1", "focus": "Simultaneous view A of Cow 1 feeding at 02:57:26."},
    {"id": "MM-19", "name": "Multi-Camera Sync View B (Cam 2)", "pred": lambda r: r["event_id"] == "ev_1690271846_c1" and r["camera_id"] == "2", "focus": "Simultaneous view B of Cow 1 feeding at 02:57:26 (same instant)."},
    {"id": "MM-20", "name": "Partial Crop / Tight Edge", "pred": lambda r: r["cow_id"] == "14" and r["camera_id"] == "3", "focus": "Check framing boundaries and whether limbs or tail are clipped."},
]

# ------------------------------------------------------------------------------
# 2. CBVD-5 SAMPLE SELECTION (20 diverse samples)
# ------------------------------------------------------------------------------
cbvd_dir = REPO_ROOT / "datasets" / "behavior" / "external" / "cbvd5"
cbvd_csv = cbvd_dir / "CBVD-5.csv"
lf_dir = cbvd_dir / "labelframes" / "labelframes"

option_map = {"0": "stand", "1": "lying down", "2": "foraging", "3": "drinking water", "4": "rumination"}
cbvd_records = []
with open(cbvd_csv, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        if '"[2,' in line:
            try:
                parts = line.strip().split('"[2,')
                fn_raw = parts[0].split('["')[1].split('"]')[0]
                fn_clean = fn_raw.replace('""', '').strip('"').strip()
                coord_str = parts[1].split(']"')[0]
                coords = [float(x.strip()) for x in coord_str.split(",")]
                meta_part = parts[1].split(']","')[1].rstrip('"\n')
                meta_dict = json.loads(meta_part.replace('""', '"'))
                opts = meta_dict.get("1", "").split(",")
                labels = [option_map.get(o.strip(), o.strip()) for o in opts if o.strip()]
                if (lf_dir / fn_clean).is_file():
                    cbvd_records.append({
                        "filename": fn_clean,
                        "video_id": fn_clean.split("_")[0],
                        "bbox": coords,
                        "labels": labels,
                    })
            except Exception:
                pass

def find_cbvd(predicate, desc):
    for a in cbvd_records:
        if predicate(a):
            return a
    raise ValueError(f"Missing CBVD sample: {desc}")

cbvd_picks = [
    # Stand behaviors
    {"id": "CBVD-01", "name": "Stand Only (Large Cow)", "pred": lambda a: a["labels"] == ["stand"] and a["bbox"][2] > 200, "focus": "Inspect pure standing posture with large pixel area."},
    {"id": "CBVD-02", "name": "Stand Only (Small / Distant Cow)", "pred": lambda a: a["labels"] == ["stand"] and a["bbox"][2] < 120, "focus": "Inspect distant standing cow; check resolution degradation."},

    # Lying down
    {"id": "CBVD-03", "name": "Lying Down (Pure Posture)", "pred": lambda a: a["labels"] == ["lying down"] and a["bbox"][2] > 180, "focus": "Resting cow on barn floor; check body contour."},
    {"id": "CBVD-04", "name": "Lying Down (Crowded / Overlapping)", "pred": lambda a: a["labels"] == ["lying down"] and a["filename"].startswith("618"), "focus": "Lying down next to other resting cows in pen."},

    # Foraging
    {"id": "CBVD-05", "name": "Foraging + Stand (Trough Feeding)", "pred": lambda a: "foraging" in a["labels"] and "stand" in a["labels"] and a["bbox"][2] > 180, "focus": "Cow standing and eating from feeding lane."},
    {"id": "CBVD-06", "name": "Foraging (Head Down at Ground)", "pred": lambda a: "foraging" in a["labels"] and a["bbox"][1] > 500, "focus": "Cow foraging near bottom edge of camera view."},

    # Drinking water
    {"id": "CBVD-07", "name": "Drinking Water (Trough)", "pred": lambda a: "drinking water" in a["labels"] and a["bbox"][2] > 150, "focus": "Muzzle at water trough; check water tank context."},
    {"id": "CBVD-08", "name": "Drinking Water (Distant Cow)", "pred": lambda a: "drinking water" in a["labels"], "focus": "Check drinking action clarity from a distance."},

    # Rumination (Co-occurring with lying and standing)
    {"id": "CBVD-09", "name": "Rumination + Lying Down (Chewing Cud)", "pred": lambda a: "rumination" in a["labels"] and "lying down" in a["labels"] and a["bbox"][2] > 200, "focus": "Lying cow labeled as chewing cud; check if mouth/jaw motion is visible in single crop."},
    {"id": "CBVD-10", "name": "Rumination + Stand (Upright Chewing)", "pred": lambda a: "rumination" in a["labels"] and "stand" in a["labels"] and a["bbox"][2] > 180, "focus": "Standing cow labeled as ruminating."},

    # Multi-cow crowded pen scenes
    {"id": "CBVD-11", "name": "Crowded Pen Scene A (Video 618)", "pred": lambda a: a["filename"] == "618_00002.jpg" and a["bbox"][0] > 800, "focus": "Scene with 6+ cows in view; check bounding box placement."},
    {"id": "CBVD-12", "name": "Crowded Pen Scene B (Video 621)", "pred": lambda a: a["filename"] == "621_00002.jpg" and a["bbox"][0] > 1000, "focus": "Multiple cows standing along fence; check inter-cow occlusion."},

    # Occlusions & Edge positions
    {"id": "CBVD-13", "name": "Fence / Railing Occlusion", "pred": lambda a: a["bbox"][0] < 100 and a["bbox"][1] > 400, "focus": "Cow partially behind metal fence gate in foreground."},
    {"id": "CBVD-14", "name": "Extreme Small Crop (<100px)", "pred": lambda a: a["bbox"][2] < 100 and a["bbox"][3] < 120, "focus": "Inspect low-resolution boundary case; is behavior distinguishable?"},

    # Temporal Sequence: Consecutive frames from Video 621 (Frames 2, 3, 4, 5)
    {"id": "CBVD-15", "name": "Consecutive Video 621 - Frame 2", "pred": lambda a: a["filename"] == "621_00002.jpg" and a["bbox"][0] < 100, "focus": "Frame t=2s of cow feeding along rail."},
    {"id": "CBVD-16", "name": "Consecutive Video 621 - Frame 3", "pred": lambda a: a["filename"] == "621_00003.jpg" and a["bbox"][0] < 100, "focus": "Frame t=3s (1 second later); check temporal continuity."},
    {"id": "CBVD-17", "name": "Consecutive Video 621 - Frame 4", "pred": lambda a: a["filename"] == "621_00004.jpg" and a["bbox"][0] < 100, "focus": "Frame t=4s (2 seconds later); check posture shift."},
    {"id": "CBVD-18", "name": "Consecutive Video 621 - Frame 5", "pred": lambda a: a["filename"] == "621_00005.jpg" and a["bbox"][0] < 100, "focus": "Frame t=5s (3 seconds later); observe feeding continuity."},

    # Different video sources (Videos 624, 689)
    {"id": "CBVD-19", "name": "Different Video Setup (Video 624)", "pred": lambda a: a["filename"].startswith("624"), "focus": "Check barn lighting and floor cleanliness in video 624."},
    {"id": "CBVD-20", "name": "Different Video Setup (Video 689)", "pred": lambda a: a["filename"].startswith("689"), "focus": "Check camera angle and perspective in video 689."},
]

# ------------------------------------------------------------------------------
# 3. EXPORT SAMPLES
# ------------------------------------------------------------------------------
manifest = {"mmcows": [], "cbvd": []}

print("Staging MmCows review images...")
for p in mm_picks:
    rec = find_mm(p["pred"], p["name"])
    src = REPO_ROOT / rec["image_path"]
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"Missing MmCows img: {src}")
    h, w = img.shape[:2]
    # Resize max width 400
    w_new = 400
    h_new = int(h * (w_new / w))
    thumb = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    out_fn = f"{p['id'].lower()}_{src.stem}.jpg"
    out_p = STAGING_DIR / out_fn
    cv2.imwrite(str(out_p), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])

    manifest["mmcows"].append({
        "id": p["id"],
        "name": p["name"],
        "asset_path": str(out_p),
        "rel_path": f"assets/agent_behavior_inspection/{out_fn}",
        "class_name": rec["class_name"],
        "class_id": rec["class_id"],
        "cow_id": rec["cow_id"],
        "camera_id": rec["camera_id"],
        "orig_res": f"{w}x{h}",
        "split": rec["canonical_split"],
        "focus": p["focus"],
    })

print("Staging CBVD-5 review images (crop + scene context)...")
for p in cbvd_picks:
    rec = find_cbvd(p["pred"], p["name"])
    frame_src = lf_dir / rec["filename"]
    frame = cv2.imread(str(frame_src))
    if frame is None:
        raise FileNotFoundError(f"Missing CBVD frame: {frame_src}")
    fh, fw = frame.shape[:2]
    bbox = rec["bbox"]
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1, y1 = max(0, min(fw - 1, x)), max(0, min(fh - 1, y))
    x2, y2 = max(0, min(fw, x + w)), max(0, min(fh, y + h))

    # Crop
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    cw_new = 360
    ch_new = max(1, int(ch * (cw_new / max(1, cw))))
    crop_r = cv2.resize(crop, (cw_new, ch_new), interpolation=cv2.INTER_AREA)
    crop_fn = f"{p['id'].lower()}_crop_{Path(rec['filename']).stem}.jpg"
    crop_p = STAGING_DIR / crop_fn
    cv2.imwrite(str(crop_p), crop_r, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Scene context thumbnail with bounding box
    scene = frame.copy()
    cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(scene, "+".join(rec["labels"]), (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    ctx_w = 480
    ctx_h = int(fh * (ctx_w / fw))
    scene_r = cv2.resize(scene, (ctx_w, ctx_h), interpolation=cv2.INTER_AREA)
    scene_fn = f"{p['id'].lower()}_scene_{Path(rec['filename']).stem}.jpg"
    scene_p = STAGING_DIR / scene_fn
    cv2.imwrite(str(scene_p), scene_r, [cv2.IMWRITE_JPEG_QUALITY, 85])

    manifest["cbvd"].append({
        "id": p["id"],
        "name": p["name"],
        "crop_path": str(crop_p),
        "scene_path": str(scene_p),
        "rel_crop_path": f"assets/agent_behavior_inspection/{crop_fn}",
        "rel_scene_path": f"assets/agent_behavior_inspection/{scene_fn}",
        "filename": rec["filename"],
        "video_id": rec["video_id"],
        "labels": rec["labels"],
        "crop_res": f"{cw}x{ch}",
        "scene_res": f"{fw}x{fh}",
        "focus": p["focus"],
    })

manifest_out = STAGING_DIR / "manifest.json"
with open(manifest_out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Staging complete: {len(manifest['mmcows'])} MmCows and {len(manifest['cbvd'])} CBVD samples saved.")
print(f"Manifest written to: {manifest_out}")
