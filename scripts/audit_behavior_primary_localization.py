# -*- coding: utf-8 -*-
"""
Phase 3 Behavior Primary Localization Sanity Check
Evaluates RT-DETR-L localization directly on the new primary Behavior stack:
CVB (25 samples) + Kaggle Beef Cattle Behavior (20 samples) from datasets/behavior/cvb_beef/train.csv.

Modal Profile: tigerwood693
GPU Tier: T4
Seed: 2026

Outputs:
  - artifacts/perception_audit/behavior_primary_localization_sanity.csv
  - docs/audits/assets/behavior_primary_localization_sanity/ (overlays + contact sheets)
  - docs/research_log/2026-09-23_behavior_primary_localization_sanity.md
"""

import os
import sys
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# Guard against Windows cross-drive ValueError in ntpath.commonpath for Modal
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

# Volumes
cvb_vol = modal.Volume.from_name("cvb-data")
beef_vol = modal.Volume.from_name("beef-behavior-data")

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.1.0",
        "opencv-python-headless",
        "pillow",
        "pandas",
        "numpy",
    )
)

app = modal.App("behavior-primary-localization-sanity", image=image)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@app.function(
    volumes={"/data": cvb_vol},
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def audit_cvb_on_modal(cvb_records):
    """
    Execute RT-DETR-L inference on 25 CVB training track segments.
    Recovers authentic target-cow GT bbox from instances_default.json.
    """
    import os
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import RTDETR

    print(f"[MODAL-CVB] Starting RT-DETR-L audit on {len(cvb_records)} CVB samples...")
    print(f"[MODAL-CVB] CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

    # Load RT-DETR-L
    model = RTDETR("rtdetr-l.pt")
    cvb_root = "/data/cvb/000058916v001"

    cached_jsons = {}
    results = []

    for idx, rec in enumerate(cvb_records):
        sample_id = rec["sample_id"]
        cut_name = rec["session_id"]
        source_vid = rec["source_video_id"]
        track_id = int(rec["tracklet_id"])
        beh_canon = rec["behavior_canonical"]
        start_f = int(rec["start_frame"])
        end_f = int(rec["end_frame"])
        mid_f = (start_f + end_f) // 2

        img_path = os.path.join(cvb_root, "data", "raw_frames", cut_name, f"img_{mid_f:05d}.jpg")
        ann_path = os.path.join(cvb_root, "data", "annotations", cut_name, "annotations", "instances_default.json")

        if not os.path.exists(img_path):
            alt_path = os.path.join(cvb_root, "data", "raw_frames", cut_name, f"img_{mid_f}.jpg")
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                print(f"[WARN] Frame image not found: {img_path}")
                continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ERROR] Could not read image: {img_path}")
            continue

        img_h, img_w = img_bgr.shape[:2]

        # 1. Recover official target GT bbox from authentic annotation source
        if cut_name not in cached_jsons:
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f:
                    cached_jsons[cut_name] = json.load(f)
            else:
                cached_jsons[cut_name] = None

        jdata = cached_jsons[cut_name]
        gt_box = None  # [x1, y1, x2, y2]
        gt_status = "gt_unavailable"

        if jdata is not None:
            target_img_id = None
            for im in jdata.get("images", []):
                fname = im.get("file_name", "")
                base_fname = os.path.basename(fname)
                if "img_" in base_fname:
                    try:
                        fnum = int(base_fname.split("img_")[-1].split(".")[0])
                        if fnum == mid_f:
                            target_img_id = im.get("id")
                            break
                    except ValueError:
                        pass
                if target_img_id is None and (f"img_{mid_f:05d}.jpg" in fname or f"_{mid_f}.jpg" in fname):
                    target_img_id = im.get("id")
                    break

            if target_img_id is not None:
                for ann in jdata.get("annotations", []):
                    if ann.get("image_id") == target_img_id:
                        attrs = ann.get("attributes", {})
                        if attrs.get("track_id") == track_id:
                            raw_box = ann.get("bbox", [])
                            if len(raw_box) == 4:
                                gx, gy, gw, gh = raw_box
                                gt_box = [float(gx), float(gy), float(gx + gw), float(gy + gh)]
                                gt_status = "available"
                                break

        # 2. Run RT-DETR-L inference
        t0 = time.perf_counter()
        preds = model(img_bgr, conf=0.25, classes=[19], verbose=False, device="cuda")
        t_infer_ms = (time.perf_counter() - t0) * 1000.0

        b_objs = preds[0].boxes
        if len(b_objs) > 0:
            pred_boxes = b_objs.xyxy.cpu().numpy().tolist()
            pred_confs = b_objs.conf.cpu().numpy().tolist()
            num_cows = len(pred_boxes)
            highest_conf = float(np.max(pred_confs))
            detection_status = "detected"
        else:
            pred_boxes = []
            pred_confs = []
            num_cows = 0
            highest_conf = 0.0
            detection_status = "no_detection"

        # 3. Match with Target GT Box (IoU evaluation)
        best_iou = 0.0
        matched_conf = 0.0
        matched_box_idx = -1
        target_overlap_flag = False

        if gt_box is not None and num_cows > 0:
            ious = [compute_iou(gt_box, pbox) for pbox in pred_boxes]
            matched_box_idx = int(np.argmax(ious))
            best_iou = float(ious[matched_box_idx])
            matched_conf = float(pred_confs[matched_box_idx])
            target_overlap_flag = (best_iou > 0.0)

        # Max box area ratio
        if num_cows > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) / (img_w * img_h) for b in pred_boxes]
            max_box_area_ratio = float(np.max(areas))
        else:
            max_box_area_ratio = 0.0

        # 4. Generate Visual Overlay
        overlay = img_bgr.copy()

        # Unmatched detections (Orange)
        for d_i, (pbox, pconf) in enumerate(zip(pred_boxes, pred_confs)):
            if d_i != matched_box_idx:
                x1, y1, x2, y2 = [int(round(c)) for c in pbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(
                    overlay,
                    f"Cow {pconf:.2f}",
                    (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 140, 255),
                    2,
                )

        # Matched detection (Cyan / Yellow)
        if matched_box_idx >= 0:
            x1, y1, x2, y2 = [int(round(c)) for c in pred_boxes[matched_box_idx]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(
                overlay,
                f"Matched Det: {matched_conf:.2f} (IoU:{best_iou:.2f})",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # Official Target GT Box (Green)
        if gt_box is not None:
            gx1, gy1, gx2, gy2 = [int(round(c)) for c in gt_box]
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
            cv2.putText(
                overlay,
                f"Target GT (Trk {track_id})",
                (gx1, min(gy2 + 20, img_h - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Top Information Banner
        banner_h = 42
        cv2.rectangle(overlay, (0, 0), (img_w, banner_h), (20, 20, 20), -1)
        banner_text = (
            f"CVB | {beh_canon} | Frame {mid_f} | Track {track_id} | "
            f"GT: {'YES' if gt_box else 'NONE'} | Dets: {num_cows} | BestIoU: {best_iou:.3f} | Lat: {t_infer_ms:.1f}ms"
        )
        cv2.putText(overlay, banner_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode to JPEG bytes
        _, jpg_buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_bytes = jpg_buf.tobytes()

        res_dict = {
            "sample_id": sample_id,
            "dataset": "cvb",
            "canonical_behavior": beh_canon,
            "source_video_id": source_vid,
            "session_id": cut_name,
            "frame_index": mid_f,
            "image_width": img_w,
            "image_height": img_h,
            "num_cattle_detections": num_cows,
            "highest_cattle_confidence": round(highest_conf, 4),
            "detection_status": detection_status,
            "inference_time_ms": round(t_infer_ms, 2),
            "gt_status": gt_status,
            "target_gt_bbox": json.dumps([round(c, 1) for c in gt_box]) if gt_box else "gt_unavailable",
            "target_best_iou": round(best_iou, 4) if gt_status == "available" else None,
            "target_overlap_flag": target_overlap_flag if gt_status == "available" else None,
            "matched_detection_conf": round(matched_conf, 4) if matched_box_idx >= 0 else None,
            "max_box_area_ratio": round(max_box_area_ratio, 4),
            "no_detection_flag": (num_cows == 0),
        }
        results.append((res_dict, overlay_bytes))
        print(f"  [CVB {idx+1}/{len(cvb_records)}] {sample_id} | {beh_canon} | Dets: {num_cows} | IoU: {best_iou:.3f} ({t_infer_ms:.1f}ms)")

    print(f"[MODAL-CVB] Completed {len(results)} samples.")
    return results


@app.function(
    volumes={"/data": beef_vol},
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def audit_beef_on_modal(beef_records):
    """
    Execute RT-DETR-L inference on 20 Kaggle Beef training clips.
    Extracts deterministic midpoint frame from 224x224 single-cow crops.
    """
    import os
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import RTDETR

    print(f"[MODAL-BEEF] Starting RT-DETR-L audit on {len(beef_records)} Beef samples...")
    print(f"[MODAL-BEEF] CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

    model = RTDETR("rtdetr-l.pt")
    beef_root = "/data/beef_behavior"
    results = []

    for idx, rec in enumerate(beef_records):
        sample_id = rec["sample_id"]
        session_id = str(rec["session_id"])
        source_vid = str(rec["source_video_id"])
        beh_canon = rec["behavior_canonical"]
        rel_path = rec["source_path"]  # e.g. clips/eat/149_3_clip_0.mp4
        n_frames = int(rec["n_frames"])
        mid_f = n_frames // 2

        sub_path = rel_path.replace("clips/", "")
        vid_candidates = [
            os.path.join(beef_root, "Category Videos", "cows", sub_path),
            os.path.join(beef_root, "clips", sub_path),
            os.path.join(beef_root, rel_path),
        ]
        vid_path = None
        for vc in vid_candidates:
            if os.path.exists(vc):
                vid_path = vc
                break

        if vid_path is None:
            fn = os.path.basename(rel_path)
            for root, _, files in os.walk(beef_root):
                if fn in files:
                    vid_path = os.path.join(root, fn)
                    break

        if vid_path is None:
            print(f"[WARN] Video clip not found: {rel_path}")
            continue

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            print(f"[ERROR] Could not read frame {mid_f} from {vid_path}")
            continue

        img_h, img_w = frame_bgr.shape[:2]

        # Run RT-DETR-L
        t0 = time.perf_counter()
        preds = model(frame_bgr, conf=0.25, classes=[19], verbose=False, device="cuda")
        t_infer_ms = (time.perf_counter() - t0) * 1000.0

        b_objs = preds[0].boxes
        if len(b_objs) > 0:
            pred_boxes = b_objs.xyxy.cpu().numpy().tolist()
            pred_confs = b_objs.conf.cpu().numpy().tolist()
            num_cows = len(pred_boxes)
            highest_conf = float(np.max(pred_confs))
            detection_status = "detected"
            areas = [(b[2] - b[0]) * (b[3] - b[1]) / (img_w * img_h) for b in pred_boxes]
            max_box_area_ratio = float(np.max(areas))
        else:
            pred_boxes = []
            pred_confs = []
            num_cows = 0
            highest_conf = 0.0
            detection_status = "no_detection"
            max_box_area_ratio = 0.0

        # Visual Overlay on 224x224 crop
        overlay = frame_bgr.copy()
        for d_i, (pbox, pconf) in enumerate(zip(pred_boxes, pred_confs)):
            x1, y1, x2, y2 = [int(round(c)) for c in pbox]
            box_area_frac = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) / (img_w * img_h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"cow {pconf:.2f} ({box_area_frac:.2f})",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )

        # Top banner on 224x224 crop
        banner_h = 24
        cv2.rectangle(overlay, (0, 0), (img_w, banner_h), (20, 20, 20), -1)
        banner_text = f"Beef | {beh_canon} | Cows: {num_cows} | Area: {max_box_area_ratio:.2f}"
        cv2.putText(overlay, banner_text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

        # Encode to JPEG bytes
        _, jpg_buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        overlay_bytes = jpg_buf.tobytes()

        res_dict = {
            "sample_id": sample_id,
            "dataset": "beef_cattle_behavior",
            "canonical_behavior": beh_canon,
            "source_video_id": source_vid,
            "session_id": session_id,
            "frame_index": mid_f,
            "image_width": img_w,
            "image_height": img_h,
            "num_cattle_detections": num_cows,
            "highest_cattle_confidence": round(highest_conf, 4),
            "detection_status": detection_status,
            "inference_time_ms": round(t_infer_ms, 2),
            "gt_status": "no_gt_single_cow_crop",
            "target_gt_bbox": "NA",
            "target_best_iou": None,
            "target_overlap_flag": None,
            "matched_detection_conf": None,
            "max_box_area_ratio": round(max_box_area_ratio, 4),
            "no_detection_flag": (num_cows == 0),
        }
        results.append((res_dict, overlay_bytes))
        print(f"  [Beef {idx+1}/{len(beef_records)}] {sample_id} | {beh_canon} | Dets: {num_cows} | Area: {max_box_area_ratio:.2f} ({t_infer_ms:.1f}ms)")

    print(f"[MODAL-BEEF] Completed {len(results)} samples.")
    return results


def select_deterministic_samples(train_csv_path, seed=2026):
    """
    Selects 25 CVB (5 per class x 5 classes) and 20 Beef (5 per class x 4 classes)
    samples deterministically from train.csv, maximizing source_video_id and session_id diversity.
    """
    df = pd.read_csv(train_csv_path)
    rng = np.random.RandomState(seed)

    # 1. CVB: 25 samples
    cvb_classes = ["Standing", "Lying", "Feeding", "Drinking", "Walking"]
    cvb_rows = []
    cvb_df = df[df["dataset"] == "cvb"].copy()

    for cls in sorted(cvb_classes):
        cls_df = cvb_df[cvb_df["behavior_canonical"] == cls].copy()
        unique_vids = sorted(cls_df["source_video_id"].unique())
        perm_vids = rng.permutation(unique_vids)
        picked_for_cls = []
        vid_idx = 0
        while len(picked_for_cls) < 5:
            vid = perm_vids[vid_idx % len(perm_vids)]
            vid_samples = cls_df[cls_df["source_video_id"] == vid]
            already_picked = [p["sample_id"] for p in picked_for_cls]
            avail = vid_samples[~vid_samples["sample_id"].isin(already_picked)]
            if len(avail) > 0:
                row = avail.sample(1, random_state=rng.randint(0, 1000000)).iloc[0]
                picked_for_cls.append(row)
            vid_idx += 1
        cvb_rows.extend(picked_for_cls)

    # 2. Beef: 20 samples
    beef_classes = ["Standing", "Lying", "Feeding", "Drinking"]
    beef_rows = []
    beef_df = df[df["dataset"] == "beef_cattle_behavior"].copy()

    for cls in sorted(beef_classes):
        cls_df = beef_df[beef_df["behavior_canonical"] == cls].copy()
        unique_sids = sorted(cls_df["session_id"].unique())
        perm_sids = rng.permutation(unique_sids)
        picked_for_cls = []
        sid_idx = 0
        while len(picked_for_cls) < 5:
            sid = perm_sids[sid_idx % len(perm_sids)]
            sid_samples = cls_df[cls_df["session_id"] == sid]
            already_picked = [p["sample_id"] for p in picked_for_cls]
            avail = sid_samples[~sid_samples["sample_id"].isin(already_picked)]
            if len(avail) > 0:
                row = avail.sample(1, random_state=rng.randint(0, 1000000)).iloc[0]
                picked_for_cls.append(row)
            sid_idx += 1
        beef_rows.extend(picked_for_cls)

    return pd.DataFrame(cvb_rows), pd.DataFrame(beef_rows)


def build_contact_sheet_cv2(images_data, grid_cols, grid_rows, tile_w, tile_h, title="Contact Sheet"):
    """
    Build a composite contact sheet using OpenCV for crisp font rendering.
    """
    header_h = 55
    sheet_w = grid_cols * tile_w
    sheet_h = grid_rows * tile_h + header_h

    sheet = np.full((sheet_h, sheet_w, 3), 25, dtype=np.uint8)

    # Top title bar
    cv2.rectangle(sheet, (0, 0), (sheet_w, header_h), (15, 15, 15), -1)
    cv2.putText(sheet, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for idx, (res, img_bytes) in enumerate(images_data):
        if idx >= grid_cols * grid_rows:
            break
        r = idx // grid_cols
        c = idx % grid_cols
        x_pos = c * tile_w
        y_pos = header_h + r * tile_h

        img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_arr is not None:
            tile_resized = cv2.resize(img_arr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            sheet[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w] = tile_resized

        # Draw tile border
        cv2.rectangle(sheet, (x_pos, y_pos), (x_pos + tile_w - 1, y_pos + tile_h - 1), (60, 60, 60), 1)

    return sheet


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  PHASE 3 BEHAVIOR PRIMARY STACK LOCALIZATION SANITY CHECK")
    print("  Detector: RT-DETR-L (ultralytics, COCO class 19 'cow', conf >= 0.25)")
    print("  Seed: 2026 | Environment: Modal (profile tigerwood693)")
    print("=" * 80)

    train_csv = "datasets/behavior/cvb_beef/train.csv"
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing canonical train split: {train_csv}")

    # 1. Deterministic Sample Selection
    cvb_s, beef_s = select_deterministic_samples(train_csv, seed=2026)
    print(f"\n[1/4] Deterministic Sample Selection (Seed 2026):")
    print(f"  - CVB samples: {len(cvb_s)} (5 classes x 5 samples; {cvb_s['source_video_id'].nunique()} unique source videos)")
    print(f"  - Beef samples: {len(beef_s)} (4 classes x 5 samples; {beef_s['session_id'].nunique()} unique sessions)")
    print(f"  - Total samples: {len(cvb_s) + len(beef_s)}")

    # 2. Dispatch to Modal Cloud
    print("\n[2/4] Dispatching CVB audit to Modal (volume cvb-data, GPU T4)...")
    cvb_records = cvb_s.to_dict(orient="records")
    cvb_modal_results = audit_cvb_on_modal.remote(cvb_records)

    print("\n[3/4] Dispatching Beef audit to Modal (volume beef-behavior-data, GPU T4)...")
    beef_records = beef_s.to_dict(orient="records")
    beef_modal_results = audit_beef_on_modal.remote(beef_records)

    # 3. Assemble Local Deliverables
    print("\n[4/4] Assembling results, saving CSV, and generating visual contact sheets...")
    out_csv_dir = Path("artifacts/perception_audit")
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / "behavior_primary_localization_sanity.csv"

    out_assets_dir = Path("docs/audits/assets/behavior_primary_localization_sanity")
    out_assets_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    # Save individual CVB overlays
    for res, img_bytes in cvb_modal_results:
        all_rows.append(res)
        s_id = res["sample_id"]
        beh = res["canonical_behavior"]
        img_out = out_assets_dir / f"cvb_{beh}_{s_id}.jpg"
        with open(img_out, "wb") as f:
            f.write(img_bytes)

    # Save individual Beef overlays
    for res, img_bytes in beef_modal_results:
        all_rows.append(res)
        s_id = res["sample_id"]
        beh = res["canonical_behavior"]
        img_out = out_assets_dir / f"beef_{beh}_{s_id}.jpg"
        with open(img_out, "wb") as f:
            f.write(img_bytes)

    # Save CSV
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(out_csv_path, index=False)
    print(f"  ✓ Saved audit CSV ({len(df_results)} rows) to: {out_csv_path}")

    # Build CVB Contact Sheet (5 cols x 5 rows = 25 tiles)
    cvb_sheet = build_contact_sheet_cv2(
        cvb_modal_results,
        grid_cols=5,
        grid_rows=5,
        tile_w=480,
        tile_h=270,
        title="CVB RT-DETR-L Localization Sanity (25 Samples: 5 per Class | Green: Target GT, Cyan: Matched Det, Orange: Other Dets)",
    )
    cvb_sheet_path = out_assets_dir / "cvb_localization_sanity_contact_sheet.jpg"
    cv2.imwrite(str(cvb_sheet_path), cvb_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Saved CVB Contact Sheet ({cvb_sheet.shape[1]}x{cvb_sheet.shape[0]}) to: {cvb_sheet_path}")

    # Build Beef Contact Sheet (5 cols x 4 rows = 20 tiles)
    beef_sheet = build_contact_sheet_cv2(
        beef_modal_results,
        grid_cols=5,
        grid_rows=4,
        tile_w=320,
        tile_h=320,
        title="Kaggle Beef RT-DETR-L Cow Detection Sanity (20 Samples: 5 per Class | Yellow: RT-DETR Cow Box on 224x224 Crop)",
    )
    beef_sheet_path = out_assets_dir / "beef_localization_sanity_contact_sheet.jpg"
    cv2.imwrite(str(beef_sheet_path), beef_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Saved Beef Contact Sheet ({beef_sheet.shape[1]}x{beef_sheet.shape[0]}) to: {beef_sheet_path}")

    # Print summary metrics to console
    print("\n" + "=" * 80)
    print("  AUDIT SUMMARY METRICS")
    print("=" * 80)

    cvb_df = df_results[df_results["dataset"] == "cvb"]
    beef_df = df_results[df_results["dataset"] == "beef_cattle_behavior"]

    print("\n--- CVB (Target Cow Localization) ---")
    cvb_gt_avail = cvb_df[cvb_df["gt_status"] == "available"]
    print(f"GT Availability: {len(cvb_gt_avail)} / {len(cvb_df)} ({len(cvb_gt_avail)/len(cvb_df)*100:.1f}%)")
    if len(cvb_gt_avail) > 0:
        mean_iou = cvb_gt_avail["target_best_iou"].mean()
        median_iou = cvb_gt_avail["target_best_iou"].median()
        overlap_cnt = cvb_gt_avail["target_overlap_flag"].sum()
        iou50_cnt = (cvb_gt_avail["target_best_iou"] >= 0.50).sum()
        print(f"Target Cow Overlap (IoU > 0.0): {overlap_cnt} / {len(cvb_gt_avail)} ({overlap_cnt/len(cvb_gt_avail)*100:.1f}%)")
        print(f"Target Cow Localization Hit (IoU >= 0.50): {iou50_cnt} / {len(cvb_gt_avail)} ({iou50_cnt/len(cvb_gt_avail)*100:.1f}%)")
        print(f"Target Cow Best IoU: Mean = {mean_iou:.4f}, Median = {median_iou:.4f}")
        print("\nPer-Class Target IoU Summary (CVB):")
        for beh, b_df in cvb_gt_avail.groupby("canonical_behavior"):
            b_overlap = b_df["target_overlap_flag"].sum()
            b_hit = (b_df["target_best_iou"] >= 0.50).sum()
            print(f"  - {beh:10s}: Mean IoU = {b_df['target_best_iou'].mean():.4f}, Overlap = {b_overlap}/{len(b_df)}, Hit@0.5 = {b_hit}/{len(b_df)}")

    print("\n--- Kaggle Beef (Single-Cow Crop Detection) ---")
    beef_det_cnt = (beef_df["num_cattle_detections"] > 0).sum()
    beef_no_det = beef_df["no_detection_flag"].sum()
    print(f"Cattle Detection Rate: {beef_det_cnt} / {len(beef_df)} ({beef_det_cnt/len(beef_df)*100:.1f}%)")
    print(f"No-Detection Count:    {beef_no_det} / {len(beef_df)} ({beef_no_det/len(beef_df)*100:.1f}%)")
    print(f"Box Area Fraction:     Mean = {beef_df['max_box_area_ratio'].mean():.4f}, Min = {beef_df['max_box_area_ratio'].min():.4f}, Max = {beef_df['max_box_area_ratio'].max():.4f}")
    print("\nPer-Class Detection Summary (Beef):")
    for beh, b_df in beef_df.groupby("canonical_behavior"):
        b_det = (b_df["num_cattle_detections"] > 0).sum()
        print(f"  - {beh:10s}: Detected = {b_det}/{len(b_df)}, Mean Area Ratio = {b_df['max_box_area_ratio'].mean():.4f}")

    print("\n✓ Sanity check run complete!")


if __name__ == "__main__":
    main()
