# -*- coding: utf-8 -*-
"""
Kaggle Beef SAM 2.1 Prompt Comparison: A4 vs A5 (Fresh 40 Canonical Train Samples)

Compares ONLY two detector-guided prompt conditions:
- A4: RT-DETR-L largest cow box -> SAM 2.1 Small
- A5: RT-DETR-L largest cow box + positive center point -> SAM 2.1 Small

Experimental Setup:
- 40 NEW samples from canonical TRAIN only (datasets/behavior/cvb_beef/train.csv)
  * 10 Drinking
  * 10 Feeding
  * 10 Lying
  * 10 Standing
- Excludes all 20 Beef samples used in previous audits
- Seed = 2026
- Maximize unique sessions (Drinking: 10, Feeding: 10, Lying: 4 [max available], Standing: 10 => 29 unique sessions)
- Midpoint frame (mid_f = n_frames // 2)
- Same 40 frames for A4 and A5
- No val/test frames inspected
- No training or fine-tuning

Models & Cloud Environment:
- RT-DETR-L: rtdetr-l.pt (COCO class 19 cow, conf >= 0.25)
- SAM 2.1 Small: sam2.1_s.pt
- Modal Profile: tigerwood693
- Volume: beef-behavior-data (/data/beef_behavior)

Deliverables:
- scripts/audit_beef_A4_A5_fresh40.py
- artifacts/perception_audit/beef_A4_A5_fresh40.csv (80 evaluation rows)
- docs/audits/assets/beef_A4_A5_fresh40/contact_sheet.jpg
- Individual 3-panel composites in docs/audits/assets/beef_A4_A5_fresh40/
- docs/research_log/2026-09-23_beef_A4_A5_fresh40.md
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

app = modal.App("beef-a4-a5-fresh40", image=image)


def make_3panel_composite(
    frame_bgr,
    sample_id,
    beh_canon,
    session_id,
    # A4 info
    largest_box_a4,
    largest_conf_a4,
    mask_a4,
    ret_a4,
    st_a4,
    ar_a4,
    cc_a4,
    # A5 info
    center_pt_a5,
    mask_a5,
    ret_a5,
    st_a5,
    ar_a5,
    cc_a5,
    # RT-DETR status
    rtdetr_failure,
    sample_idx=1,
):
    """
    Generate 3-panel side-by-side comparison composite:
    [Original | A4 Overlay | A5 Overlay]
    Composite dimensions: (252, 672, 3) (Header 28px + Panels 224x224)
    """
    img_h, img_w = frame_bgr.shape[:2]

    # Panel 1: Original
    p1 = frame_bgr.copy()
    cv2.putText(p1, "Original", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p1, f"224x224 RGB", (8, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel 2: A4 Overlay (RT-DETR Largest Box -> SAM 2.1)
    p2 = frame_bgr.copy()
    if rtdetr_failure or largest_box_a4 is None:
        cv2.putText(p2, "NO RT-DETR DET", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(p2, "A4: Box Failed", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        bx1, by1, bx2, by2 = [int(round(c)) for c in largest_box_a4]
        # Inset if border-flush
        if bx1 == 0 and by1 == 0 and bx2 >= img_w - 1 and by2 >= img_h - 1:
            bx1, by1, bx2, by2 = 1, 1, img_w - 2, img_h - 2
        cv2.rectangle(p2, (bx1, by1), (bx2, by2), (255, 255, 0), 2)  # Cyan box
        cv2.putText(p2, f"RT-DETR {largest_conf_a4:.2f}", (max(2, bx1), max(14, by1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)

        if ret_a4 and mask_a4 is not None and np.sum(mask_a4) > 0:
            overlay = p2.copy()
            overlay[mask_a4] = (overlay[mask_a4] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
            p2 = overlay
            contours, _ = cv2.findContours(mask_a4.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p2, contours, -1, (0, 255, 0), 1)
            cv2.putText(p2, f"ar:{ar_a4:.2f} cc:{cc_a4}", (8, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(p2, "NO SAM MASK", (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(p2, "A4: Box -> SAM", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 3: A5 Overlay (RT-DETR Largest Box + Positive Center Point -> SAM 2.1)
    p3 = frame_bgr.copy()
    if rtdetr_failure or largest_box_a4 is None:
        cv2.putText(p3, "NO RT-DETR DET", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(p3, "A5: Prompt Failed", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        bx1, by1, bx2, by2 = [int(round(c)) for c in largest_box_a4]
        if bx1 == 0 and by1 == 0 and bx2 >= img_w - 1 and by2 >= img_h - 1:
            bx1, by1, bx2, by2 = 1, 1, img_w - 2, img_h - 2
        cv2.rectangle(p3, (bx1, by1), (bx2, by2), (255, 255, 0), 2)  # Cyan box

        # Center point
        if center_pt_a5 is not None:
            cx, cy = int(round(center_pt_a5[0])), int(round(center_pt_a5[1]))
            cv2.circle(p3, (cx, cy), 5, (0, 255, 255), -1)  # Yellow circle
            cv2.circle(p3, (cx, cy), 5, (0, 0, 0), 1)        # Black ring
            cv2.circle(p3, (cx, cy), 1, (255, 255, 255), -1)  # White center dot
            cv2.putText(p3, f"RT-DETR+Pt", (max(2, bx1), max(14, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        if ret_a5 and mask_a5 is not None and np.sum(mask_a5) > 0:
            overlay = p3.copy()
            overlay[mask_a5] = (overlay[mask_a5] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
            p3 = overlay
            contours, _ = cv2.findContours(mask_a5.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p3, contours, -1, (0, 255, 0), 1)
            cv2.putText(p3, f"ar:{ar_a5:.2f} cc:{cc_a5}", (8, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(p3, "NO SAM MASK", (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(p3, "A5: Box+Pt -> SAM", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Border lines between panels
    cv2.line(p1, (img_w - 1, 0), (img_w - 1, img_h), (50, 50, 50), 1)
    cv2.line(p2, (img_w - 1, 0), (img_w - 1, img_h), (50, 50, 50), 1)

    # Combine 3 panels horizontally
    panels = np.hstack([p1, p2, p3])

    # Top title bar
    header_h = 28
    header = np.full((header_h, panels.shape[1], 3), 25, dtype=np.uint8)
    if rtdetr_failure:
        title_text = f"#{sample_idx:02d} | {beh_canon} | {sample_id} | Sess: {session_id} | [RT-DETR 0 DETECTIONS]"
        title_color = (100, 100, 255)  # Light Red
    else:
        title_text = (
            f"#{sample_idx:02d} | {beh_canon} | {sample_id} | Sess: {session_id} | "
            f"A4: ar={ar_a4:.2f}, cc={cc_a4} | A5: ar={ar_a5:.2f}, cc={cc_a5}"
        )
        title_color = (255, 255, 255)

    cv2.putText(header, title_text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.41, title_color, 1, cv2.LINE_AA)

    return np.vstack([header, panels])


@app.function(
    volumes={"/data/beef_behavior": beef_vol},
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def compare_beef_a4_a5_modal(beef_records):
    """
    Run RT-DETR-L localization and SAM 2.1 segmentation for A4 vs A5 on 40 fresh Kaggle Beef frames.
    """
    import os
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM, RTDETR

    print(f"[MODAL-BEEF-A4-A5] Initializing SAM 2.1 and RT-DETR-L on Tesla T4...")
    print(f"[MODAL-BEEF-A4-A5] CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

    rtdetr_model = RTDETR("rtdetr-l.pt")
    sam_model = SAM("sam2.1_s.pt")
    beef_root = "/data/beef_behavior"

    results = []

    for idx, rec in enumerate(beef_records):
        sample_id = rec["sample_id"]
        session_id = str(rec["session_id"])
        source_vid = str(rec["source_video_id"])
        beh_canon = rec["canonical_behavior"]
        rel_path = rec["source_path"]
        mid_f = int(rec["frame_index"])

        # Locate video file
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

        # -------------------------------------------------------------
        # 1. Run RT-DETR-L on frame
        # -------------------------------------------------------------
        t_rt0 = time.perf_counter()
        rt_preds = rtdetr_model(frame_bgr, conf=0.25, classes=[19], verbose=False, device="cuda")
        t_rt_ms = (time.perf_counter() - t_rt0) * 1000.0

        boxes_obj = rt_preds[0].boxes
        num_cattle_dets = len(boxes_obj)
        largest_box = None
        largest_conf = None
        largest_area_ratio = None
        rtdetr_failure = False

        if num_cattle_dets > 0:
            xyxy_arr = boxes_obj.xyxy.cpu().numpy()
            conf_arr = boxes_obj.conf.cpu().numpy()
            areas = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) * (xyxy_arr[:, 3] - xyxy_arr[:, 1])
            best_idx = int(np.argmax(areas))
            largest_box = [round(float(c), 1) for c in xyxy_arr[best_idx].tolist()]
            largest_conf = round(float(conf_arr[best_idx]), 4)
            largest_area_ratio = round(float(areas[best_idx]) / (img_w * img_h), 4)
        else:
            rtdetr_failure = True

        # Helper: Execute SAM Inference
        def run_sam_inference(bboxes=None, points=None, labels=None):
            t0 = time.perf_counter()
            try:
                kwargs = {"device": "cuda", "verbose": False}
                if bboxes is not None:
                    kwargs["bboxes"] = bboxes
                if points is not None:
                    kwargs["points"] = points
                if labels is not None:
                    kwargs["labels"] = labels

                res = sam_model(frame_bgr, **kwargs)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                if res[0].masks is not None and len(res[0].masks.data) > 0:
                    raw_mask = res[0].masks.data[0].cpu().numpy().astype(bool)
                    if raw_mask.shape != (img_h, img_w):
                        mask = cv2.resize(raw_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask = raw_mask

                    pix = int(np.sum(mask))
                    if pix > 0:
                        cc = int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)
                        ar = pix / (img_w * img_h)
                        return mask, True, "segmented", round(lat_ms, 2), pix, round(ar, 4), cc
                    else:
                        return np.zeros((img_h, img_w), dtype=bool), False, "sam_no_mask", round(lat_ms, 2), 0, 0.0, 0
                else:
                    return np.zeros((img_h, img_w), dtype=bool), False, "sam_no_mask", round(lat_ms, 2), 0, 0.0, 0
            except Exception as e:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return np.zeros((img_h, img_w), dtype=bool), False, f"error_{str(e)[:25]}", round(lat_ms, 2), 0, 0.0, 0

        # -------------------------------------------------------------
        # CONDITION A4: Largest RT-DETR-L Cow Box
        # -------------------------------------------------------------
        if largest_box is None:
            mask_a4, ret_a4, st_a4, lat_a4, pix_a4, ar_a4, cc_a4 = (
                np.zeros((img_h, img_w), dtype=bool), False, "no_rtdetr_prompt", 0.0, 0, 0.0, 0
            )
        else:
            mask_a4, ret_a4, st_a4, lat_a4, pix_a4, ar_a4, cc_a4 = run_sam_inference(bboxes=[largest_box])

        rec_a4 = {
            "sample_id": sample_id,
            "canonical_behavior": beh_canon,
            "session_id": session_id,
            "source_video_id": source_vid,
            "frame_index": mid_f,
            "prompt_condition": "A4_largest_rtdetr_box",
            "exact_prompt_coordinates": json.dumps({"box": largest_box}) if largest_box else json.dumps(None),
            "mask_returned": ret_a4,
            "mask_status": st_a4,
            "inference_time_ms": lat_a4,
            "mask_area_ratio": ar_a4,
            "number_of_connected_components": cc_a4,
            "number_of_cow_detections": num_cattle_dets,
            "rtdetr_failure": rtdetr_failure,
            "selected_box": json.dumps(largest_box) if largest_box else None,
            "selected_box_confidence": largest_conf,
            "selected_box_area_ratio": largest_area_ratio,
        }

        # -------------------------------------------------------------
        # CONDITION A5: Largest RT-DETR Box + Center Positive Point
        # -------------------------------------------------------------
        center_pt_a5 = None
        if largest_box is None:
            mask_a5, ret_a5, st_a5, lat_a5, pix_a5, ar_a5, cc_a5 = (
                np.zeros((img_h, img_w), dtype=bool), False, "no_rtdetr_prompt", 0.0, 0, 0.0, 0
            )
            coords_a5 = None
        else:
            box_cx = round(float((largest_box[0] + largest_box[2]) / 2.0), 1)
            box_cy = round(float((largest_box[1] + largest_box[3]) / 2.0), 1)
            center_pt_a5 = [box_cx, box_cy]
            pts_a5_raw = [[box_cx, box_cy]]
            pts_a5 = np.array([pts_a5_raw], dtype=np.float32)  # shape (1, 1, 2)
            lbl_a5 = np.array([[1]], dtype=np.int32)           # shape (1, 1)

            mask_a5, ret_a5, st_a5, lat_a5, pix_a5, ar_a5, cc_a5 = run_sam_inference(
                bboxes=[largest_box], points=pts_a5, labels=lbl_a5
            )
            coords_a5 = {"box": largest_box, "points": pts_a5_raw, "labels": [1]}

        rec_a5 = {
            "sample_id": sample_id,
            "canonical_behavior": beh_canon,
            "session_id": session_id,
            "source_video_id": source_vid,
            "frame_index": mid_f,
            "prompt_condition": "A5_rtdetr_box_center_point",
            "exact_prompt_coordinates": json.dumps(coords_a5) if coords_a5 else json.dumps(None),
            "mask_returned": ret_a5,
            "mask_status": st_a5,
            "inference_time_ms": lat_a5,
            "mask_area_ratio": ar_a5,
            "number_of_connected_components": cc_a5,
            "number_of_cow_detections": num_cattle_dets,
            "rtdetr_failure": rtdetr_failure,
            "selected_box": json.dumps(largest_box) if largest_box else None,
            "selected_box_confidence": largest_conf,
            "selected_box_area_ratio": largest_area_ratio,
        }

        # -------------------------------------------------------------
        # Generate 3-Panel Composite: [Original | A4 Overlay | A5 Overlay]
        # -------------------------------------------------------------
        comp = make_3panel_composite(
            frame_bgr=frame_bgr,
            sample_id=sample_id,
            beh_canon=beh_canon,
            session_id=session_id,
            largest_box_a4=largest_box,
            largest_conf_a4=largest_conf,
            mask_a4=mask_a4,
            ret_a4=ret_a4,
            st_a4=st_a4,
            ar_a4=ar_a4,
            cc_a4=cc_a4,
            center_pt_a5=center_pt_a5,
            mask_a5=mask_a5,
            ret_a5=ret_a5,
            st_a5=st_a5,
            ar_a5=ar_a5,
            cc_a5=cc_a5,
            rtdetr_failure=rtdetr_failure,
            sample_idx=idx + 1,
        )

        _, buf = cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
        comp_bytes = buf.tobytes()

        results.append({
            "sample_id": sample_id,
            "canonical_behavior": beh_canon,
            "session_id": session_id,
            "sample_idx": idx + 1,
            "record_a4": rec_a4,
            "record_a5": rec_a5,
            "comp_bytes": comp_bytes,
            "rtdetr_failure": rtdetr_failure,
        })

        fail_tag = " [RT-DETR FAILED]" if rtdetr_failure else ""
        print(f"  [Sample {idx+1:02d}/40] {sample_id} ({beh_canon:8s}) -> A4: ret={ret_a4} (ar={ar_a4:.2f}, cc={cc_a4}) | A5: ret={ret_a5} (ar={ar_a5:.2f}, cc={cc_a5}){fail_tag}")

    print(f"[MODAL-BEEF-A4-A5] Finished evaluating {len(results)} samples.")
    return results


def build_master_contact_sheet(
    sample_composites_by_behavior,
    tile_w=672,
    tile_h=252,
    header_h=100,
):
    """
    Build 4-column x 10-row master contact sheet.
    Columns correspond to the 4 canonical behaviors:
    Col 0: Drinking (10 samples)
    Col 1: Feeding (10 samples)
    Col 2: Lying (10 samples)
    Col 3: Standing (10 samples)

    Each tile is the 3-panel composite (Original | A4 Overlay | A5 Overlay).
    Total dimensions: (10 * tile_h + header_h, 4 * tile_w, 3) = (2620, 2688, 3)
    """
    target_behaviors = ["Drinking", "Feeding", "Lying", "Standing"]
    grid_cols = len(target_behaviors)  # 4
    grid_rows = 10                     # 10

    sheet_w = grid_cols * tile_w
    sheet_h = grid_rows * tile_h + header_h

    sheet = np.full((sheet_h, sheet_w, 3), 20, dtype=np.uint8)

    # Master Title Banner (height 62px)
    cv2.rectangle(sheet, (0, 0), (sheet_w, 62), (12, 12, 12), -1)
    cv2.putText(
        sheet,
        "Kaggle Beef SAM 2.1: A4 (RT-DETR Box) vs A5 (RT-DETR Box + Center Point)",
        (24, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "40 Fresh Canonical TRAIN Samples (10/class) | Seed 2026 | Modal tigerwood693 | Status: PENDING HUMAN REVIEW",
        (24, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 230, 255),  # Yellow-cyan
        1,
        cv2.LINE_AA,
    )

    # Column Sub-headers (y from 62 to header_h = 100)
    cv2.rectangle(sheet, (0, 62), (sheet_w, header_h), (28, 28, 28), -1)
    for col_idx, beh in enumerate(target_behaviors):
        x_start = col_idx * tile_w
        x_end = (col_idx + 1) * tile_w
        col_label = f"COLUMN {col_idx+1}: {beh.upper()} (10 Samples)  [Original | A4 Box | A5 Box+Pt]"
        cv2.putText(
            sheet,
            col_label,
            (x_start + 14, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if col_idx > 0:
            cv2.line(sheet, (x_start, 62), (x_start, sheet_h), (60, 60, 60), 2)

    # Place tiles in grid
    for col_idx, beh in enumerate(target_behaviors):
        items = sample_composites_by_behavior.get(beh, [])
        for row_idx, comp_bytes in enumerate(items[:grid_rows]):
            x_pos = col_idx * tile_w
            y_pos = header_h + row_idx * tile_h

            img_arr = cv2.imdecode(np.frombuffer(comp_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_arr is not None:
                if img_arr.shape[1] != tile_w or img_arr.shape[0] != tile_h:
                    tile_resized = cv2.resize(img_arr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                else:
                    tile_resized = img_arr
                sheet[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w] = tile_resized

            # Tile border
            cv2.rectangle(sheet, (x_pos, y_pos), (x_pos + tile_w - 1, y_pos + tile_h - 1), (50, 50, 50), 1)

    return sheet


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  KAGGLE BEEF SAM 2.1 PROMPT COMPARISON: ONLY A4 vs A5")
    print("  A4: RT-DETR-L largest cow box -> SAM 2.1")
    print("  A5: RT-DETR-L largest cow box + positive center point -> SAM 2.1")
    print("  N = 40 Fresh Canonical TRAIN Samples (10 Drinking, 10 Feeding, 10 Lying, 10 Standing)")
    print("  Requirements: Exclude 20 prior Beef samples | Seed 2026 | Maximize sessions")
    print("  Environment: Modal Cloud (profile tigerwood693, GPU Tier: T4)")
    print("  Status: PENDING HUMAN REVIEW")
    print("=" * 80)

    train_csv = "datasets/behavior/cvb_beef/train.csv"
    prior_sanity_csv = "artifacts/perception_audit/behavior_primary_localization_sanity.csv"

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not os.path.exists(prior_sanity_csv):
        raise FileNotFoundError(f"Missing prior sanity CSV: {prior_sanity_csv}")

    # 1. Load train dataset and filter beef
    df_train = pd.read_csv(train_csv)
    beef_df = df_train[df_train["dataset"] == "beef_cattle_behavior"].copy()

    # 2. Identify 20 prior beef samples to exclude
    df_prior = pd.read_csv(prior_sanity_csv)
    prior_beef_ids = set(df_prior[df_prior["dataset"] == "beef_cattle_behavior"]["sample_id"])
    print(f"\n[1/4] Found {len(prior_beef_ids)} prior Beef samples to exclude:")
    print(f"      {sorted(list(prior_beef_ids))[:5]} ... ({len(prior_beef_ids)} total)")

    candidate_df = beef_df[~beef_df["sample_id"].isin(prior_beef_ids)].copy()
    print(f"      Remaining candidate train pool: {len(candidate_df)} samples")

    # 3. Deterministic sampling: seed = 2026, maximize unique sessions
    seed = 2026
    rng = np.random.RandomState(seed)
    target_classes = ["Drinking", "Feeding", "Lying", "Standing"]
    picked_rows = []

    for cls in sorted(target_classes):
        cls_df = candidate_df[candidate_df["behavior_canonical"] == cls].copy()
        unique_sids = sorted(cls_df["session_id"].unique())
        perm_sids = rng.permutation(unique_sids)
        picked_for_cls = []
        sid_idx = 0
        while len(picked_for_cls) < 10:
            sid = perm_sids[sid_idx % len(perm_sids)]
            sid_samples = cls_df[cls_df["session_id"] == sid]
            already_picked = [p["sample_id"] for p in picked_for_cls]
            avail = sid_samples[~sid_samples["sample_id"].isin(already_picked)]
            if len(avail) > 0:
                row = avail.sample(1, random_state=rng.randint(0, 1000000)).iloc[0]
                picked_for_cls.append(row)
            sid_idx += 1
        picked_rows.extend(picked_for_cls)

    sample_df = pd.DataFrame(picked_rows)
    assert len(sample_df) == 40, f"Expected 40 samples, got {len(sample_df)}"
    assert len(set(sample_df["sample_id"]).intersection(prior_beef_ids)) == 0, "Overlap detected!"

    # Compute midpoint frame index for each sample
    sample_df["frame_index"] = sample_df["n_frames"] // 2
    sample_df["canonical_behavior"] = sample_df["behavior_canonical"]

    print("\n[2/4] Selected 40 FRESH Beef samples (Seed 2026):")
    for cls in target_classes:
        sub = sample_df[sample_df["behavior_canonical"] == cls]
        print(f"  - {cls:<10}: {len(sub)} samples across {sub['session_id'].nunique()} unique sessions")
    print(f"  Total unique sessions: {sample_df['session_id'].nunique()} / 40")

    beef_records = sample_df.to_dict(orient="records")

    # 4. Dispatch to Modal
    print("\n[3/4] Dispatching A4 vs A5 comparison to Modal (profile tigerwood693, GPU Tier T4)...")
    modal_output = compare_beef_a4_a5_modal.remote(beef_records)

    # 5. Assemble deliverables
    print("\n[4/4] Assembling results, writing CSV, saving individual composites & master contact sheet...")
    out_csv_dir = Path("artifacts/perception_audit")
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / "beef_A4_A5_fresh40.csv"

    out_assets_dir = Path("docs/audits/assets/beef_A4_A5_fresh40")
    out_assets_dir.mkdir(parents=True, exist_ok=True)

    all_eval_rows = []
    composites_by_behavior = {
        "Drinking": [],
        "Feeding": [],
        "Lying": [],
        "Standing": [],
    }

    rtdetr_failures = []

    for item in modal_output:
        sample_id = item["sample_id"]
        beh = item["canonical_behavior"]
        sample_idx = item["sample_idx"]
        rec_a4 = item["record_a4"]
        rec_a5 = item["record_a5"]
        comp_bytes = item["comp_bytes"]
        has_rtdetr_fail = item["rtdetr_failure"]

        all_eval_rows.append(rec_a4)
        all_eval_rows.append(rec_a5)

        if has_rtdetr_fail:
            rtdetr_failures.append({
                "sample_id": sample_id,
                "behavior": beh,
                "session_id": item["session_id"],
                "sample_idx": sample_idx,
            })

        # Save individual 3-panel composite
        comp_filename = f"{sample_idx:02d}_{beh}_{sample_id}.jpg"
        with open(out_assets_dir / comp_filename, "wb") as f:
            f.write(comp_bytes)

        composites_by_behavior[beh].append(comp_bytes)

    # Save evaluation CSV (80 rows)
    df_results = pd.DataFrame(all_eval_rows)
    df_results.to_csv(out_csv_path, index=False)
    print(f"  ✓ Saved evaluation CSV ({len(df_results)} rows) to: {out_csv_path}")

    # Build master contact sheet
    master_sheet = build_master_contact_sheet(
        sample_composites_by_behavior=composites_by_behavior,
        tile_w=672,
        tile_h=252,
        header_h=100,
    )
    master_sheet_path = out_assets_dir / "contact_sheet.jpg"
    cv2.imwrite(str(master_sheet_path), master_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Saved Master Contact Sheet ({master_sheet.shape[1]}x{master_sheet.shape[0]} px) to: {master_sheet_path}")

    # Summary Statistics
    print("\n" + "=" * 80)
    print("  KAGGLE BEEF A4 vs A5 FRESH-40 EVALUATION SUMMARY")
    print("=" * 80)

    for cond_id in ["A4_largest_rtdetr_box", "A5_rtdetr_box_center_point"]:
        cond_df = df_results[df_results["prompt_condition"] == cond_id]
        n_tot = len(cond_df)
        n_ret = cond_df["mask_returned"].sum()
        ret_rate = (n_ret / n_tot) * 100.0 if n_tot > 0 else 0.0

        succ = cond_df[cond_df["mask_returned"]]
        avg_ar = succ["mask_area_ratio"].mean() if len(succ) > 0 else 0.0
        med_ar = succ["mask_area_ratio"].median() if len(succ) > 0 else 0.0
        avg_cc = succ["number_of_connected_components"].mean() if len(succ) > 0 else 0.0
        med_cc = succ["number_of_connected_components"].median() if len(succ) > 0 else 0.0
        avg_lat = cond_df["inference_time_ms"].mean()

        print(f"\nCondition: {cond_id}")
        print(f"  Total Samples: {n_tot}")
        print(f"  Masks Returned: {n_ret} / {n_tot} ({ret_rate:.1f}%)")
        print(f"  Per-Behavior Return Rates:")
        for b in target_classes:
            b_df = cond_df[cond_df["canonical_behavior"] == b]
            b_ret = b_df["mask_returned"].sum()
            print(f"    - {b:<10}: {b_ret} / {len(b_df)} ({b_ret/len(b_df)*100:.1f}%)")
        print(f"  Mask Area Ratio (Returned): Mean = {avg_ar:.4f}, Median = {med_ar:.4f}")
        print(f"  Connected Components (Returned): Mean = {avg_cc:.1f}, Median = {med_cc:.1f}")
        print(f"  Mean Inference Latency: {avg_lat:.1f} ms")

    print("\n" + "-" * 80)
    print(f"  RT-DETR-L Localization Failures (0 cattle detections): {len(rtdetr_failures)} / 40 ({len(rtdetr_failures)/40*100:.1f}%)")
    if rtdetr_failures:
        for f_item in rtdetr_failures:
            print(f"    * Sample #{f_item['sample_idx']:02d}: {f_item['sample_id']} (Behavior: {f_item['behavior']}, Session: {f_item['session_id']})")
    else:
        print("    * None! RT-DETR-L detected at least one cow in 100% of samples.")
    print("-" * 80)

    print("\n================================================================================")
    print("  FINAL STATUS: PENDING HUMAN REVIEW")
    print("  Note: No winner has been automatically selected.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
