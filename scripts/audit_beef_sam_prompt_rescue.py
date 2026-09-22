# -*- coding: utf-8 -*-
"""
Kaggle Beef SAM 2.1 Prompt-Rescue Audit
Controlled evaluation testing whether alternative prompt strategies rescue SAM 2.1
on the SAME 20 Kaggle Beef training frames used in the segmentation sanity check.

Conditions Evaluated:
- A0: Existing baseline (Full crop box [0, 0, 223, 223])
- A1: Center positive point (cx, cy) = (112, 112)
- A2: Multi-positive body points (5 deterministic points)
- A3: Positive center (112, 112) + 4 negative corners (18, 18), (206, 18), (18, 206), (206, 206)
- A4: Largest RT-DETR-L cow box
- A5: Largest RT-DETR-L cow box + center positive point

Execution:
Modal cloud (profile tigerwood693, GPU Tier T4)
Volume: beef-behavior-data (/data/beef_behavior)
Checkpoints: sam2.1_s.pt, rtdetr-l.pt
Sample Provenance: Reused from artifacts/perception_audit/behavior_primary_localization_sanity.csv

Deliverables:
- scripts/audit_beef_sam_prompt_rescue.py
- artifacts/perception_audit/beef_sam_prompt_rescue.csv
- docs/audits/assets/beef_sam_prompt_rescue/
- docs/research_log/2026-09-23_beef_sam_prompt_rescue.md
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

app = modal.App("beef-sam-prompt-rescue", image=image)


def make_4panel_composite(
    img_bgr,
    condition_id,
    condition_name,
    prompt_box=None,
    pos_points=None,
    neg_points=None,
    pred_mask=None,
    title_text="",
    box_label="Prompt Box",
    box_color=(255, 255, 0),   # Cyan
    pos_color=(0, 255, 0),      # Bright Green
    neg_color=(0, 0, 255),      # Bright Red
):
    """
    Generate 4-panel visual inspection composite:
    [Original | Prompt Visualization | SAM 2.1 Mask | Overlay]
    Composite shape: (260, 896, 3) (Header 36px + Panels 224px)
    """
    h, w = img_bgr.shape[:2]

    # Panel 1: Original
    p1 = img_bgr.copy()
    cv2.putText(p1, "Original", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 2: Prompt Visualization
    p2 = img_bgr.copy()
    if prompt_box is not None:
        bx1, by1, bx2, by2 = [int(round(c)) for c in prompt_box]
        # Inset full-crop box slightly for border visibility
        if bx1 == 0 and by1 == 0 and bx2 >= w - 1 and by2 >= h - 1:
            bx1, by1, bx2, by2 = 1, 1, w - 2, h - 2
        cv2.rectangle(p2, (bx1, by1), (bx2, by2), box_color, 2)
        cv2.putText(p2, box_label, (max(2, bx1), max(14, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 1, cv2.LINE_AA)

    if pos_points is not None:
        for pt in pos_points:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(p2, (px, py), 5, pos_color, -1)
            cv2.circle(p2, (px, py), 5, (0, 0, 0), 1)
            cv2.line(p2, (px - 2, py), (px + 2, py), (255, 255, 255), 1)
            cv2.line(p2, (px, py - 2), (px, py + 2), (255, 255, 255), 1)

    if neg_points is not None:
        for pt in neg_points:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.line(p2, (px - 4, py - 4), (px + 4, py + 4), neg_color, 2)
            cv2.line(p2, (px - 4, py + 4), (px + 4, py - 4), neg_color, 2)

    if condition_id in ["A4", "A5"] and prompt_box is None:
        cv2.putText(p2, "NO RT-DETR DET", (25, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(p2, "Prompt Vis", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 3: SAM Mask (Binary white on black)
    p3 = np.zeros((h, w, 3), dtype=np.uint8)
    if pred_mask is not None and np.sum(pred_mask) > 0:
        p3[pred_mask] = [255, 255, 255]
        cv2.putText(p3, "SAM 2.1 Mask", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(p3, "No Mask", (60, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)

    # Panel 4: Overlay
    p4 = img_bgr.copy()
    if pred_mask is not None and np.sum(pred_mask) > 0:
        overlay = p4.copy()
        overlay[pred_mask] = (overlay[pred_mask] * 0.4 + np.array([0, 230, 0]) * 0.6).astype(np.uint8)
        p4 = overlay
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p4, contours, -1, (0, 255, 0), 1)
        cv2.putText(p4, "Overlay", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(p4, "No Mask", (60, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Combine 4 panels horizontally
    panels = np.hstack([p1, p2, p3, p4])

    # Top title bar
    header_h = 36
    header = np.full((header_h, panels.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(header, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, panels])


@app.function(
    volumes={"/data/beef_behavior": beef_vol},
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def audit_beef_prompt_rescue_modal(beef_records):
    """
    Run SAM 2.1 prompt-rescue audit across all 6 conditions on the 20 Kaggle Beef frames.
    """
    import os
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM, RTDETR

    print(f"[MODAL-BEEF-RESCUE] Initializing SAM 2.1 and RT-DETR-L on Tesla T4...")
    print(f"[MODAL-BEEF-RESCUE] CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

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
        # STEP 1: Run RT-DETR-L on frame for A4 and A5
        # -------------------------------------------------------------
        t_rt0 = time.perf_counter()
        rt_preds = rtdetr_model(frame_bgr, conf=0.25, classes=[19], verbose=False, device="cuda")
        t_rt_ms = (time.perf_counter() - t_rt0) * 1000.0

        boxes_obj = rt_preds[0].boxes
        num_cattle_dets = len(boxes_obj)
        largest_box = None
        largest_conf = None
        largest_area_ratio = None

        if num_cattle_dets > 0:
            xyxy_arr = boxes_obj.xyxy.cpu().numpy()
            conf_arr = boxes_obj.conf.cpu().numpy()
            areas = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) * (xyxy_arr[:, 3] - xyxy_arr[:, 1])
            best_idx = int(np.argmax(areas))
            largest_box = [round(float(c), 1) for c in xyxy_arr[best_idx].tolist()]
            largest_conf = round(float(conf_arr[best_idx]), 4)
            largest_area_ratio = round(float(areas[best_idx]) / (img_w * img_h), 4)

        # -------------------------------------------------------------
        # Helper: Execute SAM Inference and extract mask/components
        # -------------------------------------------------------------
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

        sample_conditions = []

        # -------------------------------------------------------------
        # CONDITION A0: Baseline Full Crop Box [0, 0, 223, 223]
        # -------------------------------------------------------------
        box_a0 = [0.0, 0.0, float(img_w - 1), float(img_h - 1)]
        mask_a0, ret_a0, st_a0, lat_a0, pix_a0, ar_a0, cc_a0 = run_sam_inference(bboxes=[box_a0])
        title_a0 = f"Beef | {beh_canon} | {sample_id} | A0: Full Crop [0,0,223,223] | Area: {ar_a0:.3f} | CC: {cc_a0} | Lat: {lat_a0:.1f}ms"
        comp_a0 = make_4panel_composite(
            frame_bgr, "A0", "Full Crop Box", prompt_box=box_a0, pred_mask=mask_a0,
            title_text=title_a0, box_label="Full Crop [0,0,223,223]", box_color=(0, 255, 255)
        )
        _, buf_a0 = cv2.imencode(".jpg", comp_a0, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample_conditions.append({
            "record": {
                "sample_id": sample_id,
                "canonical_behavior": beh_canon,
                "session_id": session_id,
                "frame_index": mid_f,
                "prompt_condition": "A0_full_crop",
                "exact_prompt_coordinates": json.dumps({"box": box_a0}),
                "mask_returned": ret_a0,
                "mask_status": st_a0,
                "inference_time_ms": lat_a0,
                "mask_area_ratio": ar_a0,
                "number_of_connected_components": cc_a0,
                "number_of_cow_detections": num_cattle_dets,
                "selected_box": None,
                "selected_box_confidence": None,
                "selected_box_area_ratio": None,
            },
            "comp_bytes": buf_a0.tobytes(),
            "condition_id": "A0",
        })

        # -------------------------------------------------------------
        # CONDITION A1: Center Positive Point (cx, cy) = (112, 112)
        # -------------------------------------------------------------
        cx_a1, cy_a1 = float(img_w // 2), float(img_h // 2)
        pts_a1_raw = [[cx_a1, cy_a1]]
        pts_a1 = np.array([pts_a1_raw], dtype=np.float32)  # shape (1, 1, 2)
        lbl_a1 = np.array([[1]], dtype=np.int32)           # shape (1, 1)
        mask_a1, ret_a1, st_a1, lat_a1, pix_a1, ar_a1, cc_a1 = run_sam_inference(points=pts_a1, labels=lbl_a1)
        title_a1 = f"Beef | {beh_canon} | {sample_id} | A1: Center Point ({int(cx_a1)},{int(cy_a1)}) | Area: {ar_a1:.3f} | CC: {cc_a1} | Lat: {lat_a1:.1f}ms"
        comp_a1 = make_4panel_composite(
            frame_bgr, "A1", "Center Positive Point", pos_points=pts_a1_raw, pred_mask=mask_a1,
            title_text=title_a1
        )
        _, buf_a1 = cv2.imencode(".jpg", comp_a1, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample_conditions.append({
            "record": {
                "sample_id": sample_id,
                "canonical_behavior": beh_canon,
                "session_id": session_id,
                "frame_index": mid_f,
                "prompt_condition": "A1_center_point",
                "exact_prompt_coordinates": json.dumps({"points": pts_a1_raw, "labels": [1]}),
                "mask_returned": ret_a1,
                "mask_status": st_a1,
                "inference_time_ms": lat_a1,
                "mask_area_ratio": ar_a1,
                "number_of_connected_components": cc_a1,
                "number_of_cow_detections": num_cattle_dets,
                "selected_box": None,
                "selected_box_confidence": None,
                "selected_box_area_ratio": None,
            },
            "comp_bytes": buf_a1.tobytes(),
            "condition_id": "A1",
        })

        # -------------------------------------------------------------
        # CONDITION A2: Multi-Positive Body Points (5 deterministic points)
        # -------------------------------------------------------------
        pts_a2_raw = [
            [round(0.50 * img_w, 1), round(0.50 * img_h, 1)],  # Center (112, 112)
            [round(0.35 * img_w, 1), round(0.50 * img_h, 1)],  # Left-mid (78, 112)
            [round(0.65 * img_w, 1), round(0.50 * img_h, 1)],  # Right-mid (146, 112)
            [round(0.50 * img_w, 1), round(0.35 * img_h, 1)],  # Top-mid (112, 78)
            [round(0.50 * img_w, 1), round(0.65 * img_h, 1)],  # Bottom-mid (112, 146)
        ]
        pts_a2 = np.array([pts_a2_raw], dtype=np.float32)      # shape (1, 5, 2)
        lbl_a2 = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)  # shape (1, 5)
        mask_a2, ret_a2, st_a2, lat_a2, pix_a2, ar_a2, cc_a2 = run_sam_inference(points=pts_a2, labels=lbl_a2)
        title_a2 = f"Beef | {beh_canon} | {sample_id} | A2: Multi-Positive (5 Pts) | Area: {ar_a2:.3f} | CC: {cc_a2} | Lat: {lat_a2:.1f}ms"
        comp_a2 = make_4panel_composite(
            frame_bgr, "A2", "Multi-Positive Body Points", pos_points=pts_a2_raw, pred_mask=mask_a2,
            title_text=title_a2
        )
        _, buf_a2 = cv2.imencode(".jpg", comp_a2, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample_conditions.append({
            "record": {
                "sample_id": sample_id,
                "canonical_behavior": beh_canon,
                "session_id": session_id,
                "frame_index": mid_f,
                "prompt_condition": "A2_multipoint",
                "exact_prompt_coordinates": json.dumps({"points": pts_a2_raw, "labels": [1, 1, 1, 1, 1]}),
                "mask_returned": ret_a2,
                "mask_status": st_a2,
                "inference_time_ms": lat_a2,
                "mask_area_ratio": ar_a2,
                "number_of_connected_components": cc_a2,
                "number_of_cow_detections": num_cattle_dets,
                "selected_box": None,
                "selected_box_confidence": None,
                "selected_box_area_ratio": None,
            },
            "comp_bytes": buf_a2.tobytes(),
            "condition_id": "A2",
        })

        # -------------------------------------------------------------
        # CONDITION A3: Positive Center + 4 Negative Corners
        # -------------------------------------------------------------
        pos_a3 = [[round(0.50 * img_w, 1), round(0.50 * img_h, 1)]]
        neg_a3 = [
            [round(0.08 * img_w, 1), round(0.08 * img_h, 1)],  # Top-left (18, 18)
            [round(0.92 * img_w, 1), round(0.08 * img_h, 1)],  # Top-right (206, 18)
            [round(0.08 * img_w, 1), round(0.92 * img_h, 1)],  # Bottom-left (18, 206)
            [round(0.92 * img_w, 1), round(0.92 * img_h, 1)],  # Bottom-right (206, 206)
        ]
        pts_a3_raw = pos_a3 + neg_a3
        pts_a3 = np.array([pts_a3_raw], dtype=np.float32)      # shape (1, 5, 2)
        lbl_a3 = np.array([[1, 0, 0, 0, 0]], dtype=np.int32)  # shape (1, 5)
        mask_a3, ret_a3, st_a3, lat_a3, pix_a3, ar_a3, cc_a3 = run_sam_inference(points=pts_a3, labels=lbl_a3)
        title_a3 = f"Beef | {beh_canon} | {sample_id} | A3: Center Pos + 4 Neg Corners | Area: {ar_a3:.3f} | CC: {cc_a3} | Lat: {lat_a3:.1f}ms"
        comp_a3 = make_4panel_composite(
            frame_bgr, "A3", "Positive Center + Negative Corners",
            pos_points=pos_a3, neg_points=neg_a3, pred_mask=mask_a3,
            title_text=title_a3
        )
        _, buf_a3 = cv2.imencode(".jpg", comp_a3, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample_conditions.append({
            "record": {
                "sample_id": sample_id,
                "canonical_behavior": beh_canon,
                "session_id": session_id,
                "frame_index": mid_f,
                "prompt_condition": "A3_center_negcorners",
                "exact_prompt_coordinates": json.dumps({"points": pts_a3_raw, "labels": [1, 0, 0, 0, 0]}),
                "mask_returned": ret_a3,
                "mask_status": st_a3,
                "inference_time_ms": lat_a3,
                "mask_area_ratio": ar_a3,
                "number_of_connected_components": cc_a3,
                "number_of_cow_detections": num_cattle_dets,
                "selected_box": None,
                "selected_box_confidence": None,
                "selected_box_area_ratio": None,
            },
            "comp_bytes": buf_a3.tobytes(),
            "condition_id": "A3",
        })

        # -------------------------------------------------------------
        # CONDITION A4: Largest RT-DETR-L Cow Box
        # -------------------------------------------------------------
        if largest_box is None:
            mask_a4, ret_a4, st_a4, lat_a4, pix_a4, ar_a4, cc_a4 = (
                np.zeros((img_h, img_w), dtype=bool), False, "no_rtdetr_prompt", 0.0, 0, 0.0, 0
            )
            title_a4 = f"Beef | {beh_canon} | {sample_id} | A4: RT-DETR Box | [NO DETECTION] | Area: 0.0 | CC: 0"
        else:
            mask_a4, ret_a4, st_a4, lat_a4, pix_a4, ar_a4, cc_a4 = run_sam_inference(bboxes=[largest_box])
            title_a4 = f"Beef | {beh_canon} | {sample_id} | A4: RT-DETR Box (conf {largest_conf:.2f}) | Area: {ar_a4:.3f} | CC: {cc_a4} | Lat: {lat_a4:.1f}ms"

        comp_a4 = make_4panel_composite(
            frame_bgr, "A4", "Largest RT-DETR Box", prompt_box=largest_box, pred_mask=mask_a4,
            title_text=title_a4, box_label=f"RT-DETR (conf {largest_conf:.2f})" if largest_conf else "RT-DETR",
            box_color=(255, 255, 0)
        )
        _, buf_a4 = cv2.imencode(".jpg", comp_a4, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample_conditions.append({
            "record": {
                "sample_id": sample_id,
                "canonical_behavior": beh_canon,
                "session_id": session_id,
                "frame_index": mid_f,
                "prompt_condition": "A4_largest_rtdetr_box",
                "exact_prompt_coordinates": json.dumps({"box": largest_box}) if largest_box else json.dumps(None),
                "mask_returned": ret_a4,
                "mask_status": st_a4,
                "inference_time_ms": lat_a4,
                "mask_area_ratio": ar_a4,
                "number_of_connected_components": cc_a4,
                "number_of_cow_detections": num_cattle_dets,
                "selected_box": json.dumps(largest_box) if largest_box else None,
                "selected_box_confidence": largest_conf,
                "selected_box_area_ratio": largest_area_ratio,
            },
            "comp_bytes": buf_a4.tobytes(),
            "condition_id": "A4",
        })

        # -------------------------------------------------------------
        # CONDITION A5: Largest RT-DETR Box + Center Positive Point
        # -------------------------------------------------------------
        if largest_box is None:
            mask_a5, ret_a5, st_a5, lat_a5, pix_a5, ar_a5, cc_a5 = (
                np.zeros((img_h, img_w), dtype=bool), False, "no_rtdetr_prompt", 0.0, 0, 0.0, 0
            )
            title_a5 = f"Beef | {beh_canon} | {sample_id} | A5: RT-DETR Box + Center Pt | [NO DETECTION] | Area: 0.0 | CC: 0"
            comp_a5 = make_4panel_composite(
                frame_bgr, "A5", "RT-DETR Box + Center Point", prompt_box=None, pos_points=None, pred_mask=mask_a5,
                title_text=title_a5
            )
            coords_a5 = None
        else:
            box_cx = round(float((largest_box[0] + largest_box[2]) / 2.0), 1)
            box_cy = round(float((largest_box[1] + largest_box[3]) / 2.0), 1)
            pts_a5_raw = [[box_cx, box_cy]]
            pts_a5 = np.array([pts_a5_raw], dtype=np.float32)  # shape (1, 1, 2)
            lbl_a5 = np.array([[1]], dtype=np.int32)           # shape (1, 1)

            mask_a5, ret_a5, st_a5, lat_a5, pix_a5, ar_a5, cc_a5 = run_sam_inference(
                bboxes=[largest_box], points=pts_a5, labels=lbl_a5
            )
            title_a5 = f"Beef | {beh_canon} | {sample_id} | A5: RT-DETR Box + Center Pt | Area: {ar_a5:.3f} | CC: {cc_a5} | Lat: {lat_a5:.1f}ms"
            comp_a5 = make_4panel_composite(
                frame_bgr, "A5", "RT-DETR Box + Center Point", prompt_box=largest_box, pos_points=pts_a5_raw, pred_mask=mask_a5,
                title_text=title_a5, box_label=f"RT-DETR (conf {largest_conf:.2f})", box_color=(255, 255, 0)
            )
            coords_a5 = {"box": largest_box, "points": pts_a5_raw, "labels": [1]}

        _, buf_a5 = cv2.imencode(".jpg", comp_a5, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample_conditions.append({
            "record": {
                "sample_id": sample_id,
                "canonical_behavior": beh_canon,
                "session_id": session_id,
                "frame_index": mid_f,
                "prompt_condition": "A5_rtdetr_box_center_point",
                "exact_prompt_coordinates": json.dumps(coords_a5),
                "mask_returned": ret_a5,
                "mask_status": st_a5,
                "inference_time_ms": lat_a5,
                "mask_area_ratio": ar_a5,
                "number_of_connected_components": cc_a5,
                "number_of_cow_detections": num_cattle_dets,
                "selected_box": json.dumps(largest_box) if largest_box else None,
                "selected_box_confidence": largest_conf,
                "selected_box_area_ratio": largest_area_ratio,
            },
            "comp_bytes": buf_a5.tobytes(),
            "condition_id": "A5",
        })

        results.append((sample_id, beh_canon, sample_conditions))
        print(f"  [Sample {idx+1}/{len(beef_records)}] {sample_id} ({beh_canon}) -> A0: ret={ret_a0} (cc={cc_a0}), A1: ret={ret_a1} (cc={cc_a1}), A2: ret={ret_a2} (cc={cc_a2}), A3: ret={ret_a3} (cc={cc_a3}), A4: ret={ret_a4} (cc={cc_a4}), A5: ret={ret_a5} (cc={cc_a5})")

    print(f"[MODAL-BEEF-RESCUE] Audit completed successfully across {len(results)} samples.")
    return results


def build_tiled_contact_sheet(images_bytes_list, grid_cols, grid_rows, tile_w, tile_h, title="Contact Sheet"):
    """
    Build a composite contact sheet tiling images in a grid with OpenCV text banner.
    """
    header_h = 50
    sheet_w = grid_cols * tile_w
    sheet_h = grid_rows * tile_h + header_h

    sheet = np.full((sheet_h, sheet_w, 3), 25, dtype=np.uint8)

    # Top title bar
    cv2.rectangle(sheet, (0, 0), (sheet_w, header_h), (15, 15, 15), -1)
    cv2.putText(sheet, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, img_bytes in enumerate(images_bytes_list):
        if idx >= grid_cols * grid_rows:
            break
        r = idx // grid_cols
        c = idx % grid_cols
        x_pos = c * tile_w
        y_pos = header_h + r * tile_h

        img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_arr is not None:
            tile_resized = cv2.resize(img_arr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            sheet[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w] = tile_resized

        cv2.rectangle(sheet, (x_pos, y_pos), (x_pos + tile_w - 1, y_pos + tile_h - 1), (50, 50, 50), 1)

    return sheet


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  KAGGLE BEEF SAM 2.1 PROMPT-RESCUE AUDIT")
    print("  Testing 6 Conditions: A0 (Baseline), A1 (Center Pt), A2 (Multi Pt),")
    print("                        A3 (Center+NegCorners), A4 (RT-DETR Box), A5 (RT-DETR+Pt)")
    print("  Sample Provenance: 20 Kaggle Beef Midpoint Frames (Seed 2026, Train Split)")
    print("  Environment: Modal Cloud (profile tigerwood693, GPU Tier: T4)")
    print("=" * 80)

    sanity_csv = "artifacts/perception_audit/behavior_primary_segmentation_sanity.csv"
    train_csv = "datasets/behavior/cvb_beef/train.csv"

    if not os.path.exists(sanity_csv):
        raise FileNotFoundError(f"Missing sanity CSV: {sanity_csv}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")

    # Load the exact 20 Beef samples
    df_sanity = pd.read_csv(sanity_csv)
    df_train = pd.read_csv(train_csv)

    beef_sanity = df_sanity[df_sanity["dataset"] == "beef_cattle_behavior"].copy()
    assert len(beef_sanity) == 20, f"Expected 20 Beef samples, got {len(beef_sanity)}"

    # Merge source_path and metadata
    beef_merged = beef_sanity.merge(
        df_train[["sample_id", "source_path", "start_frame", "end_frame", "n_frames"]],
        on="sample_id",
        how="left",
    )

    beef_records = beef_merged.to_dict(orient="records")
    print(f"\n[1/3] Verified {len(beef_records)} Beef samples across 4 canonical behaviors:")
    for beh, grp in beef_merged.groupby("canonical_behavior"):
        print(f"  - {beh}: {len(grp)} samples")

    # Dispatch to Modal
    print("\n[2/3] Dispatching Prompt-Rescue Audit to Modal (tigerwood693, T4 GPU)...")
    modal_output = audit_beef_prompt_rescue_modal.remote(beef_records)

    # Assemble deliverables
    print("\n[3/3] Assembling results, saving CSV and generating contact sheets...")
    out_csv_dir = Path("artifacts/perception_audit")
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / "beef_sam_prompt_rescue.csv"

    out_assets_dir = Path("docs/audits/assets/beef_sam_prompt_rescue")
    out_assets_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    # Dict to collect composites per condition for contact sheets
    condition_composites = {
        "A0": [],
        "A1": [],
        "A2": [],
        "A3": [],
        "A4": [],
        "A5": [],
    }

    condition_titles = {
        "A0": "A0: Baseline Full Crop Box [0, 0, 223, 223]",
        "A1": "A1: Center Positive Point (cx, cy)",
        "A2": "A2: Multi-Positive Body Points (5 Pts)",
        "A3": "A3: Center Positive + 4 Negative Corners",
        "A4": "A4: Largest RT-DETR-L Cow Box",
        "A5": "A5: Largest RT-DETR Box + Center Point",
    }

    for sample_id, beh, s_conds in modal_output:
        for c_entry in s_conds:
            rec = c_entry["record"]
            all_rows.append(rec)
            c_id = c_entry["condition_id"]
            comp_bytes = c_entry["comp_bytes"]

            # Save individual 4-panel composite
            out_fn = f"beef_{c_id}_{beh}_{sample_id}.jpg"
            with open(out_assets_dir / out_fn, "wb") as f:
                f.write(comp_bytes)

            condition_composites[c_id].append(comp_bytes)

    # Save CSV
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(out_csv_path, index=False)
    print(f"  ✓ Saved prompt-rescue audit CSV ({len(df_results)} rows) to: {out_csv_path}")

    # Build Contact Sheets for each condition
    tile_w = 600
    tile_h = 174
    for c_id, c_title in condition_titles.items():
        c_bytes_list = condition_composites[c_id]
        sheet = build_tiled_contact_sheet(
            c_bytes_list,
            grid_cols=4,
            grid_rows=5,
            tile_w=tile_w,
            tile_h=tile_h,
            title=f"Kaggle Beef SAM 2.1 Rescue — {c_title} (4x5 Grid)",
        )
        sheet_path = out_assets_dir / f"beef_{c_id}_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  ✓ Saved Contact Sheet for {c_id} ({sheet.shape[1]}x{sheet.shape[0]}) to: {sheet_path}")

    # Summary table in console
    print("\n" + "=" * 80)
    print("  KAGGLE BEEF PROMPT-RESCUE AUDIT SUMMARY")
    print("=" * 80)
    for c_id in ["A0_full_crop", "A1_center_point", "A2_multipoint", "A3_center_negcorners", "A4_largest_rtdetr_box", "A5_rtdetr_box_center_point"]:
        sub_df = df_results[df_results["prompt_condition"] == c_id]
        ret_cnt = sub_df["mask_returned"].sum()
        succ_df = sub_df[sub_df["mask_returned"]]
        avg_cc = succ_df["number_of_connected_components"].mean() if len(succ_df) > 0 else 0.0
        med_cc = succ_df["number_of_connected_components"].median() if len(succ_df) > 0 else 0.0
        avg_ar = succ_df["mask_area_ratio"].mean() if len(succ_df) > 0 else 0.0
        avg_lat = sub_df["inference_time_ms"].mean()

        print(f"\nCondition: {c_id}")
        print(f"  Mask Return Rate: {ret_cnt} / {len(sub_df)} ({ret_cnt/len(sub_df)*100:.1f}%)")
        for beh, grp in sub_df.groupby("canonical_behavior"):
            b_ret = grp["mask_returned"].sum()
            print(f"    {beh}: {b_ret}/{len(grp)}")
        print(f"  Area Ratio (Mean for returned): {avg_ar:.4f}")
        print(f"  Connected Components (Returned): Mean = {avg_cc:.1f}, Median = {med_cc:.1f}")
        print(f"  Mean Latency: {avg_lat:.1f} ms")


if __name__ == "__main__":
    main()
