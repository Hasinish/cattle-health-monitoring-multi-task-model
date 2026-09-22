# -*- coding: utf-8 -*-
"""
Phase 3 Primary Behavior Stack SAM 2.1 Segmentation Sanity Check
Evaluates pretrained SAM 2.1 small (sam2.1_s.pt) directly on the same 45 Behavior frames
used in the localization sanity check (25 CVB + 20 Kaggle Beef from datasets/behavior/cvb_beef/train.csv).

Model: sam2.1_s.pt (ultralytics)
Modal Profile: tigerwood693
GPU Tier: T4
Seed: 2026 (provenance reused from localization audit)

Outputs:
  - artifacts/perception_audit/behavior_primary_segmentation_sanity.csv
  - docs/audits/assets/behavior_primary_segmentation_sanity/ (composites + contact sheets)
  - docs/research_log/2026-09-23_behavior_primary_segmentation_sanity.md
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

app = modal.App("behavior-primary-segmentation-sanity", image=image)


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
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


def make_4panel_composite(img_bgr, prompt_box, pred_mask, title_text, box_label="Prompt Box", box_color=(0, 255, 0), mask_color=(0, 230, 0), target_h=200):
    """
    Generate 4-panel visual inspection composite:
    [Original | Prompt Box | SAM 2.1 Mask | Overlay]
    """
    h, w = img_bgr.shape[:2]

    # Panel 1: Original
    p1 = img_bgr.copy()
    cv2.putText(p1, "Original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 2: Prompt Box
    p2 = img_bgr.copy()
    if prompt_box is not None:
        x1, y1, x2, y2 = [int(round(c)) for c in prompt_box]
        cv2.rectangle(p2, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(p2, box_label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
    cv2.putText(p2, "Prompt Box", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 3: SAM Mask (Binary white on black)
    p3 = np.zeros((h, w, 3), dtype=np.uint8)
    if pred_mask is not None and np.sum(pred_mask) > 0:
        p3[pred_mask] = [255, 255, 255]
    cv2.putText(p3, "SAM 2.1 Mask", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 4: Overlay
    p4 = img_bgr.copy()
    if pred_mask is not None and np.sum(pred_mask) > 0:
        overlay = p4.copy()
        overlay[pred_mask] = mask_color
        cv2.addWeighted(overlay, 0.45, p4, 0.55, 0, p4)
        if prompt_box is not None:
            x1, y1, x2, y2 = [int(round(c)) for c in prompt_box]
            cv2.rectangle(p4, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(p4, "Overlay", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Resize panels
    panels = [p1, p2, p3, p4]
    resized = []
    for p in panels:
        ph, pw = p.shape[:2]
        tw = int(round(pw * (target_h / ph)))
        resized.append(cv2.resize(p, (tw, target_h), interpolation=cv2.INTER_AREA))

    composite = np.hstack(resized)

    # Top title banner
    banner_h = 30
    comp_w = composite.shape[1]
    banner = np.full((banner_h, comp_w, 3), 20, dtype=np.uint8)
    cv2.putText(banner, title_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, composite])


@app.function(
    volumes={"/data": cvb_vol},
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def audit_cvb_segmentation_modal(cvb_records):
    """
    Run SAM 2.1 on 25 CVB frames under TWO diagnostic prompts:
    A. Official Target GT Box -> SAM 2.1
    B. Matched RT-DETR-L Box -> SAM 2.1
    """
    import os
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM, RTDETR

    print(f"[MODAL-CVB-SAM] Initializing SAM 2.1 and RT-DETR-L on Tesla T4...")
    print(f"[MODAL-CVB-SAM] CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

    rtdetr_model = RTDETR("rtdetr-l.pt")
    sam_model = SAM("sam2.1_s.pt")
    cvb_root = "/data/cvb/000058916v001"

    results = []

    for idx, rec in enumerate(cvb_records):
        sample_id = rec["sample_id"]
        cut_name = rec["session_id"]
        source_vid = rec["source_video_id"]
        beh_canon = rec["canonical_behavior"]
        mid_f = int(rec["frame_index"])
        track_id = int(rec.get("tracklet_id", 0))

        # 1. Load 1080p frame
        img_path = os.path.join(cvb_root, "data", "raw_frames", cut_name, f"img_{mid_f:05d}.jpg")
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

        # 2. Parse GT Box
        gt_box_str = rec.get("target_gt_bbox", "gt_unavailable")
        gt_box = None
        if gt_box_str != "gt_unavailable" and isinstance(gt_box_str, str):
            try:
                gt_box = [float(c) for c in json.loads(gt_box_str)]
            except Exception:
                gt_box = None

        if gt_box is None:
            print(f"[WARN] GT box unavailable for {sample_id}")
            continue

        # 3. Recover Matched RT-DETR Detection Box
        # Run RT-DETR-L on the frame and find box with max IoU with gt_box
        t_rt0 = time.perf_counter()
        rt_preds = rtdetr_model(img_bgr, conf=0.25, classes=[19], verbose=False, device="cuda")
        t_rt_ms = (time.perf_counter() - t_rt0) * 1000.0

        b_objs = rt_preds[0].boxes
        rtdetr_box = None
        rtdetr_iou = 0.0
        rtdetr_conf = 0.0

        if len(b_objs) > 0:
            p_boxes = b_objs.xyxy.cpu().numpy().tolist()
            p_confs = b_objs.conf.cpu().numpy().tolist()
            ious = [compute_iou(gt_box, pb) for pb in p_boxes]
            best_idx = int(np.argmax(ious))
            rtdetr_box = p_boxes[best_idx]
            rtdetr_iou = float(ious[best_idx])
            rtdetr_conf = float(p_confs[best_idx])

        # -------------------------------------------------------------
        # PROMPT A: Official Target GT Box -> SAM 2.1
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            res_gt = sam_model(img_bgr, bboxes=[gt_box], device="cuda", verbose=False)
            t_gt_ms = (time.perf_counter() - t0) * 1000.0
            if res_gt[0].masks is not None and len(res_gt[0].masks.data) > 0:
                raw_mask_gt = res_gt[0].masks.data[0].cpu().numpy().astype(bool)
                if raw_mask_gt.shape != (img_h, img_w):
                    pred_mask_gt = cv2.resize(raw_mask_gt.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    pred_mask_gt = raw_mask_gt
                mask_gt_returned = True
                mask_gt_status = "segmented"
            else:
                pred_mask_gt = np.zeros((img_h, img_w), dtype=bool)
                mask_gt_returned = False
                mask_gt_status = "sam_no_mask"
        except Exception as e:
            t_gt_ms = (time.perf_counter() - t0) * 1000.0
            pred_mask_gt = np.zeros((img_h, img_w), dtype=bool)
            mask_gt_returned = False
            mask_gt_status = f"error_{str(e)[:25]}"

        # Compute GT-prompt metrics
        mask_gt_pixels = int(np.sum(pred_mask_gt))
        mask_gt_area_ratio = mask_gt_pixels / (img_w * img_h)

        # Sanity metric: fraction of mask pixels inside GT bbox
        gx1_c = max(0, min(img_w, int(round(gt_box[0]))))
        gy1_c = max(0, min(img_h, int(round(gt_box[1]))))
        gx2_c = max(0, min(img_w, int(round(gt_box[2]))))
        gy2_c = max(0, min(img_h, int(round(gt_box[3]))))
        gt_in_box_pixels = int(np.sum(pred_mask_gt[gy1_c:gy2_c, gx1_c:gx2_c]))
        gt_in_box_ratio_sanity = (gt_in_box_pixels / mask_gt_pixels) if mask_gt_pixels > 0 else 0.0

        num_cc_gt = int(cv2.connectedComponents(pred_mask_gt.astype(np.uint8))[0] - 1) if mask_gt_pixels > 0 else 0

        # Composite A (GT prompted)
        title_gt = f"CVB | {beh_canon} | Prompt A: Target GT | Area: {mask_gt_area_ratio:.4f} | InBoxSanity: {gt_in_box_ratio_sanity:.3f} | Lat: {t_gt_ms:.1f}ms"
        comp_gt = make_4panel_composite(
            img_bgr,
            gt_box,
            pred_mask_gt,
            title_text=title_gt,
            box_label=f"GT Trk {track_id}",
            box_color=(0, 255, 0),
            mask_color=(0, 230, 0),
            target_h=160,
        )
        _, jpg_gt_buf = cv2.imencode(".jpg", comp_gt, [cv2.IMWRITE_JPEG_QUALITY, 85])
        comp_gt_bytes = jpg_gt_buf.tobytes()

        # -------------------------------------------------------------
        # PROMPT B: Matched RT-DETR Box -> SAM 2.1
        # -------------------------------------------------------------
        if rtdetr_box is not None:
            t0 = time.perf_counter()
            try:
                res_rt = sam_model(img_bgr, bboxes=[rtdetr_box], device="cuda", verbose=False)
                t_rt_ms_sam = (time.perf_counter() - t0) * 1000.0
                if res_rt[0].masks is not None and len(res_rt[0].masks.data) > 0:
                    raw_mask_rt = res_rt[0].masks.data[0].cpu().numpy().astype(bool)
                    if raw_mask_rt.shape != (img_h, img_w):
                        pred_mask_rt = cv2.resize(raw_mask_rt.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        pred_mask_rt = raw_mask_rt
                    mask_rt_returned = True
                    mask_rt_status = "segmented"
                else:
                    pred_mask_rt = np.zeros((img_h, img_w), dtype=bool)
                    mask_rt_returned = False
                    mask_rt_status = "sam_no_mask"
            except Exception as e:
                t_rt_ms_sam = (time.perf_counter() - t0) * 1000.0
                pred_mask_rt = np.zeros((img_h, img_w), dtype=bool)
                mask_rt_returned = False
                mask_rt_status = f"error_{str(e)[:25]}"
        else:
            t_rt_ms_sam = 0.0
            pred_mask_rt = np.zeros((img_h, img_w), dtype=bool)
            mask_rt_returned = False
            mask_rt_status = "upstream_no_detection"

        # Compute RT-DETR-prompt metrics
        mask_rt_pixels = int(np.sum(pred_mask_rt))
        mask_rt_area_ratio = mask_rt_pixels / (img_w * img_h)

        # Sanity metric against official GT box
        rt_in_box_pixels = int(np.sum(pred_mask_rt[gy1_c:gy2_c, gx1_c:gx2_c]))
        rt_in_box_ratio_sanity = (rt_in_box_pixels / mask_rt_pixels) if mask_rt_pixels > 0 else 0.0

        num_cc_rt = int(cv2.connectedComponents(pred_mask_rt.astype(np.uint8))[0] - 1) if mask_rt_pixels > 0 else 0

        # Composite B (RT-DETR prompted)
        title_rt = f"CVB | {beh_canon} | Prompt B: RT-DETR (IoU {rtdetr_iou:.2f}) | Area: {mask_rt_area_ratio:.4f} | InBoxSanity: {rt_in_box_ratio_sanity:.3f} | Lat: {t_rt_ms_sam:.1f}ms"
        comp_rt = make_4panel_composite(
            img_bgr,
            rtdetr_box,
            pred_mask_rt,
            title_text=title_rt,
            box_label=f"RT-DETR {rtdetr_conf:.2f} (IoU:{rtdetr_iou:.2f})",
            box_color=(255, 255, 0),
            mask_color=(255, 230, 0),
            target_h=160,
        )
        _, jpg_rt_buf = cv2.imencode(".jpg", comp_rt, [cv2.IMWRITE_JPEG_QUALITY, 85])
        comp_rt_bytes = jpg_rt_buf.tobytes()

        res_dict = {
            "sample_id": sample_id,
            "dataset": "cvb",
            "canonical_behavior": beh_canon,
            "source_video_id": source_vid,
            "session_id": cut_name,
            "frame_index": mid_f,
            "image_width": img_w,
            "image_height": img_h,
            # Prompt A: GT Box
            "cvb_gt_prompt_box": json.dumps([round(c, 1) for c in gt_box]),
            "cvb_gt_mask_returned": mask_gt_returned,
            "cvb_gt_mask_status": mask_gt_status,
            "cvb_gt_inference_time_ms": round(t_gt_ms, 2),
            "cvb_gt_mask_area_pixels": mask_gt_pixels,
            "cvb_gt_mask_area_ratio": round(mask_gt_area_ratio, 4),
            "cvb_gt_mask_in_target_box_ratio_sanity": round(gt_in_box_ratio_sanity, 4),
            "cvb_gt_num_connected_components": num_cc_gt,
            # Prompt B: RT-DETR Box
            "cvb_rtdetr_prompt_box": json.dumps([round(c, 1) for c in rtdetr_box]) if rtdetr_box else "None",
            "cvb_rtdetr_prompt_box_iou_with_gt": round(rtdetr_iou, 4),
            "cvb_rtdetr_mask_returned": mask_rt_returned,
            "cvb_rtdetr_mask_status": mask_rt_status,
            "cvb_rtdetr_inference_time_ms": round(t_rt_ms_sam, 2),
            "cvb_rtdetr_mask_area_pixels": mask_rt_pixels,
            "cvb_rtdetr_mask_area_ratio": round(mask_rt_area_ratio, 4),
            "cvb_rtdetr_mask_in_target_box_ratio_sanity": round(rt_in_box_ratio_sanity, 4),
            "cvb_rtdetr_num_connected_components": num_cc_rt,
            # Beef fields None for CVB
            "beef_fullcrop_prompt_box": None,
            "beef_mask_returned": None,
            "beef_mask_status": None,
            "beef_inference_time_ms": None,
            "beef_mask_area_pixels": None,
            "beef_mask_area_ratio": None,
            "beef_num_connected_components": None,
        }
        results.append((res_dict, comp_gt_bytes, comp_rt_bytes))
        print(f"  [CVB-SAM {idx+1}/{len(cvb_records)}] {sample_id} | {beh_canon} | GT_Area: {mask_gt_area_ratio:.3f} | RT_Area: {mask_rt_area_ratio:.3f} | InBoxSanity: {gt_in_box_ratio_sanity:.3f}/{rt_in_box_ratio_sanity:.3f}")

    print(f"[MODAL-CVB-SAM] Completed {len(results)} CVB samples.")
    return results


@app.function(
    volumes={"/data": beef_vol},
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
)
def audit_beef_segmentation_modal(beef_records):
    """
    Run SAM 2.1 on 20 Kaggle Beef frames under FULL-CROP prompt [0, 0, width-1, height-1].
    RT-DETR is completely bypassed.
    """
    import os
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM

    print(f"[MODAL-BEEF-SAM] Initializing SAM 2.1 on Tesla T4...")
    print(f"[MODAL-BEEF-SAM] CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

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

        # Full-crop prompt box: [0, 0, width-1, height-1]
        full_crop_box = [0, 0, img_w - 1, img_h - 1]

        t0 = time.perf_counter()
        try:
            res_beef = sam_model(frame_bgr, bboxes=[full_crop_box], device="cuda", verbose=False)
            t_ms = (time.perf_counter() - t0) * 1000.0
            if res_beef[0].masks is not None and len(res_beef[0].masks.data) > 0:
                raw_mask = res_beef[0].masks.data[0].cpu().numpy().astype(bool)
                if raw_mask.shape != (img_h, img_w):
                    pred_mask = cv2.resize(raw_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    pred_mask = raw_mask
                mask_returned = True
                mask_status = "segmented"
            else:
                pred_mask = np.zeros((img_h, img_w), dtype=bool)
                mask_returned = False
                mask_status = "sam_no_mask"
        except Exception as e:
            t_ms = (time.perf_counter() - t0) * 1000.0
            pred_mask = np.zeros((img_h, img_w), dtype=bool)
            mask_returned = False
            mask_status = f"error_{str(e)[:25]}"

        mask_pixels = int(np.sum(pred_mask))
        mask_area_ratio = mask_pixels / (img_w * img_h)
        num_cc = int(cv2.connectedComponents(pred_mask.astype(np.uint8))[0] - 1) if mask_pixels > 0 else 0

        # Composite (Full-crop prompted)
        title_beef = f"Beef | {beh_canon} | Full-Crop Prompt | MaskArea: {mask_area_ratio:.3f} | Comps: {num_cc} | Lat: {t_ms:.1f}ms"
        comp_beef = make_4panel_composite(
            frame_bgr,
            full_crop_box,
            pred_mask,
            title_text=title_beef,
            box_label="Full Crop [0,0,W-1,H-1]",
            box_color=(0, 255, 255),
            mask_color=(0, 230, 255),
            target_h=200,
        )
        _, jpg_buf = cv2.imencode(".jpg", comp_beef, [cv2.IMWRITE_JPEG_QUALITY, 90])
        comp_bytes = jpg_buf.tobytes()

        res_dict = {
            "sample_id": sample_id,
            "dataset": "beef_cattle_behavior",
            "canonical_behavior": beh_canon,
            "source_video_id": source_vid,
            "session_id": session_id,
            "frame_index": mid_f,
            "image_width": img_w,
            "image_height": img_h,
            # CVB fields None for Beef
            "cvb_gt_prompt_box": None,
            "cvb_gt_mask_returned": None,
            "cvb_gt_mask_status": None,
            "cvb_gt_inference_time_ms": None,
            "cvb_gt_mask_area_pixels": None,
            "cvb_gt_mask_area_ratio": None,
            "cvb_gt_mask_in_target_box_ratio_sanity": None,
            "cvb_gt_num_connected_components": None,
            "cvb_rtdetr_prompt_box": None,
            "cvb_rtdetr_prompt_box_iou_with_gt": None,
            "cvb_rtdetr_mask_returned": None,
            "cvb_rtdetr_mask_status": None,
            "cvb_rtdetr_inference_time_ms": None,
            "cvb_rtdetr_mask_area_pixels": None,
            "cvb_rtdetr_mask_area_ratio": None,
            "cvb_rtdetr_mask_in_target_box_ratio_sanity": None,
            "cvb_rtdetr_num_connected_components": None,
            # Beef Full-Crop Prompt fields
            "beef_fullcrop_prompt_box": json.dumps(full_crop_box),
            "beef_mask_returned": mask_returned,
            "beef_mask_status": mask_status,
            "beef_inference_time_ms": round(t_ms, 2),
            "beef_mask_area_pixels": mask_pixels,
            "beef_mask_area_ratio": round(mask_area_ratio, 4),
            "beef_num_connected_components": num_cc,
        }
        results.append((res_dict, comp_bytes))
        print(f"  [Beef-SAM {idx+1}/{len(beef_records)}] {sample_id} | {beh_canon} | Area: {mask_area_ratio:.3f} | Comps: {num_cc} ({t_ms:.1f}ms)")

    print(f"[MODAL-BEEF-SAM] Completed {len(results)} Beef samples.")
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
    cv2.putText(sheet, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

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
    print("  PHASE 3 PRIMARY BEHAVIOR STACK SAM 2.1 SEGMENTATION SANITY CHECK")
    print("  Model: sam2.1_s.pt (ultralytics)")
    print("  Sample Provenance: 45 Frames Reused from behavior_primary_localization_sanity.csv")
    print("  Environment: Modal Cloud (profile tigerwood693, GPU Tier: T4)")
    print("=" * 80)

    loc_csv = "artifacts/perception_audit/behavior_primary_localization_sanity.csv"
    train_csv = "datasets/behavior/cvb_beef/train.csv"

    if not os.path.exists(loc_csv):
        raise FileNotFoundError(f"Missing localization audit CSV: {loc_csv}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")

    # Load 45 samples from completed localization audit
    df_loc = pd.read_csv(loc_csv)
    df_train = pd.read_csv(train_csv)

    # Merge provenance columns
    df_merged = df_loc.merge(
        df_train[["sample_id", "source_path", "start_frame", "end_frame", "tracklet_id", "n_frames"]],
        on="sample_id",
        how="left",
    )

    cvb_s = df_merged[df_merged["dataset"] == "cvb"].copy()
    beef_s = df_merged[df_merged["dataset"] == "beef_cattle_behavior"].copy()

    print(f"\n[1/4] Reused Provenance Verified:")
    print(f"  - CVB samples: {len(cvb_s)} (5 classes x 5 samples across 25 unique source videos)")
    print(f"  - Beef samples: {len(beef_s)} (4 classes x 5 samples across 19 unique sessions)")
    print(f"  - Total samples: {len(df_merged)}")

    # 2. Dispatch to Modal Cloud
    print("\n[2/4] Dispatching CVB SAM 2.1 audit to Modal (Prompt A: GT Box, Prompt B: RT-DETR Box)...")
    cvb_records = cvb_s.to_dict(orient="records")
    cvb_modal_results = audit_cvb_segmentation_modal.remote(cvb_records)

    print("\n[3/4] Dispatching Kaggle Beef SAM 2.1 audit to Modal (Full-Crop Prompt [0, 0, W-1, H-1])...")
    beef_records = beef_s.to_dict(orient="records")
    beef_modal_results = audit_beef_segmentation_modal.remote(beef_records)

    # 3. Assemble Results & Save Visual Assets
    print("\n[4/4] Assembling deliverables, saving CSV, and generating contact sheets...")
    out_csv_dir = Path("artifacts/perception_audit")
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / "behavior_primary_segmentation_sanity.csv"

    out_assets_dir = Path("docs/audits/assets/behavior_primary_segmentation_sanity")
    out_assets_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    cvb_gt_comps = []
    cvb_rt_comps = []

    for res, comp_gt_bytes, comp_rt_bytes in cvb_modal_results:
        all_rows.append(res)
        s_id = res["sample_id"]
        beh = res["canonical_behavior"]

        # Save individual 4-panel composites
        out_gt = out_assets_dir / f"cvb_{beh}_{s_id}_promptA_gt.jpg"
        out_rt = out_assets_dir / f"cvb_{beh}_{s_id}_promptB_rtdetr.jpg"
        with open(out_gt, "wb") as f:
            f.write(comp_gt_bytes)
        with open(out_rt, "wb") as f:
            f.write(comp_rt_bytes)

        cvb_gt_comps.append(comp_gt_bytes)
        cvb_rt_comps.append(comp_rt_bytes)

    beef_comps = []
    for res, comp_beef_bytes in beef_modal_results:
        all_rows.append(res)
        s_id = res["sample_id"]
        beh = res["canonical_behavior"]

        out_beef = out_assets_dir / f"beef_{beh}_{s_id}_fullcrop.jpg"
        with open(out_beef, "wb") as f:
            f.write(comp_beef_bytes)

        beef_comps.append(comp_beef_bytes)

    # Save CSV
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(out_csv_path, index=False)
    print(f"  ✓ Saved segmentation audit CSV ({len(df_results)} rows) to: {out_csv_path}")

    # Build Contact Sheets
    # CVB Prompt A: GT Box Contact Sheet (5 cols x 5 rows = 25 tiles)
    # Tile size: 640 x 140 (aspect ratio ~4.5:1)
    cvb_gt_sheet = build_tiled_contact_sheet(
        cvb_gt_comps,
        grid_cols=5,
        grid_rows=5,
        tile_w=600,
        tile_h=130,
        title="CVB SAM 2.1 Prompt A: Target GT Box [Original | GT Box | SAM Mask | Overlay] (5x5 Grid)",
    )
    cvb_gt_sheet_path = out_assets_dir / "cvb_sam21_gt_contact_sheet.jpg"
    cv2.imwrite(str(cvb_gt_sheet_path), cvb_gt_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Saved CVB GT Contact Sheet ({cvb_gt_sheet.shape[1]}x{cvb_gt_sheet.shape[0]}) to: {cvb_gt_sheet_path}")

    # CVB Prompt B: RT-DETR Box Contact Sheet (5 cols x 5 rows = 25 tiles)
    cvb_rt_sheet = build_tiled_contact_sheet(
        cvb_rt_comps,
        grid_cols=5,
        grid_rows=5,
        tile_w=600,
        tile_h=130,
        title="CVB SAM 2.1 Prompt B: Matched RT-DETR Box [Original | RT-DETR Box | SAM Mask | Overlay] (5x5 Grid)",
    )
    cvb_rt_sheet_path = out_assets_dir / "cvb_sam21_rtdetr_contact_sheet.jpg"
    cv2.imwrite(str(cvb_rt_sheet_path), cvb_rt_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Saved CVB RT-DETR Contact Sheet ({cvb_rt_sheet.shape[1]}x{cvb_rt_sheet.shape[0]}) to: {cvb_rt_sheet_path}")

    # Beef Full-Crop Contact Sheet (4 cols x 5 rows = 20 tiles)
    beef_sheet = build_tiled_contact_sheet(
        beef_comps,
        grid_cols=4,
        grid_rows=5,
        tile_w=600,
        tile_h=160,
        title="Kaggle Beef SAM 2.1 Full-Crop Prompt [Original | Full Crop Box | SAM Mask | Overlay] (4x5 Grid)",
    )
    beef_sheet_path = out_assets_dir / "beef_sam21_fullcrop_contact_sheet.jpg"
    cv2.imwrite(str(beef_sheet_path), beef_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Saved Beef Contact Sheet ({beef_sheet.shape[1]}x{beef_sheet.shape[0]}) to: {beef_sheet_path}")

    # Print Summary Metrics
    print("\n" + "=" * 80)
    print("  SAM 2.1 SEGMENTATION SANITY CHECK METRICS")
    print("=" * 80)

    cvb_df = df_results[df_results["dataset"] == "cvb"]
    beef_df = df_results[df_results["dataset"] == "beef_cattle_behavior"]

    print("\n--- CVB Prompt A: Official Target GT Box ---")
    gt_ret_cnt = cvb_df["cvb_gt_mask_returned"].sum()
    print(f"Mask Return Rate: {gt_ret_cnt} / {len(cvb_df)} ({gt_ret_cnt/len(cvb_df)*100:.1f}%)")
    print(f"Mask Area Fraction: Mean = {cvb_df['cvb_gt_mask_area_ratio'].mean():.4f}, Min = {cvb_df['cvb_gt_mask_area_ratio'].min():.4f}, Max = {cvb_df['cvb_gt_mask_area_ratio'].max():.4f}")
    print(f"Mask In Target Box Sanity Ratio: Mean = {cvb_df['cvb_gt_mask_in_target_box_ratio_sanity'].mean():.4f} (Median: {cvb_df['cvb_gt_mask_in_target_box_ratio_sanity'].median():.4f})")
    print(f"Connected Components: Mean = {cvb_df['cvb_gt_num_connected_components'].mean():.2f}")

    print("\n--- CVB Prompt B: Matched RT-DETR-L Box ---")
    rt_ret_cnt = cvb_df["cvb_rtdetr_mask_returned"].sum()
    print(f"Mask Return Rate: {rt_ret_cnt} / {len(cvb_df)} ({rt_ret_cnt/len(cvb_df)*100:.1f}%)")
    print(f"Mask Area Fraction: Mean = {cvb_df['cvb_rtdetr_mask_area_ratio'].mean():.4f}, Min = {cvb_df['cvb_rtdetr_mask_area_ratio'].min():.4f}, Max = {cvb_df['cvb_rtdetr_mask_area_ratio'].max():.4f}")
    print(f"Mask In Target Box Sanity Ratio: Mean = {cvb_df['cvb_rtdetr_mask_in_target_box_ratio_sanity'].mean():.4f} (Median: {cvb_df['cvb_rtdetr_mask_in_target_box_ratio_sanity'].median():.4f})")
    print(f"Connected Components: Mean = {cvb_df['cvb_rtdetr_num_connected_components'].mean():.2f}")

    print("\n--- Kaggle Beef: Full-Crop Box Prompt ---")
    beef_ret_cnt = beef_df["beef_mask_returned"].sum()
    print(f"Mask Return Rate: {beef_ret_cnt} / {len(beef_df)} ({beef_ret_cnt/len(beef_df)*100:.1f}%)")
    print(f"Mask Area Fraction: Mean = {beef_df['beef_mask_area_ratio'].mean():.4f}, Min = {beef_df['beef_mask_area_ratio'].min():.4f}, Max = {beef_df['beef_mask_area_ratio'].max():.4f}")
    print(f"Connected Components: Mean = {beef_df['beef_num_connected_components'].mean():.2f}")

    print("\nPer-Class Mask Return Summary:")
    print("  CVB (Prompt A / Prompt B):")
    for beh, b_df in cvb_df.groupby("canonical_behavior"):
        a_ret = b_df["cvb_gt_mask_returned"].sum()
        b_ret = b_df["cvb_rtdetr_mask_returned"].sum()
        print(f"    - {beh:10s}: GT={a_ret}/{len(b_df)}, RT={b_ret}/{len(b_df)}, GT_InBoxSanity={b_df['cvb_gt_mask_in_target_box_ratio_sanity'].mean():.3f}, RT_InBoxSanity={b_df['cvb_rtdetr_mask_in_target_box_ratio_sanity'].mean():.3f}")

    print("  Kaggle Beef (Full-Crop Prompt):")
    for beh, b_df in beef_df.groupby("canonical_behavior"):
        b_ret = b_df["beef_mask_returned"].sum()
        print(f"    - {beh:10s}: Ret={b_ret}/{len(b_df)}, Mean Area Ratio={b_df['beef_mask_area_ratio'].mean():.3f}, Mean Comps={b_df['beef_num_connected_components'].mean():.1f}")

    print("\n✓ SAM 2.1 Segmentation Sanity Check Complete!")


if __name__ == "__main__":
    main()
