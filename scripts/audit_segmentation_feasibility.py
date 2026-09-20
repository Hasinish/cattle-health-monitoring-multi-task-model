"""
Phase 3 Step 2.2: Cattle Segmentation Feasibility Audit
Evaluates pretrained instance segmentation across ScienceDB (BCS), MmCows (Behavior), and SideViewCows2026 (Re-ID).

Pipelines evaluated:
1. Main Pipeline: RT-DETR-L primary cow box -> SAM 2.1 (sam2.1_s.pt / sam2.1_hiera_small)
2. Fast Baseline: YOLO26s-seg (yolo26s-seg.pt)
3. Diagnostic (SideView only): Oracle GT box -> SAM 2.1

Usage:
    python scripts/audit_segmentation_feasibility.py --mode smoke
    python scripts/audit_segmentation_feasibility.py --mode expanded
"""

import os
import sys
import time
import json
import argparse
import gc
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import SAM, YOLO

# Ensure deterministic sampling and reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def print_progress(current, total, prefix="", suffix="", bar_len=28):
    """Clean single-line progress bar for Windows PowerShell without terminal glitches."""
    frac = current / total if total > 0 else 0
    filled = int(round(bar_len * frac))
    bar = "=" * filled + "-" * (bar_len - filled)
    percent = frac * 100.0
    line = f"\r{prefix} [{bar}] {percent:5.1f}% ({current}/{total}) {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()

def load_sample_manifest(mode='smoke'):
    """Load the deterministic sample manifest from Step 2.1."""
    csv_path = "artifacts/perception_audit/sample_manifest_expanded.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run Step 2.1 audit first!")
        
    df = pd.read_csv(csv_path)
    if mode == 'smoke':
        # Select first 10 of each dataset (30 images total)
        df_s = df[df['dataset'] == 'ScienceDB'].iloc[:10]
        df_m = df[df['dataset'] == 'MmCows'].iloc[:10]
        df_v = df[df['dataset'] == 'SideViewCows2026'].iloc[:10]
        df = pd.concat([df_s, df_m, df_v], ignore_index=True)
    return df

def load_rtdetr_primary_boxes(manifest_df):
    """
    Extract primary cow bounding box per sample from Step 2.1 RT-DETR-L detections.
    Documented Deterministic Primary-Cow Rule:
    Select detection with maximum bounding-box area `(x2 - x1) * (y2 - y1)`.
    Ties broken by higher confidence.
    If no detection, returns None (recorded as upstream_localization_failure).
    """
    csv_path = "artifacts/perception_audit/localization_detections_expanded.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run Step 2.1 audit first!")
        
    df_det = pd.read_csv(csv_path)
    rt = df_det[df_det['model_name'] == 'RT-DETR-L'].copy()
    
    # Filter for detected rows with valid coordinates
    rt_valid = rt[rt['detection_status'] == 'detected'].copy()
    rt_valid['box_x1'] = pd.to_numeric(rt_valid['box_x1'], errors='coerce')
    rt_valid['box_y1'] = pd.to_numeric(rt_valid['box_y1'], errors='coerce')
    rt_valid['box_x2'] = pd.to_numeric(rt_valid['box_x2'], errors='coerce')
    rt_valid['box_y2'] = pd.to_numeric(rt_valid['box_y2'], errors='coerce')
    rt_valid['confidence'] = pd.to_numeric(rt_valid['confidence'], errors='coerce')
    
    rt_valid['area'] = (rt_valid['box_x2'] - rt_valid['box_x1']) * (rt_valid['box_y2'] - rt_valid['box_y1'])
    
    # Primary cow: sort by area descending, confidence descending
    sorted_det = rt_valid.sort_values(by=['sample_id', 'area', 'confidence'], ascending=[True, False, False])
    primary_map = {}
    for s_id, grp in sorted_det.groupby('sample_id'):
        top = grp.iloc[0]
        primary_map[s_id] = {
            'box': [float(top['box_x1']), float(top['box_y1']), float(top['box_x2']), float(top['box_y2'])],
            'confidence': float(top['confidence']),
            'num_cows_detected': int(top['num_detections'])
        }
        
    # Return dict mapping sample_id -> box info or None
    sample_boxes = {}
    for _, row in manifest_df.iterrows():
        s_id = row['sample_id']
        sample_boxes[s_id] = primary_map.get(s_id, None)
        
    return sample_boxes

def compute_iou_dice(pred_mask, gt_mask):
    """Compute exact IoU and Dice between two 2D boolean masks."""
    pred_b = np.squeeze(pred_mask > 0).astype(bool)
    gt_b = np.squeeze(gt_mask > 0).astype(bool)
    
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    iou = float(inter) / float(union) if union > 0 else 0.0
    
    p_sum = pred_b.sum()
    g_sum = gt_b.sum()
    dice = (2.0 * float(inter)) / float(p_sum + g_sum) if (p_sum + g_sum) > 0 else 0.0
    return round(iou, 4), round(dice, 4)

def create_review_composite(img_bgr, box, pred_mask, dataset, subset, s_id, model_tag, gt_mask=None):
    """
    Generate 4-panel visual inspection composite:
    [Original (+ GT contour if available) | Bounding Box | Predicted Mask | Mask Overlay]
    """
    h, w = img_bgr.shape[:2]
    
    # Panel 1: Original (+ GT contour in yellow if available)
    p1 = img_bgr.copy()
    if gt_mask is not None and gt_mask.sum() > 0:
        gt_2d = np.squeeze(gt_mask).astype(np.uint8)
        contours, _ = cv2.findContours(gt_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, contours, -1, (0, 255, 255), 2)
    cv2.putText(p1, f"{dataset} | {subset}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 2: Box Prompt
    p2 = img_bgr.copy()
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange
        cv2.putText(p2, "Primary Box", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(p2, "NO BOX (Upstream Fail)", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(p2, "Box Prompt", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 3: Predicted Mask (Binary: white on black)
    p3 = np.zeros((h, w, 3), dtype=np.uint8)
    if pred_mask is not None and pred_mask.sum() > 0:
        pred_2d = np.squeeze(pred_mask > 0)
        p3[pred_2d] = [255, 255, 255]
    cv2.putText(p3, f"Mask ({model_tag})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 4: Mask Overlay (semi-transparent green on original)
    p4 = img_bgr.copy()
    if pred_mask is not None and pred_mask.sum() > 0:
        overlay = p4.copy()
        pred_2d = np.squeeze(pred_mask > 0)
        overlay[pred_2d] = [0, 230, 0] # Vibrant green
        cv2.addWeighted(overlay, 0.45, p4, 0.55, 0, p4)
    cv2.putText(p4, "Overlay", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Resize panels to uniform height 360px
    target_h = 360
    panels = [p1, p2, p3, p4]
    resized = []
    for p in panels:
        ph, pw = p.shape[:2]
        tw = int(pw * (target_h / ph))
        resized.append(cv2.resize(p, (tw, target_h), interpolation=cv2.INTER_AREA))
        
    composite = np.hstack(resized)
    return composite

def run_segmentation_audit(mode='smoke', device='cuda', resume=False):
    print("=" * 80)
    print(f"CATTLE SEGMENTATION FEASIBILITY AUDIT [Mode: {mode.upper()}]")
    print(f"Device: {device} | Checkpoint: sam2.1_s.pt / yolo26s-seg.pt")
    print("=" * 80)
    
    artifacts_dir = "artifacts/perception_audit"
    assets_dir = "docs/audits/assets/perception_audit"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Load samples and primary RT-DETR boxes
    df_samples = load_sample_manifest(mode=mode)
    primary_boxes = load_rtdetr_primary_boxes(df_samples)
    print(f"Loaded {len(df_samples)} samples:")
    for ds, cnt in df_samples['dataset'].value_counts().items():
        print(f"  - {ds:18s}: {cnt} images")
    print("-" * 80)
    
    records = []
    total = len(df_samples)
    
    # 2. Pipeline A: RT-DETR-L Box -> SAM 2.1
    print("\n[Pipeline A] Initializing SAM 2.1 (sam2.1_s.pt / sam2.1_hiera_small)...")
    sam_model = SAM('sam2.1_s.pt')
    
    sam_results = {}
    t_start = time.perf_counter()
    for i, (_, row) in enumerate(df_samples.iterrows(), 1):
        s_id = row['sample_id']
        ds = row['dataset']
        img_path = row['image_path']
        mask_path = row['mask_path']
        box_info = primary_boxes.get(s_id, None)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            sam_results[s_id] = {'mask': None, 'time_ms': 0.0, 'status': 'error_load'}
            continue
            
        h, w = img_bgr.shape[:2]
        
        if box_info is None:
            sam_results[s_id] = {'mask': None, 'time_ms': 0.0, 'status': 'upstream_localization_failure'}
        else:
            t0 = time.perf_counter()
            try:
                res = sam_model(img_bgr, bboxes=[box_info['box']], device=device, verbose=False)
                t_ms = (time.perf_counter() - t0) * 1000.0
                if res[0].masks is not None and len(res[0].masks) > 0:
                    raw_mask = res[0].masks.data[0].cpu().numpy()
                    pred_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                    status = 'segmented'
                else:
                    pred_mask = None
                    status = 'sam_no_mask'
            except Exception as e:
                t_ms = (time.perf_counter() - t0) * 1000.0
                pred_mask = None
                status = f'error_{str(e)[:25]}'
                
            sam_results[s_id] = {'mask': pred_mask, 'time_ms': round(t_ms, 1), 'status': status}
            
        elapsed = time.perf_counter() - t_start
        avg_s = elapsed / i
        eta_s = int((total - i) * avg_s)
        mins, secs = divmod(eta_s, 60)
        status_txt = sam_results[s_id]['status']
        print_progress(i, total, prefix="  RT-DETR -> SAM 2.1", suffix=f"| {ds:12s} | {status_txt:16s} | ETA: {mins:02d}:{secs:02d}")
        
    print()
    del sam_model
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # 3. Pipeline B: Fast Baseline YOLO26s-seg
    print("\n[Pipeline B] Initializing Fast Baseline YOLO26s-seg (yolo26s-seg.pt)...")
    yolo_model = YOLO('yolo26s-seg.pt')
    
    yolo_results = {}
    t_start = time.perf_counter()
    for i, (_, row) in enumerate(df_samples.iterrows(), 1):
        s_id = row['sample_id']
        ds = row['dataset']
        img_path = row['image_path']
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            yolo_results[s_id] = {'mask': None, 'time_ms': 0.0, 'status': 'error_load', 'box': None}
            continue
            
        h, w = img_bgr.shape[:2]
        t0 = time.perf_counter()
        try:
            res = yolo_model(img_bgr, conf=0.25, classes=[19], verbose=False, device=device)
            t_ms = (time.perf_counter() - t0) * 1000.0
            if res[0].masks is not None and len(res[0].masks) > 0:
                # Select primary cow by largest area among cow detections
                boxes = res[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(np.argmax(areas))
                
                raw_mask = res[0].masks.data[best_idx].cpu().numpy()
                pred_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                primary_box = [float(v) for v in boxes[best_idx]]
                status = 'segmented'
            else:
                pred_mask = None
                primary_box = None
                status = 'no_detection'
        except Exception as e:
            t_ms = (time.perf_counter() - t0) * 1000.0
            pred_mask = None
            primary_box = None
            status = f'error_{str(e)[:25]}'
            
        yolo_results[s_id] = {'mask': pred_mask, 'time_ms': round(t_ms, 1), 'status': status, 'box': primary_box}
        elapsed = time.perf_counter() - t_start
        avg_s = elapsed / i
        eta_s = int((total - i) * avg_s)
        mins, secs = divmod(eta_s, 60)
        status_txt = yolo_results[s_id]['status']
        print_progress(i, total, prefix="  YOLO26s-seg       ", suffix=f"| {ds:12s} | {status_txt:16s} | ETA: {mins:02d}:{secs:02d}")
        
    print()
    del yolo_model
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # 4. Diagnostic: Oracle GT Box -> SAM 2.1 (SideViewCows2026 only)
    sideview_samples = df_samples[df_samples['dataset'] == 'SideViewCows2026']
    oracle_results = {}
    
    if len(sideview_samples) > 0:
        print("\n[Diagnostic] Initializing Oracle GT Box -> SAM 2.1 (SideView only)...")
        sam_model = SAM('sam2.1_s.pt')
        t_start = time.perf_counter()
        
        for i, (_, row) in enumerate(sideview_samples.iterrows(), 1):
            s_id = row['sample_id']
            img_path = row['image_path']
            mask_path = row['mask_path']
            
            img_bgr = cv2.imread(img_path)
            gt_m = cv2.imread(mask_path, 0)
            if img_bgr is None or gt_m is None:
                oracle_results[s_id] = {'mask': None, 'time_ms': 0.0, 'iou': 0.0, 'dice': 0.0, 'status': 'error'}
                continue
                
            h, w = img_bgr.shape[:2]
            gt_bool = np.squeeze(gt_m > 127)
            if gt_bool.ndim != 2:
                gt_bool = gt_bool.reshape(h, w)
            ys, xs = np.where(gt_bool)
            if len(xs) == 0:
                oracle_results[s_id] = {'mask': None, 'time_ms': 0.0, 'iou': 0.0, 'dice': 0.0, 'status': 'empty_gt'}
                continue
                
            oracle_box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            t0 = time.perf_counter()
            try:
                res = sam_model(img_bgr, bboxes=[oracle_box], device=device, verbose=False)
                t_ms = (time.perf_counter() - t0) * 1000.0
                if res[0].masks is not None and len(res[0].masks) > 0:
                    raw_mask = res[0].masks.data[0].cpu().numpy()
                    pred_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                    iou, dice = compute_iou_dice(pred_mask, gt_bool)
                    status = 'segmented'
                else:
                    pred_mask = None
                    iou, dice = 0.0, 0.0
                    status = 'sam_no_mask'
            except Exception as e:
                t_ms = (time.perf_counter() - t0) * 1000.0
                pred_mask = None
                iou, dice = 0.0, 0.0
                status = f'error_{str(e)[:25]}'
                
            oracle_results[s_id] = {
                'mask': pred_mask,
                'time_ms': round(t_ms, 1),
                'iou': iou,
                'dice': dice,
                'status': status,
                'box': oracle_box
            }
            elapsed = time.perf_counter() - t_start
            avg_s = elapsed / i
            eta_s = int((len(sideview_samples) - i) * avg_s)
            mins, secs = divmod(eta_s, 60)
            print_progress(i, len(sideview_samples), prefix="  Oracle GT -> SAM  ", suffix=f"| IoU: {iou:.3f} | Dice: {dice:.3f} | ETA: {mins:02d}:{secs:02d}")
            
        print()
        del sam_model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # 5. Compile Records and Evaluate SideView Ground Truth
    print("\n" + "-" * 80)
    print("Compiling segmentation records and generating visual composites...")
    
    summary_rows = []
    sideview_metrics = {'RT-DETR_SAM': {'ious': [], 'dices': []},
                        'YOLO26s_seg': {'ious': [], 'dices': []},
                        'Oracle_SAM': {'ious': [], 'dices': []}}
    
    for idx, row in df_samples.iterrows():
        s_id = row['sample_id']
        ds = row['dataset']
        sub = row['category_or_subset']
        img_path = row['image_path']
        mask_path = row['mask_path']
        box_info = primary_boxes.get(s_id, None)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        # Ground truth mask (if SideView)
        gt_mask = None
        if pd.notna(mask_path) and os.path.exists(str(mask_path)):
            gt_raw = cv2.imread(str(mask_path), 0)
            if gt_raw is not None:
                gt_mask = np.squeeze(gt_raw > 127)
                
        # 1. RT-DETR -> SAM
        r_sam = sam_results.get(s_id, {})
        sam_m = r_sam.get('mask', None)
        sam_status = r_sam.get('status', 'unknown')
        sam_time = r_sam.get('time_ms', 0.0)
        
        sam_iou, sam_dice = None, None
        if gt_mask is not None and sam_m is not None:
            sam_iou, sam_dice = compute_iou_dice(sam_m, gt_mask)
            sideview_metrics['RT-DETR_SAM']['ious'].append(sam_iou)
            sideview_metrics['RT-DETR_SAM']['dices'].append(sam_dice)
        elif gt_mask is not None:
            # Mask failed or upstream localization failed
            sideview_metrics['RT-DETR_SAM']['ious'].append(0.0)
            sideview_metrics['RT-DETR_SAM']['dices'].append(0.0)
            
        # 2. YOLO26s-seg
        r_yolo = yolo_results.get(s_id, {})
        yolo_m = r_yolo.get('mask', None)
        yolo_status = r_yolo.get('status', 'unknown')
        yolo_time = r_yolo.get('time_ms', 0.0)
        
        yolo_iou, yolo_dice = None, None
        if gt_mask is not None and yolo_m is not None:
            yolo_iou, yolo_dice = compute_iou_dice(yolo_m, gt_mask)
            sideview_metrics['YOLO26s_seg']['ious'].append(yolo_iou)
            sideview_metrics['YOLO26s_seg']['dices'].append(yolo_dice)
        elif gt_mask is not None:
            sideview_metrics['YOLO26s_seg']['ious'].append(0.0)
            sideview_metrics['YOLO26s_seg']['dices'].append(0.0)
            
        # 3. Oracle SAM
        r_ora = oracle_results.get(s_id, {})
        ora_iou = r_ora.get('iou', None)
        ora_dice = r_ora.get('dice', None)
        if ora_iou is not None:
            sideview_metrics['Oracle_SAM']['ious'].append(ora_iou)
            sideview_metrics['Oracle_SAM']['dices'].append(ora_dice)
            
        summary_rows.append({
            'sample_id': s_id,
            'dataset': ds,
            'category_or_subset': sub,
            'image_path': img_path,
            'rtdetr_box_status': 'box_present' if box_info is not None else 'upstream_localization_failure',
            'sam2_status': sam_status,
            'sam2_time_ms': sam_time,
            'sam2_sideview_iou': sam_iou if sam_iou is not None else '',
            'sam2_sideview_dice': sam_dice if sam_dice is not None else '',
            'yolo26_status': yolo_status,
            'yolo26_time_ms': yolo_time,
            'yolo26_sideview_iou': yolo_iou if yolo_iou is not None else '',
            'yolo26_sideview_dice': yolo_dice if yolo_dice is not None else '',
            'oracle_sam_iou': ora_iou if ora_iou is not None else '',
            'oracle_sam_dice': ora_dice if ora_dice is not None else ''
        })
        
        # Save visual inspection composites
        # Save all in smoke, or 36 diverse samples in expanded
        save_composite = True
        if mode == 'expanded':
            s_num = int(s_id.split('_')[1])
            save_composite = (s_num <= 12) or (100 < s_num <= 112) or (200 < s_num <= 212)
            
        if save_composite:
            box_coords = box_info['box'] if box_info is not None else None
            comp = create_review_composite(
                img_bgr=img_bgr,
                box=box_coords,
                pred_mask=sam_m,
                dataset=ds,
                subset=sub,
                s_id=s_id,
                model_tag="RT-DETR->SAM2.1",
                gt_mask=gt_mask
            )
            comp_path = os.path.join(assets_dir, f"seg_{s_id}_{ds.lower()}_{sub.lower()}.jpg")
            cv2.imwrite(comp_path, comp, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
        print_progress(idx + 1, total, prefix="  Saving Composites ", suffix=f"| Assets: {assets_dir}")
        
    print()
    
    # Save CSV outputs
    df_out = pd.DataFrame(summary_rows)
    out_csv = os.path.join(artifacts_dir, f"segmentation_results_{mode}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nSegmentation results saved: {out_csv}")
    print("-" * 80)
    
    # Quantitative Report for SideView
    if len(sideview_metrics['RT-DETR_SAM']['ious']) > 0:
        print("\nQUANTITATIVE SEGMENTATION METRICS (SideViewCows2026 Ground Truth)")
        print("-" * 80)
        for model_name, m_dict in [('RT-DETR-L -> SAM 2.1', sideview_metrics['RT-DETR_SAM']),
                                    ('Oracle GT Box -> SAM 2.1', sideview_metrics['Oracle_SAM']),
                                    ('YOLO26s-seg (Fast Baseline)', sideview_metrics['YOLO26s_seg'])]:
            ious = np.array(m_dict['ious'])
            dices = np.array(m_dict['dices'])
            if len(ious) > 0:
                print(f"[{model_name}] (N={len(ious)})")
                print(f"  Mean IoU:   {ious.mean():.4f} | Median IoU:   {np.median(ious):.4f}")
                print(f"  Mean Dice:  {dices.mean():.4f} | Median Dice:  {np.median(dices):.4f}")
                print(f"  IoU >= 0.5: {(ious >= 0.5).sum()}/{len(ious)} ({(ious >= 0.5).mean()*100:.1f}%) | "
                      f"IoU >= 0.7: {(ious >= 0.7).sum()}/{len(ious)} ({(ious >= 0.7).mean()*100:.1f}%)")
        print("-" * 80)
        
    # Summary of ScienceDB and MmCows Statuses
    print("\nRAW SEGMENTATION STATUSES (ScienceDB & MmCows)")
    print("-" * 80)
    for ds in ['ScienceDB', 'MmCows']:
        sub_df = df_out[df_out['dataset'] == ds]
        print(f"\n[{ds}] (N={len(sub_df)})")
        print("  RT-DETR -> SAM 2.1 statuses:")
        for st, c in sub_df['sam2_status'].value_counts().items():
            print(f"    - {st:30s}: {c} ({c/len(sub_df)*100:.1f}%)")
        print("  YOLO26s-seg statuses:")
        for st, c in sub_df['yolo26_status'].value_counts().items():
            print(f"    - {st:30s}: {c} ({c/len(sub_df)*100:.1f}%)")
            
    print("\n" + "=" * 80)
    print("SEGMENTATION AUDIT COMPLETE!")
    print("=" * 80)
    return df_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Cattle Segmentation Feasibility Audit")
    parser.add_argument('--mode', choices=['smoke', 'expanded'], default='smoke',
                        help="Audit mode: smoke (30 images) or expanded (300 images)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use for inference")
    parser.add_argument('--resume', action='store_true',
                        help="Resume from existing results if available")
    args = parser.parse_args()
    
    run_segmentation_audit(mode=args.mode, device=args.device, resume=args.resume)
