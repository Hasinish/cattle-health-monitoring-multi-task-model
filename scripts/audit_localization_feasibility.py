"""
Phase 3 Step 2.1: Cattle Localization Feasibility Audit
Evaluates pretrained off-the-shelf object detectors (YOLOv8s, Faster R-CNN v2, RT-DETR-L)
on representative samples from ScienceDB (BCS), MmCows (Behavior), and SideViewCows2026 (Re-ID).

Usage:
    python scripts/audit_localization_feasibility.py --mode smoke
    python scripts/audit_localization_feasibility.py --mode expanded
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
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from ultralytics import YOLO, RTDETR

# Ensure deterministic sampling
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def print_progress(current, total, prefix="", suffix="", bar_len=30):
    """Clean single-line progress bar for Windows terminal without UI glitches."""
    frac = current / total if total > 0 else 0
    filled = int(round(bar_len * frac))
    bar = "=" * filled + "-" * (bar_len - filled)
    percent = frac * 100.0
    line = f"\r{prefix} [{bar}] {percent:5.1f}% ({current}/{total}) {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()

def sample_sciencedb(n_samples=30, seed=SEED):
    """Sample representative ScienceDB images covering all 5 BCS classes and diverse burst groups."""
    csv_path = "datasets/bcs/sciencedb/test.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    
    df = pd.read_csv(csv_path)
    classes = [3.25, 3.5, 3.75, 4.0, 4.25]
    per_class = n_samples // len(classes)
    remainder = n_samples % len(classes)
    
    samples = []
    rng = np.random.RandomState(seed)
    
    for i, cls_val in enumerate(classes):
        cls_df = df[df['label'] == cls_val].copy()
        unique_groups = cls_df['burst_group_id'].unique()
        k = per_class + (1 if i < remainder else 0)
        selected_groups = rng.choice(unique_groups, size=min(k, len(unique_groups)), replace=False)
        
        for grp in selected_groups:
            grp_df = cls_df[cls_df['burst_group_id'] == grp]
            row = grp_df.sample(1, random_state=seed).iloc[0]
            samples.append({
                'dataset': 'ScienceDB',
                'dataset_role': 'BCS (Primary)',
                'image_path': row['image_path'],
                'mask_path': '',
                'category_or_subset': f"BCS_{row['label']}",
                'burst_or_session': row['burst_group_id'],
                'metadata': json.dumps({
                    'bcs_label': float(row['label']),
                    'burst_group_id': str(row['burst_group_id']),
                    'original_passage_id': str(row['original_passage_id']),
                    'farm_source': str(row['farm_source'])
                })
            })
    return pd.DataFrame(samples)

def sample_mmcows(n_samples=30, seed=SEED):
    """Sample representative MmCows behavior crops covering 7 behaviors, 4 cameras, and diverse cows."""
    csv_path = "datasets/behavior/mmcows/manifest.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    
    df = pd.read_csv(csv_path)
    if n_samples == 30:
        alloc = {'Lying': 5, 'Standing': 5, 'Feeding_head_down': 5, 'Feeding_head_up': 5, 'Walking': 4, 'Drinking': 3, 'Licking': 3}
    else:
        alloc = {'Lying': 16, 'Standing': 16, 'Feeding_head_down': 16, 'Feeding_head_up': 16, 'Walking': 14, 'Drinking': 11, 'Licking': 11}
        
    samples = []
    for beh, count in alloc.items():
        beh_df = df[df['class_name'] == beh].copy()
        sampled_rows = beh_df.sample(n=min(count, len(beh_df)), random_state=seed)
        for _, row in sampled_rows.iterrows():
            samples.append({
                'dataset': 'MmCows',
                'dataset_role': 'Behavior (Primary)',
                'image_path': row['image_path'],
                'mask_path': '',
                'category_or_subset': f"Behavior_{row['class_name']}",
                'burst_or_session': f"cow_{row['cow_id']}_cam_{row['camera_id']}",
                'metadata': json.dumps({
                    'behavior': str(row['class_name']),
                    'cow_id': int(row['cow_id']),
                    'camera_id': int(row['camera_id']),
                    'timestamp_iso': str(row['timestamp_iso'])
                })
            })
    return pd.DataFrame(samples)

def sample_sideview(n_samples=30, seed=SEED):
    """Sample representative SideViewCows2026 images covering parlor, barn, snapshots, and diverse cows."""
    csv_path = "datasets/id/sideviewcows2026/manifest.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
        
    df = pd.read_csv(csv_path)
    if n_samples == 30:
        alloc = {'parlor': 10, 'barn': 10, 'snapshots': 10}
    else:
        alloc = {'parlor': 40, 'barn': 40, 'snapshots': 20}
        
    samples = []
    for subset, count in alloc.items():
        sub_df = df[df['subset'] == subset].copy()
        sampled_rows = sub_df.sample(n=min(count, len(sub_df)), random_state=seed)
        for _, row in sampled_rows.iterrows():
            samples.append({
                'dataset': 'SideViewCows2026',
                'dataset_role': 'Re-ID (Primary)',
                'image_path': row['image_path'],
                'mask_path': row['mask_path'],
                'category_or_subset': f"ReID_{row['subset']}",
                'burst_or_session': str(row['recording_id']),
                'metadata': json.dumps({
                    'subset': str(row['subset']),
                    'individual_id': int(row['individual_id']),
                    'frame_no': int(row['frame_no']),
                    'recording_id': str(row['recording_id'])
                })
            })
    return pd.DataFrame(samples)

def get_sample_manifest(mode='smoke', seed=SEED):
    """Build unified deterministic sample manifest."""
    n_per_dataset = 30 if mode == 'smoke' else 100
    df_s = sample_sciencedb(n_samples=n_per_dataset, seed=seed)
    df_m = sample_mmcows(n_samples=n_per_dataset, seed=seed)
    df_v = sample_sideview(n_samples=n_per_dataset, seed=seed)
    
    df_all = pd.concat([df_s, df_m, df_v], ignore_index=True)
    df_all.insert(0, 'sample_id', [f"sample_{i+1:04d}" for i in range(len(df_all))])
    return df_all

def draw_boxes(image, boxes, scores, label_prefix="Cow", color=(0, 255, 0), thickness=2):
    """Draw bounding boxes with label tags on an image copy."""
    img_out = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness)
        tag = f"{label_prefix}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img_out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_out, tag, (x1 + 2, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img_out

def run_candidate_inference(model_name, model_info, df_samples, device, conf_thresh):
    """Run a single candidate model across all samples with clean progress tracking and VRAM management."""
    print(f"\n[{model_name}] Initializing model on {device}...")
    
    if model_info['type'] == 'yolo':
        model = YOLO(model_info['weights'])
    elif model_info['type'] == 'rtdetr':
        model = RTDETR(model_info['weights'])
    elif model_info['type'] == 'torchvision':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        
    cow_class = model_info['cow_class']
    total = len(df_samples)
    
    results = {}
    total_time = 0.0
    
    for i, (_, row) in enumerate(df_samples.iterrows(), 1):
        s_id = row['sample_id']
        img_path = row['image_path']
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            results[s_id] = {'boxes': [], 'scores': [], 'time_ms': 0.0, 'status': 'error_load'}
            continue
            
        t0 = time.perf_counter()
        boxes = []
        scores = []
        status = 'detected'
        
        try:
            if model_info['type'] in ['yolo', 'rtdetr']:
                res = model(img_bgr, conf=conf_thresh, classes=[cow_class], verbose=False, device=device)
                b_objs = res[0].boxes
                if len(b_objs) > 0:
                    boxes = b_objs.xyxy.cpu().numpy()
                    scores = b_objs.conf.cpu().numpy()
            elif model_info['type'] == 'torchvision':
                tensor_img = TF.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(device)
                with torch.no_grad():
                    preds = model([tensor_img])[0]
                labels = preds['labels'].cpu().numpy()
                pred_scores = preds['scores'].cpu().numpy()
                pred_boxes = preds['boxes'].cpu().numpy()
                keep = (labels == cow_class) & (pred_scores >= conf_thresh)
                boxes = pred_boxes[keep]
                scores = pred_scores[keep]
                
            status = 'detected' if len(boxes) > 0 else 'no_detection'
        except Exception as e:
            status = f'error_{str(e)[:25]}'
            boxes = []
            scores = []
            
        t_ms = (time.perf_counter() - t0) * 1000.0
        total_time += t_ms
        
        results[s_id] = {
            'boxes': boxes,
            'scores': scores,
            'time_ms': round(t_ms, 2),
            'status': status,
            'w': img_bgr.shape[1],
            'h': img_bgr.shape[0]
        }
        
        # Progress reporting
        avg_ms = total_time / i
        eta_s = int((total - i) * (avg_ms / 1000.0))
        mins, secs = divmod(eta_s, 60)
        cows_found = len(boxes)
        suffix = f"| Lat: {t_ms:5.1f}ms (avg {avg_ms:5.1f}ms) | Cows: {cows_found} | ETA: {mins:02d}:{secs:02d}"
        print_progress(i, total, prefix=f"  {model_name:14s}", suffix=suffix)
        
    print() # newline after progress bar
    
    # Free model and purge CUDA cache to prevent VRAM accumulation
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def run_audit(mode='smoke', device='cuda', conf_thresh=0.25):
    print("=" * 80)
    print(f"CATTLE LOCALIZATION FEASIBILITY AUDIT [Mode: {mode.upper()}]")
    print(f"Device: {device} | Confidence Threshold: {conf_thresh}")
    print("=" * 80)
    
    artifacts_dir = "artifacts/perception_audit"
    assets_dir = "docs/audits/assets/perception_audit"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Deterministic Sampling
    df_samples = get_sample_manifest(mode=mode, seed=SEED)
    manifest_csv = os.path.join(artifacts_dir, f"sample_manifest_{mode}.csv")
    df_samples.to_csv(manifest_csv, index=False)
    print(f"\nSample manifest generated: {len(df_samples)} images ({manifest_csv})")
    for ds, count in df_samples['dataset'].value_counts().items():
        print(f"  - {ds:18s}: {count} images")
    print("-" * 80)
    
    # Candidate configurations
    candidates = {
        'YOLOv8s': {
            'type': 'yolo',
            'weights': 'yolov8s.pt',
            'cow_class': 19,
            'color': (0, 255, 0)       # Green
        },
        'FasterRCNN_v2': {
            'type': 'torchvision',
            'weights': 'DEFAULT',
            'cow_class': 21,
            'color': (255, 128, 0)     # Blue/Cyan
        },
        'RT-DETR-L': {
            'type': 'rtdetr',
            'weights': 'rtdetr-l.pt',
            'cow_class': 19,
            'color': (0, 165, 255)     # Orange
        }
    }
    
    # Run each model sequentially to keep VRAM usage low and UI clean
    all_model_results = {}
    for m_name, m_info in candidates.items():
        all_model_results[m_name] = run_candidate_inference(
            m_name, m_info, df_samples, device, conf_thresh
        )
        
    print("\n" + "-" * 80)
    print("Exporting detections and generating visual inspection composites...")
    
    detections_records = []
    summary_records = []
    
    # Composite selection: all 90 in smoke, or 36 diverse representative in expanded
    total_samples = len(df_samples)
    for idx, row in df_samples.iterrows():
        s_id = row['sample_id']
        dataset = row['dataset']
        img_path = row['image_path']
        mask_path = row['mask_path']
        cat_subset = row['category_or_subset']
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        
        # Prepare panels for composite
        preview_panels = {'Original': img_bgr.copy()}
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview_panels['Original'], contours, -1, (0, 255, 255), 2)
                
        for m_name, m_info in candidates.items():
            res = all_model_results[m_name][s_id]
            boxes = res['boxes']
            scores = res['scores']
            t_ms = res['time_ms']
            status = res['status']
            
            # Record detections
            if len(boxes) == 0:
                detections_records.append({
                    'sample_id': s_id,
                    'dataset': dataset,
                    'category_or_subset': cat_subset,
                    'image_path': img_path,
                    'model_name': m_name,
                    'detection_idx': 0,
                    'num_detections': 0,
                    'detection_status': status,
                    'box_x1': '',
                    'box_y1': '',
                    'box_x2': '',
                    'box_y2': '',
                    'confidence': '',
                    'inference_time_ms': t_ms,
                    'image_width': w,
                    'image_height': h
                })
            else:
                for d_idx, (b, s) in enumerate(zip(boxes, scores)):
                    detections_records.append({
                        'sample_id': s_id,
                        'dataset': dataset,
                        'category_or_subset': cat_subset,
                        'image_path': img_path,
                        'model_name': m_name,
                        'detection_idx': d_idx + 1,
                        'num_detections': len(boxes),
                        'detection_status': status,
                        'box_x1': round(float(b[0]), 2),
                        'box_y1': round(float(b[1]), 2),
                        'box_x2': round(float(b[2]), 2),
                        'box_y2': round(float(b[3]), 2),
                        'confidence': round(float(s), 4),
                        'inference_time_ms': t_ms,
                        'image_width': w,
                        'image_height': h
                    })
                    
            top_score = round(float(np.max(scores)), 4) if len(scores) > 0 else ''
            summary_records.append({
                'sample_id': s_id,
                'dataset': dataset,
                'category_or_subset': cat_subset,
                'model_name': m_name,
                'num_detections': len(boxes),
                'top_confidence': top_score,
                'detection_status': status,
                'inference_time_ms': t_ms
            })
            
            # Preview panel
            p_img = draw_boxes(img_bgr, boxes, scores, label_prefix="Cow", color=m_info['color'], thickness=2)
            cv2.putText(p_img, f"{m_name} ({len(boxes)} cows)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            preview_panels[m_name] = p_img

        # Save composite
        save_preview = True
        if mode == 'expanded':
            s_num = int(s_id.split('_')[1])
            save_preview = (s_num <= 12) or (100 < s_num <= 112) or (200 < s_num <= 212)
            
        if save_preview:
            target_h = 360
            resized_panels = []
            for name in ['Original', 'YOLOv8s', 'FasterRCNN_v2', 'RT-DETR-L']:
                p_img = preview_panels[name]
                p_h, p_w = p_img.shape[:2]
                target_w = int(p_w * (target_h / p_h))
                resized = cv2.resize(p_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if name == 'Original':
                    cv2.putText(resized, f"{dataset} | {cat_subset}", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                resized_panels.append(resized)
                
            composite = np.hstack(resized_panels)
            comp_name = f"{s_id}_{dataset.lower()}_{cat_subset.lower()}.jpg"
            cv2.imwrite(os.path.join(assets_dir, comp_name), composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
        print_progress(idx + 1, total_samples, prefix="  Visual Composites", suffix=f"| Saved to {assets_dir}")

    print()
    
    # Save CSVs
    df_detections = pd.DataFrame(detections_records)
    df_summary = pd.DataFrame(summary_records)
    
    det_csv = os.path.join(artifacts_dir, f"localization_detections_{mode}.csv")
    sum_csv = os.path.join(artifacts_dir, f"localization_summary_{mode}.csv")
    df_detections.to_csv(det_csv, index=False)
    df_summary.to_csv(sum_csv, index=False)
    
    print("-" * 80)
    print(f"Results successfully saved:")
    print(f"  - Detections CSV: {det_csv}")
    print(f"  - Summary CSV:    {sum_csv}")
    print("-" * 80)
    
    # Print Automated Statistics
    print("\nAUTOMATED DETECTION STATISTICS (RAW INFERENCE)")
    print("-" * 80)
    for m_name in candidates.keys():
        print(f"\nModel: {m_name}")
        m_df = df_summary[df_summary['model_name'] == m_name]
        for ds in ['ScienceDB', 'MmCows', 'SideViewCows2026']:
            sub = m_df[m_df['dataset'] == ds]
            n_imgs = len(sub)
            n_det = (sub['num_detections'] > 0).sum()
            n_zero = (sub['num_detections'] == 0).sum()
            n_multi = (sub['num_detections'] > 1).sum()
            median_lat = sub['inference_time_ms'].median()
            print(f"  [{ds:16s}] Images: {n_imgs:3d} | >=1 Cow: {n_det:3d} ({n_det/n_imgs*100:5.1f}%) | "
                  f"0 Cows: {n_zero:2d} | Multi-Cow: {n_multi:2d} | Med Latency: {median_lat:5.1f} ms")
                  
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE! REVIEW COMPOSITES AND COMPILE FEASIBILITY VERDICT.")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Cattle Localization Feasibility Audit")
    parser.add_argument('--mode', choices=['smoke', 'expanded'], default='smoke',
                        help="Audit mode: smoke (90 images) or expanded (300 images)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use for inference")
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Confidence threshold for detections")
    args = parser.parse_args()
    
    run_audit(mode=args.mode, device=args.device, conf_thresh=args.conf)
