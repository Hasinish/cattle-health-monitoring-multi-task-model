"""
Step 2.3 — Cattle Pose / Keypoint Feasibility Audit Pipeline.

Evaluates official DeepLabCut SuperAnimal-Quadruped pretrained foundation pose
models (HRNet-W32 and ResNet-50) on primary Phase 3 datasets:
  - ScienceDB (BCS rear views)
  - MmCows (Behavior postures)
  - SideViewCows2026 (Re-ID side views)

Pipeline structure:
  Full image
  -> Step 2.1 RT-DETR-L primary target-cow crop
  -> official SuperAnimal pipeline on crop (max_individuals=1)
  -> map predicted keypoints back to original-image coordinates

Distinguishes 4 failure stages:
  1. upstream_localization_failure
  2. pose_detector_failure
  3. pose_output_returned
  4. pose_inference_error

Usage:
  python scripts/audit_pose_feasibility.py --help
  python scripts/audit_pose_feasibility.py --mode smoke --model hrnet_w32 --device cuda
  python scripts/audit_pose_feasibility.py --mode expanded --model hrnet_w32 --device cuda --resume
  python scripts/audit_pose_feasibility.py --mode expanded --model resnet_50 --device cuda --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2.3 — Cattle Pose / Keypoint Feasibility Audit (Zero-Shot SuperAnimal-Quadruped)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "expanded"],
        default="smoke",
        help="Audit mode: 'smoke' (30 images: 10/dataset) or 'expanded' (300 images: 100/dataset)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hrnet_w32", "resnet_50"],
        default="hrnet_w32",
        help="Pose backbone model: 'hrnet_w32' or 'resnet_50'",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="fasterrcnn_resnet50_fpn_v2",
        help="Detector architecture in SuperAnimal pipeline (default: fasterrcnn_resnet50_fpn_v2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution: skip sample IDs already recorded in the results CSV",
    )
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=0.2,
        help="Visualization and analysis confidence threshold (NOT an accuracy threshold; default: 0.2)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="artifacts/perception_audit/sample_manifest_expanded.csv",
        help="Path to sample manifest CSV",
    )
    parser.add_argument(
        "--localization-csv",
        type=str,
        default="artifacts/perception_audit/localization_detections_expanded.csv",
        help="Path to Step 2.1 localization detections CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/perception_audit",
        help="Directory to save audit CSV results and schema JSON",
    )
    parser.add_argument(
        "--composite-dir",
        type=str,
        default="docs/audits/assets/perception_audit",
        help="Directory to save visual overlay inspection composites",
    )
    parser.add_argument(
        "--save-all-composites",
        action="store_true",
        help="Save visual inspection composites for all images (default: all in smoke, first 10/dataset in expanded)",
    )
    return parser.parse_args()


def load_manifest(manifest_path: str, mode: str) -> pd.DataFrame:
    """Load sample manifest and filter for smoke (30) or expanded (300) mode."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if mode == "smoke":
        # Deterministically select first 10 images from each dataset
        smoke_dfs = []
        for dataset_name in ["ScienceDB", "MmCows", "SideViewCows2026"]:
            ds_df = df[df["dataset"] == dataset_name]
            smoke_dfs.append(ds_df.head(10))
        df = pd.concat(smoke_dfs, ignore_index=True)
    return df


def load_primary_cow_boxes(localization_csv: str, manifest_df: pd.DataFrame) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Load Step 2.1 RT-DETR-L primary cow bounding box for each sample.
    
    Deterministic Primary-Cow Rule:
      Select detection with maximum bounding-box area (x2 - x1) * (y2 - y1).
      Ties broken by higher confidence.
      If no detection: returns None (recorded as upstream_localization_failure).
    """
    if not os.path.exists(localization_csv):
        raise FileNotFoundError(f"Localization CSV not found: {localization_csv}. Run Step 2.1 first!")

    df_det = pd.read_csv(localization_csv)
    rt = df_det[df_det["model_name"] == "RT-DETR-L"].copy()
    rt_valid = rt[rt["detection_status"] == "detected"].copy()

    rt_valid["box_x1"] = pd.to_numeric(rt_valid["box_x1"], errors="coerce")
    rt_valid["box_y1"] = pd.to_numeric(rt_valid["box_y1"], errors="coerce")
    rt_valid["box_x2"] = pd.to_numeric(rt_valid["box_x2"], errors="coerce")
    rt_valid["box_y2"] = pd.to_numeric(rt_valid["box_y2"], errors="coerce")
    rt_valid["confidence"] = pd.to_numeric(rt_valid["confidence"], errors="coerce")

    rt_valid["area"] = (rt_valid["box_x2"] - rt_valid["box_x1"]) * (rt_valid["box_y2"] - rt_valid["box_y1"])
    sorted_det = rt_valid.sort_values(by=["sample_id", "area", "confidence"], ascending=[True, False, False])

    primary_map: Dict[str, Dict[str, Any]] = {}
    for s_id, grp in sorted_det.groupby("sample_id"):
        top = grp.iloc[0]
        primary_map[str(s_id)] = {
            "box": [float(top["box_x1"]), float(top["box_y1"]), float(top["box_x2"]), float(top["box_y2"])],
            "confidence": float(top["confidence"]),
            "num_detections": int(top["num_detections"]),
        }

    sample_boxes: Dict[str, Optional[Dict[str, Any]]] = {}
    for _, row in manifest_df.iterrows():
        s_id = str(row["sample_id"])
        sample_boxes[s_id] = primary_map.get(s_id, None)

    return sample_boxes


def extract_crop(
    img_bgr: np.ndarray, box: List[float]
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Extract target-cow crop from full image, clamping to image bounds.
    Returns (crop_bgr, (x1, y1, x2, y2)) or (None, None) if crop is invalid.
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(w, int(round(box[2])))
    y2 = min(h, int(round(box[3])))

    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 2 or (y2 - y1) < 2:
        return None, None

    crop = img_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def check_points_in_mask(
    points_orig: List[Tuple[float, float, float]], gt_mask: np.ndarray
) -> Tuple[float, int, int]:
    """
    Geometric sanity check for SideViewCows2026:
    Calculate percentage of predicted keypoints falling inside ground-truth cow mask.
    Returns (inside_rate, inside_count, total_count).
    Explicitly labeled keypoints-inside-mask rate (sanity check, NOT accuracy).
    """
    h, w = gt_mask.shape[:2]
    inside_count = 0
    total_count = len(points_orig)
    if total_count == 0:
        return 0.0, 0, 0

    for x, y, _ in points_orig:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            if gt_mask[iy, ix] > 0:
                inside_count += 1

    rate = round(inside_count / total_count, 4)
    return rate, inside_count, total_count


def create_pose_inspection_composite(
    img_bgr: np.ndarray,
    crop_bgr: Optional[np.ndarray],
    crop_coords: Optional[Tuple[int, int, int, int]],
    keypoints_orig: List[Tuple[float, float, float]],
    keypoints_crop: List[Tuple[float, float, float]],
    keypoint_names: List[str],
    skeleton: List[Tuple[int, int]],
    vis_threshold: float,
    sample_id: str,
    dataset: str,
    subset: str,
    model_name: str,
    status: str,
    gt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a 3-panel visual inspection composite:
      Panel 1: Full image with RT-DETR-L crop box (+ GT mask contour if available)
      Panel 2: Cropped cow with predicted keypoints & skeleton
      Panel 3: Full original image with remapped keypoints & skeleton
    """
    h_orig, w_orig = img_bgr.shape[:2]
    panel1 = img_bgr.copy()

    # Draw GT mask contour on Panel 1 if present
    if gt_mask is not None and gt_mask.shape[:2] == (h_orig, w_orig):
        contours, _ = cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel1, contours, -1, (255, 0, 255), 2)  # Magenta GT contour

    # Draw RT-DETR-L box on Panel 1
    if crop_coords is not None:
        x1, y1, x2, y2 = crop_coords
        cv2.rectangle(panel1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(panel1, f"Target Cow Crop: [{x1},{y1},{x2},{y2}]", (x1 + 4, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Panel 2: Crop view
    target_crop_h = 400
    if crop_bgr is not None and crop_bgr.shape[0] > 0 and crop_bgr.shape[1] > 0:
        ch, cw = crop_bgr.shape[:2]
        crop_disp = crop_bgr.copy()

        # Draw skeleton lines on crop if available
        if skeleton and len(keypoints_crop) == len(keypoint_names):
            for idx1, idx2 in skeleton:
                if idx1 < len(keypoints_crop) and idx2 < len(keypoints_crop):
                    p1 = keypoints_crop[idx1]
                    p2 = keypoints_crop[idx2]
                    if p1[2] >= vis_threshold and p2[2] >= vis_threshold:
                        cv2.line(crop_disp, (int(round(p1[0])), int(round(p1[1]))),
                                 (int(round(p2[0])), int(round(p2[1]))), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw keypoint markers on crop
        for idx, (kx, ky, conf) in enumerate(keypoints_crop):
            ix, iy = int(round(kx)), int(round(ky))
            if 0 <= ix < cw and 0 <= iy < ch:
                if conf >= vis_threshold:
                    cv2.circle(crop_disp, (ix, iy), 4, (0, 255, 0), -1, cv2.LINE_AA)  # Green for >= threshold
                else:
                    cv2.circle(crop_disp, (ix, iy), 3, (0, 0, 255), -1, cv2.LINE_AA)  # Red for < threshold
    else:
        crop_disp = np.zeros((target_crop_h, target_crop_h, 3), dtype=np.uint8)
        cv2.putText(crop_disp, f"No Valid Crop ({status})", (20, target_crop_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Panel 3: Remapped keypoints on full image
    panel3 = img_bgr.copy()
    if status == "pose_output_returned" and len(keypoints_orig) > 0:
        # Draw skeleton lines on full image
        if skeleton and len(keypoints_orig) == len(keypoint_names):
            for idx1, idx2 in skeleton:
                if idx1 < len(keypoints_orig) and idx2 < len(keypoints_orig):
                    p1 = keypoints_orig[idx1]
                    p2 = keypoints_orig[idx2]
                    if p1[2] >= vis_threshold and p2[2] >= vis_threshold:
                        cv2.line(panel3, (int(round(p1[0])), int(round(p1[1]))),
                                 (int(round(p2[0])), int(round(p2[1]))), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw keypoint markers on full image
        for idx, (kx, ky, conf) in enumerate(keypoints_orig):
            ix, iy = int(round(kx)), int(round(ky))
            if 0 <= ix < w_orig and 0 <= iy < h_orig:
                if conf >= vis_threshold:
                    cv2.circle(panel3, (ix, iy), 4, (0, 255, 0), -1, cv2.LINE_AA)
                else:
                    cv2.circle(panel3, (ix, iy), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # Normalize panel heights for side-by-side concatenation
    target_h = 480
    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        new_w = max(1, int(round(iw * (h / ih))))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    p1_res = resize_to_h(panel1, target_h)
    p2_res = resize_to_h(crop_disp, target_h)
    p3_res = resize_to_h(panel3, target_h)

    # Add header bars to each panel
    def add_panel_header(img: np.ndarray, title: str, color: Tuple[int, int, int]) -> np.ndarray:
        header = np.zeros((32, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return np.vstack([header, img])

    p1_hdr = add_panel_header(p1_res, "Panel 1: RT-DETR-L Crop & Box", (0, 255, 0))
    p2_hdr = add_panel_header(p2_res, f"Panel 2: SuperAnimal Crop Pose ({model_name})", (0, 255, 255))
    p3_hdr = add_panel_header(p3_res, f"Panel 3: Remapped Keypoints (Thresh={vis_threshold})", (255, 200, 0))

    composite = np.hstack([p1_hdr, p2_hdr, p3_hdr])

    # Global title bar
    global_hdr = np.zeros((36, composite.shape[1], 3), dtype=np.uint8)
    title_text = f"{sample_id} | {dataset} ({subset}) | Model: {model_name} | Status: {status} | Green: >= {vis_threshold}, Red: < {vis_threshold}"
    cv2.putText(global_hdr, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    final_composite = np.vstack([global_hdr, composite])
    return final_composite


def format_time(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run_audit(args: argparse.Namespace) -> None:
    """Main execution function for Step 2.3 Pose Feasibility Audit."""
    print("=" * 80)
    print("STEP 2.3 — Cattle Pose / Keypoint Feasibility Audit")
    print(f"Mode:               {args.mode.upper()}")
    print(f"Pose Backbone:      {args.model}")
    print(f"Detector:           {args.detector}")
    print(f"Device:             {args.device}")
    print(f"Resume:             {args.resume}")
    print(f"Analysis Threshold: {args.vis_threshold} (Raw confidence threshold; NOT accuracy)")
    print("=" * 80)

    # 1. Check & Load Dependencies
    try:
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
            create_superanimal_inference_runners,
        )
    except ImportError as e:
        print(f"\n[ERROR] DeepLabCut is not installed or import failed: {e}")
        print("Please install DeepLabCut with ModelZoo support using:")
        print("  pip install deeplabcut dlclibrary")
        sys.exit(1)

    print(f"[INFO] DeepLabCut Version: {deeplabcut.__version__}")

    # 2. Output Directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.composite_dir, exist_ok=True)

    results_csv_path = os.path.join(args.output_dir, f"pose_results_{args.mode}_{args.model}.csv")
    keypoints_csv_path = os.path.join(args.output_dir, f"pose_keypoints_{args.mode}_{args.model}.csv")
    schema_json_path = os.path.join(args.output_dir, "superanimal_quadruped_schema.json")

    # 3. Load Sample Manifest & Step 2.1 Boxes
    manifest_df = load_manifest(args.manifest, args.mode)
    total_samples = len(manifest_df)
    print(f"[INFO] Loaded {total_samples} samples for mode '{args.mode}'.")
    for ds_name, grp in manifest_df.groupby("dataset"):
        print(f"  - {ds_name}: {len(grp)} images")

    primary_boxes = load_primary_cow_boxes(args.localization_csv, manifest_df)

    # 4. Resume Handling
    completed_sample_ids = set()
    if args.resume and os.path.exists(results_csv_path):
        try:
            existing_results = pd.read_csv(results_csv_path)
            if "sample_id" in existing_results.columns:
                completed_sample_ids = set(existing_results["sample_id"].astype(str).tolist())
                print(f"[INFO] Resuming: found {len(completed_sample_ids)} already completed samples.")
        except Exception as e:
            print(f"[WARNING] Could not read existing results CSV for resume: {e}")

    # 5. Initialize Official DeepLabCut SuperAnimal-Quadruped Pipeline
    print(f"\n[INFO] Initializing official SuperAnimal-Quadruped pipeline ({args.model})...")
    init_start = time.time()
    try:
        pose_runner, det_runner, model_cfg = create_superanimal_inference_runners(
            superanimal_name="superanimal_quadruped",
            model_name=args.model,
            detector_name=args.detector,
            max_individuals=1,
            batch_size=1,
            detector_batch_size=1,
            device=args.device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize SuperAnimal runners: {e}")
        sys.exit(1)
    init_elapsed = time.time() - init_start
    print(f"[INFO] Pipeline initialized in {init_elapsed:.2f}s.")

    # 6. Extract Dynamic Keypoint Schema & Skeleton
    keypoint_names = model_cfg["metadata"]["bodyparts"]
    skeleton_raw = model_cfg.get("skeleton", [])
    skeleton_indices: List[Tuple[int, int]] = []
    if skeleton_raw:
        for bpt1, bpt2 in skeleton_raw:
            if bpt1 in keypoint_names and bpt2 in keypoint_names:
                skeleton_indices.append((keypoint_names.index(bpt1), keypoint_names.index(bpt2)))

    schema_data = {
        "model_name": args.model,
        "superanimal_name": "superanimal_quadruped",
        "detector_name": args.detector,
        "num_keypoints": len(keypoint_names),
        "keypoint_names": keypoint_names,
        "skeleton_connections": skeleton_raw,
        "skeleton_indices": skeleton_indices,
    }
    with open(schema_json_path, "w") as f:
        json.dump(schema_data, f, indent=2)
    print(f"[INFO] Saved official keypoint schema ({len(keypoint_names)} keypoints) to {schema_json_path}.")

    # 7. Initialize CSV Headers if creating fresh files
    results_headers = [
        "sample_id",
        "dataset",
        "category_or_subset",
        "image_path",
        "model_name",
        "detector_name",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "pose_detector_status",
        "inference_status",
        "inference_time_ms",
        "num_returned_keypoints",
        "mean_raw_confidence",
        "median_raw_confidence",
        "num_below_vis_threshold",
        "num_above_vis_threshold",
        "vis_threshold",
        "keypoints_inside_mask_rate",
        "failure_reason",
    ]

    keypoint_headers = [
        "sample_id",
        "dataset",
        "model_name",
        "keypoint_idx",
        "keypoint_name",
        "x_crop",
        "y_crop",
        "x_orig",
        "y_orig",
        "raw_confidence",
        "is_above_vis_threshold",
        "inside_gt_mask",
    ]

    if not os.path.exists(results_csv_path) or not args.resume:
        pd.DataFrame(columns=results_headers).to_csv(results_csv_path, index=False)
    if not os.path.exists(keypoints_csv_path) or not args.resume:
        pd.DataFrame(columns=keypoint_headers).to_csv(keypoints_csv_path, index=False)

    # 8. Execution Loop
    print("\nStarting Pose Feasibility Audit Loop:")
    print("-" * 80)

    start_time = time.time()
    processed_count = 0
    model_display = "HRNet-W32" if args.model == "hrnet_w32" else "ResNet-50"

    # Dataset counter tracking for terminal display: e.g. "47/100"
    dataset_totals = manifest_df["dataset"].value_counts().to_dict()
    dataset_counters: Dict[str, int] = {ds: 0 for ds in dataset_totals}

    for idx, (_, row) in enumerate(manifest_df.iterrows(), 1):
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        subset = str(row.get("category_or_subset", "default"))
        img_path = str(row["image_path"])

        dataset_counters[dataset] += 1
        ds_curr = dataset_counters[dataset]
        ds_tot = dataset_totals[dataset]

        if sample_id in completed_sample_ids:
            continue

        processed_count += 1
        iter_start = time.time()

        # Defaults
        crop_coords: Optional[Tuple[int, int, int, int]] = None
        crop_bgr: Optional[np.ndarray] = None
        status = "upstream_localization_failure"
        detector_status = "not_run"
        failure_reason = ""
        keypoints_orig: List[Tuple[float, float, float]] = []
        keypoints_crop: List[Tuple[float, float, float]] = []
        inference_time_ms = 0.0
        inside_mask_rate: Optional[float] = None
        inside_mask_flags: List[Optional[bool]] = [None] * len(keypoint_names)

        # Load original image
        if not os.path.exists(img_path):
            status = "pose_inference_error"
            failure_reason = f"Image file not found: {img_path}"
        else:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                status = "pose_inference_error"
                failure_reason = f"cv2.imread returned None: {img_path}"
            else:
                h_orig, w_orig = img_bgr.shape[:2]

                # Check upstream localization
                box_info = primary_boxes.get(sample_id, None)
                if box_info is None:
                    status = "upstream_localization_failure"
                    failure_reason = "RT-DETR-L did not provide target-cow crop"
                else:
                    crop_bgr, crop_coords = extract_crop(img_bgr, box_info["box"])
                    if crop_bgr is None or crop_coords is None:
                        status = "upstream_localization_failure"
                        failure_reason = f"Invalid crop coordinates: {box_info['box']}"
                    else:
                        # Official SuperAnimal inference on crop
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        t0 = time.time()
                        try:
                            # 1. Run internal detector on crop
                            det_preds = det_runner.inference([crop_rgb]) if det_runner is not None else None

                            # Inspect detector output
                            has_valid_box = False
                            if det_preds is not None and len(det_preds) > 0:
                                d_pred = det_preds[0]
                                bboxes = d_pred.get("bboxes", [])
                                scores = d_pred.get("bbox_scores", [])
                                if len(bboxes) > 0 and (len(scores) == 0 or np.max(scores) > 0.0):
                                    has_valid_box = True

                            if not has_valid_box and det_runner is not None:
                                detector_status = "detector_no_detection"
                                status = "pose_detector_failure"
                                failure_reason = "SuperAnimal internal detector found 0 cows in crop"
                            else:
                                detector_status = "detector_detected"
                                # 2. Run pose runner on crop
                                pose_inputs = [(crop_rgb, det_preds[0])] if det_preds is not None else [crop_rgb]
                                pose_preds = pose_runner.inference(pose_inputs)

                                if pose_preds is None or len(pose_preds) == 0:
                                    status = "pose_inference_error"
                                    failure_reason = "pose_runner returned empty predictions"
                                else:
                                    p_pred = pose_preds[0]
                                    raw_bpts = p_pred.get("bodyparts", None)
                                    if raw_bpts is None or len(raw_bpts) == 0:
                                        status = "pose_inference_error"
                                        failure_reason = "Predictions dictionary missing 'bodyparts' key"
                                    else:
                                        # Shape is typically (max_individuals, num_bpts, 3) or (num_bpts, 3)
                                        bpts_arr = np.asarray(raw_bpts)
                                        if bpts_arr.ndim == 3:
                                            bpts_arr = bpts_arr[0]  # individual 0

                                        num_pts = min(len(keypoint_names), len(bpts_arr))
                                        x1, y1 = crop_coords[0], crop_coords[1]

                                        for k_idx in range(num_pts):
                                            kx_crop = float(bpts_arr[k_idx, 0])
                                            ky_crop = float(bpts_arr[k_idx, 1])
                                            kconf = float(bpts_arr[k_idx, 2])

                                            kx_orig = kx_crop + x1
                                            ky_orig = ky_crop + y1

                                            keypoints_crop.append((kx_crop, ky_crop, kconf))
                                            keypoints_orig.append((kx_orig, ky_orig, kconf))

                                        status = "pose_output_returned"

                        except Exception as e:
                            status = "pose_inference_error"
                            failure_reason = f"Exception during pose inference: {type(e).__name__}: {str(e)}"

                        inference_time_ms = round((time.time() - t0) * 1000.0, 2)

                # SideView Ground-Truth Mask Sanity Check
                gt_mask: Optional[np.ndarray] = None
                if dataset == "SideViewCows2026":
                    mask_path = row.get("mask_path", "")
                    if pd.notna(mask_path) and os.path.exists(str(mask_path)):
                        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if gt_mask is not None and len(keypoints_orig) > 0:
                            inside_mask_rate, inside_cnt, _ = check_points_in_mask(keypoints_orig, gt_mask)
                            # Flag each keypoint individually
                            h_m, w_m = gt_mask.shape[:2]
                            for ki, (kx, ky, _) in enumerate(keypoints_orig):
                                ix, iy = int(round(kx)), int(round(ky))
                                if 0 <= ix < w_m and 0 <= iy < h_m:
                                    inside_mask_flags[ki] = bool(gt_mask[iy, ix] > 0)
                                else:
                                    inside_mask_flags[ki] = False

                # Visual inspection composite
                save_composite = args.save_all_composites or (args.mode == "smoke") or (ds_curr <= 10)
                if save_composite:
                    comp_path = os.path.join(
                        args.composite_dir, f"pose_{args.mode}_{args.model}_{sample_id}.jpg"
                    )
                    comp_img = create_pose_inspection_composite(
                        img_bgr=img_bgr,
                        crop_bgr=crop_bgr,
                        crop_coords=crop_coords,
                        keypoints_orig=keypoints_orig,
                        keypoints_crop=keypoints_crop,
                        keypoint_names=keypoint_names,
                        skeleton=skeleton_indices,
                        vis_threshold=args.vis_threshold,
                        sample_id=sample_id,
                        dataset=dataset,
                        subset=subset,
                        model_name=args.model,
                        status=status,
                        gt_mask=gt_mask,
                    )
                    cv2.imwrite(comp_path, comp_img)

        # 9. Compute Confidence Summary Statistics
        if status == "pose_output_returned" and len(keypoints_orig) > 0:
            raw_confs = [p[2] for p in keypoints_orig]
            mean_conf = round(float(np.mean(raw_confs)), 4)
            median_conf = round(float(np.median(raw_confs)), 4)
            below_vis = int(sum(c < args.vis_threshold for c in raw_confs))
            above_vis = int(sum(c >= args.vis_threshold for c in raw_confs))
            num_returned = len(keypoints_orig)
        else:
            mean_conf = None
            median_conf = None
            below_vis = 0
            above_vis = 0
            num_returned = 0

        # 10. Append to CSVs incrementally (Crash Safety)
        row_res = {
            "sample_id": sample_id,
            "dataset": dataset,
            "category_or_subset": subset,
            "image_path": img_path,
            "model_name": args.model,
            "detector_name": args.detector,
            "crop_x1": crop_coords[0] if crop_coords else None,
            "crop_y1": crop_coords[1] if crop_coords else None,
            "crop_x2": crop_coords[2] if crop_coords else None,
            "crop_y2": crop_coords[3] if crop_coords else None,
            "pose_detector_status": detector_status,
            "inference_status": status,
            "inference_time_ms": inference_time_ms,
            "num_returned_keypoints": num_returned,
            "mean_raw_confidence": mean_conf,
            "median_raw_confidence": median_conf,
            "num_below_vis_threshold": below_vis,
            "num_above_vis_threshold": above_vis,
            "vis_threshold": args.vis_threshold,
            "keypoints_inside_mask_rate": inside_mask_rate,
            "failure_reason": failure_reason,
        }
        pd.DataFrame([row_res]).to_csv(results_csv_path, mode="a", header=False, index=False)

        # Append keypoints if returned
        if status == "pose_output_returned" and len(keypoints_orig) > 0:
            kpt_rows = []
            for ki, k_name in enumerate(keypoint_names):
                if ki < len(keypoints_orig):
                    kx_c, ky_c, conf = keypoints_crop[ki]
                    kx_o, ky_o, _ = keypoints_orig[ki]
                    kpt_rows.append({
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "model_name": args.model,
                        "keypoint_idx": ki,
                        "keypoint_name": k_name,
                        "x_crop": round(kx_c, 2),
                        "y_crop": round(ky_c, 2),
                        "x_orig": round(kx_o, 2),
                        "y_orig": round(ky_o, 2),
                        "raw_confidence": round(conf, 4),
                        "is_above_vis_threshold": bool(conf >= args.vis_threshold),
                        "inside_gt_mask": inside_mask_flags[ki],
                    })
            if kpt_rows:
                pd.DataFrame(kpt_rows).to_csv(keypoints_csv_path, mode="a", header=False, index=False)

        # 11. Terminal Progress Display
        elapsed_sec = time.time() - start_time
        avg_sec_per_sample = elapsed_sec / processed_count
        remaining_samples = total_samples - (len(completed_sample_ids) + processed_count)
        eta_sec = max(0, remaining_samples * avg_sec_per_sample)
        pct = int(round((ds_curr / ds_tot) * 100))

        status_tag = "OK" if status == "pose_output_returned" else status
        display_line = (
            f"{model_display} | {dataset:16s} | {sample_id} | "
            f"{ds_curr:2d}/{ds_tot:2d} | {pct:3d}% | "
            f"elapsed {format_time(elapsed_sec)} | ETA {format_time(eta_sec)} | {status_tag}"
        )
        print(display_line)

    print("-" * 80)
    print(f"[COMPLETED] Feasibility audit finished in {format_time(time.time() - start_time)}.")
    print(f"Results saved to:    {results_csv_path}")
    print(f"Keypoints saved to:  {keypoints_csv_path}")
    print(f"Composites saved to: {args.composite_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_audit(args)
