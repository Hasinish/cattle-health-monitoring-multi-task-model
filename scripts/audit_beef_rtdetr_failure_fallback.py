# -*- coding: utf-8 -*-
"""
Kaggle Beef Fallback Perception Audit:
Testing Single Positive Center Point (112, 112) -> SAM 2.1 Small
on the 3 Known RT-DETR-L Upstream Detection Failures from Fresh-40 Audit.

Target Samples:
1. beef_4_4_clip_0 — Drinking (Session 4, frame 125)
2. beef_131_5_clip_0 — Feeding (Session 131, frame 125)
3. beef_00000000580000000_27_clip_3 — Lying (Session 580000000, frame 125)

Hypothesis / Strategy Tested:
IF RT-DETR-L detects no cow (0 cattle detections at conf >= 0.25)
-> Fallback to SAM 2.1 Small with ONE positive center point (cx, cy) = (112, 112).

Environment:
- Modal Cloud: Profile tigerwood693, GPU Tier T4
- Volume: beef-behavior-data (/data/beef_behavior)
- Model: sam2.1_s.pt

Deliverables:
- artifacts/perception_audit/beef_rtdetr_failure_fallback.csv
- docs/audits/assets/beef_rtdetr_failure_fallback/contact_sheet.jpg
- Individual composites in docs/audits/assets/beef_rtdetr_failure_fallback/
- docs/research_log/2026-09-23_beef_rtdetr_failure_fallback.md
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

app = modal.App("beef-rtdetr-failure-fallback", image=image)


def make_4panel_fallback_composite(
    frame_bgr,
    sample_id,
    beh_canon,
    session_id,
    mid_f,
    point_xy,
    mask,
    mask_returned,
    mask_status,
    ar,
    cc,
    lat_ms,
    sample_idx=1,
):
    """
    Generate 4-panel visual composite:
    [Original | Center Point | SAM Mask | Overlay]
    Composite dimensions: (256, 896, 3) (Header 32px + 4 panels of 224x224)
    """
    img_h, img_w = frame_bgr.shape[:2]

    # Panel 1: Original
    p1 = frame_bgr.copy()
    cv2.putText(p1, "Original", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p1, f"Frame {mid_f}", (8, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel 2: Center Point Prompt Visualization
    p2 = frame_bgr.copy()
    cx, cy = int(round(point_xy[0])), int(round(point_xy[1]))
    # Draw point with ring and crosshair
    cv2.circle(p2, (cx, cy), 6, (0, 255, 255), -1)  # Yellow circle
    cv2.circle(p2, (cx, cy), 6, (0, 0, 0), 1)        # Black outline
    cv2.circle(p2, (cx, cy), 2, (255, 255, 255), -1)  # White center dot
    cv2.line(p2, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(p2, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(p2, "Fallback Prompt", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p2, f"Center Pt ({cx},{cy})", (8, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # Panel 3: SAM Mask (Binary white on black)
    p3 = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if mask_returned and mask is not None and np.sum(mask) > 0:
        p3[mask] = [255, 255, 255]
        cv2.putText(p3, "SAM 2.1 Mask", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(p3, f"pix: {int(np.sum(mask))}", (8, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(p3, "NO MASK", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 2, cv2.LINE_AA)

    # Panel 4: Overlay
    p4 = frame_bgr.copy()
    if mask_returned and mask is not None and np.sum(mask) > 0:
        overlay = p4.copy()
        overlay[mask] = (overlay[mask] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
        p4 = overlay
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p4, contours, -1, (0, 255, 0), 1)

        cv2.putText(p4, "Overlay", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(p4, f"ar:{ar:.3f} cc:{cc}", (8, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(p4, "NO MASK", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # Border lines between panels
    cv2.line(p1, (img_w - 1, 0), (img_w - 1, img_h), (50, 50, 50), 1)
    cv2.line(p2, (img_w - 1, 0), (img_w - 1, img_h), (50, 50, 50), 1)
    cv2.line(p3, (img_w - 1, 0), (img_w - 1, img_h), (50, 50, 50), 1)

    # Combine 4 panels horizontally
    panels = np.hstack([p1, p2, p3, p4])

    # Top title bar
    header_h = 32
    header = np.full((header_h, panels.shape[1], 3), 25, dtype=np.uint8)
    title_text = (
        f"#{sample_idx} | {beh_canon} | {sample_id} | Sess: {session_id} | "
        f"Fallback Pt ({cx},{cy}) -> ar={ar:.3f}, cc={cc}, lat={lat_ms:.1f}ms"
    )
    cv2.putText(header, title_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, panels])


@app.function(
    volumes={"/data/beef_behavior": beef_vol},
    gpu="T4",
    timeout=300,
    cpu=2.0,
    memory=4096,
)
def evaluate_beef_fallback_modal(fallback_records):
    """
    Run SAM 2.1 Small inference with single center point (112, 112) on the 3 RT-DETR failure samples.
    """
    import os
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM

    print(f"[MODAL-FALLBACK] Initializing SAM 2.1 Small on Tesla T4...")
    print(f"[MODAL-FALLBACK] CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

    sam_model = SAM("sam2.1_s.pt")
    beef_root = "/data/beef_behavior"

    results = []

    for idx, rec in enumerate(fallback_records):
        sample_id = rec["sample_id"]
        beh_canon = rec["canonical_behavior"]
        session_id = str(rec["session_id"])
        source_vid = str(rec["source_video_id"])
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
            print(f"[ERROR] Video clip not found: {rel_path}")
            continue

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            print(f"[ERROR] Could not read frame {mid_f} from {vid_path}")
            continue

        img_h, img_w = frame_bgr.shape[:2]

        # Single center positive point prompt
        cx = float(img_w // 2)  # 112.0
        cy = float(img_h // 2)  # 112.0
        pts_raw = [[cx, cy]]
        pts = np.array([pts_raw], dtype=np.float32)  # shape (1, 1, 2)
        lbls = np.array([[1]], dtype=np.int32)       # shape (1, 1)

        t0 = time.perf_counter()
        try:
            res = sam_model(frame_bgr, points=pts, labels=lbls, device="cuda", verbose=False)
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
                    mask_ret = True
                    mask_st = "segmented"
                else:
                    mask = np.zeros((img_h, img_w), dtype=bool)
                    pix = 0
                    cc = 0
                    ar = 0.0
                    mask_ret = False
                    mask_st = "sam_no_mask"
            else:
                mask = np.zeros((img_h, img_w), dtype=bool)
                pix = 0
                cc = 0
                ar = 0.0
                mask_ret = False
                mask_st = "sam_no_mask"
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            mask = np.zeros((img_h, img_w), dtype=bool)
            pix = 0
            cc = 0
            ar = 0.0
            mask_ret = False
            mask_st = f"error_{str(e)[:25]}"

        # Generate 4-panel composite
        comp = make_4panel_fallback_composite(
            frame_bgr=frame_bgr,
            sample_id=sample_id,
            beh_canon=beh_canon,
            session_id=session_id,
            mid_f=mid_f,
            point_xy=(cx, cy),
            mask=mask,
            mask_returned=mask_ret,
            mask_status=mask_st,
            ar=ar,
            cc=cc,
            lat_ms=lat_ms,
            sample_idx=idx + 1,
        )

        _, buf = cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        comp_bytes = buf.tobytes()

        eval_record = {
            "sample_id": sample_id,
            "canonical_behavior": beh_canon,
            "session_id": session_id,
            "source_video_id": source_vid,
            "frame_index": mid_f,
            "prompt_strategy": "fallback_single_center_point",
            "exact_prompt_coordinates": json.dumps({"points": pts_raw, "labels": [1]}),
            "mask_returned": mask_ret,
            "mask_status": mask_st,
            "inference_time_ms": round(lat_ms, 2),
            "mask_area_ratio": round(ar, 4),
            "number_of_connected_components": cc,
            "original_rtdetr_failure": True,
        }

        results.append({
            "sample_id": sample_id,
            "canonical_behavior": beh_canon,
            "sample_idx": idx + 1,
            "record": eval_record,
            "comp_bytes": comp_bytes,
        })

        print(f"  [Fallback {idx+1}/3] {sample_id} ({beh_canon:8s}) -> ret={mask_ret}, area_ratio={ar:.4f}, cc={cc}, lat={lat_ms:.1f}ms")

    return results


def build_fallback_master_contact_sheet(
    sample_composites,
    tile_w=896,
    tile_h=256,
    header_h=80,
):
    """
    Build master contact sheet stacking the 3 fallback sample composites vertically.
    Grid: 1 column x 3 rows.
    Sheet dimensions: (3 * 256 + 80, 896, 3) = (848, 896, 3)
    """
    sheet_w = tile_w
    sheet_h = len(sample_composites) * tile_h + header_h

    sheet = np.full((sheet_h, sheet_w, 3), 20, dtype=np.uint8)

    # Master Header
    cv2.rectangle(sheet, (0, 0), (sheet_w, 50), (12, 12, 12), -1)
    cv2.putText(
        sheet,
        "Kaggle Beef Fallback: Center Point (112,112) on RT-DETR Failures",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Sub-header
    cv2.rectangle(sheet, (0, 50), (sheet_w, header_h), (28, 28, 28), -1)
    cv2.putText(
        sheet,
        "3 Known RT-DETR-L Misses from Fresh-40 Audit | SAM 2.1 Small | Status: PENDING HUMAN REVIEW",
        (16, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )

    # Stack the 3 composites
    for idx, comp_bytes in enumerate(sample_composites):
        y_pos = header_h + idx * tile_h
        img_arr = cv2.imdecode(np.frombuffer(comp_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_arr is not None:
            if img_arr.shape[1] != tile_w or img_arr.shape[0] != tile_h:
                tile_resized = cv2.resize(img_arr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            else:
                tile_resized = img_arr
            sheet[y_pos : y_pos + tile_h, 0 : tile_w] = tile_resized

        # Horizontal separator
        cv2.rectangle(sheet, (0, y_pos), (tile_w - 1, y_pos + tile_h - 1), (60, 60, 60), 1)

    return sheet


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  KAGGLE BEEF FALLBACK PERCEPTION AUDIT")
    print("  Strategy: IF RT-DETR detects no cow -> SAM 2.1 with ONE center point (112, 112)")
    print("  Target: EXACT 3 Known RT-DETR-L Failures from Fresh-40 Audit")
    print("    1. beef_4_4_clip_0 — Drinking")
    print("    2. beef_131_5_clip_0 — Feeding")
    print("    3. beef_00000000580000000_27_clip_3 — Lying")
    print("  Environment: Modal Cloud (profile tigerwood693, GPU Tier T4)")
    print("  Status: PENDING HUMAN REVIEW")
    print("=" * 80)

    train_csv = "datasets/behavior/cvb_beef/train.csv"
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")

    df_train = pd.read_csv(train_csv)
    beef_df = df_train[df_train["dataset"] == "beef_cattle_behavior"].copy()

    # Exact target order specified by user
    target_sample_ids = [
        "beef_4_4_clip_0",
        "beef_131_5_clip_0",
        "beef_00000000580000000_27_clip_3",
    ]

    target_records = []
    for sid in target_sample_ids:
        match = beef_df[beef_df["sample_id"] == sid]
        assert len(match) == 1, f"Expected 1 match for {sid}, got {len(match)}"
        row = match.iloc[0].to_dict()
        row["frame_index"] = row["n_frames"] // 2  # exactly 125
        row["canonical_behavior"] = row["behavior_canonical"]
        target_records.append(row)

    print(f"\n[1/3] Verified {len(target_records)} target failure samples from canonical train:")
    for idx, r in enumerate(target_records):
        print(f"  {idx+1}. {r['sample_id']:<35} | {r['canonical_behavior']:<10} | Session: {r['session_id']}")

    # Dispatch to Modal
    print("\n[2/3] Dispatching fallback evaluation to Modal (profile tigerwood693, GPU Tier T4)...")
    modal_output = evaluate_beef_fallback_modal.remote(target_records)

    # Assemble deliverables
    print("\n[3/3] Assembling results, saving CSV and master contact sheet...")
    out_csv_dir = Path("artifacts/perception_audit")
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / "beef_rtdetr_failure_fallback.csv"

    out_assets_dir = Path("docs/audits/assets/beef_rtdetr_failure_fallback")
    out_assets_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    comp_bytes_list = []

    for item in modal_output:
        sample_id = item["sample_id"]
        beh = item["canonical_behavior"]
        sample_idx = item["sample_idx"]
        rec = item["record"]
        comp_bytes = item["comp_bytes"]

        eval_rows.append(rec)
        comp_bytes_list.append(comp_bytes)

        # Save individual composite
        comp_filename = f"{sample_idx:02d}_{beh}_{sample_id}_fallback.jpg"
        with open(out_assets_dir / comp_filename, "wb") as f:
            f.write(comp_bytes)

    # Save CSV
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(out_csv_path, index=False)
    print(f"  ✓ Saved fallback CSV ({len(df_eval)} rows) to: {out_csv_path}")

    # Build Master Contact Sheet (1 col x 3 rows)
    master_sheet = build_fallback_master_contact_sheet(
        sample_composites=comp_bytes_list,
        tile_w=896,
        tile_h=256,
        header_h=80,
    )
    master_sheet_path = out_assets_dir / "contact_sheet.jpg"
    cv2.imwrite(str(master_sheet_path), master_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  ✓ Saved Master Contact Sheet ({master_sheet.shape[1]}x{master_sheet.shape[0]} px) to: {master_sheet_path}")

    # Detailed Console Report
    print("\n" + "=" * 80)
    print("  KAGGLE BEEF FALLBACK PERCEPTION AUDIT RESULTS")
    print("=" * 80)

    for idx, r in df_eval.iterrows():
        print(f"\nSample #{idx+1}: {r['sample_id']} ({r['canonical_behavior']})")
        print(f"  Session ID: {r['session_id']}")
        print(f"  Midpoint Frame: {r['frame_index']}")
        print(f"  Mask Returned: {r['mask_returned']} ({r['mask_status']})")
        print(f"  Mask Area Ratio: {r['mask_area_ratio']:.4f}")
        print(f"  Connected Components: {r['number_of_connected_components']}")
        print(f"  Inference Latency: {r['inference_time_ms']:.1f} ms")

    all_returned = df_eval["mask_returned"].all()
    print("\n" + "-" * 80)
    print(f"  Fallback Success Rate: {df_eval['mask_returned'].sum()} / {len(df_eval)} ({df_eval['mask_returned'].mean()*100:.1f}%)")
    print(f"  Mean Area Ratio: {df_eval['mask_area_ratio'].mean():.4f}")
    print(f"  Mean Connected Components: {df_eval['number_of_connected_components'].mean():.1f}")
    print("-" * 80)

    print("\n================================================================================")
    print("  FINAL STATUS: PENDING HUMAN REVIEW")
    print("  Note: No winner has been automatically selected.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
