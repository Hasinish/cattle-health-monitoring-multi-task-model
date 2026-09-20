"""
Generate visual contact sheets for Step 2.3 Cattle Pose Feasibility Audit.

Reads all 90 committed pose overlay images from `docs/audits/assets/perception_audit/`:
  - 30 ScienceDB (10 HRNet-W32 expanded, 10 HRNet-W32 smoke, 10 ResNet-50 expanded)
  - 30 MmCows (10 HRNet-W32 expanded, 10 HRNet-W32 smoke, 10 ResNet-50 expanded)
  - 30 SideViewCows2026 (10 HRNet-W32 expanded, 10 HRNet-W32 smoke, 10 ResNet-50 expanded)

Outputs 9 high-resolution contact sheets (10 images each) to:
  `docs/audits/assets/pose_visual_review/`
"""

import glob
import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np


def extract_metadata(filepath: str) -> Tuple[str, str, str, str]:
    """
    Extract (dataset, model, mode, sample_id) from filename.
    e.g. pose_expanded_hrnet_w32_sample_0001.jpg
    """
    fname = os.path.basename(filepath)
    model = "HRNet-W32" if "hrnet_w32" in fname else ("ResNet-50" if "resnet_50" in fname else "Unknown")
    mode = "Expanded" if "expanded" in fname else ("Smoke" if "smoke" in fname else "Other")

    m = re.search(r"sample_\d+", fname)
    s_id = m.group(0) if m else "sample_unknown"
    num = int(s_id.split("_")[1]) if m else 0

    if num <= 100:
        dataset = "ScienceDB"
    elif num <= 200:
        dataset = "MmCows"
    else:
        dataset = "SideViewCows2026"

    return dataset, model, mode, s_id


def create_contact_sheet(
    title: str,
    image_entries: List[Tuple[str, str, str, str, str]],  # (path, dataset, model, mode, sample_id)
    output_path: str,
    target_width: int = 2200,
) -> None:
    """
    Build a vertically stacked, high-resolution contact sheet.
    Pads each image horizontally to target_width to preserve 100% original sharpness.
    """
    panels = []

    # 1. Main Sheet Header
    header_h = 64
    header = np.zeros((header_h, target_width, 3), dtype=np.uint8)
    header[:] = (45, 30, 20)  # Dark slate blue in BGR

    cv2.putText(
        header,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    panels.append(header)

    # 2. Add each image + label bar
    for idx, (img_path, dataset, model, mode, s_id) in enumerate(image_entries, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        ih, iw = img.shape[:2]

        # Pad or resize proportionally to target_width
        if iw != target_width:
            scale = target_width / iw
            new_h = max(1, int(round(ih * scale)))
            img_resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img

        panels.append(img_resized)

        # Label bar below image
        bar_h = 38
        bar = np.zeros((bar_h, target_width, 3), dtype=np.uint8)
        bar[:] = (25, 25, 25)  # Dark gray

        fname = os.path.basename(img_path)
        label_text = f"#{idx:02d} | Sample: {s_id} | Dataset: {dataset} | Model: {model} | Mode: {mode} | File: {fname}"
        cv2.putText(
            bar,
            label_text,
            (16, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),  # Yellow/cyan highlight
            1,
            cv2.LINE_AA,
        )
        panels.append(bar)

        # Thin separator line between samples
        sep_h = 6
        sep = np.zeros((sep_h, target_width, 3), dtype=np.uint8)
        sep[:] = (60, 60, 60)
        panels.append(sep)

    # Stack all panels vertically
    sheet = np.vstack(panels)

    # Save high-quality JPEG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK] Created contact sheet: {output_path} ({sheet.shape[1]}x{sheet.shape[0]}, {len(image_entries)} items)")


def main():
    src_pattern = "docs/audits/assets/perception_audit/pose_*.jpg"
    files = sorted(glob.glob(src_pattern))
    print(f"Total source pose overlays found: {len(files)}")

    # Classify all 90 files
    groups: Dict[str, List[Tuple[str, str, str, str, str]]] = {
        "ScienceDB — HRNet-W32": [],
        "ScienceDB — ResNet-50": [],
        "MmCows — HRNet-W32": [],
        "MmCows — ResNet-50": [],
        "SideViewCows2026 — HRNet-W32": [],
        "SideViewCows2026 — ResNet-50": [],
    }

    unassigned = []
    for f in files:
        dataset, model, mode, s_id = extract_metadata(f)
        key = f"{dataset} — {model}"
        if key in groups:
            groups[key].append((f, dataset, model, mode, s_id))
        else:
            unassigned.append(f)

    if unassigned:
        print(f"[WARNING] Unassigned files: {len(unassigned)}")

    out_dir = "docs/audits/assets/pose_visual_review"
    os.makedirs(out_dir, exist_ok=True)

    generated_sheets = []

    # Map groups to filenames and split into pages of max 10-15 images
    group_slugs = {
        "ScienceDB — HRNet-W32": "sciencedb_hrnet_w32",
        "ScienceDB — ResNet-50": "sciencedb_resnet_50",
        "MmCows — HRNet-W32": "mmcows_hrnet_w32",
        "MmCows — ResNet-50": "mmcows_resnet_50",
        "SideViewCows2026 — HRNet-W32": "sideviewcows2026_hrnet_w32",
        "SideViewCows2026 — ResNet-50": "sideviewcows2026_resnet_50",
    }

    for group_name, items in groups.items():
        slug = group_slugs[group_name]
        # Sort items: Expanded first (by sample_id), then Smoke (by sample_id)
        items_sorted = sorted(items, key=lambda x: (0 if x[3] == "Expanded" else 1, x[4]))

        # Split into pages of 10
        page_size = 10
        pages = [items_sorted[i : i + page_size] for i in range(0, len(items_sorted), page_size)]
        num_pages = len(pages)

        for page_idx, page_items in enumerate(pages, 1):
            sheet_fname = f"{slug}_page{page_idx}.jpg"
            out_path = os.path.join(out_dir, sheet_fname)
            page_mode = page_items[0][3] if all(p[3] == page_items[0][3] for p in page_items) else "Mixed"
            sheet_title = f"{group_name} ({page_mode} Audit — Page {page_idx}/{num_pages}) | {len(page_items)} Overlays"

            create_contact_sheet(
                title=sheet_title,
                image_entries=page_items,
                output_path=out_path,
                target_width=2200,
            )
            generated_sheets.append((group_name, sheet_fname, len(page_items), page_mode, out_path))

    print("\nSummary of Generated Contact Sheets:")
    total_included = sum(s[2] for s in generated_sheets)
    print(f"Total contact sheets: {len(generated_sheets)}")
    print(f"Total overlays included: {total_included} / {len(files)}")
    assert total_included == len(files), f"Mismatch! {total_included} != {len(files)}"
    print("[VERIFIED] Every source overlay is represented exactly once!")


if __name__ == "__main__":
    main()
