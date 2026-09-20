"""
Generate high-resolution visual contact sheets and manifest for Step 2.4 Cattle Viewpoint Audit.

Creates:
  1. artifacts/perception_audit/viewpoint_manual_review_manifest.csv
  2. docs/audits/assets/viewpoint_visual_review/*.jpg (6 contact sheets, 10 samples each)
  3. docs/audits/phase3_viewpoint_visual_review_index.md

Taxonomy:
  - rear
  - rear-oblique
  - side
  - front-oblique
  - front
  - unknown / ambiguous
"""

import csv
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Deterministic Selection Configuration (60 Samples Total, 20 per Dataset)
# ---------------------------------------------------------------------------

SCIENCEDB_PICKS = [
    # BCS 3.25 (4)
    'sample_0001', 'sample_0003', 'sample_0006', 'sample_0010',
    # BCS 3.50 (4)
    'sample_0021', 'sample_0022', 'sample_0026', 'sample_0030',
    # BCS 3.75 (4)
    'sample_0041', 'sample_0043', 'sample_0051', 'sample_0053',
    # BCS 4.00 (4)
    'sample_0061', 'sample_0063', 'sample_0065', 'sample_0070',
    # BCS 4.25 (4)
    'sample_0081', 'sample_0082', 'sample_0084', 'sample_0090'
]

MMCOWS_PICKS = [
    # Lying (3)
    'sample_0104', 'sample_0105', 'sample_0107',
    # Standing (3)
    'sample_0117', 'sample_0120', 'sample_0124',
    # Feeding head down (3)
    'sample_0133', 'sample_0137', 'sample_0140',
    # Feeding head up (3)
    'sample_0149', 'sample_0152', 'sample_0156',
    # Walking (3)
    'sample_0165', 'sample_0169', 'sample_0172',
    # Drinking (3)
    'sample_0179', 'sample_0183', 'sample_0185',
    # Licking (2)
    'sample_0190', 'sample_0195'
]

SIDEVIEW_PICKS = [
    # Parlor (8)
    'sample_0201', 'sample_0204', 'sample_0206', 'sample_0210',
    'sample_0215', 'sample_0220', 'sample_0228', 'sample_0235',
    # Barn (8)
    'sample_0241', 'sample_0245', 'sample_0250', 'sample_0255',
    'sample_0260', 'sample_0265', 'sample_0270', 'sample_0277',
    # Snapshots (4)
    'sample_0281', 'sample_0285', 'sample_0290', 'sample_0298'
]


def build_manifest(expanded_manifest_path: str, output_manifest_path: str) -> pd.DataFrame:
    """Extract selected samples and write viewpoint_manual_review_manifest.csv."""
    exp_df = pd.read_csv(expanded_manifest_path)
    all_picks = SCIENCEDB_PICKS + MMCOWS_PICKS + SIDEVIEW_PICKS
    sub_df = exp_df[exp_df['sample_id'].isin(all_picks)].copy()

    records = []
    for s_id in all_picks:
        row = sub_df[sub_df['sample_id'] == s_id].iloc[0]
        meta = json.loads(row['metadata'])

        if row['dataset'] == 'ScienceDB':
            note = f"Farm: {meta.get('farm_source')}; BCS: {meta.get('bcs_label')}; Passage: {meta.get('original_passage_id')}"
        elif row['dataset'] == 'MmCows':
            note = f"Behavior: {meta.get('behavior')}; Cow: {meta.get('cow_id')}; Cam: {meta.get('camera_id')}"
        else:
            note = f"Subset: {meta.get('subset')}; Cow: {meta.get('individual_id')}; Frame: {meta.get('frame_no')}"

        records.append({
            'sample_id': row['sample_id'],
            'dataset': row['dataset'],
            'source_image_path': row['image_path'],
            'category_or_subset': row['category_or_subset'],
            'provenance_group': row['burst_or_session'],
            'proposed_viewpoint': 'UNREVIEWED',
            'review_status': 'pending_human_review',
            'manual_notes': note
        })

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    out_df.to_csv(output_manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[OK] Wrote {len(out_df)} rows to {output_manifest_path}")
    return out_df


# ---------------------------------------------------------------------------
# 2. High-Resolution Contact Sheet Builder
# ---------------------------------------------------------------------------

def create_grid_contact_sheet(
    title: str,
    subtitle: str,
    entries: List[Dict[str, str]],
    output_path: str,
    cols: int = 2,
    cell_w: int = 1080,
    cell_h: int = 680,
) -> None:
    """
    Build a 2-column high-resolution contact sheet.
    Each cell contains:
      - RGB image fitted cleanly inside cell_w x (cell_h - label_h)
      - High-contrast label bar with sample_id and metadata
    """
    num_entries = len(entries)
    rows = (num_entries + cols - 1) // cols

    sheet_w = cols * cell_w + 40  # 20px padding left/right
    header_h = 100
    sheet_h = header_h + rows * cell_h + 30

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (20, 20, 24)  # Deep dark slate background

    # Header Bar
    header_bar = np.zeros((header_h, sheet_w, 3), dtype=np.uint8)
    header_bar[:] = (40, 28, 18)  # Rich dark bronze/slate in BGR
    cv2.putText(
        header_bar,
        title,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header_bar,
        subtitle,
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    sheet[0:header_h, :] = header_bar

    # Render each cell
    label_h = 50
    img_max_w = cell_w - 24
    img_max_h = cell_h - label_h - 16

    for idx, entry in enumerate(entries):
        r = idx // cols
        c = idx % cols

        x0 = 20 + c * cell_w
        y0 = header_h + 15 + r * cell_h

        img_path = entry['source_image_path']
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            img_box = np.zeros((img_max_h, img_max_w, 3), dtype=np.uint8)
            cv2.putText(img_box, f"IMAGE MISSING: {img_path}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            ih, iw = img.shape[:2]
            scale = min(img_max_w / iw, img_max_h / ih)
            nw = max(1, int(round(iw * scale)))
            nh = max(1, int(round(ih * scale)))

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            img_scaled = cv2.resize(img, (nw, nh), interpolation=interp)

            # Center inside image box
            img_box = np.zeros((img_max_h, img_max_w, 3), dtype=np.uint8)
            img_box[:] = (12, 12, 14)
            dx = (img_max_w - nw) // 2
            dy = (img_max_h - nh) // 2
            img_box[dy:dy+nh, dx:dx+nw] = img_scaled

        # Draw cell border
        cv2.rectangle(img_box, (0, 0), (img_max_w - 1, img_max_h - 1), (60, 60, 70), 1)

        # Place image box in sheet
        sheet[y0:y0+img_max_h, x0+12:x0+12+img_max_w] = img_box

        # Label Bar
        bar_y = y0 + img_max_h + 4
        label_bar = np.zeros((label_h, img_max_w, 3), dtype=np.uint8)
        label_bar[:] = (32, 32, 38)
        cv2.rectangle(label_bar, (0, 0), (img_max_w - 1, label_h - 1), (80, 80, 95), 1)

        # Primary label: sample_id in bright cyan, dataset/category in white
        s_id = entry['sample_id']
        ds = entry['dataset']
        cat = entry['category_or_subset']
        prov = entry['provenance_group']

        primary_txt = f"{s_id}  |  {ds}  |  {cat}"
        cv2.putText(
            label_bar,
            primary_txt,
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 0),  # Cyan
            2,
            cv2.LINE_AA,
        )

        secondary_txt = f"Provenance: {prov}  |  {entry['manual_notes']}"
        cv2.putText(
            label_bar,
            secondary_txt,
            (14, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        sheet[bar_y:bar_y+label_h, x0+12:x0+12+img_max_w] = label_bar

    # Save output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"[OK] Saved contact sheet: {output_path} ({sheet_w}x{sheet_h}, {num_entries} samples)")


# ---------------------------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------------------------

def main():
    expanded_manifest = "artifacts/perception_audit/sample_manifest_expanded.csv"
    viewpoint_manifest = "artifacts/perception_audit/viewpoint_manual_review_manifest.csv"
    contact_dir = "docs/audits/assets/viewpoint_visual_review"
    index_path = "docs/audits/phase3_viewpoint_visual_review_index.md"

    # Step A: Build manifest
    df = build_manifest(expanded_manifest, viewpoint_manifest)

    # Step B: Build contact sheets
    sheets_config = [
        {
            'dataset': 'ScienceDB',
            'page': 1,
            'picks': SCIENCEDB_PICKS[:10],
            'filename': 'sciencedb_viewpoint_page1.jpg',
            'title': 'ScienceDB — Cattle Viewpoint Manual Review (Page 1 of 2: Samples 1–10)',
            'subtitle': 'BCS Classes 3.25 & 3.50  |  Farms: GS_Gansu, YM_Farm2  |  Categories: rear, rear-oblique, side, front-oblique, front, unknown / ambiguous'
        },
        {
            'dataset': 'ScienceDB',
            'page': 2,
            'picks': SCIENCEDB_PICKS[10:],
            'filename': 'sciencedb_viewpoint_page2.jpg',
            'title': 'ScienceDB — Cattle Viewpoint Manual Review (Page 2 of 2: Samples 11–20)',
            'subtitle': 'BCS Classes 3.75, 4.00, 4.25  |  Farms: GS_Gansu, YM_Farm2, STEREO_Farm3  |  Categories: rear, rear-oblique, side, front-oblique, front, unknown / ambiguous'
        },
        {
            'dataset': 'MmCows',
            'page': 1,
            'picks': MMCOWS_PICKS[:10],
            'filename': 'mmcows_viewpoint_page1.jpg',
            'title': 'MmCows — Cattle Viewpoint Manual Review (Page 1 of 2: Samples 1–10)',
            'subtitle': 'Behaviors: Lying, Standing, Feeding (head down/up)  |  Cows: 1, 3, 4, 5, 6, 7, 12, 13, 15  |  Cams: 1, 2, 3, 4'
        },
        {
            'dataset': 'MmCows',
            'page': 2,
            'picks': MMCOWS_PICKS[10:],
            'filename': 'mmcows_viewpoint_page2.jpg',
            'title': 'MmCows — Cattle Viewpoint Manual Review (Page 2 of 2: Samples 11–20)',
            'subtitle': 'Behaviors: Feeding, Walking, Drinking, Licking  |  Cows: 2, 3, 4, 7, 8, 11, 15, 16  |  Cams: 1, 2, 3, 4'
        },
        {
            'dataset': 'SideViewCows2026',
            'page': 1,
            'picks': SIDEVIEW_PICKS[:10],
            'filename': 'sideviewcows2026_viewpoint_page1.jpg',
            'title': 'SideViewCows2026 — Cattle Viewpoint Manual Review (Page 1 of 2: Samples 1–10)',
            'subtitle': 'Subsets: Parlor (samples 1–8), Barn (samples 9–10)  |  Side-view chute walk & barn pen angles'
        },
        {
            'dataset': 'SideViewCows2026',
            'page': 2,
            'picks': SIDEVIEW_PICKS[10:],
            'filename': 'sideviewcows2026_viewpoint_page2.jpg',
            'title': 'SideViewCows2026 — Cattle Viewpoint Manual Review (Page 2 of 2: Samples 11–20)',
            'subtitle': 'Subsets: Barn (samples 11–16), Snapshots (samples 17–20)  |  Barn alleys, group pens, snapshot angles'
        },
    ]

    generated_sheets = []
    for cfg in sheets_config:
        out_file = os.path.join(contact_dir, cfg['filename'])
        sub_entries = []
        for s_id in cfg['picks']:
            row = df[df['sample_id'] == s_id].iloc[0]
            sub_entries.append(row.to_dict())

        create_grid_contact_sheet(
            title=cfg['title'],
            subtitle=cfg['subtitle'],
            entries=sub_entries,
            output_path=out_file,
            cols=2,
            cell_w=1080,
            cell_h=680
        )
        generated_sheets.append((cfg, out_file))

    # Step C: Build Visual Review Index Markdown
    md_lines = [
        "# Phase 3 Viewpoint Visual Review Index (Step 2.4)",
        "",
        "**Date**: 2026-09-20  ",
        "**Phase**: Phase 3 (Pretrained Perception Feasibility Audit — Step 2.4 Viewpoint Taxonomy Definition)  ",
        "**Manifest**: [artifacts/perception_audit/viewpoint_manual_review_manifest.csv](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_manual_review_manifest.csv)  ",
        "**Status**: `pending_human_review` (Manual review pack built; no model inference or training performed)  ",
        "",
        "---",
        "",
        "## Viewpoint Taxonomy & Evaluation Guidelines",
        "",
        "The objective of Step 2.4 is to determine whether cattle body orientation categories are visually definable and consistently usable across our primary datasets without relying on camera ID shortcuts.",
        "",
        "### Candidate Coarse Taxonomy:",
        "1. `rear` — Direct rear view (rump, tailhead, hindquarters facing camera).",
        "2. `rear-oblique` — Three-quarter rear perspective (flank and rear visible).",
        "3. `side` — Broadside profile view (full lateral body profile).",
        "4. `front-oblique` — Three-quarter frontal perspective (head, shoulder, and flank visible).",
        "5. `front` — Direct frontal view (head, chest, snout facing camera).",
        "6. `unknown / ambiguous` — Cases where orientation cannot be definitively determined (heavy occlusion, extreme top-down CCTV angle, tight or partial crop, multiple cows causing ambiguity).",
        "",
        "> [!IMPORTANT]",
        "> **Camera ID is NOT viewpoint**: In multi-camera surveillance (MmCows) or fixed chutes, camera position must not be treated as a viewpoint proxy. The true viewpoint is defined strictly by the cow's physical body orientation relative to the camera optical axis.",
        "",
        "---",
        "",
        "## Contact Sheets for Visual Review",
        ""
    ]

    for cfg, out_file in generated_sheets:
        rel_path = os.path.relpath(out_file, os.path.dirname(index_path)).replace("\\", "/")
        md_lines.extend([
            f"### {cfg['title']}",
            f"*{cfg['subtitle']}*",
            "",
            f"![{cfg['title']}]({rel_path})",
            "",
            f"- **Samples Included ({len(cfg['picks'])} images)**: `{', '.join(cfg['picks'])}`",
            f"- **Contact Sheet File**: `{cfg['filename']}`",
            "",
            "---",
            ""
        ])

    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Created viewpoint visual review index: {index_path}")


if __name__ == '__main__':
    main()
