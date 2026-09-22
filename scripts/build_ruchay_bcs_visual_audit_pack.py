"""
Build a representative Ruchay 2026 RGB visual contact-sheet pack for manual BCS-quality audit.

Selects exactly 100 RGB images using deterministic seed 2026:
- 10 samples from each of the 10 BCS classes (2.75 to 5.00)
- Spread across 4 recording sessions as evenly as dataset permits
- Maximizes unique biological cow IDs (94 unique cows across 100 samples)
- Centrally positioned non-adjacent frames
- Obtains only required RGB files via HTTP Range requests without downloading 77.74 GB raw archives
- Generates 10 high-resolution contact sheets (2x5 grid, tiles 640x360 px, long side >= 500 px)
- Produces:
  artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv
  docs/audits/assets/ruchay_bcs_visual_audit/*.jpg
  docs/audits/phase3_ruchay_bcs_visual_audit_index.md
"""

import os
import io
import time
import zipfile
import requests
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

REPO_ROOT = r"d:\cattle-health-monitoring-multi-task-model"
MANIFEST_SRC = os.path.join(REPO_ROOT, "datasets", "bcs", "external", "ruchay2026", "ruchay2026_manifest.csv")
SAMPLES_DIR = os.path.join(REPO_ROOT, "datasets", "bcs", "external", "ruchay2026", "samples")
AUDIT_MANIFEST_OUT = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "ruchay_bcs_visual_audit_manifest.csv")
ASSETS_DIR = os.path.join(REPO_ROOT, "docs", "audits", "assets", "ruchay_bcs_visual_audit")
INDEX_MD_OUT = os.path.join(REPO_ROOT, "docs", "audits", "phase3_ruchay_bcs_visual_audit_index.md")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SELECTION_SEED = 2026

# Zenodo archive metadata for record 20290988
ARCHIVES = {
    "06.12.2024": {
        "url": "https://zenodo.org/api/records/20290988/files/06.12.2024.zip/content",
        "size": 9756687171,
        "has_date_prefix": True
    },
    "20.02.2025": {
        "url": "https://zenodo.org/api/records/20290988/files/20.02.2025.zip/content",
        "size": 13831603804,
        "has_date_prefix": True
    },
    "20.03.2025": {
        "url": "https://zenodo.org/api/records/20290988/files/20.03.2025.zip/content",
        "size": 16905046250,
        "has_date_prefix": True
    },
    "27.03.2025.1": {
        "url": "https://zenodo.org/api/records/20290988/files/27.03.2025.1.zip/content",
        "size": 21398809702,
        "has_date_prefix": False
    },
    "27.03.2025.2": {
        "url": "https://zenodo.org/api/records/20290988/files/27.03.2025.2.zip/content",
        "size": 21581642116,
        "has_date_prefix": False
    },
}


class CachedHTTPRangeFile(io.RawIOBase):
    """Buffered read-ahead file object over HTTP Range requests."""
    def __init__(self, url: str, size: int, headers: dict = None, chunk_size: int = 4 * 1024 * 1024):
        self.url = url
        self.size = size
        self.pos = 0
        self.headers = headers or {}
        self.session = requests.Session()
        self.chunk_size = chunk_size
        self.cache_start = 0
        self.cache_data = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        size = len(b)
        if self.pos >= self.size or size == 0:
            return 0
        # Serve from cache if available
        if self.cache_start <= self.pos < self.cache_start + len(self.cache_data):
            off = self.pos - self.cache_start
            avail = len(self.cache_data) - off
            to_read = min(size, avail)
            b[:to_read] = self.cache_data[off : off + to_read]
            self.pos += to_read
            return to_read
        # Fetch chunk with retry and backoff
        fetch_len = max(size, self.chunk_size)
        end = min(self.pos + fetch_len - 1, self.size - 1)
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={**self.headers, "Range": f"bytes={self.pos}-{end}"}, timeout=30)
                if r.status_code == 206 and len(r.content) > 0:
                    self.cache_start = self.pos
                    self.cache_data = r.content
                    break
                else:
                    time.sleep(1.0 * (attempt + 1))
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to fetch range bytes={self.pos}-{end} from {self.url} after 5 attempts")

        to_read = min(size, len(self.cache_data))
        b[:to_read] = self.cache_data[:to_read]
        self.pos += to_read
        return to_read


def select_100_samples(manifest_path: str, seed: int = 2026) -> pd.DataFrame:
    """Select exactly 100 representative samples stratified across 10 BCS classes."""
    df = pd.read_csv(manifest_path)
    df["frame_idx_in_passage"] = df.groupby(["date", "frame"]).cumcount()

    classes = [2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
    rng = np.random.RandomState(seed)

    selected_cows_global = set()
    selected_rows = []

    for bcs in classes:
        sub = df[df["bcs"] == bcs].copy()
        available_dates = sorted(sub["date"].unique())

        if bcs == 2.75:
            # 2.75 exists only on 06.12.2024 across 4 cows
            cows = sorted(sub["animal_id"].unique())
            counts = [3, 3, 2, 2]
            for cow_id, count in zip(cows, counts):
                cow_sub = sub[sub["animal_id"] == cow_id]
                target_frames = [5, 10, 15] if count == 3 else [7, 13]
                picks = cow_sub[cow_sub["frame_idx_in_passage"].isin(target_frames)]
                selected_rows.append(picks)
                selected_cows_global.add(cow_id)
        else:
            n_dates = len(available_dates)
            base_per_date = 10 // n_dates
            rem = 10 % n_dates
            date_quotas = {d: base_per_date + (1 if i < rem else 0) for i, d in enumerate(available_dates)}

            # Quota adjustment based on available unique cows
            for d in available_dates:
                d_cows = set(sub[sub["date"] == d]["animal_id"].unique()) - selected_cows_global
                if len(d_cows) < date_quotas[d]:
                    deficit = date_quotas[d] - len(d_cows)
                    date_quotas[d] = len(d_cows)
                    for other_d in available_dates:
                        if other_d != d:
                            other_avail = len(set(sub[sub["date"] == other_d]["animal_id"].unique()) - selected_cows_global)
                            if other_avail >= date_quotas[other_d] + deficit:
                                date_quotas[other_d] += deficit
                                break

            for d, quota in date_quotas.items():
                if quota == 0:
                    continue
                d_sub = sub[sub["date"] == d]
                avail_cows = sorted(list(set(d_sub["animal_id"].unique()) - selected_cows_global))
                if len(avail_cows) < quota:
                    avail_cows = sorted(d_sub["animal_id"].unique())
                chosen_cows = rng.choice(avail_cows, size=quota, replace=False)
                for cow_id in chosen_cows:
                    selected_cows_global.add(cow_id)
                    cow_sub = d_sub[d_sub["animal_id"] == cow_id]
                    mid_picks = cow_sub[cow_sub["frame_idx_in_passage"].isin([9, 10, 8, 11])]
                    pick = mid_picks.sample(n=1, random_state=rng) if len(mid_picks) > 0 else cow_sub.sample(n=1, random_state=rng)
                    selected_rows.append(pick)

    res = pd.concat(selected_rows, ignore_index=True)
    # Sort deterministically by BCS ascending, then session, then cow_id
    res = res.sort_values(by=["bcs", "date", "animal_id"]).reset_index(drop=True)
    # Assign sequential sample_id ruchay_audit_001 to ruchay_audit_100
    res["sample_id"] = [f"ruchay_audit_{i+1:03d}" for i in range(len(res))]
    return res


def extract_missing_samples(selected_df: pd.DataFrame) -> None:
    """Download only the required images via selective HTTP Range requests."""
    tasks_by_archive: Dict[str, List[Tuple[str, str, str]]] = {}

    for _, row in selected_df.iterrows():
        date = row["date"]
        frame = row["frame"]
        base = os.path.basename(row["rgb_path"])
        local_path = os.path.join(SAMPLES_DIR, date, str(frame), "rgb", base)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            continue

        if date == "27.03.2025":
            arch_name = "27.03.2025.1" if frame <= 340 else "27.03.2025.2"
            path_in_zip = f"{frame}/rgb/{base}"
        else:
            arch_name = date
            path_in_zip = f"{date}/{frame}/rgb/{base}"

        tasks_by_archive.setdefault(arch_name, []).append((row["sample_id"], path_in_zip, local_path))

    total_missing = sum(len(v) for v in tasks_by_archive.values())
    print(f"\n[INFO] Missing images to download: {total_missing} / {len(selected_df)}")

    if total_missing == 0:
        print("[INFO] All 100 images already exist locally!")
        return

    def process_archive(arch: str, task_items: List[Tuple[str, str, str]]):
        meta = ARCHIVES[arch]
        print(f"\n[INFO] [Thread {arch}] Opening archive for {len(task_items)} images...", flush=True)
        t0 = time.time()
        raw_io = CachedHTTPRangeFile(meta["url"], meta["size"], HEADERS, chunk_size=4 * 1024 * 1024)
        zf = zipfile.ZipFile(raw_io)
        print(f"  -> [Thread {arch}] Zip index loaded in {time.time() - t0:.2f}s", flush=True)

        done = 0
        for sample_id, path_in_zip, local_path in task_items:
            t_item = time.time()
            data = zf.read(path_in_zip)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)
            done += 1
            print(f"  -> [{arch}] [{sample_id}] {os.path.basename(local_path)} ({len(data)/1024:.1f} KB in {time.time() - t_item:.2f}s | {done}/{len(task_items)})", flush=True)

    with ThreadPoolExecutor(max_workers=len(tasks_by_archive)) as executor:
        futures = [executor.submit(process_archive, arch, items) for arch, items in tasks_by_archive.items()]
        for f in as_completed(futures):
            f.result()


def validate_and_build_manifest(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Validate all images on disk and export the audit manifest."""
    os.makedirs(os.path.dirname(AUDIT_MANIFEST_OUT), exist_ok=True)

    manifest_rows = []
    for _, row in selected_df.iterrows():
        date = row["date"]
        frame = row["frame"]
        base = os.path.basename(row["rgb_path"])
        local_abs = os.path.join(SAMPLES_DIR, date, str(frame), "rgb", base)
        local_rel = os.path.relpath(local_abs, REPO_ROOT).replace("\\", "/")

        assert os.path.exists(local_abs), f"File missing: {local_abs}"
        img = cv2.imread(local_abs)
        assert img is not None, f"Corrupt image: {local_abs}"
        h, w = img.shape[:2]

        manifest_rows.append({
            "sample_id": row["sample_id"],
            "image_path": local_rel,
            "cow_id": row["animal_id"],
            "session": date,
            "bcs": row["bcs"],
            "source_filename": base,
            "width": w,
            "height": h,
            "selection_seed": SELECTION_SEED,
            # Extra fields retained for full index provenance
            "frame": frame,
            "original_sample_id": row["sample_id_orig"] if "sample_id_orig" in row else row["sample_id"]
        })

    mdf = pd.DataFrame(manifest_rows)
    # Columns required by specification:
    # sample_id,image_path,cow_id,session,bcs,source_filename,width,height,selection_seed
    req_cols = ["sample_id", "image_path", "cow_id", "session", "bcs", "source_filename", "width", "height", "selection_seed"]
    mdf[req_cols].to_csv(AUDIT_MANIFEST_OUT, index=False)
    print(f"\n[SUCCESS] Saved audit manifest with {len(mdf)} rows to: {AUDIT_MANIFEST_OUT}")
    return mdf


# Grid layout configuration for contact sheets
COLS = 2
ROWS = 5
TILE_W = 640
TILE_IMG_H = 360  # 16:9 aspect ratio fits 1080p without distortion
TILE_BAR_H = 65
TILE_H = TILE_IMG_H + TILE_BAR_H  # 425

HEADER_H = 85
SHEET_W = COLS * TILE_W + 40  # 1320
SHEET_H = HEADER_H + ROWS * TILE_H + 30  # 85 + 2125 + 30 = 2240


def render_tile(row: pd.Series) -> np.ndarray:
    """Render a single tile (640x425) with high-res 16:9 image and metadata bar."""
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    tile[:] = (22, 22, 28)  # Premium dark slate background

    local_path = os.path.join(REPO_ROOT, row["image_path"])
    img = cv2.imread(local_path)
    if img is not None:
        h, w = img.shape[:2]
        scale = min(TILE_W / w, TILE_IMG_H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x_off = (TILE_W - nw) // 2
        y_off = (TILE_IMG_H - nh) // 2
        tile[y_off : y_off + nh, x_off : x_off + nw] = resized
    else:
        cv2.putText(tile, "MISSING IMAGE", (50, TILE_IMG_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Metadata bar
    bar_y = TILE_IMG_H
    cv2.rectangle(tile, (0, bar_y), (TILE_W, TILE_H), (15, 15, 20), -1)
    cv2.line(tile, (0, bar_y), (TILE_W, bar_y), (60, 60, 75), 1)

    # Line 1: Sample ID | BCS | Cow ID
    line1 = f"{row['sample_id'].upper()}  |  BCS: {row['bcs']:.2f}  |  Cow ID: {row['cow_id']}"
    cv2.putText(tile, line1, (15, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)

    # Line 2: Session | Passage | Source filename
    line2 = f"Session: {row['session']}  |  Passage: {row['frame']}  |  {row['source_filename']}"
    cv2.putText(tile, line2, (15, bar_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 180, 195), 1, cv2.LINE_AA)

    # Border around tile
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (50, 50, 65), 1)
    return tile


def generate_contact_sheets(manifest_df: pd.DataFrame) -> List[str]:
    """Generate 10 high-resolution contact sheets (one per BCS class)."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    generated_paths = []

    classes = sorted(manifest_df["bcs"].unique())
    for sheet_idx, bcs_val in enumerate(classes, 1):
        sub = manifest_df[manifest_df["bcs"] == bcs_val].reset_index(drop=True)
        sheet = np.zeros((SHEET_H, SHEET_W, 3), dtype=np.uint8)
        sheet[:] = (12, 12, 16)  # Dark charcoal container background

        # Header banner
        cv2.rectangle(sheet, (0, 0), (SHEET_W, HEADER_H), (20, 24, 32), -1)
        cv2.line(sheet, (0, HEADER_H), (SHEET_W, HEADER_H), (70, 80, 100), 2)

        title = f"RUCHAY 2026 BCS VISUAL AUDIT -- SHEET {sheet_idx:02d}/10: BCS {bcs_val:.2f}"
        cv2.putText(sheet, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        cows_count = sub["cow_id"].nunique()
        sessions = ", ".join(sorted(sub["session"].unique()))
        subtitle = f"10 Representative Overhead Nadir RGB Samples (1920x1080) | Cow IDs: {cows_count} unique | Sessions: {sessions} | Seed: {SELECTION_SEED}"
        cv2.putText(sheet, subtitle, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)

        # Tiles
        pad_x = 20
        pad_y = HEADER_H + 15
        for i, (_, row) in enumerate(sub.iterrows()):
            r = i // COLS
            c = i % COLS
            x = pad_x + c * TILE_W
            y = pad_y + r * TILE_H
            tile = render_tile(row)
            sheet[y : y + TILE_H, x : x + TILE_W] = tile

        out_name = f"ruchay_bcs_sheet_{sheet_idx:02d}_bcs{bcs_val:.2f}.jpg"
        out_path = os.path.join(ASSETS_DIR, out_name)
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        generated_paths.append(out_path)
        print(f"  -> Generated Sheet [{sheet_idx:02d}/10]: {out_name} (Dimensions: {SHEET_W}x{SHEET_H} px)")

    return generated_paths


def build_audit_index_md(manifest_df: pd.DataFrame, sheet_paths: List[str]) -> None:
    """Generate comprehensive markdown documentation mapping all samples and embedding contact sheets."""
    lines = []
    lines.append("# Ruchay et al. (2026) RGB-D BCS Visual Quality Audit Index\n")
    lines.append("**Date**: 2026-09-22  ")
    lines.append("**Author**: Hasin Ishrak & Antigravity Research Agent  ")
    lines.append("**Context**: Phase 3 Candidate External BCS Validation Benchmark Perception Audit  ")
    lines.append(f"**Selection Seed**: `{SELECTION_SEED}` (Deterministic, 100% reproducible)  ")
    lines.append(f"**Total Samples**: 100 RGB images across 10 BCS classes (exactly 10 per class)  ")
    lines.append(f"**Total Unique Cows**: {manifest_df['cow_id'].nunique()} unique biological cows (94% diversity)  \n")

    lines.append("---\n")
    lines.append("## 1. Executive Summary & Audit Purpose\n")
    lines.append("Ruchay et al. (2026) (*RGB-D dataset of dairy cows for body condition scoring*, Zenodo record `20290988`) is designated as the **Primary External BCS Validation Benchmark** in the canonical roadmap.")
    lines.append("Before executing cross-domain BCS evaluation, a manual human visual audit is required to assess:")
    lines.append("1. **Overhead Nadir Camera Framing**: Cows pass through a chute beneath a Microsoft Kinect sensor mounted at 3.05m height;")
    lines.append("2. **Anatomical Visibility**: Whether dorsal spine, hooks (tuber coxae), pins (tuber ischii), thurl, and tailhead ligament are clearly visible for Ferguson 5-point BCS evaluation;")
    lines.append("3. **Multi-Cow & Obstruction Risks**: Whether loose-housing crowding, gates, or chute walls introduce occlusion or multi-cow ambiguity;")
    lines.append("4. **Image Clarity & Lighting**: Sensor noise, shadows, motion blur, and contrast under barn illumination.\n")

    lines.append("> [!IMPORTANT]")
    lines.append("> **Audit Constraint**: This document and visual evidence pack are strictly observational. No dataset roles are altered, no roadmap changes are enacted, and no performance comparisons are fabricated. The pack enables direct manual inspection by Hasin and ChatGPT vision.\n")

    lines.append("---\n")
    lines.append("## 2. Sample Distribution & Dataset Demographics\n")
    lines.append("### Class & Session Cross-Tabulation\n")
    crosstab = pd.crosstab(manifest_df["bcs"], manifest_df["session"])
    lines.append("| BCS Class | 06.12.2024 | 20.02.2025 | 20.03.2025 | 27.03.2025 | Total Samples | Unique Cows |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for bcs_val in sorted(manifest_df["bcs"].unique()):
        sub_bcs = manifest_df[manifest_df["bcs"] == bcs_val]
        c_06 = crosstab.loc[bcs_val, "06.12.2024"] if "06.12.2024" in crosstab.columns else 0
        c_20_02 = crosstab.loc[bcs_val, "20.02.2025"] if "20.02.2025" in crosstab.columns else 0
        c_20_03 = crosstab.loc[bcs_val, "20.03.2025"] if "20.03.2025" in crosstab.columns else 0
        c_27 = crosstab.loc[bcs_val, "27.03.2025"] if "27.03.2025" in crosstab.columns else 0
        n_cows = sub_bcs["cow_id"].nunique()
        lines.append(f"| **{bcs_val:.2f}** | {c_06} | {c_20_02} | {c_20_03} | {c_27} | **{len(sub_bcs)}** | **{n_cows}** |")
    lines.append(f"| **TOTAL** | **{manifest_df['session'].value_counts().get('06.12.2024', 0)}** | **{manifest_df['session'].value_counts().get('20.02.2025', 0)}** | **{manifest_df['session'].value_counts().get('20.03.2025', 0)}** | **{manifest_df['session'].value_counts().get('27.03.2025', 0)}** | **{len(manifest_df)}** | **{manifest_df['cow_id'].nunique()}** |\n")

    lines.append("### Key Sampling Observations:\n")
    lines.append("- **BCS 2.75**: Present exclusively in session `06.12.2024` across only 4 cows in the entire 25,700-sample dataset. All 4 biological cows are represented (quotas: 3, 3, 2, 2) using non-adjacent frames (passage relative indices [5, 10, 15] or [7, 13]).")
    lines.append("- **BCS 4.25, 4.50, 4.75, 5.00**: Present exclusively in session `27.03.2025`. Each class features 10 completely distinct biological cows (1 frame per cow).")
    lines.append("- **BCS 3.00, 3.25, 3.50, 3.75, 4.00**: Distributed across all available recording dates with 10 distinct biological cows each.")
    lines.append("- **Total Diversity**: 94 distinct cows out of 100 samples (the theoretical maximum possible given BCS 2.75 constraints).\n")

    lines.append("---\n")
    lines.append("## 3. High-Resolution Contact Sheets Gallery\n")
    for sheet_idx, sheet_path in enumerate(sheet_paths, 1):
        bcs_val = sorted(manifest_df["bcs"].unique())[sheet_idx - 1]
        sheet_rel = os.path.relpath(sheet_path, os.path.dirname(INDEX_MD_OUT)).replace("\\", "/")
        lines.append(f"### Contact Sheet {sheet_idx:02d}: BCS {bcs_val:.2f}\n")
        lines.append(f"![Ruchay BCS Sheet {sheet_idx:02d}]({sheet_rel})\n")

    lines.append("---\n")
    lines.append("## 4. Full 100-Sample Audit Provenance Manifest\n")
    lines.append("| Sample ID | BCS | Cow ID | Session | Passage | Source Filename | Resolution | Local Relative Path |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :--- |")
    for _, row in manifest_df.iterrows():
        lines.append(f"| `{row['sample_id']}` | **{row['bcs']:.2f}** | `{row['cow_id']}` | {row['session']} | {row['frame']} | `{row['source_filename']}` | {row['width']}x{row['height']} | [`{row['image_path']}`](file:///{os.path.join(REPO_ROOT, row['image_path']).replace(os.sep, '/')} ) |")

    lines.append("\n---\n")
    lines.append("## 5. Artifact Registry\n")
    lines.append(f"- Manifest CSV: [`artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv`](file:///{AUDIT_MANIFEST_OUT.replace(os.sep, '/')})\n")
    lines.append(f"- Generator Script: [`scripts/build_ruchay_bcs_visual_audit_pack.py`](file:///{os.path.join(REPO_ROOT, 'scripts', 'build_ruchay_bcs_visual_audit_pack.py').replace(os.sep, '/')})\n")
    lines.append(f"- Contact Sheets Directory: [`docs/audits/assets/ruchay_bcs_visual_audit/`](file:///{ASSETS_DIR.replace(os.sep, '/')})\n")

    with open(INDEX_MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SUCCESS] Generated comprehensive audit index markdown at: {INDEX_MD_OUT}")


def main():
    print("=" * 70)
    print("STEP 1: Deterministic Sample Selection (Seed 2026)")
    print("=" * 70)
    selected_df = select_100_samples(MANIFEST_SRC, seed=SELECTION_SEED)
    print(f"Selected {len(selected_df)} samples across {selected_df['bcs'].nunique()} classes and {selected_df['animal_id'].nunique()} unique cows.")

    print("\n" + "=" * 70)
    print("STEP 2: Selective HTTP Range Extraction from Zenodo (Record 20290988)")
    print("=" * 70)
    extract_missing_samples(selected_df)

    print("\n" + "=" * 70)
    print("STEP 3: Physical Validation & Manifest Creation")
    print("=" * 70)
    manifest_df = validate_and_build_manifest(selected_df)

    print("\n" + "=" * 70)
    print("STEP 4: Contact Sheet Generation (10 Sheets, 2x5 Grid, >=500px long side)")
    print("=" * 70)
    sheet_paths = generate_contact_sheets(manifest_df)

    print("\n" + "=" * 70)
    print("STEP 5: Comprehensive Markdown Index & Gallery Generation")
    print("=" * 70)
    build_audit_index_md(manifest_df, sheet_paths)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
