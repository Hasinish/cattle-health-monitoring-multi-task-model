# -*- coding: utf-8 -*-
"""
Deep Forensic Inspection of Kaggle Beef Behavior Dataset on Modal Volume beef-behavior-data.
Stage 3: Video decoding, resolution/FPS/duration analysis, annotation discovery, and archive tree.
"""

import os
import sys

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

volume = modal.Volume.from_name("beef-behavior-data", create_if_missing=True)
VOLUME_DIR = "/data"
DATASET_DIR = "/data/beef_behavior"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "p7zip-full")
    .pip_install("pillow", "pandas", "numpy")
)
app = modal.App("beef-behavior-audit-stage3", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1800,
    cpu=4.0,
    memory=8192,
)
def audit_stage3_video_and_annotations():
    """Analyze video properties, sample frames, detect annotations, and list archive structure."""
    import glob
    import json
    import subprocess
    import zipfile
    from collections import Counter, defaultdict
    import numpy as np
    import pandas as pd
    from PIL import Image

    print("=" * 80)
    print("  KAGGLE BEEF CATTLE BEHAVIOR DATASET: STAGE 3 FORENSIC AUDIT")
    print("=" * 80)

    cat_dir = os.path.join(DATASET_DIR, "Category Videos", "cows")
    archive_path = os.path.join(DATASET_DIR, "archive.zip")

    # ------------------------------------------------------------------
    # 1. FIND ALL ANNOTATION / METADATA FILES IN ARCHIVE & EXTRACTED
    # ------------------------------------------------------------------
    print("\n[1] Searching for annotation / label / metadata files inside archive.zip...")
    # Read zip central directory using zipfile (fast in Python)
    zf = zipfile.ZipFile(archive_path, 'r')
    infolist = zf.infolist()
    print(f"Total entries in archive.zip: {len(infolist):,}")

    top_level_dirs = set()
    extensions_in_zip = Counter()
    metadata_files = []

    for info in infolist:
        fn = info.filename
        top = fn.split("/")[0]
        top_level_dirs.add(top)
        ext = os.path.splitext(fn)[1].lower() or "no_ext"
        extensions_in_zip[ext] += 1
        if ext in [".txt", ".csv", ".json", ".xml", ".yaml", ".yml", ".md", ".pbtx", ".xlsx"]:
            metadata_files.append((fn, info.file_size))

    print(f"\nTop-level folders in archive.zip: {sorted(list(top_level_dirs))}")
    print("\nExtensions in archive.zip:")
    for ext, cnt in extensions_in_zip.most_common(15):
        print(f"  {ext:10s}: {cnt:,}")

    print(f"\nDiscovered Metadata / Annotation files in archive ({len(metadata_files)} files):")
    for fn, sz in sorted(metadata_files, key=lambda x: x[1], reverse=True)[:30]:
        print(f"  - {fn} ({sz:,} bytes)")

    # Read content of small metadata/annotation files
    print("\nReading sample metadata files:")
    for fn, sz in metadata_files:
        if sz < 10000 and any(k in fn.lower() for k in ["label", "behavior", "readme", "class", "train", "val", "test", "data"]):
            try:
                content = zf.read(fn).decode("utf-8", errors="ignore")
                print(f"\n--- Content of {fn} ({sz} bytes) ---")
                print(content[:1000])
            except Exception as e:
                print(f"Error reading {fn}: {e}")

    # ------------------------------------------------------------------
    # 2. CENSUS OF CATEGORY VIDEOS (BEHAVIOR CLIPS)
    # ------------------------------------------------------------------
    print("\n[2] Exhaustive Census of Category Videos...")
    behaviors = sorted(os.listdir(cat_dir))
    clip_records = []

    for b in behaviors:
        b_path = os.path.join(cat_dir, b)
        if not os.path.isdir(b_path):
            continue
        flist = sorted(os.listdir(b_path))
        print(f"Processing behavior: '{b}' ({len(flist)} clips)...")
        for fn in flist:
            if not fn.endswith(".mp4"):
                continue
            fp = os.path.join(b_path, fn)
            sz = os.path.getsize(fp)
            # Parse filename: e.g. 00000000082000000_3_clip_14.mp4
            parts = fn.replace(".mp4", "").split("_")
            sess = parts[0] if len(parts) >= 1 else "unknown"
            track = parts[1] if len(parts) >= 2 else "unknown"
            clip_idx = parts[-1] if len(parts) >= 4 else "unknown"

            clip_records.append({
                "behavior": b,
                "filename": fn,
                "session_id": sess,
                "track_id": track,
                "clip_idx": clip_idx,
                "size_bytes": sz,
                "filepath": fp,
            })

    df_clips = pd.DataFrame(clip_records)
    print(f"\nTotal Category Video clips cataloged: {len(df_clips):,}")
    print("\nClips per Behavior:")
    print(df_clips["behavior"].value_counts())
    print("\nUnique Sessions in Category Videos:")
    print(df_clips["session_id"].value_counts())
    print("\nTop Track IDs in Category Videos:")
    print(df_clips["track_id"].value_counts().head(20))

    # ------------------------------------------------------------------
    # 3. PROBE SAMPLE VIDEOS (RESOLUTION, FPS, DURATION, CODEC)
    # ------------------------------------------------------------------
    print("\n[3] Probing Video Clips via ffprobe...")
    import json

    def probe_video(vpath):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames,codec_name",
            "-of", "json",
            vpath,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        try:
            d = json.loads(res.stdout)
            s = d.get("streams", [{}])[0]
            w = int(s.get("width", 0))
            h = int(s.get("height", 0))
            codec = s.get("codec_name", "")
            dur = float(s.get("duration", 0) or 0)
            n_frames = int(s.get("nb_frames", 0) or 0)
            fps_str = s.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/")) if "/" in fps_str else (30, 1)
            fps = num / den if den > 0 else 30.0
            return {"width": w, "height": h, "codec": codec, "duration": dur, "frames": n_frames, "fps": fps}
        except Exception as e:
            return {"error": str(e)}

    # Probe 50 diverse clips across all behaviors
    probe_results = []
    for b in behaviors:
        subset = df_clips[df_clips["behavior"] == b]
        sample_subset = subset.sample(min(10, len(subset)), random_state=2026)
        for _, row in sample_subset.iterrows():
            info = probe_video(row["filepath"])
            info["behavior"] = b
            info["filename"] = row["filename"]
            probe_results.append(info)

    df_probes = pd.DataFrame(probe_results)
    print("\nVideo Probe Summary (N=50 sample clips across all 5 behaviors):")
    print(df_probes[["behavior", "width", "height", "fps", "duration", "frames"]].head(15))

    print("\nResolution Distribution across Probed Clips:")
    print(df_probes.groupby(["width", "height"]).size())

    print("\nFPS Distribution:")
    print(df_probes["fps"].value_counts())

    print("\nDuration Distribution (seconds):")
    print(df_probes["duration"].describe())

    print("\nFrame Count Distribution:")
    print(df_probes["frames"].describe())

    # ------------------------------------------------------------------
    # 4. INSPECT 'videos_cut/videos/' IN ARCHIVE
    # ------------------------------------------------------------------
    print("\n[4] Inspecting videos_cut/ in archive.zip...")
    vc_files = [info for info in infolist if info.filename.startswith("videos_cut/")]
    print(f"Total entries under videos_cut/: {len(vc_files)}")
    mp4_in_vc = [info for info in vc_files if info.filename.endswith(".mp4")]
    print(f"Total MP4 videos under videos_cut/: {len(mp4_in_vc)}")
    for info in mp4_in_vc[:10]:
        print(f"  - {info.filename} ({info.file_size / (1024*1024):.2f} MB)")

    # ------------------------------------------------------------------
    # 5. INSPECT 'Labelframes/' IN ARCHIVE
    # ------------------------------------------------------------------
    print("\n[5] Inspecting Labelframes/ in archive.zip...")
    lf_entries = [info for info in infolist if info.filename.startswith("Labelframes/")]
    print(f"Total entries under Labelframes/: {len(lf_entries):,}")
    lf_jpgs = [info for info in lf_entries if info.filename.endswith(".jpg")]
    print(f"Total JPEG images under Labelframes/: {len(lf_jpgs):,}")

    # Check image dimensions of sample Labelframe
    print("\n[6] Inspecting sample Labelframes images...")
    sample_lf_dir = os.path.join(DATASET_DIR, "Labelframes", "00000000082000000", "1")
    if os.path.exists(sample_lf_dir):
        imgs = sorted(os.listdir(sample_lf_dir))[:5]
        for img_fn in imgs:
            img_p = os.path.join(sample_lf_dir, img_fn)
            with Image.open(img_p) as im:
                print(f"  {img_fn}: size={im.size}, mode={im.mode}, format={im.format}")


@app.local_entrypoint()
def main():
    audit_stage3_video_and_annotations.remote()


if __name__ == "__main__":
    main()
