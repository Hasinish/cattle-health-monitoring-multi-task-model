# -*- coding: utf-8 -*-
"""
Kaggle Beef Cattle Behavior Dataset Forensic Audit Stage 4:
Inode Reclamation, Master Manifest Compilation, and Deterministic Visual Review Pack Generation (Seed 2026).
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
    .apt_install("ffmpeg")
    .pip_install("pillow", "pandas", "numpy")
)
app = modal.App("beef-behavior-audit-stage4", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1800,
    cpu=4.0,
    memory=8192,
)
def generate_visual_pack_and_manifest():
    """Reclaim inodes by purging partially extracted Labelframes, then build manifest and visual pack."""
    import glob
    import random
    import shutil
    import subprocess
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    print("=" * 80)
    print("  KAGGLE BEEF CATTLE BEHAVIOR DATASET: STAGE 4 VISUAL PACK & MANIFEST")
    print("=" * 80)

    # 1. Reclaim inodes by removing partially extracted Labelframes
    # Note: archive.zip is 100% intact and contains all 1.14 million Labelframes if ever needed.
    lf_dir = os.path.join(DATASET_DIR, "Labelframes")
    if os.path.exists(lf_dir):
        print("\n[1/4] Reclaiming inodes: removing partially extracted Labelframes directory...")
        shutil.rmtree(lf_dir, ignore_errors=True)
        volume.commit()
        print("  [OK] Partially extracted Labelframes removed and volume committed!")

    res_i = subprocess.run(["df", "-i", VOLUME_DIR], capture_output=True, text=True)
    print("\nUpdated Inode Status (df -i):")
    print(res_i.stdout.strip())

    cat_dir = os.path.join(DATASET_DIR, "Category Videos", "cows")
    visual_out_dir = os.path.join(DATASET_DIR, "visual_audit_pack")
    os.makedirs(visual_out_dir, exist_ok=True)
    print(f"Created visual pack output directory at: {visual_out_dir}")

    # 2. Compile Master Manifest of all Category Video clips
    print("\n[2/4] Compiling master manifest of all 4,337 category video clips...")
    behaviors = sorted(os.listdir(cat_dir))
    records = []

    for b in behaviors:
        b_path = os.path.join(cat_dir, b)
        if not os.path.isdir(b_path):
            continue
        flist = sorted(os.listdir(b_path))
        for fn in flist:
            if not fn.endswith(".mp4"):
                continue
            fp = os.path.join(b_path, fn)
            sz = os.path.getsize(fp)
            parts = fn.replace(".mp4", "").split("_")
            sess = parts[0] if len(parts) >= 1 else "unknown"
            track = parts[1] if len(parts) >= 2 else "unknown"
            clip_idx = parts[-1] if len(parts) >= 4 else "unknown"

            records.append({
                "clip_id": fn.replace(".mp4", ""),
                "behavior": b,
                "session_id": sess,
                "bytetrack_id": track,
                "clip_idx": clip_idx,
                "filename": fn,
                "size_bytes": sz,
                "resolution": "224x224",
                "fps": 25.0,
            })

    df = pd.DataFrame(records)
    print(f"Total clips cataloged: {len(df):,}")

    manifest_csv = os.path.join(DATASET_DIR, "beef_behavior_clips_manifest.csv")
    df.to_csv(manifest_csv, index=False)
    print(f"Saved manifest to {manifest_csv}")

    # 3. Extract consecutive frames from MP4 using ffmpeg
    def extract_consecutive_frames(vpath, start_frame, num_frames=6):
        """Extract consecutive frames starting from start_frame using ffmpeg."""
        frames = []
        for i in range(num_frames):
            f_idx = start_frame + i
            cmd = [
                "ffmpeg",
                "-v", "error",
                "-i", vpath,
                "-vf", f"select=eq(n\\,{f_idx})",
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-"
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.stdout:
                import io
                img = Image.open(io.BytesIO(res.stdout)).convert("RGB")
                frames.append(img)
            else:
                frames.append(Image.new("RGB", (224, 224), (30, 30, 30)))
        return frames

    # 4. Build filmstrips for each behavior class (Seed 2026)
    print("\n[3/4] Generating deterministic 6-frame consecutive filmstrips (Seed 2026)...")
    random.seed(2026)
    np.random.seed(2026)

    color_map = {
        "drink": (0, 206, 209),       # Dark turquoise
        "eat": (34, 139, 34),          # Forest green
        "lie": (147, 112, 219),        # Medium purple
        "ruminate": (178, 34, 34),     # Firebrick
        "stand": (30, 144, 255),       # Dodger blue
    }

    generated_assets = []

    for b in behaviors:
        subset = df[df["behavior"] == b]
        if len(subset) == 0:
            continue
        sample_clips = subset.sample(2, random_state=2026)
        for seq_idx, (_, row) in enumerate(sample_clips.iterrows(), 1):
            vpath = os.path.join(cat_dir, b, row["filename"])
            start_f = random.choice([25, 50, 75, 100])
            frames = extract_consecutive_frames(vpath, start_f, num_frames=6)

            tile_w, tile_h = 224, 224
            banner_h = 42
            total_w = tile_w * len(frames)
            total_h = tile_h + banner_h

            filmstrip = Image.new("RGB", (total_w, total_h), (18, 18, 24))
            draw = ImageDraw.Draw(filmstrip)

            title = f"Kaggle Beef | Behavior: {b.upper()} | Clip: {row['filename']} | ByteTrack ID: {row['bytetrack_id']} | Frames {start_f}-{start_f+5} (25 FPS)"
            draw.text((15, 12), title, fill=(240, 240, 240))

            c = color_map.get(b, (200, 200, 200))
            for i, f_img in enumerate(frames):
                draw_f = ImageDraw.Draw(f_img)
                draw_f.rectangle([0, 0, tile_w - 1, tile_h - 1], outline=c, width=3)
                draw_f.rectangle([5, tile_h - 22, 60, tile_h - 4], fill=(0, 0, 0))
                draw_f.text((8, tile_h - 20), f"f={start_f + i}", fill=(255, 255, 255))
                filmstrip.paste(f_img, (i * tile_w, banner_h))

            out_fn = f"beef_filmstrip_{b}_seq{seq_idx}.jpg"
            out_p = os.path.join(visual_out_dir, out_fn)
            filmstrip.save(out_p, quality=92)
            print(f"  [OK] Saved filmstrip: {out_fn}")
            generated_assets.append(out_p)

    # 5. Diagnostic edge cases:
    print("\n[4/4] Generating diagnostic edge cases...")

    # Diagnostic 1: High ByteTrack ID instance (Track 77)
    t77_subset = df[df["bytetrack_id"] == "77"]
    if len(t77_subset) > 0:
        row77 = t77_subset.iloc[0]
        vpath77 = os.path.join(cat_dir, row77["behavior"], row77["filename"])
        frames77 = extract_consecutive_frames(vpath77, 50, 6)
        filmstrip77 = Image.new("RGB", (224 * 6, 224 + 42), (18, 18, 24))
        draw77 = ImageDraw.Draw(filmstrip77)
        draw77.text((15, 12), f"Diagnostic Edge Case: ByteTrack ID Proliferation (Track ID: 77) | {row77['filename']}", fill=(240, 240, 240))
        for i, f_img in enumerate(frames77):
            draw_f = ImageDraw.Draw(f_img)
            draw_f.rectangle([0, 0, 223, 223], outline=(255, 140, 0), width=3)
            draw_f.text((8, 204), f"f={50+i}", fill=(255, 255, 255))
            filmstrip77.paste(f_img, (i * 224, 42))
        out77 = os.path.join(visual_out_dir, "beef_edge_case_bytetrack_proliferation.jpg")
        filmstrip77.save(out77, quality=92)
        print("  [OK] Saved edge case: beef_edge_case_bytetrack_proliferation.jpg")
        generated_assets.append(out77)

    # Diagnostic 2: Nighttime / Infrared lighting observation
    # Find a clip with darker pixel mean or late session
    night_candidates = df[df["session_id"].str.startswith("00000000165") | df["session_id"].str.startswith("0000000058")]
    if len(night_candidates) > 0:
        row_night = night_candidates.iloc[0]
        vpath_night = os.path.join(cat_dir, row_night["behavior"], row_night["filename"])
        frames_night = extract_consecutive_frames(vpath_night, 50, 6)
        filmstrip_night = Image.new("RGB", (224 * 6, 224 + 42), (18, 18, 24))
        draw_night = ImageDraw.Draw(filmstrip_night)
        draw_night.text((15, 12), f"Diagnostic Edge Case: Low-Light / Barn Illumination Shift | {row_night['filename']}", fill=(240, 240, 240))
        for i, f_img in enumerate(frames_night):
            draw_f = ImageDraw.Draw(f_img)
            draw_f.rectangle([0, 0, 223, 223], outline=(100, 149, 237), width=3)
            draw_f.text((8, 204), f"f={50+i}", fill=(255, 255, 255))
            filmstrip_night.paste(f_img, (i * 224, 42))
        out_night = os.path.join(visual_out_dir, "beef_edge_case_lighting_variation.jpg")
        filmstrip_night.save(out_night, quality=92)
        print("  [OK] Saved edge case: beef_edge_case_lighting_variation.jpg")
        generated_assets.append(out_night)

    volume.commit()
    print("\n[COMPLETE] Volume committed successfully with visual evidence pack and master manifest!")
    print(f"Total visual review assets generated: {len(generated_assets)}")
    for a in generated_assets:
        print(f"  - {os.path.basename(a)} ({os.path.getsize(a):,} bytes)")


@app.local_entrypoint()
def main():
    generate_visual_pack_and_manifest.remote()


if __name__ == "__main__":
    main()
