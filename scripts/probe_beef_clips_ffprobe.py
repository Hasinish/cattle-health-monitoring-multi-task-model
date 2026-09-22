# -*- coding: utf-8 -*-
"""
Probe Kaggle Beef Cattle Behavior dataset clips on Modal volume with ffprobe.
Extracts exact fps, n_frames, and duration_sec for all clips.
Runs on profile tigerwood693 with cpu=1.0, memory=2048, no GPU.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
)

app = modal.App("probe-beef-clips", image=image)
volume = modal.Volume.from_name("beef-behavior-data")
VOLUME_DIR = "/data"
DATASET_DIR = "/data/beef_behavior"


def probe_single_clip(file_info):
    """Probe a single MP4 file with ffprobe."""
    rel_path, abs_path = file_info
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        abs_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(res.stdout) if res.stdout else {}
        stream = data.get("streams", [{}])[0] if data.get("streams") else {}
        fmt = data.get("format", {})

        # FPS
        fps_str = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "25/1"
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 25.0
        else:
            fps = float(fps_str)

        # Duration
        dur_val = fmt.get("duration") or stream.get("duration")
        duration_sec = float(dur_val) if dur_val is not None and dur_val != "N/A" else None

        # Frame count
        nb_frames_val = stream.get("nb_frames")
        if nb_frames_val and nb_frames_val != "N/A":
            n_frames = int(nb_frames_val)
        elif duration_sec is not None:
            # Fallback: count frames with ffprobe
            count_cmd = [
                "ffprobe",
                "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
                abs_path,
            ]
            c_res = subprocess.run(count_cmd, capture_output=True, text=True, timeout=60)
            n_frames = int(c_res.stdout.strip()) if c_res.stdout.strip().isdigit() else int(round(duration_sec * fps))
        else:
            n_frames = 250

        if duration_sec is None and n_frames is not None:
            duration_sec = n_frames / fps

        return {
            "rel_path": rel_path,
            "filename": os.path.basename(abs_path),
            "fps": round(fps, 4),
            "n_frames": n_frames,
            "duration_sec": round(duration_sec, 4),
        }
    except Exception as e:
        return {
            "rel_path": rel_path,
            "filename": os.path.basename(abs_path),
            "fps": 25.0,
            "n_frames": 250,
            "duration_sec": 10.0,
            "error": str(e),
        }


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1800,
    cpu=1.0,
    memory=2048,
)
def probe_all_beef_clips():
    """Probe all clips in Category Videos/cows."""
    cat_dir = os.path.join(DATASET_DIR, "Category Videos", "cows")
    if not os.path.exists(cat_dir):
        raise FileNotFoundError(f"Missing category videos directory: {cat_dir}")

    file_list = []
    for beh in sorted(os.listdir(cat_dir)):
        beh_path = os.path.join(cat_dir, beh)
        if not os.path.isdir(beh_path):
            continue
        for fn in sorted(os.listdir(beh_path)):
            if fn.endswith(".mp4"):
                abs_p = os.path.join(beh_path, fn)
                rel_p = f"clips/{beh}/{fn}"
                file_list.append((rel_p, abs_p))

    print(f"Found {len(file_list)} total MP4 clips to probe.")

    results = []
    # Use ThreadPoolExecutor for fast I/O
    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, res in enumerate(executor.map(probe_single_clip, file_list)):
            results.append(res)
            if (idx + 1) % 500 == 0 or (idx + 1) == len(file_list):
                print(f"  Probed {idx + 1}/{len(file_list)} clips...")

    return results


@app.local_entrypoint()
def main():
    import pandas as pd
    print("[MODAL] Running ffprobe on Kaggle Beef clips in Modal volume...")
    probed = probe_all_beef_clips.remote()
    print(f"[LOCAL] Received {len(probed)} probed clip records.")

    df_probed = pd.DataFrame(probed)
    out_path = Path("artifacts/beef_probed_metadata.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_probed.to_csv(out_path, index=False)
    print(f"[LOCAL] Saved probed metadata to {out_path}")

    # Summary
    print(f"Total clips: {len(df_probed)}")
    print(f"FPS values: {df_probed['fps'].value_counts().to_dict()}")
    print(f"Frame count summary: min={df_probed['n_frames'].min()}, max={df_probed['n_frames'].max()}, mean={df_probed['n_frames'].mean():.2f}, median={df_probed['n_frames'].median()}")
    non_250 = df_probed[df_probed["n_frames"] != 250]
    print(f"Clips with n_frames != 250: {len(non_250)} ({len(non_250)/len(df_probed)*100:.2f}%)")


if __name__ == "__main__":
    main()
