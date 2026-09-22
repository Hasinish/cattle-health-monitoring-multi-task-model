# -*- coding: utf-8 -*-
"""
Fast Forensic Inspection of Kaggle Beef Behavior Dataset on Modal Volume beef-behavior-data.
Stage 2: Archive listing (7z l), extraction completeness check, Category Videos census, and Labelframes census.
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
    .apt_install("p7zip-full")
    .pip_install("pandas", "tqdm")
)
app = modal.App("beef-behavior-audit-stage2", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1200,
    cpu=4.0,
    memory=8192,
)
def inspect_archive_and_directories():
    """Inspect archive contents, Category Videos, and Labelframes."""
    import subprocess
    import glob

    archive_path = os.path.join(DATASET_DIR, "archive.zip")
    print("=" * 80)
    print("  KAGGLE BEEF CATTLE BEHAVIOR DATASET: STAGE 2 ARCHIVE & DATA AUDIT")
    print("=" * 80)

    # 1. 7z l archive.zip summary
    print("\n[1] Checking 7z archive listing summary...")
    # Get total file count and uncompressed size from archive
    cmd = ["7z", "l", "-slt", archive_path]
    # We can run 7z l without -slt first or grep lines
    p = subprocess.Popen(["7z", "l", archive_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tail_lines = []
    total_lines = 0
    for line in p.stdout:
        total_lines += 1
        tail_lines.append(line.strip())
        if len(tail_lines) > 25:
            tail_lines.pop(0)
    p.wait()
    print(f"Archive listing total lines: {total_lines}")
    print("Archive tail:")
    for l in tail_lines:
        print(f"  {l}")

    # 2. Inspect 'Category Videos/cows/'
    cat_dir = os.path.join(DATASET_DIR, "Category Videos", "cows")
    print("\n[2] Inspecting Category Videos/cows/ ...")
    if os.path.exists(cat_dir):
        behaviors = sorted(os.listdir(cat_dir))
        print(f"Subdirectories in Category Videos/cows/: {behaviors}")
        for b in behaviors:
            b_path = os.path.join(cat_dir, b)
            if os.path.isdir(b_path):
                files = os.listdir(b_path)
                total_sz = sum(os.path.getsize(os.path.join(b_path, f)) for f in files)
                print(f"  Behavior '{b}': {len(files):,} video clips ({total_sz / (1024*1024):.1f} MB)")
                if files:
                    print(f"    Sample files: {files[:3]}")
    else:
        print(f"Category Videos/cows does not exist at {cat_dir}")

    # 3. Inspect 'Labelframes/'
    lf_dir = os.path.join(DATASET_DIR, "Labelframes")
    print("\n[3] Inspecting Labelframes/ ...")
    if os.path.exists(lf_dir):
        sub_sessions = sorted(os.listdir(lf_dir))
        print(f"Labelframes session directories: {sub_sessions}")
        for s in sub_sessions:
            s_path = os.path.join(lf_dir, s)
            if os.path.isdir(s_path):
                track_dirs = sorted(os.listdir(s_path))
                print(f"  Session '{s}': {len(track_dirs)} track directories: {track_dirs[:15]}...")
                # Sample one track dir
                if track_dirs:
                    td = track_dirs[0]
                    td_p = os.path.join(s_path, td)
                    frames = os.listdir(td_p)
                    print(f"    Track '{td}': {len(frames)} frames. Sample frames: {frames[:3]}")
    else:
        print(f"Labelframes does not exist at {lf_dir}")

    # 4. Check whether any other files or directories were inside archive.zip
    # Find all top-level items in archive
    print("\n[4] Inspecting top-level items inside archive.zip...")
    res = subprocess.run(["7z", "l", "-ba", archive_path], capture_output=True, text=True)
    top_items = set()
    total_archive_files = 0
    total_archive_dirs = 0
    for line in res.stdout.split("\n"):
        parts = line.split()
        if len(parts) >= 6:
            # 7z output format: Date Time Attr Size Compressed Name
            name = " ".join(parts[5:])
            if name:
                top = name.split("/")[0].split("\\")[0]
                top_items.add(top)
                if "D" in parts[2]:
                    total_archive_dirs += 1
                else:
                    total_archive_files += 1

    print(f"Top-level items in archive: {sorted(list(top_items))}")
    print(f"Total files in archive: {total_archive_files:,}")
    print(f"Total directories in archive: {total_archive_dirs:,}")


@app.local_entrypoint()
def main():
    inspect_archive_and_directories.remote()


if __name__ == "__main__":
    main()
