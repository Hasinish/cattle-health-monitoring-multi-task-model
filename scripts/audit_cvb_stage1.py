# -*- coding: utf-8 -*-
"""
Deep Forensic Inspection of CVB Dataset on Modal Volume cvb-data.
Stage 1: Metadata, File System Structure, and Annotation Schemas.
"""

import os
import sys
from pathlib import Path

# Guard against Windows cross-drive ValueError in ntpath.commonpath for Modal
if sys.platform == "win32":
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

volume = modal.Volume.from_name("cvb-data", create_if_missing=True)
VOLUME_DIR = "/data"
CVB_DIR = "/data/cvb"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pillow", "pandas", "tqdm")
)

app = modal.App("cvb-forensic-audit", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=1800,  # 30 mins max
    cpu=2.0,
    memory=4096,
)
def inspect_cvb_stage1():
    """Exhaustive inspection of CVB directory tree, metadata, and JSON schemas."""
    import json
    import glob
    from collections import Counter, defaultdict

    print("=" * 80)
    print("  CVB (CATTLE VISUAL BEHAVIORS) STAGE 1: METADATA & SCHEMA AUDIT")
    print("=" * 80)

    # 1. Locate root folder inside CVB_DIR
    print("\n--- 1. Root Directory Listing ---")
    subdirs = os.listdir(CVB_DIR)
    print(f"Top-level items in {CVB_DIR}: {subdirs}")

    root_path = os.path.join(CVB_DIR, "000058916v001") if "000058916v001" in subdirs else CVB_DIR
    print(f"Working root path: {root_path}")
    print(f"Subdirectories under root: {os.listdir(root_path)}")

    data_dir = os.path.join(root_path, "data")
    meta_dir = os.path.join(root_path, "metadata")

    # 2. Inspect Metadata Files
    print("\n--- 2. Metadata Files Inspection ---")
    if os.path.exists(meta_dir):
        print(f"Metadata files in {meta_dir}: {os.listdir(meta_dir)}")
        for f in os.listdir(meta_dir):
            fp = os.path.join(meta_dir, f)
            print(f"\n>>> File: {f} ({os.path.getsize(fp)} bytes) <<<")
            if f.endswith((".md", ".txt", ".xml", ".pbtx", ".pbtxt")):
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    print(content[:2000])  # First 2000 chars

    # Inspect cvb_in_ava_format
    ava_dir = os.path.join(data_dir, "cvb_in_ava_format") if os.path.exists(data_dir) else None
    if ava_dir and os.path.exists(ava_dir):
        print(f"\n--- 3. AVA Format Files in {ava_dir} ---")
        print(f"Files: {os.listdir(ava_dir)}")
        for f in os.listdir(ava_dir):
            fp = os.path.join(ava_dir, f)
            print(f"\n>>> File: {f} ({os.path.getsize(fp)} bytes) <<<")
            if f.endswith((".pbtx", ".pbtxt", ".csv", ".txt")):
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [fh.readline() for _ in range(25)]
                    print("".join(lines))

    # 4. Inspect Cut Directories and JSON Annotations
    ann_dir = os.path.join(data_dir, "annotations")
    raw_dir = os.path.join(data_dir, "raw_frames")

    raw_cuts = sorted(os.listdir(raw_dir)) if os.path.exists(raw_dir) else []
    ann_cuts = sorted(os.listdir(ann_dir)) if os.path.exists(ann_dir) else []

    print(f"\n--- 4. Cut Directories Count ---")
    print(f"Raw cuts count: {len(raw_cuts)}")
    print(f"Annotation cuts count: {len(ann_cuts)}")
    print(f"Raw cuts == Annotation cuts? {raw_cuts == ann_cuts}")

    # Inspect 5 sample JSON files from different cuts
    print("\n--- 5. Sample JSON Schemas ---")
    sample_jsons = glob.glob(os.path.join(ann_dir, "*", "**", "*.json"), recursive=True)
    print(f"Total JSON files found: {len(sample_jsons)}")
    for jpath in sample_jsons[:3]:
        print(f"\n>>> JSON Path: {os.path.relpath(jpath, ann_dir)} <<<")
        with open(jpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        print("Keys:", list(data.keys()))
        if "info" in data:
            print("Info:", data["info"])
        if "categories" in data:
            print("Categories:", data["categories"])
        if "images" in data:
            print(f"Images count: {len(data['images'])}")
            print("Sample image record:", data["images"][0])
        if "annotations" in data:
            print(f"Annotations count: {len(data['annotations'])}")
            if data["annotations"]:
                print("Sample annotation record:", data["annotations"][0])

    # 6. Parse behavior tags from directory names
    print("\n--- 6. Cut Directory Name Taxonomy ---")
    beh_counter = Counter()
    ani_counter = Counter()
    cam_counter = Counter()
    date_counter = Counter()

    for cut in raw_cuts:
        parts = cut.split("_")
        for p in parts:
            if p.startswith("beh"):
                beh_counter[p] += 1
            elif p.startswith("ani"):
                ani_counter[p] += 1
            elif p.startswith("gopro") or p.startswith("arm"):
                cam_counter[p] += 1
            elif len(p) == 8 and p.isdigit() and p.startswith("2020"):
                date_counter[p] += 1

    print("Behaviors in cut names:", sorted(beh_counter.items()))
    print("Animals in cut names:", sorted(ani_counter.items()))
    print("Cameras in cut names:", sorted(cam_counter.items()))
    print("Dates in cut names:", sorted(date_counter.items()))

