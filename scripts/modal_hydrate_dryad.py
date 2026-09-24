# -*- coding: utf-8 -*-
"""Modal Hydration and Verification Pipeline for Dryad Cattle BCS Dataset.

Source: Winkler & Boucheron (New Mexico State University), Dryad DOI: 10.5061/dryad.tqjq2bw4s
Volume: 'dryad-bcs-data' mounted at /data

Note on Dryad Upstream:
  DataDryad uses Cloudflare/Anubis JavaScript Proof-of-Work bot challenge that blocks
  automated server-side HTTP downloads (HTTP 403 Forbidden).
  The authentic 5,940 DGE TIFF images (Classes 2-7, 54 biological cows, 854 MB)
  are already verified locally in datasets/bcs/dryad_bcs/Total_sorted_DGE_images/.
  This script uploads the authentic local dataset directly to the persistent Modal volume
  in ~20-30 seconds using Modal's high-speed batch_upload engine.

Usage:
  # Upload to Modal volume on profile hasinishrak2015:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_dryad.py::upload_dryad

  # Upload to Modal volume on profile dryousufmozumder:
  modal run --profile dryousufmozumder scripts/modal_hydrate_dryad.py::upload_dryad

  # Verify on either profile:
  modal run --profile hasinishrak2015 scripts/modal_hydrate_dryad.py::verify
  modal run --profile dryousufmozumder scripts/modal_hydrate_dryad.py::verify
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATASET_DIR = REPO_ROOT / "datasets" / "bcs" / "dryad_bcs" / "Total_sorted_DGE_images"

# Persistent storage volume for Dryad BCS
volume = modal.Volume.from_name("dryad-bcs-data", create_if_missing=True)
VOLUME_DIR = "/data"
REMOTE_DATASET_DIR = "/data/Total_sorted_DGE_images"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pillow", "tqdm")
)

app = modal.App("dryad-bcs-pipeline", image=image)


# ==============================================================================
# REMOTE VERIFICATION FUNCTION
# ==============================================================================
@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def verify_dryad():
    """Verify extracted Dryad Cattle BCS dataset in Modal volume."""
    from PIL import Image

    print("=" * 70)
    print("  DRYAD CATTLE BCS PHYSICAL VERIFICATION")
    print(f"  Volume Directory: {REMOTE_DATASET_DIR}")
    print("=" * 70)

    if not os.path.exists(REMOTE_DATASET_DIR):
        print(f"[ERROR] Directory {REMOTE_DATASET_DIR} does not exist in volume.")
        return {"status": "not_found", "total_images": 0}

    expected_classes = ["2", "3", "4", "5", "6", "7"]
    class_counts = {}
    total_images = 0
    cows_found = set()
    sample_images = []

    for root, _, files in os.walk(REMOTE_DATASET_DIR):
        for f in files:
            if f.lower().endswith((".tiff", ".tif", ".jpg", ".png")):
                total_images += 1
                p = os.path.join(root, f)
                rel = os.path.relpath(p, REMOTE_DATASET_DIR)
                parts = rel.replace("\\", "/").split("/")
                if len(parts) >= 1 and parts[0] in expected_classes:
                    c = parts[0]
                    class_counts[c] = class_counts.get(c, 0) + 1
                # Cow folder detection (e.g. Cow_1)
                for part in parts:
                    if part.startswith("Cow_") or part.startswith("cow_"):
                        cows_found.add(part)
                if len(sample_images) < 5:
                    sample_images.append(p)

    total_bytes = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(REMOTE_DATASET_DIR)
        for f in files
    )
    size_mb = total_bytes / (1024 * 1024)

    print(f"✓ Total Images Found: {total_images:,} (expected 5,940)")
    print(f"✓ Total Disk Space:  {size_mb:.1f} MB")
    print(f"✓ Unique Cows Found: {len(cows_found)} (expected 54)")
    print("\nClass Counts (1-9 Wagner scale, active 2-7):")
    for c in expected_classes:
        cnt = class_counts.get(c, 0)
        print(f"  - Class {c}: {cnt:,} images")

    print("\nSample PIL TIFF Decodes:")
    for p in sample_images:
        try:
            with Image.open(p) as img:
                rel_p = os.path.relpath(p, REMOTE_DATASET_DIR)
                print(f"  ✓ {rel_p}: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"  ✗ {p}: {e}")

    is_verified = total_images == 5940 and len(cows_found) >= 50
    status_str = "verified" if is_verified else "incomplete"
    print(f"\n[STATUS] Dryad BCS volume status: {status_str.upper()} ({total_images}/5,940)")

    return {
        "status": status_str,
        "total_images": total_images,
        "unique_cows": len(cows_found),
        "size_mb": round(size_mb, 1),
        "class_counts": class_counts,
    }


# ==============================================================================
# LOCAL UPLOAD ENTRYPOINT
# ==============================================================================
@app.local_entrypoint()
def upload_dryad():
    """Upload verified local Dryad BCS dataset directly to Modal volume."""
    print("=" * 70)
    print("  DRYAD CATTLE BCS MODAL VOLUME UPLOADER")
    print(f"  Source Directory: {LOCAL_DATASET_DIR}")
    print(f"  Target Volume:    dryad-bcs-data mounted at {REMOTE_DATASET_DIR}")
    print("=" * 70)

    if not LOCAL_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Local Dryad dataset not found at {LOCAL_DATASET_DIR}!\n"
            "Please ensure datasets/bcs/dryad_bcs/Total_sorted_DGE_images/ exists."
        )

    all_files = [p for p in LOCAL_DATASET_DIR.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in all_files)
    print(f"[*] Found {len(all_files):,} files ({total_bytes / (1024*1024):.1f} MB) to upload.")

    vol = modal.Volume.from_name("dryad-bcs-data")
    t0 = time.time()
    print("[*] Uploading files to Modal volume via multi-part batch upload...")

    with vol.batch_upload(force=True) as batch:
        for f in all_files:
            rel = f.relative_to(LOCAL_DATASET_DIR).as_posix()
            remote_path = f"/Total_sorted_DGE_images/{rel}"
            batch.put_file(f, remote_path)

    elapsed = time.time() - t0
    rate = (total_bytes / (1024 * 1024)) / max(elapsed, 0.001)
    print(f"✓ Upload complete in {elapsed:.1f}s ({rate:.1f} MB/s)!")

    print("\n[*] Running remote verification check...")
    res = verify_dryad.remote()
    print(f"✓ Verification result: {res}")


@app.local_entrypoint()
def verify():
    """Run standalone remote verification."""
    res = verify_dryad.remote()
    print(f"✓ Verification result: {res}")
