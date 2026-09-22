# -*- coding: utf-8 -*-
"""
Fetch Kaggle Beef Cattle Behavior visual audit pack and summary CSV from Modal Volume beef-behavior-data.
"""

import os
import sys
from pathlib import Path

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

image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("beef-fetch-assets", image=image)


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def get_audit_asset_bytes(rel_path: str) -> bytes:
    """Read file bytes from volume and return to local client."""
    full_path = os.path.join(DATASET_DIR, rel_path)
    if not os.path.exists(full_path):
        return b""
    with open(full_path, "rb") as f:
        return f.read()


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=1.0,
    memory=2048,
)
def list_visual_pack_files() -> list[str]:
    """List all files in visual_audit_pack."""
    vp_dir = os.path.join(DATASET_DIR, "visual_audit_pack")
    if not os.path.exists(vp_dir):
        return []
    return sorted(os.listdir(vp_dir))


@app.local_entrypoint()
def main():
    repo_root = Path(__file__).resolve().parent.parent
    local_asset_dir = repo_root / "docs" / "audits" / "assets" / "beef_behavior_audit"
    local_artifact_dir = repo_root / "artifacts" / "behavior_audit"
    local_dataset_dir = repo_root / "datasets" / "behavior" / "beef_cattle_behavior"

    os.makedirs(local_asset_dir, exist_ok=True)
    os.makedirs(local_artifact_dir, exist_ok=True)
    os.makedirs(local_dataset_dir, exist_ok=True)

    print("Fetching master manifest CSV...")
    csv_bytes = get_audit_asset_bytes.remote("beef_behavior_clips_manifest.csv")
    if csv_bytes:
        csv_dest = local_artifact_dir / "beef_behavior_audit_summary.csv"
        with open(csv_dest, "wb") as f:
            f.write(csv_bytes)
        print(f"[OK] Saved {csv_dest} ({len(csv_bytes):,} bytes)")

        # Copy to dataset directory
        with open(local_dataset_dir / "manifest.csv", "wb") as f:
            f.write(csv_bytes)
        print(f"[OK] Saved {local_dataset_dir / 'manifest.csv'}")

    print("\nListing visual pack files on volume...")
    files = list_visual_pack_files.remote()
    print(f"Found {len(files)} visual pack files on volume:")
    for fn in files:
        print(f"  Downloading {fn}...")
        b = get_audit_asset_bytes.remote(f"visual_audit_pack/{fn}")
        if b:
            dest = local_asset_dir / fn
            with open(dest, "wb") as f:
                f.write(b)
            print(f"  [OK] Saved {dest.relative_to(repo_root)} ({len(b):,} bytes)")

    print("\n[SUCCESS] All Kaggle Beef audit assets fetched locally!")


if __name__ == "__main__":
    main()
