# -*- coding: utf-8 -*-
"""Cloud-to-Cloud Transfer of Certified Viewpoint Checkpoint to hasinishrak2015.

Transfers the frozen real-cattle viewpoint checkpoint from:
  Source: tigerwood693, volume 'viewpoint-checkpoints'
          path: viewpoint_real_finetune/viewpoint_resnet18_real_best.pth
To:
  Destination: hasinishrak2015, volume 'mtl-checkpoints'
          path: viewpoint_aux/viewpoint_resnet18_real_best.pth

Ensures:
  1. Exact SHA-256 identity verification (source and destination).
  2. Zero modification/deletion of source checkpoint.
  3. Safe placement under viewpoint_aux/ on destination.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

try:
    import modal
except ImportError:
    print("[ERROR] modal package not found. Run: pip install modal")
    sys.exit(1)


def get_profile_creds(profile_name: str) -> Dict[str, str]:
    cfg_path = Path(os.path.expanduser("~/.modal.toml"))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    if profile_name not in cfg:
        raise KeyError(f"Profile '{profile_name}' not found in ~/.modal.toml")
    return {
        "token_id": cfg[profile_name]["token_id"],
        "token_secret": cfg[profile_name]["token_secret"],
    }


def transfer_checkpoint() -> Dict[str, Any]:
    print("=" * 70)
    print("  🚀 STEP 1: TRANSFER CERTIFIED VIEWPOINT CHECKPOINT TO hasinishrak2015")
    print("  Source     : tigerwood693 (viewpoint-checkpoints)")
    print("  Destination: hasinishrak2015 (mtl-checkpoints)")
    print("=" * 70)

    src_creds = get_profile_creds("tigerwood693")
    dst_creds = get_profile_creds("hasinishrak2015")

    src_client = modal.Client.from_credentials(src_creds["token_id"], src_creds["token_secret"])
    dst_client = modal.Client.from_credentials(dst_creds["token_id"], dst_creds["token_secret"])

    src_vol = modal.Volume.from_name("viewpoint-checkpoints", client=src_client)
    dst_vol = modal.Volume.from_name("mtl-checkpoints", client=dst_client)

    src_rel_path = "viewpoint_real_finetune/viewpoint_resnet18_real_best.pth"
    dst_rel_path = "viewpoint_aux/viewpoint_resnet18_real_best.pth"

    # Step 1: Read source into temp file & hash
    print(f"\n[1/4] Reading source checkpoint from tigerwood693...")
    t0 = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp_path = Path(tmp.name)
        src_vol.read_file_into_fileobj(src_rel_path, tmp)

    dl_time = time.time() - t0
    src_size = tmp_path.stat().st_size
    print(f"  ✓ Downloaded {src_size:,} bytes ({src_size / (1024**2):.2f} MB) in {dl_time:.2f}s")

    # Step 2: Compute SHA-256
    print("\n[2/4] Verifying source SHA-256 hash...")
    with open(tmp_path, "rb") as f:
        src_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  Source SHA-256: {src_hash}")

    EXPECTED_HASH = "a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a"
    if src_hash != EXPECTED_HASH:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Source SHA-256 mismatch! Expected {EXPECTED_HASH}, got {src_hash}")
    print("  ✓ Source hash matches known certified hash!")

    # Step 3: Upload to destination volume
    print(f"\n[3/4] Uploading to hasinishrak2015 -> {dst_rel_path}...")
    t_up = time.time()
    with dst_vol.batch_upload(force=True) as batch:
        batch.put_file(tmp_path, dst_rel_path)
    up_time = time.time() - t_up
    print(f"  ✓ Uploaded to destination in {up_time:.2f}s")

    # Step 4: Verify destination file
    print(f"\n[4/4] Verifying destination file on hasinishrak2015...")
    entries = list(dst_vol.listdir("viewpoint_aux"))
    found = False
    for e in entries:
        print(f"  - Found: {e.path} ({e.size} bytes)")
        if e.path.endswith("viewpoint_resnet18_real_best.pth"):
            found = True
            if e.size != src_size:
                raise ValueError(f"Destination size mismatch! {e.size} vs {src_size}")

    if not found:
        raise FileNotFoundError(f"Destination file not found after upload: {dst_rel_path}")

    # Verify destination hash by reading it back
    print("  Reading back destination file to verify bit-level identity...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as verify_tmp:
        verify_path = Path(verify_tmp.name)
        dst_vol.read_file_into_fileobj(dst_rel_path, verify_tmp)

    with open(verify_path, "rb") as f:
        dst_hash = hashlib.sha256(f.read()).hexdigest()
    verify_path.unlink(missing_ok=True)
    tmp_path.unlink(missing_ok=True)

    print(f"  Destination SHA-256: {dst_hash}")
    assert dst_hash == src_hash == EXPECTED_HASH, f"Hash mismatch: dst={dst_hash}, src={src_hash}"
    print(f"  ✓ Bit-identical verification PASSED! Source == Destination == {EXPECTED_HASH}")

    res = {
        "status": "SUCCESS",
        "source_profile": "tigerwood693",
        "source_volume": "viewpoint-checkpoints",
        "source_path": src_rel_path,
        "destination_profile": "hasinishrak2015",
        "destination_volume": "mtl-checkpoints",
        "destination_path": dst_rel_path,
        "size_bytes": src_size,
        "sha256": dst_hash,
    }
    return res


if __name__ == "__main__":
    res = transfer_checkpoint()
    print("\nTransfer summary:", res)
