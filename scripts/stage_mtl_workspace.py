# -*- coding: utf-8 -*-
"""Maximum-Speed Resumable MTL Data Staging Controller.

Coordinates the high-throughput staging of all Train/Val datasets for:
- Run 7 (E1 Hard-Shared MTL)
- Run 8 (E3 Modular/Task-Private MTL)
on target Modal profile: 'hasinishrak2015'.

Data Staging Architecture:
1. BCS (tigerwood697 -> hasinishrak2015):
   Source-side chunking of pre-computed monolithic tensors (train_bcs_224.pt, val_bcs_224.pt)
   + manifests. Relayed 1 chunk at a time via PC, verified by SHA-256, local chunks
   deleted immediately to maintain <= 1 GB peak local disk footprint. Reassembled on target.
   Strictly zero test data transferred.
2. Behavior (tigerwood693 -> hasinishrak2015):
   Source-side fast uncompressed tar archive of 4,271 retained sequence directories (3,641 Train,
   630 Val) + manifests. Relayed chunk by chunk via PC, verified by SHA-256, local chunks deleted
   immediately. Extracted directly into /mtl-data/behavior/. Strictly zero test data transferred.
3. Re-ID (Zenodo -> hasinishrak2015 DIRECT):
   16-stream HTTP Range download of parlor.zip directly inside hasinishrak2015 ephemeral /tmp/.
   Selective extraction of exactly 15,436 Train/Val image & mask pairs for the 41 representation
   cows. Zero bytes relayed through PC. Temporary parlor.zip purged immediately.

Commands:
  python scripts/stage_mtl_workspace.py --task all --dry-run
  python scripts/stage_mtl_workspace.py --task all --fast --yes
  python scripts/stage_mtl_workspace.py --task bcs --fast
  python scripts/stage_mtl_workspace.py --task behavior --fast
  python scripts/stage_mtl_workspace.py --task reid --fast
  python scripts/stage_mtl_workspace.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# CLEAN LIVE PROGRESS BAR (Enhanced for Task, Stage, Chunks, MB/s, and ETA)
# ==============================================================================
class CleanProgressBar:
    """Thread-safe, live in-place carriage-return (\\r) progress bar.

    Displays:
    Task | Stage | [=========>   ] % | files/chunks | GB done / total | MB/s | Elapsed | ETA
    """

    def __init__(
        self,
        total_bytes: int,
        desc: str = "",
        initial_bytes: int = 0,
        unit_label: str = "bytes",
        bar_len: int = 22,
    ) -> None:
        self.total = max(1, total_bytes)
        self.current = initial_bytes
        self.desc = desc
        self.unit_label = unit_label
        self.bar_len = bar_len
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_render_time = 0.0
        self.last_bytes = initial_bytes
        self.speed = 0.0
        self._render(force=True)

    def update(self, n_bytes: int) -> None:
        with self.lock:
            self.current += n_bytes
            now = time.time()
            dt = now - self.last_render_time
            if dt >= 0.10 or self.current >= self.total:  # ~10 Hz update
                if dt > 0:
                    self.speed = (self.current - self.last_bytes) / dt
                    self.last_render_time = now
                    self.last_bytes = self.current
                self._render(force=False)

    def set_current(self, current_bytes: int) -> None:
        with self.lock:
            now = time.time()
            dt = now - self.last_render_time
            if dt > 0:
                self.speed = (current_bytes - self.last_bytes) / dt
                self.last_render_time = now
                self.last_bytes = current_bytes
            self.current = current_bytes
            self._render(force=False)

    def write(self, msg: str) -> None:
        with self.lock:
            sys.stdout.write(f"\r{' ' * 110}\r{msg}\n")
            sys.stdout.flush()
            self._render(force=True)

    def _render(self, force: bool = False) -> None:
        pct = (self.current / self.total * 100) if self.total > 0 else 100.0
        filled = int(self.bar_len * self.current / self.total) if self.total > 0 else self.bar_len
        filled = min(self.bar_len, max(0, filled))
        arrow = ">" if (filled < self.bar_len and self.current < self.total) else "="
        bar = "=" * max(0, filled - (1 if arrow == ">" else 0)) + (arrow if filled > 0 else "")
        bar = bar.ljust(self.bar_len)

        def fmt_size(b: int) -> str:
            if b >= 1024**3:
                return f"{b / (1024**3):.2f} GB"
            if b >= 1024**2:
                return f"{b / (1024**2):.1f} MB"
            if b >= 1024:
                return f"{b / 1024:.1f} KB"
            return f"{b} B"

        curr_str = fmt_size(self.current)
        total_str = fmt_size(self.total)
        speed_str = f"{self.speed / (1024**2):.1f} MB/s" if self.speed > 0 else "-- MB/s"

        elapsed_sec = int(time.time() - self.start_time)
        em, es = divmod(elapsed_sec, 60)
        eh, em = divmod(em, 60)
        elapsed_str = f"{eh:02d}:{em:02d}:{es:02d}" if eh > 0 else f"{em:02d}:{es:02d}"

        rem_bytes = max(0, self.total - self.current)
        if self.speed > 0 and rem_bytes > 0:
            eta_sec = int(rem_bytes / self.speed)
            m, s = divmod(eta_sec, 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
        else:
            eta_str = "--:--"

        line = f"\r{self.desc} [{bar}] {pct:5.1f}% | {curr_str} / {total_str} | {speed_str} | Elapsed {elapsed_str} | ETA {eta_str}"
        sys.stdout.write(line.ljust(110))
        sys.stdout.flush()

    def close(self) -> None:
        with self.lock:
            self._render(force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()


def _compute_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute SHA-256 with 16MB stream buffers."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# MODAL CLIENT & PROFILE HELPERS
# ==============================================================================
def get_modal_client_for_profile(profile_name: str) -> Any:
    """Reads credentials from ~/.modal.toml for a specific profile and returns a modal.Client."""
    try:
        import modal
        import toml
    except ImportError as e:
        raise RuntimeError(f"Required package missing: {e}. Run pip install modal toml")

    config_path = Path(os.environ.get("USERPROFILE", os.environ.get("HOME", ""))) / ".modal.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Modal config at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        toml_data = toml.load(f)

    if profile_name not in toml_data:
        raise KeyError(f"Profile '{profile_name}' not found in ~/.modal.toml. Available: {list(toml_data.keys())}")

    profile_conf = toml_data[profile_name]
    token_id = profile_conf.get("token_id")
    token_secret = profile_conf.get("token_secret")

    if not token_id or not token_secret:
        raise ValueError(f"Profile '{profile_name}' is missing token_id or token_secret")

    return modal.Client.from_credentials(token_id, token_secret)


# ==============================================================================
# RELAY ENGINE: SOURCE -> PC CHUNK -> TARGET (Sequential, Low Disk)
# ==============================================================================
def transfer_chunk_relay(
    source_vol: Any,
    source_path: str,
    target_vol: Any,
    target_path: str,
    local_temp_file: Path,
    expected_size: int,
    expected_sha: str,
    task_name: str,
    chunk_idx: int,
    num_chunks: int,
    buffer_mb: int = 16,
    keep_temp: bool = False,
) -> None:
    """Transfers a single chunk from source volume to target volume via a low-memory local buffer.

    Verifies checksums and immediately purges local temporary storage.
    """
    import modal

    local_temp_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Download chunk from source
    dl_pbar = CleanProgressBar(
        total_bytes=expected_size,
        desc=f"{task_name} | DL CHUNK {chunk_idx + 1}/{num_chunks}",
    )

    # Read in stream blocks
    with open(local_temp_file, "wb") as f:
        def _cb(n_bytes: int):
            dl_pbar.update(n_bytes)

        # Using modal.Volume.read_file_into_fileobj with progress callback
        source_vol.read_file_into_fileobj(source_path, f, progress_cb=_cb)
    dl_pbar.close()

    if local_temp_file.stat().st_size != expected_size:
        raise ValueError(f"Downloaded chunk size mismatch: expected {expected_size}, got {local_temp_file.stat().st_size}")

    # 2. Checksum verification with configured buffer size
    actual_sha = _compute_sha256(local_temp_file, chunk_size=buffer_mb * 1024 * 1024)
    if actual_sha != expected_sha:
        local_temp_file.unlink()
        raise ValueError(f"Checksum mismatch for chunk {source_path}: expected {expected_sha}, got {actual_sha}")

    # 3. Upload chunk to target volume
    up_pbar = CleanProgressBar(
        total_bytes=expected_size,
        desc=f"{task_name} | UP CHUNK {chunk_idx + 1}/{num_chunks}",
    )

    # Wrap file in monitored reader for progress
    class MonitoredReader(io.BufferedReader):
        def __init__(self, raw, pbar):
            super().__init__(raw)
            self._pbar = pbar

        def read(self, size=-1):
            chunk = super().read(size)
            if chunk:
                self._pbar.update(len(chunk))
            return chunk

    with open(local_temp_file, "rb") as raw_f:
        wrapped_f = MonitoredReader(raw_f, up_pbar)
        with target_vol.batch_upload(force=True) as batch:
            batch.put_file(wrapped_f, target_path)
    up_pbar.close()

    # 4. Immediate local deletion to preserve low disk requirement
    if not keep_temp and local_temp_file.exists():
        local_temp_file.unlink()


# ==============================================================================
# PIPELINE EXECUTION FOR BCS
# ==============================================================================
def stage_bcs(
    dry_run: bool = False,
    chunk_size_mb: int = 1024,
    buffer_mb: int = 16,
    keep_temp: bool = False,
    temp_dir: Path = REPO_ROOT / "scratch" / "mtl_staging_temp",
) -> Dict[str, Any]:
    """Stages BCS tensors from tigerwood697 to hasinishrak2015."""
    import modal

    print("\n" + "=" * 70)
    print("  TASK A: BCS DATASET STAGING")
    print("  Source:      tigerwood697 (sciencedb-perception-cache)")
    print("  Destination: hasinishrak2015 (mtl-data/bcs)")
    print("=" * 70)

    # 1. Trigger export preparation on tigerwood697
    if dry_run:
        print("[DRY-RUN] Would execute modal_export_bcs.py::prepare_bcs_export_remote on tigerwood697")
        print("[DRY-RUN] Estimated payload: train_bcs_224.pt (~6.89 GB) + val_bcs_224.pt (~1.57 GB) = ~8.46 GB")
        est_chunks = math.ceil(8467234812 / (chunk_size_mb * 1024 * 1024))
        print(f"[DRY-RUN] Chunk size: {chunk_size_mb} MB (~{est_chunks} sequential chunks, relay buffer: {buffer_mb} MB)")
        print("[DRY-RUN] Target reassembly would verify SHA-256 and assert test_bcs_224.pt is absent")
        return {"status": "DRY_RUN", "estimated_bytes": 8467234812}

    print("[*] Contacting source Modal profile tigerwood697 for BCS export...")
    src_client = get_modal_client_for_profile("tigerwood697")
    tgt_client = get_modal_client_for_profile("hasinishrak2015")

    # Run remote export preparation function
    cmd = [
        "modal", "run", "--profile", "tigerwood697",
        "scripts/modal_export_bcs.py::prepare_bcs_export_remote",
        "--chunk-size-mb", str(chunk_size_mb),
    ]
    subprocess.run(cmd, check=True)

    src_vol = modal.Volume.from_name("sciencedb-perception-cache", client=src_client)
    tgt_vol = modal.Volume.from_name("mtl-data", client=tgt_client)

    # Read export manifest from source volume
    buf = io.BytesIO()
    src_vol.read_file_into_fileobj("export_bcs/bcs_export_manifest.json", buf)
    manifest = json.loads(buf.getvalue().decode("utf-8"))

    total_bytes = manifest["total_bytes"]
    print(f"[*] BCS Export Manifest loaded: {total_bytes / (1024**3):.2f} GB across {sum(len(f['chunks']) for f in manifest['files'].values())} chunks.")

    # State tracking file for resume
    state_file = temp_dir / "bcs_transfer_state.json"
    temp_dir.mkdir(parents=True, exist_ok=True)
    completed_chunks = set()
    if state_file.exists():
        try:
            completed_chunks = set(json.loads(state_file.read_text()).get("completed", []))
        except Exception:
            pass

    # Check which chunks already exist on target volume
    try:
        staging_files = {f.path for f in tgt_vol.listdir("bcs/staging")}
    except Exception:
        staging_files = set()

    # Transfer manifest CSVs first
    for m_name in manifest.get("manifests", {}):
        tgt_path = f"/bcs/staging/{m_name}"
        if tgt_path.lstrip("/") not in staging_files:
            print(f"[*] Uploading BCS manifest: {m_name}")
            m_buf = io.BytesIO()
            src_vol.read_file_into_fileobj(f"export_bcs/{m_name}", m_buf)
            m_buf.seek(0)
            with tgt_vol.batch_upload(force=True) as batch:
                batch.put_file(m_buf, tgt_path)

    # Transfer chunks sequentially
    all_chunks: List[Tuple[str, Dict[str, Any]]] = []
    for file_key, f_info in manifest["files"].items():
        for c in f_info["chunks"]:
            all_chunks.append((file_key, c))

    num_chunks = len(all_chunks)
    for idx, (file_key, c) in enumerate(all_chunks):
        c_name = c["chunk_name"]
        remote_src = f"export_bcs/{c_name}"
        remote_tgt = f"/bcs/staging/{c_name}"
        local_chunk_path = temp_dir / c_name

        if c_name in completed_chunks or remote_tgt.lstrip("/") in staging_files:
            print(f"    BCS Chunk {idx + 1}/{num_chunks}: {c_name} (already uploaded to target ✅)")
            completed_chunks.add(c_name)
            continue

        transfer_chunk_relay(
            source_vol=src_vol,
            source_path=remote_src,
            target_vol=tgt_vol,
            target_path=remote_tgt,
            local_temp_file=local_chunk_path,
            expected_size=c["size"],
            expected_sha=c["sha256"],
            task_name="BCS",
            chunk_idx=idx,
            num_chunks=num_chunks,
            buffer_mb=buffer_mb,
            keep_temp=keep_temp,
        )

        completed_chunks.add(c_name)
        with open(state_file, "w") as f:
            json.dump({"completed": list(completed_chunks)}, f)

    tgt_vol.commit()

    # Reassemble on target
    print("\n[*] Triggering target reassembly for BCS on hasinishrak2015...")
    cmd_reassemble = [
        "modal", "run", "--profile", "hasinishrak2015",
        "scripts/modal_stage_mtl_target.py::reassemble_bcs_remote",
        "--manifest", json.dumps(manifest),
    ]
    subprocess.run(cmd_reassemble, check=True)

    # Cleanup source chunks on tigerwood697
    print("[*] Cleaning up temporary export chunks on source profile tigerwood697...")
    subprocess.run(["modal", "run", "--profile", "tigerwood697", "scripts/modal_export_bcs.py::cleanup_bcs_export_remote"], check=False)

    if state_file.exists():
        state_file.unlink()

    return {"status": "SUCCESS", "bytes_transferred": total_bytes}


# ==============================================================================
# PIPELINE EXECUTION FOR BEHAVIOR
# ==============================================================================
def stage_behavior(
    dry_run: bool = False,
    chunk_size_mb: int = 1024,
    buffer_mb: int = 16,
    keep_temp: bool = False,
    temp_dir: Path = REPO_ROOT / "scratch" / "mtl_staging_temp",
) -> Dict[str, Any]:
    """Stages Behavior sequences from tigerwood693 to hasinishrak2015."""
    import modal

    print("\n" + "=" * 70)
    print("  TASK B: BEHAVIOR DATASET STAGING")
    print("  Source:      tigerwood693 (behavior-perception-cache)")
    print("  Destination: hasinishrak2015 (mtl-data/behavior)")
    print("=" * 70)

    if dry_run:
        print("[DRY-RUN] Would execute modal_export_behavior.py::prepare_behavior_export_remote on tigerwood693")
        print("[DRY-RUN] Packages 4,271 retained sequences (3,641 Train, 630 Val) into uncompressed tar archive (~900MB - 1.1GB)")
        est_chunks = math.ceil(1050000000 / (chunk_size_mb * 1024 * 1024))
        print(f"[DRY-RUN] Chunk size: {chunk_size_mb} MB (~{est_chunks} sequential chunks, relay buffer: {buffer_mb} MB)")
        print("[DRY-RUN] Target extraction would extract directly into /mtl-data/behavior/ and assert test data absent")
        return {"status": "DRY_RUN", "estimated_bytes": 1050000000}

    print("[*] Contacting source Modal profile tigerwood693 for Behavior export...")
    src_client = get_modal_client_for_profile("tigerwood693")
    tgt_client = get_modal_client_for_profile("hasinishrak2015")

    # Run remote export preparation function
    cmd = [
        "modal", "run", "--profile", "tigerwood693",
        "scripts/modal_export_behavior.py::prepare_behavior_export_remote",
        "--chunk-size-mb", str(chunk_size_mb),
    ]
    subprocess.run(cmd, check=True)

    src_vol = modal.Volume.from_name("behavior-perception-cache", client=src_client)
    tgt_vol = modal.Volume.from_name("mtl-data", client=tgt_client)

    buf = io.BytesIO()
    src_vol.read_file_into_fileobj("export_behavior/behavior_export_manifest.json", buf)
    manifest = json.loads(buf.getvalue().decode("utf-8"))

    total_bytes = manifest["total_bytes"]
    num_chunks = manifest["num_chunks"]
    print(f"[*] Behavior Export Manifest loaded: {total_bytes / (1024**2):.1f} MB across {num_chunks} chunks ({manifest['total_sequences']} sequences).")

    state_file = temp_dir / "behavior_transfer_state.json"
    temp_dir.mkdir(parents=True, exist_ok=True)
    completed_chunks = set()
    if state_file.exists():
        try:
            completed_chunks = set(json.loads(state_file.read_text()).get("completed", []))
        except Exception:
            pass

    try:
        staging_files = {f.path for f in tgt_vol.listdir("behavior/staging")}
    except Exception:
        staging_files = set()

    for idx, c in enumerate(manifest["chunks"]):
        c_name = c["chunk_name"]
        remote_src = f"export_behavior/{c_name}"
        remote_tgt = f"/behavior/staging/{c_name}"
        local_chunk_path = temp_dir / c_name

        if c_name in completed_chunks or remote_tgt.lstrip("/") in staging_files:
            print(f"    Behavior Chunk {idx + 1}/{num_chunks}: {c_name} (already uploaded to target ✅)")
            completed_chunks.add(c_name)
            continue

        transfer_chunk_relay(
            source_vol=src_vol,
            source_path=remote_src,
            target_vol=tgt_vol,
            target_path=remote_tgt,
            local_temp_file=local_chunk_path,
            expected_size=c["size"],
            expected_sha=c["sha256"],
            task_name="Behavior",
            chunk_idx=idx,
            num_chunks=num_chunks,
            buffer_mb=buffer_mb,
            keep_temp=keep_temp,
        )

        completed_chunks.add(c_name)
        with open(state_file, "w") as f:
            json.dump({"completed": list(completed_chunks)}, f)

    tgt_vol.commit()

    # Reassemble on target
    print("\n[*] Triggering target extraction for Behavior on hasinishrak2015...")
    cmd_reassemble = [
        "modal", "run", "--profile", "hasinishrak2015",
        "scripts/modal_stage_mtl_target.py::reassemble_behavior_remote",
        "--manifest", json.dumps(manifest),
    ]
    subprocess.run(cmd_reassemble, check=True)

    # Cleanup source chunks on tigerwood693
    print("[*] Cleaning up temporary export chunks on source profile tigerwood693...")
    subprocess.run(["modal", "run", "--profile", "tigerwood693", "scripts/modal_export_behavior.py::cleanup_behavior_export_remote"], check=False)

    if state_file.exists():
        state_file.unlink()

    return {"status": "SUCCESS", "bytes_transferred": total_bytes}


# ==============================================================================
# PIPELINE EXECUTION FOR RE-ID (Direct Cloud Download inside Modal)
# ==============================================================================
def stage_reid(dry_run: bool = False, workers: int = 16) -> Dict[str, Any]:
    """Stages Re-ID directly inside Modal profile hasinishrak2015 via Zenodo HTTP Range requests."""
    print("\n" + "=" * 70)
    print("  TASK C: RE-ID DATASET STAGING (DIRECT CLOUD DOWNLOAD)")
    print("  Source:      Zenodo Record 21605650 (parlor.zip, 9.60 GB)")
    print("  Destination: hasinishrak2015 (mtl-data/reid)")
    print("  PC Relay:    0 BYTES (Direct cloud-to-cloud)")
    print("=" * 70)

    if dry_run:
        print("[DRY-RUN] Would execute modal_stage_mtl_target.py::stage_reid_direct_remote on hasinishrak2015")
        print(f"[DRY-RUN] Direct {workers}-worker HTTP Range download of parlor.zip (9,598,064,961 bytes) to ephemeral /tmp/")
        print("[DRY-RUN] Ephemeral disk allocated: 20480 MiB (20 GiB)")
        print("[DRY-RUN] Persistent range chunk staging at /mtl-data/reid/.download_staging/ guarantees cross-invocation resume")
        print("[DRY-RUN] Computes exact 15,436 Train/Val records for the 41 representation learning cows (12,753 Train, 2,683 Val)")
        print("[DRY-RUN] Selectively extracts ONLY the 15,436 image files and 15,436 mask files (30,872 files) into /mtl-data/reid/")
        print("[DRY-RUN] Unlinks temporary parlor.zip and purges download_staging immediately; asserts 0 held-out cow overlap and zero barn/snapshots data")
        return {"status": "DRY_RUN", "direct_cloud_bytes": 9598064961}

    print(f"[*] Launching direct cloud download & extraction inside hasinishrak2015 with {workers} range workers...")
    cmd = [
        "modal", "run", "--profile", "hasinishrak2015",
        "scripts/modal_stage_mtl_target.py::stage_reid_direct_remote",
        "--threads", str(workers),
    ]
    subprocess.run(cmd, check=True)
    return {"status": "SUCCESS"}


# ==============================================================================
# VERIFICATION & AUDIT GATEWAY
# ==============================================================================
def verify_workspace(dry_run: bool = False) -> Dict[str, Any]:
    """Runs the comprehensive audit on /mtl-data in hasinishrak2015 and syncs manifest."""
    print("\n" + "=" * 70)
    print("  MTL WORKSPACE VERIFICATION & CERTIFICATION GATE")
    print("  Target Profile: hasinishrak2015")
    print("=" * 70)

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"

    if dry_run:
        print("[DRY-RUN] Would execute modal_stage_mtl_target.py::verify_mtl_workspace_remote on hasinishrak2015")
        print(f"[DRY-RUN] Git provenance SHA: {git_sha}")
        print("[DRY-RUN] Checks BCS: sequential 16 GB load of train_bcs_224.pt and val_bcs_224.pt (tensors, targets, raw_labels), test absent")
        print("[DRY-RUN] Checks Behavior: all 4,271 sequences (frame_00.jpg..frame_07.jpg, mask_00.png..mask_07.png, disjoint train/val), test absent")
        print("[DRY-RUN] Checks Re-ID: 15,436 pairs, 41 cows, 0 held-out overlap, no persistent barn/snapshots")
        print("[DRY-RUN] Checks mtl-checkpoints volume writeability")
        print("[DRY-RUN] Generates /mtl-data/staging_manifest.json with full provenance (git SHA, sources, hashes, byte sizes) and syncs to artifacts/mtl_staging/staging_manifest.json")
        return {"status": "DRY_RUN", "git_sha": git_sha}

    print(f"[*] Running remote audit on hasinishrak2015 (staging_git_sha={git_sha})...")
    cmd = [
        "modal", "run", "--profile", "hasinishrak2015",
        "scripts/modal_stage_mtl_target.py::verify_mtl_workspace_remote",
        "--staging-git-sha", git_sha,
    ]
    subprocess.run(cmd, check=True)

    # Sync staging_manifest.json locally
    print("[*] Syncing staging_manifest.json to local repository...")
    tgt_client = get_modal_client_for_profile("hasinishrak2015")
    import modal
    tgt_vol = modal.Volume.from_name("mtl-data", client=tgt_client)

    buf = io.BytesIO()
    tgt_vol.read_file_into_fileobj("staging_manifest.json", buf)
    manifest_dict = json.loads(buf.getvalue().decode("utf-8"))

    local_manifest_dir = REPO_ROOT / "artifacts" / "mtl_staging"
    local_manifest_dir.mkdir(parents=True, exist_ok=True)
    local_manifest_path = local_manifest_dir / "staging_manifest.json"
    with open(local_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    print(f"[+] Saved local staging manifest copy to {local_manifest_path}")
    return manifest_dict


# ==============================================================================
# CLI CONTROLLER ENTRYPOINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maximum-Speed Resumable MTL Data Staging Pipeline for hasinishrak2015.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["all", "bcs", "behavior", "reid"],
        default="all",
        help="Task to stage (default: all)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode with high-throughput defaults (16 workers, 1024 MB chunks, 64 MB buffer)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without downloading or transferring full datasets",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for automated runs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run standalone verification audit on existing target volume",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers/threads for HTTP Range downloads (default: 16 in --fast, 8 in standard)",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=None,
        help="Size of sequential transfer chunks in megabytes (default: 1024 in --fast, 512 in standard)",
    )
    parser.add_argument(
        "--buffer-mb",
        type=int,
        default=None,
        help="Streaming I/O buffer size in megabytes (default: 64 in --fast, 16 in standard)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary downloaded chunks on PC (for debugging)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=REPO_ROOT / "scratch" / "mtl_staging_temp",
        help="Local staging directory for transient chunk relay",
    )

    args = parser.parse_args()

    # Determine high-throughput vs standard defaults based on --fast flag
    if args.fast:
        workers = args.workers if args.workers is not None else 16
        chunk_size_mb = args.chunk_size_mb if args.chunk_size_mb is not None else 1024
        buffer_mb = args.buffer_mb if args.buffer_mb is not None else 64
    else:
        workers = args.workers if args.workers is not None else 8
        chunk_size_mb = args.chunk_size_mb if args.chunk_size_mb is not None else 512
        buffer_mb = args.buffer_mb if args.buffer_mb is not None else 16

    mode_str = "DRY-RUN (Simulated)" if args.dry_run else ("ACTIVE (Fast Transfer)" if args.fast else "ACTIVE (Standard Transfer)")

    print("\n" + "=" * 76)
    print("  🚀 MAXIMUM-SPEED RESUMABLE MTL DATA STAGING CONTROLLER")
    print(f"  Target Modal Profile: hasinishrak2015")
    print(f"  Execution Mode:      {mode_str}")
    print(f"  Fast Flag:           {'ENABLED' if args.fast else 'DISABLED'}")
    print(f"  Selected Task(s):    {args.task.upper() if not args.verify else 'VERIFICATION ONLY'}")
    print(f"  Chunk Size:          {chunk_size_mb} MB")
    print(f"  Streaming Buffer:    {buffer_mb} MB")
    print(f"  Parallel Workers:    {workers}")
    print("=" * 76)

    if args.verify:
        verify_workspace(dry_run=args.dry_run)
        return

    if not args.dry_run and not args.yes:
        confirm = input("\n[?] Ready to begin MTL data staging to hasinishrak2015? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("[-] Staging cancelled by user.")
            sys.exit(0)

    t0 = time.time()

    # Task A: BCS
    if args.task in ("all", "bcs"):
        stage_bcs(
            dry_run=args.dry_run,
            chunk_size_mb=chunk_size_mb,
            buffer_mb=buffer_mb,
            keep_temp=args.keep_temp,
            temp_dir=args.temp_dir,
        )

    # Task B: Behavior
    if args.task in ("all", "behavior"):
        stage_behavior(
            dry_run=args.dry_run,
            chunk_size_mb=chunk_size_mb,
            buffer_mb=buffer_mb,
            keep_temp=args.keep_temp,
            temp_dir=args.temp_dir,
        )

    # Task C: Re-ID
    if args.task in ("all", "reid"):
        stage_reid(
            dry_run=args.dry_run,
            workers=workers,
        )

    # Verification
    if args.task == "all":
        verify_workspace(dry_run=args.dry_run)

    total_time = time.time() - t0
    m, s = divmod(int(total_time), 60)
    print("\n" + "=" * 76)
    print(f"  ✅ STAGING PIPELINE COMPLETED IN {m}m {s}s")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

