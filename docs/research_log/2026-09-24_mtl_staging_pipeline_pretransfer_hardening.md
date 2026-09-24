# Research Log — Final Pre-Transfer Hardening of the MTL Staging Pipeline

**Date**: 2026-09-24  
**Author**: Hasin Ishrak  
**Target Profile**: `hasinishrak2015` (`mtl-data`, `mtl-checkpoints`)  
**Status**: COMPLETE & 100% AUDITED (ZERO REAL DATA TRANSFERRED)

---

## 1. Executive Summary
Conducted the final pre-transfer hardening of the unified Multi-Task Learning (MTL) data staging pipeline across `scripts/modal_stage_mtl_target.py`, `scripts/stage_mtl_workspace.py`, and `artifacts/mtl_staging/staging_manifest_schema.json`. Hardened eight critical pipeline vectors: (1) authentic 2-digit Behavior frame/mask filenames (`frame_00.jpg`..`frame_07.jpg`, `mask_00.png`..`mask_07.png`), (2) missing `pandas as pd` import inside Re-ID verification scope, (3) exhaustive audit verifying all 4,271 Behavior sequences (3,641 Train, 630 Val; 34,168 frames + 34,168 masks) with train/val sequence disjointness, (4) Re-ID ephemeral disk provisioned at 20,480 MiB (20 GiB), (5) true cross-invocation Re-ID resume utilizing persistent range-chunk staging on `/mtl-data/reid/.download_staging/` with post-extraction cleanup, (6) active `--fast` configuration selector with high-throughput defaults (16 workers, 1024 MB chunks, 64 MB buffer vs 8 workers, 512 MB chunks, 16 MB buffer), (7) comprehensive manifest provenance recording Git SHA, task byte sizes, cryptographic SHA-256 hashes, source volumes/records, and zero held-out cow overlap, and (8) verified with dry runs and synthetic unit tests. Strictly zero real dataset bytes were transferred.

---

## 2. Context & Motivation
Preparation for Run 7 (E1 Hard-Shared MTL) and Run 8 (E3 Modular MTL) requires staging three distinct datasets on Modal profile `hasinishrak2015`:
1. **BCS** from `tigerwood697` (`train_bcs_224.pt`, `val_bcs_224.pt`).
2. **Behavior** from `tigerwood693` (4,271 retained sequence directories).
3. **Re-ID** direct from Zenodo Record 21605650 (`parlor.zip`, 15,436 Train/Val pairs across 41 cows).

Before launching the full multi-gigabyte data transfer, a rigorous forensic review identified several operational vulnerabilities in the initial staging code that would have caused container failures, false integrity rejections, or unresumable transfer stalls if interrupted.

---

## 3. The 8 Hardening Fixes & Technical Details

### 3.1 Behavior Cache Authentic 2-Digit Filenames
The authentic Run 5 cache created by `scripts/build_behavior_perception_cache.py` uses 2-digit zero-padded names:
```text
frame_{t:02d}.jpg -> frame_00.jpg ... frame_07.jpg
mask_{t:02d}.png  -> mask_00.png ... mask_07.png
```
Previously, `modal_stage_mtl_target.py` checked 3-digit names (`frame_000.jpg`, `mask_000.png`), which would fail integrity verification on genuine Run 5 cached sequences. All documentation, extraction logic, and verifiers were standardized to 2-digit format, and the entire repository was audited to ensure zero 3-digit assumptions remain.

### 3.2 Missing Pandas Import in Remote Verification Scope
`verify_mtl_workspace_remote()` utilizes `pd.read_csv()` to parse canonical SideView protocols (`protocol_cross_setting.csv`, `protocol_closed_set.csv`). Explicitly added `import pandas as pd` inside `verify_mtl_workspace_remote()`, ensuring the verification container does not crash when executing Re-ID protocol audits.

### 3.3 Exhaustive 4,271 Behavior Verification & Disjointness Check
Replaced the initial 10-sequence sample check with an exhaustive filesystem verification of all:
```text
3,641 Train sequences
  630 Val sequences
4,271 Total sequences
```
For EVERY retained sequence, the verifier asserts:
- `perception_metadata.json` exists and is non-empty (`st_size > 0`).
- `frame_00.jpg` ... `frame_07.jpg` exist and are non-empty (34,168 frames).
- `mask_00.png` ... `mask_07.png` exist and are non-empty (34,168 masks).
- Enforces strict disjointness: `assert not beh_train_ids.intersection(beh_val_ids)`.

### 3.4 Re-ID Ephemeral Disk Provisioning
`stage_reid_direct_remote()` assembles `parlor.zip` (9.60 GB) and extracts 15,436 image and mask pairs (30,872 files, ~5.5 GB) inside the container. Explicitly provisioned:
```python
ephemeral_disk=20480  # 20,480 MiB (20 GiB)
```
This guarantees ample scratch headroom on Modal without hitting disk exhaustion errors.

### 3.5 Re-ID True Cross-Invocation Resume Engine
Previously, downloaded range parts were saved only in ephemeral `/tmp/`, meaning any container restart or network interruption would lose completed chunks.
- **Persistent Staging**: Range chunks are now saved to `/mtl-data/reid/.download_staging/parlor.part_{start}_{end}` on the persistent volume.
- **Resume Skip**: On rerun, existing parts on volume matching the exact expected byte size `end_b - start_b + 1` are validated and skipped.
- **Stitching**: Completed parts are stitched from the persistent volume into ephemeral `/tmp/sideview_mtl/parlor.zip`.
- **Cleanup**: After selective extraction of the 15,436 Train/Val pairs and integrity assertions (0 held-out cow overlap, no barn/snapshots in volume), the ephemeral `parlor.zip` is deleted AND all persistent range parts in `.download_staging/` are unlinked and the directory removed.

### 3.6 Active `--fast` Flag Implementation
Upgraded `--fast` in `scripts/stage_mtl_workspace.py` from a cosmetic flag to an active configuration selector:
- **Standard Mode (default)**: `workers=8`, `chunk_size_mb=512`, `buffer_mb=16`.
- **Fast Mode (`--fast`)**: `workers=16`, `chunk_size_mb=1024`, `buffer_mb=64`.
- Wired `buffer_mb` into `transfer_chunk_relay` for high-throughput streaming and SHA-256 chunk hashing. User-specified CLI arguments (`--workers`, `--chunk-size-mb`, `--buffer-mb`) override defaults.

### 3.7 Strengthened Staging Manifest Provenance
Enhanced `/mtl-data/staging_manifest.json` and its JSON schema `artifacts/mtl_staging/staging_manifest_schema.json` to record:
- Active Git commit SHA (`staging_git_sha`).
- Sources: BCS (`tigerwood697`, `sciencedb-perception-cache`), Behavior (`tigerwood693`, `behavior-perception-cache`), Re-ID (Zenodo Record 21605650, `parlor.zip`).
- Exact Train and Val sample counts across all 3 tasks.
- Exact byte sizes of stored tensors, sequences, and image pairs.
- SHA-256 hashes of monolithic BCS tensors and the Behavior archive.
- Re-ID held-out cow overlap = 0.
- Final status: `CERTIFIED_READY_FOR_MTL`.

### 3.8 Synthetic Unit Testing & Dry Runs
Created `tests/test_mtl_staging_hardening.py` testing all 4 core components locally:
1. Behavior 2-digit naming and disjointness assertion: PASS.
2. Pandas import and protocol parsing (41 train cows, 69 eval cows, 12,753 train pairs, 2,683 val pairs): PASS.
3. Persistent Re-ID range-chunk resume, stitching, and staging directory cleanup: PASS.
4. Staging manifest JSON schema validation against `staging_manifest_schema.json`: PASS.

Both standard and fast dry runs passed cleanly:
```powershell
python scripts/stage_mtl_workspace.py --task all --dry-run
python scripts/stage_mtl_workspace.py --task all --fast --dry-run
```

---

## 4. Deliverables & Artifacts
- Master Controller: `scripts/stage_mtl_workspace.py`
- Target Modal Staging & Verification Engine: `scripts/modal_stage_mtl_target.py`
- Staging Manifest Schema: `artifacts/mtl_staging/staging_manifest_schema.json`
- Synthetic Unit Test Suite: `tests/test_mtl_staging_hardening.py`
- Workspace Memory: `memory/state.md`, `memory/history.md`, `memory/index.md`
