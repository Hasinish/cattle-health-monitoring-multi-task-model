# Research Log — 2026-09-24: MTL Workspace Staging & Zero-Copy Certification on hasinishrak2015

## 1. Executive Summary
On 2026-09-24, executed zero-GPU forensic certification of the unified Multi-Task Learning (MTL) workspace on Modal profile `hasinishrak2015` via `scripts/modal_stage_mtl_target.py` (App `ap-0QtIRqRuI3mDlksS8TnvLn`). Verified 100% data integrity, tensor loadability, exact hashes, and the new **Zero-Copy SideView Train/Val Access Policy** across all three canonical tasks (BCS, Behavior, Re-ID) without routing bytes through local Wi-Fi. Generated and validated official `/mtl-data/staging_manifest.json` and synced locally to `artifacts/mtl_staging/staging_manifest.json`.

---

## 2. Forensic Audit Receipts

### Task A: ScienceDB BCS (Monolithic Tensors)
- **Train Tensor**: `train_bcs_224.pt` (6.42 GB, 34,369 samples, shape `[34369, 4, 224, 224]`, `torch.uint8`). SHA-256: `b45b3b14e871381b88de59ba542a72ad67287bb92278b16de7ce4ac69ed2818a`.
- **Val Tensor**: `val_bcs_224.pt` (1.46 GB, 7,817 samples, shape `[7817, 4, 224, 224]`, `torch.uint8`). SHA-256: `3be6daace45b051f9bf51a594f7fd02290fe16c9221cfdda7abb704a6a522da0`.
- **Sequential In-Memory Verification**: Both tensors sequentially loaded via `torch.load(..., weights_only=False)` and memory reclaimed cleanly.
- **Leakage Protection**: Verified `test_bcs_224.pt` and `test_perception.csv` are strictly ABSENT from `/mtl-data/bcs/`.

### Task B: Behavior Perception Sequences
- **Source Export**: Packaged on `tigerwood693` using 64-worker parallel NVMe pre-staging in 72.4s (59.0 seq/s) and local tar in 3.62s. Archive SHA-256: `c50bc36c853c8dc722dbc118c2bf08646eb28a1bad1c8ec37719a11d6a20ecc3`.
- **Cloud Transfer**: Downloaded 652.3 MB chunk cloud-to-cloud in 7.3s (90.0 MB/s) and extracted into `/mtl-data/behavior/` in 124.4s.
- **Exhaustive Sequence Check**: Verified ALL 4,271 sequences (3,641 Train, 630 Val):
  - 34,168 frames (`frame_00.jpg` ... `frame_07.jpg`), all non-empty.
  - 34,168 masks (`mask_00.png` ... `mask_07.png`), all non-empty.
  - 4,271 `perception_metadata.json` files, all non-empty.
- **Disjointness**: 0 overlapping sequence IDs between Train and Val (`train_val_disjoint: true`).
- **Leakage Protection**: Verified `production_test/`, `retained_test.csv`, and `failed_test.csv` are strictly ABSENT from `/mtl-data/behavior/`.

### Task C: SideViewCows2026 Re-ID (Zero-Copy Access Policy)
- **Zero-Copy Architecture**: Avoided duplicating 30,872 files into `/mtl-data/reid`. MTL training mounts `sideview-data` volume directly at `/sideview/sideviewcows2026/parlor`.
- **Protocol Alignment**: Evaluated against canonical `protocol_cross_setting.csv` (Protocol A) and `protocol_closed_set.csv` (Protocol D):
  - **Training Cows**: Strictly 41 representation-learning identities.
  - **Evaluation Cows**: Strictly 69 held-out identities (`held_out_cow_overlap: 0`).
  - **Train Pairs**: Exactly 12,753 pairs (12,753 RGB images + 12,753 binary masks).
  - **Val Pairs**: Exactly 2,683 pairs (2,683 RGB images + 2,683 binary masks).
  - **Total Files Verified on Volume**: Exactly 30,872 files verified existing and non-zero bytes on `/sideview/sideviewcows2026/parlor/`.
- **Leakage Protection**: Zero access to any of the 69 evaluation cows during train/val verification.

---

## 3. Staging Manifest Provenance
- Manifest saved to `/mtl-data/staging_manifest.json` on persistent volume and committed.
- Local replica written to `artifacts/mtl_staging/staging_manifest.json`.
- Schema validation against `artifacts/mtl_staging/staging_manifest_schema.json`: **100% PASS**.
- Status: **`CERTIFIED_READY_FOR_MTL`**.
