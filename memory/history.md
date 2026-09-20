# Session Summary — 2026-09-20 (Phase 3 Step 1 ScienceDB Burst Split Repair Script)

- Engineered `scripts/repair_sciencedb_splits.py` to repair the ScienceDB Cattle BCS dataset split by clustering overlapping video burst passages into leak-free connected burst groups.
- Designed multi-signal conservative burst evidence:
  1. Exact SHA-256 content byte hashing.
  2. Multi-Index Hashing (MIH) on 64-bit dHash and aHash (Hamming distance <= 2) within the same farm source.
  3. In-memory normalized pixel Mean Absolute Error (MAE) on 64x64 grayscale with threshold <= 5.0 (out of 255).
  4. Disjoint Set Union (Union-Find) connected component graph clustering.
- Built rock-solid Windows PowerShell compatible `tqdm` progress UI (`ascii=True`, fixed width, multi-stage, eta, rate, zero flickering, zero spammy prints).
- Implemented multi-check verification suite (53,566 image count, group disjointness, 0 exact cross-duplicates, 0 confirmed cross-burst links).
- Executed manual run in terminal: 5,653 repaired burst groups formed from 5,662 initial passages (9 confirmed video burst overlaps merged into connected components); 70/15/15 stratified split generated (train: 37,045, val: 8,481, test: 8,040 across 3,958 train, 850 val, 845 test groups).
- Verified 0 exact duplicates, 0 cross-burst leakage; confirmed GS_1818 and GS_1823 are 100% unified in val.
- Promoted staging deliverables to canonical `datasets/bcs/sciencedb/` and updated `sciencedb_bcs_index.csv`.
- Documented in `docs/research_log/2026-09-20_sciencedb_burst_group_split_repair.md` and updated research log README.md index.

# Session Summary — 2026-09-20 (Phase 3 Duplicate Audit & Cold Turkey System Purge)

- Conducted exhaustive exact (SHA-256) and perceptual near-duplicate (Multi-Index Hashed dHash/aHash, $d \le 6$) leakage audit across all split-bearing Phase 3 datasets:
  - **ScienceDB Cattle BCS (53,566 images)**: 0 exact cross-duplicates; 88,944 near-duplicate cross-partition suspects flagged. Discovered critical vulnerability: `GS_1818` (val) and `GS_1823` (train) are consecutive frames of the same video passage shifted by 1 frame (MAE 0.25). Split **NEEDS CORRECTION** before Step 4 training by clustering passages into connected temporal burst blocks.
  - **MmCows Behavior (213,686 images)**: 0 exact cross-duplicates. 100% cow disjointness verified across canonical split (11 train, 2 val, 3 test cows) and all 4 GroupKFold splits (`overlap: set()`). Split is **CLEAN & LOCKED**.
  - **OpenCows2020 (4,736 images)**: 0 exact cross-duplicates; 1,239 near-duplicates (100% within same cow identity due to author randomization). Designated **LEGACY BASELINE ONLY**.
- Documented findings in `docs/research_log/2026-09-20_phase3_duplicate_nearduplicate_audit.md` and updated `docs/research_log/README.md`.
- Performed forensic audit and complete eradication of Cold Turkey Blocker from host system:
  - Verified background services, active processes, and main installation directories were already deleted.
  - Found and purged lingering browser extensions in Google Chrome (`pganeibhckoanndahmnfggfoeofncnii`) and Microsoft Edge (`jfphahkinplobmabmgjmjgflbhjjddeb`).
  - Purged user registry key `HKCU:\Software\Cold Turkey`.
  - Executed elevated registry cleanup removing orphaned `HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{6498E673-B9C2-4544-A722-1E854B5B573E}_is1` and `HKLM:\Software\Cold Turkey`. Verified 100% removal across all system hives.

# Session Summary — 2026-09-20 (Modal Billing Monitor & BILLING.md Integration)

- Added real-time multi-account Modal billing monitor and dashboard generator identical to `modal-qwen`.
- Created `scripts/billing_monitor.py` supporting profile discovery from `~/.modal.toml`, active profile detection, thread-safe JSON query of metered/billed cost across all 6 accounts, remaining credit calculation, and L40S/H100/L4 GPU runtime estimations.
- Created `scripts/modal_billing.py` for one-shot terminal summaries and root wrapper `billing_monitor.py` for direct command-line execution (`python billing_monitor.py` and `python billing_monitor.py --loop`).
- Successfully generated `BILLING.md` in workspace root and mirrored it to `D:\custom-antigravity\BILLING.md`.

# Session Summary — 2026-09-20 (External Benchmarks Hydration: SideViewCows, BECA, CBVD-5)

- Built high-speed multi-threaded resumable downloader `scripts/fast_download_sideviewcows.py` for SideViewCows2026 (Zenodo record 21605650, 80,260 images + masks, 23.33 GB) with browser header spoofing and custom `\r` `CleanProgressBar`. Fully downloaded and extracted `snapshots.zip` (607 imgs), `parlor.zip` (108,786 files), and `barn.zip` (15.05 GB).
- Created `scripts/fast_download_beca.py` for BECA dataset (Figshare file 63928608, Article 32070171: BECA-D + BECA-L, 18.91 GB, 29,061 images) with multi-threaded HTTP Range and automatic S3 redirect resolution. Fully downloaded and extracted `BECA.zip` (68,350 files).
- Created `scripts/download_cbvd5.py` for CBVD-5 behavior dataset from Kaggle via `kagglehub` API with automatic syncing into canonical `datasets/behavior/external/cbvd5/`. Fully downloaded and synced 887 MP4 videos, 206,100 mini frames, 5,322 annotated label frames (10.84 GB).
- Verified and updated canonical `datasets/dataset_registry.csv` via `scripts/build_dataset_registry.py` to transition SideViewCows2026, BECA-D, BECA-L, and CBVD-5 to `AVAILABLE`.
- Documented complete investigation in `docs/research_log/2026-09-20_external_benchmarks_hydration_audit.md` and updated `docs/research_log/README.md`.
- Updated hardware configuration in `personal_info.md` to document local GTX 1050 Ti laptop and remote BRACU lab RTX 5090 research rig. Mirrored changes to `D:\custom-antigravity`.

# Session Summary — 2026-09-20 (OpenCows2020 Legacy Re-ID Protocol Rebuild)

- Audited OpenCows2020 (4,736 images across 46 identities) and proved that sequence / tracklet structure cannot be recovered from provenance (frame numbers are unordered crops; consecutive MAE is identical to random pairs within cow).
- Quantified extensive random within-identity mixing in legacy `context/preprocess_id.py` random split: 1,023 frame-index adjacent pairs ($|f_1 - f_2| = 1$), 2,942 near frame-index pairs ($|f_1 - f_2| \le 5$), 1,760 visually similar pairs (MAE < 15), and 3 exact-duplicate pairs (identical SHA256) crossed train and val.
- Preserved the official benchmark `identification-test` set (496 images, 46 cows) 100% untouched.
- Rebuilt training-side train/val via contiguous frame-index heuristic + duplicate harmonization: Train=3,586, Val=654, Test=496 (total 4,736 images across 46 cows).
- Eliminated exact-duplicate leakage (0 duplicate hash overlap across any split) and reduced frame-index adjacency crossings from 1,023 down to 48 (single boundary transition per cow).
- Explicitly documented that true tracklet/temporal leakage cannot be verified because provenance is unavailable.
- Created `datasets/id/opencow2020/manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `split_report.md`, and updated `datasets/id/id_index.csv`.
- Created `scripts/build_opencows_splits.py` and standalone verification `scripts/verify_opencows_splits.py` (100% pass).
- Updated `datasets/dataset_registry.csv` and documented findings in `docs/research_log/2026-09-20_opencows2020_legacy_reid_audit.md`.
- Maintained OpenCows2020 strictly as a **LEGACY BASELINE ONLY**; MultiCamCows2024 remains intended primary Re-ID.

# Session Summary — 2026-09-20

- Completed Dryad Cattle BCS Discrepancy Audit (`docs/research_log/2026-09-20_dryad_bcs_discrepancy_audit.md`).
- Fully reconciled the discrepancy between the older ~5,923 expectation and 5,940 physical files: proved that legacy `preprocess_bcs.py` hardcoded classes 2–6 (5,923 imgs) to fit a 5-class head and silently dropped class folder '7' (17 imgs from `Cow_52`).
- Proved Class 7 is 100% authentic Criollo beef cattle data on the 1–9 Wagner scale (Winkler & Boucheron, NMSU, Dryad DOI: 10.5061/dryad.tqjq2bw4s); `Cow_52` was also recorded at BCS 6 (`Cow_52_29`).
- Reconstructed 54 biological cows (`Cow_1`..`Cow_55`, `Cow_39` absent) across 148 session folders; exposed cross-session animal identity leakage if splitting by folder name instead of biological cow.
- Verified 100% image readability (all 5,940 are valid 224x224 RGB DGE TIFFs) and 0 cross-cow duplicate leakage.
- Exported master manifest `datasets/bcs/dryad/manifest.csv` (5,940 rows), `cow_audit.csv` (54 cows), `audit_report.md`, and updated legacy `datasets/bcs/bcs_index.csv` (5,940 rows).
- Completed MmCows Behavior Grouped Evaluation Protocol and Leakage Audit (`docs/research_log/2026-09-20_mmcows_grouped_protocol_and_leakage_audit.md`).
- Verified that 213,686 image crops stem from 16 genuine biological Holstein dairy cows recorded across 4 synchronized CCTV cameras over 21.0 hours (NeurIPS 2024 Spotlight, Purdue NEIS Lab).
- Protected 87.15% synchronized multi-camera events (64,830 / 74,388 events) and 15s contiguous periodic time blocks via strict cow-disjoint grouping.
- Identified Class 5 (Licking) rarity (2,009 crops, 41.7:1 imbalance) with 5 cows having zero licking crops; constructed balanced 4-Fold GroupKFold cross-validation suite evaluating 100% of cows with guaranteed positive support for all 7 classes.
- Exported master manifest `datasets/behavior/mmcows/manifest.csv` (213,686 rows), `provenance_audit.csv` (16 cows), canonical splits (`train.csv`, `val.csv`, `test.csv`), 4-fold suite (`folds/fold_[0-3].csv`), and `split_report.md`.
- Implemented and passed all leak-free assertions in `scripts/build_mmcows_splits.py` and `scripts/verify_mmcows_splits.py` (0 identity overlap, 0 multi-cam leak, 100/100 sampled path resolution).
- Conducted read-only physical filesystem inventory audit on GTX 1050 Ti machine (`docs/research_log/2026-09-20_local_dataset_inventory_audit.md`).
- Clarified physical presence vs scientific roles: ScienceDB BCS marked `METADATA_ONLY` locally (raw files absent; stale 53,566-row index); MultiCamCows2024 marked `BLOCKED` upstream (connection reset verified locally and on Modal cloud); Dryad BCS verified locally (5,940 TIFFs across classes 2-7, discrepancy flagged for future audit); MmCows verified locally (213,686 active behavior crops valid and indexed; 427,390 total JPGs include auxiliary folders; raw videos purged); OpenCows2020 verified locally (4,736 images across 46 classes).
- Created deterministic canonical dataset registry `datasets/dataset_registry.csv` via `scripts/build_dataset_registry.py` (13 datasets, 28 columns, machine-awareness fields `local_status_1050ti` and `local_verified_date`).

# Session Summary — 2026-09-19

- Adopted and locked the 13-step Phase 3 Canonical Roadmap (`phase3_canonical_roadmap.md` and `docs/phase3_canonical_roadmap.md`).
- Established primary tasks & datasets: BCS (Primary: ScienceDB, Ext: Ruchay 2026, Dryad), Behavior (Primary: MmCows, Ext: CBVD-5), Cow ID / Re-ID (New Primary: MultiCamCows2024, Ext: SideViewCows2026, Legacy: OpenCows2020).
- Locked immediate priority to STEP 1: Data Registry (`datasets/dataset_registry.csv`) and Clean Splits.
- Documented roadmap in research log (`docs/research_log/2026-09-19_phase3_canonical_roadmap.md`) and synchronized workspace memory.

# Session Summary — 2026-09-18

- Audited primary Phase 3 dataset split protocols across ScienceDB BCS (53,566 images), MmCows behavior (213,686 crops), and OpenCows2020 cow ID (4,736 images).
- Corrected status claims: verified that indexing does not equate to methodologically clean splits; verified MmCows has 7 active classes (not 5); verified Dryad BCS labels are folder classes '2'-'6' (not continuous 1-5).
- Re-aligned experiment order: locked 6-step sequence (Split Audit -> Single-Task Baselines -> Hard-Sharing MTL -> Partial-Sharing MTL -> Transfer Analysis -> optional PCGrad/GradNorm).
- Updated memory/state.md to reflect split validation status and prevent premature model training.

# Session Summary — 2026-09-17

- Conducted exhaustive forensic audit of Mendeley CattleLameness dataset (50 clips, 9,950 frames); exposed critical train/test leakage (`N (9).mp4` in Train vs `N (3).mp4` in Test from identical YouTube source).
- Built 42-animal/source grouping manifest (`cattle_lameness_manifest.csv`) with leak-free 5-fold StratifiedGroupKFold cross-validation (marked provisional/historical).
- Forensically audited 4 candidate lameness datasets: evaluated Russello 2026 (98 cows, 272 trajectories, pose-only), rejected Wu NWAFU (static pose only), whsu2s (missing repo data), and Duan 2025 (closed/private data).
- Established centralized Research Logging Hub under `docs/research_log/` and created `.agents/rules/research_logging.md` and workspace root `AGENTS.md` to permanently enforce documentation of all findings, manifests, and audits.
- Codified canonical Phase 3 scope: Body Condition Scoring (Primary: ScienceDB, Secondary: Dryad), Behavior Recognition (MmCows), and Individual Cow Identification (OpenCows2020); Lameness excluded from primary MTL training.

# Session Summary — 2026-09-16

- Fully restored and indexed 3 of 4 core datasets: Mendeley CattleLameness (9,950 frames), MmCows Behavior (213,686 images), and OpenCows2020 Cow ID (4,736 images across 46 classes).
- Built high-speed Kagglehub restore for OpenCows2020 and automated disk cleanup removing 36+ GB of raw video/zip clutter.
- Corrected Dryad BCS dataset DOI reference (doi:10.5061/dryad.tqjq2bw4s) and built automated browser launcher & real-time download watcher in `download_all.py` (commit `f543f16`).
- Packaged complete custom Antigravity environment into dedicated private GitHub repo `https://github.com/Hasinish/custom-antigravity.git` (commit `7391dc1`, `20d72e8`).
- Packaged all 4 Modal compute accounts into `custom-antigravity/credentials/modal.toml` with 1-click deployment in `setup.ps1`.
- Resurrected the lost custom Modal billing script from historical transcripts into `custom-antigravity/scripts/modal_billing.py` (commit `5fc1c76`), verified real-time balances ($104.97 total remaining credit).

# Session Summary — 2026-09-15

- Workspace synchronization check against GitHub remote origin/main. Local commit 6c3437f matches remote exactly. Untracked compiled PDFs detected. Workspace memory initialized.

# history.md — Conversation Log History

<!-- IMPORTANT FOR AGENTS: Always prepend new conversation log entries to the top of this list (most recent first). Do not append to the bottom. -->

- **[2026-09-20] Convo 27375138-e032-457f-a2a6-753e72f4a342**: Completed forensic local dataset inventory on GTX 1050 Ti machine; distinguished physical local availability from canonical scientific roles; marked MultiCamCows2024 download as BLOCKED (upstream issue); created canonical dataset registry `datasets/dataset_registry.csv` (13 datasets, 28 columns).
- **[2026-09-19] Convo 27375138-e032-457f-a2a6-753e72f4a342**: Synchronized workspace and custom-antigravity from origin/main (+287k lines, 12 config files). Deployed Antigravity customizations via setup.ps1. Added Modal accounts hasinishrak2015 and dryousufmozumder ($30 grants each), upgraded tigerwood697 ($30 grant), reaching 6 accounts and $107.01 total credit (~55 hrs L40S). Synced canonical roadmap clarifications to docs/ and pushed to GitHub. Clean start locked for STEP 1 execution.
- **[2026-09-19] Convo fa09b269-a73b-49f3-aecc-c870aef77dac**: Git synchronization status check across workspace and custom-antigravity repository. Verified clean working trees and all commits pushed to origin/main.
- **[2026-09-19] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Formally adopted Phase 3 Canonical Roadmap (13-step pipeline). Replaced OpenCows2020 with MultiCamCows2024 as primary Re-ID; designated Ruchay 2026, CBVD-5, and SideViewCows2026 as external validation; locked Step 1 (Data Registry & Clean Splits) as immediate next priority.
- **[2026-09-18] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Audited split integrity across ScienceDB, MmCows, and OpenCows2020; corrected 7-class MmCows and '2'-'6' Dryad label definitions; codified 6-step experiment sequence.
- **[2026-09-17] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Audited CattleLameness & 4 candidate datasets. Uncovered train/test leak, built 42-group leak-safe manifest, and established research log hub + agent persistence rules.
- **[2026-09-16] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Started session to guide and execute dataset downloads on local laptop environment.
- **[2026-09-16] Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef**: Restored Lameness (9,950), Behavior (213,686), and ID (4,736) datasets. Purged 36+ GB raw behavior videos. Integrated Dryad browser stream automation. Pushed commits to GitHub. Initialized IDE customization transfer document.
- **[2026-09-15] Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef**: Verified workspace is fully up to date with remote GitHub repo `Hasinish/cattle-health-monitoring-multi-task-model` (HEAD at 6c3437f). Flagged untracked PDF reports. Initialized memory files.
