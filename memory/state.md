# state.md — Current Workspace State

## Active Goals & Todo
- [x] Check workspace synchronization against remote repository (`origin/main`)
- [x] Protect `personal_info.md` via `.gitignore` to prevent leaking private info
- [x] Build automated dataset restoration script for research PC (`scripts/setup_research_pc.ps1`)
- [x] Download & extract Mendeley CattleLameness dataset (9,950 frames extracted, index CSV generated)
- [x] Download & extract 213,686 MmCows behavior images via Hugging Face (`cropped_bboxes.zip`) directly to `datasets/behavior/`
- [x] Complete OpenCows2020 dataset download & indexing (4,736 images indexed across 46 classes via Kagglehub)
- [x] Purge 36+ GB raw behavior videos, zip archives, and cache to reclaim local disk space
- [x] Automate Dryad BCS download launcher & watcher in `scripts/download_all.py` (commit `f543f16`)
- [x] Export comprehensive Antigravity customization replication repository to `D:\custom-antigravity` and push to remote (`https://github.com/Hasinish/custom-antigravity.git`)
- [x] Install local laptop dataset dependencies (opencv-python, pandas, tqdm, kagglehub, huggingface_hub, scikit-learn)
- [x] Fix extraction syntax bug in `scripts/download_mmcows.py`
- [x] Complete forensic CattleLameness dataset audit & leakage investigation (`docs/audits/cattle_lameness_audit_report.md`)
- [x] Build leakage-safe CattleLameness manifest (`datasets/lameness/cattle_lameness_manifest.csv`) & 5-fold evaluation architecture (`docs/audits/cattle_lameness_grouping_report.md`)
- [x] Complete forensic audit of 4 candidate lameness datasets (`docs/audits/candidate_lameness_datasets_audit.md`)
- [x] Establish research log hub (`docs/research_log/`) and automated persistence rules (`.agents/rules/research_logging.md`, `AGENTS.md`)
- [ ] Choose lameness implementation path: Russello 2026 trajectory encoder vs CattleLameness 5-fold CV
- [ ] Place Dryad BCS archive (`Total_sorted_DGE_images.zip` from doi:10.5061/dryad.tqjq2bw4s) and run `preprocess_bcs.py`
- [ ] Restore datasets locally on laptop (`python scripts/download_all.py`)
- [ ] Connect to Research PC (RTX 5090) and run master restoration (`git pull; python scripts/download_all.py --all`)
- [ ] Launch PCGrad multi-task training run
- [ ] Prepare P3 draft submission by September 26

## Last Session (Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879)
- Conducted forensic audit of Mendeley CattleLameness dataset (50 clips, 9,950 frames) and exposed critical train/test leakage (`N (9).mp4` vs `N (3).mp4`).
- Generated 42-group leak-safe 5-fold cross-validation manifest (`datasets/lameness/cattle_lameness_manifest.csv`).
- Audited 4 candidate repos: recommended Russello 2026 (98 cows, 272 trajectories), rejected Wu NWAFU (static pose), whsu2s (missing data), and Duan 2025 (closed).
- Built centralized research log repository (`docs/research_log/`) and created workspace rules (`.agents/rules/research_logging.md`, `AGENTS.md`) to enforce perpetual tracking.
- Analyzed P2 Hostile Review (`P2_hostile_review.md`) and diagnosed root causes of negative transfer, threshold leakage, and unit mismatches.
- Created and committed automated restoration script `scripts/setup_research_pc.ps1` and `scripts/download_all.py`.
- Created and committed `scripts/pcgrad.py` implementing Projecting Conflicting Gradients (NeurIPS 2020).
- Successfully restored CattleLameness (9,950 frames), MmCows behavior (213,686 frames), and OpenCows2020 (4,736 frames).
- Cleaned up 36+ GB of raw behavior videos/archives.
- Built automated browser launcher and download watcher for Dryad BCS in `download_all.py` (pushed to `origin/main`).
- Packaged complete custom Antigravity environment into dedicated private GitHub repo `https://github.com/Hasinish/custom-antigravity.git` (20 files, 1-click installer).
- Packaged all 4 Modal compute accounts into `custom-antigravity/credentials/modal.toml` and updated `setup.ps1` for 1-click credential deployment across devices.
- Created serverless dataset volume sync (`modal_sync_datasets.py`) and multi-task PCGrad training pipeline (`modal_train_pcgrad.py`) for Modal cloud execution.
- Restored dynamic real-time Modal billing monitor daemon (`scripts/billing_monitor.py`) and auto-generated `BILLING.md` dashboard tracking all 4 accounts (pushed to `custom-antigravity` commit `ff5a941`).

## Current Blockers & Notes
- Dryad BCS pending stream download via browser from https://datadryad.org/downloads/file_stream/2391628.
- Research PC ready to bootstrap using `Hasinish/custom-antigravity` and `cattle-health-monitoring-multi-task-model`.


