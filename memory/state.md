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
- [ ] Place Dryad BCS archive (`Total_sorted_DGE_images.zip` from doi:10.5061/dryad.tqjq2bw4s) and run `preprocess_bcs.py`
- [ ] Connect to Research PC (RTX 5090) and run master restoration (`git pull; python scripts/download_all.py --all`)
- [ ] Launch PCGrad multi-task training run
- [ ] Prepare P3 draft submission by September 26

## Last Session (Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef)
- Analyzed P2 Hostile Review (`P2_hostile_review.md`) and diagnosed root causes of negative transfer, threshold leakage, and unit mismatches.
- Created and committed automated restoration script `scripts/setup_research_pc.ps1` and `scripts/download_all.py`.
- Created and committed `scripts/pcgrad.py` implementing Projecting Conflicting Gradients (NeurIPS 2020).
- Successfully restored CattleLameness (9,950 frames), MmCows behavior (213,686 frames), and OpenCows2020 (4,736 frames).
- Cleaned up 36+ GB of raw behavior videos/archives.
- Built automated browser launcher and download watcher for Dryad BCS in `download_all.py` (pushed to `origin/main`).
- Packaged complete custom Antigravity environment into dedicated GitHub repo `https://github.com/Hasinish/custom-antigravity.git` (20 files, 1-click installer).

## Current Blockers & Notes
- Dryad BCS pending stream download via browser from https://datadryad.org/downloads/file_stream/2391628.
- Research PC ready to bootstrap using `Hasinish/custom-antigravity` and `cattle-health-monitoring-multi-task-model`.


