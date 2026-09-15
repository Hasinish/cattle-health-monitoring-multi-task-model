# state.md — Current Workspace State

## Active Goals & Todo
- [x] Check workspace synchronization against remote repository (`origin/main`)
- [x] Protect `personal_info.md` via `.gitignore` to prevent leaking private info
- [x] Build automated dataset restoration script for research PC (`scripts/setup_research_pc.ps1`)
- [x] Download & extract Mendeley CattleLameness dataset (9,950 frames extracted, index CSV generated)
- [x] Download & extract 22.3 GB MmCows behavior dataset via KaggleHub to `datasets/kaggle_cache/`
- [ ] Complete OpenCows2020 dataset download via `download_opencows.py`
- [ ] Connect to Research PC (RTX 5090) at 1:00 AM and run dataset restoration
- [ ] Launch PCGrad multi-task training run
- [ ] Prepare P3 draft submission by September 26

## Last Session (Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef)
- Analyzed P2 Hostile Review (`P2_hostile_review.md`) and diagnosed root causes of negative transfer, threshold leakage, and unit mismatches.
- Created and committed automated restoration script `scripts/setup_research_pc.ps1`.
- Created and committed `scripts/pcgrad.py` implementing Projecting Conflicting Gradients (NeurIPS 2020) to resolve task interference in the shared backbone.
- Successfully restored CattleLameness (9,950 frames indexed) and downloaded full MmCows (22.3 GB) on local HDD.

## Current Blockers & Notes
- OpenCows2020 download currently running in user terminal.
- Lab slot starts at 1:00 AM (Wed-Thu 1 AM to 1 PM). Ready to deploy scripts to Research PC.


