# state.md — Current Workspace State

## Active Goals & Todo
- [x] Check workspace synchronization against remote repository (`origin/main`)
- [x] Protect `personal_info.md` via `.gitignore` to prevent leaking private info
- [x] Build automated dataset restoration script for research PC (`scripts/setup_research_pc.ps1`)
- [x] Implement PyTorch PCGrad gradient surgery optimizer (`scripts/pcgrad.py`) for P3 novelty
- [ ] Connect to Research PC (RTX 5090) at 1:00 AM and run dataset restoration
- [ ] Launch PCGrad multi-task training run
- [ ] Prepare P3 draft submission by September 26

## Last Session (Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef)
- Analyzed P2 Hostile Review (`P2_hostile_review.md`) and diagnosed root causes of negative transfer, threshold leakage, and unit mismatches.
- Created and committed automated restoration script `scripts/setup_research_pc.ps1`.
- Created and committed `scripts/pcgrad.py` implementing Projecting Conflicting Gradients (NeurIPS 2020) to resolve task interference in the shared backbone.

## Current Blockers & Notes
- Lab slot starts at 1:00 AM (Wed-Thu 1 AM to 1 PM). Restoring datasets on research PC is the immediate next step.


