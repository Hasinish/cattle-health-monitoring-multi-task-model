# Workspace Rules — Cattle Health Monitoring Multi-Task Model

## Research Logging & Audit Preservation
1. **Mandatory Research Log Reading on Startup**:
   - At the beginning of EVERY conversation or before proposing ANY dataset changes, architecture designs, or experiment setups, the AI assistant **MUST read `docs/research_log/README.md` and the latest entry in `docs/research_log/`**.
   - NEVER propose rejected datasets (e.g., Wu NWAFU, whsu2s, Duan 2025) or superseded tasks (e.g., lameness in primary Phase 3 MTL) documented in the research logs.

2. **Mandatory Research Logging**:
   - Every forensic audit, leakage analysis, dataset investigation, ablation study, or model experiment must be documented in `docs/research_log/YYYY-MM-DD_<topic>.md`.
   - Update `docs/research_log/README.md` index table whenever a new research log is created.
   - Comprehensive audit reports belong in `docs/audits/`.

2. **Dataset Manifests**:
   - All dataset manifests, de-leakage grouping files, and CV fold definitions must be saved as deterministic CSVs in `datasets/<task>/` and tracked by Git (unignored in `.gitignore`).
   - Generation scripts belong in `scripts/`.

3. **Memory & State Synchronization**:
   - Synchronize `memory/state.md` immediately upon completing tasks or reaching milestones.
   - Prepend new conversation summaries to `memory/history.md`.
   - Keep `memory/index.md` up to date with new directory trees and file purposes.

4. **Git Hygiene & Environment Sync**:
   - Ensure all documentation, logs, manifests, and scripts are committed to Git so the local laptop and remote cloud environments (rotating Modal profiles) stay in 100% lockstep.
   - Never commit `scratch/` or raw GB-scale video/image folders.

5. **Antigravity Customization & Rules Sync**:
   - Whenever workspace rules, `.agents/rules/`, or global agent instructions are modified, immediately mirror them to `D:\custom-antigravity`, commit, and push to GitHub.

6. **Compute Execution Policy & Cloud / Modal Usage**:
   - **Local Machine (GTX 1050 Ti)**: Strictly reserved for smoke tests, path checks, tensor/loss/metric unit checks, checkpoint save/resume verification, and tiny 1–2 epoch validation runs. NEVER run full training or heavy preprocessing locally.
   - **Cloud Execution (Rotating Modal Profiles)**: All heavy preprocessing, full training runs, ablations, and final experimental suites are dispatched to Modal/cloud.
   - **Rotating Modal Accounts**: Modal accounts/profiles rotate depending on available credits and resource quotas. Never hard-code one Modal account or profile as universally required.
   - **Disqualified Hardware**: The BRACU Lab Research PC (RTX 5090) is permanently DISQUALIFIED and unusable due to an unrecoverable locked/bloated OS/driver environment. NEVER propose or schedule training runs on the RTX 5090.
   - **Cost Optimization**: Strictly use minimal container resources: `cpu=1.0, memory=2048` (or 1024) for downloads, data transfers, and extraction. NEVER attach GPUs or allocate excessive CPUs/RAM for pure network I/O.
   - Always implement periodic `volume.commit()` checkpoints (e.g. every 60s) for long-running downloads so partial progress is saved and resumable if interrupted.
   - Default to lowest-cost GPU tier (e.g. `gpu="T4"`) for all smoke tests and feasibility runs.
