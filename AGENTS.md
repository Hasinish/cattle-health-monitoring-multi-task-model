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

4. **Git Hygiene & Dual-Device Sync**:
   - Ensure all documentation, logs, manifests, and scripts are committed to Git so the local laptop and the BRACU Lab Research PC (RTX 5090) stay in 100% lockstep.
   - Never commit `scratch/` or raw GB-scale video/image folders.
