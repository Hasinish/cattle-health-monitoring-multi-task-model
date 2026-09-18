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
- **[2026-09-18] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Audited split integrity across ScienceDB, MmCows, and OpenCows2020; corrected 7-class MmCows and '2'-'6' Dryad label definitions; codified 6-step experiment sequence.
- **[2026-09-17] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Audited CattleLameness & 4 candidate datasets. Uncovered train/test leak, built 42-group leak-safe manifest, and established research log hub + agent persistence rules.
- **[2026-09-16] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Started session to guide and execute dataset downloads on local laptop environment.
- **[2026-09-16] Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef**: Restored Lameness (9,950), Behavior (213,686), and ID (4,736) datasets. Purged 36+ GB raw behavior videos. Integrated Dryad browser stream automation. Pushed commits to GitHub. Initialized IDE customization transfer document.
- **[2026-09-15] Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef**: Verified workspace is fully up to date with remote GitHub repo `Hasinish/cattle-health-monitoring-multi-task-model` (HEAD at 6c3437f). Flagged untracked PDF reports. Initialized memory files.
