# state.md — Current Workspace State

## Active Goals & Todo
- [x] Check workspace synchronization against remote repository (`origin/main`)
- [x] Protect `personal_info.md` via `.gitignore` to prevent leaking private info
- [x] Build automated dataset restoration script for research PC (`scripts/setup_research_pc.ps1`)
- [x] Download & extract Mendeley CattleLameness dataset (9,950 frames extracted, index CSV generated)
- [x] Download & extract 213,686 MmCows behavior images via Hugging Face (`cropped_bboxes.zip`) directly to `datasets/behavior/`
- [x] Complete OpenCows2020 dataset download & indexing (4,736 images indexed across 46 classes via Kagglehub)
- [x] Download & extract ScienceDB Cattle BCS dataset (53,566 RGB images, index CSV generated)
- [x] Purge 36+ GB raw behavior videos, zip archives, and cache to reclaim local disk space
- [x] Automate Dryad BCS download launcher & watcher in `scripts/download_all.py` (commit `f543f16`)
- [x] Export comprehensive Antigravity customization replication repository to `D:\custom-antigravity` and push to remote
- [x] Install local laptop dataset dependencies (opencv-python, pandas, tqdm, kagglehub, huggingface_hub, scikit-learn)
- [x] Complete forensic CattleLameness dataset audit & leakage investigation (`docs/audits/cattle_lameness_audit_report.md`)
- [x] Build leakage-safe CattleLameness manifest (`datasets/lameness/cattle_lameness_manifest.csv`) & 5-fold evaluation architecture
- [x] Establish research log hub (`docs/research_log/`) and automated persistence rules (`.agents/rules/research_logging.md`, `AGENTS.md`)
- [x] Finalize P3 task scope: focus on 3 core RGB tasks (BCS, Behavior, Cow ID); lameness removed from primary P3 MTL model
- [ ] Place Dryad BCS archive (Secondary / Comparison, `Total_sorted_DGE_images.zip` in Downloads) and run `preprocess_bcs.py`
- [ ] STEP 1: Audit and verify dataset splits for ScienceDB, MmCows, and OpenCows2020 (cow/session/source leakage)
- [ ] STEP 2: Run clean Single-Task baselines on verified splits
- [ ] STEP 3: Run clean 3-task Hard-Sharing MTL
- [ ] STEP 4: Run Partial-Sharing MTL
- [ ] STEP 5: Compare Single-Task vs Hard-Sharing vs Partial-Sharing and measure negative transfer
- [ ] STEP 6: Optionally test PCGrad / GradNorm if it answers a useful question
- [ ] Prepare P3 draft submission by September 26

## Active P3 Task Scope
- **Current Status**: Over 270,000 images are downloaded/indexed and ready for split validation. Dataset split integrity has NOT yet been fully verified.
- **1. Body Condition Scoring (BCS)**:
  - **Primary**: ScienceDB (53,566 RGB images across 5 classes: 3.25, 3.5, 3.75, 4.0, 4.25 from 10,898 cows)
  - **Secondary / Comparison**: Dryad (5,923 Depth Grayscale Edge images across folder classes '2', '3', '4', '5', '6')
- **2. Behavior Recognition**:
  - **Primary**: MmCows (213,686 bounding-box crops across 7 active behavior classes: 1 to 7 from 16 cows)
- **3. Individual Cow Identification**:
  - **Primary**: OpenCows2020 (4,736 images across 46 cow classes, closed-set identification)
- **Lameness Status**:
  - Excluded from primary Phase 3 multi-task experiment
  - CattleLameness retained for historical Phase 2 documentation and audit/leakage evidence only
  - Russello 2026 retained as a possible future pose/keypoint-based extension

## Last Session (Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879)
- Conducted forensic audit of Mendeley CattleLameness dataset (50 clips, 9,950 frames) and exposed critical train/test leakage (`N (9).mp4` vs `N (3).mp4`).
- Generated 42-group leak-safe 5-fold cross-validation manifest (`datasets/lameness/cattle_lameness_manifest.csv`).
- Audited 4 candidate repos: recommended Russello 2026 (98 cows, 272 trajectories), rejected Wu NWAFU, whsu2s, and Duan 2025.
- Built centralized research log repository (`docs/research_log/`) and created workspace rules (`.agents/rules/research_logging.md`, `AGENTS.md`).
- Restored CattleLameness (9,950 frames), MmCows behavior (213,686 frames), OpenCows2020 (4,736 frames), and ScienceDB BCS (53,566 frames).
- Cleaned up 36+ GB of raw behavior videos/archives.
- Built automated browser launcher and download watcher for Dryad BCS in `download_all.py`.
- Packaged complete custom Antigravity environment into dedicated private GitHub repo `https://github.com/Hasinish/custom-antigravity.git`.
- Corrected experimental order: split audit first, followed by single-task baselines, hard sharing, partial sharing, transfer evaluation, and optional PCGrad/GradNorm last.

## Current Blockers & Notes
- Over 270,000 images are downloaded/indexed and ready for split validation. Indexing does not prove splits are leak-free.
- Next immediate step is to audit and resolve split logic across ScienceDB, MmCows, and OpenCows2020.
- OpenCows2020 train/val split currently uses random shuffling on sequential video frames, risking high near-duplicate leakage between train and val.
- MmCows contains only 16 cows total, leaving validation (2 cows) and test (3 cows) vulnerable to subject-specific evaluation bias.
- PCGrad/GradNorm are deferred to Step 6 and must not be treated as the current training plan.


