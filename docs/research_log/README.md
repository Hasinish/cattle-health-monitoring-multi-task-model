# Research & Investigation Log Hub

Welcome to the centralized Research Log repository for the Cattle Health Monitoring Multi-Task Model thesis project.

## Purpose
Every forensic dataset audit, data leakage discovery, hyperparameter experiment, architectural pivot, and ablation study is permanently documented in this directory. This ensures:
1. Complete reproducibility across the local laptop and the BRACU Lab Research PC (RTX 5090).
2. Bulletproof defense during thesis defense and hostile reviews (P2/P3).
3. Zero lost findings, decisions, or hard-won insights across AI conversation boundaries.

---

## Log Entry Naming Convention
All log entries are saved as markdown documents following the strict pattern:
```
docs/research_log/YYYY-MM-DD_<topic_in_snake_case>.md
```
*Example:* `docs/research_log/2026-09-17_lameness_investigation.md`

---

## Standard Entry Structure
Every log entry MUST contain the following sections:
1. **Executive Summary**: 2-3 sentences explaining the core problem and key outcome.
2. **Context & Motivation**: Why the investigation or experiment was conducted.
3. **Forensic Findings & Data**: Exact numbers, file paths, class distributions, leakage risks, and evidence.
4. **Architectural Decisions & Action Plan**: Concrete, definitive choices made with trade-off analysis.
5. **Artifacts & File Registry**: Paths to all generated code, manifests, and detailed reports.
6. **Next Steps**: Exact immediate checklist.

---

## Historical Log Index
| Date | Title / Topic | Primary Outcome / Decision | Key Artifacts |
| **2026-09-20** | [OpenCows2020 Legacy Re-ID Protocol & Leakage Audit](2026-09-20_opencows2020_legacy_reid_audit.md) | Audited OpenCows2020 (4,736 images, 46 cows). Disproved sequence recoverability from provenance (frame numbers are unordered crops; consecutive MAE is identical to random pairs). Exposed legacy `preprocess_id.py` random shuffle flaw (1,023 adjacent-frame leaks, 1,760 near-duplicate leaks, 3 exact-duplicate leaks). Preserved official benchmark `identification-test` (496 images) 100% untouched. Rebuilt training-side train/val via contiguous frame blocks + duplicate harmonization (train: 3,586, val: 654). Verified 0 duplicate leakage. Maintained legacy-only status. | `datasets/id/opencow2020/manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `split_report.md`, `datasets/id/id_index.csv`, `build_opencows_splits.py`, `verify_opencows_splits.py` |
| **2026-09-20** | [Dryad BCS Dataset Discrepancy Audit](2026-09-20_dryad_bcs_discrepancy_audit.md) | Resolved discrepancy between older ~5,923 expectation and 5,940 physical TIFFs. Proved older count omitted class folder '7' (17 images from Cow_52). Proved Class 7 is 100% authentic Criollo beef data on 1–9 Wagner scale. Reconstructed 54 biological cows across 148 session folders. Verified 0 cross-cow duplicate leakage and 100% image readability. Exported master manifest, cow census, and updated legacy `bcs_index.csv`. | `datasets/bcs/dryad/manifest.csv`, `cow_audit.csv`, `audit_report.md`, `datasets/bcs/bcs_index.csv`, `scripts/build_dryad_manifest.py` |
| **2026-09-20** | [MmCows Grouped Evaluation Protocol & Leakage Audit](2026-09-20_mmcows_grouped_protocol_and_leakage_audit.md) | Audited MmCows behavior dataset (213,686 crops, 7 classes). Proved biological cow ID validity (16 Holstein cows, 4 CCTV cams, 21.0h, NeurIPS 2024 Spotlight). Protected 87.15% multi-camera synchronized events and 15s contiguous time blocks via cow-disjoint grouping. Identified Class 5 (Licking) rarity with 5 zero-licking cows. Generated master manifest, provenance audit, canonical split (11/2/3 cows), and balanced 4-Fold GroupKFold suite evaluating 100% of cows with 0 leakage. | `datasets/behavior/mmcows/manifest.csv`, `provenance_audit.csv`, `train.csv`, `val.csv`, `test.csv`, `folds/fold_[0-3].csv`, `split_report.md`, `build_mmcows_splits.py`, `verify_mmcows_splits.py` |
| **2026-09-20** | [ScienceDB Roadmap Correction](2026-09-20_sciencedb_roadmap_correction.md) | Corrected roadmap documentation to classify ScienceDB as passage-disjoint / sequence-safe rather than cow-disjoint due to publisher not releasing biological cow IDs. Maintained 5,662 passage clusters and 0 leakage split. | `docs/phase3_canonical_roadmap.md`, `phase3_canonical_roadmap.md` |
| **2026-09-20** | [ScienceDB Cattle BCS Identity Audit & Leakage-Free Split](2026-09-20_sciencedb_identity_audit_and_leakage_free_split.md) | Audited ScienceDB identity parser; proved that claimed '10,898 cows' was a mixture of 5,403 burst passages and 5,495 individual video frames with 94.64% stereo sequence leakage in legacy split. Found 19 exact duplicate pairs (38 files). Built leak-free 5,662-passage stratified split across 3 coarse farms. Exported train (37,126), val (8,099), test (8,341) and identity audit manifest. | `datasets/bcs/sciencedb/train.csv`, `val.csv`, `test.csv`, `identity_audit.csv`, `split_report.md`, `build_sciencedb_splits.py` |
| **2026-09-20** | [Local Dataset Inventory Audit & Canonical Dataset Registry](2026-09-20_local_dataset_inventory_audit.md) | Executed read-only forensic physical inventory and generated canonical `datasets/dataset_registry.csv` (13 datasets, 28 fields). Corrected false ScienceDB presence claim (raw files absent locally, index stale). Recorded MultiCamCows2024 as BLOCKED upstream (connection reset verified locally & on Modal cloud). Cataloged Dryad BCS (5,940 TIFFs across classes 2-7, discrepancy flagged), MmCows (213,686 behavior crops verified out of 427,390 total JPGs, raw videos purged), and OpenCows2020 (4,736 images across 46 classes verified). | `datasets/dataset_registry.csv`, `scripts/build_dataset_registry.py`, `2026-09-20_local_dataset_inventory_audit.md`, `memory/state.md` |
| **2026-09-19** | [Adoption of Phase 3 Canonical Roadmap](2026-09-19_phase3_canonical_roadmap.md) | Locked the 13-step Phase 3 Canonical Roadmap. Replaced OpenCows2020 with MultiCamCows2024 as primary Re-ID; designated Ruchay 2026, CBVD-5, and SideViewCows2026 as external validation; locked Step 1 (Data Registry & Clean Splits) as immediate next priority. | `phase3_canonical_roadmap.md`, `docs/phase3_canonical_roadmap.md`, `2026-09-19_phase3_canonical_roadmap.md` |
| **2026-09-18** | [Proposed Direction: Segmentation-Guided, Anatomy-Aware & Viewpoint-Aware Cattle Health Monitoring](2026-09-18_cattle_centered_anatomy_aware_direction.md) | Proposed new candidate research direction: isolating cows via segmentation, pretraining anatomy/pose embeddings, and conditioning representations on camera viewpoint to eliminate background shortcuts and build a cattle-specific shared representation for MTL. Status: Feasibility audit required. | `2026-09-18_cattle_centered_anatomy_aware_direction.md`, `memory/state.md` |
| **2026-09-18** | [Primary Task Dataset Split Integrity & Leakage Audit](2026-09-18_dataset_split_integrity_audit.md) | Audited ScienceDB, MmCows, and OpenCows2020 splits. Identified OpenCows2020 train/val sequence leakage from random frame shuffling. Verified 7 active MmCows behavior classes and discrete '2'-'6' Dryad classes. Codified 6-step experiment sequence (PCGrad deferred to Step 6). | `2026-09-18_dataset_split_integrity_audit.md`, `memory/state.md` |
| **2026-09-17** | [Lameness Dataset Forensic Audit & Leakage Resolution](2026-09-17_lameness_investigation.md) | Discovered severe source leakage and dataset-quality problems in CattleLameness. A provisional 42-group manifest was created but later marked historical after manual review found unreliable grouping assumptions and CGI/Blender clips. Russello 2026 was identified as the strongest public lameness candidate, but lameness was subsequently removed from the primary Phase 3 multi-task experiment. | `cattle_lameness_manifest.csv`, `cattle_lameness_audit_report.md`, `candidate_lameness_datasets_audit.md` |

