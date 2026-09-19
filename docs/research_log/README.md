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
| :--- | :--- | :--- | :--- |
| **2026-09-20** | [Local Dataset Inventory Audit & Canonical Dataset Registry](2026-09-20_local_dataset_inventory_audit.md) | Executed read-only forensic physical inventory and generated canonical `datasets/dataset_registry.csv` (13 datasets, 28 fields). Corrected false ScienceDB presence claim (raw files absent locally, index stale). Recorded MultiCamCows2024 as BLOCKED upstream (connection reset verified locally & on Modal cloud). Cataloged Dryad BCS (5,940 TIFFs across classes 2-7, discrepancy flagged), MmCows (213,686 behavior crops verified out of 427,390 total JPGs, raw videos purged), and OpenCows2020 (4,736 images across 46 classes verified). | `datasets/dataset_registry.csv`, `scripts/build_dataset_registry.py`, `2026-09-20_local_dataset_inventory_audit.md`, `memory/state.md` |
| **2026-09-19** | [Adoption of Phase 3 Canonical Roadmap](2026-09-19_phase3_canonical_roadmap.md) | Locked the 13-step Phase 3 Canonical Roadmap. Replaced OpenCows2020 with MultiCamCows2024 as primary Re-ID; designated Ruchay 2026, CBVD-5, and SideViewCows2026 as external validation; locked Step 1 (Data Registry & Clean Splits) as immediate next priority. | `phase3_canonical_roadmap.md`, `docs/phase3_canonical_roadmap.md`, `2026-09-19_phase3_canonical_roadmap.md` |
| **2026-09-18** | [Proposed Direction: Segmentation-Guided, Anatomy-Aware & Viewpoint-Aware Cattle Health Monitoring](2026-09-18_cattle_centered_anatomy_aware_direction.md) | Proposed new candidate research direction: isolating cows via segmentation, pretraining anatomy/pose embeddings, and conditioning representations on camera viewpoint to eliminate background shortcuts and build a cattle-specific shared representation for MTL. Status: Feasibility audit required. | `2026-09-18_cattle_centered_anatomy_aware_direction.md`, `memory/state.md` |
| **2026-09-18** | [Primary Task Dataset Split Integrity & Leakage Audit](2026-09-18_dataset_split_integrity_audit.md) | Audited ScienceDB, MmCows, and OpenCows2020 splits. Identified OpenCows2020 train/val sequence leakage from random frame shuffling. Verified 7 active MmCows behavior classes and discrete '2'-'6' Dryad classes. Codified 6-step experiment sequence (PCGrad deferred to Step 6). | `2026-09-18_dataset_split_integrity_audit.md`, `memory/state.md` |
| **2026-09-17** | [Lameness Dataset Forensic Audit & Leakage Resolution](2026-09-17_lameness_investigation.md) | Discovered severe source leakage and dataset-quality problems in CattleLameness. A provisional 42-group manifest was created but later marked historical after manual review found unreliable grouping assumptions and CGI/Blender clips. Russello 2026 was identified as the strongest public lameness candidate, but lameness was subsequently removed from the primary Phase 3 multi-task experiment. | `cattle_lameness_manifest.csv`, `cattle_lameness_audit_report.md`, `candidate_lameness_datasets_audit.md` |

