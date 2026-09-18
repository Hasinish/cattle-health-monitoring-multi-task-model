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
| **2026-09-18** | [Primary Task Dataset Split Integrity & Leakage Audit](2026-09-18_dataset_split_integrity_audit.md) | Audited ScienceDB, MmCows, and OpenCows2020 splits. Identified OpenCows2020 train/val sequence leakage from random frame shuffling. Verified 7 active MmCows behavior classes and discrete '2'-'6' Dryad classes. Codified 6-step experiment sequence (PCGrad deferred to Step 6). | `2026-09-18_dataset_split_integrity_audit.md`, `memory/state.md` |
| **2026-09-17** | [Lameness Dataset Forensic Audit & Leakage Resolution](2026-09-17_lameness_investigation.md) | Discovered severe source leakage and dataset-quality problems in CattleLameness. A provisional 42-group manifest was created but later marked historical after manual review found unreliable grouping assumptions and CGI/Blender clips. Russello 2026 was identified as the strongest public lameness candidate, but lameness was subsequently removed from the primary Phase 3 multi-task experiment. | `cattle_lameness_manifest.csv`, `cattle_lameness_audit_report.md`, `candidate_lameness_datasets_audit.md` |

