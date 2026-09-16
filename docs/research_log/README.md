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
| **2026-09-17** | [Lameness Dataset Forensic Audit & Leakage Resolution](file:///d:/cattle-health-monitoring-multi-task-model/docs/research_log/2026-09-17_lameness_investigation.md) | Discovered train/test clip leakage in CattleLameness (N(9) vs N(3)). Built 42-group leak-safe 5-fold CV manifest. Audited 4 candidates; selected Russello 2026 (98 cows, 272 trajectories) as primary trajectory candidate. | `cattle_lameness_manifest.csv`, `cattle_lameness_audit_report.md`, `candidate_lameness_datasets_audit.md` |
