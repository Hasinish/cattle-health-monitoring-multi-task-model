# Phase 3 writing status

Reviewed source snapshot: `4ff3de8e667514b40875ba46bf7cb1c890d02219` on 2026-09-25.

Status: All 8 focused deadline runs (Runs 1–8) are 100% complete and held-out evaluated. Experimental results, checkpoints, metrics, and outcome-level negative-transfer analysis are certified across Runs 1–8 (single-task baselines Runs 1–3, perception-enhanced Runs 4–6, E1 hard-shared MTL Run 7, and E3 modular task-private adapter MTL Run 8). Manuscript insertion of the completed MTL methodology (Chapter 4), multi-task results & integrated discussion (Chapter 5), final RQ answers/conclusion (Chapter 6), post-MTL Abstract, and evidence appendices remains pending.

| Section/file | Status | Remaining boundary |
|---|---|---|
| Front matter | Provisional | Administrative names, signatures, approval, acknowledgments, ethics/AI disclosure |
| Abstract | Interim pre-MTL version | Pending final post-MTL rewrite incorporating certified E1/E3 results and calibrated negative-transfer findings |
| Chapter 1: Introduction | Stable before MTL | Final integrated contribution wording and RQ alignment only |
| Chapter 2: Literature Review | Stable before MTL, normal final polishing allowed | No speculative reference expansion |
| Chapter 3: Requirements, Impacts and Constraints | Stable before MTL | Deployment-specific costs, approvals, and user evidence remain unavailable |
| Chapter 4: Proposed Methodology | Runs 1–6 stable; MTL architectures executed and certified | Insert executed E1 hard-shared and E3 modular task-private residual adapter methodology |
| Chapter 5: Result Analysis | Runs 1–6 stable; E1/E3 held-out evaluation complete | Insert certified E0/E1/E3 results, outcome-level negative transfer analysis, and integrated discussion |
| Chapter 6: Conclusion | Structured interim synthesis | Insert final research-question answers, contribution list, limitations, conclusion, and future work |
| Evidence appendices | Updated through Runs 1–6 | Integrate certified Run 7/8 checkpoints, evaluation logs, and artifact manifests |
| Evidence map | Updated through E34 | Update with certified Run 7 and Run 8 evidence items while preserving claim boundaries |
| Rubric checklist | Updated | Complete final MTL-dependent CO5–CO7 and final CO14 deliverables |
| LaTeX build | PASS: 69-page PDF, no fatal error, undefined citation/reference, or overfull box | See `BUILD_REPORT.md`; only inherited/nonfatal warnings remain |

## Completed & Certified Experimental State (Runs 1–8)

- **All 8 Focused Deadline Runs Complete**: All 8 focused deadline runs (Runs 1–8) are complete and held-out evaluated with test-set isolation preserved.
- **Run 7 (E1 Hard-Shared MTL Control)**: Architecture (11.18M shared trunk, 747k task heads), training (Epoch 3 checkpoint, val objective 0.40036), and held-out evaluation are complete across all three tasks (BCS MAE 0.1788, Behavior Acc 85.00% / Loss 0.6226, Re-ID Barn Rank-1 57.38% / mAP 30.37%).
- **Run 8 (E3 Modular MTL / Task-Private Adapters)**: Architecture (11.18M shared trunk, 395k private residual adapters, 747k task heads; 12.32M total trainable params, +3.32% over E1), training (Epoch 2 checkpoint, val objective 0.38888), and held-out evaluation are complete across all three tasks (BCS MAE 0.1916 on matched test population, Behavior Acc 85.77% / Loss 0.4193, Re-ID Barn Rank-1 49.08% / mAP 28.04%).
- **Negative-Transfer Outcome Analysis Complete**: Run 7 demonstrated held-out performance degradation under hard sharing, consistent with negative transfer at the outcome level (underlying optimization mechanism not directly measured). Run 8 task-private residual adapters showed selective, metric-dependent benefit (improving test cross-entropy loss by -0.2033 and CVB barn CCTV accuracy by +3.55 pp, while balanced accuracy dropped by 0.60 pp and Re-ID degraded) rather than universal negative-transfer mitigation.
- **Exact E1 vs E3 Matched Comparison Available**: Deterministic head-to-head comparison metrics, confusion matrices, and per-class breakdowns are fully verified and synced in `artifacts/mtl_e3_evaluation/mtl_e3_test_evaluation_metrics.json` and research logs.
- **Canonical Roadmap Preservation**: Deferred canonical experiments such as E2 (partial sharing), E4/PCGrad, E5/GradNorm, larger ablation ladders, and Step 12 pretraining remain deferred roadmap work, NOT completed or cancelled.

## Remaining Manuscript Insertion Tasks

- **Chapter 4 (Proposed Methodology)**: Insert executed E1 hard-shared architecture and E3 modular task-private residual bottleneck adapter implementation details, parameter allocations (12.32M params; +3.32% private capacity), loss formulations, and training schedules.
- **Chapter 5 (Result Analysis)**: Insert final matched E0/E1/E3 comparative results, metric breakdown tables, per-task findings, and calibrated discussion on selective negative-transfer mitigation and cross-setting retrieval trade-offs.
- **Chapter 6 (Conclusion)**: Insert final answers to the 3 Research Questions, consolidated contribution list, explicit limitations, and future work directions.
- **Abstract**: Rewrite with final post-MTL scope, empirical multi-task findings, and certified performance metrics.
- **Evidence Appendices, Evidence Map & Rubric Checklist**: Integrate certified Run 7 and Run 8 checkpoints, held-out evaluation payloads, and artifact manifests.

## Remaining Non-MTL Administration

Confirm author order, submission metadata, committee details, signatures, acknowledgments, dataset permissions, ethics and AI-assistance disclosure, team contribution records, and any separate IEEE-format submission requirement.
