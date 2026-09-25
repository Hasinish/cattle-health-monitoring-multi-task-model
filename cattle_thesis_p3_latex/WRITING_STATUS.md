# Phase 3 writing status

Reviewed source snapshot: `4af058bdee84bfcc879a366dcf7d7b5b441af7f1` on 2026-09-26.

Status: All 8 focused deadline runs + one final additional E4 PCGrad control are 100% complete and held-out evaluated. Experimental model training/evaluation is finished. Experimental results, checkpoints, metrics, and outcome-level negative-transfer analysis are certified across all completed configurations (single-task baselines Runs 1–3, perception-enhanced Runs 4–6, E1 hard-shared MTL Run 7, E3 modular task-private adapter MTL Run 8, and E4 PCGrad optimization control). GradNorm E5 and partial sharing E2 remain deferred. Chapter 4 E4 PCGrad methodology integration is complete. Chapter 5 final E0/E1/E3/E4 result integration complete (includes complete single-task and multi-task evaluations, Tables 5.8–5.12, PCGrad conflict diagnostics, and calibrated negative-transfer synthesis). Chapter 6 final RQ answers/contributions/limitations/conclusion complete. Abstract rewrite is complete from final experimental evidence. Appendices, evidence map, and rubric checklist need final E4 integration.

| Section/file | Status | Remaining boundary |
|---|---|---|
| Front matter | Provisional | Administrative names, signatures, approval, acknowledgments, ethics/AI disclosure |
| Abstract | Complete | Self-contained 331-word synthesis integrated from final single-task and multi-task experimental evidence |
| Chapter 1: Introduction | Stable before MTL | Final integrated contribution wording and RQ alignment only |
| Chapter 2: Literature Review | Stable before MTL, normal final polishing allowed | No speculative reference expansion |
| Chapter 3: Requirements, Impacts and Constraints | Stable before MTL | Deployment-specific costs, approvals, and user evidence remain unavailable |
| Chapter 4: Proposed Methodology | Complete | Chapter 4 E4 PCGrad methodology integration complete |
| Chapter 5: Result Analysis | Complete | Chapter 5 final E0/E1/E3/E4 result integration complete |
| Chapter 6: Conclusion | Complete | Chapter 6 final RQ answers/contributions/limitations/conclusion complete |
| Evidence appendices | Updated through Runs 1–6 | Needs final E4 integration (certified E1/E3/E4 checkpoints, evaluation logs, and artifact manifests) |
| Evidence map | Updated through E34 | Needs final E4 integration while preserving claim boundaries |
| Rubric checklist | Updated | Needs final E4 integration (complete final MTL-dependent CO5–CO7 and final CO14 deliverables) |
| LaTeX build | PASS: 87-page PDF, no fatal error, undefined citation/reference, or overfull box | See `BUILD_REPORT.md`; only inherited/nonfatal warnings remain |

## Completed & Certified Experimental State (8 Focused Deadline Runs + One Final Additional E4 PCGrad Control)

- **Experimental Model Training & Evaluation Finished**: All experimental model training and evaluation is officially complete and closed. No further training runs or ablations will be launched.
- **8 Focused Deadline Runs Complete**: All 8 focused deadline runs (Runs 1–8) are complete and held-out evaluated with test-set isolation strictly preserved.
- **Run 7 (E1 Hard-Shared MTL Control)**: Architecture (11.18M shared trunk, 747k task heads; 11,926,706 trainable params), training (Epoch 3 checkpoint, val objective 0.40036), and held-out evaluation complete across all three tasks (BCS MAE 0.1788, Behavior Acc 85.00% / Loss 0.6226, Re-ID Barn Rank-1 57.38% / mAP 30.37%).
- **Run 8 (E3 Modular MTL / Task-Private Adapters)**: Architecture (11.18M shared trunk, 395k private residual adapters, 747k task heads; 12.32M total trainable params, +3.32% over E1), training (Epoch 2 checkpoint, val objective 0.38888), and held-out evaluation complete across all three tasks (BCS MAE 0.1916 on matched test population, Behavior Acc 85.77% / Loss 0.4193, Re-ID Barn Rank-1 49.08% / mAP 28.04%).
- **E4 PCGrad Optimization Control**: E4 PCGrad is the final user-approved additional optimization control and is now fully trained and held-out evaluated. Architecture (11.18M shared trunk, 747k task heads; exactly 11,926,706 trainable params matching E1, 0 adapters, 0 gates), training (Epoch 3 checkpoint, val objective 0.40098; 44,177 projections across 16,140 super-steps, approx 45%–54% pairwise conflict frequency directly observed), and held-out evaluation complete across all three tasks (BCS MAE 0.1828, Behavior Acc 86.92% / Loss 0.5454, Re-ID Barn Rank-1 54.97% / mAP 30.23%; Snapshot Rank-1 57.00% / mAP 33.87%).
- **Calibrated Scientific Findings Across Controls**:
  - E4 vs E1: BCS slightly worse on MAE, Acc@0, Acc@1, Macro-F1; Behavior improved on overall accuracy (+1.92 pp), balanced accuracy (+1.85 pp), Macro-F1 (+0.0248), test loss (-0.0772), Walking F1 (0.0909 vs 0.0408), and CVB metrics; Re-ID mixed / near E1 on principal retrieval metrics (Barn Rank-1 lower by 2.41 pp, Barn mAP lower by 0.14 pp, Snapshot Rank-1 lower by 0.17 pp, Snapshot mAP higher by 0.18 pp).
  - E4 vs E3: E4 improved MAE, Acc@0, Acc@1, and Macro-F1 on BCS; E4 improved overall accuracy, balanced accuracy, Macro-F1, and Walking F1 on Behavior; E4 improved Rank-1 and mAP across both Barn and Snapshot queries on Re-ID; deeper-rank retrieval metrics were similar or lower under E4 (Barn Rank-10: 74.99% vs 75.03%; Snapshot Rank-5: 72.16% vs 72.32%; Snapshot Rank-10: 78.25% vs 80.23%).
  - Safe synthesis: E4 provided broader held-out improvements than E3 on the principal BCS and Behavior metrics and on Re-ID Rank-1/mAP, while some deeper-rank retrieval metrics remained similar or lower.
  - Negative transfer: Neither architectural modularity (E3) nor gradient projection (E4) completely eliminated negative transfer relative to single-task baselines. Shared-backbone pairwise gradient conflicts were directly observed during E4 training (44,177 projections across 16,140 super-steps, approx 45%–54% pairwise conflict frequency), but we do NOT claim gradient conflicts caused all E1 negative transfer.
- **Canonical Roadmap Preservation**: GradNorm E5 and partial sharing E2 remain deferred roadmap work, NOT completed or cancelled.

## Remaining Manuscript Integration Tasks

- **Chapter 1 (Introduction)**: Final integrated contribution wording and RQ alignment only.
- **Abstract**: Complete (self-contained 331-word synthesis of problem, cattle-centered single-task results, E1/E3/E4 multi-task findings, and bounded trade-off conclusion).
- **Chapter 4 (Proposed Methodology)**: Complete (includes E1/E3/E4 methodology, parameter allocation in Table 4.5/4.6, 538/16,140 super-step correction in Table 4.6/4.7, and PCGrad gradient routing in Figure 4.4).
- **Chapter 5 (Result Analysis)**: Complete (includes complete single-task and multi-task evaluations, Tables 5.8–5.12, PCGrad conflict diagnostics, and calibrated negative-transfer synthesis).
- **Chapter 6 (Conclusion)**: Complete (includes final RQ answers, 6 contributions, 12 limitations, 7 future work directions, and final synthesis).
- **Appendices, Evidence Map & Rubric Checklist**: Final E4 integration while preserving claim boundaries.
- **Whole-Document Proof & Front Matter**: Final proofing and administrative front matter.

## Remaining Non-MTL Administration

Confirm author order, submission metadata, committee details, signatures, acknowledgments, dataset permissions, ethics and AI-assistance disclosure, team contribution records, and any separate IEEE-format submission requirement.

