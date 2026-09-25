# Evidence map

Reviewed repository snapshot: `d55ea2df436f504c3de4d83bcf9db845a6b706e3` (2026-09-26).

This file keeps repository provenance out of the main academic narrative. Published claims use the bibliography; E-identifiers connect project-specific claims to versioned records. All single-task baselines (Runs 1–3), perception-enhanced models (Runs 4–6), and multi-task configurations (E1 hard-shared, E3 modular task-private, and E4 PCGrad optimization control) are complete and held-out evaluated.

| ID | Source path at the reviewed snapshot | Supported content |
|---|---|---|
| E01 | `phase3_canonical_roadmap.md` | Phase 3 scope, dataset roles, cattle-centered direction, and MTL boundary |
| E02 | `memory/state.md` | Latest project status and completed experiment milestones |
| E03 | `phase3_deadline_execution_2026-09-26.md` | Focused eight-run sequence and planned E1/E3 roles |
| E04 | `datasets/bcs/sciencedb/split_report.md` | ScienceDB counts, repaired burst groups, split protection, and identity limitation |
| E05 | `datasets/behavior/cvb_beef/split_report.md` | Behavior mappings, counts, source/session grouping, split seed, and Walking limitation |
| E06 | `datasets/id/sideviewcows2026/split_report.md` | SideView scale and verified protocols, including identity-disjoint evaluation |
| E07 | `datasets/dataset_registry.csv` | Primary/external dataset roles, availability, modalities, and license fields |
| E08 | `artifacts/bcs_baseline/bcs_baseline_metrics.json` | Run 1 configuration, full-test metrics, confusion matrix, and split hashes |
| E09 | `artifacts/behavior_baseline/behavior_baseline_metrics.json` | Run 2 full-test, class-specific, and source-specific metrics |
| E10 | `scripts/train_sciencedb_bcs_baseline.py` | Run 1 ordinal-BCE implementation and physical-score mapping |
| E11 | `docs/audits/phase3_perception_feasibility.md` | Detector, segmentation, pose, and restricted manual-review findings |
| E12 | `scripts/train_cvb_beef_behavior_baseline.py` | Run 2 single-frame extraction and classifier implementation |
| E13 | `docs/research_log/2026-09-23_behavior_primary_stack_correction.md` | Approved CVB+Beef primary Behavior role |
| E14 | `docs/research_log/2026-09-23_sciencedb_bcs_baseline_full_training_results.md` | Run 1 training/runtime narrative and result interpretation |
| E15 | `artifacts/behavior_baseline/behavior_baseline_30epoch_summary.md` | Run 2 selected checkpoint and full-run summary |
| E16 | `thesis_marking_rubrics.md` | BRACU report criteria and chapter mapping |
| E17 | `thesis template/main.tex` | Supplied report structure and front-matter conventions |
| E18 | `cattle_thesis_p2_latex/core/titlepage.tex` | Existing author roster, subject to final administrative confirmation |
| E19 | `docs/research_log/2026-09-23_sideviewcows2026_reid_baseline_smoke_test.md` | Run 3 pre-training execution validation |
| E20 | `scripts/train_sideview_reid_baseline.py` | Run 3 raw-feature classifier and normalized retrieval implementation |
| E21 | `artifacts/reid_baseline_smoke/reid_baseline_metrics.json` | Run 3 smoke-test scope; not used as final retrieval evidence |
| E22 | `artifacts/reid_baseline/reid_baseline_metrics.json` | Run 3 full Protocol A Barn/Snapshot retrieval results |
| E23 | `docs/research_log/2026-09-23_sideviewcows2026_reid_baseline_full_training_results.md` | Run 3 training and evaluation record |
| E24 | `artifacts/bcs_perception_run4/bcs_perception_matched_test_comparison.json` | Run 4 coverage, matched Run 1 control, Run 4 metrics, and deltas |
| E25 | `scripts/train_sciencedb_bcs_perception.py` | Run 4 four-channel ordinal model and binary-mask handling |
| E26 | `artifacts/behavior_perception_cache/perception_summary.json` | Run 5 Train/Validation cache coverage, mask counts, exclusions, and class retention |
| E27 | `artifacts/behavior_run5_test/run2_vs_run5_matched_comparison.json` | Run 5 matched test, class/source metrics, and Run 2 control |
| E28 | `scripts/train_cvb_beef_behavior_tcn.py` | Run 5 sampling, four-channel frame encoder, and residual Conv1D implementation |
| E29 | `artifacts/reid_perception_run6/reid_perception_metrics.json` | Run 6 oracle representation, integrity checks, training, and Protocol A results |
| E30 | `scripts/train_sideview_reid_perception.py` | Run 6 GT-mask crop, fourth channel, and matched retrieval implementation |
| E31 | `artifacts/viewpoint_real_finetune/test_evaluation_metrics.json` | Final held-out real-cattle viewpoint metrics |
| E32 | `docs/research_log/2026-09-24_sciencedb_bcs_perception_full_training_results.md` | Run 4 full-training and matched-test narrative |
| E33 | `docs/research_log/2026-09-24_run5_behavior_perception_test_evaluation_results.md` | Run 5 strict test gate and matched comparison narrative |
| E34 | `docs/research_log/2026-09-24_sideviewcows2026_reid_perception_run6_full_training_results.md` | Run 6 full-training and Protocol A narrative |
| E35 | `artifacts/mtl_e4_training/mtl_e4_metrics.json` | E4 training diagnostics; 16,140 super-steps; 44,177 triggered projections; 2.737 projections/super-step; pairwise shared-gradient conflict measurements |
| E36 | `artifacts/mtl_e1_evaluation/mtl_e1_test_evaluation_metrics.json` | Official E1 held-out BCS, Behavior, and Re-ID results; matched single-task comparison values embedded only where appropriate |
| E37 | `artifacts/mtl_e3_evaluation/mtl_e3_test_evaluation_metrics.json` | Official E3 held-out BCS, Behavior, and Re-ID metrics |
| E38 | `artifacts/mtl_e4_evaluation/mtl_e4_test_evaluation_metrics.json` | Official E4 held-out BCS, Behavior, and Re-ID metrics |
| E39 | `scripts/train_mtl_e1_hard_shared.py` | Executed monolithic hard-shared architecture, shared backbone / task heads, joint optimization procedure |
| E40 | `scripts/train_mtl_e3_modular.py` | Task-private residual adapter architecture, +395,904 private parameters, +3.32% capacity difference |
| E41 | `scripts/train_mtl_e4_pcgrad.py` | E1-matched architecture, PCGrad projection implementation, shared-gradients-only intervention, unprojected task-head gradients, deterministic projection ordering |

## Claim coverage

| Manuscript area | Evidence | Interpretation boundary |
|---|---|---|
| Introduction and scope | E01–E07 | Objectives and scope aligned with completed single-task and multi-task evidence. |
| Literature review | Verified bibliography | Sample theses are style references only, never scientific evidence. |
| Requirements, impacts, constraints | E01–E07, E16–E18 | No invented interviews, approval, ROI, emissions, or deployment trial. |
| Run 1 BCS | E04, E08, E10, E14 | Burst-group-disjoint, not biological cow-disjoint; single run. |
| Run 4 BCS | E24, E25, E32 | Matched perception-successful subset; combined crop+mask configuration. |
| Run 2 Behavior | E05, E09, E12, E15 | Source/session-grouped; Walking is CVB-only; single frame. |
| Run 5 Behavior | E26–E28, E33 | Combined representation+temporal change; matched 780-sequence test. |
| Run 3 Re-ID | E06, E20, E22, E23 | Protocol A identity-disjoint; Barn and Snapshot reported separately. |
| Run 6 Re-ID | E29, E30, E34 | Oracle GT segmentation-guided; not automatic SAM deployment evidence. |
| Viewpoint | E31 | Own held-out test only; downstream transfer and synthetic-pretraining causality unproven. |
| Hard-Shared MTL (E1) | E36, E39 | Held-out degradation supports outcome-level negative transfer; gradient conflict mechanism not directly diagnosed in E1. |
| Modular Task-Private MTL (E3) | E37, E40 | Task-private residual adapters; +3.32% parameter capacity means routing-only attribution is not possible. |
| PCGrad MTL (E4) | E35, E38, E41 | Gradient conflict measurements recorded during E4 only; do not retroactively assign identical conflict frequencies to E1/E3; selective mitigation, not universal recovery. |

## Unresolved evidence boundaries

1. All reported single-task baselines and multi-task models were evaluated from single trained checkpoints. Computational constraints precluded repeated-seed training distributions, confidence intervals, standard deviations, or formal hypothesis testing; reported metric differences represent descriptive point comparisons.
2. The later Run 2-versus-Run 5 comparison artifact embeds a historical Run 2 class/source support breakdown that disagrees with the original Run 2 baseline artifact. The thesis treats the original artifact (E09) as authoritative for the canonical 809-sample result and uses E27 only for the explicitly matched 780-sample comparison; the inconsistent embedded historical breakdown is not reproduced.
3. ScienceDB has no verified released biological cow IDs. Its results represent sequence-safe burst-group evaluation, not biological cow-disjoint evaluation.
4. The primary Behavior protocol is source/session-grouped, not cow-disjoint; Walking is present only in CVB.
5. In single-task comparisons (Runs 4–6), localization, cropping, binary masks, and temporal convolutions were evaluated in combined configurations. The individual marginal contribution of each isolated component was not ablated independently.
6. Run 6 uses released ground-truth masks. It does not validate automatic upstream segmentation for Re-ID, and crop-versus-mask effects are not separated.
7. The viewpoint classifier has a valid held-out result, but its transfer to the three downstream datasets was not verified. No same-split ImageNet-only control isolates the effect of MOO pretraining.
8. External BCS, Behavior, and Re-ID stress tests (such as Ruchay, Dryad, MmCows, CBVD-5, and BECA) remain unexecuted roadmap items; results do not imply verified cross-domain generalization.
9. Modular adapter evaluation (E3) added 395,904 trainable parameters (+3.32% capacity over E1); observed differences cannot be attributed purely to architectural routing in isolation from capacity scale.
10. Direct gradient conflict measurements were recorded exclusively during E4 PCGrad training; conflict was not directly tracked during E1 optimization. PCGrad provided selective mitigation on some metrics, but dedicated single-task models remained superior on several task-specific measures.
11. Canonical roadmap items E2 (partial parameter sharing) and E5 (GradNorm adaptive loss weighting) remain deferred roadmap items, NOT completed and NOT required to claim the executed thesis comparisons.
12. Final administrative details, permissions, ethics/AI-assistance disclosure, and team contribution records require author and institutional confirmation.

No experimental source, dataset, roadmap, memory file, sample report, or prior thesis is modified by this thesis-writing task.
