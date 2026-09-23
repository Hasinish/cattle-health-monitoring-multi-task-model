# Evidence map

Read-only source: `main` at `9baa0c68355d59a039a4c3ce17d71a6c1e16d4e2`. Writing branch: `thesis-writing`.
This review revision is not a substitute for each experiment's historical execution revision.

Published sources use the bibliography. E-identifiers link project claims to pinned records below. This is a writing ledger, not a replacement roadmap.

| ID | Source path at the reviewed snapshot | Supported content |
|---|---|---|
| E01 | `phase3_canonical_roadmap.md` | Scope, roles and scientific constraints |
| E02 | `memory/state.md` | Current status, completed hydration and diagnostic limits |
| E03 | `phase3_deadline_execution_2026-09-26.md` | Deadline comparison plan; stale status text not substituted for results |
| E04 | `datasets/bcs/sciencedb/split_report.md` | Repaired bursts, counts and limitations |
| E05 | `datasets/behavior/cvb_beef/split_report.md` | Labels, groups, partitions and split seed |
| E06 | `datasets/id/sideviewcows2026/split_report.md` | Approved primary role and prepared protocols |
| E07 | `datasets/dataset_registry.csv` | Metadata, availability and licenses; contradictions retained |
| E08 | `artifacts/bcs_baseline/bcs_baseline_metrics.json` | BCS results, configuration, hashes and missing execution revision |
| E09 | `artifacts/behavior_baseline/behavior_baseline_metrics.json` | Recorded Behavior results and selected epoch |
| E10 | `scripts/train_sciencedb_bcs_baseline.py` | Ordinal implementation, input procedure and metrics |
| E11 | `docs/audits/phase3_perception_feasibility.md` | Detection, segmentation and pose audit; restricted manual review |
| E12 | `scripts/train_cvb_beef_behavior_baseline.py` | Input extraction, fallbacks, code defaults and aggregation |
| E13 | `docs/research_log/2026-09-23_behavior_primary_stack_correction.md` | Approved Behavior role correction |
| E14 | `docs/research_log/2026-09-23_sciencedb_bcs_baseline_full_training_results.md` | Runtime/cost narrative; history discrepancies retained |
| E15 | `artifacts/behavior_baseline/behavior_baseline_30epoch_summary.md` | Full-run summary, not a cache audit |
| E16 | `thesis_marking_rubrics.md` | All fourteen marking criteria and allocations |
| E17 | `thesis template/main.tex` | Official format and chapter mapping |
| E18 | `cattle_thesis_p2_latex/core/titlepage.tex` | Existing author roster; final details pending |
| E19 | `docs/research_log/2026-09-23_sideviewcows2026_reid_baseline_smoke_test.md` | New Re-ID implementation and smoke log |
| E20 | `scripts/train_sideview_reid_baseline.py` | Raw-feature classifier and normalized retrieval implementation |
| E21 | `artifacts/reid_baseline_smoke/reid_baseline_metrics.json` | Smoke artifact explicitly skips Protocol A retrieval |

## Claim coverage

| Location | Evidence | Boundary |
|---|---|---|
| Introduction | E01--E06 | Objectives are not completed findings. |
| Literature Review | Verified bibliography plus E01/E11 | Focused starting set, not a systematic review. |
| Requirements, impacts and management | E01--E07, E14--E18 | No invented interviews, approvals, emissions or ROI. |
| BCS methods/results | E04, E08, E10 | Burst-safe, not cow-disjoint; historical Git SHA missing. |
| Behavior methods/results | E05, E09, E12, E15 | Full-run fallback use and execution provenance unresolved. |
| Re-ID method/readiness | E19--E21 | Implemented and smoke-tested, not full retrieval evaluated. |
| Perception findings | E02, E11 | Limited audit/diagnostic samples, not downstream benefits. |
| Enhanced/temporal/MTL/external results | E01--E03 | Explicitly pending. |

## Selected source blob identifiers

BCS result: `862bb2377d3abad86a2f971319cedf26bb948457`.
Behavior result: `6cb42799bcff29ebf9f5205fe819e5449427e4ac`.
Behavior code: `587d32cb9b7b60aac8758529f31639bd63bde4b9`.
Re-ID code: `b1fea82f598d3acd11efdef21b28b4199326872d`.
Re-ID smoke result: `4231fdd2bc254fbac9d8c53dfcc60c925ff107e8`.
Re-ID smoke log: `d2e5dfe1aaf8609957a6bb682704f6b46804485a`.
Rubric: `b1f13eb4e3e238697fbd5270cf27cbdd66f12368`.
Template main: `7e7be9d92e643738768d63270f59b5b06695becc`.

## Recorded split SHA-256 values

ScienceDB train: `9f6b0bc716e01a2ab22208daff1c49e49fd450a4d7cf0a57b2979275ed33497a`

ScienceDB validation: `e223e3c4c081ca5c9f993f7156dc791df97b6ea6d011b8b4b6f068590b3d975d`

ScienceDB test: `eae459e031d06c4b1150ce2cbcdcb8259724b070b831341222b15c99e3626e5f`

Behavior train: `117d3191b175f4a6f43dc3cfb92f1ecbe42230f7f46a01c2d67cb81d84177e30`

Behavior validation: `897105d6266eba01b2b7bd45e2a7eb63bca7e9107faa202b07ba82e6d866b925`

Behavior test: `0a67faf182a5ce8d3c02188656553310a6720a54d23f114b80d8b6093e00b30e`

These values come from the result/split records. The raw datasets and complete split files were not independently re-audited in the writing environment.

## Unresolved issues

1. Behavior code allows synthetic failed-extraction inputs and CVB full-frame fallbacks. Existence is verified; use/count in the full run is not established.
2. Behavior JSON lacks executed config, training seed, revision and split hashes. Split seed 2026 is not automatically the training seed.
3. Source-specific Behavior Macro-F1 uses an implicit class list; Beef lacks true Walking.
4. BCS run revision is UNKNOWN. Some narrative epoch-history values disagree with JSON; training curves are withheld.
5. Deadline BCS pose configuration conflicts with the restricted negative manual pose review. Record an explicit design decision.
6. CBVD-5 registry taxonomy conflicts with the verified publication. Reconcile imported labels before external mapping.
7. SideView's stale external-role registry wording is superseded by its approved primary protocol report. Hydration and Re-ID smoke testing are complete; full retrieval is not.
8. Ruchay registry availability and later visual sample availability do not establish complete raw hydration in every environment. Wider BCS support and Dryad's different scale/modality need a frozen protocol.
9. Kaggle Beef reuse rights and final administrative/ethics details remain unresolved.
10. Re-ID log diagram simplifies normalization placement; the draft follows actual code: classifier on raw features, retrieval on normalized features. Smoke log/JSON scopes differ and are not merged into a full result.

No dataset, experiment script, result artifact, roadmap, memory file, previous thesis, sample paper or original template is changed by this writing task.
