# Phase 3 writing status

Reviewed source snapshot: `a695a92e7c3b52b165053e182f6d3e3331b0866f` on 2026-09-24.

Status: nearly complete pre-MTL thesis manuscript. Runs 1–6 are incorporated. Runs 7–8, final integrated synthesis, final Abstract, and final Conclusion remain pending.

| Section/file | Status | Remaining boundary |
|---|---|---|
| Front matter | Provisional | Administrative names, signatures, approval, acknowledgments, ethics/AI disclosure |
| Abstract | Interim pre-MTL version | Replace after E1/E3 results and negative-transfer analysis |
| Chapter 1: Introduction | Stable before MTL | Final integrated contribution wording only |
| Chapter 2: Literature Review | Stable before MTL, normal final polishing allowed | No speculative reference expansion |
| Chapter 3: Requirements, Impacts and Constraints | Stable before MTL | Deployment-specific costs, approvals, and user evidence remain unavailable |
| Chapter 4: Proposed Methodology | Runs 1–6 stable; MTL role defined | Insert exact executed E1/E3 implementation after Runs 7–8 |
| Chapter 5: Result Analysis | Runs 1–6 stable | Add E0/E1/E3 results, negative transfer, and integrated discussion |
| Chapter 6: Conclusion | Structured interim synthesis | Final research-question answers, contribution list, limitations, and conclusion after Runs 7–8 |
| Evidence appendices | Updated through Runs 1–6 | Add Run 7/8 artifacts and any final external/repeated-run evidence |
| Evidence map | Updated through E34 | Preserve claim boundaries when adding MTL evidence |
| Rubric checklist | Updated | MTL-dependent CO5–CO7 and final CO14 deliverables remain partial |
| LaTeX build | PASS: 69-page PDF, no fatal error, undefined citation/reference, or overfull box | See `BUILD_REPORT.md`; only inherited/nonfatal warnings remain |

## Stable before MTL

- Introduction, motivation, problem statement, research questions, objectives, scope, and task-specific supported contributions.
- Literature review and synthesis of BCS, Behavior, Re-ID, perception, temporal modeling, MTL, negative transfer, and task-specific representations.
- Functional/non-functional/scientific requirements; impacts, constraints, ethics, project/resource management, risks, and partial economics.
- Dataset selection, label harmonization, split protection, leakage boundaries, and primary/external roles.
- Perception feasibility and the evidence-based exclusion of pose/viewpoint from Runs 4–6.
- Run 1/4 BCS methodology, coverage, matched results, interpretation, and limitations.
- Run 2/5 Behavior methodology, coverage, matched results, source/class analysis, negative findings, and limitations.
- Run 3/6 Re-ID methodology, identity-disjoint protocol, Barn/Snapshot results, oracle boundary, and limitations.
- The separate held-out viewpoint result and its downstream transfer boundary.

## Pending Runs 7–8

- Exact executed E1 hard-shared architecture, task sampling, losses, capacity, and compute.
- Exact executed E3 modular/task-private architecture, adapters/gates/private pathways, task sampling, losses, capacity, and compute.
- Matched E0/E1/E3 task-wise results.
- Phase 3 negative-transfer measurement.
- E1-versus-E3 comparison and final integrated architecture verdict.
- Final overall Discussion and research-question synthesis.
- Final contribution list, Abstract, Conclusion, and Future Work synthesis.

## Remaining non-MTL administration

Confirm author order, submission metadata, committee details, signatures, acknowledgments, dataset permissions, ethics and AI-assistance disclosure, team contribution records, and any separate IEEE-format submission requirement.
