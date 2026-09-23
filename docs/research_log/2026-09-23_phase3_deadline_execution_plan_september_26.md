# Research Log: Phase 3 Deadline Execution Priority Overlay (Target: 26 September 2026)

**Date:** 2026-09-23  
**Status:** CANONICAL / ACTIVATED  
**Log Type:** Deadline Execution Priority Overlay / Thesis Scope Management (NOT an empirical data audit)  
**Parent Blueprint:** [`phase3_canonical_roadmap.md`](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md)  
**Overlay Document:** [`phase3_deadline_execution_2026-09-26.md`](file:///d:/cattle-health-monitoring-multi-task-model/phase3_deadline_execution_2026-09-26.md)  
**Associated Deliverables:**
- `phase3_deadline_execution_2026-09-26.md` (Standalone priority overlay)
- `phase3_canonical_roadmap.md` and `docs/phase3_canonical_roadmap.md` (Cross-referenced with active overlay notice)
- `memory/state.md` (Updated active goals and roadmap tracking)
- `docs/research_log/README.md` (Updated index table)

---

## 1. Executive Summary & Intent

With the thesis submission deadline set for **26 September 2026**, this decision log establishes an active, focused **Deadline Execution Priority Overlay**.

The guiding operational principle is:
> **"Before the 26 Sep deadline, execute only the minimum defensible thesis runs. All exhaustive ablations remain deferred roadmap work and can be completed later if needed."**

### Critical Boundary:
**THIS IS NOT A SCIENTIFIC CANCELLATION OF THE CANONICAL ROADMAP.**  
The 13-step canonical roadmap remains the authoritative long-term scientific blueprint. The overlay establishes an immediate execution filter to ensure that a complete, evaluated Multi-Task Deep Learning (MTL) framework integrating Body Condition Scoring (BCS), Behavior Recognition, and Cow Re-Identification (Re-ID) is fully realized and benchmarked before the deadline.

---

## 2. The 8-Run Focused Execution Sequence

Phase 3 execution is focused onto exactly 8 canonical runs:

### Group A: Single-Task Baselines (Prerequisite Controls)
1. **Run 1 — BCS RGB Baseline:** **COMPLETE & 100% CERTIFIED** ✅ (ScienceDB, ResNet-18, Ordinal BCE; Real MAE = 0.1848, Acc@1 = 86.74%).
2. **Run 2 — Behavior RGB Baseline:** **SMOKE TEST CERTIFIED** (CVB + Kaggle Beef protocol, ResNet-18, 224x224 RGB; 100% Modal smoke pass). **FULL 30-EPOCH TRAINING IS THE IMMEDIATE NEXT ACTION.**
3. **Run 3 — Re-ID RGB Baseline:** SideViewCows2026 primary protocol (ResNet-18, identity discrimination control).

### Group B: Perception-Enhanced Single-Task Models (Combined Configurations)
4. **Run 4 — BCS Perception-Enhanced Model:** ONE main combined configuration: `RGB + localization/crop + soft segmentation mask + anatomy/pose`. No multi-way factorial sweeps before deadline.
5. **Run 5 — Behavior Perception-Enhanced Temporal Model:** ONE main configuration: `cattle-centered RGB + segmentation (where operational) + anatomy/pose (where operational) + ONE lightweight temporal model (TCN)`. No GRU/LSTM/VideoMAE sweeps before deadline.
6. **Run 6 — Re-ID Perception-Enhanced Model:** ONE main configuration: `cattle-centered crop + soft segmentation mask + anatomy/pose (only if operational/useful)`.

### Group C: Multi-Task Learning Integration & Evaluation
7. **Run 7 — Basic Final MTL Control (E1: Hard Sharing):** Fully hard-shared vision trunk with separate heads. Detects and measures negative transfer across tasks.
8. **Run 8 — Main Deadline MTL Model (E3: Modular / Adapters / Task-Private Pathways):** Shared representation augmented with task-specific pathways or adapter modules. Main candidate for mitigating negative transfer.

---

## 3. Cattle Perception Module Definition & Viewpoint Policy

The canonical cattle-perception flow remains:
```text
image / video
-> localization (RT-DETR-L)
-> segmentation / soft mask (SAM 2.1)
-> anatomy / pose / keypoints (SuperAnimal-Quadruped)
-> viewpoint
```

### Strict Viewpoint Constraint:
- Viewpoint remains part of the theoretical perception module.
- However, zero-shot VLMs were empirically rejected, and MOO synthetic transfer achieved only 27.37% real diagnostic accuracy with severe rear-view bias.
- An operational real-cattle viewpoint generator is **NOT YET SELECTED**.
- Therefore, viewpoint **MUST NOT be silently forced into deadline training models**.
- If no reliable viewpoint generator is validated before deadline, viewpoint will be explicitly documented as *"empirically investigated in feasibility audits, but deferred from downstream feature fusion due to synthetic-to-real transfer domain gaps"*.
- **NEVER substitute camera ID or dataset session as a proxy for viewpoint.**

---

## 4. Scientific Claim Boundaries

Because the deadline plan uses combined perception configurations rather than exhaustive individual ablations:
- **PERMISSIBLE CLAIM:** *"The combined cattle-centered perception-enhanced representation improved (or did not improve) performance and robustness compared to generic RGB baselines."*
- **IMPERMISSIBLE CLAIMS (Unless an isolated ablation is executed):** *"Segmentation individually caused the improvement"* or *"Anatomy individually caused the improvement."*

---

## 5. Deferred Experiment Registry (Deferred, NOT Deleted)

The following roadmap items are formally deferred to post-deadline thesis follow-ups:
- Full A0–A4 localization/segmentation ladder
- Full B0–B3 anatomy/pose controls (including shuffled-pose baselines)
- Full C0–C3 viewpoint conditioning experiments
- Full D0–D6 temporal behavior ladder (GRU, LSTM, pose-only temporal, VideoMAE, SlowFast)
- E2 exhaustive layer-wise partial-sharing search
- E4 (PCGrad) and E5 (GradNorm) optimization sweeps
- Optional cattle-specific pretraining (Step 12)
- Exhaustive Cartesian combinations of perception x temporal x MTL

---

## 6. Immediate Next Step

Launch full 30-epoch training of Run 2 (Step 4.2 Behavior RGB baseline) on Modal L40S:
```bash
modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior.py::main --epochs 30 --batch-size 64
```
