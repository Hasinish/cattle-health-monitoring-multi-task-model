# Research Log: Official Thesis Title Lock & Mandatory Final MTL Integration Scope Clarification

**Date:** 2026-09-23  
**Status:** CANONICAL / LOCKED BY USER APPROVAL  
**Log Type:** User-Approved Thesis-Scope Constraint / Clarification (NOT an empirical result or data audit)  
**Fixed Official Thesis Title:** **“Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring”**  
**Associated Deliverables:**
- `phase3_canonical_roadmap.md` (Updated header, current thesis direction, overview diagram, Step 11 title and goal, summary list)
- `docs/phase3_canonical_roadmap.md` (Mirrored 100% bit-identical)
- `memory/state.md` (Updated Phase 3 status and Step 11 active goal)
- `docs/research_log/README.md` (Updated historical index table)

---

## 1. Executive Summary

This log documents an explicit, binding thesis-scope clarification approved by Hasin Ishrak regarding the official thesis title and the ultimate structural deliverable of Phase 3.

The official thesis title is permanently locked as:
> **“Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring”**

This title cannot be changed, superseded, or diluted. Consequently, the research program **MUST culminate in an evaluated multi-task deep learning (MTL) framework** integrating the three core downstream tasks:
1. **Body Condition Scoring (BCS)**
2. **Behavior Recognition**
3. **Individual Cow Identification / Re-Identification**

The working title *"Vision-Based AI for Cattle Health Monitoring"* remains valid solely as an informal project shorthand, but the formal academic and thesis requirement demands an integrated multi-task deep learning framework.

---

## 2. Mandatory Final MTL vs. Single-Task-First Methodology

A critical scientific boundary is formalized: **making the final MTL framework mandatory does NOT permit skipping or abbreviating the single-task-first methodology.**

### The Prerequisite Sequence
The canonical research workflow proceeds strictly in sequence:
```text
Step 4: Single-Task Baselines
        - Step 4.1: BCS RGB baseline (COMPLETE & CERTIFIED: Real MAE 0.1848, Acc@1 86.74%)
        - Step 4.2: Single-frame RGB Behavior baseline (CVB + Kaggle Beef protocol)
        - Step 4.3: Single-frame RGB Re-ID baseline (SideViewCows2026 protocol)
   ↓
Steps 5–7: Cattle-Centered Representation Ablations per Task
        - Step 5: Localization / segmentation ablations
        - Step 6: Anatomy / pose ablations
        - Step 7: Viewpoint ablations
   ↓
Step 8: Temporal Behavior Experiments (Ladder of temporal modeling)
   ↓
Step 9: Consolidated Task-Conditioned Single-Task Models
   ↓
Step 10: Cross-Domain & Robustness Evaluations
   ↓
Step 11: Mandatory Final MTL Integration & Evaluation
        - Evaluate sharing topologies (E1–E5) against single-task reference (E0)
        - Quantify negative transfer, gradient conflict, and representation alignment
```

Single-task models are indispensable because:
1. They establish the unconstrained upper-bound reference performance (**E0 control**).
2. They identify which visual priors (masks, keypoints, viewpoint) each distinct biological task preserves or discards.
3. Without strong single-task baselines, any multi-task degradation cannot be isolated as negative transfer versus poor baseline representations.

---

## 3. Scope of Step 11: Determining HOW to Share, Not WHETHER to Share

Roadmap Step 11 has been retitled from *"Revisit Sharing / MTL"* to:
> **“Mandatory Final MTL Integration & Evaluation”**

Its status remains **LATE PHASE ONLY**, blocked until Steps 4 through 10 provide validated single-task cattle-centered representations.

The research objective of Step 11 is to discover **HOW** the tasks should share visual information, rather than debating *whether* an MTL model will be delivered.

### Architectural Agnosticism & Empirical Alternatives
The final MTL architecture must **NOT** be predetermined in advance. All of the following topologies remain open empirical alternatives to compare:
- **E0 (Control):** Three independent single-task cattle-centered models.
- **E1 (Hard Sharing):** Fully shared vision backbone with bifurcated task heads.
- **E2 (Partial Sharing):** Shared low-level/early spatial layers with private late representation trunks.
- **E3 (Modular / Gated Sharing):** Shared backbone augmented with task-specific adapter modules, FiLM conditioning, or cross-attention gates.
- **E4 (Optimization Control — PCGrad):** Hard-shared backbone trained with Projecting Conflicting Gradients.
- **E5 (Optimization Control — GradNorm):** Hard-shared backbone trained with dynamic loss balancing.

### Task Asymmetry & Task-Private Routing
Crucially, the final MTL framework **does not require every task to consume identical feature representations**.
- BCS requires dorsal/pelvic anatomical morphology while discarding identity cues.
- Behavior requires posture, limb kinematics, and spatial context.
- Re-ID requires coat pigmentation patterns and idiosyncratic contour cues while remaining invariant to pasture context.

The unified MTL framework may therefore incorporate task-private pathways, asymmetric feature routing, or modular adapters while functioning as a single, coherent multi-task model.

---

## 4. Preservation of Scope Boundaries

In accordance with strict research hygiene:
1. **Core Tasks:** Fixed to BCS, Behavior, and Cow ID/Re-ID.
2. **Lameness Status:** Strictly preserved as excluded from Phase 3 MTL (retained as historical Phase 2 audit evidence only).
3. **Datasets & Splits:** Zero modifications to existing dataset roles, split definitions, manifests, or audit CSVs.
4. **Compute & Execution:** Zero training runs or inference jobs were scheduled or launched. This update is purely structural scope alignment.
