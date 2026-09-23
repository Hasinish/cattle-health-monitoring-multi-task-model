# Phase 3 Deadline Execution Priority Overlay (Target: 26 September 2026)

**Thesis Title:** Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring (FIXED & LOCKED)  
**Document Status:** ACTIVE DEADLINE PRIORITY OVERLAY  
**Effective Date:** 2026-09-23  
**Target Deadline:** 2026-09-26 (Submission / Defense Draft)  
**Parent Document:** [phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md)  

---

## 1. Executive Intent & Guiding Principle

> **"Before the 26 Sep deadline, execute only the minimum defensible thesis runs. All exhaustive ablations remain deferred roadmap work and can be completed later if needed."**

### Critical Scientific Principle:
**THIS IS NOT A SCIENTIFIC CANCELLATION OF THE CANONICAL ROADMAP.**  
The full 13-step canonical roadmap remains the authoritative long-term scientific blueprint. This document defines a time-boxed execution priority overlay to ensure that an empirical, evaluated, and methodologically sound Multi-Task Deep Learning framework across the three core tasks (BCS, Behavior, Re-ID) is fully realized and benchmarked before September 26.

All non-essential hyper-combinations, exhaustive component sweeps, and secondary temporal architectures are deferred to post-deadline thesis follow-ups.

---

## 2. The 8-Run Deadline Execution List

To satisfy the thesis scope while respecting the strict time constraint, Phase 3 execution is focused onto exactly 8 canonical runs:

```text
               [SINGLE-TASK BASELINES]
1. BCS RGB Baseline (Done)      2. Behavior RGB Baseline      3. Re-ID RGB Baseline
         ↓                                ↓                            ↓
               [PERCEPTION-ENHANCED SINGLE-TASK]
4. BCS Perception Model         5. Behavior Perception+TCN    6. Re-ID Perception Model
         \                                |                           /
          \                               |                          /
           \                              |                         /
            ------------------------------+-------------------------
                                          |
                               [FINAL MTL EVALUATION]
                               7. Basic MTL Control (E1: Hard Sharing)
                               8. Main Deadline MTL (E3: Modular/Adapters/Task-Private)
```

### Run-by-Run Specifications:

#### Run 1: BCS RGB Single-Task Baseline
- **Status:** **COMPLETE & 100% CERTIFIED** ✅
- **Details:** ScienceDB (53,566 images, 5,653 repaired burst groups), ResNet-18, Ordinal BCE.
- **Results:** Real MAE = 0.1848 BCS units, Within +-1 Class Accuracy (Acc@1) = 86.74%, Balanced Accuracy = 40.41%, Macro-F1 = 0.4110. Global best checkpoint saved at Epoch 8.

#### Run 2: Behavior RGB Single-Frame Baseline
- **Status:** **SMOKE-TEST CERTIFIED; FULL TRAINING IS IMMEDIATE NEXT ACTION**
- **Details:** Canonical CVB + Kaggle Beef protocol (`datasets/behavior/cvb_beef/`; Train 3,785, Val 680, Test 809; Seed 2026).
- **Architecture:** ResNet-18, 224x224 RGB, CrossEntropyLoss, AdamW.
- **Critical Input Rule:** Target cow cropped via GT bbox on deterministic midpoint frame for CVB; midpoint frame used directly for single-cow Kaggle Beef. Walking is CVB-only.
- **Verification:** Modal smoke test passed 100% (App `ap-mqOh4m8zkMkHLtfOunYE5c`, GPU T4, runtime 63.2s, test.csv untouched).

#### Run 3: Re-ID RGB Single-Task Baseline
- **Status:** **PENDING (Prerequisite for MTL)**
- **Details:** SideViewCows2026 primary protocol.
- **Architecture:** ResNet-18 feature extractor with identity loss (CrossEntropy / CosFace / Triplet). Establishes single-task identity discrimination control.

#### Run 4: BCS Perception-Enhanced Single-Task Model
- **Configuration:** **ONE main combined configuration:**
  `RGB + cattle localization/crop + segmentation / soft mask + anatomy / pose`.
- **Constraint:** Do NOT run every individual combination sweep before the deadline. Combine the validated perception features to evaluate whether cattle-centered representation beats generic RGB.

#### Run 5: Behavior Perception-Enhanced Temporal Model
- **Configuration:** **ONE main configuration:**
  `cattle-centered RGB + segmentation (where operational) + anatomy/pose (where operational) + ONE lightweight temporal model (TCN)`.
- **Constraint:** Do NOT run GRU, LSTM, VideoMAE, or SlowFast before deadline unless core work is fully finished. TCN provides the fastest, most stable, and least computationally intensive temporal baseline.

#### Run 6: Re-ID Perception-Enhanced Single-Task Model
- **Configuration:** **ONE main configuration:**
  `cattle-centered crop + soft segmentation mask + anatomy/pose (only if operational/useful)`.
- **Constraint:** Do not run large combination sweeps. Focus on background suppression and coat preservation.

#### Run 7: Basic Final MTL Control (E1: Hard Sharing)
- **Configuration:** Fully hard-shared vision trunk with bifurcated task heads for BCS, Behavior, and Re-ID.
- **Scientific Role:** Serves as the primary control to detect, measure, and document negative transfer across disparate task objectives.

#### Run 8: Main Deadline MTL Model (E3: Modular / Adapters / Task-Private Pathways)
- **Configuration:** Shared trunk + task-specific private pathways, lightweight adapter modules, or cross-attention gates.
- **Scientific Role:** The primary thesis contribution for mitigating negative transfer, allowing BCS (morphology), Behavior (posture/motion), and Re-ID (coat pattern) to preserve specialized features while benefiting from unified multi-task representations.

---

## 3. Cattle Perception Module Architecture & Flow

The definition of the cattle perception pipeline is preserved identically to the canonical roadmap:

```text
image / video
    ↓
cow localization (RT-DETR-L)
    ↓
segmentation / soft mask (SAM 2.1)
    ↓
anatomy / pose / keypoints (SuperAnimal-Quadruped)
    ↓
coarse viewpoint
    ↓
cache reproducible cattle-centered representations
```

---

## 4. Strict Viewpoint Policy for Deadline Execution

Viewpoint remains a formal theoretical component of the cattle-perception module in the canonical roadmap. However, the following empirical realities govern deadline execution:

1. **Empirical Status:** The frozen zero-shot VLM probe (CLIP/OpenCLIP/SigLIP) was empirically rejected under the tested setup. The fully fine-tuned MOO synthetic ResNet-18 model achieved only 27.37% real diagnostic accuracy on non-ambiguous cattle crops, displaying severe rear-view bias.
2. **Operational Constraint:** A robust, validated real-cattle viewpoint generator is **NOT YET SELECTED**.
3. **Strict Prohibition:** Viewpoint **MUST NOT** be silently or artificially forced into deadline training models.
4. **Fallback Handling:** If no reliable real-cattle viewpoint generator is validated in time, viewpoint will be explicitly documented in the thesis as:
   > *"Tested under Phase 3 feasibility audits, but currently deferred from downstream feature fusion due to synthetic-to-real transfer domain gaps."*
5. **Absolute Boundary:** **NEVER substitute camera ID or dataset session as a proxy for viewpoint.** Camera ID encodes farm and background shortcuts, directly violating anti-shortcut learning principles.

---

## 5. Scientific Claim Boundaries

Because the deadline plan evaluates combined perception configurations rather than exhaustive individual component ablations:

- **PERMISSIBLE CLAIM:**
  > *"The combined cattle-centered perception-enhanced representation improved (or did not improve) out-of-distribution robustness and accuracy compared with generic RGB baselines across BCS, Behavior, and Re-ID."*
- **IMPERMISSIBLE CLAIMS (Unless an isolated ablation run is executed):**
  > *"Segmentation individually caused the observed improvement."*  
  > *"Anatomy/pose keypoints individually caused the observed improvement."*

All claims in the thesis defense must strictly align with the exact modular comparisons performed.

---

## 6. Comprehensive Deferred Experiment Registry

The following experiments from the canonical roadmap are formally **DEFERRED** until after the September 26 deadline:

| Roadmap Component | Deferred Experiments | Reason for Deferral |
| :--- | :--- | :--- |
| **Segmentation / Localization (Step 5)** | Full A0–A4 component ladder (A0 full, A1 box, A2 soft mask, A3 hard mask, A4 background-only) | Time and compute constraint; evaluate combined A2+pose first |
| **Anatomy / Pose (Step 6)** | Full B0–B3 controls (B1 heatmaps, B2 keypoint coords, B3 shuffled keypoint controls) | Evaluated as part of combined perception representation |
| **Viewpoint (Step 7)** | Full C0–C3 viewpoint experiments (C1 one-hot, C2 embedding, C3 viewpoint-conditioned norm) | Operational real-cattle generator not yet certified; prevents shortcut injection |
| **Temporal Behavior (Step 8)** | Full D0–D6 ladder: D2 LSTM, D3 GRU, D4 pose-only sequence, D5 two-stream, D6 VideoMAE/SlowFast | TCN selected as sole deadline temporal architecture for stability and speed |
| **Consolidated MTL (Step 11)** | E2 exhaustive partial-sharing layer search, E4 PCGrad, E5 GradNorm sweeps | E0 (single), E1 (hard), and E3 (adapters/private) form the minimum defensible set |
| **Pretraining (Step 12)** | Cattle-specific masked autoencoding / self-supervised pretraining | Optional stretch goal; non-essential for primary thesis defense |
| **Combinatorial Sweeps** | Exhaustive N-way Cartesian product of all perception x temporal x MTL options | Scientifically secondary to proving core MTL feasibility and negative-transfer mitigation |

---

## 7. Immediate Action Checklist

1. [x] Step 4.1 BCS RGB Single-Task Baseline: 100% COMPLETE & CERTIFIED.
2. [x] Step 4.2 Behavior RGB Single-Task Baseline: Smoke-tested on Modal with 100% pass.
3. [ ] **IMMEDIATE NEXT ACTION:** Launch full 30-epoch training of Step 4.2 Behavior RGB baseline on Modal L40S:
   ```bash
   modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior.py::main --epochs 30 --batch-size 64
   ```
4. [ ] Implement and verify Step 4.3 Re-ID RGB baseline on SideViewCows2026.
5. [ ] Execute Step 3 perception caching for combined configurations (BCS, Behavior, Re-ID).
6. [ ] Execute Runs 4, 5, 6 (Perception-enhanced single-task models).
7. [ ] Execute Run 7 (E1 Basic MTL control) and Run 8 (E3 Main Deadline MTL framework).
8. [ ] Compile comparative thesis tables and submit Phase 3 draft by September 26.
