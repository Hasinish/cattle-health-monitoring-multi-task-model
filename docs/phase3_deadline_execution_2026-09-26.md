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
1. BCS RGB Baseline (Done ✅)   2. Behavior RGB Baseline (Done ✅) 3. Re-ID RGB Baseline (Done ✅)
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
- **Results:** Real MAE = 0.1848 BCS units, Within +-1 Class Accuracy (Acc@1) = 86.74%, Balanced Accuracy = 40.41%, Macro-F1 = 0.4110. Global best checkpoint saved at Epoch 8. Test set strictly frozen and isolated.

#### Run 2: Behavior RGB Single-Frame Baseline
- **Status:** **COMPLETE & 100% CERTIFIED** ✅
- **Details:** Canonical CVB + Kaggle Beef protocol (`datasets/behavior/cvb_beef/`; Train 3,785, Val 680, Test 809; Seed 2026).
- **Architecture:** ResNet-18, 224x224 RGB, CrossEntropyLoss, AdamW.
- **Critical Input Rule:** Target cow cropped via GT bbox on deterministic midpoint frame for CVB; midpoint frame used directly for single-cow Kaggle Beef. Walking is CVB-only.
- **Results:** Full 30-epoch training executed on Modal L40S (`tigerwood693`, App `ap-ZrBKKvGcVzM7IB1AYMs2LH`). Global best checkpoint captured at Epoch 25. Final evaluation on held-out sequence-safe test split (809 unseen clips across 44 groups): **Overall Accuracy: 88.88%** (719/809), **Balanced Accuracy: 71.72%**, **Macro-F1: 0.7413**, **Test Loss: 0.5312**. Per-class F1: Lying 0.9552, Feeding 0.9225, Drinking 0.8430, Standing 0.7684, Walking 0.2174.

#### Run 3: Re-ID RGB Single-Task Baseline
- **Status:** **COMPLETE & 100% CERTIFIED** ✅
- **Details:** SideViewCows2026 primary protocol (`protocol_cross_setting.csv`; 41 training identities / 69 held-out evaluation identities).
- **Architecture:** ResNet-18 -> 512-D L2-normalized embedding -> Linear(512, 41) classifier with CrossEntropyLoss.
- **Results:** Full 30-epoch training executed on Modal L40S (`tigerwood697`, App `ap-2v7eXL7tv414v518NkLBPN`). Best checkpoint captured at Epoch 13 (Val Top-1 Acc: 98.73%, Val Bal Acc: 98.65%, Val Macro-F1: 0.9871). Canonical Protocol A held-out retrieval evaluation on 69 unseen cows across 62,678 total images against 36,811 parlor gallery images:
  - **Query Barn -> Gallery Parlor** (25,260 queries, 69 unseen cows): **Rank-1 / Top-1 Accuracy: 58.64%**, **Rank-5: 78.19%**, **Rank-10: 83.72%**, **mAP: 38.32%**
  - **Query Snapshots -> Gallery Parlor** (607 handheld queries, 63 unseen cows): **Rank-1 / Top-1 Accuracy: 38.88%**, **Rank-5: 57.17%**, **Rank-10: 64.58%**, **mAP: 27.05%**

#### Run 4: BCS Perception-Enhanced Single-Task Model
- **Configuration:** **ONE main combined configuration:**
  `RGB + cattle crop/localization + SAM 2.1 soft foreground mask`.
- **Exclusion of Pose:** SuperAnimal-Quadruped pose is **EXCLUDED** from the deadline BCS run.
  - **Empirical Grounding:** In the Step 2.3 pose audit (`docs/research_log/2026-09-20_cattle_pose_feasibility_audit.md`), zero-shot pose predictions on reviewed ScienceDB rear-view chute images were evaluated as visually bad and anatomically unreliable (hallucinating cranial keypoints on occluded heads behind the torso). Furthermore, the 39-keypoint SuperAnimal quadruped schema completely lacks pelvic and skeletal landmarks (*tuber coxae* / hooks, *tuber ischiadicum* / pins, pelvic hollows) essential for assessing bovine body condition.
- **Control to Beat:** Run 1 BCS RGB baseline (`Real MAE: 0.1848`, `Acc@1: 86.74%`, `Bal Acc: 40.41%`, `Macro-F1: 0.4110`).

#### Run 5: Behavior Perception-Enhanced Temporal Model
- **Configuration:** **ONE main configuration:**
  `cattle-centered RGB/crop + segmentation/mask + ONE lightweight temporal model (TCN)`.
- **Exclusion of Pose:** Do NOT force SuperAnimal pose into the deadline Behavior run because current pose evidence is insufficient/unreliable for the primary behavior setup (Step 2.3 audit showed RT-DETR missed recumbent cows in bedding, keypoints distorted in stall bars, zero keypoint ground truth).
- **Temporal Constraint:** Do NOT run GRU, LSTM, VideoMAE, or SlowFast before deadline. TCN provides the fastest, most stable, and least computationally intensive temporal baseline.

#### Run 6: Re-ID Perception-Enhanced Single-Task Model
- **Configuration:** **ONE main configuration:**
  `cattle crop + foreground/soft-mask representation`.
- **Pose Policy:** Retain pose strictly as an **OPTIONAL future/isolated ablation**. Although parlor side views appeared visually plausible in the Step 2.3 audit (72.7%–77.2% points inside mask), pose must NOT be automatically included in the deadline Re-ID configuration to maintain focus on coat pattern preservation and background suppression.
- **Control to Beat:** Run 3 Re-ID RGB baseline (Barn Rank-1: 58.64%, Snapshots Rank-1: 38.88%).

#### Run 7: Basic Final MTL Control (E1: Hard Sharing)
- **Configuration:** Fully hard-shared vision trunk with bifurcated task heads for BCS, Behavior, and Re-ID.
- **Scientific Role:** Serves as the primary control to detect, measure, and document negative transfer across disparate task objectives.

#### Run 8: Main Deadline MTL Model (E3: Modular / Adapters / Task-Private Pathways)
- **Configuration:** Shared trunk + task-specific private pathways, lightweight adapter modules, or cross-attention gates.
- **Scientific Role:** The primary thesis contribution for mitigating negative transfer, allowing BCS (morphology), Behavior (posture/motion), and Re-ID (coat pattern) to preserve specialized features while benefiting from unified multi-task representations.

---

## 3. Cattle Perception Module Architecture & Flow

The definition of the cattle perception pipeline adapted for deadline execution:

```text
image / video
    ↓
cow localization / crop (RT-DETR-L)
    ↓
segmentation / soft mask (SAM 2.1)
    ↓
cache reproducible cattle-centered representations
```

*(Pose remains an optional future ablation; Viewpoint is verified on real cattle but not yet validated for cross-domain transfer to downstream tasks).*

---

## 4. Viewpoint Status for Deadline Execution

The empirical status of bovine viewpoint estimation is updated as follows:

1. **Real-Cattle Viewpoint Classifier Exists & Certified:**
   A real-cattle 3-class (`front`, `side`, `rear`) viewpoint classifier was trained by fine-tuning the MOO-pretrained ResNet-18 on authentic bounding-box crops (`datasets/viewpoint/self_clean_v1_rtdetr_crop/`; 616 train crops across 616 duplicate groups).
   - Evaluated on the strictly frozen, unseen held-out test split (`test.csv`: 131 crops across 131 unique duplicate groups):
     - **Test Accuracy:** **86.26%** (113 / 131 correct)
     - **Balanced Test Accuracy:** **84.96%**
     - **Macro-F1 Score:** **0.8573**
     - **Per-Class Recall:** `front`: 89.83%, `side`: 72.73%, `rear`: 92.31%
     - **Per-Class Precision:** `front`: 82.81%, `side`: 85.71%, `rear`: 92.31%
   - Stored on Modal persistent volume `viewpoint-checkpoints` (`viewpoint_resnet18_real_best.pth`).
2. **Operational Boundary for Runs 4–6:**
   - **Do NOT automatically insert viewpoint into Runs 4–6.**
   - While the viewpoint model is certified on its own held-out test split, it has **not yet been validated for cross-domain transfer** to ScienceDB (overhead/rear chute), CVB+Beef (barn/feedlot CCTV), or SideViewCows2026 (parlor/barn chutes).
   - Silent injection without cross-domain verification risks propagating viewpoint misclassifications into downstream representations.
3. **Absolute Boundary:**
   **NEVER substitute camera ID or dataset session as a proxy for viewpoint.** Camera ID encodes farm and background shortcuts, directly violating anti-shortcut learning principles.

---

## 5. Scientific Claim Boundaries

Because the deadline plan evaluates targeted perception configurations rather than exhaustive combinatorial sweeps:

- **PERMISSIBLE CLAIM:**
  > *"The cattle-centered perception-enhanced representation (localization + soft mask) improved (or did not improve) out-of-distribution robustness and accuracy compared with generic RGB baselines across BCS, Behavior, and Re-ID."*
- **IMPERMISSIBLE CLAIMS (Unless an isolated ablation run is executed):**
  > *"Segmentation individually caused the observed improvement."*  
  > *"Anatomy/pose keypoints individually caused the observed improvement."*
- **PRETRAINING ATTRIBUTION BOUNDARY (MANDATORY):**
  > **Do NOT claim that the numerical difference (+58.89%) between the synthetic-only MOO diagnostic (27.37% on 95 samples) and the fine-tuned real test result (86.26% on 131 samples) proves MOO synthetic pretraining caused the improvement.**
  > The synthetic-only diagnostic benchmark and the new real held-out test split are **different evaluation populations**. Furthermore, no same-split ImageNet-initialized control was evaluated. Isolating the specific causal benefit of MOO pretraining over generic ImageNet transfer requires a controlled comparison on the identical train/test split.

---

## 6. Comprehensive Deferred Experiment Registry

The following experiments from the canonical roadmap are formally **DEFERRED** until after the September 26 deadline:

| Roadmap Component | Deferred Experiments | Reason for Deferral |
| :--- | :--- | :--- |
| **Segmentation / Localization (Step 5)** | Full A0–A4 component ladder (A0 full, A1 box, A2 soft mask, A3 hard mask, A4 background-only) | Time and compute constraint; evaluate main A2 soft mask representation first |
| **Anatomy / Pose (Step 6)** | Full B0–B3 controls (B1 heatmaps, B2 keypoint coords, B3 shuffled keypoint controls) | Evaluated as an optional future ablation; excluded from deadline BCS/Behavior runs due to anatomical unreliability on rear/lying postures |
| **Viewpoint (Step 7)** | Full C0–C3 viewpoint experiments (C1 one-hot, C2 embedding, C3 viewpoint-conditioned norm) | Real-cattle classifier trained (86.26% test acc), but cross-domain transfer to ScienceDB/CVB/SideView is not yet validated |
| **Temporal Behavior (Step 8)** | Full D0–D6 ladder: D2 LSTM, D3 GRU, D4 pose-only sequence, D5 two-stream, D6 VideoMAE/SlowFast | TCN selected as sole deadline temporal architecture for stability and speed |
| **Consolidated MTL (Step 11)** | E2 exhaustive partial-sharing layer search, E4 PCGrad, E5 GradNorm sweeps | E0 (single), E1 (hard), and E3 (adapters/private) form the minimum defensible set |
| **Pretraining (Step 12)** | Cattle-specific masked autoencoding / self-supervised pretraining | Optional stretch goal; non-essential for primary thesis defense |
| **Combinatorial Sweeps** | Exhaustive N-way Cartesian product of all perception x temporal x MTL options | Scientifically secondary to proving core MTL feasibility and negative-transfer mitigation |

---

## 7. Immediate Action Checklist

1. [x] Step 4.1 BCS RGB Single-Task Baseline: 100% COMPLETE & CERTIFIED (Real MAE: 0.1848, Acc@1: 86.74%).
2. [x] Step 4.2 Behavior RGB Single-Task Baseline: 100% COMPLETE & CERTIFIED (Test Acc: 88.88%, Macro-F1: 0.7413).
3. [x] Step 4.3 Re-ID RGB Single-Task Baseline: 100% COMPLETE & CERTIFIED (Protocol A Barn Rank-1: 58.64%, Snapshots Rank-1: 38.88%).
4. [x] Real Viewpoint 3-Class Classifier: 100% COMPLETE & CERTIFIED (Test Acc: 86.26%, Bal Acc: 84.96%, Macro-F1: 0.8573 on held-out test.csv).
5. [ ] **IMMEDIATE NEXT ACTION:** Prepare and execute Run 4 (BCS Perception-Enhanced Model: RGB + cattle crop + SAM 2.1 soft mask) on ScienceDB (`tigerwood697`).
6. [ ] Execute Run 5 (Behavior Perception-Enhanced Model: RGB crop + soft mask + TCN) on CVB + Kaggle Beef (`tigerwood693`).
7. [ ] Execute Run 6 (Re-ID Perception-Enhanced Model: cattle crop + soft mask) on SideViewCows2026 (`tigerwood697`).
8. [ ] Execute Run 7 (E1 Basic MTL control) and Run 8 (E3 Main Deadline MTL framework).
9. [ ] Compile comparative thesis tables and submit Phase 3 draft by September 26.

