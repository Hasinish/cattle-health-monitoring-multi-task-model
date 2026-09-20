# Phase 3 Canonical Roadmap — Vision-Based AI for Cattle Health Monitoring

**Project:** Vision-Based AI for Cattle Health Monitoring  
**Roadmap status:** CANONICAL / LOCKED FOR EXECUTION  
**Last updated:** 2026-09-20 — ScienceDB identity/leakage audit incorporated  
**Purpose:** Single source of truth for the coding/research agent.  
**Important:** Do not restart the project from zero. Do not silently change the scope without recording the decision in the research log.

---

# 0. Current Thesis Direction

## Core Phase 3 question

**Can cattle-centered visual representations reduce shortcut learning and improve robustness for Body Condition Scoring, Behavior Recognition, and Cow Re-Identification compared with generic RGB representations?**

The Phase 3 idea is not simply:

> "Add segmentation + pose + viewpoint."

The real scientific question is:

> **Which cattle-specific visual priors are useful for which task, and which information should each task preserve or ignore?**

The tasks require different information:

- **BCS:** preserve morphology/anatomy; reduce irrelevant background and identity cues.
- **Behavior:** preserve posture, motion, and sometimes scene/context.
- **Re-ID:** preserve coat pattern and individual morphology; reduce dependence on farm/camera/background.

This task-dependent conflict is one of the reasons a single hard-shared representation can produce negative transfer.

---

# 1. Scope Lock

## Core downstream tasks

1. **Body Condition Scoring**
2. **Behavior Recognition**
3. **Individual Cow Identification / Re-Identification**

## Core cattle-centered priors to investigate

1. **Cow localization / crop**
2. **Segmentation / soft mask guidance**
3. **Anatomy / pose / keypoints**
4. **Viewpoint information**
5. **Temporal information for Behavior**

## Core evaluation themes

1. Clean leakage-resistant evaluation
2. In-domain accuracy
3. Cross-camera / cross-view robustness
4. Cross-dataset / cross-environment robustness
5. Background shortcut analysis
6. Negative-transfer analysis
7. Task-specific vs shared representation design

## Not core right now

- Lameness
- Skin disease detection
- Monocular 3D cow reconstruction
- Giant Video Transformers
- Cattle foundation model training from scratch
- PCGrad/GradNorm as the main novelty
- Training a new segmentation model from scratch
- Annotating thousands of new cattle keypoints
- A giant monolithic model containing every idea at once

These can only be reconsidered after the core roadmap is stable.

---

# 2. Canonical Dataset Stack

## 2.1 BCS

### Primary
**ScienceDB**

Role:
- Main training dataset
- Main in-domain BCS benchmark
- Passage-disjoint / sequence-safe evaluation

Verified 2026-09-20:
- 53,566 images
- true biological cow IDs are not released in the available dataset metadata
- the former project claim of 10,898 cows was invalid because passage/frame identifiers were parsed as biological identities
- 5,662 reconstructed passage/sequence clusters are the defensible grouping units for split protection
- the legacy split placed 247 of 261 stereo passage blocks (94.64%) across multiple partitions
- the replacement split has zero passage-cluster overlap and zero exact-duplicate leakage across train/val/test

Main strengths:
- Large image count
- Passage/sequence structure can be reconstructed for leakage-resistant grouping
- Multiple coarse acquisition/source groups are recoverable

Main weakness:
- No verified biological cow identity across visits/days
- Narrow BCS range
- Mostly rear-view imagery

### Primary external validation
**Ruchay et al. RGB-D BCS 2026**

Role:
- External BCS generalization test
- Wider BCS range
- Multi-breed / different geography / different camera geometry
- Optional RGB vs RGB-D comparison

### Secondary external
**Dryad BCS**

Role:
- Breed/modality/domain-shift comparison
- Use carefully because ordinal label semantics may differ

---

## 2.2 Behavior

### Primary
**MmCows**

Role:
- Main behavior training/evaluation dataset
- Identity-aware behavior evaluation
- Temporal modeling experiments

Main strengths:
- Cow IDs
- Timestamps
- Multi-camera data
- Behavior labels
- Strong provenance

Main weakness:
- Only 16 cows
- Highly correlated video-derived frames
- Severe class imbalance

### Primary external validation
**CBVD-5**

Role:
- Larger-herd external behavior test
- Cross-dataset / cross-environment evaluation

Important:
- Do not claim cow-disjoint evaluation unless persistent biological cow IDs are verified.

### Optional external behavior datasets
- CVB
- XGain
- 2026 Simmental behavior dataset

Use only for compatible label intersections.

---

## 2.3 Cow ID / Re-ID

### Primary Benchmark (Approved Contingency)
**SideViewCows2026**

Role:
- Main Re-ID training and evaluation benchmark
- Nested cross-setting evaluation (parlor fixed entrance gallery -> barn handheld video query -> snapshots unconstrained query)
- Long-term temporal evaluation (>279 days span)
- Open-set and closed-set metric-learning experiments
- Ground-truth segmentation mask ablation (80,260 verified binary masks)

Why it replaces MultiCamCows2024 as primary under approved contingency:
- MultiCamCows2024 was the intended primary dataset (90 cows, 101k images, 3 ceiling cameras, 7 days). However, its official distribution endpoint (`https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip`) remained completely inaccessible due to persistent server-side connection resets from both local and Modal cloud environments.
- On 2026-09-20, an evidence-based contingency assessment (`docs/research_log/2026-09-20_multicam_contingency_assessment.md`) proved SideViewCows2026 provides scale parity (110 biological cattle, 80,260 images, mean 730 imgs/cow), long temporal depth (>9 months), and 80,260 ground-truth binary segmentation masks that directly test Phase 3's core hypothesis (segmentation-guided cattle representations).
- Formally approved by Hasin on 2026-09-20.

Status:
- 100% downloaded and verified locally (`datasets/id/external/sideviewcows2026/`).
- Deterministic protocol generation and leakage audit pending in Step 1.

### Primary External Longitudinal Re-ID Validation
**BECA-L**

Role:
- Out-of-domain longitudinal appearance change test (7+ continuous months, 134 capture dates)
- Cross-cowshed generalization (cowshed 0 vs 2 vs 3)
- Viewpoint/domain shift stress test (top-down dorsal beef cattle vs side-view dairy cattle)
- 103 biological beef cattle, 12,172 images

### External Large-Scale / Population Stress Benchmark
**BECA-D**

Role:
- Extreme-scale identity retrieval test across 5,661 unseen individuals (16,889 images)
- Evaluates open-set feature generalization to large populations

### Historical / Contingency-Excluded Intended Primary
**MultiCamCows2024**

Role:
- Intended primary Re-ID benchmark in original Phase 3 design
- Currently BLOCKED / excluded from active Phase 3 execution due to upstream archive connection resets
- Preserved in project records; may be re-evaluated if upstream access is ever restored

### Legacy Comparison
**OpenCows2020**
- Retained strictly as a legacy literature baseline for backward comparability
- Not used as primary Re-ID dataset due to unrecoverable tracklet provenance and author frame randomization
- Preserve official 496-image test set

---

# 3. Phase 3 Roadmap Overview

```text
STEP 1  Data registry + clean splits
   ↓
STEP 2  Cattle-perception feasibility audit
   ↓
STEP 3  Cache masks / pose / viewpoint
   ↓
STEP 4  Clean RGB single-task baselines
   ↓
STEP 5  Localization / segmentation ablation
   ↓
STEP 6  Anatomy / pose ablation
   ↓
STEP 7  Viewpoint ablation
   ↓
STEP 8  Temporal Behavior experiments
   ↓
STEP 9  Build the final task-conditioned P3 model
   ↓
STEP 10 Cross-domain / robustness evaluation
   ↓
STEP 11 Revisit sharing / MTL
   ↓
STEP 12 Optional cattle-specific pretraining
   ↓
STEP 13 Final repeated runs + thesis tables
```

---

# 4. STEP 1 — Data Registry and Clean Splits

**Status:** NEXT / IMMEDIATE

## Goal

Before training any serious model, every dataset must have a scientifically defensible manifest and split protocol.

## Required dataset registry

Create:

```text
datasets/dataset_registry.csv
```

Recommended fields:

```text
dataset
dataset_version
task
paper_url
dataset_url
license
download_status
local_root
sha256_or_manifest_hash
n_images
n_videos
n_cows
n_farms
n_sessions
n_cameras
cow_id_available
farm_id_available
session_id_available
tracklet_id_available
camera_id_available
timestamp_available
source_video_available
viewpoint_available
modalities
label_schema
notes
```

Unknown values must be `NA`.

Never guess metadata.

---

## 4.1 ScienceDB split

**Status: VERIFIED / UPDATED 2026-09-20**

The identity audit showed that the released ScienceDB data do **not** contain trustworthy biological cow IDs. Therefore ScienceDB must not be described as cow-disjoint.

Required protocol:

- passage-disjoint / sequence-safe train/val/test
- reconstructed passage/sequence clusters are the maximal defensible independent grouping unit
- zero passage/sequence overlap across train/val/test
- duplicate-linked sequences must remain in the same partition
- do not claim cross-cow generalization from ScienceDB
- do not infer biological identity from filename prefixes

Required checks:

```text
train_passages ∩ val_passages = ∅
train_passages ∩ test_passages = ∅
val_passages ∩ test_passages = ∅
```

Verified split:

```text
5,662 passage clusters
train: 3,963 clusters / 37,126 images
val:     849 clusters /  8,099 images
test:    850 clusters /  8,341 images
seed: 42
```

Required outputs:

```text
datasets/bcs/sciencedb/train.csv
datasets/bcs/sciencedb/val.csv
datasets/bcs/sciencedb/test.csv
datasets/bcs/sciencedb/identity_audit.csv
datasets/bcs/sciencedb/split_report.md
```

Acceptance criteria:

- [x] parsed identity semantics audited
- [x] former 10,898-cow claim rejected
- [x] zero passage/sequence overlap
- [x] split seed recorded
- [x] class distribution recorded
- [x] exact duplicate leakage prevented
- [ ] perceptual near-duplicate audit completed across finalized primary datasets
- [x] biological cow-ID limitation documented
- [x] source/location metadata status documented

---

## 4.2 MmCows split

Do not randomly split frames.

Primary grouping:

```text
cow_id
```

Secondary protection:

```text
timestamp / contiguous time block / synchronized multi-view event
```

Preferred evaluation:

- repeated Group K-Fold
- or repeated leave-k-cows-out
- report per-cow metrics
- confidence intervals must resample cows, not frames

Required outputs:

```text
datasets/behavior/mmcows/folds/
datasets/behavior/mmcows/manifest.csv
datasets/behavior/mmcows/provenance_audit.csv
datasets/behavior/mmcows/split_report.md
```

Manifest should contain:

```text
image_path
cow_id
behavior
camera_id
timestamp
source_video
time_block_id
sync_group_id
bbox
```

Acceptance criteria:

- [ ] no cow crosses train/test within a fold
- [ ] synchronized views do not leak across partitions
- [ ] time blocks protected
- [ ] behavior counts per cow recorded
- [ ] class imbalance documented
- [ ] source provenance recoverable

---

## 4.3 SideViewCows2026 split (Primary Re-ID — Approved Contingency)

Status: COMPLETE & VERIFIED (2026-09-20). Generated and verified leak-free via `scripts/build_sideview_reid_protocols.py`.

Four canonical evaluation protocols across 110 individuals, 80,260 images, and 80,260 binary masks:

### Protocol A — Cross-setting domain shift (Primary Benchmark Protocol)
- **Gallery**: Fixed camera `parlor` entrance frames (controlled lighting, consistent framing; 36,811 images across 69 cows).
- **Query 1**: Handheld video frames in `barn` (motion blur, varying camera angles; 25,260 images across 69 cows).
- **Query 2**: Unconstrained `snapshots` (outdoor/indoor, varied postures including lying down; 607 images across 63 cows).
- **Train**: Parlor frames from 41 parlor-only cows (17,582 images) available for representation learning without test query contamination.

### Protocol B — Longitudinal / cross-temporal
- Exploit `time_offset_s` (>279 days span; 0 to 634 days): earlier parlor capture interval (first 60% of sessions per cow; 35,433 images) -> later parlor capture interval (subsequent 40% of sessions; 18,960 images).
- Query long-range: barn video (25,260 images) and snapshots (607 images) recorded >200 days after parlor.
- Strict assertion: for every cow, all query sessions occur strictly after all gallery sessions (`min_query_time > max_gallery_time`).

### Protocol C — Open-set / identity-disjoint
- Stratified 70% Train (77 cows, 57,605 images) / 10% Val (11 cows, 7,225 images) / 20% Test (22 cows, 15,430 images) balanced across subset types (multi, parlor_barn, parlor_only).
- Strict assertion: train, val, and test cow identities are 100% disjoint.

### Protocol D — Closed-set identification
- Standard 110-class metric identification with sequence-safe recording protection (`dt <= 60s` session clustering; 3,604 sessions).
- Chronological partition: in-domain parlor (70% train: 40,745 images, 15% val: 7,373 images, 15% test_parlor: 6,275 images) + out-of-domain test sets (test_barn: 25,260 images, test_snapshots: 607 images).
- Strict assertion: no adjacent-frame video burst leakage detected under the implemented checks.

Required manifest:
```text
image_path
mask_path
individual_id
subset
frame_no
time_offset_s
width
height
sha256
```

Required outputs:
```text
datasets/id/sideviewcows2026/manifest.csv
datasets/id/sideviewcows2026/protocol_cross_setting.csv
datasets/id/sideviewcows2026/protocol_longitudinal.csv
datasets/id/sideviewcows2026/protocol_open_set.csv
datasets/id/sideviewcows2026/protocol_closed_set.csv
datasets/id/sideviewcows2026/split_report.md
```

Acceptance criteria:
- [x] 0 image overlap across train/gallery/query partitions
- [x] exact duplicate (SHA-256) and perceptual near-duplicate audit run (0 exact duplicates, min perceptual distance 7 bits on 10,094 sampled session anchors; no leakage detected under implemented checks)
- [x] open-set protocol strictly identity-disjoint (77 Train / 11 Val / 22 Test cows, 0 overlap)
- [x] segmentation masks verified and matched 1-to-1 with images (80,260 masks, 0 stem mismatches)
- [x] sequence provenance recorded (3,604 discrete recording sessions; no adjacent-frame leakage detected under the implemented checks)

### 4.3.1 MultiCamCows2024 Contingency Record (Historical)
- MultiCamCows2024 was the originally intended primary Re-ID benchmark (90 cows, 101k images, 3 ceiling cameras, 7 days).
- Official distribution endpoint (`data.bris.ac.uk`) failed with persistent connection resets from both local and cloud environments.
- On 2026-09-20, Hasin formally approved the contingency recommendation (`docs/research_log/2026-09-20_multicam_contingency_assessment.md`) adopting SideViewCows2026 as primary Re-ID.
- Preserved in project records; re-evaluation deferred unless upstream access is restored.

---

# 5. STEP 2 — Cattle-Perception Feasibility Audit

**Status:** READY / CURRENT (Gate 1 Cleared)

## Goal

Before building the final architecture, test whether existing tools can reliably give:

1. cow localization
2. cow segmentation
3. cattle pose/keypoints
4. viewpoint information

Use a small representative sample first.

Recommended sample:

- 100–300 ScienceDB images
- 100–300 MmCows images
- 100–300 SideViewCows2026 images

Include difficult examples:

- occlusion
- lying cows
- overhead views
- rear views
- side views
- partial crops
- multiple cows
- low light
- clutter

---

## Candidate resources

### Segmentation / localization
- CattleEyeView
- existing cattle detectors
- general modern detector/segmenter if cattle-specific model is insufficient

Provisional pretrained candidates to audit first (not selected in advance):
- detector/bounding-box prompt → SAM 2 / SAM 2.1 mask generation
- Grounded-SAM-style pipeline for automatic cow localization + mask generation
- pretrained YOLO instance-segmentation model as a faster baseline
- cattle-specific CattleEyeView-derived segmentation checkpoint if a verified usable checkpoint is available
- conventional Mask R-CNN/Detectron2 baseline only if needed for comparison

Selection rule:
- test the same representative images across candidates
- compare mask usability, failure rate, speed, and difficult-view behavior
- do not assume the heaviest model is best
- freeze the selected upstream model/checkpoint before downstream ablations

### Pose
- SuperAnimal-Quadruped
- CattleEyeView-aligned pose model
- BECA keypoints
- other verified livestock pose resources

### Viewpoint
- coarse rule/model
- MOO synthetic viewpoint supervision
- simple classifier if needed

---

## Required outputs

```text
docs/audits/phase3_perception_feasibility.md
artifacts/perception_audit/
```

For every sampled image store:

```text
image_id
dataset
bbox_ok
mask_ok
pose_ok
view_ok
pose_confidence
failure_reason
manual_notes
```

Acceptance criteria:

- [ ] segmentation visually usable on most representative samples
- [ ] pose quality manually inspected
- [ ] viewpoint categories are definable
- [ ] failure modes documented
- [ ] no decision to retrain pose/segmentation before this audit is complete

---

# 6. STEP 3 — Cache Upstream Cattle Information

**Status:** BLOCKED BY STEP 2

## Goal

Run perception tools once and reuse the same outputs across all downstream ablations.

Cache:

```text
bbox
soft_mask
pose_coordinates
pose_confidence
optional_pose_heatmaps
coarse_viewpoint
```

Recommended structure:

```text
data/processed/perception/
    sciencedb/
    mmcows/
    sideviewcows2026/
```

Each cache must store:

```text
source_image_path
source_hash
model_name
model_version
checkpoint_hash
preprocessing_version
output_path
```

Acceptance criteria:

- [ ] exact upstream model version recorded
- [ ] outputs deterministic/reproducible
- [ ] all experiments use the same cached perception outputs
- [ ] missing/failed predictions explicitly marked
- [ ] no silent zero-filling of failed pose coordinates

---

## Operational implementation plan for cattle perception

This clarifies execution without changing the scientific roadmap.

**First modeling task after Gate 1:** build and validate the cattle-perception pipeline before official downstream task training.

```text
image / video
    ↓
cow localization
    ↓
segmentation / soft mask
    ↓
anatomy / pose / keypoints
    ↓
coarse viewpoint
    ↓
cache reproducible cattle-centered outputs
```

Execution strategy:

1. Develop and debug the perception pipeline on a small representative subset first.
2. Prefer pretrained/frozen tools; do not train segmentation or pose from scratch unless the feasibility audit shows that existing tools fail.
3. Once a perception component is accepted, freeze its exact code/config/checkpoint/version.
4. Copy the same frozen perception code/checkpoints to the BCS, Behavior, and Re-ID execution environments.
5. Each task environment may download only its own task datasets and generate its own local perception cache.
6. A shared cloud dataset store is optional, not required for the initial parallel execution plan.
7. All downstream ablations must reuse the same cached upstream outputs for that dataset/version.
8. Smoke-test scripts on low-cost/local hardware before paid cloud or high-end GPU runs.

Storage/versioning rule:

```text
Git
→ code + configs + manifests + split files + small logs/results

Persistent GPU/cloud volume or local disk
→ datasets + cached masks/pose/viewpoint + large checkpoints

Research PC backup
→ final/best checkpoints + final manifests/results
```

Do not put large datasets or large `.pt` / `.pth` checkpoints directly in normal Git history.

---

# 7. STEP 4 — Clean RGB Single-Task Baselines

**Status:** BLOCKED BY STEP 1

## Goal

Establish trusted baselines before adding cattle-centered information.

---

## 7.1 BCS baseline

```text
RGB
→ image encoder
→ ordinal head
```

Metrics:

- MAE
- ordinal/rank agreement
- balanced accuracy
- macro-F1 if useful

Important:
- Do not report class-index MAE as physical BCS units unless converted correctly.

---

## 7.2 Behavior baseline

```text
single RGB frame
→ image encoder
→ behavior classifier
```

Metrics:

- Macro-F1
- Balanced Accuracy
- Per-class recall
- Per-cow F1
- cow-level confidence interval

---

## 7.3 Re-ID baseline

Prefer:

```text
RGB
→ image encoder
→ normalized embedding
→ metric / ID loss
```

Metrics may include:

- Top-1
- mAP
- CMC / retrieval metrics
- cross-day / cross-camera metrics

Do not rely only on closed-set softmax classification.

Acceptance criteria:

- [ ] all baselines use clean split protocols
- [ ] config stored
- [ ] seed stored
- [ ] Git commit stored
- [ ] checkpoint stored
- [ ] metrics stored
- [ ] no external test set used for hyperparameter tuning

Parallel execution is allowed after Gate 1 and the shared perception pipeline is frozen:

```text
BCS environment      → ScienceDB / BCS datasets
Behavior environment → MmCows / behavior datasets
Re-ID environment    → SideViewCows2026 / Re-ID datasets
```

These environments should use the same Git-tracked code and the same frozen upstream perception versions, while keeping task datasets and large caches local to the relevant environment.

---

# 8. STEP 5 — Localization / Segmentation Ablation

**Status:** BLOCKED BY STEPS 3–4

## Goal

Determine whether forcing the model to focus on cattle helps.

Run staged conditions.

### A0 — Original RGB

```text
full image
```

### A1 — Cow bounding-box crop

Tests localization alone.

### A2 — Crop + soft segmentation guidance

Tests whether mask information improves cattle-focused representation.

### A3 — Foreground-only cow

Background removed.

Use as a diagnostic, not automatically the deployable solution.

### A4 — Cow stream + separate context stream

Most relevant for Behavior.

```text
cow-centered stream
+
low-resolution/full-scene context stream
```

Scientific questions:

- Does localization help?
- Does soft mask guidance help beyond cropping?
- Does complete background removal improve robustness?
- Does Behavior lose useful semantic context when background is removed?
- Does Re-ID depend on environmental shortcuts?

Acceptance criteria:

- [ ] same backbone/training schedule across ablations
- [ ] trainable parameter counts recorded
- [ ] no unfair extra capacity unless controlled
- [ ] background-only diagnostic considered for Re-ID/Behavior
- [ ] cross-domain impact recorded, not only in-domain accuracy

---

# 9. STEP 6 — Anatomy / Pose Ablation

**Status:** BLOCKED BY STEP 5 AND POSE FEASIBILITY

## Goal

Test whether explicit anatomy contributes real information.

Core comparisons:

### B0
Visual representation only.

### B1
Visual + correct pose.

### B2
Visual + shuffled pose.

### B3
Visual + confidence-aware pose.

Optional representation comparison:

```text
pose coordinates
vs
pose heatmaps
```

Possible pose features:

```text
normalized coordinates
angles
distances
body-axis geometry
pose heatmaps
small pose MLP
small GCN
```

Task hypotheses:

- BCS: likely useful
- Behavior: likely useful
- Re-ID: uncertain / possibly smaller benefit

Important:

If correct pose and shuffled pose perform similarly, anatomy is probably not the reason for the gain.

Acceptance criteria:

- [ ] shuffled-pose control implemented
- [ ] pose confidence preserved
- [ ] missing keypoints masked correctly
- [ ] parameter-matched control considered
- [ ] pose quality separately audited

---

# 10. STEP 7 — Viewpoint Ablation

**Status:** BLOCKED BY STEP 3

## Goal

Determine whether explicit orientation information improves robustness.

Recommended conditions:

### C0
Normal augmentation only.

### C1
Add viewpoint token.

### C2
Add shuffled viewpoint token.

### C3
View-conditioned gating / FiLM.

Recommended coarse viewpoint categories:

```text
rear
rear-oblique
side
front-oblique
front
top/overhead
```

Exact taxonomy can differ by dataset.

Important rule:

**Camera ID is not viewpoint.**

Camera ID can leak:

- floor
- pen
- lighting
- background
- fixed farm scene

Acceptance criteria:

- [ ] viewpoint label definition documented
- [ ] camera ID not used as a shortcut proxy
- [ ] shuffled-view control implemented
- [ ] evaluate especially under cross-view/cross-camera conditions
- [ ] if no measurable benefit, do not force viewpoint into final model

---

# 11. STEP 8 — Temporal Behavior Experiments

**Status:** BLOCKED BY CLEAN MmCows PROVENANCE

## Goal

Turn Behavior from a frame-only task into a proper temporal task.

Required ladder:

### D0 — Single frame

Baseline.

### D1 — Multiple frames + average pooling

Tests whether simply seeing more frames helps.

### D2 — Multiple frames + TCN

Tests learned temporal order efficiently.

### D3 — Multiple frames + GRU/LSTM

Alternative temporal baseline.

### D4 — Pose sequence only

Tests skeleton motion.

### D5 — RGB temporal + pose temporal fusion

Tests complementarity.

Optional later:

### D6
Pretrained 3D CNN / SlowFast / VideoMAE benchmark.

Do not start with D6.

Scientific interpretation:

```text
D0 → D1
= benefit of more visual evidence

D1 → D2/D3
= benefit of actual temporal modeling

D4 → D5
= whether RGB/context adds information beyond pose motion
```

Acceptance criteria:

- [ ] clips built from provenance-safe time blocks
- [ ] no frame leakage
- [ ] D1 average-pooling baseline included
- [ ] temporal window length recorded
- [ ] per-cow metrics reported
- [ ] heavy video models only considered after lightweight models justify temporal complexity

---

# 12. STEP 9 — Build the Final P3 Architecture

**Status:** BLOCKED UNTIL STEPS 5–8 PRODUCE RESULTS

## Principle

Do not decide the final architecture in advance.

Only keep components that survive ablation.

Likely moderate architecture:

```text
RGB / video
    ↓
frozen cow localizer
    ↓
cow-centered RGB + soft mask
    ↓
image encoder
    ↓
optional pose encoder
+
optional viewpoint token
+
optional context stream
    ↓
task-conditioned gated fusion
    ↓
 ┌────────────┬───────────────┬─────────────┐
 │            │               │             │
BCS        Behavior          Re-ID
ordinal     temporal          metric
head        TCN/GRU           embedding
```

Expected task-specific usage:

### BCS
- cow-centered appearance
- morphology
- useful anatomy
- possibly viewpoint
- little/no context

### Behavior
- cow-centered appearance
- pose
- temporal information
- controlled scene context
- possibly viewpoint

### Re-ID
- coat pattern
- individual morphology
- soft mask / cattle focus
- viewpoint if useful
- context strongly suppressed

Acceptance criteria:

- [ ] every component justified by earlier ablation
- [ ] no unused complexity
- [ ] task pathways explicitly documented
- [ ] final model is reproducible from config

---

# 13. STEP 10 — Cross-Domain / Robustness Evaluation

**Status:** BLOCKED BY FINAL SINGLE-TASK MODELS

## Goal

Test whether the model learned cattle rather than dataset shortcuts.

---

## 13.1 BCS external evaluation

Primary:

```text
Train: ScienceDB
Test zero-shot: Ruchay 2026
```

Secondary:

```text
Dryad
```

Questions:

- Does cattle-centered representation improve external transfer?
- Does RGB-D provide extra value on Ruchay after RGB-only testing?

---

## 13.2 Behavior external evaluation

Primary:

```text
Train: MmCows
External test: CBVD-5
```

Optional:

```text
CVB
XGain
Simmental 2026
```

Important:

Only map behavior labels with defensible semantic overlap.

Do not force ambiguous mappings.

---

## 13.3 Re-ID external evaluation

Primary:

```text
Train / develop: SideViewCows2026 (Approved Contingency)
External: BECA-L (Longitudinal) & BECA-D (Population Scale)
```

Additional / Historical:

```text
OpenCows2020 (legacy literature comparison only)
MultiCamCows2024 (blocked upstream; excluded from primary execution)
```

Questions:

- cross-camera robustness?
- cross-setting robustness?
- long-term appearance robustness?
- open-set transfer?

Acceptance criteria:

- [ ] external datasets never used for core hyperparameter tuning
- [ ] label mapping documented
- [ ] zero-shot and fine-tuned results clearly separated
- [ ] same-domain and cross-domain results both reported

---

# 14. STEP 11 — Revisit Sharing / MTL

**Status:** LATE PHASE ONLY

## Goal

Only after strong single-task cattle-centered systems exist, test whether sharing helps.

Required comparison:

### E0
Three independent single-task models.

### E1
Fully hard-shared backbone.

### E2
Shared early layers + private late layers.

### E3
Shared backbone + task adapters/gates.

### E4
Hard sharing + PCGrad.

### E5
Hard sharing + GradNorm.

Track:

```text
task metrics
gradient cosine similarity
gradient norms
training stability
negative transfer
shared-layer feature similarity if feasible
```

Scientific interpretation:

If:

```text
E1 < E0
E2/E3 recover
E4/E5 do not recover
```

then the main problem is likely **what information is shared**, not merely optimizer conflict.

Important:

PCGrad/GradNorm are controls, not the thesis novelty.

---

# 15. STEP 12 — Optional Cattle-Specific Pretraining

**Status:** STRETCH ONLY

Only attempt if core roadmap succeeds and compute/time remain.

Possible idea:

```text
same cow crop
    ├─ original background
    └─ background randomized/blurred/swapped

encoder(original)
≈
encoder(background-changed)
```

Optional auxiliary objectives:

- pose consistency
- mask prediction
- viewpoint prediction
- contrastive cattle embedding

Goal:

Reduce environmental shortcut dependence.

Do not call this a "cattle foundation model."

---

# 16. STEP 13 — Final Experiment Protocol

Development:

```text
1 seed
```

Final headline experiments:

```text
3 seeds
```

Suggested:

```text
42
123
2026
```

Report:

```text
mean ± standard deviation
```

Also record:

- dataset manifest hash
- split hash
- Git commit
- config
- seed
- checkpoint
- trainable parameters
- inference latency/FLOPs if measured
- metric confidence intervals where appropriate

Do not run every minor ablation 3 times.

Use 3 seeds only for final comparisons.

---

# 17. Experiment Naming Convention

Recommended IDs:

```text
BCS_A0_RGB
BCS_A1_CROP
BCS_A2_MASK
BCS_B1_POSE
BCS_C1_VIEW

BEH_D0_FRAME
BEH_D1_AVG
BEH_D2_TCN
BEH_D3_GRU
BEH_D4_POSESEQ
BEH_D5_RGBPOSE

REID_A0_RGB
REID_A1_CROP
REID_A2_MASK
REID_A3_FGONLY
REID_C1_VIEW

MTL_E0_SINGLE
MTL_E1_HARD
MTL_E2_PARTIAL
MTL_E3_ADAPTER
MTL_E4_PCGRAD
MTL_E5_GRADNORM
```

Every experiment directory should contain:

```text
config.yaml
metrics.json
train.log
checkpoint.pt
git_commit.txt
dataset_manifest_hash.txt
split_hash.txt
notes.md
```

---

# 18. Logging Rules

Every meaningful action must be logged.

Use existing repo system:

```text
memory/state.md
memory/history.md
memory/index.md

docs/research_log/YYYY-MM-DD_<topic>.md
docs/audits/

datasets/<task>/
```

For every dataset audit:

Record:

- source URL
- download date
- license
- checksum
- exact local counts
- discrepancies vs paper
- split decision
- leakage risks

For every experiment:

Record:

- hypothesis
- dataset version
- split version
- model/config
- seed
- result
- failure notes
- next decision

Do not create a second competing memory system.

---

# 19. Hard Scientific Rules

## Never

- tune on the test set
- threshold-tune on the test set
- randomly split adjacent video frames
- call tracking IDs permanent cow IDs
- fabricate cow IDs
- claim zero leakage without evidence
- claim masks/pose/viewpoint help before experiments
- claim external generalization from same-farm random splits
- call camera ID viewpoint
- call PCGrad/GradNorm novel
- re-add lameness to core Phase 3 without an explicit decision
- call a dataset public without verifying access
- call a model real-time without measuring it
- call a cattle encoder a foundation model without scale/breadth evidence

## Always

- version split files
- preserve source provenance
- record seed
- record Git commit
- save checkpoints
- save metrics
- log failures
- run duplicate/near-duplicate checks
- separate verified fact from inference
- keep external test sets frozen

---

# 20. Go / No-Go Gates

## Gate 1 — Data Ready
**Status: PASSED & LOCKED (2026-09-20)**

Proceed to perception/baselines only if:

- [x] ScienceDB burst-group-disjoint / sequence-safe split verified (repaired 2026-09-20; 0 cross-burst leakage)
- [x] MmCows grouped evaluation defined (canonical split + 4-fold GroupKFold suite verified 2026-09-20; 0 cow overlap)
- [x] SideViewCows2026 protocols generated and duplicate audit passed (4 canonical protocols verified 2026-09-20; 0 exact duplicates, min perceptual distance 7 bits)
- [x] required duplicate / near-duplicate and protocol leakage checks pass across all primary splits

---

## Gate 2 — Perception Feasible

Proceed to pose/view experiments only if:

- [ ] segmentation is usable
- [ ] pose is usable enough to interpret
- [ ] viewpoint labels are defensible
- [ ] failure rates documented

If pose fails badly:
- do not force pose into the thesis
- continue with segmentation + viewpoint + temporal experiments

---

## Gate 3 — Component Value

A component enters the final model only if:

- it improves a relevant metric, robustness condition, or interpretability objective
- the gain survives appropriate controls
- the result is not obviously due only to extra parameters

---

## Gate 4 — Final Model

Build the consolidated final architecture only after:

- [ ] segmentation ablation complete
- [ ] pose ablation complete or rejected
- [ ] viewpoint ablation complete or rejected
- [ ] temporal Behavior ladder complete

---

## Gate 5 — MTL

Do not begin MTL until:

- [ ] clean single-task final models exist
- [ ] negative transfer can be measured fairly

---

# 20.1 Operational Compute / Storage Plan

This is an execution note, not a scientific contribution.

Current practical strategy:

- use the low-end local GPU for smoke tests, path checks, tensor-shape checks, checkpoint/resume tests, metric tests, and 1–2 epoch tiny-subset runs
- reserve stronger GPUs / paid cloud for real preprocessing, full training, ablations, and final repeated runs
- keep one task per cloud environment where practical to simplify data movement
- use persistent volumes for large checkpoints/caches and Git for reproducible code/config state
- back up final/best checkpoints and final results outside the cloud environment

Do not spend paid GPU time debugging basic script failures that can be reproduced locally.

---

# 21. Current Exact Position

## Current state

**STEP 1 — Data Registry and Clean Splits: COMPLETE (2026-09-20)**
**GATE 1: PASSED & LOCKED**

Completed deliverables:

1. [x] Build canonical dataset registry (`datasets/dataset_registry.csv` verified)
2. [x] Audit ScienceDB identity semantics and repair burst-group split (`datasets/bcs/sciencedb/` verified leak-free)
3. [x] Retrieve/index Ruchay 2026 metadata and manifest
4. [x] Rebuild MmCows grouped evaluation protocol with cow, time-block, and synchronized-view protection (`datasets/behavior/mmcows/folds/` verified)
5. [x] Document MultiCamCows2024 upstream blocker and formally adopt SideViewCows2026 contingency (`docs/research_log/2026-09-20_multicam_contingency_assessment.md`)
6. [x] Download/index SideViewCows2026 (80,260 images + masks verified on disk)
7. [x] Download/index BECA-D / BECA-L (29,061 images verified on disk)
8. [x] Verify/index CBVD-5 raw data and identity metadata (887 videos, 206,100 frames verified on disk)
9. [x] Run automatic duplicate / near-duplicate audit across ScienceDB, MmCows, OpenCows
10. [x] Build and verify deterministic SideViewCows2026 primary Re-ID protocols and run duplicate/near-duplicate audit (`datasets/id/sideviewcows2026/` verified)

---

# 22. Agent Instruction — What To Do Next

Gate 1 is fully CLEARED. Step 1 is COMPLETE.

Immediate next task:

> **STEP 2 — Cattle-Perception Feasibility Audit**

Required next deliverable:

```text
docs/audits/phase3_perception_feasibility.md
```

Then update:
- `memory/state.md`
- `docs/research_log/README.md`

Then update:

```text
memory/state.md
```

with:

- datasets downloaded
- verified metadata
- split status
- leakage status
- unresolved issues
- next action

---

# 23. Canonical One-Line Roadmap

```text
DATA
→ verify cattle segmentation/pose/viewpoint
→ cache cattle knowledge
→ clean RGB baselines
→ test crop/mask
→ test pose
→ test viewpoint
→ add temporal Behavior
→ build final task-conditioned cattle-centered model
→ cross-domain testing
→ revisit MTL
→ optional cattle-specific pretraining
→ final repeated runs
```

---

# 24. Canonical Thesis Story

## Phase 2

```text
generic RGB
→ hard-shared CNN
→ BCS + Behavior + Lameness + Cow ID
```

Problems:

- negative transfer
- weak lameness data
- leakage
- background shortcut concern
- limited novelty

## Phase 3

```text
cattle-centered perception
→ localization / mask
→ anatomy / pose
→ viewpoint
→ task-appropriate context/time
→ task-conditioned representations
→ BCS / Behavior / Re-ID
```

The central contribution is not merely a new classifier.

The thesis investigates:

> **What information should a cattle vision model preserve, what should it ignore, and how does that change across morphology, behavior, and identity tasks?**

This roadmap remains the canonical plan until evidence from an experiment or dataset audit justifies a documented change.

---

# 25. High-Reasoning Review Resource — GPT-6 Astra

The user currently has approximately **12 GPT-6 Astra messages** available as a scarce thesis-support resource.

Use Astra selectively for high-value review tasks rather than routine coding/chat. Priority uses:

- full Phase 3 roadmap / methodology red-team review
- cattle-perception module design review (segmentation + anatomy/pose + viewpoint)
- dataset split / leakage audit review
- experiment design and ablation sanity checks
- interpretation of final results and unsupported-claim detection
- final paper / defense review

Do not treat Astra output as ground truth. Any suggested change to the canonical roadmap must still be justified by evidence, logged, and treated as a proposed update before adoption.
