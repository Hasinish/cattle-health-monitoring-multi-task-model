# Research Log — 2026-09-20: MultiCamCows2024 Contingency Assessment & Evidence-Based Re-ID Protocol Proposal

## 1. Why a Contingency is Necessary
In the Phase 3 Canonical Roadmap (`phase3_canonical_roadmap.md`), **MultiCamCows2024** was designated as the intended primary Individual Cow Identification / Re-Identification (Re-ID) benchmark. The roadmap codified Go / No-Go **Gate 1 (Data Ready)** as:
> *"MultiCamCows indexed and protocols generated, OR a documented contingency is adopted if upstream access remains unavailable."*

Gate 1 is currently open and blocking Phase 3 Step 2 (Cattle-Perception Feasibility Audit) solely because MultiCamCows2024 is physically unavailable. The other primary datasets (ScienceDB for BCS and MmCows for Behavior) have been downloaded, forensically audited, repaired, and locked into verified leak-free splits.

Proceeding with model development requires resolving the Re-ID data foundation. Leaving MultiCamCows2024 as an unresolved blocker indefinitely halts the thesis pipeline. This document provides a forensic, read-only scientific evaluation of all locally accessible Re-ID candidate datasets to formulate a single, evidence-based contingency proposal for user review.

---

## 2. Evidence That MultiCamCows2024 is Upstream Blocked
The official dataset publication endpoint for MultiCamCows2024 is hosted by the University of Bristol Data Repository:
- **Landing Page**: `https://data.bris.ac.uk/data/dataset/2inu67jru7a6821kkgehxg3cv2` (HTTP 200 OK)
- **Direct Archive URL**: `https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip` (Reported size: 36.5 GiB)

Repeated download attempts conducted on 2026-09-20 produced the following verified forensic failure states:
1. **Local Windows 11 Client**:
   - Web browsers (Chrome, Edge) fail immediately with `ERR_CONNECTION_RESET`.
   - Python `urllib` / `requests` fail with `TimeoutError: The read operation timed out` or `ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host`.
   - `curl -I` and `wget` terminate with `curl: (35) Recv failure: Connection was reset`.
2. **Remote Cloud Environments (Modal Datacenters)**:
   - Automated cloud downloaders launched from high-bandwidth US-East and EU-West Modal containers fail identically: `RemoteDisconnected: Remote end closed connection without response`.
3. **Repository-Wide Failure**:
   - Other large dataset archives hosted under `data.bris.ac.uk/datasets/tar/` fail with identical server-side TCP resets, indicating institution-level egress blocking, server-side maintenance, or IP-whitelisting on the Bristol archive server.
4. **Mirror Investigation**:
   - Comprehensive searches across Zenodo, Hugging Face, Kaggle, Figshare, Dryad, and Academic Torrents confirm that no authorized secondary mirrors or re-hosted distributions exist.

**Conclusion**: MultiCamCows2024 is completely inaccessible upstream. A formal contingency is mandatory to clear Gate 1.

---

## 3. Verified Comparison Table of Accessible Re-ID Candidates
A physical inspection of the local filesystem was conducted across all accessible candidate datasets:
- **SideViewCows2026**: `datasets/id/external/sideviewcows2026/`
- **BECA-L**: `datasets/id/external/beca/BECA-L/`
- **BECA-D**: `datasets/id/external/beca/BECA-D/`
- **OpenCows2020**: `datasets/id/opencow2020/` & `datasets/id/opencow2020-DatasetNinja/`

| Dimension | MultiCamCows2024 (Blocked) | SideViewCows2026 (Local) | BECA-L (Local) | BECA-D (Local) | OpenCows2020 (Local Legacy) |
|---|---|---|---|---|---|
| **Local Status** | **NOT_FOUND (Blocked)** | **AVAILABLE (100% on disk)** | **AVAILABLE (100% on disk)** | **AVAILABLE (100% on disk)** | **AVAILABLE (100% on disk)** |
| **Local Size** | 0 GB | 23.33 GB | Included in 18.91 GB BECA | Included in 18.91 GB BECA | 0.85 GB |
| **Total Images** | 101,329 (reported) | **80,260** (+ 80,260 masks) | **12,172** | **16,889** | **4,736** |
| **Verified Biological Cows** | 90 (reported) | **110** | **103** | **5,661** | **46** |
| **Images / Cow (Mean ± Std)** | ~1,125 (reported) | **729.6 ± 315.2** | **118.2 ± 88.9** | **3.0 ± 0.0** (exactly 3/cow) | **103.0 ± 29.4** |
| **Images / Cow (Range)** | Not reported | **195 to 2,058** | **32 to 417** | **3 (train) / 2–3 (val)** | **40 to 180** |
| **Camera Geometry / View** | Top-down (3 ceiling cams) | **Side-view (Right flank)** | **Top-down / Dorsal** | **Top-down / Dorsal** | **Mixed (Oblique/Side/Rear)** |
| **Multi-Camera / Setting** | 3 synchronized cameras | **3 nested settings** (Parlor, Barn, Snapshots) | **3 cowsheds** (cowshed 0, 2, 3) | Single acquisition setup | Single handheld camera |
| **Temporal Span** | 7 consecutive days | **>279 days (>9 months)** | **229 days (>7 months, 134 dates)** | Single snapshot event | Stripped / Unknown |
| **Temporal Provenance** | Timestamps & tracklets | `time_offset_s` (high-res seconds) | ISO dates (`YYYY-MM-DD`) | None | None (unordered crops) |
| **Segmentation Masks** | None | **80,260 ground-truth binary masks** | None (OBB/keypoint crops) | None (AABB crops) | None |
| **Cattle Breed** | Holstein | Holstein Friesian (major), Simmental, Jersey, Brown Swiss | Beef cattle (Simmental, Angus crosses) | Beef cattle (extensive population) | Holstein |
| **Tracklet / Sequence Safe** | Yes (author tracklets) | **Yes** (chronological frame ordering) | **Yes** (chronological date sequence) | No (static 3-shot clusters) | **No** (randomized crops; near-duplicate leakage) |

---

## 4. Protocol Feasibility Analysis for Each Candidate

### A. SideViewCows2026
- **Identity Count & Distribution**: 110 biological cattle with massive sample depth (195 to 2,058 images per individual; mean 730 images).
- **Subsets & Structure**:
  - `parlor`: 54,393 images across 110 cows (fixed entrance camera, highly consistent framing, 156-day span).
  - `barn`: 25,260 images across 69 cows (handheld video inside housing, varying camera motion, 236-day span).
  - `snapshots`: 607 images across 63 cows (indoor/outdoor photos, lying down, varied angles, 279-day span).
- **Nested Cross-Setting Design**: The 63 individuals in `snapshots` are a subset of the 69 in `barn`, which are a subset of the 110 in `parlor`.
- **Protocol Feasibility**:
  1. *Cross-Setting Domain Shift Protocol*: Gallery formed from fixed `parlor` entrance frames -> Query evaluated on handheld `barn` video and unconstrained `snapshots`.
  2. *Longitudinal / Cross-Month Protocol*: Gallery on earlier months (days 0–100) -> Query on late months (days 200–279).
  3. *Open-Set Protocol*: 80 cows for training/embedding learning, 30 held-out cows for zero-shot gallery/query rank-1 retrieval and mAP.
  4. *Closed-Set Identification*: 110-class metric classification.
- **Hypothesis Alignment (Crucial)**: Includes 80,260 ground-truth binary segmentation masks. This directly tests the core thesis question: *"Does cattle-specific segmentation eliminate background shortcut learning in cattle Re-ID?"*

### B. BECA-L (Longitudinal Beef Cattle)
- **Identity Count & Distribution**: 103 biological beef cattle, 12,172 images (mean 118 images/cow, range 32–417).
- **Subsets & Structure**: Divided into 3 physical facilities (`cowshed0`: 6,686 imgs, `cowshed2`: 2,158 imgs, `cowshed3`: 3,328 imgs). Overlapping cow populations cross facilities (23 cows shared between shed 0 & 2; 32 cows shared between shed 0 & 3; 25 cows shared between shed 2 & 3).
- **Temporal Span**: 134 distinct capture dates spanning 2023-12-10 to 2024-07-26 (more than 7 continuous months).
- **Protocol Feasibility**:
  1. *Cross-Cowshed Protocol*: Train on `cowshed0` -> Query on `cowshed2` and `cowshed3`.
  2. *Longitudinal Protocol*: Train on winter/spring (Dec–Mar) -> Query on summer (Apr–Jul) to stress seasonal coat and body weight changes.
  3. *Open-Set Protocol*: Hold out 23 cows for zero-shot evaluation.
- **Limitations**: Only 12,172 images (substantially smaller than MultiCam's 101k or SideView's 80k); top-down dorsal view only; lacks pixel-level segmentation masks for all frames.

### C. BECA-D (Large-Scale Diversity Benchmark)
- **Identity Count & Distribution**: 5,661 individual cattle, 16,889 images.
- **Crucial Structural Constraint**: Every cow in the training set (5,361 cows) has **EXACTLY 3 images**. The validation set has 300 cows with 2–3 images each.
- **Protocol Feasibility**:
  - Unusable as a primary Re-ID training benchmark for deep metric learning: 3 images per class cannot support robust intra-class variance modeling, temporal tracklet splitting, cross-camera testing, or longitudinal generalization.
  - **Ideal Role**: Frozen out-of-domain scale stress test evaluating zero-shot gallery/query retrieval on 5,661 unseen individuals.

### D. OpenCows2020 (Legacy Baseline)
- **Identity Count & Distribution**: 46 cows, 4,736 images.
- **Audit Findings**:
  - The dataset consists of unordered bounding box crops stripped of video timestamps.
  - A comprehensive duplicate audit proved that near-identical frames of the same cow permeate train, val, and test due to the original authors' random frame shuffling.
  - While our contiguous frame-index blocking heuristic eliminated exact-duplicate overlap, tracklet-safe separation cannot be guaranteed.
- **Conclusion**: OpenCows2020 **CANNOT** be promoted to primary Re-ID. It must remain strictly a legacy comparison baseline.

---

## 5. Scientific Trade-Off Analysis

| Strategy | Primary Re-ID Dataset | External Validation Benchmarks | Scientific Strengths | Scientific Risks / Vulnerabilities |
|---|---|---|---|---|
| **Strategy 1 (Recommended)** | **SideViewCows2026** (80k imgs, 110 cows, side-view, 80k masks) | **BECA-L** (longitudinal dorsal), **BECA-D** (5,661 cows scale), **OpenCows2020** (legacy) | - Matched scale to MultiCam (80k vs 101k imgs, 110 vs 90 cows).<br>- Ground-truth segmentation masks available for all images (perfectly evaluates Phase 3 hypothesis).<br>- Rigorous nested cross-setting evaluation (parlor vs barn vs snapshots).<br>- External validation is NOT lost: BECA-L and BECA-D provide out-of-domain dorsal beef cattle evaluation. | - Primary Re-ID shifts from top-down ceiling cameras to side-view corridor/parlor cameras.<br>- Both primary Re-ID and external validation benchmarks must be maintained. |
| **Strategy 2** | **BECA-L** (12k imgs, 103 cows, dorsal view, 7 months) | **SideViewCows2026** (side-view external), **BECA-D** (scale), **OpenCows2020** (legacy) | - Maintains dorsal/overhead viewpoint similar to MultiCam.<br>- 134 dates and 3 cowsheds provide strong cross-date/cross-shed protocols. | - Image count is small (12k images vs 101k in MultiCam).<br>- Underutilizes the 80k images and ground-truth segmentation masks of SideViewCows2026 during representation learning.<br>- Lacks pixel-level segmentation masks for representation training. |
| **Strategy 3** | **OpenCows2020** (4.7k imgs, 46 cows) | **SideViewCows2026**, **BECA-L**, **BECA-D** | - Zero roadmap changes required. | - Scientifically bankrupt: promotes a flawed dataset with unrecoverable tracklet provenance and known near-duplicate frame contamination.<br>- Contradicts previous audit decisions. |

---

## 6. ONE Recommended Contingency Configuration: **Strategy 1**

We formally recommend **Strategy 1**: **Promote SideViewCows2026 to Primary Re-ID Benchmark, and Establish BECA-L and BECA-D as the External Re-ID Validation & Stress Benchmarks.**

### Why This is the Strongest Scientific Configuration:
1. **Dataset Scale Parity**: MultiCamCows2024 had 90 cows and 101k images. SideViewCows2026 provides 110 cows and 80,260 images — virtually identical statistical power.
2. **Direct Alignment with Phase 3 Hypothesis**: SideViewCows2026 provides **80,260 ground-truth binary segmentation masks**. Our core thesis question is whether cattle-specific visual priors (segmentation masks, anatomy/pose) eliminate background shortcut learning in multi-task networks. SideView allows us to train and ablate ground-truth masks vs. predicted masks vs. RGB baselines in Re-ID.
3. **Multi-Setting & Longitudinal Depth**: The nested architecture (`parlor` fixed gallery -> `barn` handheld query -> `snapshots` unconstrained query) spanning >9 months provides realistic, industrially relevant cross-domain retrieval.
4. **Preserved External Benchmark Rigor**: Promoting SideView does not eliminate external validation. Instead, **BECA-L** (103 cows, 7-month longitudinal tracking, top-down view) and **BECA-D** (5,661 cows, large-scale one-shot retrieval) provide out-of-domain external testing across a different breed category (beef cattle) and camera geometry (top-down dorsal).

---

## 7. What Dataset Roles Would Change Under This Proposal

| Dataset | Current Roadmap Role | Proposed Contingency Role | Rationale |
|---|---|---|---|
| **MultiCamCows2024** | Primary Re-ID Benchmark | **FROZEN / CONTINGENCY EXCLUDED** (Documented upstream outage) | Endpoint unreachable; contingency adopted to clear Gate 1. |
| **SideViewCows2026** | Primary External Re-ID Validation | **PRIMARY RE-ID BENCHMARK** | 110 cows, 80k images, 80k masks, 3 nested settings, >9 months span. |
| **BECA-L** | Longitudinal Stress Test | **PRIMARY EXTERNAL LONGITUDINAL RE-ID BENCHMARK** | 103 beef cattle, 7 months (134 dates), top-down view, cross-cowshed evaluation. |
| **BECA-D** | Scale/Pretraining Stress Test | **EXTERNAL SCALE STRESS BENCHMARK** | 5,661 cows, 16.8k images; evaluates open-set feature generalization to large populations. |
| **OpenCows2020** | Legacy Baseline Only | **LEGACY BASELINE ONLY (UNCHANGED)** | Retained strictly for literature comparability; never used as primary. |

---

## 8. What External & Stress Evaluations Remain Available Under This Proposal

With SideViewCows2026 promoted to Primary Re-ID, the evaluation framework retains three distinct, frozen external benchmarks:
1. **Out-of-Domain Longitudinal Stress (BECA-L)**:
   - Evaluates whether representations trained on dairy cows (side-view) transfer to beef cattle (dorsal view) tracked across 7 months (134 dates).
2. **Extreme-Scale Population Stress (BECA-D)**:
   - Evaluates open-set zero-shot retrieval across 5,661 unseen beef cattle individuals.
3. **Legacy Literature Baseline (OpenCows2020)**:
   - Evaluates backward comparability on the 496-image official test set using our contiguous frame-index protocol.

---

## 9. Scientific Claims: What We Could and Could NOT Claim

### What We COULD Claim:
- Comprehensive multi-task learning across BCS (ScienceDB), Behavior (MmCows), and Re-ID (SideViewCows2026).
- Evaluation on 110 biological cattle with >80,000 images under controlled gallery (`parlor`) vs. handheld video query (`barn`) vs. unconstrained posture query (`snapshots`).
- Empirical validation of segmentation-guided representation learning using 80,260 ground-truth masks.
- Sequence-safe and burst-safe evaluation with zero cross-partition leakage.
- True out-of-domain external generalization to a completely different camera viewpoint (dorsal) and breed category (beef cattle) via BECA-L and BECA-D.

### What We Could NOT Claim:
- We could **NOT** claim primary Re-ID evaluation on MultiCamCows2024.
- We could **NOT** claim top-down ceiling camera re-identification as our primary setup (primary is side-view; dorsal is external validation).
- We could **NOT** claim zero leakage on OpenCows2020 beyond our documented contiguous frame heuristic.

---

## 10. Exact Proposed Roadmap Correction (For Future User Approval)
If Hasin approves this proposal, the following minimal, atomic changes will be made in Step 1:
1. **`phase3_canonical_roadmap.md` Section 2.3 & 4.3**:
   - Record MultiCamCows2024 under documented contingency due to upstream server outage.
   - Transition SideViewCows2026 to Primary Re-ID.
   - Designate BECA-L and BECA-D as the formal external Re-ID validation benchmarks.
   - Replace the MultiCam protocol requirements with SideViewCows2026 protocols (Protocol A: Cross-setting parlor-to-barn/snapshots; Protocol B: Longitudinal cross-month; Protocol C: Open-set 80/30 identity-disjoint; Protocol D: Closed-set 110-identity).
2. **`datasets/dataset_registry.csv`**:
   - Update `task` column for SideViewCows2026 to `Re-ID (Primary)` and BECA-L to `Re-ID (Primary External Validation)`.
3. **Gate 1 Clearance**:
   - Generate deterministic SideViewCows2026 protocol CSVs and run exact/near-duplicate audit.
   - Formally mark Gate 1 as **PASSED / COMPLETE**.
   - Proceed to **Phase 3 Step 2: Cattle-Perception Feasibility Audit**.

---

## 11. Artifacts & Audit Registry
- Manifest: `datasets/id/external/sideviewcows2026/manifest.csv` (80,260 rows verified).
- BECA Manifests: `datasets/id/external/beca/BECA-L/reid/` (12,172 images) and `BECA-D/` (16,889 images).
- OpenCows Protocol: `datasets/id/opencow2020/manifest.csv` (4,736 images).
- Contingency Assessment: `docs/research_log/2026-09-20_multicam_contingency_assessment.md`.
