# Run 5 Behavior Perception Fast Caching Equivalence Certification (NVIDIA L40S)

**Date:** 2026-09-24  
**Task:** Phase 3 Run 5 (Behavior Perception Caching Optimization)  
**Execution Profile:** `tigerwood693` (Modal Cloud)  
**Target Hardware:** NVIDIA L40S GPU (1x L40S, 4 CPUs, 16 GB RAM)  
**Modal App ID:** `ap-LnhAP3RcL9DtfILgobryt5`  
**Git HEAD:** `0c7395fb084dfe8e598a5c06de3760dbc3be29a1`  

---

## 1. Executive Summary
Designed, verified, and certified an optimized single-GPU fast perception caching pipeline for Phase 3 Run 5 (Behavior Perception-Enhanced Temporal Model) without modifying the scientific perception policy. Across the certified 40-sequence balanced smoke subset (320 frames; 192 CVB, 128 Kaggle Beef), the fast path achieved **100.00% exact binary mask equality (304/304 masks)**, **Global Minimum Mask IoU = 1.000000**, **Maximum RGB Crop Pixel Diff = 0**, and **100% agreement on dataset partitioning (38 retained, 2 excluded)**. Throughput on NVIDIA L40S improved from 140.8 to 180.1 candidates/minute (17.8 to 22.8 fps), projecting full Train+Val production caching (4,465 candidates) in approximately 24.6 minutes (0.41h). Canonical `test.csv` (809 candidates) remained strictly isolated and was never loaded, sampled, or evaluated.

---

## 2. Technical Optimizations

### 2.1 Monotonic Single-Pass Beef Video Decoder
- **Reference Bottleneck**: The reference serial code invoked OpenCV `cap.set(CAP_PROP_POS_FRAMES, f_idx)` repeatedly for each of the T=8 frames, forcing repeated container seeking and forward keyframe re-decoding.
- **Fast Path Solution**: `decode_beef_video_monotonic` opens the video container exactly once, advances through frames strictly in ascending forward order using sequential `cap.grab()` for skipped frames and `cap.read()` for requested target frames.
- **Verification**: Verified 100% bit-identical decoded frames across all 304 frames (`Maximum RGB Pixel Diff = 0`).

### 2.2 Batched RT-DETR-L Detection for Kaggle Beef
- **Reference Bottleneck**: 8 independent sequential forward calls to RT-DETR-L per video clip.
- **Fast Path Solution**: Feeds all 8 monotonically decoded frames as a single batch list to RT-DETR-L (`conf=0.25`, COCO cow `class=19`) in one forward pass on the L40S GPU.
- **Policy Invariance**: Preserves independent per-frame largest-box selection and condition A5 (box + center point) vs fallback (112, 112 center point) logic with zero policy approximation.

### 2.3 Pure FP32 Precision for Deterministic Equivalence
- **Forensic Discovery**: PyTorch 2.x on Ada Lovelace GPUs defaults `torch.backends.cudnn.allow_tf32 = True`. Because cuDNN selects different Tensor Core tiling kernels for batch size 8 vs batch size 1, TF32 (10-bit mantissa) introduced a 0.000023-pixel numerical difference which crossed the rounding threshold on borderline coordinates, causing a single frame to exhibit IoU = 0.998380.
- **Remediation**: Explicitly enforced `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False` during caching, guaranteeing bit-level deterministic convergence and elevating minimum mask IoU from 0.998380 to **1.000000 (100.00% exact binary equality)** across all 304 frames.

---

## 3. Quantitative Equivalence Gate Verification

| Metric | Scientific Gate Target | Serial Reference | Fast Optimized Path | Gate Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Retained Sequences** | Exact Match (38) | 38 | 38 | **PASS (100%)** |
| **Excluded Sequences** | Exact Match (2) | 2 | 2 | **PASS (100%)** |
| **Sampled Frame Indices** | 100% Match | 320 / 320 | 320 / 320 | **PASS (100%)** |
| **CVB GT Prompt Frames** | 192 | 192 | 192 | **PASS (100%)** |
| **Beef A5 Prompt Frames** | 93 | 93 | 93 | **PASS (100%)** |
| **Beef Fallback Frames** | 19 | 19 | 19 | **PASS (100%)** |
| **Exact Binary Mask Match** | High | 100% | **304 / 304 (100.00%)** | **PASS** |
| **Minimum Mask IoU** | >= 0.999 | 1.000000 | **1.000000** | **PASS** |
| **Mean Mask IoU** | >= 0.999 | 1.000000 | **1.000000** | **PASS** |
| **Max RGB Crop Pixel Diff** | 0 | 0 | **0** | **PASS** |

---

## 4. Single-GPU L40S Throughput Benchmarks

| Metric | Serial Reference | Fast Optimized Path | Gain / Speedup |
| :--- | :--- | :--- | :--- |
| **Wall-clock Runtime (40 seqs)** | 17.04 s | **13.32 s** | **1.28x Faster** |
| **Sec / Attempted Candidate** | 0.426 s/cand | **0.333 s/cand** | -0.093 s/cand |
| **Attempted Candidates / Min** | 140.8 cand/min | **180.1 cand/min** | +39.3 cand/min |
| **Frames / Second** | 17.8 fps | **22.8 fps** | **+28.1% Throughput** |
| **Projected Full Cache Time (4,465)** | 0.53 hours (~31.7 min) | **0.41 hours (~24.6 min)** | -7.1 minutes |

---

## 5. Entrypoints & Artifacts
- **Equivalence Verification Entrypoint**:
  ```bash
  modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::verify_equivalence
  ```
- **60-Second Fast L40S Benchmark Entrypoint**:
  ```bash
  modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::benchmark_cache_l40s_fast --time-limit-sec 60
  ```
- **Equivalence Report**: `artifacts/behavior_perception_equivalence/fast_path_equivalence_report.json`
- **Markdown Audit**: `docs/audits/2026-09-24_fast_path_equivalence_report.md`
