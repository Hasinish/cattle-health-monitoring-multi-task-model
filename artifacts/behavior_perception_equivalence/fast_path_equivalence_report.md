# Scientific Equivalence Audit: Serial Reference vs Fast Caching Path

**Date:** 2026-09-23 21:23:30  
**Target GPU:** NVIDIA L40S  
**Candidate Sequences Evaluated:** 40 (Certified balanced smoke subset: 30 train, 10 val)  
**Total Frames Evaluated:** 304 (38 retained sequences * 8 frames)  
**Equivalence Status:** **PASS (CERTIFIED)**  

---

## 1. Executive Summary
Validated the single-GPU optimized fast perception caching pipeline against the certified serial reference implementation on the exact 40-sequence balanced smoke subset. The fast path incorporates:
1. **Monotonic single-pass Beef video decoding**: eliminates repeated `cap.set` seeking calls while producing bit-identical video frames.
2. **Batched RT-DETR-L detection**: feeds all 8 frames in a single batch forward pass (`conf=0.25`, COCO cow `class=19`) with independent per-frame largest-box selection.
3. **Validated SAM 2.1 Small prompt execution**: exact GT bbox prompts for CVB and exact A5 / center-point fallback prompts for Kaggle Beef.

The scientific equivalence gate requirements were met with zero discrepancies in dataset partitioning decisions:
- **Retained Sequence IDs**: 100% agreement (38/38)
- **Excluded Sequence IDs**: 100% agreement (2/2)
- **Sampled Frame Indices**: 100% bit-identical across all 320 frames
- **Prompt Strategy**: 100% identical (192 CVB GT bbox, 93 Beef A5, 19 Beef center fallback)
- **Minimum Mask IoU**: **1.000000** (Target: >= 0.999)
- **Mean Mask IoU**: **1.000000**
- **Exact Mask Equality Rate**: **100.00%**
- **Maximum RGB Pixel Diff**: **0**

---

## 2. Quantitative Equivalence Gate Metrics

| Metric | Target | Serial Reference | Fast Optimized | Equivalence Result |
| :--- | :--- | :--- | :--- | :--- |
| **Retained Sequences** | 38 | 38 | 38 | **100% Match** |
| **Excluded Sequences** | 2 | 2 | 2 | **100% Match** |
| **Sampled Frame Indices** | 100% Match | 320 / 320 | 320 / 320 | **100% Match** |
| **CVB GT Prompt Frames** | 192 | 192 | 192 | **100% Match** |
| **Beef A5 Prompt Frames** | 93 | 93 | 93 | **100% Match** |
| **Beef Fallback Frames** | 19 | 19 | 19 | **100% Match** |
| **Minimum Mask IoU** | >= 0.999 | 1.000000 | **1.000000** | **PASS** |
| **Mean Mask IoU** | >= 0.999 | 1.000000 | **1.000000** | **PASS** |
| **Exact Mask Equality** | High | 100% | **100.00%** | **PASS** |
| **Max RGB Pixel Diff** | <= 1 | 0 | **0** | **PASS** |

---

## 3. Throughput & Speedup Comparison on NVIDIA L40S

| Metric | Serial Reference | Fast Optimized | Gain / Speedup |
| :--- | :--- | :--- | :--- |
| **Wall-clock Runtime (40 seqs)** | 17.04 s | **13.32 s** | **1.28x Faster** |
| **Sec / Attempted Candidate** | 0.426 s/cand | **0.333 s/cand** | -0.093 s |
| **Attempted Candidates / Min** | 140.8 cand/min | **180.1 cand/min** | +39.3 cand/min |
| **Frames / Second** | 17.8 fps | **22.8 fps** | **1.28x Throughput** |
| **Projected Full Cache Time (4,465)** | 0.53 hours | **0.41 hours** | -0.12 hours |

---
