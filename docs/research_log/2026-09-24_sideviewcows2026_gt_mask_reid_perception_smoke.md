# Run 6 SideViewCows2026 GT-Mask Perception-Enhanced Re-ID Smoke Certification

**Date:** 2026-09-24
**Status:** PIPELINE PREPARED AND LOCAL SMOKE-CERTIFIED; FULL TRAINING AND PROTOCOL A EVALUATION PENDING
**Compute:** Local NVIDIA GeForce GTX 1050 Ti (4 GB VRAM)
**Task:** Phase 3 Run 6 — SideViewCows2026 perception-enhanced Re-ID

## 1. Executive Summary

Run 6 was implemented as a controlled **GT/oracle segmentation-guided Re-ID representation** using SideViewCows2026's paired target-cow masks. The pipeline derives a target box from each GT mask, expands it by a deterministic 5% proportional margin, applies the identical crop to RGB and mask, resizes them to 224x224, and concatenates ImageNet-normalized RGB with an unnormalized binary mask channel to produce `[B,4,224,224]`.

A local-only 2-epoch smoke test on 64 training and 64 validation images passed all plumbing checks on the GTX 1050 Ti. All 128 real RGB-mask pairs were valid, the forward/backward path and finite loss were verified, the 512-D embeddings had unit L2 norm, and checkpoint reload reproduced logits bit-identically (maximum absolute difference `0.00000000`). No Protocol A gallery, barn-query, or snapshot-query image was loaded or evaluated. This smoke accuracy is not a thesis result.

## 2. Context & Motivation

Run 3 established the matched generic RGB control on canonical Protocol A:

- Barn -> Parlor Rank-1: 58.64%; mAP: 38.32%.
- Snapshots -> Parlor Rank-1: 38.88%; mAP: 27.05%.

Run 6 tests whether a cattle-centered, foreground-guided input improves robustness relative to that RGB control. SideViewCows2026 supplies a verified 1:1 target-cow GT mask for every image. Using those masks in this deadline configuration avoids silent target switching in crowded scenes and isolates the scientific effect of cattle-focused representation. It is an oracle experimental condition, not an automatic deployment segmentation pipeline; SAM did not generate these masks.

## 3. Forensic Findings & Data

### 3.1 Canonical protocol preservation

- Protocol A representation-learning identities: 41.
- Protocol A held-out evaluation identities: 69, with zero identity overlap.
- Protocol D restricted full training set: 12,753 images across the same 41 cows.
- Protocol D restricted full validation set: 2,683 images across the same 41 cows.
- Smoke subset: 64 train + 64 validation images, deterministically sampled with seed 2026.
- Smoke identities physically accessed: 39, all a strict subset of the 41 representation-learning cows.
- Held-out Protocol A gallery/query images physically loaded: 0.
- Protocol splits and identity mapping were not modified.

### 3.2 Pair and mask integrity

Every smoke row was decoded and checked before training:

- RGB paths present/readable: 128/128.
- Matching mask paths present/readable: 128/128.
- Filename stem and cow-directory provenance aligned: 128/128.
- RGB and mask dimensions identical: 128/128.
- Non-empty, non-full target masks: 128/128.
- Positive, in-bounds mask-derived crops: 128/128.
- Identical RGB/mask crop coordinates: 128/128.
- Nearest-neighbor resized mask values: exactly `{0.0, 1.0}`.
- Invalid pairs: 0.

Example pre-resize crops include `708x453` (cow 565), `705x548` (cow 507), and `858x537` (cow 545). Across the smoke set, crop widths ranged from 508 to 2,431 pixels and crop heights from 362 to 1,092 pixels before resizing.

### 3.3 Tensor and gradient assertions

- Input tensor: `[8,4,224,224]`.
- Raw ResNet-18 feature: `[8,512]`.
- L2-normalized embedding: `[8,512]`.
- Training logits: `[8,41]`.
- Observed embedding norm range: `0.9999999404` to `1.0000000000`.
- Forward pass: passed.
- Backward pass with finite gradients: passed.
- Cross-entropy loss finite in every batch: passed.

### 3.4 Parameter-controlled architecture

The implementation subclasses the Run 3 model and changes only `conv1` from 3 to 4 channels. RGB weights copy the ImageNet-pretrained values exactly; the fourth channel is initialized from the mean of the three RGB kernels.

- Run 3 RGB model trainable parameters: 11,197,545.
- Run 6 RGB+mask model trainable parameters: 11,200,681.
- Exact additional capacity: 3,136 parameters.

All other architecture and training choices remain matched: ResNet-18, 512-D L2 embedding, `Linear(512,41)`, cross-entropy, AdamW (`lr=1e-4`, `weight_decay=1e-4`), cosine scheduler, seed 2026, and 224x224 resolution. No pose, viewpoint, attention, adapters, temporal model, ArcFace, or triplet-loss redesign was introduced.

### 3.5 Local smoke execution

Exact command:

```powershell
python scripts/train_sideview_reid_perception.py --smoke --smoke-samples 64 --epochs 2 --batch-size 8 --workers 0 --output-dir artifacts/reid_perception_smoke
```

- GPU: NVIDIA GeForce GTX 1050 Ti, 4 GB.
- Script-measured total pipeline runtime: 16.96 seconds (excluding Python import/startup).
- Training/checkpoint section: 11.98 seconds.
- Epoch 1: train loss 3.9235; validation loss 3.6387.
- Epoch 2: train loss 2.8123; validation loss 3.4641.
- Checkpoint saved and loaded into a fresh model.
- Maximum absolute pre/post-reload logit difference: `0.00000000` (bit-identical).

Tiny-smoke accuracy is intentionally not interpreted because the smoke set is small, omits some identities in each partition, and exists solely to certify data/model plumbing.

## 4. Architectural Decisions & Action Plan

1. **Adopt for Run 6:** `[RGB crop, binary GT target-mask channel]` with a 5% mask-derived crop margin.
2. **Oracle terminology is mandatory:** this condition uses SideView target-cow GT masks and must not be described as automatic segmentation or SAM output.
3. **Aligned augmentation:** horizontal flipping is synchronized between RGB and mask; color jitter is RGB-only; RGB uses bilinear resizing and the mask uses nearest-neighbor resizing.
4. **No foreground multiplication:** RGB is not multiplied by the mask; the mask remains an explicit fourth channel.
5. **Preserve fair comparison:** full Run 6 must retain Run 3's protocol, identities, optimizer, objective, scheduler, resolution, checkpoint-selection metric, and retrieval implementation.
6. **Current boundary:** no full 30-epoch training, Modal launch, or held-out Protocol A retrieval evaluation was performed.

## 5. Artifacts & File Registry

| Artifact | Path | Purpose |
| :--- | :--- | :--- |
| Run 6 trainer | `scripts/train_sideview_reid_perception.py` | GT-mask paired transforms, 4-channel model, smoke/full training, and matched retrieval path |
| Smoke metrics | `artifacts/reid_perception_smoke/reid_perception_smoke_metrics.json` | Integrity, protocol isolation, shapes, parameters, losses, runtime, and checkpoint round-trip evidence |
| Visual audit sheet | `artifacts/reid_perception_smoke/gt_mask_crop_contact_sheet.jpg` | Six recording-session examples: original RGB, GT binary mask, crop+mask overlay, final crop |
| Git-tracked audit copy | `docs/audits/assets/reid_perception_smoke/gt_mask_crop_contact_sheet.jpg` | Permanent byte-identical copy of the visual audit sheet |
| Local checkpoints | `artifacts/reid_perception_smoke/reid_perception_{best,latest}.pth` | Local verification only; ignored by Git under the repository checkpoint policy |

## 6. Next Steps

- Prepare/verify a low-cost Modal wrapper only when the full Run 6 execution is authorized.
- Execute the full 30-epoch Run 6 training in cloud compute, not locally.
- Evaluate the frozen best checkpoint once on Protocol A barn and snapshot queries against the parlor gallery.
- Compare the matched Run 6 retrieval metrics directly with Run 3.

**Run 6 GT-mask perception pipeline is smoke-certified only; full training and held-out Protocol A evaluation were NOT launched.**
