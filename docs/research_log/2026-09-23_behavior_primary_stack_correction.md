# Behavior Primary Stack Correction — CVB + Kaggle Beef

**Date:** 2026-09-23  
**Status:** APPROVED / CANONICAL ROADMAP CORRECTION  
**Decision authority:** Explicit user approval after completed repository audits and review of the Behavior dataset evidence.

## 1. Executive Summary

Phase 3 Behavior Recognition will no longer use MmCows as the sole primary training dataset. The canonical primary Behavior training stack is now **CVB + Kaggle Beef Cattle Behavior**, because both provide dense continuous video suited to the thesis's temporal Behavior objective. MmCows is retained as an **external identity-aware / cow-disjoint stress test**, and CBVD-5 becomes secondary external validation.

This is an evidence-based roadmap correction. Historical audit documents are preserved unchanged as records of the earlier decision; their recommendation that MmCows remain primary is superseded by this entry and the updated canonical roadmap.

## 2. Evidence Supporting the Correction

### CVB
- 225,829 1080p frames across 502 cuts from 66 recoverable source videos.
- 30 FPS continuous 15-second clips.
- 1,163,408 annotated cattle bounding boxes.
- Dense temporal motion is available, including Walking.
- Critical limitation: no biological cow IDs.
- Critical split issue: the official AVA split has source-video leakage; 32/36 validation source videos have sister cuts in training.
- Therefore CVB is usable only after a new **source-video-grouped** split is built.

### Kaggle Beef Cattle Behavior
- 4,337 verified single-cow MP4 clips.
- Approximately 11.84 hours of continuous 25 FPS video.
- Official classes: ruminate, lie, stand, eat, drink.
- Walking is absent.
- Biological cow IDs are not released; ByteTrack IDs are ephemeral.
- Therefore the dataset must be split by **recording session/source video**, never by random clip or frame.

### MmCows
- 213,686 behavior crops across 16 biological cows with strong identity/multi-camera provenance.
- Cow-disjoint evaluation is scientifically useful.
- However, the available Phase 3 representation is scan-sampled at 15-second intervals rather than dense video.
- The 1,000-image visual reassessment also found substantial occlusion and viewpoint ambiguity in the reviewed MmCows subset.
- Therefore MmCows remains valuable, but its strongest role is now **external identity-aware generalization testing**, not primary dense-motion training.

## 3. Canonical 5-Class Mapping

### CVB
- `resting-standing` -> Standing
- `resting-lying` -> Lying
- `grazing` -> Feeding
- `drinking` -> Drinking
- `walking` -> Walking

Exclude from the initial canonical baseline:
- `ruminating-standing`
- `ruminating-lying`
- `hidden`
- `other`
- `grooming`
- `none`
- `running`

### Kaggle Beef
- `stand` -> Standing
- `lie` -> Lying
- `eat` -> Feeding
- `drink` -> Drinking

Exclude `ruminate` from the initial canonical 5-class baseline because it has no direct shared class under the selected taxonomy.

### Walking caveat
Kaggle Beef contains no Walking. Walking samples in the primary combined training stack therefore originate from CVB only. This creates a potential dataset-source shortcut. The project must:
1. report per-dataset and per-class metrics,
2. avoid interpreting pooled Walking accuracy as cross-domain evidence,
3. evaluate Walking on held-out CVB source videos, and
4. use the frozen MmCows cow-disjoint split as an external Walking/generalization stress test.

## 4. Leakage-Safe Protocol Requirements

### CVB
Group all cuts by recoverable `source_video_id`. No source video may cross train/validation/test.

### Kaggle Beef
Group all derived clips by recording session and, where recoverable, source surveillance video. No session/source video may cross train/validation/test.

Random frame splitting and random clip splitting are prohibited.

## 5. Dataset Roles After Correction

- **Primary Behavior training stack:** CVB + Kaggle Beef Cattle Behavior
- **External identity-aware validation:** MmCows
- **Secondary external validation:** CBVD-5
- **Optional later datasets:** XGain / Simmental only after independent audit and defensible label mapping

## 6. Governance Consequences

- Step 1 / Gate 1 is reopened **for Behavior only** because the new primary stack does not yet have its canonical combined manifest and leakage-safe split files.
- Existing BCS and Re-ID split decisions remain unchanged.
- Existing MmCows split artifacts remain valid and frozen for external validation.
- Existing perception audits on MmCows are not automatically transferable to CVB or Kaggle Beef; any new-primary perception assumptions require a small direct sanity check before large-scale caching.

## 7. Immediate Next Task

Build only the combined Behavior protocol:

```text
datasets/behavior/cvb_beef/manifest.csv
datasets/behavior/cvb_beef/train.csv
datasets/behavior/cvb_beef/val.csv
datasets/behavior/cvb_beef/test.csv
datasets/behavior/cvb_beef/label_mapping.csv
datasets/behavior/cvb_beef/split_report.md
```

Do not start full Behavior training until this protocol is verified.

## 8. Updated Artifacts

- `phase3_canonical_roadmap.md`
- `docs/phase3_canonical_roadmap.md`
- `memory/state.md`
- `memory/index.md`
- `datasets/dataset_registry.csv`
- `docs/research_log/README.md`
- `docs/research_log/2026-09-23_behavior_primary_stack_correction.md`
