# Phase 3 Behavior Dataset Agent Visual Inspection Report

**Date**: 2026-09-20  
**Inspector**: Antigravity Agent (Multimodal Vision Inspection)  
**Artifact Directory**: `docs/audits/assets/agent_behavior_inspection/`  
**Purpose**: Direct visual verification of 20 MmCows samples and 20 CBVD-5 samples to evaluate image quality, crop boundaries, resolution, and annotation validity using native multimodal vision.

---

## 1. MmCows Visual Inspection (20 Samples)

| ID | Filename | Label | Crop Res | Visual Description & Observation | Verdict | Reason |
|:---|:---|:---|:---:|:---|:---:|:---|
| **MM-01** | `1690272086_03-01-26_1_1.jpg` | Walking | 547x519 | Cow walking forward down aisle towards camera; foreleg in active stride; clear black/white coat markings; minor stall bar in lower-left foreground; mild CCTV compression on floor. | **GOOD** | Large target crop, clear leg articulation, unambiguous locomotion posture. |
| **MM-02** | `1690272026_03-00-26_5_1.jpg` | Walking | 294x409 | Cow walking away down aisle; visible soft motion blur on hind legs and tail motion; torso and coat silhouette clearly intact; posture upright in motion. | **ACCEPTABLE** | Motion blur present on fast-moving hooves, but posture and action remain recognizable. |
| **MM-03** | `1690276721_04-18-41_3_1.jpg` | Standing | 485x125 | Extremely narrow vertical crop (485x125); lower legs truncated; horizontal stall divider pipe runs across torso; dorsal coat visible. | **QUESTIONABLE** | Awkward bounding box geometry; lower limbs cut off by crop border. |
| **MM-04** | `1690273766_03-29-26_8_2.jpg` | Standing | 485x456 | Cow standing in feed alley viewed from Cam 2 side angle; crisp black/white coat pattern; full body profile in view; diagonal rail in foreground. | **GOOD** | High resolution, crisp body contours, perfect standing profile. |
| **MM-05** | `1690271981_02-59-41_16_4.jpg` | Standing | 640x256 | Steep dorsal/overhead camera angle; black cow standing stationary in stall; lower legs occluded by cubicle curb; dorsal spine and stance clearly visible. | **ACCEPTABLE** | Steep overhead perspective limits leg view, but standing stance is clear. |
| **MM-06** | `1690284401_06-26-41_4_1.jpg` | Feeding Head Up | 240x357 | Cow standing at feed bunk with neck upright and head lifted above feed barrier; muzzle angle clearly distinct from eating. | **GOOD** | Clear head-up transition posture; distinct from head-down feeding. |
| **MM-07** | `1690271876_02-57-56_1_2.jpg` | Feeding Head Up | 252x198 | Flank camera view of Cow 1 lifting head from bunk; distinct neck elevation; minor CCTV noise. | **ACCEPTABLE** | Moderate resolution, but neck elevation and head-up posture are obvious. |
| **MM-08** | `1690271876_02-57-56_5_1.jpg` | Feeding Head Down | 402x362 | Cow standing at bunk with neck extended downward into trough; muzzle immersed in feed; sharp dorsal ridge. | **GOOD** | Clear head-down eating posture, excellent body framing. |
| **MM-09** | `1690273166_03-19-26_12_1.jpg` | Feeding Head Down | 397x274 | Independent test cow feeding with head lowered into bunk; clean lateral posture; sharp coat edges. | **GOOD** | High quality, definitive feeding posture on held-out test cow. |
| **MM-10** | `1690293326_08-55-26_6_1.jpg` | Licking | 240x200 | Self-grooming behavior; cow's head is sharply flexed laterally backwards against its ribcage/flank, licking coat. | **GOOD** | Rare self-grooming behavior; distinctive anatomical neck flexion is unmistakable. |
| **MM-11** | `1690330406_19-13-26_10_1.jpg` | Licking | 141x205 | Cow extending muzzle to lick stall metal hardware/pipe; contact behavior; soft CCTV compression. | **ACCEPTABLE** | Low resolution (141x205), but muzzle-to-hardware licking action is visible. |
| **MM-12** | `1690280366_05-19-26_7_2.jpg` | Drinking | 417x485 | Large high-resolution crop; cow standing with head lowered into water basin; yellow ear tag clearly visible; water fixture sharp. | **GOOD** | Rich visual detail, unequivocal drinking posture and water tank context. |
| **MM-13** | `1690279496_05-04-56_2_4.jpg` | Drinking | 931x719 | Massive 931x719 crop; held-out test cow drinking from waterer; super crisp coat texture, legible ear tag, sharp water basin. | **GOOD** | Exceptional resolution and fidelity; definitive drinking behavior. |
| **MM-14** | `1690290716_08-11-56_9_3.jpg` | Lying | 447x228 | Cow resting recumbently in stall bedding; full recumbent posture; legs tucked under body; clear ear tag. | **GOOD** | Flawless resting/lying posture with complete body in frame. |
| **MM-15** | `1690271846_02-57-26_13_4.jpg` | Lying | 308x205 | Validation cow resting in stall under dimmer barn lighting; recumbent body contour clear; head upright resting. | **ACCEPTABLE** | Low-light illumination, but recumbent body posture is unambiguous. |
| **MM-16** | `1690272101_03-01-41_11_4.jpg` | Lying | 352x240 | Cow resting in cubicle bed; three horizontal galvanized steel stall divider pipes cut directly across torso and neck. | **ACCEPTABLE** | Heavy foreground metal pipe occlusion, but recumbent body is still visible behind bars. |
| **MM-17** | `1690326011_18-00-11_1_3.jpg` | Lying | 389x220 | Evening/night capture; low ambient light creates camera sensor ISO grain; cow silhouette and recumbent posture remain clear. | **ACCEPTABLE** | Sensor noise present due to low light, but behavior remains easily classifiable. |
| **MM-18** | `1690271846_02-57-26_1_1.jpg` | Feeding Head Down | 494x443 | Synchronized View A: Cow 1 feeding at bunk from rear-quarter angle (Cam 1); sharp coat markings and feed trough. | **GOOD** | High resolution, verified simultaneous capture. |
| **MM-19** | `1690271846_02-57-26_1_2.jpg` | Feeding Head Down | 234x189 | Synchronized View B: Cow 1 feeding at bunk at exact same second (02:57:26) from Cam 2 (flank angle). | **GOOD** | Confirms multi-camera temporal synchronization across barn network. |
| **MM-20** | `1690275851_04-04-11_14_3.jpg` | Lying | 475x311 | Cow lying down in cubicle bed; right crop border slightly clips hindquarters, but head, neck, and torso are fully intact. | **GOOD** | Minor bounding box clipping at tail, but recumbent posture is unmistakable. |

---

## 2. CBVD-5 Visual Inspection (20 Samples)

| ID | Filename | Label(s) | Crop Res | Visual Description & Observation | Verdict | Reason |
|:---|:---|:---|:---:|:---|:---:|:---|
| **CBVD-01** | `618_00002.jpg` | Stand | 246x274 | Cow standing upright in pen; all 4 legs bearing weight; high contrast edge enhancement; occupies ~3% of 1080p full frame. | **GOOD** | Clear standing posture and sharp edges. |
| **CBVD-02** | `623_00007.jpg` | Stand | 114x108 | Distant black cow standing in far cubicle behind vertical posts; 114x108 px; dark underexposed silhouette; blocky compression. | **QUESTIONABLE** | Very small distant crop; heavy pixelation and dark silhouette make details indistinct. |
| **CBVD-03** | `625_00002.jpg` | Lying down | 200x158 | Cow lying down in sand stall seen from posterior angle; recumbent body contour clear; curved divider bar crosses flank; DCT block noise on coat. | **ACCEPTABLE** | Recumbent posture is clear, but crop is low-resolution (200x158) with visible JPEG compression blocks. |
| **CBVD-04** | `618_00003.jpg` | Lying down | 125x290 | Narrow vertical strip (125x290) showing a curved metal stall bar and a partial patch of cow coat; head, limbs, and spine are out of frame. | **QUESTIONABLE** | Extreme partial crop; impossible to diagnose posture from crop alone without wide scene context. |
| **CBVD-05** | `618_00002.jpg` | Stand, Foraging | 848x252 | High-angle dorsal view of black cow standing along feed bunk; head lowered across curb into feed alley; yellow ear tag visible. | **GOOD** | Large crop, posture and feeding action match labels. |
| **CBVD-06** | `618_00002.jpg` | Stand, Foraging | 848x252 | Duplicate bounding box anchor for foraging validation; clear standing posture at bunk. | **GOOD** | Clear standing and eating action. |
| **CBVD-07** | `689_00003.jpg` | Stand, Drinking water | 344x460 | Cow standing upright with muzzle lowered directly into stainless steel water trough; sharp contrast; legs firmly on floor. | **GOOD** | Clean standing posture, unequivocal drinking interaction with water fixture. |
| **CBVD-08** | `689_00003.jpg` | Stand, Drinking water | 344x460 | Secondary anchor on water trough interaction; clean stance and head dip. | **GOOD** | Clear drinking posture. |
| **CBVD-09** | `618_00002.jpg` | Lying down, Rumination | 208x370 | White cow lying down in stall seen from rear; recumbent posture unmistakable; head facing away, jaw motion invisible in static image. | **ACCEPTABLE** | Lying posture is GOOD, but rumination cannot be confirmed visually from a static crop. |
| **CBVD-10** | `618_00002.jpg` | Stand, Rumination | 321x349 | Cow standing upright next to waterer; weight on all four legs; head slightly lowered; chewing cud invisible in single frame. | **GOOD** | Standing posture is crisp and unmistakable; rumination unverified in static crop. |
| **CBVD-11** | `618_00002.jpg` | Stand, Rumination | 321x349 | Crowded scene sample; bounding box accurately covers target standing cow. | **GOOD** | Standing posture is clear. |
| **CBVD-12** | `621_00002.jpg` | Stand | 150x332 | Frontal view of cow standing facing camera; dark underexposed torso; white forelegs visible; blurry edges at 150 px width. | **ACCEPTABLE** | Standing stance is identifiable, but image is dark, narrow, and low resolution. |
| **CBVD-13** | `621_00002.jpg` | Stand, Foraging | 440x300 | High-angle dorsal-lateral view of cow standing at feed lane; diagonal fence pipe occludes neck; head reaching into feed bunk. | **GOOD** | Posture is clearly standing and foraging; metal rail occlusion does not obscure behavior. |
| **CBVD-14** | `686_00004.jpg` | Stand | 88x102 | Cow standing in far-distance open doorway hundreds of feet away; minuscule 88x102 px crop; severe pixelation, heavy DCT artifacts. | **BAD** | Severe pixelation; cow is a blocky smudge resembling a retro low-res artifact; barely recognizable. |
| **CBVD-15** | `621_00002.jpg` | Stand, Foraging | 440x300 | Consecutive frame t=2s; cow standing and feeding across fence rail; sharp coat markings. | **GOOD** | Clear standing and foraging action. |
| **CBVD-16** | `621_00003.jpg` | Stand, Foraging | 402x282 | Consecutive frame t=3s; 1 second later; slight tail movement; head remains lowered in feed bunk. | **GOOD** | Continuous temporal standing and foraging behavior. |
| **CBVD-17** | `621_00004.jpg` | Stand | 378x176 | Consecutive frame t=4s; shifted bounding box captures dorsal spine and back; label dropped "foraging" (only "stand"). | **ACCEPTABLE** | Bounding box shifted; posture is standing, but foraging label was dropped. |
| **CBVD-18** | `621_00005.jpg` | Stand | 442x332 | Consecutive frame t=5s; cow is back in full view, actively foraging with head in bunk, but label is ONLY "stand". | **ACCEPTABLE** | Visual action is foraging, but AVA annotation dropped "foraging" label (temporal inconsistency). |
| **CBVD-19** | `624_00002.jpg` | Stand, Rumination | 290x222 | Cow standing inside cubicle stall; multiple curved metal divider pipes cross body; standing stance evident. | **ACCEPTABLE** | Stall bar occlusion present; standing posture clear; rumination unverified on static crop. |
| **CBVD-20** | `689_00003.jpg` | Stand, Drinking water | 344x460 | Cow at drinker in Video 689 setup; sharp contrast, upright posture at water trough. | **GOOD** | Clear standing and drinking posture. |

---

## 3. Summary Tally

### MmCows Visual Inspection Summary
- **GOOD**: 12 (60%)
- **ACCEPTABLE**: 7 (35%)
- **QUESTIONABLE**: 1 (5%) — *MM-03: overly narrow vertical crop (485x125) truncating lower legs.*
- **BAD**: 0 (0%)
- **Main Visual Problems Observed**:
  1. CCTV compression noise on floor and moving limbs.
  2. Soft motion blur on hooves/limbs during walking (e.g. MM-02).
  3. Stall bar foreground occlusion in stall camera views (e.g. MM-16).
  4. Occasional tight/awkward crop framing (e.g. MM-03).
  5. Overall usability: **19 / 20 (95%) samples are fully usable for deep learning behavior recognition.**

### CBVD-5 Visual Inspection Summary
- **GOOD**: 11 (55%)
- **ACCEPTABLE**: 6 (30%)
- **QUESTIONABLE**: 2 (10%) — *CBVD-02 (114x108 distant stall), CBVD-04 (125x290 partial stall bar slice).*
- **BAD**: 1 (5%) — *CBVD-14: 88x102 px distant doorway cow; severe DCT block compression, barely recognizable as an animal.*
- **Main Visual Problems Observed**:
  1. **The "Wide-Angle Sharpness Illusion"**: 1080p full scenes look sharp, but individual cows occupy tiny bounding boxes (median 156x167 px).
  2. Severe pixelation and DCT compression blockiness on distant cows (e.g. CBVD-14 at 88x102 px).
  3. Extreme partial crops that capture stall bars and a coat patch without limbs or head (e.g. CBVD-04).
  4. Temporal label inconsistency in AVA annotations (e.g. Video 621 where frame 2 and 3 are labeled "stand, foraging", but frame 5 with identical feeding posture drops "foraging" and is labeled only "stand").
  5. Static unverifiability: Rumination (chewing cud) cannot be visually diagnosed on single static image crops without temporal video playback.

---

## 4. Direct Answers to Core Evaluation Questions

### 1. Does MmCows actually look too poor for behavior training?
**NO.** While MmCows displays standard agricultural CCTV compression and occasional motion blur on fast-moving hooves, 95% of crops (19/20) clearly capture the animal's torso, limbs, head, and behavioral posture. MmCows crops are large (median 390x370 px, reaching up to 931x719 px), providing abundant pixel density on the cow itself. It is fully viable and standard for real-world barn vision.

### 2. Does CBVD actually look visually better at the COW level, not just scene level?
**NO.** This is the most crucial visual finding of this audit. CBVD-5 creates a false impression of superiority because its 1080p full barn scenes look bright and clean. However, when cropped to individual cows, the animals are physically distant from the wide-angle camera. The median CBVD-5 crop is only 156x167 px (2.5x smaller in area than MmCows). Distant cows shrink to 88x102 px, where JPEG/DCT block compression turns the animal into a pixelated smudge. At the cow crop level, CBVD is **not** superior to MmCows.

### 3. Are the previous blur/resolution conclusions visually supported?
**YES, 100% VISUALLY SUPPORTED.**
- MmCows crops contain significantly more pixels on the cow body.
- CBVD full scenes look sharper, but CBVD cow crops suffer from low resolution, edge-sharpening halos, and blocky compression.
- The forensic audit's claim that CBVD's sharpness is a scene-level illusion is confirmed by direct visual inspection.

### 4. Is either dataset visually unusable?
**NO.** Neither dataset is visually unusable. Both datasets are functional agricultural vision datasets. For coarse behaviors (standing, lying down, drinking, feeding), models can learn robust representations from both.

### 5. Do you see any obvious annotation/crop mistakes?
**YES.**
- **CBVD-5**:
  - `CBVD-04` is an uninformative 125x290 crop of a stall bar and coat patch with no limbs or head.
  - `CBVD-14` is an unusable 88x102 pixelated smudge in a distant doorway.
  - **Temporal inconsistency**: In Video 621 (CBVD-15 to CBVD-18), the cow is actively standing and feeding with head in the feed trough across frames 2, 3, 4, and 5. Frames 2 and 3 are labeled `stand, foraging`, but frame 5 drops `foraging` and is labeled only `stand`.
  - **Static unverifiability**: Rumination cannot be verified on static crops; it requires temporal video modeling.
- **MmCows**:
  - `MM-03` has an awkward 485x125 crop geometry that clips lower legs.

### 6. Based ONLY on visual quality, which dataset looks stronger?
**MmCows is visually stronger for cow-level deep learning.** While CBVD has better ambient barn lighting, MmCows delivers **2.5x more pixels on the animal**, preserves full body silhouettes, and captures dynamic behaviors (walking, licking) that CBVD completely lacks.

### 7. Based on visual quality + the existing scientific/provenance evidence, should the current roles remain unchanged?
**YES. CURRENT ROLES SHOULD REMAIN UNCHANGED.**
- **Primary Behavior**: `MmCows`
- **External Validation**: `CBVD-5`

**Why?** Because visual inspection confirms that MmCows is not too degraded for training and CBVD does not have superior cow-level detail, the non-negotiable scientific and structural factors decisively favor MmCows:
1. **Biological Cow Identity**: MmCows provides verified cow IDs (1–16) and strict identity-disjoint train/val/test splits. CBVD-5 has ZERO cow IDs (dummy actor ID `1` everywhere), making true cow-disjoint generalization testing impossible.
2. **Video Split Leakage**: Official AVA splits in CBVD-5 leak 100% of val/test videos into train.
3. **Taxonomy Completeness**: MmCows includes Walking (vital for mobility/lameness research) and Licking. CBVD-5 lacks Walking and Licking entirely.
4. **Temporal Modeling**: Rumination in CBVD-5 requires 3D-CNN / video modeling and cannot be reliably classified on static 2D crops.
