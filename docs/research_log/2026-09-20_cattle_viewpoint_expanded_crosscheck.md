# Cattle Viewpoint 100-Sample Expanded Cross-Check & Adjudication Audit (Step 2.4)

## 1. Executive Summary
To provide a rigorous, independent validation of cattle viewpoint labeling beyond the initial 60-image human-verified baseline, we constructed a new 100-image sample (ScienceDB: 34, MmCows: 33, SideViewCows2026: 33; seed 2026) with zero sample overlap. Visual labeling was independently executed by the coding/research agent and cross-checked against blind contact sheet predictions from ChatGPT vision, yielding an initial raw concordance of 79.0% (79/100). The 21 disagreements were isolated into a dedicated user review document and adjudicated directly by the user (20 ChatGPT labels accepted, 1 explicit user override to `rear`), producing a finalized 100-sample cross-checked benchmark with strict provenance tracking.

---

## 2. Context & Motivation
In Phase 3 Step 2.4, evaluating coarse viewpoint estimation requires reliable ground truth. While the initial 60-image manual review pack (`viewpoint_manual_review_manifest.csv`) established the visual feasibility of the 6-class taxonomy, 60 samples is small for assessing multi-model zero-shot performance across Diverse cattle environments. Furthermore, relying entirely on a single agent's visual labels introduces potential bias.

To solve this without compromising scientific integrity:
1. We selected 100 new, non-overlapping images sampled across all primary Phase 3 domains.
2. We generated 10 blind contact sheets containing only sample IDs (zero bounding boxes, zero metadata, zero hints).
3. We submitted these blind sheets for independent vision evaluation by ChatGPT.
4. We presented all 21 disagreements to the user in `docs/audits/phase3_viewpoint_mismatch_user_review.md` for human adjudication.
5. We strictly maintained distinct provenance: this 100-sample set is labeled as `agent visual labeling + independent ChatGPT vision cross-check + user adjudication of disagreements`, preserving the pristine status of the original 60-image human-verified baseline.

---

## 3. Forensic Findings & Data

### 3.1 Sample Selection & Distribution (Seed: 2026)
- **ScienceDB (34 samples)**: Across 5 BCS classes (3.25: 7, 3.5: 7, 3.75: 7, 4.0: 7, 4.25: 6) and 3 farm facilities (GS: 14, YM: 14, STEREO: 6).
- **MmCows (33 samples)**: Across 7 behaviors (lying: 5, standing: 5, walking: 5, feeding: 5, drinking: 5, headup: 5, licking: 3), 4 CCTV cameras (C1: 9, C2: 8, C3: 8, C4: 8), and 15 unique cow identities.
- **SideViewCows2026 (33 samples)**: Across parlor (15), barn (12), and snapshots (6) subsets across 28 distinct cow identities.
- **Total**: Exactly 100 unique samples; 0 overlap with the 60 samples in `viewpoint_manual_review_manifest.csv`.

### 3.2 Inter-Model Concordance (Agent vs. ChatGPT Vision)
- **Agreed Samples**: 79 / 100 (79.0%)
- **Disagreements**: 21 / 100 (21.0%)

#### Nature of Disagreements:
1. **Rear vs. Rear-Oblique Boundary (7 cases)**: e.g. `vp2_0010`, `vp2_0012`, `vp2_0018`, `vp2_0022`, `vp2_0027`. Cows in chute walking slightly at an angle where the agent called `rear-oblique` and ChatGPT called `rear`. User adjudicated all 7 to `rear`.
2. **Side vs. Front-Oblique / Rear-Oblique (7 cases)**: e.g. `vp2_0041`, `vp2_0044`, `vp2_0048`, `vp2_0054`, `vp2_0064`, `vp2_0084`. Loose barn / parlor angles where flank dominates but slight perspective angle exists. User confirmed `side` (5) or `rear-oblique` (2).
3. **Severe Occlusion / Ambiguity (4 cases)**: e.g. `vp2_0055`, `vp2_0061`, `vp2_0065`, `vp2_0060`. Barn stall bars / top-down occlusions. ChatGPT called `unknown / ambiguous` for all 4. User accepted `unknown / ambiguous` for 3, and issued an explicit override to `rear` for `vp2_0060`.
4. **Direct Orientation Inversion (3 cases)**: e.g. `vp2_0017`, `vp2_0023`, `vp2_0030`. Chute lighting or cow bending made agent call `front` or `front-oblique`, whereas ChatGPT identified caudal/rear anatomy. User adjudicated all 3 to `rear`.

### 3.3 Final Adjudicated Label Distribution (N=100)
| Viewpoint Class | Overall Count | Percentage | ScienceDB | MmCows | SideViewCows2026 |
|---|---|---|---|---|---|
| `side` | 54 | 54.0% | 0 | 21 | 33 |
| `rear` | 27 | 27.0% | 25 | 2 | 0 |
| `rear-oblique` | 12 | 12.0% | 9 | 3 | 0 |
| `unknown / ambiguous` | 5 | 5.0% | 0 | 5 | 0 |
| `front-oblique` | 2 | 2.0% | 0 | 2 | 0 |
| `front` | 0 | 0.0% | 0 | 0 | 0 |
| **Total** | **100** | **100.0%** | **34** | **33** | **33** |

---

## 4. Architectural Decisions & Action Plan
1. **Manifest Provenance Discipline**:
   - The 60-image `viewpoint_manual_review_manifest.csv` remains the pure human-verified baseline (`human_verified`).
   - The 100-image `viewpoint_expanded_agent_review_manifest.csv` is preserved under `crosschecked_consensus` (79 rows) and `user_adjudicated_crosscheck` (21 rows).
2. **Zero-Shot VLM Evaluation Readiness**:
   - Both manifests are ready to be used as evaluation targets for `scripts/audit_viewpoint_zeroshot.py`.
   - The expanded 100-sample set provides greater statistical power and coverage across all three datasets while maintaining high label quality.

---

## 5. Artifacts & File Registry
- Selection & Pack Script: `scripts/build_viewpoint_expanded_crosscheck_pack.py`
- Finalization Script: `scripts/finalize_viewpoint_expanded_manifest.py`
- Adjudication Manifest: `artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv`
- Contact Sheets: `docs/audits/assets/viewpoint_expanded_crosscheck/viewpoint_crosscheck_01.jpg` to `10.jpg`
- Cross-Check Index: `docs/audits/phase3_viewpoint_expanded_crosscheck_index.md`
- User Review File: `docs/audits/phase3_viewpoint_mismatch_user_review.md`
- User Review Assets: `docs/audits/assets/viewpoint_mismatch_user_review/vp2_*.jpg`

---

## 6. Next Steps
- [x] Select 100 diverse, non-overlapping samples across ScienceDB, MmCows, SideViewCows2026.
- [x] Render 10 blind contact sheets and index.
- [x] Obtain independent ChatGPT vision predictions.
- [x] Audit 21 disagreements and compile user review pack.
- [x] Apply user adjudications to expanded manifest.
- [ ] Run comparative zero-shot VLM evaluation (`openai_clip`, `openclip_laion`, `google_siglip`) on the 60 human-verified samples and/or the 100 cross-checked samples.
- [ ] Finalize Step 2.4 viewpoint synthesis and proceed to Step 2 synthesis report.
