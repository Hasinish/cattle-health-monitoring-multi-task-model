# Thesis Review: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring

**Reviewer Role:** Academic Supervisor / Technical Reviewer  
**Document:** `main_test.pdf` (46 pages)  
**Date:** 2026-06-09

---

## 1. RESEARCH CLARITY & PROBLEM STATEMENT

**Is the problem clearly defined?** Mostly yes. Chapter 1 identifies four gaps: fragmented single-task research, private-dataset dependency, lack of cross-dataset evaluation, and limited multimodal public data. These are real problems in the field.

**Is the motivation strong?** The motivation section (Section 1.2) is earnest but reads like a blog post, not an academic argument. Sentences like *"This unlabeled video just sits there"* and *"most farms already run surveillance cameras that record hours of footage daily"* are colloquial. The motivation needs to be rewritten in formal academic prose with quantitative evidence backing each claim.

**Are objectives specific and measurable?** Partially. Objective 1 ("develop a unified multi-task framework") is vague — what constitutes "unified"? Objective 3 ("evaluate sequential task training strategies") overlaps heavily with Objective 1. There is no success criterion defined. What MAE, F1, or AUC would constitute a successful outcome? Without measurable targets, these are aspirations, not objectives.

**Is the scope well-defined?** Yes. Section 1.6 is actually one of the strongest parts of the introduction — clear boundaries on tasks, modalities, datasets, and parameter budget (<10M). Well done here.

---

## 2. LITERATURE REVIEW

**Is it comprehensive?** For an undergraduate thesis, it is reasonably comprehensive. 29 papers reviewed across BCS, lameness, behavior, and MTL is adequate. The summary tables (Tables 2.1–2.5) are excellent — they show systematic thinking.

**Is there critical analysis?** This is where it falls short. The literature review is predominantly a **chronological summary** — "X did this, Y did that." There is almost no critical comparative analysis. For example:
- Why does Tun 2024 achieve 99.94% but Russello 2024 only 80%? Is it the modality? The dataset size? The scoring granularity? The review lists them but never **compares** them analytically.
- The claim "no existing paper integrates BCS, behavior, lameness, and ID in a unified framework" (Section 2.3) is the central novelty claim but is asserted without evidence of an exhaustive search methodology (e.g., which databases were searched, what search terms, what inclusion/exclusion criteria).

**Missing references:** There is no discussion of:
- GradNorm or uncertainty weighting for MTL loss balancing (Kendall et al., 2018; Chen et al., 2018) — directly relevant to your multi-task loss weighting.
- Any cattle re-identification literature beyond OpenCows2020.
- The broader MTL literature beyond Crawshaw 2020 (e.g., Standley et al., 2020 "Which Tasks Should Be Learned Together?").

---

## 3. METHODOLOGY

**Is it clearly explained?** Yes. Chapter 4 (chapter_5.tex) is the **strongest chapter** in the report. The architecture diagram, the mathematical formulations for CORAL, Focal Loss, BCE, and Cross-Entropy are all correct and clearly presented. The phased sequential training pipeline (Phase 1–6) is well-structured and demonstrates genuine engineering thought.

**Is MTL justified over single-task?** Here's the uncomfortable truth: **the results do not justify MTL**. Table 5.6 shows:
- BCS MAE **degrades** from 0.5566 (single-task) → 0.7849 (MTL). That's a **41% degradation**.
- Behavior F1 **degrades** from 0.7445 (single-task) → 0.4775 (MTL). That's a **36% degradation**.

The report acknowledges this as "negative transfer" but then boldly claims the MTL model is "exceptionally viable for farm surveillance deployment" in Section 5.7. This is intellectually dishonest. You cannot dismiss a 41% MAE increase as "minor performance drops" (line 191 of chapter_6_test.tex). A vet would notice a full BCS class difference.

**Are datasets appropriate?** The dataset selection is actually excellent and well-justified. The cow-wise group splitting to prevent identity leakage (Section 4.2.2) is a sophisticated choice that many published papers miss. The sparse temporal sampling is clever engineering.

> [!CAUTION]
> **Critical Issue: The lameness dataset is dangerously small.** 50 videos (25 lame, 25 sound) is not a dataset — it's an anecdote. Reporting "AUC = 1.0000" and "100% accuracy" on 50 samples is statistically meaningless. Any binary classifier can achieve perfect separation on 50 samples with enough capacity. This result should be presented with extreme caution, not bold-faced as a headline achievement.

**Loss weighting strategy:** The multi-task loss weights (0.35/0.35/0.15/0.15) in Equation 4.6 are stated but **never justified**. Why these specific weights? Were they tuned? Grid-searched? Based on task uncertainty? This is a critical design decision presented without any rationale.

---

## 4. TECHNICAL DEPTH & ACCURACY

**Are DL concepts used correctly?** Mostly yes. The CORAL formulation (Equations 4.1–4.2), CBAM description (Equations 4.3–4.4), and Focal Loss (Equation 4.5) are all technically correct.

**Technically incorrect or misleading claims:**

1. **"Reduces edge computer memory overhead by 75%"** (Section 5.7, line 247). This claim assumes you would otherwise run 4 separate EfficientNet-B0 models. But nobody does that. A fair comparison would be 4 models with lighter backbones (MobileNet) vs. 1 model with EfficientNet-B0. The 75% number is misleading.

2. **"Inference latency from 120ms to 32ms per frame"** (same section). Where does 120ms come from? Which hardware? At what batch size? This number appears fabricated — there is no benchmarking table or methodology supporting it.

3. **"Lameness AUC = 1.0000"** — Technically correct but practically meaningless on n=50. The report does not compute confidence intervals. A bootstrapped 95% CI on AUC with n=50 would be extremely wide. Claiming "perfect" performance without statistical qualification is a red flag for any reviewer.

4. **Ablation 3 (RGB vs. DGE)** compares EfficientNet-B0 on Dryad (DGE, MAE=0.6175) vs. ScienceDB (RGB, MAE=0.5566) and concludes the model "generalizes well across both modalities." This is **not** a valid modality comparison — these are **different datasets** with different cows, different scoring scales, different sizes. You cannot attribute the performance difference to modality when every other variable is also different.

**Conceptual gaps:**
- No discussion of **gradient conflict** between tasks during joint training (Phase 6). You mention "negative transfer" descriptively but never analyze the actual gradient dynamics.
- No training curves (loss vs. epoch) for any task. How do we know the models converged? Were there signs of overfitting?

---

## 5. RESULTS & ANALYSIS

**Are results presented clearly?** Yes. The tables are well-formatted and the new bar charts are a nice addition. The per-class behavior accuracy breakdown (Table 5.3) and the CBVD-5 cross-dataset bar chart (Figure 5.2) add real analytical value.

**Sufficient baseline comparison?** Partially. The 5-backbone comparison is thorough for single-task. But there are **no comparisons against published methods**. For example:
- Your BCS MAE of 0.5566 on ScienceDB — how does this compare to Liang 2025's 93.77% accuracy? You never make this comparison even though you reviewed that paper.
- Your behavior F1 of 0.7445 — how does this compare to Dottorini 2025's 91.8% on the same MmCows dataset?

Without these comparisons, the reader cannot assess whether your results are competitive.

**Are error cases discussed?** The CBVD-5 generalization analysis (Ablation 5) is genuinely excellent. Identifying that Drinking drops to 2.99% and discussing domain shift honestly is the kind of analysis I want to see more of. The threshold calibration section (5.3.2) is also good — it shows real debugging of model behavior.

**Are claimed improvements significant?** No statistical significance tests are reported anywhere. No confidence intervals, no paired t-tests, no bootstrapping. For a report that makes strong claims about "perfect" performance, this is a serious omission.

---

## 6. WRITING QUALITY & ACADEMIC STYLE

**Overall quality:** The writing quality is **inconsistent**. Chapter 4 (Methodology) reads like a proper academic paper. Chapter 1 (Introduction) reads like a first draft with casual language.

**Specific issues:**

| Location | Problem | Fix |
|---|---|---|
| Ch1, Sec 1.1, para 1 | Single 450-word paragraph. Unreadable wall of text. | Break into 4–5 focused paragraphs with topic sentences. |
| Ch1, Sec 1.2 | *"This unlabeled video just sits there"* | Rewrite: "This abundant unlabeled footage remains unutilized" |
| Ch1, Sec 1.2 | *"which is surprising"* | Remove editorial commentary. State the gap factually. |
| Ch1, Sec 1.3 | *"but most of the cattle monitoring studies use their own data which are private"* | Informal. Rewrite: "the majority of studies employ proprietary datasets" |
| Ch5, Sec 5.7 | Only 2 bullet points in the Discussion section. | This is supposed to be the intellectual core of a results chapter. Expand significantly. |
| Abstract | 1 paragraph, ~240 words. Acceptable length but crams too many numbers. | Reduce to key metrics only; add a 1-sentence limitation acknowledgment. |

**Figure/Table referencing:** After our recent fixes, cross-references are now properly present. Good.

**Consistency issues:**
- "ScienceDB" vs. "SciDB" — used interchangeably (Tables use "SciDB" but text uses "ScienceDB").
- "Macro F1" vs. "Macro F1-score" — inconsistent throughout.

---

## 7. CONTRIBUTIONS & ORIGINALITY

**Are contributions clearly stated?** Yes. Section 6.2 lists four contributions. They are well-articulated.

**Are they genuinely novel?**

1. *"First end-to-end multi-task framework for BCS + behavior + lameness + ID"* — This is a valid novelty claim **if** you can substantiate the systematic search. Currently unsubstantiated.
2. *"Full reproducibility via public datasets"* — This is a genuine and commendable contribution. Most ag-CV papers fail here.
3. *"Spatiotemporal integration with attention"* — CBAM + LSTM is not novel architecturally. The novelty is in applying it to cattle monitoring, which is incremental.
4. *"Edge optimization (<10M params)"* — The claim is valid but there is no actual edge deployment benchmark (no Jetson inference times, no ONNX export, no TensorRT profiling). Without evidence of actual edge deployment, this is aspirational, not a contribution.

---

## 8. CONCLUSION & FUTURE WORK

**Does the conclusion reflect what was done?** Mostly. But the framing is too triumphant given the results. Saying the model achieves "outstanding computational efficiency" when BCS and behavior tasks degraded significantly under MTL is misleading.

**Are limitations acknowledged?** Partially. The negative transfer is mentioned but downplayed. The 50-video lameness dataset limitation is never explicitly called out as a limitation. It should be.

**Future work:** The future work section (Section 6.3) is actually quite strong and realistic:
- Joint fine-tuning (unfreezing backbone) — directly addresses the negative transfer problem.
- Federated learning for cross-dataset generalization — thoughtful.
- Behavior sequential modeling — logical next step.
- Thermal imaging — appropriately flagged as requiring new data collection.

---

## 9. FORMATTING & PRESENTATION

**Structure:** The report follows a standard thesis structure (Title, Declaration, Approval, Abstract, Acknowledgment, ToC, LoF, LoT, Nomenclature, 6 Chapters, Bibliography). This is well-organized.

**Citations:** IEEE format via biblatex. Consistent formatting. 

**Visual quality:** Clean LaTeX output. The new `pgfplots` bar charts with grid lines and custom colors look professional. The architecture diagram (mtl_architecture.png) is clear.

**Minor formatting issues:**
- The Nomenclature page exists but I cannot verify its completeness from the source files alone.
- Chapter numbering in `main_test.tex` maps Chapter 4 → "Proposed Methodology" and Chapter 5 → "Result Analysis". The internal `.tex` files are named `chapter_5.tex` and `chapter_6_test.tex` respectively, which is confusing for maintainability but does not affect the PDF output.

---

## ✅ STRENGTHS (What was done well)

1. **Methodology chapter is excellent.** The phased sequential training pipeline, mathematical formulations (CORAL, Focal Loss, CBAM), and preprocessing decisions (cow-wise splits, sparse temporal sampling, data capping) demonstrate strong engineering rigor. This is publication-quality methodology writing.

2. **Dataset selection and reproducibility commitment.** Using 6 public datasets exclusively is a genuine contribution. The cow-wise group splitting to prevent identity leakage is sophisticated.

3. **Honest generalization analysis.** The CBVD-5 cross-dataset evaluation (Ablation 5) and threshold calibration section show real intellectual honesty. Reporting a 0.1245 F1 cross-dataset score takes courage.

4. **Comprehensive ablation studies.** Five structured ablations isolating CORAL vs. CE, CBAM contribution, RGB vs. DGE, Focal Loss, and cross-dataset generalization. This is thorough experimental design.

5. **Literature review tables.** The summary tables for BCS, lameness, behavior, and MTL papers are excellent reference material and show systematic thinking.

6. **Requirements chapter (Chapter 3).** The ABET/BAETE compliance sections (societal impact, environmental impact, ethics, risk management, economic analysis) are well-written and relevant. Many undergrad theses phone these in — yours are substantive.

---

## ⚠️ AREAS FOR IMPROVEMENT (Good but can be better)

1. **Discussion section is skeletal.** Section 5.7 has only 2 bullet points. This should be 2–3 pages discussing: why negative transfer occurred, how your results compare to published baselines, practical deployment considerations, and honest limitations. **Fix:** Expand to at least 6–8 paragraphs covering trade-offs, comparisons, and deployment realities.

2. **No comparison against published SOTA.** You review 29 papers but never compare your numbers against them. **Fix:** Add a table comparing your BCS MAE against Liang 2025, your behavior F1 against Dottorini 2025, etc.

3. **Introduction writing quality.** Sections 1.1–1.3 use informal language and have wall-of-text paragraphs. **Fix:** Break Section 1.1 into 5 focused paragraphs. Remove all colloquialisms. Add paragraph topic sentences.

4. **Objective 1 and 3 overlap.** **Fix:** Merge into one objective or sharpen the distinction (e.g., Obj 1 = "build the framework", Obj 3 = "compare sequential vs. joint training strategies").

5. **Inconsistent dataset naming.** "ScienceDB" vs. "SciDB" used interchangeably. **Fix:** Pick one and use it everywhere (or define the abbreviation once and use the short form consistently).

6. **No training curves anywhere.** **Fix:** Add loss-vs-epoch and metric-vs-epoch plots for at least the main tasks (BCS and Behavior). This is standard practice and helps readers assess convergence/overfitting.

---

## ❌ CRITICAL WEAKNESSES (Serious problems)

1. **The MTL results undermine the thesis claim.** The entire thesis argues for a "unified multi-task framework" but the experimental results show that MTL **degrades** the two primary tasks (BCS MAE +41%, Behavior F1 −36%) compared to single-task baselines. The report dismisses this as "minor" and calls the model "exceptionally viable." This is the central contradiction of the report. **Why it matters:** Any examiner will immediately ask "Why would I use your MTL model when single-task models are significantly better?" **Fix:** Reframe the narrative. Acknowledge the trade-off honestly. Argue that the MTL value proposition is computational efficiency (1 model vs. 4) and that future gradient balancing (GradNorm, PCGrad) could resolve the negative transfer. Do not claim the results are good when they demonstrably aren't for 2/4 tasks.

2. **Lameness results on n=50 are statistically unreliable.** Reporting AUC=1.0000 and accuracy=100% on 50 videos without any statistical qualification (confidence intervals, cross-validation folds) is a serious methodological weakness. **Why it matters:** This will be the first thing a technically competent examiner challenges. **Fix:** Add bootstrapped confidence intervals. Use k-fold cross-validation. At minimum, add a prominent caveat acknowledging the tiny sample size.

3. **Multi-task loss weights are unjustified.** The weights (0.35/0.35/0.15/0.15) in Equation 4.6 are stated as fact with zero rationale. **Why it matters:** These weights directly control what the model learns. If they're wrong, they explain the negative transfer. **Fix:** Either (a) report a grid search over weight combinations, (b) cite uncertainty weighting (Kendall et al., 2018), or (c) at minimum justify the choice heuristically (e.g., "BCS and behavior are given higher weight because they have more training data").

4. **No statistical significance testing anywhere.** Zero p-values, zero confidence intervals, zero effect sizes across the entire results chapter. **Why it matters:** Without statistics, all comparisons ("improved from X to Y") are anecdotal. **Fix:** Add at minimum bootstrapped 95% CIs for the key metrics in Table 5.6, especially for the lameness and ID results where sample sizes are small.

5. **Ablation 3 (RGB vs. DGE) is methodologically invalid.** Comparing EfficientNet-B0 on Dryad (DGE) vs. ScienceDB (RGB) conflates modality with dataset. The datasets have different cows, different scales, different sizes (5,923 vs. 53,566 images). **Why it matters:** The conclusion that "the model generalizes well across both modalities" is not supported by the evidence. **Fix:** Either remove this claim or add a proper disclaimer: "A controlled modality comparison requires the same dataset with paired RGB and DGE images, which is not available in public data."

---

## 🔥 PRIORITY FIX LIST (Top 5 most important things to fix RIGHT NOW)

| Priority | Issue | Impact | Effort |
|---|---|---|---|
| **1** | **Reframe MTL narrative.** Stop claiming MTL results are good when BCS and behavior degraded 36–41%. Acknowledge the trade-off honestly and argue for computational efficiency instead. | Prevents examiner from immediately dismissing the thesis | ~2 hours of rewriting Section 5.3.1 and 5.7 |
| **2** | **Add statistical caveats for lameness (n=50).** Add a paragraph explicitly flagging the sample size limitation. Ideally add cross-validation or bootstrapped CIs. | Prevents credibility damage from the "perfect score" claim | ~1 hour for caveat text; ~3 hours for CV |
| **3** | **Expand Discussion section.** Currently 2 bullet points. Needs comparison with published SOTA, honest limitation analysis, and deployment considerations. | This section is the intellectual heart of a thesis; currently hollow | ~3 hours |
| **4** | **Justify multi-task loss weights.** Add 1 paragraph explaining why 0.35/0.35/0.15/0.15 was chosen. | Addresses a critical unexplained design decision | ~30 minutes |
| **5** | **Clean up Chapter 1 writing.** Break wall-of-text paragraphs, remove colloquialisms, sharpen objectives. | First impression for examiners; currently unprofessional | ~2 hours |

---

## 📊 OVERALL SCORE

| Dimension | Score | Notes |
|---|---|---|
| Research Clarity | **6/10** | Problem is identified but objectives are vague and overlapping. Motivation is informal. |
| Literature Review | **7/10** | Good coverage and excellent summary tables, but lacks critical comparative analysis. |
| Methodology | **8/10** | Strongest chapter. Clear math, sound preprocessing, well-structured training pipeline. |
| Technical Depth | **6/10** | Correct formulations but several misleading claims and one invalid ablation. |
| Results & Analysis | **5/10** | Tables and charts are good, but no SOTA comparison, no statistics, and the narrative contradicts the data. |
| Writing Quality | **5/10** | Wildly inconsistent. Ch4 is professional; Ch1 reads like a draft. |
| Overall Report Quality | **6/10** | A solid undergraduate effort with strong methodology, but the results narrative needs significant correction. |

### Verdict

This is a **competent undergraduate thesis with strong engineering work** — the preprocessing decisions, architecture design, and ablation framework are genuinely well-thought-out. However, the report's biggest problem is **intellectual honesty in results interpretation**. The multi-task model degrades two of four tasks significantly, and the report tries to spin this as a success rather than analyzing it critically. The lameness "perfect score" on 50 samples is a credibility risk. **Fix the narrative, add statistical rigor, expand the discussion, and this becomes a strong P2 report.** The raw work is good — the story you're telling about it needs to catch up.
