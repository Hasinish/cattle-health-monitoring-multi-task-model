# Hostile P2 Review: *Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring*

Audit date: 2 September 2026  
Source audited: complete 58-page P2 PDF (46 numbered report pages plus front matter)  
Revision: second-pass audit; corrected one earlier overstatement and added equation-, data-unit-, architecture-, statistics-, figure-, deployment-, and 2026-literature findings  
Review stance: methodology, statistical rigor, literature integrity, novelty, and defense survivability

## 1. EXECUTIVE VERDICT

This report is **not defensible as-is**. It describes a four-task cattle-monitoring system built from EfficientNet-B0, CBAM, CORAL, focal loss, an LSTM, YOLOv8-nano crops, and six public datasets, but its main empirical story rests on results whose experimental units, splits, training schedule, and uncertainty are either invalid or insufficiently specified. The single biggest risk is **evaluation contamination and pseudo-replication**: frame-level lameness explicitly leaks video identity/background; cow-ID uses image-wise splits and then unexplained temporal voting; video-derived datasets are not shown to be separated by animal, recording, farm, day, or source; and the test set is used to choose a lameness threshold. That risk infects the strongest numbers—0.9829/0.9921/1.0000 lameness AUC and 97.58% ID—and therefore infects backbone selection, architecture claims, and the “unified benchmark” contribution. The report is also statistically incapable of supporting causal language: every ablation is one run, “default PyTorch random seed” is not a documented seed or deterministic protocol, no confidence interval or hypothesis test appears, and the tiny lameness test apparently changes in 10-percentage-point steps. The novelty claim may still be narrowly salvageable as “one engineering prototype combining these four named outputs,” but it is not established by the literature review, and it is weakened by recent multitask, multi-camera, video, behavior–identity, and public-dataset work. The P3 defense can survive only if the team stops defending the current headline scores, rebuilds leakage-resistant splits, reports multi-seed/group-bootstrap uncertainty, defines the multi-dataset optimization algorithm precisely, and reframes the contribution as a rigorously evaluated study of when cross-dataset hard sharing fails—not as a deployment-ready unified monitor.

## 2. CRITICAL FLAWS

### 1. The strongest lameness result is knowingly contaminated and is still used throughout the paper

- **Exact location/quote:** Abstract: “Frame-level lameness AUC reached 0.9829, but this is treated as an optimistic upper bound due to possible identity/background leakage.” Table 5.6 nevertheless uses that task result in backbone selection; Table 5.7 headlines 0.9829, 0.9921, and 1.0000; §5.4.1 says the results “suggest that sequential modeling may capture locomotion kinematics.”
- **Why damaging:** Once frames from the same source video—or from the same animal/background—appear across partitions, the model can recognize the recording rather than lameness. Calling it an “upper bound” does not make it valid evidence. It also contaminates the claimed selection of EfficientNet-B0 and the comparison between spatial and temporal MTL. On the original dataset, 50 clips contain only 42 cattle and were collected from online videos, so a video-only split may still place the same animal or source context on both sides. The source paper itself reports 3D-CNN and ConvLSTM video-level results, which are the obvious baselines this report fails to reproduce ([Sohan et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12764510/)).
- **Concrete fix:** Delete all frame-level lameness scores from model-selection evidence. Group by original animal and source video/uploader where recoverable. Use nested, repeated group-stratified cross-validation: outer folds estimate performance; inner folds choose epochs, loss weights, and threshold. Report fold-level accuracy, balanced accuracy, sensitivity, specificity, F1, AUROC, and AUPRC with group-bootstrap 95% intervals. Compare directly against 3D-CNN and ConvLSTM under the same folds.

### 2. The report tunes a classification threshold on the test set

- **Exact location/quote:** §5.4.2: “After adjusting the decision threshold to 0.70 … the classes were separated perfectly,” followed by “this is considered as invalid for final evaluation because it introduces test-set leakage.” The abstract and conclusion still mention the perfect separation.
- **Why damaging:** This is not merely a limitation; it is an invalid analysis. Repeating the perfect result in the abstract gives it rhetorical weight despite admitting that it cannot estimate generalization. It also reveals that the team inspected test probabilities and modified the decision rule post hoc.
- **Concrete fix:** Remove the perfect result from the abstract, result table, conclusion, and defense slides. Retain one sentence in an “analysis error corrected before P3” note. Calibrate threshold only in each inner validation fold, lock it, then evaluate once on the corresponding outer fold. If calibration is a research question, use Platt scaling or isotonic regression on validation predictions and report Brier score and a calibration curve.

### 3. Dataset facts central to the experimental design are wrong or undocumented

- **Exact location/quote:** §4.2 says MmCows contains “213,686 cropped frames.” Table 5.8 labels Sohan et al. as “RGB-D,” and the paragraph below says Sohan used “a different dataset.” The same table calls Asim et al. “Sensor Data,” labels the metric “Accuracy / Macro F1,” and reports 91.11%. §5.5.5 says CBVD-5 is “2,000 balanced images.”
- **Why damaging:** MmCows documents about **20,000 annotated source frames and roughly 213,000 cow bounding boxes** ([primary record](https://openreview.net/forum?id=X4nq0W2qZX)). The repository does distribute cropped bounding-box images for behavior classification, so 213,686 may be a plausible count of **cow-instance crops**; the defensible criticism is that the report calls those instances “frames,” obscuring that many samples share the same timestamp, scene, camera, and cow. This is a unit-of-analysis error, not proof that every crop count is fabricated. Sohan et al. use RGB online video, not RGB-D, and report 90% 3D-CNN and 85% ConvLSTM accuracy on the **same 50-clip CattleLameness source**, not a different dataset ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12764510/)). Asim et al. is a **vision-based overhead-camera/YOLO** study; 91.11% is reported as mAP for YOLOv8-L, not ordinary activity-classification accuracy or macro-F1 ([paper](https://www.mdpi.com/1424-8220/26/3/785)). CBVD contains 687 video segments, 206,100 extracted images, 107 cows, and five named behaviors; the report never defines how its 2,000 “balanced” images were sampled or kept independent by video/cow ([CBVD paper](https://www.nature.com/articles/s41598-024-65953-x)). These errors and omissions undermine the authors’ claim of a documented public-data benchmark.
- **Concrete fix:** Create a dataset audit table from primary sources: raw units, unique animals, videos, frames, boxes, farms, cameras, dates, original labels, license, selected units, exclusion rules, and final train/validation/test group counts. Publish immutable split manifests and a script that regenerates every sample list. Correct Table 5.8 and all counts.

### 4. The actual multi-dataset MTL optimization algorithm is not specified

- **Exact location/quote:** §4.5 lists six phases and §4.4 gives `0.35 BCS + 0.35 Behavior + 0.15 Lameness + 0.15 ID`, stating the weights were chosen “empirically based on dataset size and task complexity.”
- **Why damaging:** Four datasets have different lengths, sampling units, modalities, labels, and tensor shapes; lameness consumes 20-frame sequences while the other heads are spatial. The report never says whether an iteration draws one task or all tasks, how missing labels are masked, how loaders are cycled, whether loss is averaged per frame/clip/sample, how frequently each task updates the shared encoder, whether CBAM remains trainable, or whether batch-normalization statistics are shared across domains. The equation therefore does not define an executable experiment. Dataset frequency may dominate the nominal weights. This is the core method, so non-reproducibility here is fatal.
- **Concrete fix:** Provide pseudocode with one explicit optimizer step. State task-sampling probabilities, batch sizes, epoch definition, loader cycling, label masks, loss normalization, gradient accumulation, BN mode/statistics, frozen modules per phase, optimizer state resets, scheduler, checkpoint criterion, and exact random seeds. Log per-task gradient norm, cosine similarity, update count, and effective weighted contribution.

### 5. BCS metrics are numerically incomparable and mislabeled

- **Exact location/quote:** Table 5.1 places Dryad MAE and ScienceDB MAE in adjacent columns; the abstract calls 0.5566 “BCS Mean Absolute Error”; Table 5.8 writes “90.50% Acc (±0.25) / 0.5566 MAE.”
- **Why damaging:** The implementation predicts class indices 0–4, but ScienceDB’s five levels are 3.25–4.25 in 0.25-unit steps, whereas Dryad is reported as scores 2–6 in 1-unit steps. An index MAE of 0.5566 on ScienceDB corresponds to roughly 0.139 BCS units if the mapping is linear; on Dryad, 0.6175 index units corresponds to 0.6175 BCS units. “Within one class” therefore means ±0.25 on one dataset and ±1.0 on the other. The table and cross-modality discussion invite a comparison that has no common unit.
- **Concrete fix:** Convert all predictions and errors back to the original biological score before evaluation. Label every tolerance in score units, not “one class.” Report MAE, median absolute error, quadratic weighted Cohen’s kappa, exact agreement, Bland–Altman bias/limits of agreement, and confusion matrices. Never average or rank index-scale errors across incompatible scoring protocols.

### 6. Every claimed improvement is a one-run point estimate with no valid randomness protocol

- **Exact location/quote:** §5.5: “all reported results are derived from a single run using a fixed random seed and lack variance estimates.” §4.5 instead says “single run using default PyTorch random seed.”
- **Why damaging:** “Default seed” is not a reproducibility specification, and PyTorch explicitly warns that completely reproducible results are not guaranteed across releases/platforms and requires deliberate seeding and deterministic configuration ([PyTorch reproducibility guide](https://docs.pytorch.org/docs/stable/notes/randomness.html)). Differences such as focal loss 0.7445 vs. CE 0.7074, CBAM MAE 0.6175 vs. 0.7000, and CORAL 0.5566 vs. 0.6940 may be ordinary run or split variance. No claim of “confirms,” “significant,” or “critical” is supportable.
- **Second-pass statistical finding:** The sequence accuracies change only in 10-point steps and sequence AUROCs in 0.04 steps, which is exactly consistent with a test set of 10 videos containing five positive and five negative cases (`1/(5×5)=0.04`). If so, the headline 80% accuracy is only 8/10; its approximate Wilson 95% interval is about 49%–94%. MmCows has only 16 cows, so a 70/15/15 cow-wise division leaves roughly 2–3 animals for validation and 2–3 for testing depending on rounding. Thousands of frames do not repair those tiny independent-unit counts.
- **Concrete fix:** Run at least five independent seeds on fixed group splits. For the 50-video dataset, prefer repeated nested group CV. Report mean ± SD across seeds/folds and group-bootstrap 95% CIs. Use paired permutation/bootstrap tests on identical held-out units, DeLong or stratified bootstrap for AUROC, and Holm correction across planned comparisons. Predeclare the primary metric per task.

### 7. The novelty claim is not established and is already crowded by omitted recent work

- **Exact location/quote:** §6.2: “To the best of our knowledge, this is the first end-to-end multi-task deep learning framework that simultaneously monitors BCS, behavior, lameness, and individual cow identity … using a single shared encoder.” §1.1 says the review covered “33 primary scientific sources.”
- **Why damaging:** “First exact four-label bundle” is a combinatorial claim, not necessarily a research contribution. The review omits closely adjacent work: behavior-conditioned multitask cow identification ([Hooker et al., DOI 10.3168/jds.2025-26731](https://www.sciencedirect.com/science/article/pii/S0022030225009014)), a genuinely multi-annotation cattle video benchmark covering detection/tracking/counting/pose/segmentation ([CattleEyeView](https://arxiv.org/abs/2312.08764)), large multi-camera self-supervised re-identification ([MultiCamCows2024](https://research-information.bris.ac.uk/en/publications/holstein-friesian-re-identification-using-multiple-cameras-and-se/)), and a January 2026 multimodal framework that performs cow-day health monitoring from physiological, behavioral, production, and thermal inputs against veterinarian-confirmed disorders ([Paulauskaite-Taraseviciene et al., 2026](https://doi.org/10.3390/ani16030411)). More importantly, the four tasks here are learned from disjoint datasets rather than jointly annotated animals, so shared-representation benefit is assumed rather than grounded in paired task information.
- **Concrete fix:** Replace “first” with a falsifiable scoped statement after a documented search protocol and date. State the actual delta: “a public-dataset engineering study of hard parameter sharing across four disjoint cattle tasks, with quantified negative transfer.” Compare against recent closest work and explain why disjoint-dataset MTL is scientifically interesting beyond reducing stored encoder copies.

### 8. Cow-identification evaluation is exposed to image/track leakage, and the 97.58% result is uninterpretable

- **Exact location/quote:** §4.2 says ID uses an “image-wise” split. §5.4.1 says 97.58% comes from “temporal majority voting across the 20 frames of each video sequence,” while “the frame-level ID performance itself rose unexpectedly from 86.49% … to 94.96%.”
- **Why damaging:** Near-adjacent frames of the same cow, recording, background, camera, or day can make re-identification trivial. The report does not say where OpenCows video sequences came from, how frames were linked into 20-frame units, whether any voted track overlaps training images, or how MTL causes a 8.47-point frame-level jump. The canonical OpenCows protocol should not be silently replaced with a random image split. Majority voting magnifies correlated leakage rather than providing independent evidence.
- **Concrete fix:** Use the official split where available, plus a harder day/session/camera-separated protocol. Define each tracklet and guarantee that no source video crosses partitions. Report single-frame Top-1/Top-5, tracklet-level Top-1, per-ID recall, macro-F1, and open-set rejection if deployment includes unseen cows. Compare majority vote with probability averaging and a temporal model under identical tracks.

### 9. The 75% “memory and latency” result is not measured and is technically overclaimed

- **Exact location/quote:** §5.4.1: “one shared backbone provides a 75% reduction in memory and latency.” §5.6.5 equates four EfficientNet-B0 encoders (about 21.2M parameters) with one 5.3M encoder and calls this a 75% GPU-memory reduction. §6.1 repeats “VRAM requirements by approximately 75%.”
- **Why damaging:** Parameter-count duplication is not peak VRAM: activations, heads, detector, LSTM, buffers, input streams, framework overhead, and precision matter. Nor does loading one set of weights eliminate forward passes from two cameras or four different temporal/spatial inputs. Latency can remain similar or worsen through task heads, buffering, YOLO, and scheduling. Comparing 0.39 GFLOPs to “40 TOPS” also mixes floating-point operations with hardware’s precision-specific peak throughput.
- **Concrete fix:** Rename the current result “75% reduction in duplicated encoder parameters.” Benchmark total parameters, serialized size, peak allocated VRAM/RAM, end-to-end latency, throughput, energy, and sustained temperature on the claimed device at stated precision and batch size. Include detection, cropping, 20-frame buffering, all active heads, and both cameras. Compare four separate processes, one multi-head model, and a simple shared-feature cache.

### 10. “Gradient conflict” is asserted as a causal result without measuring gradients

- **Exact location/quote:** §5.4.1 calls the performance drop “an obvious example of gradient conflict”; §5.6.1 says updates “directly conflict” and sharing “inevitably degrades both.”
- **Why damaging:** Performance degradation alone does not identify the cause. It could arise from loss-scale mismatch, task sampling frequency, dataset-domain BN contamination, catastrophic forgetting during sequential phases, different checkpoint criteria, reduced task-specific capacity, poor task weights, or hyperparameter unfairness. “Inevitable” is false in principle and contradicted by extensive MTL literature on selective sharing and task affinity.
- **Concrete fix:** Measure pairwise gradient cosine similarity and conflict frequency at shared layers, per-task gradient norms, and task-wise loss trajectories. Run equal-weight, uncertainty-weighted, GradNorm, and PCGrad baselines; add task-specific BN/adapters and grouped task variants. Use Standley et al.’s task-affinity framing and only call it gradient conflict if the gradients are measured ([Standley et al., 2020](https://proceedings.mlr.press/v119/standley20a.html)).

### 11. Backbone selection is a confounded rank exercise, not a fair experiment

- **Exact location/quote:** §5.3/Table 5.6 calls EfficientNet-B0 the “highest overall rank” but supplies no ranking equation. ID models use a fixed 10-epoch budget and §5.2.4 admits larger ResNets have not converged; lameness selection includes the contaminated frame-level metric; CBAM placement across candidates is unclear.
- **Why damaging:** A model cannot be selected on a metric known to leak, nor by comparing architectures trained for unequal convergence behavior. If only EfficientNet receives CBAM in some tables, architecture and attention are confounded. Choosing the winning backbone on the same held-out test sets also turns the test set into validation data.
- **Concrete fix:** Select hyperparameters/backbone only on validation data under a fixed compute budget or convergence-matched protocol. Define a precommitted multi-objective rule including accuracy and measured efficiency. Evaluate the selected configuration once on untouched test groups. Apply identical heads, attention, augmentation, optimizer search space, and early-stopping rule to every candidate.

### 12. The behavior experiment does not support its imbalance and generalization claims

- **Exact location/quote:** §5.5.4: focal loss with `γ=2, α=0.25` “reduced the influence of majority classes”; §5.2.2 reports “per-class accuracy”; §5.5.5 reports CBVD class “accuracy” and macro-F1 0.1245.
- **Why damaging:** A scalar α=0.25 applied uniformly does not rebalance classes; it rescales the entire focal objective. The focusing term may help hard examples, and per-epoch capping changes the distribution, but the report attributes the gain to class weighting it did not implement. “Per-class accuracy” appears to mean recall, not accuracy. No precision, support, confusion matrix, or AUPRC is shown despite the evaluation protocol promising them. Cross-dataset labels are not equivalent—CBVD’s “foraging” is apparently mapped to “feeding”—and the 2,000-image subset may contain adjacent frames from the same video.
- **Concrete fix:** Define class-specific α values or compare class-balanced loss, balanced sampler, logit adjustment, and capped sampling separately. Report precision/recall/F1/support for every class and macro/micro/weighted summaries. For CBVD, publish the label ontology and mapping, sample at video level, evaluate the full eligible set, and state which classes are excluded.

### 13. The CORAL implementation may not be rank-consistent

- **Exact location/quote:** §4.3 says the BCS head is a “Linear layer outputting four ordinal logits”; Eq. 4.4 defines `y_i = I(y > i)` for `i = 1,…,K−1`; §5.1 says the actual class indices are `0–4`.
- **Why damaging:** There are two separate defects. First, genuine CORAL is not merely four independent BCE outputs: rank consistency relies on a constrained/shared weight structure with ordered biases. A generic `Linear(..., 4)` can produce rank violations. Second, the written indexing is off by one if labels really are 0–4. Four cumulative targets should correspond to `I(y>0), I(y>1), I(y>2), I(y>3)`. The report instead writes `I(y>1)…I(y>4)`, which collapses classes 0 and 1 and makes the top class unreachable by the sum rule. This may be only a notation error, but until code is shown it is also a plausible implementation error ([Cao, Mirjalili & Raschka, 2020](https://doi.org/10.1016/j.patrec.2020.11.008)).
- **Concrete fix:** Show the exact target-encoding and layer code. Add unit tests for all five labels showing their expected four-bit targets and decoded predictions. Use a vetted CORAL layer, enforce/verify ordered logits, report rank-violation frequency, and compare against independent cumulative BCE, categorical CE, and scalar regression on identical seeds/groups.

### 14. The stated LSTM dropout likely does nothing

- **Exact location/quote:** §4.3 specifies “a single-layer LSTM with hidden size 256 and dropout 0.5.”
- **Why damaging:** In PyTorch, `nn.LSTM(dropout=p)` applies dropout between recurrent layers except after the last layer; with one layer, the internal dropout parameter has no operative inter-layer location ([PyTorch LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)). The report therefore claims regularization that likely was never applied.
- **Concrete fix:** Add explicit dropout to encoder features and/or LSTM output, or use two recurrent layers if justified. State the exact forward pass and verify with code/tests that dropout changes outputs during training.

### 15. The “cross-dataset modality ablation” cannot identify a modality effect

- **Exact location/quote:** §5.5.3 says the DGE result shows “depth contours … helping to offset the significantly smaller training sample size”; §5.6.4 admits datasets differ in size, breed, environment, and annotation.
- **Why damaging:** The caveat directly destroys the causal claim. Different datasets, breeds, score scales, sample dependence, modalities, labels, and backbones prevent attribution to depth. This is not an ablation and not even a clean generalization test because the target definition differs.
- **Concrete fix:** Rename it “descriptive cross-dataset result” and remove all causal language about depth compensating for data. A valid modality ablation needs paired RGB and depth/DGE from the same cows and partitions, with identical model/training and RGB-only, depth-only, early-fusion, and late-fusion conditions.

### 16. The claimed open benchmark is not currently reproducible

- **Exact location/quote:** §6.2 calls the work a “transparent and reproducibility-oriented benchmark via documented preprocessing and task-specific splits”; §4.5 says the repository “will be released.”
- **Why damaging:** Public source data alone does not make the experiment reproducible. The report lacks code, split manifests, dependency lockfile, seed values, data checksums, crop-generation details, best-checkpoint rules, complete hyperparameters, model selection protocol, and the MTL batch schedule. Some source datasets may also carry redistribution or usage conditions that must be respected.
- **Concrete fix:** Release code before making the benchmark claim. Include environment lockfile/container, config files, exact commit, seed list, checksums, licenses/terms, download instructions, deterministic split manifests, training logs, and scripts that regenerate all tables from saved predictions.

### 17. The model never learns the health relationships used to justify MTL

- **Exact location/quote:** §1.2 justifies sharing because “physical well-being, such as lameness, influences feeding behavior and affects body condition.” §4.2 then assigns every task to a different dataset, and §5 reports no sample carrying more than one of the four target labels.
- **Why damaging:** Dataset identity is perfectly confounded with task identity: ScienceDB/Dryad means BCS, MmCows means behavior, CattleLameness means lameness, and OpenCows means ID. The model never sees whether a particular cow’s lameness co-occurs with changed behavior or body condition, so it cannot exploit the biological correlation used as motivation. It shares parameters across domains; it does not learn a joint cattle-health state or fuse the four outputs. Hard sharing may therefore learn dataset/camera/modality routing rather than transferable health features.
- **Concrete fix:** Reframe the current work as **multi-domain, multi-head parameter sharing across separately labeled tasks**. Add a task/domain-discrimination probe to test whether the shared representation mainly encodes dataset identity. If “integrated health monitoring” remains the claim, collect or use a cohort with synchronized cow IDs and at least two co-labeled outcomes, then evaluate whether one task improves another.

### 18. The preprocessing may destroy the lameness signal the LSTM is claimed to learn

- **Exact location/quote:** §4.1 crops each detected cow; §4.2 resizes every crop to `224×224`; the 20 frames are sampled at equal relative positions regardless of clip duration; §5.4.1 says the LSTM captures “locomotion kinematics and temporal context.”
- **Why damaging:** Independently detecting, cropping, and square-resizing every frame can normalize away absolute translation, change body/stride aspect ratios, and inject detector-box jitter. Uniformly sampling 20 frames from clips of different duration means one recurrent step has no fixed physical time. No FPS, elapsed duration, stride cycle, optical flow, bounding-box trajectory, or pose coordinate is supplied. The model may classify appearance/background across 20 images, but the report has not established that it measures kinematics.
- **Concrete fix:** Preserve aspect ratio with padding; retain timestamps, original crop scale, box-center trajectory, and camera calibration where available. Compare independent resized crops against fixed-window crops, optical flow, pose/stride features, and a contiguous clip model. Report clip duration/FPS and evaluate sensitivity to sampling rate and sequence length.

### 19. The headline “health monitoring” endpoints do not actually test poor-condition or abnormal-behavior detection

- **Exact location/quote:** §1.1 motivates “early detection of health problems like poor body condition … and abnormal behavior.” ScienceDB—the source of the headline BCS result—contains only 3.25–4.25 scores. The behavior head predicts seven ordinary activities but has no illness/anomaly label, longitudinal baseline, or clinical outcome.
- **Why damaging:** A classifier restricted to relatively high/narrow BCS values cannot demonstrate detection of thin cows, obese extremes, or clinically actionable change. Naming normal activities is not equivalent to detecting abnormal behavior or illness; the report never defines abnormal duration/frequency, establishes an individual baseline, or links predictions to diagnosis. Cow ID is an enabling function, not a health endpoint. The title and motivation therefore claim a health system that the evaluated labels do not instantiate.
- **Concrete fix:** Narrow the claim to “cattle visual attribute and activity monitoring,” or add clinically relevant endpoints: extreme BCS ranges, longitudinal BCS change, abnormal activity duration, and a prospectively defined health event. Specify the decision supported, alert threshold, time horizon, sensitivity, and false-alert cost.

### 20. Repeated use of the same test sets makes them development sets

- **Exact location/quote:** Tables 5.1–5.6 use test results from five backbones to select EfficientNet-B0; §5.5 then reports test results for CORAL/CE, CBAM/no-CBAM, focal/CE, and cross-dataset choices; Table 5.7 evaluates final MTL configurations on the same named datasets.
- **Why damaging:** Even if model weights never directly see test labels, repeatedly inspecting test performance to choose backbone, attention, loss, narrative, and final architecture adaptively overfits the research process to those test sets. The final Table 5.7 is therefore not a single untouched evaluation. This is test-set leakage at the experiment-selection level.
- **Concrete fix:** Develop every architecture and ablation using training/validation groups only. Lock one final configuration and analysis plan, then evaluate once on a fresh held-out group set. If no fresh data remain, call current results development/validation results and create a new outer test fold or nested-CV estimate for P3.

## 3. MODERATE ISSUES

### 1. The problem statement confuses model consolidation with simultaneous monitoring

- **Location/quote:** §1.3 says no framework addresses all four tasks “together”; §5.6.3 later requires different rear-view and side-view cameras and activates heads selectively.
- **Damage:** The system does not obtain four labels from a common observation, and disjoint datasets prevent evidence that the tasks help one another. It is closer to a shared model package than a unified measurement model.
- **Fix:** Define “unified” operationally: shared stored parameters, shared runtime process, which heads run on which camera, and whether outputs are synchronized per cow/time.

### 2. The motivation statistics are weakly or incorrectly sourced

- **Location/quote:** §1.1 attributes “global cattle population is over one billion” to FAO report [7] and “Up to 25%” lameness to Whay et al. [2].
- **Damage:** [7] is a broad SDG livestock report, not the clean primary source for a current cattle census. A 2003 welfare-assessment paper is not an adequate current global prevalence source.
- **Fix:** Use current FAOSTAT livestock stocks and a current systematic prevalence review; specify country, production system, case definition, and uncertainty.

### 3. “Behavior changes before clinical symptoms” is too broad for the cited review

- **Location/quote:** §1.1 attributes the assertion to von Keyserlingk et al. [3].
- **Damage:** A broad welfare review does not establish that every listed behavior reliably precedes clinically detectable illness or by how long.
- **Fix:** Cite condition-specific prospective studies and state the target condition, temporal lead, sensor/camera modality, and false-alarm tradeoff.

### 4. The report promises methods it does not implement

- **Location/quote:** §1.6 says the framework “applies … domain adaptation approaches” and “gradient normalization during fine-tuning.”
- **Damage:** Results use neither domain adaptation nor GradNorm; §6.4 lists them as future work.
- **Fix:** Change present tense to planned future work or implement them in P3.

### 5. Scope contradicts the actual modality choice

- **Location/quote:** §1.6 says “RGB imaging is the primary input modality” for Dryad and depth is auxiliary; §5.2.1 reports DGE inputs.
- **Damage:** DGE is not plain RGB, and the report’s modality narrative shifts between scope and experiment.
- **Fix:** Precisely state which Dryad channel representation was used in each run and why.

### 6. “Real-time” and “edge-deployable” are labels without measurements

- **Location/quote:** Objectives and conclusion repeatedly use “real-time” and “edge-deployable”; §6.3 concedes that latency and energy are estimated, not measured.
- **Damage:** A parameter cap does not demonstrate a real-time pipeline, especially with YOLO, two cameras, 20-frame windows, and multiple heads.
- **Fix:** Until hardware results exist, use “edge-oriented design target.” Define latency/FPS requirements and measure them.

### 7. Economic claims are invented rather than analyzed

- **Location/quote:** §3.8 gives RFID at `$80–120/cow`, cameras at `$150`, edge device at `$300`, total `$1,500`, and “>90%” cost reduction.
- **Damage:** No citations, camera count derivation, installation, enclosure, networking, lighting, calibration, power, maintenance, replacement, labor, or depreciation are included. “Zero recurring cloud cost” ignores local maintenance and storage.
- **Fix:** Cite vendor/market sources, define a farm size and camera topology, calculate total cost of ownership over 3–5 years, and give a sensitivity range.

### 8. Hardware feasibility compares incompatible quantities

- **Location/quote:** §3.1 compares EfficientNet’s ~0.39 GFLOP/frame and 7.8 GFLOP/sequence with Jetson Orin Nano “40 TOPS.”
- **Damage:** TOPS is typically a precision-dependent theoretical peak, often INT8; model GFLOPs may be FP32/FP16 and exclude detector/heads/memory movement.
- **Fix:** Benchmark the actual exported model at a declared precision using TensorRT/ONNX, batch size 1, full pipeline, and sustained load.

### 9. The selected 20-frame center-sampling policy is arbitrary

- **Location/quote:** §4.2 divides every clip into 20 segments and takes the center frame.
- **Damage:** It discards variable gait-cycle timing, may miss short behaviors, and lacks comparison to contiguous clips, random segment sampling, or motion-aware sampling.
- **Fix:** Ablate frame count and sampling scheme; report physical duration and FPS; consider temporal jitter during training and deterministic uniform sampling at test time.

### 10. Batch normalization across heterogeneous domains is ignored

- **Location/quote:** The shared EfficientNet-B0 is trained across RGB, DGE-like, farm-video, rear-view, and side-view data with no BN policy.
- **Damage:** Shared BN running statistics can create domain interference even when gradients are not intrinsically conflicting.
- **Fix:** Test frozen ImageNet BN, task-specific BN, GroupNorm, or domain-specific adapters.

### 11. Single-task versus MTL comparison may not be budget-matched

- **Location/quote:** §4.5 uses task-specific phases plus joint fine-tuning; single-task budgets vary by task.
- **Damage:** Extra exposure to a task during multiple phases can inflate or degrade performance independently of sharing.
- **Fix:** Match optimizer steps and augmentation exposures per task, and compare both equal-compute and equal-convergence settings.

### 12. The lameness endpoint lacks credible ground truth

- **Location/quote:** The dataset is presented as binary lameness labels; §6 calls deployment “clinical.”
- **Damage:** The source dataset was labeled from visual gait characteristics and online metadata, not a blinded veterinary gold standard. Binary online-video labels cannot justify clinical language.
- **Fix:** Say “screening” or “research classification.” Document annotators, scoring rubric, agreement, uncertainty, and ideally obtain veterinary re-annotation of the 50 clips.

### 13. The behavior head is treated inconsistently as spatial and temporal

- **Location/quote:** §4.3 describes a spatial seven-logit head; §5.4.1 attributes behavior improvement to temporal modeling, but the architecture description gives only one LSTM lameness head.
- **Damage:** It is unclear whether behavior consumes frame features, LSTM features, averaged logits, or another sequence aggregator.
- **Fix:** Draw the exact computation graph and tensor shapes for both frame-level and spatiotemporal configurations.

### 14. Identification gains may reflect shared label/domain artifacts rather than beneficial MTL

- **Location/quote:** §5.4.1 calls the 86.49→94.96 rise “unexpected” and leaves it unexplained.
- **Damage:** An unexplained nine-point jump is a warning signal, not a result to advertise. It may expose split, checkpoint, preprocessing, or bookkeeping differences.
- **Fix:** Reproduce with fixed split/seed; verify predictions and sample IDs; run ID-only with the identical preprocessing, epochs, and backbone state used by MTL.

### 15. The conclusion uses causal and superlative language beyond the evidence

- **Location/quote:** §6.1: “successfully developed”; “LSTM layers is highly effective”; “outstanding computational efficiency.”
- **Damage:** The MTL model is materially worse on two major tasks, lameness is tiny/unstable, and efficiency is estimated.
- **Fix:** Replace with neutral quantitative statements and reserve conclusions for independently supported findings.

### 16. Ethical and privacy analysis is superficial

- **Location/quote:** §3.4 says visual monitoring avoids pain and local processing protects privacy.
- **Damage:** Cameras can capture workers and visitors; animal IDs and farm operations can be commercially sensitive; downloaded online videos may have licensing/consent constraints. RFID harms are asserted without a risk comparison.
- **Fix:** Add data provenance/license matrix, human bystander handling, retention/access policy, threat model, model-error consequences, and human-in-the-loop escalation.

### 17. Claimed “standards” are not standards

- **Location/quote:** §3.5 is titled “IEEE Software Quality Standards” but discusses PEP 8 and performance metrics.
- **Damage:** PEP 8 is a Python style guide, not an IEEE software-quality standard; metrics are not standards compliance.
- **Fix:** Either delete the section or cite/apply actual standards such as ISO/IEC 25010 for quality models and relevant AI risk/data governance standards.

### 18. Detector performance is absent from the end-to-end evaluation

- **Location/quote:** §4.1 and §4.5 use YOLOv8-nano to detect/crop cattle; all task results appear to use prepared crops.
- **Damage:** Deployment accuracy depends on missed detections, truncation, multi-cow association, and crop jitter. Crop-level task metrics do not estimate system performance.
- **Fix:** Report detector dataset, split, mAP/recall, tracking/association method, and end-to-end task performance on raw scenes.

### 19. The stated “under 10 million parameters” scope is internally violated

- **Location/quote:** §1.6 says backbone selection is “limited to architectures with fewer than 10 million parameters”; Tables 5.1–5.6 evaluate ResNet-18 and ResNet-50. §3.1 and §6.2 then claim the **total model** remains under 10M.
- **Damage:** Standard ResNet-18 is roughly 11.7M parameters and ResNet-50 roughly 25.6M, so the claimed selection constraint is false as written. The total-pipeline claim is also unproven: EfficientNet-B0 (~5.3M) + YOLOv8-nano (~3.2M) + a 1280→256 LSTM (~1.57M) already approaches or exceeds 10M before CBAM and all heads, depending on exactly what is counted. Excluding the detector while calling the result “total model size” is deceptive accounting.
- **Fix:** Publish a component-wise parameter table generated from the exact code: detector, encoder with classifier removed, CBAM, LSTM(s), each head, and total. State whether the <10M constraint applies to the classifier module or the deployable pipeline. Rewrite the scope accordingly.

### 20. The CBAM ablation ignores that EfficientNet-B0 already contains attention

- **Location/quote:** §5.5.2 calls the comparison “with CBAM and without CBAM” and concludes that “targeted spatial attention” improves BCS.
- **Damage:** EfficientNet-B0’s MBConv blocks already contain squeeze-and-excitation channel attention. The actual comparison is therefore **SE-only EfficientNet versus SE+CBAM**, not attention versus no attention. The report offers no reason that a second channel gate after the final 7×7 feature map is the right placement, and no attention visualization verifies that CBAM focuses on spine/hooks/pins.
- **Fix:** Describe the baseline correctly. Compare SE-only, spatial-only CBAM, channel-only CBAM, full CBAM, and a parameter-matched non-attention layer. Add Grad-CAM/attention-map sanity checks and multi-seed uncertainty before making anatomical claims.

### 21. ImageNet normalization is unjustified for DGE pseudo-channels

- **Location/quote:** §4.2 defines DGE as grayscale intensity, depth coordinates, and Canny edges packed into three channels, then says all samples use ImageNet RGB means/stds.
- **Damage:** Those three channels do not represent red, green, and blue and have radically different ranges/distributions. Applying RGB channel statistics—especially to 16-bit depth and sparse binary-like edges—can produce arbitrary scaling. The Dryad source explicitly warns users to normalize each depth image carefully before 8-bit conversion to avoid corruption ([Dryad usage notes](https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw4s)).
- **Fix:** Document depth datatype/conversion, missing-depth handling, Canny parameters, and per-channel distributions. Compare modality-specific normalization against ImageNet normalization, and explain how pretrained first-layer filters are adapted to non-RGB semantics.

### 22. The “systematic” literature review has no reproducible search method

- **Location/quote:** §1.1 says “a literature review comprising 33 primary scientific sources”; §1.2 claims 88% (29/33) used private data; Chapter 2 calls the review systematic; the bibliography contains 44 mixed entries.
- **Damage:** No databases, search strings, date range, inclusion/exclusion criteria, screening process, duplicate handling, quality assessment, or source-by-source public/private classification is provided. The denominator of 33 cannot be reconstructed from 44 references. The 88% statistic and “no existing paper” novelty claim are therefore unauditable.
- **Fix:** Add a compact systematic-search appendix with search date, databases, exact queries, screening flow, included-paper list, and coded columns supporting 29/33. Use the 2025 survey of 67 public PLF computer-vision datasets as a starting point rather than an undocumented manual sample ([Bhujel et al., 2025](https://doi.org/10.1016/j.compag.2024.109718)).

### 23. Architecture performance is confounded with team member and implementation

- **Location/quote:** §5.3: “The five team members trained single-task baseline models on their assigned architectures”; Tables 5.1–5.6 identify each backbone by member.
- **Damage:** Unless every run uses the same data manifests, code path, augmentations, optimizer, scheduler, precision, checkpoint rule, head, and attention configuration, the comparison measures both architecture and operator-specific implementation. The ID table explicitly gives CBAM only to EfficientNet-B0, confirming at least one configuration difference.
- **Fix:** Run every backbone through one config-driven training script and one evaluator. Version-control configs and predictions; do not use member assignment as an experimental factor.

### 24. Figures 5.2–5.4 claim validation evidence that is not plotted

- **Location/quote:** Text before Figure 5.2 calls it the “training and validation loss trajectory,” but the plot/caption show only training loss. Figure 5.3’s caption says “Training and validation loss,” yet the plot contains one line labeled training loss. Figure 5.4 repeats the same caption error and also contains only training loss.
- **Damage:** §5.2.2 says validation macro-F1 stabilized; §5.2.3 infers “without significant divergence” and greater stability; §5.2.4 discusses validation convergence. Training loss alone cannot support any of those generalization or early-stopping claims.
- **Fix:** Regenerate each figure with clearly labeled train and validation curves for the actual checkpoint-selection metric, mark the selected epoch, and include confidence bands across seeds. If validation logs do not exist, remove the claims rather than reconstructing them after the fact.

### 25. The temporal architecture and labels are still undefined

- **Location/quote:** §4.3 says behavior sequences pass through “the shared LSTM pathway,” then separately names a “Lameness Head (Temporal LSTM).” Behavior sequences are made by grouping “consecutive sampled MmCows frames,” but no sequence-label rule is stated.
- **Damage:** It is impossible to tell whether behavior and lameness share one LSTM, have two LSTMs, or merely share the encoder. If behavior changes within 20 frames, the ground-truth sequence label is undefined. Overlapping windows could also cross split boundaries or make test examples nearly duplicates.
- **Fix:** Publish the computation graph and tensor shapes; define window stride, elapsed time, sequence inclusion criteria, label aggregation, transition handling, and overlap constraints. State exactly which recurrent parameters are shared.

### 26. The identification system is closed-set while the deployment claim is open-world

- **Location/quote:** §4.2 states all 46 OpenCows identities occur in train, validation, and test; the ID head is a 46-class linear softmax.
- **Damage:** A softmax classifier cannot enroll a new cow without changing/retraining the head and cannot reliably reject an unknown cow. Real herds add, sell, and move animals. Recent benchmarks explicitly separate verification, identification, limited-data, and unseen-identity protocols, exposing how weak a random within-ID image test is ([ReCowGnition, 2026](https://arxiv.org/abs/2607.22071)).
- **Fix:** Either state that ID is closed-set only or switch to metric learning/prototypical enrollment with an unknown threshold calibrated on validation identities. Evaluate known-ID identification, unseen-ID verification, new-cow enrollment, and cross-camera/session shift separately.

### 27. The live system has no identity-preserving tracker or cross-camera association

- **Location/quote:** §4.5.3 queues 20 crops and votes for cow ID; §5.6.3 uses separate rear- and side-view cameras but says only that task heads are activated selectively.
- **Damage:** A 20-frame buffer assumes the same animal has already been associated across frames. With multiple cows, detection alone cannot prevent identity switches. Two cameras also require cross-view/time association before BCS, behavior, and lameness outputs can be attached to one cow. Without this, “unified” per-cow monitoring is operationally undefined.
- **Fix:** Add ByteTrack/DeepSORT or another justified tracker, report IDF1/HOTA/ID switches, and define cross-camera association/synchronization. Rao et al. provide a current cattle-specific example coupling YOLO detection with ByteTrack for persistent identity ([Rao et al., 2026](https://www.nature.com/articles/s44433-026-00004-x)).

### 28. The temporal-benefit conclusion is selectively framed

- **Location/quote:** §5.4.1 is titled “Sequence Tasks Benefit from Temporal Modeling”; §6.1 says LSTM modeling is “highly effective.” Table 5.7 shows lameness accuracy falling from 95.28% spatial MTL to 80.00% temporal MTL; the ID gain is explicitly majority voting, not LSTM.
- **Damage:** The only direct improvement attributed to the LSTM is behavior from 0.3771 to 0.4948, still far below the 0.7445 single-task model. The lameness comparison is invalid because the spatial result leaks, but that means it cannot be used either for or against temporal modeling. The evidence does not establish a general sequence benefit.
- **Fix:** Replace the conclusion with task-specific language. Compare temporal mean pooling, LSTM, 3D-CNN, SlowFast/TimeSformer or another feasible video baseline under the same leakage-free video splits; report per-task uncertainty.

### 29. Table 5.4 contains two unexplained ResNet18-LSTM results

- **Location/quote:** Table 5.4 lists `ResNet18-LSTM (Sequence)` under Hasin as AUC 0.8800/accuracy 70%, then lists another `ResNet18-LSTM (Sequence)*` under Nusrat as AUC 0.9600/accuracy 60%.
- **Damage:** No architectural, split, seed, preprocessing, or ownership difference explains why the same named model appears twice with materially different results. The asterisk points to a footnote about **frame-level spatial leakage** even though it is attached to a sequence row. This looks like a copy/paste or bookkeeping error and weakens confidence in the entire result table.
- **Fix:** Trace both rows to exact run IDs/configs and rename them with the true difference, or delete the erroneous duplicate. Make footnote markers attach only to the results they qualify.

## 4. MINOR/POLISH ISSUES

- Abstract and conclusion should not repeat the invalid test-tuned “perfect separation,” even with caveats.
- §1.2 says “The proposed method is not utilizing Multi-Task Learning”; context suggests “existing methods.” As written, it directly contradicts the thesis.
- “Clinical deployment” is inappropriate for a farm-screening prototype without clinical-grade ground truth or prospective validation.
- “Recurrent 3D time-series” in §6.2 is technically wrong; an LSTM over feature vectors is one-dimensional temporal sequence modeling, not 3D convolutional modeling.
- The exact performance-drop percentages disagree: §5.4.1 gives 40.6%/33.5%; §5.6.1 gives 41.0%/35.8%. The behavior change from 0.7445 to 0.4948 is 33.5% relative, not 35.8%.
- Table 5.3 and Figure 5.6 use “per-class accuracy” where “recall” appears intended.
- Table 5.7 mixes frame-level and video-level endpoints in one row and uses an asterisk without a sufficiently local definition.
- Figure 5.5 compares heterogeneous metrics on one axis and omits BCS; this is visually misleading.
- Figure 5.1 shows training loss approaching zero while validation diverges; identify the selected checkpoint and epoch directly on the graph.
- Report test-set sample counts and supports next to every metric.
- The 70/15/15 split statement is inconsistent with lameness accuracy changing in apparent 10-point increments; disclose actual counts.
- Replace “possible” leakage with confirmed protocol diagnosis where frames are knowingly split from the same video.
- Remove unused nomenclature entries (e.g., GAN, DWT, BERT-like generic terms if absent) rather than padding the abbreviation list.
- Standardize “Acknowledgment” versus “Acknowledgement.”
- Fix grammar throughout: “trained to multiple related tasks,” “made the model to learn,” “which is the model achieved,” and similar constructions.
- Reference metadata and punctuation are inconsistent; add volumes, issues, pages/article numbers, publication type, and access dates consistently.
- Do not call a Kaggle mirror the primary MmCows source when a paper/repository exists.
- Explain whether “53,566 images” are independent images, video frames, augmented crops, or repeated observations of the same cows.
- Avoid “SOTA” unless the comparison uses the same dataset, split, metric, and conditions.
- A 10M-parameter FP16 model is roughly 20 MB for weights, not 40 MB; distinguish FP32, FP16, and total runtime memory.

## 5. ADD

### A. Leakage-resistant evaluation package

Add immutable CSV/JSON manifests with `dataset`, `sample_id`, `animal_id`, `video_id`, `farm`, `camera`, `date/session`, `label`, and `split`. Assert in code that no protected group overlaps. For datasets without reliable animal identifiers, group at least by source video/recording and state the residual risk.

### B. Five-seed and grouped-uncertainty analysis

For BCS, behavior, and ID, run five seeds on a frozen group split. For lameness, use repeated nested group CV because 50 videos cannot support a stable single holdout. Report seed/fold distributions, 95% group-bootstrap intervals, and paired tests.

### C. Honest MTL baseline matrix

Run the following under matched task exposure and backbone initialization:

1. Four independent single-task models.
2. Hard sharing with equal weights.
3. Current static weights.
4. Uncertainty weighting.
5. GradNorm.
6. PCGrad.
7. Shared backbone with task-specific BN or small adapters.

The point is not to stack techniques; it is to separate loss scaling, gradient geometry, and domain-statistics failure.

### D. Gradient-conflict diagnostics

Log pairwise cosine similarity, conflict rate, gradient norm, and update magnitude for each task at early/middle/late backbone blocks. Plot these over training. This turns “negative transfer” from storytelling into an observed mechanism.

### E. Simpler baselines

- BCS: mean/median class, scalar regression, categorical CE, true CORAL.
- Behavior: majority class, class-weighted CE, balanced sampling, focal loss, temporal mean pooling.
- Lameness: majority class, optical-flow summary + logistic regression, CNN mean-pooling, original 3D-CNN, original ConvLSTM.
- ID: nearest-neighbor/metric-learning embedding, official OpenCows baseline, probability averaging versus vote.

### F. Correct task-specific metrics

- **BCS:** score-unit MAE/median AE, quadratic weighted kappa, exact and clinically meaningful tolerance agreement, Bland–Altman.
- **Behavior:** macro-F1 primary; per-class precision/recall/F1/support; balanced accuracy; confusion matrix; AUPRC for rare classes.
- **Lameness:** balanced accuracy, sensitivity, specificity, F1, AUROC, AUPRC, Brier/calibration; group-level CIs.
- **ID:** Top-1/Top-5, macro-F1, per-ID recall, tracklet-level versus frame-level, open-set test if claimed.

### G. Real cross-domain behavior experiment

Define a shared label ontology between MmCows and CBVD, map only semantically equivalent classes, hold out entire CBVD videos/cows, and use all eligible examples. Compare source-only, simple color/contrast normalization, AdaBN/task-specific BN, and one domain-adaptation baseline. Do not tune on the target test labels.

### H. End-to-end raw-video test

Include YOLO detection and cow association. Report failure cascades: detection miss, wrong cow crop, task-head error, and temporal association error. This is necessary for any real-time/deployment claim.

### I. Actual efficiency benchmark

Benchmark on the exact target hardware at batch size 1: total parameters, serialized weight size, peak RAM/VRAM, median/P95 latency, FPS, power, and temperature after sustained operation. Test one and two camera streams, FP32/FP16/INT8 if supported, and include YOLO and buffering.

### J. Reproducibility artifact

Release the repository, dependency lockfile, configs, split manifests, data checksums, seed list, logs, saved test predictions, and one command that regenerates each table. Add a data/license card and model card.

### K. Task-versus-domain representation audit

Train a frozen-feature linear probe to predict **dataset/domain identity** from the shared embedding, and compare its accuracy with task-relevant probes. Add centered-kernel alignment (CKA) or another representation-similarity analysis across task batches. If domain identity is nearly perfectly decoded, test task-specific BN/adapters and pairwise task sharing before claiming that biological task relationships drive the representation.

### L. Code-level correctness tests before more training

Add unit tests for all five CORAL target encodings/decodings, the one-layer LSTM dropout behavior, component-wise parameter totals, split-group overlap, sequence-window overlap, and metric calculations. These checks are cheaper and higher priority than another expensive full run because they can invalidate every later experiment if wrong.

## 6. CHANGE

1. **Change the thesis claim.** Current: “first unified deployment-ready four-task framework.” Defensible P3 version: “A leakage-controlled evaluation of hard parameter sharing across four public cattle-monitoring tasks, exposing task/domain interference and testing mitigation strategies.”
2. **Change the evaluation unit from frame to independent group.** The statistical unit is cow/video/session/farm—not an extracted frame.
3. **Change every BCS metric to original score units.** Keep class-index outputs internal only.
4. **Change the lameness design to nested group CV.** Remove any test-set threshold choice.
5. **Change ID splitting to official or session/video-separated splits.** Define tracklets before temporal voting.
6. **Change the MTL method section into executable pseudocode.** Include task sampling and missing-label handling.
7. **Change causal language.** “Associated with degradation” until gradient conflict is measured; “parameter duplication reduction” until memory/latency is measured.
8. **Change literature-review organization.** Replace one-paragraph paper summaries with comparison dimensions: task, data unit, group split, public availability, modality, metric, external validation, and code.
9. **Change Table 5.8.** Either reproduce prior methods on the same data or remove the numeric comparison. Correct Sohan’s RGB modality and published results.
10. **Change behavior imbalance handling.** A uniform α is not class weighting. Separate focal focusing, class weights, and sampling/capping.
11. **Change edge language.** Use “candidate for edge optimization” until full-pipeline hardware measurements exist.
12. **Change ethics from generic benefits to risk controls.** Cover workers, farm confidentiality, video licenses, false negatives/positives, and human review.
13. **Change “health monitoring” to the measured claim.** Until clinically actionable change/abnormality is evaluated, call the outputs BCS estimation, activity classification, lameness screening, and closed-set identification.
14. **Change “213,686 frames” to the verified data unit.** Distinguish source frames, cow bounding boxes, and exported crop instances; report all three counts where used.

## 7. CUT

- Cut the post-hoc “perfect” lameness result from all headline locations.
- Cut contaminated frame-level lameness from backbone selection and performance claims.
- Cut “75% latency/VRAM reduction”; retain only the arithmetic about duplicated encoder parameters until measured.
- Cut “first” unless supported by a documented systematic search and a carefully bounded definition.
- Cut the “cross-dataset modality ablation” as an ablation; retain only a descriptive note.
- Cut irrelevant literature padding on pears [36], poultry review [39], tomato disease [40], pregnancy ultrasound [30], and ewe carcass traits [42] unless the text extracts a method directly reused and explains transferability.
- Cut the contextual SOTA number table if datasets/splits/metrics remain non-equivalent.
- Cut unsupported economic savings and “zero recurring cost.”
- Cut federated learning and Unreal Engine synthetic cattle from the immediate P3 plan. They are scope explosions that do not repair current validity.
- Cut broad claims that behavior change precedes clinical illness unless tied to a condition and validated lead-time evidence.
- Cut unused abbreviations and generic textbook background that does not lead to an experimental decision.
- Cut the word “systematic” from the literature review unless a reproducible search/screening method is added.
- Cut claims that the current model learns relationships among BCS, behavior, lameness, and identity; the datasets contain no jointly labeled cows to test that proposition.

## 8. LITERATURE AUDIT TABLE

“Accurately represented?” evaluates how the report uses the source, not whether the bibliography metadata is perfect. “Partial” includes overbroad inference, incomplete characterization, or use only as generic background.

| Citation | Real? | Accurately represented? | Still relevant? | Verdict |
|---|---:|---|---|---|
| [1] [Edmonson et al., *A body condition scoring chart for Holstein dairy cows* (1989)](https://doi.org/10.3168/jds.S0022-0302(89)79081-0) | Yes | Mostly; foundational 1–5 chart, but does not validate this model/dataset’s label protocol | Foundational, not current evidence | **Keep**, add modern automated BCS validation work |
| [2] [Whay et al., *Assessment of the welfare of dairy cattle using animal-based measurements* (2003)](https://doi.org/10.1136/vr.153.7.197) | Yes | **Partial/weak** for the generalized “up to 25%” prevalence claim | Outdated for current prevalence | **Replace** for prevalence; keep only if discussing its assessment method |
| [3] [von Keyserlingk et al., welfare review (2009)](https://doi.org/10.3168/jds.2009-2326) | Yes | **Overextended** to claim listed behaviors precede clinical symptoms | Foundational welfare context | **Keep with narrower claim**; add condition-specific prospective evidence |
| [4] [Andrew et al., coat-pattern identification in RGB-D (ICIP 2016)](https://doi.org/10.1109/ICIP.2016.7532404) | Yes | Substantively relevant; report text calls it 2017 in one place while bibliography says 2016 | Still foundational for visual ID | **Keep and correct year** |
| [5] [Lin et al., Focal Loss (2017)](https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) | Yes | Formula is broadly correct; report incorrectly treats scalar α=0.25 as class rebalancing | Foundational | **Keep; correct implementation/interpretation** |
| [6] [Chen et al., GradNorm (2018)](https://proceedings.mlr.press/v80/chen18a.html) | Yes | Accurate as a future MTL-balancing method; not actually used despite an earlier implementation claim | Relevant | **Keep and implement or label future** |
| [7] [FAO, *World Livestock: Transforming the livestock sector through the SDGs* (2018)](https://doi.org/10.4060/ca1201en) | Yes | **Weakly matched** to the specific current global cattle-count assertion | Broad policy context, not census source | **Replace for population statistic** with FAOSTAT |
| [8] [Kendall et al., uncertainty-weighted MTL (2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) | Yes | Accurate generic MTL technique; not experimentally compared | Relevant | **Keep and add as baseline if discussed** |
| [9] [Woo et al., CBAM (2018)](https://doi.org/10.1007/978-3-030-01234-2_1) | Yes | Architecture concept represented correctly; anatomical attention claim is unverified without maps | Relevant foundational module | **Keep; remove localization claim or show evidence** |
| [10] [Howard et al., MobileNetV3 (2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | Yes | Accurate backbone background | Relevant baseline | **Keep** |
| [11] [Tan & Le, EfficientNet (2019)](https://proceedings.mlr.press/v97/tan19a.html) | Yes | Accurate backbone background; does not prove edge deployment | Relevant baseline | **Keep; narrow inference** |
| [12] [OpenCows2020 dataset mirror](https://datasetninja.com/opencows2020) | Yes, as a dataset page | Dataset facts broadly real, but secondary mirror and split handling is not justified | Relevant | **Replace/augment** with primary Bristol source and official protocol |
| [13] [Cao et al., CORAL (2020)](https://doi.org/10.1016/j.patrec.2020.11.008) | Yes | Theory summarized, but rank-consistent implementation is not demonstrated | Highly relevant | **Keep; verify actual CORAL layer** |
| [14] [Crawshaw, MTL survey (2020)](https://arxiv.org/abs/2009.09796) | Yes | **Misrepresented** where the report says it validates frozen-backbone sequential training/prevention of negative transfer | Useful but older survey | **Keep only for taxonomy; remove validation claim** |
| [15] [Neethirajan, sensors/big data/ML in animal farming (2020)](https://doi.org/10.1016/j.sbsr.2020.100367) | Yes | Broad background use is reasonable; cannot support model-specific claims | Broad, somewhat dated | **Keep sparingly** |
| [16] [Standley et al., *Which Tasks Should Be Learned Together?* (2020)](https://proceedings.mlr.press/v119/standley20a.html) | Yes | Relevant to task affinity; report does not actually apply its methodology | Highly relevant | **Keep and use analytically** |
| [17] [Yu et al., PCGrad (2020)](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Yes | Accurately described as future gradient surgery; not used | Highly relevant | **Keep and implement** |
| [18] [Andrew et al., cattle ID via deep metric learning (2021)](https://doi.org/10.1016/j.compag.2021.106133) | Yes | Relevant; the report does not reproduce its protocol for a fair claim | Relevant | **Keep; align evaluation** |
| [19] [Cows2021 dataset (2021)](https://doi.org/10.5523/bris.4vnrca7qw1642qlwxjadp87h7) | Yes | Dataset citation appears real but is not central to the experiments | Relevant optional ID dataset | **Keep only if used/comparing** |
| [20] [Gao et al., self-supervised video cattle ID/Cows2021 (2021)](https://arxiv.org/abs/2105.01938) | Yes | Relevant prior video-ID work; underused when claiming temporal-voting novelty | Highly relevant | **Keep and compare directly** |
| [21] [Weng et al., two-branch cattle face recognition (2022)](https://doi.org/10.1016/j.compag.2022.106871) | Yes | Broadly accurate ID literature entry | Relevant, not closest to coat/tracklet setup | **Keep selectively** |
| [22] [Fischer et al., Dryad RGB/depth BCS dataset (2023)](https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw4s) | Yes | **Incomplete**: source warns that video-derived images have low sample uniqueness and recommends temporally distant sampling | Central and relevant | **Keep; explicitly address dependence warning** |
| [23] [CattleLameness GitHub dataset](https://github.com/fahimsohan/CattleLameness) | Yes, as repository | Dataset is real; repository alone is insufficient ground-truth/method citation | Central but weakly validated | **Keep with [33] and document provenance/license** |
| [24] [Li et al., CBVD (2024)](https://doi.org/10.1038/s41598-024-65953-x) | Yes | **Partly inaccurate/under-documented**: 2,000 balanced-image subset and label mapping are not supported by the cited dataset description | Central cross-domain source | **Keep; correct counts and protocol** |
| [25] [ScienceDB cattle BCS dataset (2024)](https://www.scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c) | Yes, as dataset record | Core description plausible; narrow 3.25–4.25 range must be emphasized | Central | **Keep; add dataset paper/protocol if available** |
| [26] [MmCows Kaggle mirror](https://www.kaggle.com/datasets/hienvuvg/mmcows) | Yes, as mirror | **Inaccurate use**: ~213k boxes are reported as cropped frames | Central but secondary citation | **Replace/augment** with primary paper/repository; correct unit |
| [27] [Vu et al., MmCows dataset paper (2024)](https://openreview.net/forum?id=X4nq0W2qZX) | Yes | **Unit is mislabeled:** about 20k annotated source frames contain ~213k boxes; 213,686 may describe exported cow crops, not unique frames | Central and current | **Keep; report frames, boxes, and crop instances separately** |
| [28] [Antognoli et al., computer vision in dairy management review (2025)](https://doi.org/10.3390/ani15172508) | Yes | Relevant broad survey; does not by itself prove the exact four-task novelty claim | Current review | **Keep; use its search trail systematically** |
| [29] [Attri et al., SAAM-VetNet (2025)](https://doi.org/10.1097/MS9.0000000000003728) | Yes | Likely accurately summarized, but animal disease/severity imagery is only cross-domain MTL evidence | Tangential | **Move to methods background or remove** |
| [30] [Gonçalves et al., pregnancy ultrasound computer vision (2025)](https://doi.org/10.1093/jas/skaf166) | Yes | Appears real; not relevant to the four tasks or shared RGB/video architecture | Tangential filler | **Remove** |
| [31] [Liu et al., lightweight attention BCS model (2025)](https://doi.org/10.3390/vetsci12090906) | Yes | Relevant and broadly aligned; numerical comparison needs same metric/data caveat | Highly relevant | **Keep and compare design/efficiency protocol** |
| [32] [Palma et al., AI/data analytics on dairy farms review (2025)](https://doi.org/10.3390/ani15091291) | Yes | Broad review characterization is reasonable | Current but broad | **Keep sparingly** |
| [33] [Sohan et al., direct video lameness (2025)](https://doi.org/10.1038/s41598-025-29118-8) | Yes | **Materially false in Table 5.8:** RGB video is called RGB-D; published 90%/85% results are marked N/A; prose incorrectly calls the same 50-clip source “a different dataset” | Central and closest baseline | **Keep, correct, and reproduce** |
| [34] [Szyc et al., video-based automated lameness (2025)](https://doi.org/10.3390/s25185771) | Yes | Relevant; report gives too little protocol/result detail to establish comparison | Highly relevant | **Keep and expand** |
| [35] [Asim et al., cattle activity monitoring (2026)](https://www.mdpi.com/1424-8220/26/3/785) | Yes | **Mischaracterized as “sensor data”:** it is a vision-based overhead-camera/YOLO study on a custom commercial-farm image dataset, and 91.11% is reported as mAP rather than ordinary classification accuracy | Current and relevant behavior work | **Keep; correct modality and metric; avoid direct comparison** |
| [36] [Bu et al., pear spectroscopy MTL (2026)](https://doi.org/10.1016/j.saa.2026.127684) | Yes | Probably summarized correctly, but it offers no cattle/task/domain evidence | Irrelevant | **Remove** |
| [37] [Guzhva et al., PickAMoo weight estimation (2026)](https://doi.org/10.1038/s41598-026-54742-3) | Yes | Real, but weight estimation is not BCS and must not be treated as equivalent | Current but tangential | **Remove or segregate as adjacent application** |
| [38] [Lee et al., biosensors in dairy PLF review (2026)](https://doi.org/10.5713/ab.260154) | Yes | Broad context, not vision/MTL evidence | Current but broad | **Keep only for motivation** |
| [39] [Paneru et al., poultry computer-vision review (2026)](https://doi.org/10.1016/j.psj.2026.106887) | Yes | Likely real/accurate but cross-species and not needed | Irrelevant filler | **Remove** |
| [40] [Sandal & Ghosh, tomato disease review preprint (2026)](https://doi.org/10.21203/rs.3.rs-9600064/v1) | Yes, preprint | Citation may exist, but it is non-peer-reviewed plant-disease material with no direct role | Irrelevant filler | **Remove** |
| [41] [Sani et al., PLF challenges/opportunities review (2026)](https://doi.org/10.5713/ab.250895) | Yes | Broad PLF framing only | Current but broad | **Keep only if tied to a concrete deployment constraint** |
| [42] [Shalaldeh et al., ewe weight/carcass multimodal fusion (2026)](https://doi.org/10.3390/biology15100815) | Yes | Likely accurate summary, but cross-species outcome and modality are remote | Tangential filler | **Remove** |
| [43] [Yao et al., side-view automated dairy-cow BCS (2026)](https://doi.org/10.3168/jds.2025-27759) | Yes | Relevant, but Table 5.8 lists its result as N/A rather than accurately extracting its protocol/results | Highly relevant/current | **Keep; perform full comparison** |
| [44] [Zin & Tin, computer vision in PLF review (2026)](https://doi.org/10.5713/ab.260165) | Yes | Broad survey use appears reasonable; insufficient to prove novelty | Current review | **Keep; mine it for omitted primary work** |

### Literature-forensics verdict

No citation in the bibliography was conclusively fabricated. That does **not** make the literature review credible. Its overall rating is **Mixed, with strong signs of AI-assisted citation padding**. The evidence is structural: after the relevant core sources, the review abruptly accumulates one-sentence 2025–2026 descriptions from unrelated domains (pears, tomato disease, poultry, pregnancy ultrasound, ewe carcass traits); table cells use vague placeholders such as “Various”; central source details are wrong; and formulaic “this review rather than an empirical model” language replaces critical synthesis. Exact-phrase checks did not establish verbatim plagiarism, so it would be irresponsible to accuse the authors of plagiarism or definitively identify authorship from prose alone. The defensible allegation is narrower: sources appear to have been gathered and paraphrased without adequate primary-source verification, producing citation padding and factual errors.

Important omissions include recent public-data surveys, multi-camera identity work, true multi-annotation cattle-video benchmarks, behavior-conditioned identity MTL, modern automated BCS validation, and stronger video-lameness pipelines. Those omissions make the “first” and “state of the art” framing unreliable.

## 9. NEW PAPERS TO ADD

### Competitive benchmark against the closest work

| Closest work | What it does better than this report | Edge it has that this report lacks | Honest delta for this thesis |
|---|---|---|---|
| [Hooker et al., *Multitask contrastive learning for individual dairy cow recognition across different behavior classes* (2025)](https://www.sciencedirect.com/science/article/pii/S0022030225009014) | Jointly reasons about behavior and identity and uses contrastive learning designed for small image sets | Task relationship is explicit rather than merely sharing disjoint datasets | This report covers four outputs, but breadth alone is not a demonstrated technical advantage |
| [CattleEyeView (VCIP 2023/2024)](https://arxiv.org/abs/2312.08764) | Provides multiple annotations on the same top-down videos for detection, tracking, counting, pose, and segmentation; 30,703 frames and 753 cow instances | True multi-task data and coherent shared observations | This report targets health outcomes but lacks jointly annotated data and end-to-end multi-object evaluation |
| [Yu et al., MultiCamCows2024 / multi-camera self-supervised re-identification (2025)](https://research-information.bris.ac.uk/en/publications/holstein-friesian-re-identification-using-multiple-cameras-and-se/) | Uses 101,329 images, 90 cows, three cameras, seven days, self-supervision, and code; reports >96% single-image ID under its protocol | Camera/day diversity and representation learning aligned to re-ID | This report’s 97.58% vote is not competitive evidence until split and track independence are proven |
| [Sohan et al., direct video cattle-lameness detection (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12764510/) | On the same 50-clip source, reports 90% 3D-CNN and 85% ConvLSTM video accuracy with code/data | Direct matched baselines and correct source characterization | This report’s 80% EfficientNet-LSTM is currently worse and statistically unqualified |
| [Paulauskaite-Taraseviciene et al., multimodal early health-disorder detection (2026)](https://doi.org/10.3390/ani16030411) | Integrates physiological, behavioral, production, and thermal inputs at the cow-day level against veterinarian-confirmed udder/leg/hoof disorders | Actually predicts health status from co-located multimodal evidence rather than attaching unrelated heads trained on disjoint datasets | This report is cheaper/RGB-oriented, but its “integrated health” claim is much weaker and not clinically grounded |

### Twelve specific additions

1. **Hooker et al. (2025), “Multitask contrastive learning for individual dairy cow recognition across different behavior classes based on small image sets.”** Add because it is the closest omitted behavior–identity MTL work and directly challenges the novelty framing. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0022030225009014)
2. **Yu et al. (2025), “Holstein-Friesian re-identification using multiple cameras and self-supervision: MultiCamCows2024.”** Add because it supplies a much stronger multi-camera/day identity protocol and a self-supervised direction the motivation mentions but never evaluates. [Research record](https://research-information.bris.ac.uk/en/publications/holstein-friesian-re-identification-using-multiple-cameras-and-se/)
3. **Bhujel et al. (2025), review of publicly available computer-vision datasets for precision livestock farming, *Computers and Electronics in Agriculture* 229, 109718.** Add because a public-dataset benchmark claim must be grounded in the dedicated dataset survey, not an informal count of 33 papers. [DOI](https://doi.org/10.1016/j.compag.2024.109718)
4. **CattleEyeView (2023/2024), a multitask top-down cattle-video benchmark.** Add because it is a genuine same-scene multi-annotation benchmark and provides a sharper comparator for what “multitask cattle monitoring” means. [Paper](https://arxiv.org/abs/2312.08764)
5. **BECA (2025/2026), a longitudinal beef-cattle image dataset for computer vision.** Add because it brings 16,889 images, thousands of animals/observations, and longitudinal identity variation that exposes the narrowness of image-wise ID tests. [Scientific Data](https://www.nature.com/articles/s41597-025-06326-5)
6. **Siachos et al. (2024), “Automated body condition scoring in dairy cows using 2-dimensional images.”** Add because it uses 34,150 images and BCS agreement metrics that should shape the report’s evaluation rather than class-index MAE alone. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37977440/)
7. **Russello et al. (2024), video-based automatic lameness detection using pose estimation and multiple traits, *Computers and Electronics in Agriculture* 224, 109040.** Add as a stronger gait-structured comparator to an unconstrained frame CNN/LSTM. [DOI](https://doi.org/10.1016/j.compag.2024.109040)
8. **Myint et al. (2024), real-time cattle-lameness detection from a single side-view camera.** Add because it addresses the exact deployment geometry and real-time claim with a lameness-specific system. [Scientific Reports](https://doi.org/10.1038/s41598-024-72436-8)
9. **Paulauskaite-Taraseviciene et al. (2026), “AI-Driven Multimodal Sensing for Early Detection of Health Disorders in Dairy Cows.”** Add because it predates P2 and directly exposes the difference between monitoring proxy tasks and predicting health: it integrates physiological, behavioral, production, and thermal data at cow-day level against veterinarian-confirmed udder, leg, and hoof disorders. [Animals](https://doi.org/10.3390/ani16030411)
10. **Islam et al. (2026), “Computer Vision for Cattle Health and Welfare Monitoring: A Comprehensive Review of Methods, Applications, and Interdisciplinary Integration in Smart Agriculture.”** This appeared just after P2 but should anchor P3’s updated scope and missing-work search. [Sensors](https://doi.org/10.3390/s26134271)
11. **Rao, Garcia & Neethirajan (2026), “Video-based cattle behaviour detection for digital twin development in precision dairy systems.”** This paper predates the P2 submission and is a direct threat to the deployment narrative: it combines YOLOv11, ByteTrack identity persistence, SlowFast/TimeSformer behavior recognition, 4,964 annotated clips, macro-F1 0.84, and measured throughput. It exposes the missing tracker, temporal baselines, and hardware measurement in this report. [npj Veterinary Sciences](https://www.nature.com/articles/s44433-026-00004-x)
12. **Huber et al. (2026), “ReCowGnition: A Realistic Biometric Benchmark for Cow Face Recognition.”** This appeared after the June P2 but must be in P3. It contributes 6,838 images of 161 cows plus two verification and four identification protocols, making the report’s 46-way within-identity softmax test look especially weak and unrealistic. [arXiv](https://arxiv.org/abs/2607.22071)

Emerging directions the report currently ignores are task/domain-specific normalization, lightweight adapters instead of full hard sharing, cattle-specific self-supervised/video pretraining, contrastive re-identification conditioned on behavior/view, pose- or gait-structured lameness models, calibration/selective prediction, identity-persistent multi-object tracking, open-set cow ID, and evaluation under camera/day/farm shift. The active groups most visibly overlapping this space include the Bristol cattle-identification group, the MmCows/NEIS-Lab team, authors behind CattleEyeView, the Dalhousie video-behavior/digital-twin group, the ReCowGnition biometric-benchmark team, and current video-lameness groups represented by Sohan and Russello. The literature review does not map these research programs or explain how this project differs.

## 10. TOP 15 HARSHEST DEFENSE QUESTIONS

| Rank | Hostile committee question | Ready now? | What is needed to turn N into Y |
|---:|---|:---:|---|
| 1 | **What is the independent statistical unit in each task, and prove that no cow, video, track, day, camera, farm, or source crosses your train/validation/test boundary.** | **N** | Split manifests, group-overlap assertions, counts by protected unit, official protocol citations |
| 2 | **Why should we believe 0.9829–1.0000 lameness AUC when you admit frame leakage and tuned a threshold on the test set?** | **N** | Remove contaminated/test-tuned results; nested group CV; locked validation threshold; CIs |
| 3 | **Your source lameness paper reports 90% with 3D-CNN and 85% with ConvLSTM on the same 50 clips. Why does your table call it RGB-D and N/A, and why is your model only 80%?** | **N** | Correct source audit; reproduce both baselines on identical folds; explain protocol differences |
| 4 | **Exactly how does one optimizer step combine a 20-frame lameness batch with unrelated single images from three other datasets?** | **N** | Executable pseudocode, tensor shapes, masks, sampling probabilities, normalization, BN policy |
| 5 | **You call the failure “gradient conflict.” Show me the gradient measurements that rule out loss scaling, BN domain contamination, and catastrophic forgetting.** | **N** | Gradient cosine/norm logs; equal-weight, GradNorm, PCGrad, task-specific BN/adapters ablations |
| 6 | **Why are Dryad and ScienceDB MAEs in the same table when one class step is 1.0 BCS and the other is 0.25? What unit is 0.5566?** | **N** | Recompute score-unit metrics; relabel tolerances; add weighted kappa/Bland–Altman |
| 7 | **Are “213,686 MmCows frames” actually frames, bounding boxes, or exported cow crops—and how many independent cows, times, and camera scenes do they represent?** | **N** | Correct unit inventory from primary source; report frame/box/crop counts and independent groups |
| 8 | **What exactly are the 2,000 “balanced” CBVD images, how were they sampled, and how did you map foraging to feeding without target-label leakage?** | **N** | Published extraction script, video-level groups, ontology/mapping, all eligible test data |
| 9 | **How can 97.58% cow ID be credible under an image-wise split and temporal voting? Are adjacent frames of the same video on both sides?** | **N** | Official/session-separated split, track manifests, single-frame vs tracklet evaluation, overlap proof |
| 10 | **You claim a 75% latency and VRAM reduction. Show the measurements including YOLO, two camera streams, heads, activations, and the 20-frame buffer.** | **N** | On-device full-pipeline benchmark and four-model comparator; precision/batch details |
| 11 | **What makes this research rather than packaging four classifiers behind one encoder, especially when the tasks never share labels on the same cows and no output actually detects abnormal behavior or an integrated health state?** | **N** | Sharper measurable hypothesis; task/domain probes; affinity study; clinically meaningful endpoint or narrower claim |
| 12 | **How did you choose loss weights 0.35/0.35/0.15/0.15 without using the test set, and what happens under plausible alternatives?** | **N** | Validation-only selection, sensitivity grid, equal-weight/uncertainty/GradNorm/PCGrad results |
| 13 | **Why does a one-layer PyTorch LSTM list dropout 0.5 when that internal dropout is inactive? What other implementation claims have you verified?** | **N** | Correct architecture/code excerpt, unit tests, explicit dropout, code release |
| 14 | **Your labels are 0–4, but Eq. 4.4 uses thresholds 1–4. Did you collapse classes 0/1 and make class 4 unreachable, and is the head true rank-consistent CORAL at all?** | **N** | Exact code; five target/decode unit tests; constrained weights/biases; rank-violation analysis |
| 15 | **Name the closest paper and state your one-sentence technical delta without using the words “first,” “unified,” “lightweight,” or “public.”** | **N** | Choose Hooker/CattleEyeView/Sohan as task-specific comparators; articulate measured scientific delta |

## 11. PRIORITY ACTION LIST FOR NEXT PHASE

1. **Freeze all headline claims and run correctness tests before training.** Verify CORAL’s five encodings/decodings, split overlap, metric code, LSTM dropout, sequence overlap, and full parameter count; clarify MmCows units and correct Sohan/Asim metadata.
2. **Replace every split with a leakage-resistant manifest.** Cow/video/session/source grouping is the non-negotiable foundation; verify with automated assertions.
3. **Redo lameness completely.** Use repeated nested group CV, validation-only thresholds, original 3D-CNN/ConvLSTM comparators, and group-level intervals; permanently retire frame AUC.
4. **Redo ID with official and track/session-separated protocols.** Separate single-frame and tracklet claims; investigate the unexplained 86.49→94.96 jump; add unknown-cow/open-set evaluation or explicitly limit the claim to closed-set ID.
5. **Make the MTL training procedure executable on paper.** Specify task sampling, masks, loss normalization, BN, freezing, schedules, and checkpointing; release config-driven code.
6. **Run a small, disciplined MTL matrix.** Single-task, equal-weight hard sharing, current static weights, PCGrad, and GradNorm; add task-specific BN if domain mixing remains severe. Do not add five unrelated innovations.
7. **Run at least five seeds and compute group-aware 95% CIs/tests.** No more conclusion from a fourth decimal place produced by one run.
8. **Repair metrics and claims.** Convert BCS to score units; report behavior per-class precision/recall/F1; add calibration; narrow “health monitoring” to measured endpoints; remove “significant,” “inevitable,” “real-time,” and “75% memory/latency” until evidenced.
9. **Rewrite novelty/literature around closest work.** Add Hooker, CattleEyeView, MultiCamCows, Bhujel, Siachos, Russello, Rao, and ReCowGnition; delete cross-domain padding; document the search protocol; correct every misrepresentation.
10. **Build and measure the actual deployment pipeline only after validity is repaired.** Add identity-preserving tracking/cross-camera association, then benchmark YOLO + tracker + shared encoder + heads + sequence buffer on target hardware with median/P95 latency, peak memory, and power. A valid modest result is safer in defense than an impressive unmeasured one.

---

### Bottom-line defense posture

Do not try to defend the current P2 numbers as final evidence. Defend P2 as the experiment that exposed protocol defects and negative transfer, then show that P3 corrected them. A committee can forgive a failed architecture; it will not forgive knowingly contaminated tests, untraceable dataset counts, or causal claims made from a single run.
