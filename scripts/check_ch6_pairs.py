import re

raw_pairs = """
Original (Do Paraphrase)
This thesis investigated whether cattle-centered visual representations and multi-task learning can support unified monitoring of dairy cattle across three distinct vision tasks: body condition scoring (BCS), behavior recognition, and individual cow re-identification (Re-ID). Across all experimental investigations, rigorous provenance tracking, leakage-aware evaluation protocols, and matched-population controls were enforced to ensure reproducible and reliable empirical conclusions.

Original (Do Paraphrase)
For Body Condition Scoring, evaluating on the matched population of N = 7,549 ScienceDB test images where automatic detection and segmentation succeeded (93.89% perception coverage), localized cropping and binary foreground masks reduced physical error from 0.1929 to 0.1709 Mean Absolute Error (MAE) in BCS units and increased Acc@1 within ±0.25 units from 84.95% to 89.40% relative to the generic RGB baseline. However, exact classification accuracy remained modest (43.57%), and class-balanced metrics showed minor declines (balanced accuracy of 39.70% vs. 40.23%; Macro-F1 of 0.4039 vs. 0.4074).

Original (Do Paraphrase)
For Behavior Recognition, evaluating on N = 780 matched video sequences (T = 8 frames; 96.42% retained coverage), combining cattle-centered crops, foreground masks, and a lightweight 1D temporal convolutional network increased balanced accuracy from 71.30% to 74.43%, decreased test loss from 0.5505 to 0.4430, and improved minority-class Walking F1 from 0.2174 to 0.2456 relative to the single-frame RGB baseline. However, overall accuracy decreased slightly from 88.46% to 87.44%, and performance redistributed across sources (improving on feedlot camera footage while declining slightly on barn CCTV surveillance).

Original (Do Paraphrase)
For Cattle Re-Identification, evaluated under SideView Protocol A across 69 unseen cattle identities, target-centered cropping guided by released ground-truth masks improved first-match retrieval across both query conditions: Barn-to-Parlor Rank-1 increased from 58.64% to 63.90% (mAP from 38.32% to 40.68%), while Snapshot-to-Parlor Rank-1 rose from 38.88% to 62.93% (mAP from 27.05% to 40.42%). This substantial gain under handheld snapshot queries indicates that target-centered isolation is particularly beneficial when handling extreme posture, angle, and illumination shifts; however, because released ground-truth masks were used, this represents an oracle condition rather than an automatic segmentation deployment pipeline.

Original (Do Paraphrase)
The Monolithic Hard-Shared Multi-Task Control (E1), which forced all three tasks through a single four-channel ResNet-18 spatial backbone, resulted in outcome-level negative transfer across all three tasks relative to matched single-task references: BCS MAE degraded from 0.1709 to 0.1788; Behavior balanced accuracy dropped from 74.43% to 67.30% (with Walking F1 falling from 0.2456 to 0.0408); and Re-ID Barn Rank-1 fell from 63.90% to 57.38% (mAP falling from 40.68% to 30.37%).

Original (Do Paraphrase)
These outcomes demonstrate that cattle-centered visual representations provide tangible utility, but they do not universally benefit all evaluation metrics or eliminate shortcut learning across tasks. Furthermore, because these comparisons evaluated combined perception pipelines (crop plus mask for BCS and Re-ID; crop, mask, and temporal convolution for Behavior), the gains cannot be attributed to segmentation alone.

Original (Do Paraphrase)
The experimental results support task-specific incorporation of temporal dynamics and structural cues rather than forcing heterogeneous livestock monitoring tasks into a uniform input representation:

Original (Do Paraphrase)
For Body Condition Scoring, static morphology is the primary biological signal. Incorporating localized bounding-box crops and foreground binary masks provided a localized foreground representation while retaining visual morphology relevant to BCS, achieving superior error proximity without requiring temporal aggregation.

Original (Do Paraphrase)
For Behavior Recognition, posture and motion dynamics require temporal context. The combined eight-frame foreground-guided TCN configuration improved balanced accuracy and Walking F1 relative to the matched single-frame reference.

Original (Do Paraphrase)
For Cow Re-Identification, biometric coat patterns and identity features require fine-grained surface spatial detail. Cropping and mask guidance preserved localized surface geometry; the oracle-mask experiments demonstrated substantial retrieval gains under unconstrained query conditions when reliable foreground isolation is available.

Original (Do Paraphrase)
Anatomical pose keypoints and viewpoint priors were systematically audited. Frozen SuperAnimal pose estimation exhibited high missingness (45.60% detector failure rate on behavior sequences, deteriorating to 72.92% failure on recumbent/lying cattle) and lacked critical rear-pelvic landmarks required for BCS; pose was consequently excluded from the primary downstream pipeline. Real-cattle viewpoint classification achieved 86.26% accuracy on its own domain, but was not fused into primary models because cross-domain transfer to downstream farm environments remained unverified, with high camera and source shortcut risks.

Original (Do Paraphrase)
Thus, preserving task-specific requirements is best accomplished by tailoring visual cues to the physical demands of each domain—spatial morphology for BCS, temporal sequence modeling for behavior, and surface biometrics for re-identification—rather than imposing a single monolithic cue across all tasks.

Original (Do Paraphrase)
Outcome-Level Negative Transfer under Hard Sharing: Monolithic hard parameter sharing (E1) degraded held-out performance across all three task domains relative to matched single-task references: BCS MAE rose by +0.0079 units; Behavior balanced accuracy dropped by -7.13 percentage points (with Walking F1 falling from 0.2456 to 0.0408); and Re-ID Barn Rank-1 dropped by -6.52 percentage points (mAP falling by -10.31 percentage points). These consistent held-out degradations confirm outcome-level negative transfer when forcing disparate cattle vision tasks into a single shared ResNet-18 spatial trunk.

Original (Do Paraphrase)
Task-Specific Cattle Representation Strategy: Formulated and verified a principled, evidence-led input representation pipeline tailored to the physical demands of each task: static morphology-focused crop and mask for BCS; 8-frame sequential modeling with 1D temporal convolution for behavior; and biometric surface pattern preservation for re-identification. Audited and documented empirical boundaries for anatomical pose estimation and viewpoint priors.

Original (Do Paraphrase)
Systematic Multi-Task Sharing Benchmark: Conducted a controlled multi-task evaluation comparing monolithic hard parameter sharing (E1), modular task-private residual adapters (E3), and gradient-projected optimization (E4) against matched single-task baselines on identical held-out test sets, providing an empirical benchmark for livestock multi-task vision.

Original (Do Paraphrase)
Direct Gradient Conflict Diagnostics: Documented direct empirical measurements of shared-parameter gradient interference during multi-task training under PCGrad, recording 44,177 pairwise conflict projections across 16,140 super-steps and demonstrating regular gradient opposition (46.5% to 47.8% frequency) between agricultural monitoring tasks.

Original (Do Paraphrase)
Calibrated Understanding of Multi-Task Trade-Offs: Provided empirical evidence that multi-task learning does not universally benefit individual agricultural tasks through shared representations, demonstrating that negative transfer is outcome-level and that mitigation strategies provide selective, metric-dependent trade-offs rather than uniform performance gains.

Original (Do Paraphrase)
Persistence of Negative Transfer: Despite selective improvements from PCGrad, final multi-task models underperformed dedicated single-task references on several primary metrics, indicating that joint multi-task deployment involves deliberate engineering compromises.

Original (Do Paraphrase)
Repeated-Seed Training and Uncertainty Estimation: Future investigations should conduct multi-seed training runs to compute confidence intervals, variance bounds, and formal statistical tests across single-task and multi-task configurations.

Original (Do Paraphrase)
End-to-End Automatic Re-ID Segmentation: Integrating an upstream automated segmentation model (such as fine-tuned SAM or YOLO-seg) to replace oracle masks with predicted masks, evaluating real-world degradation under segmentation error.

Original (Do Paraphrase)
External Multi-Herd Generalization Testing: Evaluating trained models on independent external datasets, including Ruchay2026 and Dryad for BCS, and MmCows or CBVD-5 for behavior recognition, to establish cross-farm generalization.

Original (Do Paraphrase)
Longitudinal Identity Retrieval: Stress-testing cow re-identification across longitudinal cohorts (such as BECA-L and BECA-D) to assess biometric stability across lactation stages, seasonal coat changes, and physical growth.

Original (Do Paraphrase)
Isolated Perception Component Ablations: Conducting factorial ablation studies to isolate the individual contributions of bounding-box localization, binary foreground masking, soft probability masking, and temporal sequence length.

Original (Do Paraphrase)
Robust Cattle Pose Estimation: Fine-tuning animal pose models specifically on recumbent, lying, and rear-view cattle postures to overcome the keypoint missingness that currently prevents pose integration into livestock behavior monitoring.

Original (Do Paraphrase)
Advanced Multi-Task Optimization Strategies: Investigating partial parameter sharing (such as branched trunks) and adaptive loss-weighting algorithms (such as GradNorm) to further mitigate cross-task gradient tension.

Original (Do Paraphrase)
This thesis established an empirical and methodological framework for multi-task deep learning in precision cattle monitoring, evaluating body condition scoring, behavior recognition, and cow re-identification under controlled single-task and joint multi-task settings.

Original (Do Paraphrase)
The empirical evidence demonstrates that cattle-centered visual representations provide substantial, measurable utility for individual livestock monitoring tasks, particularly by improving BCS error tolerance and enhancing cow re-identification under unconstrained camera shifts. However, these benefits are task-dependent and do not eliminate the distinct representational requirements inherent to morphology regression, temporal sequence modeling, and biometric identity matching.

Original (Do Paraphrase)
When integrated into a unified multi-task framework, naive hard parameter sharing resulted in clear outcome-level negative transfer across all three tasks. Incorporating task-private residual adapters failed to resolve this degradation, while gradient-projected optimization (PCGrad) directly revealed that opposing shared-backbone gradient directions occurred regularly during training. By resolving opposing gradient components, PCGrad provided selective mitigation—notably boosting behavior recognition metrics and authentic barn surveillance performance—yet dedicated single-task models remained superior on several task-specific measures.

Original (Do Paraphrase)
Ultimately, these findings demonstrate that multi-task learning in agricultural computer vision does not provide a universal performance benefit through shared representations; successful deployment requires balancing shared cattle features with task-specific capacity and conflict-aware optimization.


TO:
Paraphrased:
The main aim of this thesis was to determine the effectiveness of cattle visual representations, in addition to multi-task learning, for monitoring dairy cattle. The tasks carried out under the thesis were: body condition scoring(BCS), behaviour recognition and re-identification of the cattle. The entire procedure took place through the process of record keeping, no data leakage, and matched-population controls to make sure its reproducible and reliable.

Paraphrased:
The BCS task used 7,549 ScienceDB test images, as detection and segmentation were correct on these images, which accounted for 93.89% of the test images. The results were better than the normal RGB model with the use of cropped cattle images and binary masks. The MAE dropped from 0.1929 to 0.1709 BCS units, while Acc@1 within ±0.25 increased from 84.95% to 89.40%. However, exact accuracy of classification was only 43.57%. Balanced accuracy and Macro-F1 also became slightly lower, from 40.23% to 39.70% and from 0.4074 to 0.4039.

Paraphrased:
The testing for the behavior task was conducted on 780 video sequences, 8 frames per sequence, and 96.42% of the matched data. The cropped cattle images, foreground masks and a small 1D temporal network achieved better results in several measures than the single-frame RGB model. Balanced accuracy increased from 71.30% to 74.43%, test loss decreased from 0.5505 to 0.4430, and Walking F1 improved from 0.2174 to 0.2456. But the overall accuracy decreased slightly from 88.46% to 87.44%. The method was more effective with feedlot camera data, and slightly less effective with barn CCTV data.

Paraphrased:
The model was tested on 69 unseen cattle during the Cattle Re-Identification using SideView Protocol A experiment. First-match accuracy for both query approaches was boosted by employing target centered crops from the ground-truth mask. For the Barn-to-Parlor case, Rank-1 was boosted from 58.64% to 63.90%, and mAP from 38.32% to 40.68%. For Snapshot-to-Parlor case, Rank-1 was boosted from 38.88% to 62.93%, and mAP from 27.05% to 40.42%. The significant improvement for snapshots indicates that this approach may prove especially useful in cases when there is drastic difference in the pose, view and lighting of the cattle. This test scenario, however,  was oracle-based, as ground-truth mask was used.

Paraphrased:
The Monolithic Hard-Shared Multi-Task Control (E1) that forced all three tasks to pass through a single four-channel ResNet-18 spatial backbone, experienced negative transfer across all three tasks compared to matched single-task baselines: BCS MAE degraded from 0.1709 to 0.1788; Behavior balanced accuracy decreased from 74.43% to 67.30% (Walking F1 decreased from 0.2456 to 0.0408); and Re-ID Barn Rank-1 decreased from 63.90% to 57.38% (mAP decreased from 40.68% to 30.37%).

Paraphrased:
Overall, the cattle-centred representations were useful, but not as effective for all measures and did not eliminate shortcut learning between tasks. Combined processing steps also contributed to the improvements. Cropping and masking were applied to BCS and Re-ID, and temporal convolution was applied in addition to these two to Behavior. For this reason, it is not possible to conclude that the better results are only due to segmentation.

Paraphrased:
The results show that each task should use the visual and temporal information that best fits its needs, instead of using the same input format for all cattle monitoring tasks: 

Paraphrased:
For Body Condition Scoring, body shape is the key biological information. The use of bounding-box crops and binary foreground masks made it easier for the model to focus on the cow while retaining the biological information of body shape that was essential for BCS. This improved prediction error without using information from multiple frames.

Paraphrased:
In the case of Behavior Recognition, body pose and movement are both important aspects, thus requiring data from multiple frames for the model. In fact, a model using 8 frames with foreground guided TCN performed better than the single-frame model.

Paraphrased:
For Cattle Re-ID system, details of the coat patterns and skin texture are required. The use of crops and segmentation masks played an important role in this regard. The Oracle masks turned out to be extremely useful in difficult conditions where the correct segmentation of the foreground was done.

Paraphrased:
Anatomical pose keypoints and viewpoints have been thoroughly tested. The frozen SuperAnimal pose had a very poor detection rate (45.60% in behavior sequence cases and 72.92% in lying cattle cases) and was unable to generate the rear pelvic keypoints required for BCS. For these reasons, the pose information is not incorporated into the primary downstream pipeline. The real-cattle viewpoint had an 86.26% accuracy rate on the dataset created just for it, but it has not been incorporated into the primary models since it has not been tested in other farm environments.

Paraphrased:
Thus, visual information has to be chosen based on requirements for each specific task. The body shape and morphology are needed for BCS, for behavior recognition change over time is necessary, and unique coat surface details are necessary for re-identification. Thus, it would be more useful to do that instead of using the same visual information for all the tasks.

Paraphrased:
Outcome-Level Negative Transfer under Hard Sharing: The hard parameter sharing method (E1) showed inferior performance compared to the matched single-task models for all three tasks. For BCS, there was an increase in the MAE metric by +0.0079. In Behavior, there was a decline in the balanced accuracy score by -7.13 percent, and Walking F1 went down from 0.2456 to 0.0408. In Re-ID, there was a decrease in Barn Rank-1 by -6.52 percent and mAP by -10.31 percent. This is outcome-level negative transfer due to different cattle vision tasks being made to share the ResNet-18 spatial backbone.

Paraphrased:
Task-Specific Cattle Representation Strategy: A task-specific approach for designing the representation of input data for a particular task was proposed and validated. Static crops of cows and masks were used for BCS in order to capture body shapes. An 8-frame sequence of cow videos was represented using a 1D temporal convolution. For the task of re-identification, important surface patterns were maintained. Limits of pose estimation and viewpoint were also explored.

Paraphrased:
Benchmark for Systematic Multi-Task Sharing: Performed a controlled experiment in multi-task settings to compare monolithic hard parameter sharing (E1), modular task-private residual adapters (E3), and gradient-projected optimization (E4) against matched single-task baselines using identical held-out test sets, providing an empirical benchmark for multi-task livestock vision.

Paraphrased:
Direct Gradient Conflict Diagnostics: Direct empirical evidence of interference among gradients of common parameters when training multiple tasks using PCGrad, with 44,177 pairs of gradient conflicts measured in 16,140 super-steps, and showing regular gradient conflicts between agricultural monitoring tasks at a frequency of 46.5–47.8%.

Paraphrased:
Calibrated Insights Into Multi-Task Trade-Offs: Established evidence showing that multi-task learning is not always advantageous for individual agriculture tasks due to shared representation, showing that the problem of negative transfer is an outcome-based one and that solutions lead to selective, metric-dependent trade-offs instead of uniformly improved performance.

Paraphrased:
Persistence of Negative Transfer: Although there are some gains made through PCGrad, in the end, the multi-task models performed worse than single-task models on many key metrics, which demonstrates that joint multi-tasking deployment is an deliberate engineering compromise.

Paraphrased:
Repeated Seed Training and Uncertainty Estimation: Future work needs to carry out multiple seed trainings in order to estimate confidence intervals, variance bounds, and statistical tests in both single and multitask scenarios.

Paraphrased:
End-To-End Automatic Re-ID Segmentation: Combining an upstream automatic segmentation model (e.g., fine-tuned SAM or YOLO-seg) which uses predicted masks instead of oracle masks, studying the effect of real-world degradation due to segmentation errors.

Paraphrased:
Generalization via External Testing using Multiple Herds: Testing of our trained networks using independent external test databases, such as Ruchay2026 and Dryad for BCS and MmCows or CBVD-5 for behavior recognition, to assess cross-farm generalization.

Paraphrased:
Longitudinal Identity Retrieval: Stress-testing cow Re-ID over time to determine the stability of the biometric identity across different cohorts (BECA-L and BECA-D), and across different lactation stages, seasonal coat appearance and body growth.

Paraphrased:
Ablation Studies for Individual Perception Components: Performing factorial ablation studies to analyze the separate contributions of bounding box localization, soft probabilistic masking, binary foreground masking and temporal length of sequences.

Paraphrased:
Robust Cattle Pose Estimation: Fine-tuning animal pose models for lying, recumbent, and rear-view cattle postures to address the problem of missing keypoints, which hinders the integration of pose information into cattle behavior monitoring.

Paraphrased:
Advanced Multi-Task Optimization Techniques: Exploring partial parameter sharing approaches (including branched trunk) and adaptive loss weighting techniques (including GradNorm) to address the cross-task gradient conflict.

Paraphrased:
The thesis was successful in establishing an empirical and methodological framework for multi-task deep learning in the context of precision cattle monitoring, involving body condition scoring, behavior recognition, and cow re-identification under controlled single-task and joint multi-task settings.

Paraphrased:
Evidence from experiments has shown that cow-focused visual representations offer significant utility in the performance of specific individual livestock monitoring tasks by improving BCS error tolerance and improving the re-identification of cows when there are unconstrained shifts in the cameras. This is however task dependent and does not address the separate needs of representation for morphological regression, temporal modeling, and biometric identification.

Paraphrased:
Naive hard parameter sharing in a multi-task integration framework led to negative transfer effects at the outcome level in all three tasks. Adding task-private residual adapters did not help mitigate the performance drop, whereas PCGrad explicitly showed that gradient directions for shared backbones were often opposite in training. Through the balancing of opposing gradient components, PCGrad delivered a selective remedy: especially improving behavior classification metrics and authentic barn surveillance performance, and yet specific single-task models were better on some task-specific metrics.

Paraphrased:
In the end, the results show that multi-task learning in agricultural computer vision is not universally beneficial in terms of performance using shared representation; successful deployment should entail a balance between shared characteristics of cattle, task-oriented abilities, and optimization strategies that account for task conflicts.
"""

orig_part, para_part = raw_pairs.split("TO:\n")

orig_blocks = [b.strip() for b in orig_part.split("Original (Do Paraphrase)") if b.strip()]
para_blocks = [b.strip() for b in para_part.split("Paraphrased:") if b.strip()]

print(f"Orig blocks: {len(orig_blocks)}, Para blocks: {len(para_blocks)}")

with open("cattle_thesis_p3_latex/chapters/chapter_9.tex", "r", encoding="utf-8") as f:
    tex_content = f.read()

for i, (orig, para) in enumerate(zip(orig_blocks, para_blocks)):
    # find where orig or key phrases of orig appear in tex_content
    # extract first 30 chars
    probe = orig[:35].replace("N = 7,549", "").replace("N = 780", "").strip()
    match = probe in tex_content
    print(f"Item {i+1}: match={match} | probe='{probe[:30]}'")
