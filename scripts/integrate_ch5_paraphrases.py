import re

tex_path = "cattle_thesis_p3_latex/chapters/chapter_6.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Opening paragraph
old_p1 = "This chapter reports the completed single-task and multi-task evaluations, including the monolithic hard-shared control, modular task-private architecture, and PCGrad optimization control. The single-task evaluations first assess whether task-specific cattle-centered representations provide empirical utility over generic RGB baselines on matched populations. The subsequent multi-task evaluations directly examine the central questions of joint training: whether a unified spatial representation produces outcome-level negative transfer, whether architectural modularity mitigates cross-task degradation, whether gradient projection mitigates shared-backbone conflicts, and what training diagnostics reveal about gradient interference during optimization."
new_p1 = "The chapter will include all the finished results from single-task experiments and multi-task experiments which include Hard-Shared Baseline, Modular Task-Specific Model, and PCGrad control. Single-task experiments will first prove whether the cattle-centric representation outperforms the traditional RGB input on the same data. Multi-task experiments will then address the key issues regarding joint training. They will examine whether there is a negative transfer when only one spatial representation is being shared; whether having different task-specific modules mitigates performance loss; whether PCGrad solves any gradient conflict within the shared backbone."
assert old_p1 in content, "old_p1 not found"
content = content.replace(old_p1, new_p1, 1)

# 2. Baseline RGB Run 1 (preserve \evidence{E08})
old_p2 = r"The Baseline RGB model (Run~1) provides the canonical benchmark across all 8,040 ScienceDB test images, achieving a physical MAE of 0.1848 BCS units with 86.74\% tolerance accuracy ($\mathrm{Acc}@1$). Predictions generally stayed within $\pm0.25$ units, though exact score separation remained challenging. This benchmark is a single point estimate under a burst-group-disjoint, rather than biological cow-disjoint, protocol.\evidence{E08}"
new_p2 = r"The Baseline RGB Run 1 serves as the principal reference result for the ScienceDB images used for testing which number 8,040 in total. The physical MAE achieved by the model is equal to 0.1848 BCS units, while the Acc@1 value is 86.74\%. Most of the time, the prediction error was within $\pm0.25$ units, while predicting the score precisely was not easy. This should be considered as one estimate since the split was disjoint only among the burst groups and not cows.\evidence{E08}"
assert old_p2 in content, "old_p2 not found"
content = content.replace(old_p2, new_p2, 1)

# 3. Matched behavior evaluation (preserve \evidence{E27} and Table~\ref{tab:behaviorcomparison})
old_p3 = r"On the matched evaluation (Table~\ref{tab:behaviorcomparison}), temporal foreground modeling raised balanced accuracy from 71.30\% to 74.43\% and reduced test loss from 0.5505 to 0.4430. Overall accuracy dipped slightly from 88.46\% to 87.44\%, indicating a redistribution of predictive weight rather than a uniform gain across all metrics.\evidence{E27}"
new_p3 = r"As per Table~\ref{tab:behaviorcomparison}, temporal foreground modeling has helped enhance balanced accuracy and test loss. While balanced accuracy went up from 71.30\% to 74.43\%, test loss has gone down from 0.5505 to 0.4430. However, overall accuracy was affected adversely and went down from 88.46\% to 87.44\%. This means that there is an improvement of the model for some classes but not necessarily in all measures.\evidence{E27}"
assert old_p3 in content, "old_p3 not found"
content = content.replace(old_p3, new_p3, 1)

# 4. Open-set biometric retrieval SideView Protocol A (preserve \evidence{E22,E29})
old_p4 = r"Open-set biometric retrieval was evaluated under SideView Protocol~A, querying the fixed 36,811-image parlor gallery with barn CCTV and handheld snapshot imagery across 69 unseen cows disjoint from the 41 training identities.\evidence{E22,E29}"
new_p4 = r"The open set biometric identification experiment was done by using SideView Protocol~A. The system scanned through a fixed database of 36,811 images from the parlor using barn CCTV and handheld camera images. The experiment involved 69 cows that were not among the 41 training cows.\evidence{E22,E29}"
assert old_p4 in content, "old_p4 not found"
content = content.replace(old_p4, new_p4, 1)

# 5. Barn-to-Parlor surveillance (preserve Table~\ref{tab:reidbarn})
old_p5 = r"For Barn-to-Parlor surveillance queries, the Oracle Segmentation-Guided model (Run~6) improved primary retrieval quality (Table~\ref{tab:reidbarn}), raising Rank-1 accuracy from 58.64\% to 63.90\% and mean Average Precision (mAP) from 38.32\% to 40.68\%. The oracle target-centered configuration improved top-rank retrieval while deeper-rank measures remained similar or slightly lower."
new_p5 = r"In case of Barn-to-Parlor experiments, the Oracle Segmentation-Guided model (Run~6) enhanced the main retrieval performance in Table~\ref{tab:reidbarn}. Accuracy for Rank-1 increased from 58.64\% to 63.90\%, whereas the mAP increased from 38.32\% to 40.68\%. It indicates that the model performed better when identifying the right cow among the retrieved images. But the performance on the deeper ranks was more or less similar or even worse."
assert old_p5 in content, "old_p5 not found"
content = content.replace(old_p5, new_p5, 1)

# 6. E1 negative transfer BCS (preserve Table~\ref{tab:mtl_bcs_results})
old_p6 = r"Joint training under the Monolithic Hard-Shared MTL Control (E1) degraded the principal held-out BCS outcomes relative to the matched single-task reference (Run~4; Table~\ref{tab:mtl_bcs_results}). Physical MAE increased from 0.1709 to 0.1788 BCS units, alongside drops in tolerance accuracy and class-balanced measures. Although E1 recorded a slightly lower ordinal cross-entropy loss, this loss reduction did not translate into tighter physical score estimation. This pattern demonstrates outcome-level negative transfer when forcing ordinal morphological regression into a shared backbone alongside behavior and biometric identification."
new_p6 = r"The Table~\ref{tab:mtl_bcs_results} demonstrates that BCS performance deteriorated when hard-shared multi-task approach (E1) was applied relative to the performance of the baseline (Run~4). Physical mean absolute error became equal to 0.1709 for Run~4 and 0.1788 for E1. Other metrics like tolerance accuracy and balanced performance metric became worse as well. Ordinal cross-entropy loss became somewhat better but did not contribute to the improvement of BCS score prediction."
assert old_p6 in content, "old_p6 not found"
content = content.replace(old_p6, new_p6, 1)

# 7. Architectural and optimization interventions Re-ID
old_p7 = r"Architectural and optimization interventions yielded divergent retrieval outcomes. Modular adapters (E3) produced secondary degradation, driving Barn Rank-1 down to 49.08\%. While PCGrad (E4) recovered substantial ground relative to E3—lifting Barn Rank-1 back to 54.97\% and Snapshot mAP to 33.87\%—it remained mixed relative to E1 and well below the single-task reference. Isolated exceptions appeared at deeper retrieval ranks (such as E3 maintaining slightly higher Rank-10 in snapshots), confirming that multi-task feature sharing alters retrieval rank distributions without matching dedicated single-task biometric precision."
new_p7 = r"Different results were obtained from the architecture and optimization approaches in Re-ID. The modular adapters (E3) approach decreased the performance to Barn Rank-1 of 49.08\%. PCGrad (E4) improved the performance in comparison with the previous one, and achieved a result of Barn Rank-1 of 54.97\%, and a higher value of Snapshot mAP of 33.87\%. Nevertheless, its results were inconsistent with E1 and were still lower than those of the single-task network. Minor improvements occurred at higher ranks, for example, Snapshot Rank-10 was slightly increased in E3. It can be concluded that there is a difference in the retrieval performance by the implementation of multiple tasks for feature sharing; however, this method is still not as accurate as a specific single-task Re-ID approach."
assert old_p7 in content, "old_p7 not found"
content = content.replace(old_p7, new_p7, 1)

# 8. E4 training diagnostics
old_p8 = r"These diagnostics provide direct empirical evidence that opposing gradient directions on the shared four-channel ResNet-18 spatial backbone occurred regularly during joint training. Crucially, strict scientific claim boundaries must be maintained. E1 used ordinary shared-gradient accumulation without PCGrad projection, whereas E4 explicitly detected and projected opposing shared-backbone gradient directions. The E4 diagnostics establish that such conflicts occurred frequently during E4 training, but they do not establish identical conflict frequencies or a causal mechanism for E1. These measurements describe optimization dynamics under projection rather than proving that gradient interference alone accounted for observed performance gaps."
new_p8 = r"As per the training diagnostics, conflicting gradients are observed quite often in the four-channel ResNet-18 backbone that is shared by both tasks. Nevertheless, one has to make sure that an essential distinction between E1 and E4 is not forgotten. E1 just summed up the gradients, whereas E4 applied the PCGrad method in order to determine the conflicting gradient directions and to project them. As a result, the E4 results demonstrate that conflicts are often observed only in the case of E4 training. They cannot be used in arguing that E1 experienced an equivalent rate of conflict or that gradient interference is the only explanation for the performance difference."
assert old_p8 in content, "old_p8 not found"
content = content.replace(old_p8, new_p8, 1)

# 9. Monolithic Hard Parameter Sharing (E1) bullet
old_p9 = r"\item \textbf{Monolithic Hard Parameter Sharing (E1)}: Naively forcing three heterogeneous agricultural vision tasks into a single four-channel ResNet-18 spatial trunk produced held-out degradation across BCS, Behavior, and principal Re-ID retrieval metrics. This outcome-level negative transfer demonstrates that mutual synergy cannot be assumed when combining coarse body shape estimation (BCS), localized temporal motion (Behavior), and fine-grained biometric surface patterns (Re-ID). The specific internal feature corruption, however, cannot be isolated from E1 alone."
new_p9 = r"\item \textbf{Monolithic Hard Parameter Sharing (E1)}: In the case of the Monolithic Hard Parameter Sharing model (E1), all three tasks used a single shared ResNet-18 spatial backbone. In doing so, worse results were obtained for BCS, Behavior, and Re-ID. These findings indicate the presence of negative transfer and that the three tasks do not necessarily have to reinforce each other. The three tasks differ in the type of information used; body shape is used in BCS, motion in Behavior, and subtle visual cues in Re-ID. However, E1 does not allow us to specify precisely what was affected. E1 is unable to provide exact information about the impact on the common characteristics."
assert old_p9 in content, "old_p9 not found"
content = content.replace(old_p9, new_p9, 1)

# 10. Modular Task-Private Adapters (E3) bullet
old_p10 = r"\item \textbf{Modular Task-Private Adapters (E3)}: Introducing task-private residual bottleneck adapters between the shared backbone and each prediction head failed to provide general negative-transfer mitigation. While selected surveillance and loss measures improved, BCS and Re-ID suffered further degradation, and Behavior balanced accuracy declined. Because E3 added 395,904 trainable parameters (+3.32\% capacity), its performance differences cannot be attributed purely to architectural routing."
new_p10 = r"\item \textbf{Modular Task-Private Adapters (E3)}: The Modular Task-Private Adapters (E3) introduced separate adapter blocks for each task between the backbone and the prediction head, yet the problem of negative transfer was not solved consistently. While some surveillance and loss measures have become better, BCS and Re-ID measures worsened, as well as Behavior balance accuracy. In addition to this, 395,904 trainable parameters were added in E3, making an increase of model capacity by 3.32\%. This is why the improvements of the performance cannot be explained solely by task-specific routing."
assert old_p10 in content, "old_p10 not found"
content = content.replace(old_p10, new_p10, 1)

# 11. PCGrad Optimization Control (E4) bullet
old_p11 = r"\item \textbf{PCGrad Optimization Control (E4)}: By applying PCGrad projection exclusively to shared backbone gradients under exact parameter parity with E1 (11,926,706 trainable parameters; zero adapters), E4 provided a controlled optimization comparison. E4 achieved selective, metric-dependent mitigation of negative transfer—notably improving Behavior overall accuracy, balanced accuracy, Macro-F1, and CVB surveillance performance, while recovering principal Re-ID and BCS metrics relative to E3. However, E4 remained below dedicated single-task models on several primary metrics."
new_p11 = r"\item \textbf{PCGrad Optimization Control (E4)}: Model (E4) PCGrad modified the optimization process without modifying the model size. The number of trainable parameters in E4 was the same as that of E1, which is 11,926,706, while there were no adapter layers in it. PCGrad could address the negative transfer problem in some aspects, like Behavior accuracy, balanced accuracy, Macro-F1, and CVB surveillance effectiveness. PCGrad did better in important results of BCS and Re-ID than E3. Nevertheless, E4 was worse than single-task models on some major criteria."
assert old_p11 in content, "old_p11 not found"
content = content.replace(old_p11, new_p11, 1)

# 12. SideView Protocol A evaluates 69
old_p12 = r"\item SideView Protocol~A evaluates 69 biologically distinct cows excluded from representation learning. However, all Re-ID models use released dataset ground-truth masks, representing an oracle condition rather than autonomous field segmentation."
new_p12 = r"\item SideView Protocol~A: The test comprises 69 biologically unique cows which have been left out of the representation learning process. However, the Re-ID systems rely on the ground-truth masks which have been released. It means that the results of the tests have been obtained using an oracle segmentation framework."
assert old_p12 in content, "old_p12 not found"
content = content.replace(old_p12, new_p12, 1)

# 13. External datasets referenced
old_p13 = r"External datasets referenced in the broader roadmap (such as Ruchay, Dryad, MmCows, CBVD-5, and BECA) remain unexecuted stress tests; their documentation does not imply cross-domain generalization."
new_p13 = r"The external datasets mentioned in the bigger scheme of the research, including Ruchay, Dryad, MmCows, CBVD-5, and BECA, have not yet been validated. They are scheduled for stress testing in the future. Simply by referring to these datasets, we cannot imply that the model will generalize to other datasets/environments."
assert old_p13 in content, "old_p13 not found"
content = content.replace(old_p13, new_p13, 1)

# 14. First, cattle-centered visual representations provide tangible utility
old_p14_prefix = "First, cattle-centered visual representations provide tangible utility over generic RGB frames on matched single-task benchmarks, but the nature of this benefit is highly task-dependent."
new_p14_prefix = "In fact, applying visual information about cattle is quite useful in single task tests compared to RGB pictures. Nevertheless, the advantages of such approach may differ from task to task."
assert old_p14_prefix in content, "old_p14_prefix not found"
content = content.replace(old_p14_prefix, new_p14_prefix, 1)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESSFULLY REPLACED ALL 14 PARAGRAPHS IN CHAPTER 6 (RESULT ANALYSIS)!")
