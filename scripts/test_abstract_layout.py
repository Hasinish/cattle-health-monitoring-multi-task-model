p1 = "Cattle monitoring includes different activities such as measuring BCS (body condition scoring), behavioral recognition, and re-identification of cows. However, these activities do not rely on the same visual cues. For instance, BCS mainly involves body shape whereas behavior requires motion. The re-identification process requires features that will differentiate one animal from another like coat pattern. This makes the use of common visual cues across all three processes difficult, and it can even lower their performance."

p2 = "This thesis examines several approaches to help the model focus on the target---cattle, rather than the entire visual scene. Localization of cows and foreground masks are two approaches that help concentrate on the target, whereas temporal modeling is a straightforward technique that captures temporal information across multiple frames."

p3 = "Each task had its own set of leakage-free evaluation protocols. In ScienceDB BCS, the dataset was split into burst groups due to the lack of reliable cow identities. For behavior recognition, the sequences from barns and feedlots were split according to their sources and sessions to avoid data leakage from training to test. For SideView Re-ID, the protocol of open-set retrieval was applied on 69 held-out cows."

p4 = "Single-task experiments have shown that the contributions of cattle-centered inputs depended on the task. In case of BCS, the use of localization and masking led to an improvement in performance through the reduction of MAE from 0.1929 to 0.1709 BCS units. The percentage of accuracy within $\\pm 0.25$ was increased from 84.95\\% to 89.40\\%. Concerning behavior recognition, foreground information combined with temporal convolution improved the balanced accuracy from 71.30\\% to 74.43\\%, while overall accuracy slightly decreased from 88.46\\% to 87.44\\%. With regard to Re-ID, oracle mask increased Snapshot-to-Parlor Rank-1 from 38.88\\% to 62.93\\%, and mAP from 27.05\\% to 40.42\\%. However, due to the simultaneous change of several perception components, the above-mentioned improvements should not be attributed to segmentation alone."

p5 = "In the context of multi-task learning, hard parameter sharing led to outcome-level degradation in relation to comparable single-task baselines on all three datasets: BCS MAE degraded to 0.1788, behavioral balanced accuracy degraded to 67.30\\%, and Re-ID Barn Rank-1 degraded to 57.38\\%."

p6 = "Adding task-specific residual adapters to the model made a change in size of 3.32\\%. The adapters affected behavior performance in some ways, but did not make improvements in BCS and Re-ID tasks consistently. Then PCGrad method was used with the same model capacity to resolve interference between task gradients. During training process there were 44,177 gradient conflicts within 16,140 super-steps, which indicated that the interference was common. With PCGrad some of the interference was resolved and behavior balanced accuracy was increased up to 69.15\\% and Macro-F1 up to 0.7114."

p7 = "Based on the results, the utilization of cattle-centric representation has proved to be effective, although the benefits depend on the specific tasks. Utilization of the same representation across all tasks has been ineffective, and even though the inclusion of the adapter layer has tried to alleviate the problem, the issue cannot be solved completely. PCGrad has been of great assistance in some tasks but not uniformly across all tasks. Therefore, a unified cattle monitoring system has to incorporate elements that are both common and unique."

keywords = "\\vspace{0.4em}\n\\noindent\\textbf{Keywords:} cattle monitoring; body condition scoring; behavior recognition; cattle re-identification; cattle-centered representation; multi-task learning; negative transfer."

content = f"\\section*{{Abstract}}\n\n{p1} {p2} {p3} {p4} {p5} {p6} {p7}\n\n{keywords}\n\\clearpage\n"

with open("cattle_thesis_p3_latex/core/abstract.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated abstract.tex with vspace{0.4em} and em-dash")
