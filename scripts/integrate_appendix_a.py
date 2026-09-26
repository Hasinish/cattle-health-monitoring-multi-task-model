import re

# 1. Update appendix/appendix_1.tex
path_app1 = "cattle_thesis_p3_latex/appendix/appendix_1.tex"
with open(path_app1, "r", encoding="utf-8") as f:
    text1 = f.read()

# Paragraph 1
old_p1 = r"""The manuscript was reviewed against repository commit
\sourcepath{d55ea2df436f504c3de4d83bcf9db845a6b706e3}.
This is the latest Git commit inspected for writing; individual experiment artifacts may record an earlier or unknown execution revision. No model was rerun for the thesis rewrite."""

new_p1 = r"""The document was checked against Git commit \sourcepath{d55ea2df436f504c3de4d83bcf9db845a6b706e3}. It is the latest Git commit examined while writing this document, although individual experiment artifacts may come from an earlier or unknown execution revision. None of the models was rerun during the thesis rewrite."""

assert old_p1 in text1, "old_p1 not found in appendix_1.tex"
text1 = text1.replace(old_p1, new_p1)

# Paragraph 2
old_p2 = r"""ScienceDB contains 37,045 training, 8,481 validation, and 8,040 test images in 3,958, 850, and 845 repaired burst groups. This supports sequence-safe evaluation, not a cow-disjoint claim. The Behavior protocol contains 3,785/680/809 Train/Validation/Test samples grouped by 267 source videos or sessions; Walking is available only in CVB. SideView Protocol A learns from 41 cows and evaluates 69 different cows using a 36,811-image parlor gallery, 25,260 barn queries, and 607 snapshot queries.\evidence{E04,E05,E06}"""

new_p2 = r"""ScienceDB contains 37,045 training images, 8,481 validation images, and 8,040 test images, organized into 3,958, 850, and 845 repaired burst groups, respectively. This setup supports sequence-safe evaluation rather than a cow-disjoint evaluation. The Behavior protocol includes 3,785 training, 680 validation, and 809 test samples grouped according to 267 source videos or sessions. Walking is available only in the CVB dataset. SideView Protocol A is trained using 41 cows and evaluated on 69 different cows, with a 36,811-image parlor gallery, 25,260 barn queries, and 607 snapshot queries.\evidence{E04,E05,E06}"""

assert old_p2 in text1, "old_p2 not found in appendix_1.tex"
text1 = text1.replace(old_p2, new_p2)

# Paragraph 3
old_p3 = r"""Runs 4 and 5 use matched controls because their perception pipelines exclude some canonical samples. Run 4 and its Run 1 control share 7,549 test images. Run 5 and its Run 2 control share 780 test sequences. Run 6 uses all Protocol A inputs because SideView releases paired ground-truth masks, but its result is an oracle condition rather than automatic segmentation.\evidence{E24,E27,E29}"""

new_p3 = r"""Runs 4 and 5 employ matched controls because their perception processing streams exclude certain canonical examples. Run 4 and its Run 1 control have 7,549 test images in common. Run 5 and its Run 2 control have 780 test sequences in common. Run 6 includes all inputs from Protocol A because SideView provides paired ground-truth masks, but the result of Run 6 represents an oracle condition rather than automatic segmentation.\evidence{E24,E27,E29}"""

assert old_p3 in text1, "old_p3 not found in appendix_1.tex"
text1 = text1.replace(old_p3, new_p3)

# Paragraph 4
old_p4 = r"""All reported single-task baselines and multi-task models were evaluated from single trained checkpoints; repeated-seed distributions, confidence intervals, and hypothesis tests were precluded by compute constraints. External cross-dataset stress tests on uncalibrated herds remain unexecuted roadmap items. Re-ID evaluations utilized released ground-truth masks rather than an automated upstream segmentation model. Downstream transfer of anatomical pose estimation and viewpoint priors remains unverified. Final administrative permissions, committee details, and ethics/AI disclosures require institutional confirmation."""

new_p4 = r"""The baselines and multi-task models were validated using just one trained checkpoint. Due to computational restrictions, the research could not perform repeated-seed analysis, calculate confidence intervals, or carry out hypothesis tests. The cross-dataset experiments on uncalibrated herds have not been performed yet and remain a task for the future. For the Re-ID experiments, the provided ground-truth masks were used instead of an automated upstream segmentation model. It has not yet been checked whether anatomical pose estimation and viewpoint priors can be transferred to other downstream tasks. Institutional approval is required for all administrative, committee, ethics, and AI-related disclosures."""

assert old_p4 in text1, "old_p4 not found in appendix_1.tex"
text1 = text1.replace(old_p4, new_p4)

with open(path_app1, "w", encoding="utf-8") as f:
    f.write(text1)
print("Updated appendix_1.tex successfully!")

# 2. Update appendix/methodology_crosswalk.tex
path_cross = "cattle_thesis_p3_latex/appendix/methodology_crosswalk.tex"
with open(path_cross, "r", encoding="utf-8") as f:
    text2 = f.read()

# Paragraph 5
old_p5 = r"""Descriptive model names are used in the methodology. The identifiers below retain the connection to the results tables and archived experiment records across all completed single-task and multi-task configurations."""

new_p5 = r"""Descriptive model names are used in the methodology. The identifiers listed below preserve the connection to the result tables and archived experiment records for all completed single-task and multi-task configurations."""

assert old_p5 in text2, "old_p5 not found in methodology_crosswalk.tex"
text2 = text2.replace(old_p5, new_p5)

# Paragraph 6
old_p6 = r"""The following table groups the supporting records for Chapter 4. Evidence identifiers refer to the versioned register in this appendix. The grouping replaces repeated record identifiers in the methodology prose without changing the underlying sources."""

new_p6 = r"""The following table categorizes the supporting records for Chapter 4. These evidence identifiers refer to the versioned register included in this appendix. This categorization replaces repeated record references in the methodology text without changing the underlying sources."""

assert old_p6 in text2, "old_p6 not found in methodology_crosswalk.tex"
text2 = text2.replace(old_p6, new_p6)

with open(path_cross, "w", encoding="utf-8") as f:
    f.write(text2)
print("Updated methodology_crosswalk.tex successfully!")
