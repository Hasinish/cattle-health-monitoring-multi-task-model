import re

path_app2 = "cattle_thesis_p3_latex/appendix/appendix_2.tex"
with open(path_app2, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Reproducibility Boundary
old_1 = r"""The thesis records deterministic split files, executed implementations, machine-readable metrics, and task-specific coverage. Raw datasets, full checkpoints, and cloud volumes are not embedded in the manuscript. Reproducing the results therefore requires the repository revision, the referenced manifests, access to the corresponding data, and the saved model artifacts."""
new_1 = r"""In order to ensure reproducibility, the exact data splitting records, code used in the experiments, performance results in machine readable format and task-specific information is provided in the thesis itself. Larger resources like the original datasets, fully-trained checkpoints and cloud repository contents are not included in the thesis. It is thus clear that in order to repeat the experiments one needs access to the right version of the repository, manifest and dataset files."""
assert old_1 in text, "old_1 not found"
text = text.replace(old_1, new_1)

# 2. Run details
old_2 = r"""Run 1 records seed 42 and split hashes. Run 5 records seed 2026, retained-sample hashes, and a bit-identical checkpoint reload. Run 6 also records a bit-identical reload and integrity checks for synchronized RGB/mask crops. Multi-task configurations E1, E3, and E4 used versioned implementations, validation-only checkpoint selection, and identical held-out test populations for direct comparison; E4 additionally records deterministic PCGrad RNG seeds and training diagnostics. Some historical artifacts record an unknown Git revision because Git metadata was unavailable in their execution container; the thesis review SHA is not substituted for those missing execution fields.\evidence{E08,E27,E29,E35,E36,E37,E38,E39,E40,E41}"""
new_2 = r"""Reproducibility details were stored separately for each run. For Run 1, seed 42 and split hashes, and for Run 5 - seed 2026 and retained data sample hashes were stored. Also, the checkpoint reload for Run 5 was checked to be bit-exact relative to the stored version. The same checkpoint check was performed for Run 6, along with the synchronization of RGB and mask crops. For the multi-task comparison, E1, E3, and E4 used fixed implementation versions, and checkpoint selection was based only on validation performance. They are evaluated on exactly the same held-out samples to allow direct comparison. E4 provides additional reproducibility details by storing random seeds of PCGrad as well as training diagnostics. In some old runs, it was not possible to save the original Git commit due to lack of Git metadata in the container. Instead of speculating or providing the missing details later, unknown values are kept in these fields, and thesis-review SHA is not used as a substitute.\evidence{E08,E27,E29,E35,E36,E37,E38,E39,E40,E41}"""
assert old_2 in text, "old_2 not found"
text = text.replace(old_2, new_2)

# 3. Independence units
old_3 = r"""\textbf{Independence units.} ScienceDB protects connected bursts rather than biological identities. CVB and Beef protect source videos or sessions rather than verified cows. Only SideView Protocol A supports the stated identity-disjoint learning/evaluation comparison.\evidence{E04,E05,E06}"""
new_3 = r"""\textbf{Independence units.} ScienceDB keeps connected bursts separate across the evaluation splits rather than separating the data based on individual cow identities. CVB and Beef are organized based on source videos or sessions rather than verified individual cows. Only SideView Protocol A supports a comparison where the cow identities used for learning and evaluation are different.\evidence{E04,E05,E06}"""
assert old_3 in text, "old_3 not found"
text = text.replace(old_3, new_3)

# 4. Matched coverage
old_4 = r"""\textbf{Matched coverage.} Run 4 applies to 7,549 of 8,040 test images, and Run 5 applies to 780 of 809 test sequences. Their corresponding RGB checkpoints were evaluated on exactly those retained samples. Coverage failures remain part of the system limitation.\evidence{E24,E27}"""
new_4 = r"""\textbf{Matched coverage.} Run 4 includes 7,549 of 8,040 test images, and Run 5 includes 780 of 809 test sequences. Their corresponding RGB checkpoints were evaluated on the same retained samples. The lack of full coverage is still a limitation of the system.\evidence{E24,E27}"""
assert old_4 in text, "old_4 not found"
text = text.replace(old_4, new_4)

# 5. Combined configurations
old_5 = r"""\textbf{Combined configurations.} Run 4 changes localization, crop, and mask guidance; Run 5 changes representation and temporal aggregation; Run 6 changes crop geometry and adds a GT mask channel. None of these experiments isolates one component's causal effect."""
new_5 = r"""\textbf{Combined configurations.} Each of runs 4, 5, and 6 alters multiple components simultaneously. For example, run 4 alters the localization, cropping, and mask components; run 5 alters the representation and temporal aggregation components; and run 6 alters the crop design and uses a ground truth mask channel. This makes it difficult to isolate the contribution of each individual component."""
assert old_5 in text, "old_5 not found"
text = text.replace(old_5, new_5)

# 6. Oracle Re-ID condition
old_6 = r"""\textbf{Oracle Re-ID condition.} Run 6 uses SideView ground-truth target masks. It does not demonstrate automatic SAM segmentation, and it does not prove that shortcut learning has been eliminated.\evidence{E29,E30}"""
new_6 = r"""\textbf{Oracle Re-ID condition.} In Run 6, the ground-truth target masks are provided by SideView. The results do not demonstrate automatic SAM segmentation, nor do they prove that shortcut learning has been eliminated.\evidence{E29,E30}"""
assert old_6 in text, "old_6 not found"
text = text.replace(old_6, new_6)

# 7. Capacity confounding in modular MTL
old_7 = r"""\textbf{Capacity confounding in modular MTL.} The modular task-private adapter architecture (E3) added 395,904 trainable parameters (+3.32\% capacity over E1). Performance differences cannot be attributed purely to architectural routing in isolation from parameter scale.\evidence{E37,E40}"""
new_7 = r"""\textbf{Capacity confounding in modular MTL.} The modular MTL architecture (E3) introduced 395,904 additional parameters that increased the capacity of the model by 3.32\% relative to E1. For that reason, performance gaps can be attributed not only to the novel architecture but also to the increase in size of the model.\evidence{E37,E40}"""
assert old_7 in text, "old_7 not found"
text = text.replace(old_7, new_7)

# 8. Gradient conflict diagnostics
old_8 = r"""\textbf{Gradient conflict diagnostics.} Direct gradient conflict measurements were recorded exclusively during E4 PCGrad training; opposing gradient directions were directly observed across task pairs, but conflict statistics cannot be retroactively assigned to E1 or E3.\evidence{E35,E41}"""
new_8 = r"""\textbf{Gradient conflict diagnostics.} Gradient conflicts were measured only for the E4 PCGrad method; opposing gradients were observed between tasks, yet no statistics can now be applied to E1 or E3.\evidence{E35,E41}"""
assert old_8 in text, "old_8 not found"
text = text.replace(old_8, new_8)

# 9. Selective mitigation
old_9 = r"""\textbf{Selective mitigation.} PCGrad provided selective mitigation on some metrics (notably behavior recognition), but dedicated single-task models remained superior on several primary metrics; negative transfer was not universally eliminated.\evidence{E36,E38}"""
new_9 = r"""\textbf{Selective mitigation.} It was beneficial for tackling negative transfer for specific instances where behavior recognition was considered. The PCGrad approach was not effective for improving all metrics, and single-task models proved to be superior in many cases. Thus, negative transfer could not be fully addressed.\evidence{E36,E38}"""
assert old_9 in text, "old_9 not found"
text = text.replace(old_9, new_9)

# 10. Statistical scope
old_10 = r"""\textbf{Statistical scope.} All reported single-task baselines and multi-task models are single-run point estimates. No confidence interval, repeated-seed standard deviation, p-value, or statistical-significance claim is supplied. External cross-dataset generalization to unseen herds remains unverified."""
new_10 = r"""\textbf{Statistical scope.} This paper presents the results of a single experiment run for the single task and multitask models. The confidence intervals, standard deviation, p-value, and statistical significance of the findings are unknown due to the lack of repeated seed experiments. Generalization to novel data (herds) has not been confirmed at the present time."""
assert old_10 in text, "old_10 not found"
text = text.replace(old_10, new_10)

# 11. Future experimental research
old_11 = r"""Future experimental research includes executing external cross-herd stress tests (e.g., on Ruchay, Dryad, MmCows, CBVD-5, and BECA) and multi-seed training runs to quantify parameter variance and confidence intervals."""
new_11 = r"""The future experiments will be performed by validating the models on different sets of cows from various datasets other than those used in the training process. The datasets to be used in such validations will include Ruchay, Dryad, MmCows, CBVD-5, and BECA. In addition, the models will also be tested under different random seeds to measure the variability of their accuracies."""
assert old_11 in text, "old_11 not found"
text = text.replace(old_11, new_11)

# 12. Administrative review
old_12 = r"""Administrative review must confirm author order, semester, committee details, ethics and AI-assistance requirements, dataset permissions, acknowledgments, and any separate IEEE-format deliverable or presentation/demonstration obligations. No signature, approval, user study, or institutional exemption is invented."""
new_12 = r"""This verification process is mandatory in order to prove that the authorship order, semester, committee information, ethical and AI considerations, permissions for the datasets, acknowledgments, and all other IEEE format requirements have been fulfilled in accordance with the standards. If there are any signatures required, then those must be done."""
assert old_12 in text, "old_12 not found"
text = text.replace(old_12, new_12)

with open(path_app2, "w", encoding="utf-8") as f:
    f.write(text)

print("All 12 paragraphs in appendix_2.tex updated successfully!")
