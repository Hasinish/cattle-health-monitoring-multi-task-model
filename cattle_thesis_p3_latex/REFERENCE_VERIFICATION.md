# Reference verification status

The manuscript bibliography contains 51 entries, and all 51 are cited in the current LaTeX sources. The validation script reports no undefined citation keys and no unused bibliography entries. This rewrite added no speculative reference and did not invent authors, titles, venues, years, pages, URLs, or DOIs.

## Verification policy

- Project-specific dataset counts, split sizes, model configurations, and experiment results use the internal E-evidence register rather than pretending that repository artifacts are publications.
- Literature claims use entries already present in the verified thesis bibliography.
- Preprints remain identified as arXiv/misc entries; they are not described as peer-reviewed journal papers.
- Publications with protocol differences are used for context, not as controlled numerical competitors.
- The P3 sample reports are writing-style references only and are never cited as scientific evidence.

## Core source groups used

| Topic | Citation keys | Qualification |
|---|---|---|
| BCS foundations and models | `edmonson1989bcs`, `ferguson1994bcs`, `rodriguez2018bcs`, `liu2025vets`, `yao2026jds2` | Supports anatomical scoring, ordered targets, and prior automated BCS approaches; not evidence for this thesis's dataset split or metrics. |
| Precision livestock monitoring | `weary2009understanding`, `antognoli2025ani1`, `sani2026ab25`, `palma2025ani1`, `lee2026ab26`, `zin2026ab26` | Supports motivation and monitoring context; no farm adoption or welfare outcome is inferred. |
| Behavior datasets and temporal modeling | `zia2023cvb`, `vu2024mmcows`, `li2024cbvd5`, `asim2026s260`, `bai2018tcn`, `feichtenhofer2019slowfast`, `yan2018stgcn` | Supports dataset/task context and temporal alternatives; Run 5 is described from its actual lightweight Conv1D code. |
| Cattle identification and Re-ID | `andrew2017cattle`, `andrew2021opencows`, `gao2021cows2021`, `weng2022cattleface`, `yu2025multicam`, `zhang2026beca`, `grolleau2026moo`, `feng2025catr` | Supports cattle biometric and cross-setting context; thesis metrics use SideView artifacts, not literature benchmarks. |
| Perception and representation | `he2016resnet`, `efficientnet2019`, `cbam2018`, `he2017maskrcnn`, `zhao2024rtdetr`, `kirillov2023sam`, `ravi2024sam2`, `mathis2018deeplabcut`, `ye2024superanimal`, `ong2023cattleeyeview`, `guzhva2026s415` | Supports model concepts and tool families; no off-the-shelf component is treated as newly invented. |
| Robustness and leakage | `geirhos2020shortcut`, `xiao2021background`, `beery2018terra`, `koh2021wilds`, `kapoor2023leakage` | Supports cautious discussion of shortcuts, domain shift, and leakage; does not prove that a particular thesis model learned a shortcut. |
| Ordinal and multi-task learning | `coral2020`, `crawshaw2020mtl`, `kendall2018multi`, `chen2018gradnorm`, `yu2020pcgrad`, `standley2020tasks`, `misra2016crossstitch`, `liu2019mtan`, `attri2025saamvetnet` | Supports design alternatives and optimization methods. E1 hard sharing, E3 task-private adapters, and E4 PCGrad were executed; PCGrad literature supports the optimization method, while all thesis-specific results come from project artifacts. E2 partial sharing and GradNorm E5 remain deferred. No literature citation is used as evidence for thesis-specific metrics. |

## Specific safeguards

- `coral2020` is used to explain rank-consistent ordinal learning; Run 1 is documented as independent cumulative ordinal BCE.
- `li2024cbvd5` is not used to overwrite the thesis's verified primary Behavior mapping.
- `yu2025multicam` describes related multi-camera research; MultiCamCows2024 remains inaccessible and is not presented as the active dataset.
- `ravi2024sam2` supports the SAM 2 model family; Run 6 is still labeled oracle because its masks come from SideView ground truth.
- No citation supports a claim of statistical significance, deployment success, measured ROI, or emissions reduction because those studies were not performed here.
