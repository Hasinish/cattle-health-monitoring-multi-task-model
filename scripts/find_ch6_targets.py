import re

tex_path = "cattle_thesis_p3_latex/chapters/chapter_6.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

targets = [
    "This chapter reports the completed single-task and multi-task evaluations",
    "The Baseline RGB model (Run 1) provides the canonical benchmark across all 8,040 ScienceDB test images",
    "On the matched evaluation (Table 5.2), temporal foreground modeling raised balanced accuracy from 71.30%",
    "Open-set biometric retrieval was evaluated under SideView Protocol A",
    "For Barn-to-Parlor surveillance queries, the Oracle Segmentation-Guided model (Run 6)",
    "Joint training under the Monolithic Hard-Shared MTL Control (E1) degraded the principal held-out BCS outcomes",
    "Architectural and optimization interventions yielded divergent retrieval outcomes",
    "These diagnostics provide direct empirical evidence that opposing gradient directions",
    "Monolithic Hard Parameter Sharing (E1): Naively forcing three heterogeneous agricultural vision tasks",
    "Modular Task-Private Adapters (E3): Introducing task-private residual bottleneck adapters",
    "PCGrad Optimization Control (E4): By applying PCGrad projection exclusively",
    "SideView Protocol A evaluates 69 biologically distinct cows",
    "External datasets referenced in the broader roadmap",
    "First, cattle-centered visual representations provide tangible utility"
]

print(f"Total lines in {tex_path}: {len(lines)}")
for idx, t in enumerate(targets, 1):
    found = False
    for l_idx, line in enumerate(lines, 1):
        if t.lower() in line.lower() or t[:40].lower() in line.lower():
            print(f"Target {idx} found at line {l_idx}: {line[:70]}...")
            found = True
            break
    if not found:
        print(f"Target {idx} NOT FOUND: {t[:60]}")
