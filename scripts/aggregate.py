import os
import glob

output_file = os.path.join("docs", "detailed_p2_info.md")
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the order of files to aggregate
sections = [
    ("1. MASTER PLAN & CONTEXTS", [
        "context/context1_master_plan.txt",
        "context/context2_bcs.txt",
        "context/context2_behavior.txt"
    ]),
    ("2. PRESENTATION & KEY CONCEPTS", [
        "docs/presentation_key_concepts.md"
    ]),
    ("3. INDIVIDUAL CONTEXT 3 REPORTS", glob.glob(os.path.join(workspace_dir, "context", "Context3_*.txt"))),
    ("4. ALL RAW RESULTS", glob.glob(os.path.join(workspace_dir, "workspaces", "*", "*_results.txt"))),
    ("5. PREPROCESSING CODE", glob.glob(os.path.join(workspace_dir, "context", "preprocess_*.py"))),
    ("6. TRAINING CODE (HASIN - ResNet-18)", [
        "workspaces/hasin/train_bcs.py",
        "workspaces/hasin/train_behavior.py"
    ]),
    ("7. TRAINING CODE (NUSRAT - EfficientNetB0 & Lameness)", [
        "workspaces/nusrat/train_bcs.py",
        "workspaces/nusrat/train_behavior.py",
        "workspaces/nusrat/train_id.py",
        "workspaces/nusrat/train_lameness.py",
        "workspaces/nusrat/train_spatiotemporal_lameness.py",
        "workspaces/nusrat/train_spatiotemporal_lameness_efficientnet.py",
        "workspaces/nusrat/predict_spatiotemporal_lameness.py",
        "workspaces/nusrat/evaluate_all_50_videos.py",
        "workspaces/nusrat/evaluate_cbvd.py",
        "workspaces/nusrat/visualize_cow_detection.py",
        "workspaces/nusrat/visualize_lameness_realtime.py",
        "workspaces/nusrat/cut_video_segment.py",
        "workspaces/nusrat/crop_cow_detection.py",
        "workspaces/nusrat/crop_sciencedb_dataset.py",
        "workspaces/nusrat/visualize_bcs_crops.py"
    ]),
    ("8. TRAINING CODE (NAMIRA - MobileNetV3-Small)", [
        "workspaces/namira/train_bcs.py",
        "workspaces/namira/train_behavior.py"
    ]),
    ("9. TRAINING CODE (BITHI - ResNet-50)", [
        "workspaces/bithi/train_bcs.py",
        "workspaces/bithi/train_behavior.py"
    ]),
    ("10. TRAINING CODE (SHOUVIK - DenseNet121)", [
        "workspaces/shouvik/train_bcs.py",
        "workspaces/shouvik/train_behavior.py"
    ])
]

with open(os.path.join(workspace_dir, output_file), "w", encoding="utf-8") as outfile:
    outfile.write("# DETAILED P2 INFO - MASTER AGGREGATION\n\n")
    outfile.write("This document contains a full dump of every context, plan, result, and code file in the P2 project.\n\n")
    
    for section_title, files in sections:
        outfile.write(f"## {section_title}\n")
        outfile.write("=" * 80 + "\n\n")
        
        # Flatten and normalize paths
        resolved_files = []
        for f in files:
            if os.path.isabs(f):
                resolved_files.append(f)
            else:
                resolved_files.append(os.path.join(workspace_dir, f))
                
        # Sort for deterministic output
        resolved_files = sorted(list(set(resolved_files)))
        
        for file_path in resolved_files:
            if os.path.exists(file_path):
                outfile.write(f"### FILE: {os.path.relpath(file_path, workspace_dir)}\n")
                outfile.write("---\n")
                
                # Determine markdown block type
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".py":
                    outfile.write("```python\n")
                elif ext in [".txt", ".md", ".csv"]:
                    outfile.write("```text\n")
                else:
                    outfile.write("```text\n")
                    
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}\n")
                    
                outfile.write("\n```\n\n")
            else:
                outfile.write(f"### FILE: {os.path.relpath(file_path, workspace_dir)} (NOT FOUND)\n\n")

print(f"Successfully generated {output_file}")
