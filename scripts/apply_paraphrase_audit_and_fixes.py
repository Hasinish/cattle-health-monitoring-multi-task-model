"""
scripts/apply_paraphrase_audit_and_fixes.py
============================================
Executes targeted marking of defective paraphrased cells in Column C with LIGHT YELLOW
and writes specific '🛠️ Required Fixes:' in Column D.
Preserves EVERY OTHER CELL intact.
"""

import sys
import os
import json
import time

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute

# Light yellow RGB (Pastel / soft accent yellow)
LIGHT_YELLOW = {"red": 1.0, "green": 0.98, "blue": 0.8}

UPDATES = {
    "Chapter 1: Introduction": {
        27: (
            "📌 Words/phrases to keep intact:\n"
            "• BCS, body shape, Behavior recognition, posture, Re-identification, coat markings\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Remove stray open quotation mark ('“') before 'Coat pattern'.\n"
            "• Fix 2: Paraphrase the near-verbatim copied sentence ('Coat pattern can be a way to tell two cows apart, but it shouldn't be a shortcut for assessing body condition') into your own words."
        ),
        88: (
            "📌 Words/phrases to keep intact:\n"
            "• research questions, cattle-centered representations, RGB baselines, multi-task framework\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Missing paraphrase: Cell currently contains 'No paraphase....'. Paraphrase the three research questions into a smooth narrative paragraph."
        ),
        92: (
            "📌 Words/phrases to keep intact:\n"
            "• ScienceDB, BCS, CVB, Kaggle Beef, SideViewCows2026\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Remove rogue floating '1.' enumeration marker from the middle of the narrative paragraph ('...comparative experimental design. 1. The first stage...')."
        ),
        101: (
            "📌 Words/phrases to keep intact:\n"
            "• Chapter 4, Chapter 5, model settings, training methods, experimental controls, evaluation metrics\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Fix broken grammar ('In the chapter 4 it discuss about' -> 'Chapter 4 outlines', 'interpretetion' -> 'interpretation', 'these experiments looks at' -> 'these experiments examine').\n"
            "• Fix 2: Paraphrase the second half into your own words instead of near-verbatim copying."
        ),
        108: (
            "📌 Words/phrases to keep intact:\n"
            "• BCS, posture, activity, Re-identification, ScienceDB, burst-group-disjoint, SideViewCows2026\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change 'assignments' -> 'tasks' (machine learning models evaluate distinct computer vision tasks, not school assignments)."
        ),
        117: (
            "📌 Words/phrases to keep intact:\n"
            "• leakage-aware evaluation protocols, RGB, BCS, Re-ID, multi-task, negative transfer\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Direct copy-paste alert: Paragraph is copied almost 100% verbatim from original (79% 4-gram overlap) with only trivial word swaps. Paraphrase substantially into your own words."
        )
    },
    "Chapter 2: Literature Review": {
        13: (
            "📌 Words/phrases to keep intact:\n"
            "• ResNet, residual connections, EfficientNet, transfer learning\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Direct copy-paste alert: 100% verbatim copy-paste from original (not paraphrased). Paraphrase in your own words while keeping model names and transfer learning intact."
        ),
        51: (
            "📌 Words/phrases to keep intact:\n"
            "• Lee et al., biosensors, cameras, body shape, posture, coat patterns\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change 'fur patterns' -> 'coat patterns / coat markings' (cattle have hair coats and markings, not fur)."
        ),
        54: (
            "📌 Words/phrases to keep intact:\n"
            "• task-specific, preprocessing, shared features, integration\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Complete truncated final sentence: Paragraph cuts off mid-thought ('Therefore, the integration is not only about merging multiple predictions.'). Add the missing second half regarding how to determine what feature capacity can be shared without sacrificing individual task needs."
        ),
        124: (
            "📌 Words/phrases to keep intact:\n"
            "• RT-DETR, Zhao et al., instance segmentation, Mask R-CNN, foreground mask\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change 'modern sensors' -> 'modern object detectors' (RT-DETR and Mask R-CNN are computer vision detector models, not physical hardware sensors)."
        ),
        203: (
            "📌 Words/phrases to keep intact:\n"
            "• MTL, Kendall et al., task uncertainty, GradNorm, gradient statistics, visual features\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Paraphrase the verbatim copied final sentence ('They control how strongly tasks update the network, but they do not decide which visual features should be shared.') into your own words."
        ),
        212: (
            "📌 Words/phrases to keep intact:\n"
            "• Task clustering, Standley et al., visual features, single-task references\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Paraphrase the verbatim copied concluding sentences ('One grouping may help one task while hurting another. Strong single-task references are therefore necessary before judging a shared model.') into your own words."
        ),
        218: (
            "📌 Words/phrases to keep intact:\n"
            "• Re-ID, behavior recognition, BCS, morphology, hard sharing, task-conditioned\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Paraphrase the verbatim copied final sentence ('This is why the thesis compares hard sharing with task-conditioned or partly private processing.') into your own words."
        )
    },
    "Chapter 3: Requirements & Constraints": {
        59: (
            "📌 Words/phrases to keep intact:\n"
            "• protocol verification, perception assessment, single-task reference baselines, ablation experiments, multi-task evaluation\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.1:timeline' -> 'Table 3.1' (remove the ':timeline' code tag)."
        ),
        72: (
            "📌 Words/phrases to keep intact:\n"
            "• scientific and operational risks, train-test splits, perception failures, training budgets\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.2:risks' -> 'Table 3.2' (remove the ':risks' code tag)."
        ),
        79: (
            "📌 Words/phrases to keep intact:\n"
            "• research cost, farm-deployment business case, accelerators, preprocessing steps, storage needs\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.4:costs' -> 'Table 3.4' (remove the ':costs' code tag).\n"
            "• Fix 2: Fix broken grammar in final clause ('In Table 3.4, reports only information...' -> 'Table 3.4 reports only information that was actually recorded')."
        )
    }
}

def apply_updates():
    service = get_service()
    
    total_yellow_requests = []
    total_val_data = []
    
    print("Preparing batch updates...")
    
    for tab_name, rows_dict in UPDATES.items():
        sheet_id = SHEET_IDS[tab_name]
        print(f"\nTab: {tab_name} (ID: {sheet_id}) - {len(rows_dict)} rows to update")
        
        for row_num, col_d_text in rows_dict.items():
            row_idx = row_num - 1 # 0-indexed
            
            # 1. Background color request for Column C (col index 2 to 3)
            # ONLY updates backgroundColor field; leaves text, formatting, borders intact!
            total_yellow_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 2,
                        "endColumnIndex": 3
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": LIGHT_YELLOW
                        }
                    },
                    "fields": "userEnteredFormat.backgroundColor"
                }
            })
            
            # 2. Value update for Column D
            total_val_data.append({
                "range": f"'{tab_name}'!D{row_num}",
                "values": [[col_d_text]]
            })
            print(f"  Row {row_num:3d}: Col C -> LIGHT YELLOW | Col D -> Fix text ({len(col_d_text)} chars)")
            
    print(f"\nExecuting batch updates:")
    print(f"  - {len(total_yellow_requests)} cells in Column C to color LIGHT YELLOW")
    print(f"  - {len(total_val_data)} cells in Column D to write Required Fixes")
    
    # Execute Column D value updates
    body_val = {
        "valueInputOption": "USER_ENTERED",
        "data": total_val_data
    }
    res_val = robust_execute(lambda: service.values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=body_val
    ).execute())
    print(f"[+] Values updated successfully: {res_val.get('totalUpdatedCells')} cells updated.")
    
    # Execute Column C format updates (batchUpdate)
    body_fmt = {
        "requests": total_yellow_requests
    }
    res_fmt = robust_execute(lambda: service.batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=body_fmt
    ).execute())
    print(f"[+] Formatting updated successfully: {len(res_fmt.get('replies', []))} format replies.")
    print("\n[SUCCESS] ALL TARGETED PARAPHRASED CELLS MARKED LIGHT YELLOW AND REQUIRED FIXES RECORDED!")

if __name__ == "__main__":
    apply_updates()
