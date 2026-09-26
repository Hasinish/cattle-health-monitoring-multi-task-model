"""
scripts/replace_latex_in_col_b.py
==================================
Replaces raw LaTeX tags and math notation in Column B (Original text)
with actual meant plain-text English (Table numbers, clean formula notation).
Leaves all other cells and formatting intact.
"""

import sys
import os

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, robust_execute

UPDATES = [
    {
        "range": "'Chapter 3: Requirements & Constraints'!B59",
        "values": [[
            "The project is organized into several main stages: protocol verification, perception assessment, "
            "single-task reference models, cattle-centered alternatives, and multi-task comparison. "
            "Each stage provides the information needed for the next. Table 3.2 records the main project milestones. "
            "The timeline shows project progress, not scientific evidence."
        ]]
    },
    {
        "range": "'Chapter 3: Requirements & Constraints'!D59",
        "values": [[
            "📌 Words/phrases to keep intact:\n"
            "• protocol verification, perception assessment, single-task reference baselines, ablation experiments, multi-task evaluation\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.1:timeline' -> 'Table 3.2' (remove the ':timeline' code tag and use correct Table 3.2 numbering)."
        ]]
    },
    {
        "range": "'Chapter 3: Requirements & Constraints'!B72",
        "values": [[
            "Table 3.3 lists the main scientific and operational risks. "
            "These include related observations crossing train and test splits, unsupported identity claims, "
            "classes linked to one data source, perception failures, incorrect causal claims from combined model changes, "
            "and unfair comparisons caused by different training budgets."
        ]]
    },
    {
        "range": "'Chapter 3: Requirements & Constraints'!D72",
        "values": [[
            "📌 Words/phrases to keep intact:\n"
            "• scientific and operational risks, train-test splits, perception failures, training budgets\n\n"
            "🛠️ Required Fixes:\n"
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.2:risks' -> 'Table 3.3' (remove the ':risks' code tag and use correct Table 3.3 numbering)."
        ]]
    },
    {
        "range": "'Chapter 3: Requirements & Constraints'!B79",
        "values": [[
            "The available records allow only a partial discussion of research cost, not a full farm-deployment business case. "
            "The experiments use different accelerators, preprocessing steps, storage needs, and billing conditions. "
            "Their runtime therefore cannot be turned into directly comparable cost values without consistent rates or invoices. "
            "Table 3.4 reports only information that was actually recorded."
        ]]
    },
    {
        "range": "'Chapter 2: Literature Review'!B33",
        "values": [[
            "[Formula:\n L_MTL = sum_{t=1}^T (lambda_t * L_t)\n]"
        ]]
    },
    {
        "range": "'Chapter 2: Literature Review'!B36",
        "values": [[
            "where L_t is the loss for task t and lambda_t controls how much that task contributes. "
            "This equation describes the combined loss, but it does not define how much of the network should be shared. "
            "A model may share almost everything, only early layers, or selected information between task-specific paths."
        ]]
    }
]

def main():
    service = get_service()
    print("Replacing raw LaTeX artifacts in Column B with actual intended text...")
    
    body = {
        "valueInputOption": "USER_ENTERED",
        "data": UPDATES
    }
    
    res = robust_execute(lambda: service.values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=body
    ).execute())
    
    print(f"[+] Successfully updated {res.get('totalUpdatedCells')} cells.")
    for u in UPDATES:
        print(f"  - {u['range']} -> Updated.")

if __name__ == "__main__":
    main()
