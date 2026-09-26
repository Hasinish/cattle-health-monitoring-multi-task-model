"""
scripts/approve_ch3_fixed_rows.py
=================================
Applies approval ("✅ অর্থ ঠিক রাখা হইছে。") to Chapter 3 Rows 59, 72, and 79,
removes the resolved fixes, and resets Column C background from yellow to clean white.
"""

import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, robust_execute

CH3_SHEET_ID = 521635664

UPDATES = [
    {
        "range": "'Chapter 3: Requirements & Constraints'!D59",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• protocol verification, perception assessment, single-task reference baselines, ablation experiments, multi-task evaluation"
        ]]
    },
    {
        "range": "'Chapter 3: Requirements & Constraints'!D72",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• scientific and operational risks, train-test splits, perception failures, training budgets"
        ]]
    },
    {
        "range": "'Chapter 3: Requirements & Constraints'!D79",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• research cost, farm-deployment business case, accelerators, preprocessing steps, storage needs"
        ]]
    }
]

FORMAT_REQUESTS = [
    {
        "repeatCell": {
            "range": {
                "sheetId": CH3_SHEET_ID,
                "startRowIndex": r - 1,
                "endRowIndex": r,
                "startColumnIndex": 2,  # Col C
                "endColumnIndex": 3
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": {
                        "red": 1.0,
                        "green": 1.0,
                        "blue": 1.0
                    }
                }
            },
            "fields": "userEnteredFormat.backgroundColor"
        }
    }
    for r in [59, 72, 79]
]

def main():
    service = get_service()
    print("Approving Chapter 3 Rows 59, 72, and 79...")
    
    # 1. Update Column D texts
    body_vals = {
        "valueInputOption": "USER_ENTERED",
        "data": UPDATES
    }
    res_vals = robust_execute(lambda: service.values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=body_vals
    ).execute())
    print(f"[+] Updated {res_vals.get('totalUpdatedCells')} cells in Column D.")
    
    # 2. Reset Column C background to white
    body_fmt = {
        "requests": FORMAT_REQUESTS
    }
    res_fmt = robust_execute(lambda: service.batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=body_fmt
    ).execute())
    print(f"[+] Reset background color to white for Rows 59, 72, 79 in Column C.")

if __name__ == "__main__":
    main()
