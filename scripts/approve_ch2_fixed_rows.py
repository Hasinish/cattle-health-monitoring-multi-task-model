"""
scripts/approve_ch2_fixed_rows.py
=================================
Applies approval ("✅ অর্থ ঠিক রাখা হইছে。") to Chapter 2 Rows 124, 203, 212, and 218,
removes the resolved fixes, and resets Column C background from light yellow to clean white.
"""

import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, robust_execute

CH2_SHEET_ID = 1111150292

UPDATES = [
    {
        "range": "'Chapter 2: Literature Review'!D124",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• RT-DETR, Zhao et al., instance segmentation, Mask R-CNN, foreground mask"
        ]]
    },
    {
        "range": "'Chapter 2: Literature Review'!D203",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• MTL, Kendall et al., task uncertainty, GradNorm, gradient statistics, visual features"
        ]]
    },
    {
        "range": "'Chapter 2: Literature Review'!D212",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• Task clustering, Standley et al., visual features, single-task references"
        ]]
    },
    {
        "range": "'Chapter 2: Literature Review'!D218",
        "values": [[
            "✅ অর্থ ঠিক রাখা হইছে。\n\n"
            "📌 Words/phrases to keep intact:\n"
            "• Re-ID, behavior recognition, BCS, morphology, hard sharing, task-conditioned"
        ]]
    }
]

FORMAT_REQUESTS = [
    {
        "repeatCell": {
            "range": {
                "sheetId": CH2_SHEET_ID,
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
    for r in [124, 203, 212, 218]
]

def main():
    service = get_service()
    print("Approving Chapter 2 Rows 124, 203, 212, and 218...")
    
    # 1. Update Column D text
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
    print(f"[+] Reset background color to white for Rows 124, 203, 212, 218 in Column C.")

if __name__ == "__main__":
    main()
