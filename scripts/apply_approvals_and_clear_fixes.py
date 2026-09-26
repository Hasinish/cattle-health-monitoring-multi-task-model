"""
scripts/apply_approvals_and_clear_fixes.py
=========================================
Applies approvals ("✅ অর্থ ঠিক রাখা হইছে。") to rows where the paraphrase meaning
is confirmed accurate and faithful, removes "🛠️ Required Fixes:" where already fixed,
and un-highlights (resets background to white) Column C for approved rows.
Leaves all other cells and formatting 100% intact.
"""

import re
import sys

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, robust_execute

SHEET_METADATA = {
    "Chapter 1: Introduction": {"sheetId": 0},
    "Chapter 2: Literature Review": {"sheetId": 1111150292},
    "Chapter 3: Requirements & Constraints": {"sheetId": 521635664},
}

# The explicit remaining defective rows that MUST keep fixes and yellow highlight:
DEFECTIVE_ROWS = {
    "Chapter 1: Introduction": {
        27: [
            "• Fix 1: Remove stray open quotation mark ('“') before 'Coat pattern'.",
            "• Fix 2: Paraphrase the near-verbatim copied sentence ('Coat pattern can be a way to tell two cows apart, but it shouldn't be a shortcut for assessing body condition') into your own words."
        ],
        88: [
            "• Fix 1: Missing paraphrase: Cell currently contains 'No paraphase....'. Paraphrase the three research questions into a smooth narrative paragraph."
        ],
        92: [
            "• Fix 1: Remove rogue floating '1.' enumeration marker from the middle of the narrative paragraph ('...comparative experimental design. 1. The first stage...')."
        ],
        101: [
            "• Fix 1: Fix broken grammar ('In the chapter 4 it discuss about' -> 'Chapter 4 outlines', 'interpretetion' -> 'interpretation', 'these experiments looks at' -> 'these experiments examine').",
            "• Fix 2: Paraphrase the second half into your own words instead of near-verbatim copying."
        ],
        108: [
            "• Fix 1: Change 'assignments' -> 'tasks' (machine learning models evaluate distinct computer vision tasks, not school assignments)."
        ],
        117: [
            "• Fix 1: Direct copy-paste alert: Paragraph is copied almost 100% verbatim from original (79% 4-gram overlap) with only trivial word swaps. Paraphrase substantially into your own words."
        ]
    },
    "Chapter 2: Literature Review": {
        13: [
            "• Fix 1: Direct copy-paste alert: 100% verbatim copy-paste from original (not paraphrased). Paraphrase in your own words while keeping model names and transfer learning intact."
        ],
        51: [
            "• Fix 1: Change 'fur patterns' -> 'coat patterns / coat markings' (cattle have hair coats and markings, not fur)."
        ],
        54: [
            "• Fix 1: Complete truncated final sentence: Paragraph cuts off mid-thought ('Therefore, the integration is not only about merging multiple predictions.'). Add the missing second half regarding how to determine what feature capacity can be shared without sacrificing individual task needs."
        ],
        124: [
            "• Fix 1: Change 'modern sensors' -> 'modern object detectors' (RT-DETR and Mask R-CNN are computer vision detector models, not physical hardware sensors)."
        ],
        203: [
            "• Fix 1: Paraphrase the verbatim copied final sentence ('They control how strongly tasks update the network, but they do not decide which visual features should be shared.') into your own words."
        ],
        212: [
            "• Fix 1: Paraphrase the verbatim copied concluding sentences ('One grouping may help one task while hurting another. Strong single-task references are therefore necessary before judging a shared model.') into your own words."
        ],
        218: [
            "• Fix 1: Paraphrase the verbatim copied final sentence ('This is why the thesis compares hard sharing with task-conditioned or partly private processing.') into your own words."
        ]
    },
    "Chapter 3: Requirements & Constraints": {
        59: [
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.1:timeline' -> 'Table 3.2' (remove the ':timeline' code tag and use correct Table 3.2 numbering)."
        ],
        72: [
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.2:risks' -> 'Table 3.3' (remove the ':risks' code tag and use correct Table 3.3 numbering)."
        ],
        79: [
            "• Fix 1: Change raw LaTeX tag remnant 'Table 3.4:costs' -> 'Table 3.4' (remove the ':costs' code tag).",
            "• Fix 2: Fix broken grammar in final clause ('In Table 3.4, reports only information...' -> 'Table 3.4 reports only information that was actually recorded')."
        ]
    }
}

def clean_col_d_for_approval(current_col_d: str) -> str:
    """Cleans Column D text by stripping fixes and ensuring approval header."""
    lines = current_col_d.split("\n")
    intact_lines = []
    in_fixes = False
    
    for line in lines:
        stripped = line.strip()
        if "🛠️ Required Fixes:" in stripped:
            in_fixes = True
            continue
        if in_fixes and stripped.startswith("📌 Words/phrases to keep intact:"):
            in_fixes = False
        if in_fixes:
            continue
        if stripped.startswith("✅ অর্থ ঠিক রাখা হইছে"):
            continue
        intact_lines.append(line)
        
    intact_text = "\n".join(intact_lines).strip()
    return f"✅ অর্থ ঠিক রাখা হইছে。\n\n{intact_text}"

def main():
    service = get_service()
    
    value_updates = []
    format_requests = []
    
    approved_summary = {}
    
    for sheet_title, meta in SHEET_METADATA.items():
        sheet_id = meta["sheetId"]
        res = service.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"'{sheet_title}'!A1:D300"
        ).execute()
        rows = res.get("values", [])
        
        defective_dict = DEFECTIVE_ROWS.get(sheet_title, {})
        approved_count = 0
        
        for idx, r in enumerate(rows, start=1):
            col_b = r[1] if len(r) > 1 else ""
            col_c = r[2] if len(r) > 2 else ""
            col_d = r[3] if len(r) > 3 else ""
            
            # Skip header or empty rows
            if not col_b or col_b.startswith("Original (DONT PARAPHRASE") or col_b.startswith("Original (Do Paraphrase"):
                continue
            if not col_c.strip():
                continue
                
            if idx in defective_dict:
                # Still defective -> keep fixes and keep yellow
                continue
                
            # This row is approved!
            # 1. Check if Col D needs update
            new_col_d = clean_col_d_for_approval(col_d)
            if new_col_d != col_d:
                value_updates.append({
                    "range": f"'{sheet_title}'!D{idx}",
                    "values": [[new_col_d]]
                })
                
            # 2. Reset Col C background to clean white
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": idx - 1,
                        "endRowIndex": idx,
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
            })
            approved_count += 1
            
        approved_summary[sheet_title] = approved_count
        
    print(f"Summary of approvals to apply:")
    for s, c in approved_summary.items():
        print(f"  - {s}: {c} approved rows")
    print(f"Total Column D text updates: {len(value_updates)}")
    print(f"Total Column C un-highlight requests: {len(format_requests)}")
    
    # 1. Execute value updates
    if value_updates:
        print("\nUpdating Column D texts via batchUpdate...")
        body = {
            "valueInputOption": "USER_ENTERED",
            "data": value_updates
        }
        res_vals = robust_execute(lambda: service.values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=body
        ).execute())
        print(f"[+] Successfully updated {res_vals.get('totalUpdatedCells')} cells in Column D.")
        
    # 2. Execute formatting requests
    if format_requests:
        print("\nResetting Column C backgrounds to white for approved rows...")
        body_fmt = {
            "requests": format_requests
        }
        res_fmt = robust_execute(lambda: service.batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=body_fmt
        ).execute())
        print(f"[+] Successfully applied {len(res_fmt.get('replies', []))} formatting updates.")

if __name__ == "__main__":
    main()
