import sys
import os
import re
import time

sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute
from scripts.apply_super_smart_paraphrase_workbench import (
    extract_verbatim_intact_terms,
    parse_markdown_bold,
    is_section_title
)

CH3_REVIEWS = {
    9: [
        "Change 'not leaking' -> 'leakage resistance / preventing data leakage' (formal ML requirement for train/test splits)",
        "Change 'being able to be used in real life' -> 'practical execution' (the model is an evaluated research prototype, not a finished product)"
    ],
    22: [
        "Change 'How accessible a farm is also depends on its setting' -> 'The accessibility of the monitoring system also depends on the farm environment' (refers to system deployability and accessibility, not physical farm visitation)"
    ],
    29: [
        "Change 'Processing perceptions' -> 'Perception preprocessing' (technical term for upstream detection and segmentation)",
        "Change 'It doesn't just work with the final prediction network' -> 'It is not limited to the final prediction network' (total computational cost includes upstream processing)"
    ],
    59: [
        "Change raw LaTeX tag 'Tabletab:timeline' -> 'Table 3.1' (or 'the project timeline table')"
    ],
    72: [
        "Change raw LaTeX tag 'Tabletab:risks' -> 'Table 3.2' (or 'the risk summary table')"
    ],
    75: [
        "Change 'The names of the cows in ScienceDB are unknown' -> 'Individual cow identities in ScienceDB are unknown' (in computer vision, cattle have IDs, not personal names)"
    ],
    79: [
        "Change raw LaTeX tag 'Tabletab:costs' -> 'Table 3.4' (or 'the cost table')",
        "Change 'deploying the farm' -> 'farm deployment' (or 'deploying the system on a farm')",
        "Change 'only information that was entered correctly is shown' -> 'reports only information that was actually recorded' (refers to empirically logged compute time, not data-entry error checking)"
    ],
    82: [
        "Change 'combining different jobs' -> 'multi-task integration' (multi-task learning is an AI framework, not combining farm workers' chores)",
        "Change 'recovery period' -> 'payback period' (the formal financial term for when an investment pays for itself)"
    ]
}

def run_update():
    service = get_service()
    tab_name = "Chapter 3: Requirements & Constraints"
    sheet_id = SHEET_IDS[tab_name]
    
    print(f"Fetching '{tab_name}'...")
    res = robust_execute(lambda: service.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{tab_name}'!A1:D100"
    ).execute())
    
    rows = res.get("values", [])
    print(f"Fetched {len(rows)} rows.")
    
    requests = []
    paras_count = 0
    headers_count = 0
    titles_count = 0
    flagged_count = 0
    approved_count = 0
    
    for idx, r in enumerate(rows, start=1):
        row_idx = idx - 1
        col_b = r[1].strip() if len(r) > 1 else ""
        col_c = r[2].strip() if len(r) > 2 else ""
        
        # 1. Section Title Row
        if is_section_title(col_b):
            titles_count += 1
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": ""},
                                    "textFormatRuns": []
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns"
                }
            })
            continue
            
        # 2. Table Header Row ('Original (Do Paraphrase...')
        if col_b.startswith("Original (Do Paraphrase") or col_b.startswith("Original (DONT"):
            headers_count += 1
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": "রিভিউ ও ফিডব্যাক 📝 (Review Notes)"},
                                    "textFormatRuns": [
                                        {"startIndex": 0, "format": {"bold": True}}
                                    ],
                                    "userEnteredFormat": {
                                        "textFormat": {"bold": True, "fontSize": 10},
                                        "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                                        "verticalAlignment": "MIDDLE",
                                        "wrapStrategy": "WRAP"
                                    }
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns,userEnteredFormat(textFormat,backgroundColor,verticalAlignment,wrapStrategy)"
                }
            })
            continue
            
        # 3. Genuine Academic Paragraph
        if len(col_b) > 35:
            paras_count += 1
            verbatim_terms = extract_verbatim_intact_terms(col_b)
            intact_str = ", ".join(verbatim_terms) if verbatim_terms else "Task-specific terminology"
            
            if idx in CH3_REVIEWS:
                flagged_count += 1
                fixes_lines = [f"• **Fix {f_idx+1}:** {f}" for f_idx, f in enumerate(CH3_REVIEWS[idx])]
                fixes_str = "\n".join(fixes_lines)
                md_text = (
                    f"**🛠️ Required Fixes:**\n{fixes_str}\n\n"
                    f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
                )
                bg_color = {"red": 1.0, "green": 0.98, "blue": 0.8} # Soft pastel yellow for attention
            else:
                approved_count += 1
                md_text = (
                    f"**✅ অর্থ ঠিক রাখা হইছে।**\n\n"
                    f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
                )
                bg_color = {"red": 1.0, "green": 1.0, "blue": 1.0} # Clean white for approved
                
            clean_text, runs = parse_markdown_bold(md_text)
            
            # Update Col D
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": clean_text},
                                    "textFormatRuns": runs,
                                    "userEnteredFormat": {
                                        "wrapStrategy": "WRAP",
                                        "verticalAlignment": "TOP",
                                        "textFormat": {"fontSize": 10}
                                    }
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns,userEnteredFormat(wrapStrategy,verticalAlignment,textFormat)"
                }
            })
            
            # Update Col C background highlight
            requests.append({
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
                            "backgroundColor": bg_color
                        }
                    },
                    "fields": "userEnteredFormat.backgroundColor"
                }
            })
            
    print(f"Prepared {len(requests)} update requests:")
    print(f"  Titles: {titles_count}")
    print(f"  Headers: {headers_count}")
    print(f"  Total paras: {paras_count}")
    print(f"  Flagged rows (need attention): {flagged_count}")
    print(f"  Approved rows (meaning correct): {approved_count}")
    
    if requests:
        print("\nExecuting live batchUpdate to Google Sheets...")
        res = robust_execute(lambda: service.batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"requests": requests}
        ).execute())
        print(f"SUCCESS! Perfectly updated Chapter 3 workbench reviews in Google Sheets!")

if __name__ == "__main__":
    run_update()
