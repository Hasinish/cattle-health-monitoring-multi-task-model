"""
Apply Native Rich Text Bold Formatting to Thesis Paraphrasing Workbench.
Replaces all raw markdown asterisks (**) with Google Sheets API native textFormatRuns.
Eliminates all asterisks from Column D while making:
  - 📌 Words/phrases to keep intact: (BOLD)
  - 🛠️ Required Fixes: (BOLD)
  - • Fix 1:, • Fix 2: (BOLD)
  - ✅ অর্থ ঠিক রাখা হইছে। (BOLD)
Also ensures:
  - Section titles in Column B are bolded (fontSize 11, bold True)
  - Column D for section titles is completely cleared ("")
  - Table header in Column D is bolded with light gray background
  - Flagged rows in Column C preserve light pastel yellow background
"""

import sys
import re
import socket
import time

socket.setdefaulttimeout(25.0)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute
from scripts.format_all_chapters_comprehensive import FLAGGED_BY_TAB, extract_intact_terms

def parse_markdown_bold(md_text):
    """
    Parses a markdown string containing **bold** spans.
    Returns:
      clean_text: string with all '**' removed
      runs: list of Google Sheets API textFormatRun dicts with UTF-16 code unit offsets
    """
    clean_chars = []
    runs = []
    in_bold = False
    utf16_idx = 0
    
    i = 0
    while i < len(md_text):
        if md_text[i:i+2] == '**':
            in_bold = not in_bold
            runs.append({
                "startIndex": utf16_idx,
                "format": {"bold": in_bold}
            })
            i += 2
        else:
            char = md_text[i]
            clean_chars.append(char)
            # UTF-16 code units for this code point
            utf16_units = len(char.encode('utf-16-le')) // 2
            utf16_idx += utf16_units
            i += 1

    clean_text = "".join(clean_chars)
    
    if in_bold:
        runs.append({
            "startIndex": utf16_idx,
            "format": {"bold": False}
        })
        
    # Deduplicate / clean adjacent identical states
    filtered_runs = []
    current_bold = False
    for r in runs:
        if r["format"]["bold"] != current_bold:
            filtered_runs.append(r)
            current_bold = r["format"]["bold"]
            
    return clean_text, filtered_runs

def is_section_title(text):
    t = text.strip()
    if re.match(r'^\d+\.\d+', t):
        return True
    if t.startswith("[Table Reference") or t.startswith("Chapter "):
        return True
    return False

def build_markdown_content(tab_name, row_num, col_b, col_c):
    flagged_dict = FLAGGED_BY_TAB.get(tab_name, {})
    
    if row_num in flagged_dict:
        item = flagged_dict[row_num]
        roast = item["roast"]
        intact_str = ", ".join(item["intact"])
        fixes_lines = []
        for idx, f in enumerate(item["fixes"], start=1):
            clean_f = re.sub(r'^Fix\s*\d+:\s*', '', f)
            fixes_lines.append(f"• **Fix {idx}:** {clean_f}")
        fixes_str = "\n".join(fixes_lines)
        
        return (
            f"{roast}\n\n"
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}\n\n"
            f"**🛠️ Required Fixes:**\n{fixes_str}"
        )
    
    terms = extract_intact_terms(col_b)
    intact_str = ", ".join(terms) if terms else "Core task terminology and model names"
    
    if col_c and col_c.strip() and col_c.strip() not in ["Paraphrased:", "LEAVE BLANK (Do Not Paraphrase):"]:
        return (
            f"**✅ অর্থ ঠিক রাখা হইছে।**\n\n"
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
        )
    else:
        return (
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
        )

def process_chapter(service, tab_name, sheet_id):
    print(f"\n========================================================")
    print(f"[*] Processing '{tab_name}' (sheetId: {sheet_id})...")
    
    # Read columns A to D
    res = robust_execute(lambda: service.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{tab_name}'!A1:D265"
    ).execute())
    
    rows = res.get("values", [])
    print(f"[*] Fetched {len(rows)} rows.")
    
    requests = []
    titles_count = 0
    headers_count = 0
    paras_count = 0
    flagged_count = 0
    
    for idx, r in enumerate(rows, start=1):
        row_idx = idx - 1 # 0-indexed for batchUpdate
        col_b = r[1].strip() if len(r) > 1 else ""
        col_c = r[2].strip() if len(r) > 2 else ""
        
        # 1. Section Title Row
        if is_section_title(col_b):
            titles_count += 1
            # Clear Col D
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
            # Make Col B bold (11pt)
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex": 2
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"bold": True, "fontSize": 11}
                        }
                    },
                    "fields": "userEnteredFormat.textFormat(bold,fontSize)"
                }
            })
            
        # 2. Table Header Row
        elif col_b.startswith("Original (Do Paraphrase"):
            headers_count += 1
            # Format Col D header
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
                                    "userEnteredValue": {"stringValue": "রিভিউ ও রোস্টিং ☕ (Review Notes)"},
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
            
        # 3. Paragraph Row
        elif len(col_b) > 35:
            paras_count += 1
            md_text = build_markdown_content(tab_name, idx, col_b, col_c)
            clean_text, runs = parse_markdown_bold(md_text)
            
            # Update Col D with clean text and native rich text format runs
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
            
            # If flagged -> pastel yellow on Col C
            if idx in FLAGGED_BY_TAB.get(tab_name, {}):
                flagged_count += 1
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
                                "backgroundColor": {"red": 1.0, "green": 0.98, "blue": 0.8}
                            }
                        },
                        "fields": "userEnteredFormat.backgroundColor"
                    }
                })

    print(f"[*] Prepared {len(requests)} atomic requests: {paras_count} paras, {titles_count} titles, {headers_count} headers, {flagged_count} flagged.")
    
    # Execute batchUpdate
    if requests:
        start_t = time.time()
        res = robust_execute(lambda: service.batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"requests": requests}
        ).execute())
        elapsed = time.time() - start_t
        print(f"[SUCCESS] Updated '{tab_name}' in {elapsed:.2f}s!")

def main():
    service = get_service()
    total_start = time.time()
    
    for tab_name, sheet_id in SHEET_IDS.items():
        process_chapter(service, tab_name, sheet_id)
        
    print(f"\n🎉 ALL 3 CHAPTERS SUCCESSFULLY UPDATED WITH NATIVE RICH TEXT BOLD in {time.time() - total_start:.2f}s!")
    print("All ** asterisks have been stripped, and headings/labels are natively bolded in Google Sheets!")

if __name__ == "__main__":
    main()
