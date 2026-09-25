"""
Fix titles (clear Col D for section headings and bold Col B) and apply bold markdown
formatting for all review guides across Chapters 1, 2, and 3.
"""

import sys
import re
import socket
import time
socket.setdefaulttimeout(20.0)

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute
from scripts.format_all_chapters_comprehensive import CH1_FLAGGED, CH2_FLAGGED, CH3_FLAGGED, FLAGGED_BY_TAB, extract_intact_terms

def is_section_title(text):
    t = text.strip()
    if re.match(r'^\d+\.\d+', t):
        return True
    if t.startswith("[Table Reference") or t.startswith("Chapter "):
        return True
    return False

def format_cell(tab_name, row_num, col_b, col_c):
    flagged_dict = FLAGGED_BY_TAB.get(tab_name, {})
    
    # Flagged paragraph
    if row_num in flagged_dict:
        item = flagged_dict[row_num]
        roast = item["roast"]
        intact_str = ", ".join(item["intact"])
        
        fixes_lines = []
        for idx, f in enumerate(item["fixes"], start=1):
            # Clean if already has Fix X:
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
    
    # Approved paragraph
    if col_c and col_c.strip() and col_c.strip() not in ["Paraphrased:", "LEAVE BLANK (Do Not Paraphrase):"]:
        return (
            f"**✅ অর্থ ঠিক রাখা হইছে।**\n\n"
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
        )
    else:
        # Pending paragraph
        return (
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
        )

def run():
    service = get_service()
    ranges = [f"'{tab}'!A1:D265" for tab in SHEET_IDS.keys()]
    res = robust_execute(lambda: service.values().batchGet(spreadsheetId=SPREADSHEET_ID, ranges=ranges).execute())
    
    for tab_name, val_range in zip(SHEET_IDS.keys(), res.get("valueRanges", [])):
        sheet_id = SHEET_IDS[tab_name]
        rows = val_range.get("values", [])
        
        updates = []
        format_requests = []
        titles_cleared = 0
        paras_updated = 0
        
        for idx, r in enumerate(rows, start=1):
            row_num = idx
            col_b = r[1].strip() if len(r) > 1 else ""
            col_c = r[2].strip() if len(r) > 2 else ""
            
            # 1. Section Title Row
            if is_section_title(col_b):
                # Clear Col D completely for section titles!
                updates.append({
                    "range": f"'{tab_name}'!D{row_num}",
                    "values": [[""]]
                })
                # Make Section Title in Col B BOLD
                format_requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": idx - 1,
                            "endRowIndex": idx,
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
                titles_cleared += 1
                
            # 2. Table Header Row ('Original (Do Paraphrase...')
            elif col_b.startswith("Original (Do Paraphrase"):
                updates.append({
                    "range": f"'{tab_name}'!D{row_num}",
                    "values": [["রিভিউ ও রোস্টিং ☕ (Review Notes)"]]
                })
                format_requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": idx - 1,
                            "endRowIndex": idx,
                            "startColumnIndex": 3,
                            "endColumnIndex": 4
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "textFormat": {"bold": True, "fontSize": 10},
                                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(textFormat,backgroundColor,verticalAlignment)"
                    }
                })
                
            # 3. Paragraph Row (content row)
            elif len(col_b) > 35:
                content = format_cell(tab_name, row_num, col_b, col_c)
                updates.append({
                    "range": f"'{tab_name}'!D{row_num}",
                    "values": [[content]]
                })
                paras_updated += 1
                
                # If flagged -> pastel yellow on Col C
                if row_num in FLAGGED_BY_TAB.get(tab_name, {}):
                    format_requests.append({
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": idx - 1,
                                "endRowIndex": idx,
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

        # Apply value updates
        if updates:
            print(f"[*] '{tab_name}': Updating {len(updates)} rows (Cleared {titles_cleared} titles, Updated {paras_updated} paras)...")
            robust_execute(lambda: service.values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "USER_ENTERED", "data": updates}
            ).execute())
            
        # Apply formatting
        if format_requests:
            robust_execute(lambda: service.batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"requests": format_requests}
            ).execute())
            
    print("\n🏆 FINISHED! All titles cleared from Col D, Col B titles bolded, and paragraph labels formatted with clean bold markers!")

if __name__ == "__main__":
    run()
