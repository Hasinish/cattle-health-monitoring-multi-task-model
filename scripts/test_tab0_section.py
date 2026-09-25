"""
Test building section 1.1 with exact user formatting.
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.export_paraphrase_docs import clean_latex, parse_latex_sections

DOCUMENT_ID = "1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI"
TAB_ID = "t.0"
SCOPES = ["https://www.googleapis.com/auth/documents"]

token_path = WORKSPACE_ROOT / "token.json"
creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
docs_service = build("docs", "v1", credentials=creds)

# 1. Fetch current tab t.0
doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
body_content = t0["documentTab"]["body"]["content"]
end_idx = body_content[-1].get("endIndex", 1)

print(f"Tab t.0 current endIndex: {end_idx}")
# Clear tab t.0
if end_idx > 2:
    clear_req = [{"deleteContentRange": {"range": {"tabId": TAB_ID, "startIndex": 1, "endIndex": end_idx - 1}}}]
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": clear_req}).execute()
    print("Cleared tab t.0.")

# 2. Insert Section 1.1 Heading
heading_text = "1.1 Background\n"
insert_heading = [{
    "insertText": {
        "location": {"tabId": TAB_ID, "index": 1},
        "text": heading_text
    }
}]
docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_heading}).execute()

# Style heading as bold
style_heading = [{
    "updateTextStyle": {
        "range": {"tabId": TAB_ID, "startIndex": 1, "endIndex": 1 + len("1.1 Background")},
        "textStyle": {"bold": True, "fontSize": {"magnitude": 14, "unit": "PT"}},
        "fields": "bold,fontSize"
    }
}]
docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": style_heading}).execute()

# 3. Insert Table for Section 1.1 (6 paras -> 24 rows)
ch1 = (WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex").read_text(encoding="utf-8")
sections = parse_latex_sections(ch1)
bg_paras = [p.strip() for p in sections[1][2].split("\n\n") if p.strip()]

num_paras = len(bg_paras)
total_rows = num_paras * 4
print(f"Inserting table with {total_rows} rows for {num_paras} paras...")

# Insert table at end of segment
insert_tbl = [{
    "insertTable": {
        "rows": total_rows,
        "columns": 1,
        "endOfSegmentLocation": {"tabId": TAB_ID}
    }
}]
docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_tbl}).execute()

# 4. Fetch table
doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
tbl = [el["table"] for el in t0["documentTab"]["body"]["content"] if "table" in el][-1]
table_rows = tbl.get("tableRows", [])

row_texts = {}
for p_i, p_text in enumerate(bg_paras):
    r_orig_hdr = p_i * 4
    r_orig_txt = r_orig_hdr + 1
    r_para_hdr = r_orig_txt + 1
    r_para_txt = r_para_hdr + 1

    row_texts[r_orig_hdr] = "Original (Do Paraphrase)\n"
    row_texts[r_orig_txt] = p_text + "\n"
    row_texts[r_para_hdr] = "Paraphrased:\n"
    row_texts[r_para_txt] = "\n\n\n\n\n"

# Insert text in reverse row order
insert_reqs = []
for r_idx in range(len(table_rows) - 1, -1, -1):
    cell = table_rows[r_idx]["tableCells"][0]
    cell_start = cell["content"][0]["paragraph"]["elements"][0]["startIndex"]
    insert_reqs.append({
        "insertText": {
            "location": {"tabId": TAB_ID, "index": cell_start},
            "text": row_texts[r_idx]
        }
    })

docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_reqs}).execute()
print("Populated table cells with text.")

# 5. Fetch doc again and apply highlights to 'Original...' and 'Paraphrased:'
doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
tbl = [el["table"] for el in t0["documentTab"]["body"]["content"] if "table" in el][-1]

style_reqs = []
for r_i, r in enumerate(tbl.get("tableRows", [])):
    cell = r["tableCells"][0]
    p_el = cell["content"][0]["paragraph"]["elements"][0]
    text_content = p_el.get("textRun", {}).get("content", "")
    s_idx = p_el.get("startIndex", 0)

    if text_content.startswith("Original"):
        style_reqs.append({
            "updateTextStyle": {
                "range": {"tabId": TAB_ID, "startIndex": s_idx, "endIndex": s_idx + len(text_content.strip())},
                "textStyle": {
                    "bold": True,
                    "backgroundColor": {"color": {"rgbColor": {"red": 1.0, "green": 0.0, "blue": 0.0}}}
                },
                "fields": "bold,backgroundColor"
            }
        })
    elif text_content.startswith("Paraphrased:"):
        style_reqs.append({
            "updateTextStyle": {
                "range": {"tabId": TAB_ID, "startIndex": s_idx, "endIndex": s_idx + len(text_content.strip())},
                "textStyle": {
                    "bold": True,
                    "backgroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 1.0, "blue": 0.0}}}
                },
                "fields": "bold,backgroundColor"
            }
        })

if style_reqs:
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": style_reqs}).execute()
    print(f"Applied highlight styling to {len(style_reqs)} cells.")

print("SUCCESS: Section 1.1 formatted with exact user style!")
