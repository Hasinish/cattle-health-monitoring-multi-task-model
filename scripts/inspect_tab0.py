import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
token_path = WORKSPACE_ROOT / "token.json"
creds = Credentials.from_authorized_user_file(str(token_path), ["https://www.googleapis.com/auth/documents"])
docs_service = build("docs", "v1", credentials=creds)
doc_id = "1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI"

doc = docs_service.documents().get(documentId=doc_id, includeTabsContent=True).execute()
t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == "t.0"][0]
body = t0["documentTab"]["body"]["content"]

print(f"Total elements in Tab t.0 body: {len(body)}")
for idx, el in enumerate(body):
    if "paragraph" in el:
        p_text = "".join([e.get("textRun", {}).get("content", "") for e in el["paragraph"]["elements"]])
        if p_text.strip():
            print(f"Element {idx} [Paragraph]: {repr(p_text)}")
    elif "table" in el:
        tbl = el["table"]
        rows = tbl.get("tableRows", [])
        print(f"Element {idx} [Table]: {len(rows)} rows, {tbl.get('columns', 0)} cols")
        for r_i, r in enumerate(rows):
            for c_i, c in enumerate(r.get("tableCells", [])):
                cell_text = ""
                for c_el in c.get("content", []):
                    if "paragraph" in c_el:
                        cell_text += "".join([e.get("textRun", {}).get("content", "") for e in c_el["paragraph"]["elements"]])
                cell_style = c.get("tableCellStyle", {})
                bg = cell_style.get("backgroundColor", {})
                print(f"  Row {r_i}, Col {c_i} (bg={bg}):")
                print(f"    Content: {repr(cell_text.strip())}")
                # Print formatting details of elements
                for c_el in c.get("content", []):
                    if "paragraph" in c_el:
                        for e in c_el["paragraph"]["elements"]:
                            if "textRun" in e:
                                tr = e["textRun"]
                                ts = tr.get("textStyle", {})
                                if ts:
                                    print(f"      TextRun: {repr(tr.get('content', ''))} | style={ts}")
