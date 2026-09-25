"""
Scan Tab t.0 in Google Docs for any residual LaTeX syntax, escapes, math mode, or formatting artifacts.
"""

import sys
import re
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
tab_id = "t.0"

doc = docs_service.documents().get(documentId=doc_id, includeTabsContent=True).execute()
t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == tab_id][0]
body = t0["documentTab"]["body"]["content"]

print(f"=== SCANNING TAB {tab_id} FOR LATEX ISSUES ===")

issues_found = []

for el_idx, el in enumerate(body):
    if "paragraph" in el:
        p_text = "".join([e.get("textRun", {}).get("content", "") for e in el["paragraph"]["elements"]]).strip()
        # Check headings
        if "\\" in p_text or "{" in p_text or "$" in p_text or "~" in p_text or "--" in p_text:
            issues_found.append(("Heading", el_idx, p_text))
    elif "table" in el:
        for r_i, r in enumerate(el["table"].get("tableRows", [])):
            cell = r["tableCells"][0]
            for c_el in cell.get("content", []):
                if "paragraph" in c_el:
                    cell_text = "".join([e.get("textRun", {}).get("content", "") for e in c_el["paragraph"]["elements"]]).strip()
                    if cell_text.startswith("Original"):
                        continue
                    if cell_text.startswith("Paraphrased:"):
                        continue
                    if not cell_text:
                        continue

                    # Check for LaTeX patterns
                    latex_patterns = [
                        (r'\\[a-zA-Z]+', "Backslash command"),
                        (r'\$', "Math delimiter ($)"),
                        (r'\{|\}', "Curly brace"),
                        (r'~', "Tilde non-breaking space"),
                        (r'--|---', "LaTeX dash (-- or ---)"),
                        (r'``|\'\'', "LaTeX quotes (`` or '')"),
                        (r'_\{|\^\{|_|\^', "Subscript/superscript"),
                        (r'\[eq:\s*[^\]]+\]', "Raw eq ref tag"),
                        (r'\[ref:\s*[^\]]+\]', "Raw section ref tag"),
                        (r'\[cite:\s*[^\]]+\]', "Residual cite tag"),
                    ]

                    cell_issues = []
                    for pat, name in latex_patterns:
                        matches = re.findall(pat, cell_text)
                        if matches:
                            cell_issues.append((name, matches))

                    if cell_issues:
                        issues_found.append((f"Table Row {r_i}", cell_text, cell_issues))

print(f"Total issues/cells with artifacts found: {len(issues_found)}")
for item in issues_found[:15]:
    if item[0] == "Heading":
        print(f"\n[HEADING ISSUE] Element {item[1]}: {item[2]}")
    else:
        print(f"\n[{item[0]}]:")
        print(f"  Snippet: {item[1][:100]}...")
        print(f"  Patterns: {item[2]}")
