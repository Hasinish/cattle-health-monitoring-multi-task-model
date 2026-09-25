"""
Google Docs Automated Uploader for Thesis Paraphrasing Tabs.

Target Document ID: 1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI
Target URL: https://docs.google.com/document/d/1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI/edit

Prerequisites:
  1. `credentials.json` (OAuth 2.0 Desktop Client ID downloaded from Google Cloud Console)
     placed in the workspace root or script directory.
  2. Google Docs API and Google Drive API enabled in your Google Cloud project.
"""

import os
import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
LATEX_ROOT = WORKSPACE_ROOT / "cattle_thesis_p3_latex"
DOCS_DIR = LATEX_ROOT / "paraphrasing_docs"
MASTER_DOCX = DOCS_DIR / "Master_Freeze_Safe_Paraphrasing_All_Tabs.docx"

DOCUMENT_ID = "1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI"
SCOPES = [
    "https://www.googleapis.com/auth/documents"
]


def find_credentials():
    candidates = [
        WORKSPACE_ROOT / "credentials.json",
        Path.cwd() / "credentials.json",
        Path.home() / "credentials.json",
        Path.home() / "Downloads" / "credentials.json",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 10:
            return c
    # Check for client_secret_*.json
    for pattern in ["client_secret*.json", "credentials*.json"]:
        matches = list(WORKSPACE_ROOT.glob(pattern)) + list(Path.home().glob(pattern)) + list((Path.home() / "Downloads").glob(pattern))
        if matches:
            return matches[0]
    return None


def authenticate():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    token_path = WORKSPACE_ROOT / "token.json"

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            cred_file = find_credentials()
            if not cred_file:
                print("\n" + "=" * 70)
                print("[ERROR] `credentials.json` NOT FOUND!")
                print("=" * 70)
                print("Google API strictly requires an OAuth 2.0 Desktop Client ID.")
                print("\nTo get `credentials.json` in 2 minutes:")
                print("1. Go to: https://console.cloud.google.com/apis/credentials")
                print("2. Select or create any project.")
                print("3. Click 'Enable APIs and Services' -> Enable 'Google Docs API'.")
                print("4. Go to 'Credentials' -> 'Create Credentials' -> 'OAuth client ID'.")
                print("   (Application type: Desktop app, Name: ThesisUploader)")
                print("5. Click 'Download JSON', rename it to `credentials.json`,")
                print(f"   and save it to: {WORKSPACE_ROOT / 'credentials.json'}")
                print("=" * 70 + "\n")
                sys.exit(1)

            print(f"[*] Found credentials: {cred_file}", flush=True)
            flow = InstalledAppFlow.from_client_secrets_file(str(cred_file), SCOPES)
            print("[*] Opening your default browser for Google OAuth sign-in...", flush=True)
            print("[*] Waiting for authorization in browser...", flush=True)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())
        print(f"[OK] Authentication saved to {token_path}", flush=True)

    return creds


def update_target_document(docs_service, document_id: str):
    """Update existing Google Doc with structured freeze-safe tabs content."""
    try:
        from scripts.export_paraphrase_docs import clean_latex, parse_latex_sections
    except ModuleNotFoundError:
        from export_paraphrase_docs import clean_latex, parse_latex_sections

    print(f"[*] Fetching target document metadata for ID: {document_id}...", flush=True)
    doc = docs_service.documents().get(documentId=document_id).execute()
    title = doc.get("title", "")
    print(f"[OK] Target Document Title: '{title}'", flush=True)

    # Read the 5 tabs content
    tabs_data = [
        ("TAB 1: INTRODUCTION AND PROBLEM FORMULATION", LATEX_ROOT / "chapters" / "chapter_1.tex", True),
        ("TAB 2: LITERATURE REVIEW (DOMAIN AND METHODS)", LATEX_ROOT / "chapters" / "chapter_2.tex", True),
        ("TAB 3: REQUIREMENTS, IMPACTS AND CONSTRAINTS", LATEX_ROOT / "chapters" / "chapter_3.tex", True),
        ("TAB 4: DATASETS, PREPROCESSING, AND SPLIT INTEGRITY", LATEX_ROOT / "chapters" / "chapter_5.tex", False),
        ("TAB 5: SINGLE-TASK BASELINES AND PERCEPTION ARCHITECTURES", LATEX_ROOT / "chapters" / "chapter_5.tex", False),
    ]

    # Build the full text to insert
    full_text = "🔒 CSE400 P3 THESIS FREEZE-SAFE PARAPHRASING MASTER\n"
    full_text += "=" * 60 + "\n"
    full_text += "STRICT RULES FOR TEAMMATES:\n"
    full_text += "1. NEVER delete or modify '[cite: key]' tags. Move them naturally with the sentence.\n"
    full_text += "2. Paraphrase in academic third-person tone. Never use 'Phase 1' or 'Phase 2'.\n"
    full_text += "3. Never change numbers, cow counts, dates, or metrics.\n"
    full_text += "=" * 60 + "\n\n"

    # Tab 1
    ch1 = (LATEX_ROOT / "chapters" / "chapter_1.tex").read_text(encoding="utf-8")
    for lvl, t, b in parse_latex_sections(ch1):
        full_text += f"\n### {t.upper()}\n"
        for p in b.split("\n\n"):
            if p.strip():
                full_text += f"[ORIGINAL]: {p.strip()}\n"
                full_text += f"[PARAPHRASED]: [Type humanized version here...]\n\n"

    # Tab 2
    full_text += "\n\n" + "=" * 60 + "\nTAB 2: LITERATURE REVIEW (DOMAIN AND METHODS)\n" + "=" * 60 + "\n\n"
    ch2 = (LATEX_ROOT / "chapters" / "chapter_2.tex").read_text(encoding="utf-8")
    for lvl, t, b in parse_latex_sections(ch2):
        full_text += f"\n### {t.upper()}\n"
        for p in b.split("\n\n"):
            if p.strip():
                full_text += f"[ORIGINAL]: {p.strip()}\n"
                full_text += f"[PARAPHRASED]: [Type humanized version here...]\n\n"

    # Tab 3
    full_text += "\n\n" + "=" * 60 + "\nTAB 3: REQUIREMENTS, IMPACTS AND CONSTRAINTS\n" + "=" * 60 + "\n\n"
    ch3 = (LATEX_ROOT / "chapters" / "chapter_3.tex").read_text(encoding="utf-8")
    for lvl, t, b in parse_latex_sections(ch3):
        full_text += f"\n### {t.upper()}\n"
        for p in b.split("\n\n"):
            if p.strip():
                full_text += f"[ORIGINAL]: {p.strip()}\n"
                full_text += f"[PARAPHRASED]: [Type humanized version here...]\n\n"

    # Tab 4 & 5
    ch5 = (LATEX_ROOT / "chapters" / "chapter_5.tex").read_text(encoding="utf-8")
    all_ch5 = parse_latex_sections(ch5)

    full_text += "\n\n" + "=" * 60 + "\nTAB 4: DATASETS, PREPROCESSING, AND SPLIT INTEGRITY\n" + "=" * 60 + "\n\n"
    for lvl, t, b in all_ch5:
        tl = t.lower()
        if "dataset" in tl or "data collection" in tl or "split" in tl or "feasibility" in tl or "evaluation protocol" in tl:
            full_text += f"\n### {t.upper()}\n"
            for p in b.split("\n\n"):
                if p.strip():
                    full_text += f"[ORIGINAL]: {p.strip()}\n"
                    full_text += f"[PARAPHRASED]: [Type humanized version here...]\n\n"

    full_text += "\n\n" + "=" * 60 + "\nTAB 5: SINGLE-TASK BASELINES AND PERCEPTION ARCHITECTURES\n" + "=" * 60 + "\n\n"
    for lvl, t, b in all_ch5:
        tl = t.lower()
        if not ("dataset" in tl or "data collection" in tl or "split" in tl or "feasibility" in tl or "evaluation protocol" in tl or "multi-task" in tl or "reporting boundary" in tl):
            full_text += f"\n### {t.upper()}\n"
            for p in b.split("\n\n"):
                if p.strip():
                    full_text += f"[ORIGINAL]: {p.strip()}\n"
                    full_text += f"[PARAPHRASED]: [Type humanized version here...]\n\n"

    print(f"[*] Prepared payload: {len(full_text):,} characters across all 5 freeze-safe tabs.")

    # In Google Docs API, we insert text at index 1
    requests = [
        {
            "insertText": {
                "location": {"index": 1},
                "text": full_text
            }
        }
    ]

    print("[*] Dispatching batchUpdate request to Google Docs API...")
    result = docs_service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
    print("[SUCCESS] Google Doc successfully populated via Google Docs API! 🎉")
    print(f" -> URL: https://docs.google.com/document/d/{document_id}/edit")


def main():
    from googleapiclient.discovery import build

    print("=== GOOGLE DOCS AUTOMATED API UPLOADER ===")
    creds = authenticate()
    docs_service = build("docs", "v1", credentials=creds)
    update_target_document(docs_service, DOCUMENT_ID)


if __name__ == "__main__":
    main()
