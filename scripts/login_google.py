import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
token_path = WORKSPACE_ROOT / "token.json"
creds_path = WORKSPACE_ROOT / "credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file"
]

from google_auth_oauthlib.flow import InstalledAppFlow

print("[*] Starting OAuth flow...", flush=True)
flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
creds = flow.run_local_server(port=0, open_browser=True)

with open(token_path, "w") as token_file:
    token_file.write(creds.to_json())

print("\n[SUCCESS] token.json successfully created and saved!", flush=True)
