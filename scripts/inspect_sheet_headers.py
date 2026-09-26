import pickle
import os
from googleapiclient.discovery import build

TOKEN_PATH = r"C:\Users\Hasin\.gemini\config\google_token.pickle"
SPREADSHEET_ID = "14UIi22gtPx_ogVGBPfhV45rN1R-zTREAQG3Aymcqk0A"

with open(TOKEN_PATH, "rb") as token:
    creds = pickle.load(token)
service = build("sheets", "v4", credentials=creds)

sheet_metadata = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
sheets = sheet_metadata.get('sheets', '')

for s in sheets:
    title = s['properties']['title']
    if any(k in title for k in ["Chapter 1", "Chapter 2", "Chapter 3"]):
        print(f"\n--- Checking Sheet: {title} ---")
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"'{title}'!A1:E20"
        ).execute()
        rows = result.get('values', [])
        for r_idx, row in enumerate(rows[:10]):
            print(f"Row {r_idx+1}: {[c[:30] for c in row]}")
