import sys, time, json, hashlib
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Ensure utf-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SPREADSHEET_ID = '14UIi22gtPx_ogVGBPfhV45rN1R-zTREAQG3Aymcqk0A'

def get_service():
    creds = Credentials.from_authorized_user_file('token.json')
    return build('sheets', 'v4', credentials=creds)

def main():
    service = get_service()
    reported = {}  # key: (tab, row), value: hash(text)
    
    print("[WATCHER_ACTIVE] Paraphrase Sheet Watcher running in background. Polling every 15 seconds.")
    sys.stdout.flush()
    
    while True:
        try:
            meta = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
            sheets = meta.get('sheets', [])
            
            found_any = False
            for s in sheets:
                title = s['properties']['title']
                res = service.spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f'{title}!A1:D100'
                ).execute()
                vals = res.get('values', [])
                
                for idx, row in enumerate(vals, start=1):
                    a = row[0].strip() if len(row) > 0 and row[0] else ''
                    b = row[1].strip() if len(row) > 1 and row[1] else ''
                    c = row[2].strip() if len(row) > 2 and row[2] else ''
                    d = row[3].strip() if len(row) > 3 and row[3] else ''
                    
                    if a.startswith('Para'):
                        if c and c != 'Paraphrased:' and c != 'Paraphrased (AI Detection Free)':
                            if d != 'Correct ✅':
                                text_hash = hashlib.md5(c.encode('utf-8')).hexdigest()
                                key = (title, idx)
                                if reported.get(key) != text_hash:
                                    reported[key] = text_hash
                                    found_any = True
                                    print(f"\n[NEW_PARAPHRASE_DETECTED]")
                                    print(f"Tab: {title}")
                                    print(f"Row: {idx}")
                                    print(f"Para: {a}")
                                    print(f"ORIGINAL:\n{b}")
                                    print(f"PARAPHRASE:\n{c}")
                                    print(f"[END_PARAPHRASE]\n")
                                    sys.stdout.flush()
            
        except Exception as e:
            print(f"[WATCHER_ERROR] {e}")
            sys.stdout.flush()
            time.sleep(10)
            try:
                service = get_service()
            except Exception:
                pass
                
        time.sleep(15)

if __name__ == '__main__':
    main()
