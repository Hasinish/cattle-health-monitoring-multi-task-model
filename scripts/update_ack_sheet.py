import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def main():
    service = get_service()
    updated_orig = "We also extend our sincere thanks to our co-supervisors, Mehedi Hasan Emo, Lecturer, and Mollah MD Saif, Lecturer, Department of Computer Science and Engineering, Brac University, for their constructive feedback and assistance during our research evaluations."
    
    # Update only Row 8 Col B
    service.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range="'Acknowledgement'!B8",
        valueInputOption="USER_ENTERED",
        body={"values": [[updated_orig]]}
    ).execute()
    print("Updated Row 8 Col B in 'Acknowledgement' successfully!")

if __name__ == "__main__":
    main()
