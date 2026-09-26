import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()

# Let's inspect the Acknowledgement tab rows
res = service.values().get(spreadsheetId=SPREADSHEET_ID, range="'Acknowledgement'!A1:D16").execute()
rows = res.get('values', [])
for idx, r in enumerate(rows, 1):
    print(f"Row {idx}: {r}")

# The user's paraphrased paragraphs:
paras = [
    "First of all, all praises to Almighty Allah for giving us the power, determination, and endurance to finish this undergraduate thesis.",
    "We would like to take this opportunity to thank our supervisor for all the valuable help that he has provided during this research project. His knowledge about computer vision, robotics and machine intelligence was of great assistance to us in finishing the work.",
    "We would also like to sincerely thank our co-supervisors for his valuable feedback and support during the evaluation of our research.",
    "It is our great pleasure to thank the Department of Computer Science and Engineering, Brac University, for the provision of state-of-the-art computational facilities and an enabling academic atmosphere for our research. We also take this opportunity to extend our heartfelt gratitude to the authors and institutions behind the benchmark datasets used in this research, without whose open access efforts this study would not have been possible.",
    "Lastly, our deepest appreciation goes out to our parents, families, and friends who have provided us with constant love, guidance, and prayers through our entire academic experience."
]

# Update Column C for Rows 2, 5, 8, 11, 14
row_indices = [2, 5, 8, 11, 14]
data = []
for r_idx, p in zip(row_indices, paras):
    data.append({
        'range': f"'Acknowledgement'!C{r_idx}",
        'values': [[p]]
    })
    data.append({
        'range': f"'Acknowledgement'!D{r_idx}",
        'values': [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]
    })

service.values().batchUpdate(
    spreadsheetId=SPREADSHEET_ID,
    body={'valueInputOption': 'USER_ENTERED', 'data': data}
).execute()

print("Successfully synced Acknowledgement tab in Google Sheets!")
