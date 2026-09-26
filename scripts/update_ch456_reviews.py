import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def update_reviews():
    service = get_service()

    # Chapter 4 reviews (Rows 2, 5)
    ch4_updates = [
        {"range": "'Chapter 4: Proposed Methodology'!D2", "values": [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]},
        {"range": "'Chapter 4: Proposed Methodology'!D5", "values": [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]}
    ]

    # Chapter 5 reviews (Rows 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41)
    ch5_rows = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41]
    ch5_updates = [
        {"range": f"'Chapter 5: Result Analysis'!D{r}", "values": [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]}
        for r in ch5_rows
    ]

    # Chapter 6 reviews (Rows 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86)
    ch6_rows = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86]
    ch6_updates = []
    for r in ch6_rows:
        if r == 14:
            # Subtle observation on Row 14: "decreased from 0.1709 to 0.1788"
            ch6_updates.append({
                "range": f"'Chapter 6: Conclusion'!D{r}",
                "values": [["⚠️ ছোট নোট: 'decreased from 0.1709 to 0.1788' -> এরর বাড়ছে তাই 'increased' বা 'degraded' লিখলে ভালো। বাকি সব ঠিক আছে!"]]
            })
        else:
            ch6_updates.append({
                "range": f"'Chapter 6: Conclusion'!D{r}",
                "values": [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]
            })

    all_data = ch4_updates + ch5_updates + ch6_updates

    body = {
        "valueInputOption": "USER_ENTERED",
        "data": all_data
    }
    res = service.values().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
    print(f"Updated {len(all_data)} review cells in Column D successfully!")

if __name__ == "__main__":
    update_reviews()
