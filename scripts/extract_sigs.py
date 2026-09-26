import os
import pypdf

def extract_page_info(pdf_path, page_num, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    reader = pypdf.PdfReader(pdf_path)
    page = reader.pages[page_num - 1]
    print(f"\n==========================================")
    print(f"File: {pdf_path} (Page {page_num})")
    print("--- TEXT ---")
    print(page.extract_text())
    print("--- IMAGES ---")
    print(f"Found {len(page.images)} images.")
    for idx, img in enumerate(page.images):
        fname = f"{os.path.basename(pdf_path)}_p{page_num}_img{idx}_{img.name}"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(img.data)
        print(f"  Saved image {idx}: {fpath} ({len(img.data)} bytes)")

out_dir = "scratch/extracted_signatures"
os.makedirs(out_dir, exist_ok=True)

# 1. Defense_Report_FINAL_DRAFT - APARUP CHOWDHURY.pdf
extract_page_info("P3 Samples/fall 2025/Defense_Report_FINAL_DRAFT - APARUP CHOWDHURY.pdf", 3, out_dir)
extract_page_info("P3 Samples/fall 2025/Defense_Report_FINAL_DRAFT - APARUP CHOWDHURY.pdf", 4, out_dir)

# 2. Final_Report - ABRAR MAHIR ROHAN.pdf
extract_page_info("P3 Samples/fall 2025/Final_Report - ABRAR MAHIR ROHAN.pdf", 3, out_dir)
extract_page_info("P3 Samples/fall 2025/Final_Report - ABRAR MAHIR ROHAN.pdf", 4, out_dir)

# 3. Signed Zihad report
extract_page_info("P3 Samples/fall 2025/T2510593_FINAL_YEAR_THESIS_Report_CSE400_Fall_2024_ONWARDS Report_signed - Mamnun Ahmed Zihad.pdf", 3, out_dir)
extract_page_info("P3 Samples/fall 2025/T2510593_FINAL_YEAR_THESIS_Report_CSE400_Fall_2024_ONWARDS Report_signed - Mamnun Ahmed Zihad.pdf", 4, out_dir)
