import pdfplumber

def check_pdf_layout(pdf_path, page_num):
    print(f"\n==========================================")
    print(f"File: {pdf_path} (Page {page_num})")
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        print("Text:")
        print(page.extract_text())
        print(f"\nImages on page: {len(page.images)}")
        for idx, img in enumerate(page.images):
            print(f"  Image {idx}: x0={img['x0']:.1f}, top={img['top']:.1f}, x1={img['x1']:.1f}, bottom={img['bottom']:.1f}, width={img['width']:.1f}, height={img['height']:.1f}")

# Check the files with images on HOD or supervisor pages
check_pdf_layout("P3 Samples/[final draft] report - Farhan Haseen Prantor.pdf", 3)
check_pdf_layout("P3 Samples/Final_Report_Group7.pdf", 4)
check_pdf_layout("P3 Samples/T2410228_Final Report - TAFSIRUL HOQUE.pdf", 4)
check_pdf_layout("P3 Samples/fall 2025/Final_Report - ABRAR MAHIR ROHAN.pdf", 3)
check_pdf_layout("P3 Samples/fall 2025/Final_Report - ABRAR MAHIR ROHAN.pdf", 4)
