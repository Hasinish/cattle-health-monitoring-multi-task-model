import os
import pypdf
from PIL import Image

samples_dir = "P3 Samples"
all_pdfs = []
for root, dirs, files in os.walk(samples_dir):
    for f in files:
        if f.lower().endswith(".pdf"):
            all_pdfs.append(os.path.join(root, f))

out_dir = "scratch/hod_signatures"
os.makedirs(out_dir, exist_ok=True)

print("Checking Head of Department (Dr. Sadia Hamid Kazi) pages across all PDFs...")

found_sigs = []

for pdf_path in all_pdfs:
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i in range(min(6, len(reader.pages))):
            page = reader.pages[i]
            txt = page.extract_text() or ""
            if "sadia" in txt.lower() and "kazi" in txt.lower():
                imgs = page.images
                print(f"\nPDF: {pdf_path} (Page {i+1})")
                print(f"  Images on page: {len(imgs)}")
                for idx, img in enumerate(imgs):
                    fname = f"{os.path.basename(pdf_path)}_p{i+1}_img{idx}_{img.name}"
                    fpath = os.path.join(out_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(img.data)
                    im = Image.open(fpath)
                    print(f"    Image {idx}: {fname} | size={im.size} | format={im.format} | bytes={len(img.data)}")
                    found_sigs.append((pdf_path, i+1, fpath, im.size))
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

print(f"\nTotal images found on HOD pages: {len(found_sigs)}")
