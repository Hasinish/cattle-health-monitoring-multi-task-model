import fitz
import os

pdf_path = r"cattle_thesis_p3_latex\main.pdf"
out_dir = r"scripts\fig_checks"
os.makedirs(out_dir, exist_ok=True)

# 0-indexed page numbers for pages 40, 42, 48, 49, 55, 56, 59
# PDF pages (1-indexed): 40, 42, 48, 49, 55, 56, 59
pages_to_render = {
    40: "fig_4_1_research_design_p40.png",
    42: "fig_4_2_input_pipelines_p42.png",
    48: "fig_4_3_mtl_arch_p48.png",
    49: "fig_4_3_mtl_arch_p49.png",
    55: "fig_4_4_superstep_routing_p55.png",
    56: "fig_4_4_superstep_routing_p56.png",
    59: "fig_5_1_behavior_recall_p59.png"
}

doc = fitz.open(pdf_path)
print(f"Total pages in PDF: {len(doc)}")

zoom = 2.0  # ~144-150 DPI
mat = fitz.Matrix(zoom, zoom)

for p_num, out_name in pages_to_render.items():
    page_idx = p_num - 1
    if page_idx < len(doc):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(out_dir, out_name)
        pix.save(out_path)
        print(f"Rendered page {p_num} -> {out_path} ({pix.width}x{pix.height})")
    else:
        print(f"Page {p_num} out of range!")
doc.close()
