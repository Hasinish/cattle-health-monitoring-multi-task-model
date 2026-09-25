"""
Export Freeze-Safe Thesis Sections to Clean .docx Documents for Paraphrasing.

This script parses authentic LaTeX source files from `cattle_thesis_p3_latex/`
and generates clean, human-readable Microsoft Word (.docx) documents for each of the
5 dedicated freeze-safe tabs identified in `P3_SAMPLE_WRITING_AUDIT.md`.

It strips LaTeX markup, converts citations to preserved tags `[cite: key]`,
and builds a side-by-side or structured block template for teammates to paraphrase.
"""

import os
import re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
LATEX_ROOT = WORKSPACE_ROOT / "cattle_thesis_p3_latex"
OUTPUT_DIR = LATEX_ROOT / "paraphrasing_docs"


def clean_latex(text: str) -> str:
    """Clean common LaTeX commands while preserving content and citations."""
    # Remove comments
    lines = []
    for line in text.splitlines():
        line_clean = re.sub(r'(?<!\\)%.*$', '', line)
        lines.append(line_clean)
    text = "\n".join(lines)

    # Strip citations completely (zero citations in doc per user request)
    text = re.sub(r'~?\\cite[tp]?\{[^}]+\}', '', text)
    text = re.sub(r'\[cite:\s*[^\]]+\]', '', text)
    text = re.sub(r'~?\\ref\{([^}]+)\}', r'\1', text)
    text = re.sub(r'~?\\eqref\{([^}]+)\}', r'(\1)', text)
    text = re.sub(r'~', ' ', text)

    # Environments handling
    text = re.sub(r'\\begin\{quote\}\s*(.*?)\s*\\end\{quote\}', r'"\1"', text, flags=re.DOTALL)

    def replace_enum(match):
        block = match.group(1).strip()
        items = [it.strip() for it in re.split(r'\\item\s+', block) if it.strip()]
        return "\n".join([f"{idx+1}. {it}" for idx, it in enumerate(items)])
    text = re.sub(r'\\begin\{enumerate\}(?:\[.*?\])?\s*(.*?)\s*\\end\{enumerate\}', replace_enum, text, flags=re.DOTALL)

    def replace_item(match):
        block = match.group(1).strip()
        items = [it.strip() for it in re.split(r'\\item\s+', block) if it.strip()]
        return "\n".join([f"• {it}" for it in items])
    text = re.sub(r'\\begin\{itemize\}(?:\[.*?\])?\s*(.*?)\s*\\end\{itemize\}', replace_item, text, flags=re.DOTALL)
    text = re.sub(r'\\item\s*', '• ', text)

    # Tables handling (strip raw LaTeX tabular environments from narrative text)
    def replace_table(match):
        cap_m = re.search(r'\\caption(?:\[[^\]]*\])?\{([^}]+)\}', match.group(0))
        cap = cap_m.group(1) if cap_m else "Summary Table"
        return f"\n[Table Reference: {cap} — (Omitted from paraphrasing prose)]\n"
    text = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', replace_table, text, flags=re.DOTALL)
    text = re.sub(r'\\label\{[^}]+\}', '', text)

    # Formatting macros (handle multiple/nested passes)
    for _ in range(3):
        text = re.sub(r'\\(textbf|textit|emph|texttt|textsc|underline|textsf)\{([^}]+)\}', r'\2', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'\1', text)

    # Math symbols & expressions
    text = re.sub(r'\\mathcal\{([a-zA-Z])\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\pm(?![a-zA-Z])', '±', text)
    text = re.sub(r'\\times(?![a-zA-Z])', '×', text)
    text = re.sub(r'\\(le|leq)(?![a-zA-Z])', '≤', text)
    text = re.sub(r'\\(ge|geq)(?![a-zA-Z])', '≥', text)
    text = re.sub(r'\\approx(?![a-zA-Z])', '≈', text)
    text = re.sub(r'\\(to|rightarrow)(?![a-zA-Z])', '→', text)
    text = re.sub(r'\\lambda(?![a-zA-Z])', 'λ', text)
    text = re.sub(r'\\sum(?![a-zA-Z])', 'Σ', text)
    text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'\n[Formula: \1]\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', r'\n[Formula: \1]\n', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text)

    # Escaped LaTeX characters
    text = re.sub(r'\\([%&_#$])', r'\1', text)

    # LaTeX dashes and quotes
    text = re.sub(r'---', ' — ', text)
    text = re.sub(r'--', '-', text)
    text = re.sub(r'``|\'\'', '"', text)
    text = re.sub(r'`', "'", text)

    # Remove generic LaTeX macros
    text = re.sub(r'\\(noindent|clearpage|cleardoublepage)\s*', '', text)
    text = re.sub(r'\\(vspace|hspace)\{[^}]+\}', '', text)
    text = re.sub(r'\\drafttodo\{[^}]+\}\{[^}]+\}', '', text)
    text = re.sub(r'\\input\{[^}]+\}', '', text)

    # Clean residual curly braces if any left
    text = re.sub(r'\{([^{}]+)\}', r'\1', text)

    # Clean punctuation spacing after stripped citations (e.g. "management ." -> "management.")
    text = re.sub(r'\s+([,\.\?!;:])', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Clean multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_latex_sections(content: str):
    """Parse text into a list of tuples: (level, title, body_text)."""
    # Regex to find \section, \subsection, \subsubsection
    pattern = re.compile(r'(\\section\{([^}]+)\}|\\subsection\{([^}]+)\}|\\subsubsection\{([^}]+)\})')
    
    parts = []
    last_pos = 0
    current_level = 0
    current_title = "Preamble"

    matches = list(pattern.finditer(content))
    if not matches:
        return [(0, "Main Content", clean_latex(content))]

    first_match = matches[0]
    if first_match.start() > 0:
        preamble = content[:first_match.start()].strip()
        if preamble:
            parts.append((0, "Overview", clean_latex(preamble)))

    for i, match in enumerate(matches):
        full_match = match.group(0)
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start_idx:end_idx].strip()

        if full_match.startswith(r'\section'):
            lvl = 1
            title = match.group(2)
        elif full_match.startswith(r'\subsection'):
            lvl = 2
            title = match.group(3)
        else:
            lvl = 3
            title = match.group(4)

        parts.append((lvl, title, clean_latex(body)))

    return parts


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def add_header_box(doc, tab_name: str, desc: str, rules: list):
    """Add a professional styled intro box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")

    p = cell.paragraphs[0]
    run_title = p.add_run(f"🔒 FREEZE-SAFE PARAPHRASING TAB: {tab_name.upper()}\n")
    run_title.bold = True
    run_title.font.size = Pt(13)
    run_title.font.color.rgb = RGBColor(16, 44, 87)

    run_desc = p.add_run(f"Scope: {desc}\n\n")
    run_desc.font.size = Pt(10)
    run_desc.italic = True

    run_rules_hdr = p.add_run("TEAM WRITING RULES (STRICT):\n")
    run_rules_hdr.bold = True
    run_rules_hdr.font.size = Pt(10)
    run_rules_hdr.font.color.rgb = RGBColor(192, 0, 0)

    for rule in rules:
        p_rule = cell.add_paragraph(f"• {rule}")
        p_rule.paragraph_format.space_after = Pt(2)
        p_rule.paragraph_format.left_indent = Inches(0.2)
        for r in p_rule.runs:
            r.font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)


def create_paraphrase_doc(tab_title: str, description: str, sections: list, output_path: Path):
    """Create a formatted Word document with a two-column or structured block layout."""
    doc = Document()

    # Set page margins to 0.7 inches
    for s in doc.sections:
        s.top_margin = Inches(0.7)
        s.bottom_margin = Inches(0.7)
        s.left_margin = Inches(0.7)
        s.right_margin = Inches(0.7)

    rules = [
        "DO NOT delete or change '[cite: key]' tags. Move them naturally with the sentence, but keep the exact key intact.",
        "Paraphrase in academic, authoritative third-person tone. Never use 'Phase 1', 'Phase 2', or 'P2' (use 'preliminary investigations').",
        "Never invent or alter numbers, percentages, cow counts, or dates.",
        "Type your humanized / paraphrased text directly into the right-hand column or the designated block."
    ]

    add_header_box(doc, tab_title, description, rules)

    for lvl, title, body in sections:
        # Add heading
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True

        run_h = h.add_run(title)
        run_h.bold = True
        if lvl == 1:
            run_h.font.size = Pt(15)
            run_h.font.color.rgb = RGBColor(16, 44, 87)
        elif lvl == 2:
            run_h.font.size = Pt(13)
            run_h.font.color.rgb = RGBColor(53, 89, 143)
        else:
            run_h.font.size = Pt(11.5)
            run_h.font.color.rgb = RGBColor(80, 80, 80)

        # Break body into paragraphs
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paras:
            continue

        # Add structured 2-column table for paragraphs
        table = doc.add_table(rows=1, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False

        # Set column widths
        widths = [Inches(3.4), Inches(3.6)]
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = w

        # Header row
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "ORIGINAL ACADEMIC DRAFT (READ-ONLY)"
        hdr_cells[1].text = "PARAPHRASED / HUMANIZED TEXT (TEAM INPUT)"
        set_cell_background(hdr_cells[0], "E8ECEF")
        set_cell_background(hdr_cells[1], "D1E7DD")

        for c in hdr_cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(4)
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9.5)

        for p_text in paras:
            row = table.add_row()
            c_orig = row.cells[0]
            c_para = row.cells[1]

            c_orig.width = Inches(3.4)
            c_para.width = Inches(3.6)

            set_cell_background(c_orig, "FAFAFA")
            set_cell_background(c_para, "FFFFFF")

            p_o = c_orig.paragraphs[0]
            p_o.text = p_text
            p_o.paragraph_format.space_after = Pt(4)
            for r in p_o.runs:
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(60, 60, 60)

            p_p = c_para.paragraphs[0]
            p_p.text = "[Type paraphrased version here...]"
            p_p.paragraph_format.space_after = Pt(4)
            for r in p_p.runs:
                r.font.size = Pt(9.5)
                r.italic = True
                r.font.color.rgb = RGBColor(150, 150, 150)

        doc.add_paragraph().paragraph_format.space_after = Pt(8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"[OK] Generated: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== EXPORTING FREEZE-SAFE THESIS SECTIONS TO DOCX ===")

    # --- TAB 1: Introduction and Problem Formulation ---
    ch1_path = LATEX_ROOT / "chapters" / "chapter_1.tex"
    if ch1_path.exists():
        content = ch1_path.read_text(encoding="utf-8")
        sections = parse_latex_sections(content)
        create_paraphrase_doc(
            "Tab 1: Introduction and Problem Formulation",
            "Covers Background, Welfare Motivation, Problem Statement, Objectives (RQ1–RQ3), and Scope. 100% Freeze-Safe.",
            sections,
            OUTPUT_DIR / "Tab1_Introduction_and_Problem_Formulation.docx"
        )

    # --- TAB 2: Literature Review ---
    ch2_path = LATEX_ROOT / "chapters" / "chapter_2.tex"
    if ch2_path.exists():
        content = ch2_path.read_text(encoding="utf-8")
        sections = parse_latex_sections(content)
        create_paraphrase_doc(
            "Tab 2: Literature Review (Domain and Methods)",
            "Comprehensive review of 80+ papers across BCS, Behavior, Re-ID, and Multi-Task Transfer theory. 100% Freeze-Safe.",
            sections,
            OUTPUT_DIR / "Tab2_Literature_Review_Domain_and_Methods.docx"
        )

    # --- TAB 3: Requirements, Impacts and Constraints ---
    ch3_path = LATEX_ROOT / "chapters" / "chapter_3.tex"
    if ch3_path.exists():
        content = ch3_path.read_text(encoding="utf-8")
        sections = parse_latex_sections(content)
        create_paraphrase_doc(
            "Tab 3: Requirements, Impacts and Constraints",
            "ABET/BAETE accreditation criteria, ethics disclosures, environmental sustainability, and economic sensor analysis. 100% Freeze-Safe.",
            sections,
            OUTPUT_DIR / "Tab3_Requirements_Impacts_and_Constraints.docx"
        )

    # --- TAB 4 & 5: Chapter 5 Splits ---
    ch5_path = LATEX_ROOT / "chapters" / "chapter_5.tex"
    if ch5_path.exists():
        content = ch5_path.read_text(encoding="utf-8")
        all_ch5_sections = parse_latex_sections(content)

        # Tab 5: Single-Task Baselines and Perception Architectures (Section 4.1 & 4.2)
        # Tab 4: Datasets, Preprocessing, and Split Integrity (Section 4.3 & 4.3.3)
        tab5_sections = []
        tab4_sections = []

        for lvl, title, body in all_ch5_sections:
            t_lower = title.lower()
            if "dataset" in t_lower or "data collection" in t_lower or "split" in t_lower or "feasibility" in t_lower or "evaluation protocol" in t_lower:
                tab4_sections.append((lvl, title, body))
            elif "multi-task" in t_lower or "reporting boundary" in t_lower:
                # STRICT RED LOCK: Exclude section 4.4 from paraphrasing doc!
                continue
            else:
                tab5_sections.append((lvl, title, body))

        create_paraphrase_doc(
            "Tab 4: Datasets, Preprocessing, and Split Integrity",
            "ScienceDB (7,549 test), CVB+Beef (780 test), SideViewCows2026 Protocol A (69 held-out cows), leak-free splitting, and feasibility audits. 100% Freeze-Safe.",
            tab4_sections,
            OUTPUT_DIR / "Tab4_Datasets_Preprocessing_and_Split_Integrity.docx"
        )

        create_paraphrase_doc(
            "Tab 5: Single-Task Baseline and Perception Architectures",
            "Pipeline overview, Ordinal BCE mathematical formulations, TCN temporal structures, and Re-ID Cosine embedding head. 100% Freeze-Safe.",
            tab5_sections,
            OUTPUT_DIR / "Tab5_Single_Task_Baseline_and_Perception_Architectures.docx"
        )

    # --- MASTER DOCUMENT WITH ALL TABS COMBINED ---
    print("\n[*] Assembling Master Consolidated Paraphrasing Document...")
    master_doc = Document()
    for s in master_doc.sections:
        s.top_margin = Inches(0.7)
        s.bottom_margin = Inches(0.7)
        s.left_margin = Inches(0.7)
        s.right_margin = Inches(0.7)

    master_rules = [
        "DO NOT delete or change '[cite: key]' tags. Keep the citation keys exact.",
        "Paraphrase in academic, authoritative third-person tone. Never use 'Phase 1', 'Phase 2', or 'P2'.",
        "Never invent or alter numbers, percentages, cow counts, or dates.",
        "Type your humanized / paraphrased text directly into the right-hand column."
    ]

    tabs_data = [
        ("Tab 1: Introduction and Problem Formulation", "Covers Background, Welfare Motivation, Problem Statement, Objectives (RQ1–RQ3), and Scope. 100% Freeze-Safe.", parse_latex_sections(ch1_path.read_text(encoding="utf-8")) if ch1_path.exists() else []),
        ("Tab 2: Literature Review (Domain and Methods)", "Comprehensive review of 80+ papers across BCS, Behavior, Re-ID, and Multi-Task Transfer theory. 100% Freeze-Safe.", parse_latex_sections(ch2_path.read_text(encoding="utf-8")) if ch2_path.exists() else []),
        ("Tab 3: Requirements, Impacts and Constraints", "ABET/BAETE accreditation criteria, ethics disclosures, environmental sustainability, and economic sensor analysis. 100% Freeze-Safe.", parse_latex_sections(ch3_path.read_text(encoding="utf-8")) if ch3_path.exists() else []),
        ("Tab 4: Datasets, Preprocessing, and Split Integrity", "ScienceDB (7,549 test), CVB+Beef (780 test), SideViewCows2026 Protocol A (69 held-out cows), leak-free splitting, and feasibility audits. 100% Freeze-Safe.", tab4_sections),
        ("Tab 5: Single-Task Baseline and Perception Architectures", "Pipeline overview, Ordinal BCE mathematical formulations, TCN temporal structures, and Re-ID Cosine embedding head. 100% Freeze-Safe.", tab5_sections),
    ]

    for t_idx, (t_title, t_desc, t_sects) in enumerate(tabs_data):
        if t_idx > 0:
            master_doc.add_page_break()
        add_header_box(master_doc, t_title, t_desc, master_rules)

        for lvl, title, body in t_sects:
            h = master_doc.add_paragraph()
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True

            run_h = h.add_run(title)
            run_h.bold = True
            if lvl == 1:
                run_h.font.size = Pt(15)
                run_h.font.color.rgb = RGBColor(16, 44, 87)
            elif lvl == 2:
                run_h.font.size = Pt(13)
                run_h.font.color.rgb = RGBColor(53, 89, 143)
            else:
                run_h.font.size = Pt(11.5)
                run_h.font.color.rgb = RGBColor(80, 80, 80)

            paras = [p.strip() for p in body.split("\n\n") if p.strip()]
            if not paras:
                continue

            table = master_doc.add_table(rows=1, cols=2)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False

            widths = [Inches(3.4), Inches(3.6)]
            for row in table.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = w

            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = "ORIGINAL ACADEMIC DRAFT (READ-ONLY)"
            hdr_cells[1].text = "PARAPHRASED / HUMANIZED TEXT (TEAM INPUT)"
            set_cell_background(hdr_cells[0], "E8ECEF")
            set_cell_background(hdr_cells[1], "D1E7DD")

            for c in hdr_cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(4)
                    for r in p.runs:
                        r.bold = True
                        r.font.size = Pt(9.5)

            for p_text in paras:
                row = table.add_row()
                c_orig = row.cells[0]
                c_para = row.cells[1]

                c_orig.width = Inches(3.4)
                c_para.width = Inches(3.6)

                set_cell_background(c_orig, "FAFAFA")
                set_cell_background(c_para, "FFFFFF")

                p_o = c_orig.paragraphs[0]
                p_o.text = p_text
                p_o.paragraph_format.space_after = Pt(4)
                for r in p_o.runs:
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = RGBColor(60, 60, 60)

                p_p = c_para.paragraphs[0]
                p_p.text = "[Type paraphrased version here...]"
                p_p.paragraph_format.space_after = Pt(4)
                for r in p_p.runs:
                    r.font.size = Pt(9.5)
                    r.italic = True
                    r.font.color.rgb = RGBColor(150, 150, 150)

            master_doc.add_paragraph().paragraph_format.space_after = Pt(8)

    master_path = OUTPUT_DIR / "Master_Freeze_Safe_Paraphrasing_All_Tabs.docx"
    master_doc.save(str(master_path))
    print(f"[OK] Generated Master Doc: {master_path}")

    print("\n[SUCCESS] All 5 Individual Tab Docs + 1 Master Consolidated Doc Generated in:")
    print(f" -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
