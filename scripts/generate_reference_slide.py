import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_reference_slide():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.500)
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Design System Constants
    FONT_MAIN = "Aptos"
    
    C_LIGHT_BG = RGBColor(247, 248, 250)    # #F7F8FA
    C_WHITE_CARD = RGBColor(255, 255, 255)  # #FFFFFF
    C_CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0
    
    C_TEXT_DARK = RGBColor(16, 41, 58)      # #10293A
    C_TEXT_BODY = RGBColor(71, 85, 105)     # #475569
    C_TEXT_MUTED = RGBColor(138, 155, 168)  # #8A9BA8
    C_TEXT_WHITE = RGBColor(255, 255, 255)
    
    C_TEAL = RGBColor(14, 150, 151)         # #0E9697
    C_TEAL_LIGHT = RGBColor(91, 194, 190)   # #5BC2BE
    C_MINT_PILL = RGBColor(230, 244, 241)   # #E6F4F1
    C_AMBER = RGBColor(212, 148, 33)        # #D49421
    C_PURPLE = RGBColor(107, 93, 189)       # #6B5DBD

    # 1. Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_LIGHT_BG
    bg.line.fill.background()

    # 2. Header
    # Eyebrow
    tb_eye = slide.shapes.add_textbox(Inches(0.58), Inches(0.40), Inches(8.0), Inches(0.28))
    tf_e = tb_eye.text_frame
    tf_e.word_wrap = True
    tf_e.margin_left = tf_e.margin_top = tf_e.margin_right = tf_e.margin_bottom = 0
    p_e = tf_e.paragraphs[0]
    p_e.text = "07 / REFERENCES"
    p_e.font.name = FONT_MAIN
    p_e.font.size = Pt(11)
    p_e.font.bold = True
    p_e.font.color.rgb = C_TEAL

    # Title
    tb_t = slide.shapes.add_textbox(Inches(0.58), Inches(0.72), Inches(12.0), Inches(0.50))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = "Key Literature and Methodological References"
    p_t.font.name = FONT_MAIN
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = C_TEXT_DARK

    # Subtitle / context note
    tb_sub = slide.shapes.add_textbox(Inches(0.58), Inches(1.22), Inches(12.0), Inches(0.30))
    tf_sub = tb_sub.text_frame
    tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "Foundational studies guiding cattle perception, multi-task optimization, and evaluation design"
    p_sub.font.name = FONT_MAIN
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = C_TEXT_BODY

    # 3. Reference Categories (4 Cards Grid: 2 columns x 2 rows)
    cards_data = [
        {
            "category": "CATTLE ANATOMY & BODY CONDITION",
            "pill_bg": C_MINT_PILL,
            "pill_fg": C_TEAL,
            "refs": [
                ("[1] Edmonson et al. (1989)", "A Body Condition Scoring Chart for Holstein Dairy Cows.", "J. Dairy Sci., 72(1), 68-78. Formulated the definitive 5-point pelvic anatomy assessment."),
                ("[2] Ferguson et al. (1994)", "Principal Descriptors of Body Condition Score in Holstein Cows.", "J. Dairy Sci., 77(9), 2695-2703. Established hook and pin bone validation criteria.")
            ]
        },
        {
            "category": "MULTI-TASK LEARNING & OPTIMIZATION",
            "pill_bg": RGBColor(254, 243, 199), # Light Amber
            "pill_fg": C_AMBER,
            "refs": [
                ("[3] Yu et al. (2020)", "Gradient Surgery for Multi-Task Learning (PCGrad).", "NeurIPS 2020. Introduced conflicting gradient projection to mitigate negative transfer."),
                ("[4] Kendall et al. (2018)", "Multi-Task Learning Using Uncertainty to Weigh Losses.", "CVPR 2018. Principled homoscedastic uncertainty weighting across heterogeneous tasks.")
            ]
        },
        {
            "category": "CATTLE PERCEPTION & SEGMENTATION",
            "pill_bg": RGBColor(243, 232, 255), # Light Purple
            "pill_fg": C_PURPLE,
            "refs": [
                ("[5] Ravi et al. (2024)", "SAM 2: Segment Anything in Images and Videos.", "arXiv:2408.00714. Foundation model providing zero-shot cattle foreground segmentation masks."),
                ("[6] Zhao et al. (2024)", "DETRs Beat YOLOs on Real-Time Object Detection (RT-DETR).", "CVPR 2024. Efficient real-time transformer for robust bounding-box cow localization.")
            ]
        },
        {
            "category": "LIVESTOCK BEHAVIOR, RE-ID & ROBUSTNESS",
            "pill_bg": C_MINT_PILL,
            "pill_fg": C_TEAL,
            "refs": [
                ("[7] Zia et al. (2023)", "CVB: A Dataset for Cattle Video Behavior Recognition.", "Comput. Electron. Agric., 211. Established surveillance CCTV cattle activity benchmarks."),
                ("[8] Geirhos et al. (2020)", "Shortcut Learning in Deep Neural Networks.", "Nature Machine Intelligence, 2(11). Defined background shortcut exploitation risks in ML.")
            ]
        }
    ]

    # Grid coordinates
    card_w = Inches(5.95)
    card_h = Inches(2.45)
    xs = [Inches(0.58), Inches(6.80)]
    ys = [Inches(1.68), Inches(4.35)]

    for idx, cdata in enumerate(cards_data):
        col = idx % 2
        row = idx // 2
        cx = xs[col]
        cy = ys[row]

        # Card container
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = C_WHITE_CARD
        card.line.color.rgb = C_CARD_BORDER
        card.line.width = Pt(1)

        # Category Pill Badge
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx + Inches(0.20), cy + Inches(0.18), Inches(3.4), Inches(0.28))
        pill.fill.solid()
        pill.fill.fore_color.rgb = cdata["pill_bg"]
        pill.line.fill.background()
        tf_p = pill.text_frame
        tf_p.margin_left = tf_p.margin_top = tf_p.margin_right = tf_p.margin_bottom = 0
        p = tf_p.paragraphs[0]
        p.text = cdata["category"]
        p.font.name = FONT_MAIN
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = cdata["pill_fg"]

        # References text box inside card
        tb_r = slide.shapes.add_textbox(cx + Inches(0.20), cy + Inches(0.55), card_w - Inches(0.40), card_h - Inches(0.65))
        tf_r = tb_r.text_frame
        tf_r.word_wrap = True
        tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

        for r_idx, (r_author, r_title, r_source) in enumerate(cdata["refs"]):
            # Author + Year line
            p_auth = tf_r.paragraphs[0] if r_idx == 0 else tf_r.add_paragraph()
            p_auth.text = r_author + "  "
            p_auth.font.name = FONT_MAIN
            p_auth.font.size = Pt(11)
            p_auth.font.bold = True
            p_auth.font.color.rgb = C_TEXT_DARK
            if r_idx > 0:
                p_auth.space_before = Pt(8)

            # Title
            run_title = p_auth.add_run()
            run_title.text = f'"{r_title}"'
            run_title.font.name = FONT_MAIN
            run_title.font.size = Pt(10.5)
            run_title.font.bold = False
            run_title.font.color.rgb = C_TEXT_BODY

            # Source / annotation
            p_src = tf_r.add_paragraph()
            p_src.text = r_source
            p_src.font.name = FONT_MAIN
            p_src.font.size = Pt(9.5)
            p_src.font.italic = True
            p_src.font.color.rgb = C_TEXT_MUTED
            p_src.space_after = Pt(2)

    # 4. Footer
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(7.05), Inches(12.17), Pt(1))
    line.fill.solid()
    line.fill.fore_color.rgb = C_CARD_BORDER
    line.line.fill.background()

    tb_l = slide.shapes.add_textbox(Inches(0.58), Inches(7.12), Inches(10.0), Inches(0.25))
    tf_l = tb_l.text_frame
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0
    p_l = tf_l.paragraphs[0]
    p_l.text = "Submitted Report: Chapter 2 Literature Review and Bibliography | 8 Key Citations"
    p_l.font.name = FONT_MAIN
    p_l.font.size = Pt(8.5)
    p_l.font.color.rgb = C_TEXT_MUTED

    tb_r = slide.shapes.add_textbox(Inches(11.0), Inches(7.12), Inches(1.75), Inches(0.25))
    tf_r = tb_r.text_frame
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0
    p_r = tf_r.paragraphs[0]
    p_r.text = "REFERENCES"
    p_r.font.name = FONT_MAIN
    p_r.font.size = Pt(8.5)
    p_r.font.bold = True
    p_r.font.color.rgb = C_TEXT_MUTED

    output_path = "Reference_Slide.pptx"
    prs.save(output_path)
    print(f"Reference slide created successfully: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_reference_slide()
