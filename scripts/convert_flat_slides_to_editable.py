"""
Convert Cattle_Thesis_Defense_Review_v4_with_cow_visuals.pptx flat image slides
into 100% native, editable PowerPoint elements while strictly preserving
the exact layout, visual identity, fonts, colors, icons, and cow visuals.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def convert_deck(pptx_path="Cattle_Thesis_Defense_Review_v4_with_cow_visuals.pptx"):
    prs = Presentation(pptx_path)

    # --- Design System Constants ---
    FONT_MAIN = "Aptos"
    
    # Colors
    C_DARK_BG = RGBColor(18, 40, 57)        # #122839
    C_DARK_CARD = RGBColor(28, 53, 73)      # #1C3549
    C_DARK_BORDER = RGBColor(39, 72, 97)    # #274861
    
    C_LIGHT_BG = RGBColor(247, 248, 250)    # #F7F8FA
    C_WHITE_CARD = RGBColor(255, 255, 255)  # #FFFFFF
    C_CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0
    
    C_TEXT_DARK = RGBColor(16, 41, 58)      # #10293A
    C_TEXT_BODY = RGBColor(71, 85, 105)     # #475569
    C_TEXT_MUTED = RGBColor(138, 155, 168)  # #8A9BA8
    C_TEXT_WHITE = RGBColor(255, 255, 255)
    
    C_TEAL = RGBColor(14, 150, 151)         # #0E9697
    C_TEAL_LIGHT = RGBColor(91, 194, 190)   # #5BC2BE
    C_TEAL_CYAN = RGBColor(102, 209, 205)   # #66D1CD
    C_MINT_PILL = RGBColor(230, 244, 241)   # #E6F4F1
    
    C_AMBER = RGBColor(212, 148, 33)        # #D49421
    C_PURPLE = RGBColor(107, 93, 189)       # #6B5DBD
    C_GRAY_BAR = RGBColor(142, 157, 169)    # #8E9DA9

    def clear_slide(slide):
        # Remove all shapes
        sp_ids = [sp.shape_id for sp in slide.shapes]
        for sid in sp_ids:
            for s in list(slide.shapes):
                if s.shape_id == sid:
                    sp_elem = s.element
                    sp_elem.getparent().remove(sp_elem)

    def set_bg(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, eyebrow, title, is_dark=False):
        # Eyebrow
        tb_eye = slide.shapes.add_textbox(Inches(0.58), Inches(0.40), Inches(8.0), Inches(0.28))
        tf_e = tb_eye.text_frame
        tf_e.word_wrap = True
        tf_e.margin_left = tf_e.margin_top = tf_e.margin_right = tf_e.margin_bottom = 0
        p_e = tf_e.paragraphs[0]
        p_e.text = eyebrow.upper()
        p_e.font.name = FONT_MAIN
        p_e.font.size = Pt(11)
        p_e.font.bold = True
        p_e.font.color.rgb = C_TEAL_CYAN if is_dark else C_TEAL

        # Title
        tb_t = slide.shapes.add_textbox(Inches(0.58), Inches(0.85), Inches(12.0), Inches(0.70))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_MAIN
        p_t.font.size = Pt(26)
        p_t.font.bold = True
        p_t.font.color.rgb = C_TEXT_WHITE if is_dark else C_TEXT_DARK

    def add_footer(slide, left_text, slide_str, is_dark=False):
        # Line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(7.05), Inches(12.17), Pt(1))
        line.fill.solid()
        line.fill.fore_color.rgb = C_DARK_BORDER if is_dark else C_CARD_BORDER
        line.line.fill.background()

        # Left Text
        tb_l = slide.shapes.add_textbox(Inches(0.58), Inches(7.12), Inches(10.0), Inches(0.25))
        tf_l = tb_l.text_frame
        tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0
        p_l = tf_l.paragraphs[0]
        p_l.text = left_text
        p_l.font.name = FONT_MAIN
        p_l.font.size = Pt(8.5)
        p_l.font.color.rgb = C_TEXT_MUTED

        # Right Text
        tb_r = slide.shapes.add_textbox(Inches(11.5), Inches(7.12), Inches(1.25), Inches(0.25))
        tf_r = tb_r.text_frame
        tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0
        p_r = tf_r.paragraphs[0]
        p_r.text = slide_str
        p_r.alignment = PP_ALIGN.RIGHT
        p_r.font.name = FONT_MAIN
        p_r.font.size = Pt(9.5)
        p_r.font.bold = True
        p_r.font.color.rgb = C_TEXT_MUTED

    def add_banner(slide, text, top=Inches(6.25), is_dark=False):
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.58), top, Inches(12.17), Inches(0.56))
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor(28, 48, 64) if is_dark else C_MINT_PILL
        pill.line.fill.background()

        tb = slide.shapes.add_textbox(Inches(0.8), top + Inches(0.12), Inches(11.7), Inches(0.35))
        tf = tb.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = text
        p.font.name = FONT_MAIN
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_TEAL_CYAN if is_dark else C_TEAL

    def add_card(slide, left, top, width, height, bg_color=C_WHITE_CARD, border_color=C_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # REBUILD SLIDES 1 TO 24
    # =========================================================================
    
    # --- SLIDE 1 ---
    s1 = prs.slides[1]
    clear_slide(s1)
    set_bg(s1, C_LIGHT_BG)
    add_header(s1, "01  /  MOTIVATION", "Cattle monitoring needs a connected view of the animal")
    
    col_w = Inches(3.82)
    gap = Inches(0.35)
    c1_top = Inches(1.85)
    c1_h = Inches(4.15)
    
    cards_data1 = [
        ("scratch/icons/icon_bcs_scale.png", "BODY CONDITION", C_TEAL, "Nutritional reserves", "Track visible condition-related morphology."),
        ("scratch/icons/icon_behavior_walk.png", "DAILY ACTIVITY", C_AMBER, "Posture and movement", "Recognize feeding, drinking, standing, lying and walking."),
        ("scratch/icons/icon_reid_fingerprint.png", "INDIVIDUAL IDENTITY", C_PURPLE, "The same cow over time", "Link observations to the animal they describe.")
    ]
    for idx, (icon_path, tag, tag_color, heading, body) in enumerate(cards_data1):
        cl = Inches(0.58) + idx * (col_w + gap)
        add_card(s1, cl, c1_top, col_w, c1_h)
        if os.path.exists(icon_path):
            s1.shapes.add_picture(icon_path, cl + Inches(0.35), c1_top + Inches(0.35), Inches(0.65), Inches(0.65))
        
        # Tag
        tb = s1.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(1.25), col_w - Inches(0.7), Inches(0.25))
        p = tb.text_frame.paragraphs[0]
        p.text = tag
        p.font.name = FONT_MAIN
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = tag_color
        
        # Heading
        tb2 = s1.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(1.75), col_w - Inches(0.7), Inches(0.75))
        tb2.text_frame.word_wrap = True
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = heading
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(21)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT_DARK
        
        # Body
        tb3 = s1.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(2.70), col_w - Inches(0.7), Inches(1.1))
        tb3.text_frame.word_wrap = True
        p3 = tb3.text_frame.paragraphs[0]
        p3.text = body
        p3.font.name = FONT_MAIN
        p3.font.size = Pt(13.5)
        p3.font.color.rgb = C_TEXT_BODY

    add_banner(s1, "Cameras offer a common, non-contact source for these complementary observations.")
    add_footer(s1, "Submitted report: Chapter 1, background and motivation", "02 / 25")

    # --- SLIDE 2 ---
    s2 = prs.slides[2]
    clear_slide(s2)
    set_bg(s2, C_LIGHT_BG)
    add_header(s2, "01  /  THE MONITORING TASKS", "Three questions. Three different prediction tasks.")

    # Card 1 (BCS)
    cl = Inches(0.58)
    add_card(s2, cl, c1_top, col_w, c1_h)
    tb = s2.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(0.35), col_w - Inches(0.7), Inches(0.25))
    p = tb.text_frame.paragraphs[0]
    p.text = "BCS  /  ORDERED PREDICTION"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    
    tb2 = s2.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(0.95), col_w - Inches(0.7), Inches(1.2))
    tb2.text_frame.word_wrap = True
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = "What is the cow's body condition?"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(23)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_DARK

    if os.path.exists("scratch/icons/palette_bcs.png"):
        s2.shapes.add_picture("scratch/icons/palette_bcs.png", cl + Inches(0.35), c1_top + Inches(2.7), col_w - Inches(0.7), Inches(0.65))
    tb_lbl = s2.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(3.45), col_w - Inches(0.7), Inches(0.3))
    p_lbl = tb_lbl.text_frame.paragraphs[0]
    p_lbl.text = "ScienceDB label range"
    p_lbl.font.name = FONT_MAIN
    p_lbl.font.size = Pt(10)
    p_lbl.font.color.rgb = C_TEXT_MUTED

    # Card 2 (Behavior)
    cl2 = cl + col_w + gap
    add_card(s2, cl2, c1_top, col_w, c1_h)
    tb_b = s2.shapes.add_textbox(cl2 + Inches(0.35), c1_top + Inches(0.35), col_w - Inches(0.7), Inches(0.25))
    p_b = tb_b.text_frame.paragraphs[0]
    p_b.text = "BEHAVIOR  /  CLASSIFICATION"
    p_b.font.name = FONT_MAIN
    p_b.font.size = Pt(10.5)
    p_b.font.bold = True
    p_b.font.color.rgb = C_AMBER

    tb2_b = s2.shapes.add_textbox(cl2 + Inches(0.35), c1_top + Inches(0.95), col_w - Inches(0.7), Inches(1.2))
    tb2_b.text_frame.word_wrap = True
    p2_b = tb2_b.text_frame.paragraphs[0]
    p2_b.text = "What is the cow doing?"
    p2_b.font.name = FONT_MAIN
    p2_b.font.size = Pt(23)
    p2_b.font.bold = True
    p2_b.font.color.rgb = C_TEXT_DARK

    tb3_b = s2.shapes.add_textbox(cl2 + Inches(0.35), c1_top + Inches(2.65), col_w - Inches(0.7), Inches(1.2))
    p3_b = tb3_b.text_frame.paragraphs[0]
    p3_b.text = "Standing  /  Lying\nFeeding  /  Drinking\nWalking"
    p3_b.font.name = FONT_MAIN
    p3_b.font.size = Pt(15.5)
    p3_b.font.bold = True
    p3_b.font.color.rgb = C_AMBER

    # Card 3 (Re-ID)
    cl3 = cl2 + col_w + gap
    add_card(s2, cl3, c1_top, col_w, c1_h)
    tb_r = s2.shapes.add_textbox(cl3 + Inches(0.35), c1_top + Inches(0.35), col_w - Inches(0.7), Inches(0.25))
    p_r = tb_r.text_frame.paragraphs[0]
    p_r.text = "RE-ID  /  RETRIEVAL"
    p_r.font.name = FONT_MAIN
    p_r.font.size = Pt(10.5)
    p_r.font.bold = True
    p_r.font.color.rgb = C_PURPLE

    tb2_r = s2.shapes.add_textbox(cl3 + Inches(0.35), c1_top + Inches(0.95), col_w - Inches(0.7), Inches(1.2))
    tb2_r.text_frame.word_wrap = True
    p2_r = tb2_r.text_frame.paragraphs[0]
    p2_r.text = "Which cow is it?"
    p2_r.font.name = FONT_MAIN
    p2_r.font.size = Pt(23)
    p2_r.font.bold = True
    p2_r.font.color.rgb = C_TEXT_DARK

    if os.path.exists("scratch/icons/reid_graphic.png"):
        s2.shapes.add_picture("scratch/icons/reid_graphic.png", cl3 + Inches(0.35), c1_top + Inches(2.6), col_w - Inches(0.7), Inches(0.85))
    tb_lbl_r = s2.shapes.add_textbox(cl3 + Inches(0.35), c1_top + Inches(3.55), col_w - Inches(0.7), Inches(0.3))
    p_lbl_r = tb_lbl_r.text_frame.paragraphs[0]
    p_lbl_r.text = "Query image  →  ranked gallery"
    p_lbl_r.font.name = FONT_MAIN
    p_lbl_r.font.size = Pt(10)
    p_lbl_r.font.color.rgb = C_TEXT_MUTED

    add_banner(s2, "Condition, activity and identity belong in one monitoring framework.")
    add_footer(s2, "Submitted report: Chapters 1 and 4; task definitions and target labels", "03 / 25")

    # --- SLIDE 3 ---
    s3 = prs.slides[3]
    clear_slide(s3)
    set_bg(s3, C_LIGHT_BG)
    add_header(s3, "01  /  RESEARCH PROBLEM", "Related tasks do not necessarily need identical features")

    table_shape3 = s3.shapes.add_table(4, 3, Inches(0.58), Inches(1.85), Inches(12.17), Inches(4.15))
    tbl3 = table_shape3.table
    tbl3.columns[0].width = Inches(2.2)
    tbl3.columns[1].width = Inches(5.5)
    tbl3.columns[2].width = Inches(4.47)

    headers3 = ["TASK", "INFORMATION TO PRESERVE", "POTENTIAL DISTRACTION"]
    for c_i, h_t in enumerate(headers3):
        c = tbl3.cell(0, c_i)
        c.text = h_t
        c.fill.solid()
        c.fill.fore_color.rgb = C_DARK_BG
        p = c.text_frame.paragraphs[0]
        p.font.name = FONT_MAIN
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE

    m_rows3 = [
        ("BCS", "Morphology and visible body condition", "Identity or background correlations"),
        ("Behavior", "Posture, motion and relevant context", "A location standing in for an activity"),
        ("Re-ID", "Coat pattern and individual appearance", "Camera or farm background")
    ]
    for r_i, (t_val, p_val, d_val) in enumerate(m_rows3):
        for c_i, val in enumerate([t_val, p_val, d_val]):
            cell = tbl3.cell(r_i + 1, c_i)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_WHITE_CARD
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_MAIN
            p.font.size = Pt(13)
            p.font.color.rgb = C_TEXT_DARK
            if c_i == 0:
                p.font.bold = True
                p.font.size = Pt(16)
                if r_i == 0: p.font.color.rgb = C_TEAL
                elif r_i == 1: p.font.color.rgb = C_AMBER
                elif r_i == 2: p.font.color.rgb = C_PURPLE

    add_banner(s3, "Research question: what should be shared, and what should remain task-specific?")
    add_footer(s3, "Submitted report: Chapter 1, task-appropriate visual representations", "04 / 25")

    # --- SLIDE 4 ---
    s4 = prs.slides[4]
    clear_slide(s4)
    set_bg(s4, C_LIGHT_BG)
    add_header(s4, "01  /  RESEARCH GAP AND THESIS FOCUS", "Our focus: jointly learning condition, activity and identity")

    card_left_w = Inches(5.8)
    card_right_w = Inches(6.0)
    add_card(s4, Inches(0.58), c1_top, card_left_w, c1_h)
    
    # Left Big Card
    tb4_l = s4.shapes.add_textbox(Inches(0.95), c1_top + Inches(0.6), card_left_w - Inches(0.7), Inches(0.3))
    p = tb4_l.text_frame.paragraphs[0]
    p.text = "BCS + Behavior + Re-ID"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    tb4_l2 = s4.shapes.add_textbox(Inches(0.95), c1_top + Inches(1.3), card_left_w - Inches(0.7), Inches(1.4))
    tb4_l2.text_frame.word_wrap = True
    p2 = tb4_l2.text_frame.paragraphs[0]
    p2.text = "One evaluated\nmulti-task framework"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_DARK

    tb4_l3 = s4.shapes.add_textbox(Inches(0.95), c1_top + Inches(2.9), card_left_w - Inches(0.7), Inches(0.8))
    tb4_l3.text_frame.word_wrap = True
    p3 = tb4_l3.text_frame.paragraphs[0]
    p3.text = "Not simply three unrelated prediction models."
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(14)
    p3.font.color.rgb = C_TEXT_BODY

    # Right 2 Stacked Cards
    r_left = Inches(0.58) + card_left_w + gap
    add_card(s4, r_left, c1_top, card_right_w, Inches(1.95))
    tb_rq = s4.shapes.add_textbox(r_left + Inches(0.35), c1_top + Inches(0.25), card_right_w - Inches(0.7), Inches(0.25))
    p = tb_rq.text_frame.paragraphs[0]
    p.text = "Representation question"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    tb_rq2 = s4.shapes.add_textbox(r_left + Inches(0.35), c1_top + Inches(0.75), card_right_w - Inches(0.7), Inches(1.0))
    tb_rq2.text_frame.word_wrap = True
    p = tb_rq2.text_frame.paragraphs[0]
    p.text = "Which cattle-centered inputs are useful for each task?"
    p.font.name = FONT_MAIN
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    add_card(s4, r_left, c1_top + Inches(2.2), card_right_w, Inches(1.95))
    tb_sq = s4.shapes.add_textbox(r_left + Inches(0.35), c1_top + Inches(2.45), card_right_w - Inches(0.7), Inches(0.25))
    p = tb_sq.text_frame.paragraphs[0]
    p.text = "Sharing question"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    tb_sq2 = s4.shapes.add_textbox(r_left + Inches(0.35), c1_top + Inches(2.95), card_right_w - Inches(0.7), Inches(1.0))
    tb_sq2.text_frame.word_wrap = True
    p = tb_sq2.text_frame.paragraphs[0]
    p.text = "How do joint training and task-specific pathways affect performance?"
    p.font.name = FONT_MAIN
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    add_banner(s4, "The contribution is the integrated investigation, not a claim of universal superiority.")
    add_footer(s4, "Submitted report: Chapters 1 and 2; research problem and synthesis", "05 / 25")

    # --- SLIDE 5 ---
    s5 = prs.slides[5]
    clear_slide(s5)
    set_bg(s5, C_LIGHT_BG)
    add_header(s5, "02  /  RESEARCH QUESTIONS", "Three research questions guide the experiments")

    rq_cards = [
        ("RQ1", "REPRESENTATION", "How do cattle-centered inputs compare with RGB single-task references?"),
        ("RQ2", "TASK-SPECIFIC INFORMATION", "How should visual and temporal cues be incorporated for each task?"),
        ("RQ3", "MULTI-TASK SHARING", "How does task-conditioned sharing compare with hard sharing?")
    ]
    rq_card_h = Inches(1.35)
    rq_gap = Inches(0.25)
    rq_top_base = Inches(1.95)

    for i, (rq_num, rq_tag, rq_txt) in enumerate(rq_cards):
        ct = rq_top_base + i * (rq_card_h + rq_gap)
        add_card(s5, Inches(0.58), ct, Inches(12.17), rq_card_h)
        
        # Badge
        badge = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), ct + Inches(0.3), Inches(0.85), Inches(0.75))
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_DARK_BG
        badge.line.fill.background()
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.text = rq_num
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE

        # Tag
        tb = s5.shapes.add_textbox(Inches(2.0), ct + Inches(0.25), Inches(10.0), Inches(0.25))
        p = tb.text_frame.paragraphs[0]
        p.text = rq_tag
        p.font.name = FONT_MAIN
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = C_TEAL

        # Text
        tb2 = s5.shapes.add_textbox(Inches(2.0), ct + Inches(0.55), Inches(10.0), Inches(0.65))
        tb2.text_frame.word_wrap = True
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = rq_txt
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(15.5)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT_DARK

    add_footer(s5, "Submitted report: Chapter 1, Research Questions 1-3", "06 / 25")

    # --- SLIDE 6 ---
    s6 = prs.slides[6]
    clear_slide(s6)
    set_bg(s6, C_LIGHT_BG)
    add_header(s6, "02  /  OVERALL THESIS FRAMEWORK", "From task-specific evidence to unified multi-task learning")

    col4_w = Inches(2.75)
    gap4 = Inches(0.38)
    cols_data6 = [
        ("Protected data", ["ScienceDB", "CVB + Beef", "SideView"]),
        ("Compare inputs", ["RGB references", "vs.", "cattle-centered inputs"]),
        ("Single-task tests", ["BCS", "Behavior", "Re-ID"]),
        ("Joint MTL tests", ["E1 Hard sharing", "E3 Adapters", "E4 PCGrad"])
    ]
    for i, (title_6, items_6) in enumerate(cols_data6):
        cl = Inches(0.58) + i * (col4_w + gap4)
        add_card(s6, cl, Inches(1.85), col4_w, Inches(2.9))
        
        # Header
        tb = s6.shapes.add_textbox(cl + Inches(0.25), Inches(2.05), col4_w - Inches(0.5), Inches(0.3))
        p = tb.text_frame.paragraphs[0]
        p.text = title_6
        p.font.name = FONT_MAIN
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_TEAL
        
        # Items
        tb2 = s6.shapes.add_textbox(cl + Inches(0.25), Inches(2.6), col4_w - Inches(0.5), Inches(1.9))
        tb2.text_frame.word_wrap = True
        for j, it in enumerate(items_6):
            p2 = tb2.text_frame.paragraphs[0] if j == 0 else tb2.text_frame.add_paragraph()
            p2.text = it
            p2.font.name = FONT_MAIN
            p2.font.size = Pt(13)
            p2.font.bold = (it != "vs.")
            p2.font.color.rgb = C_TEXT_MUTED if it == "vs." else C_TEXT_DARK
            p2.space_after = Pt(4)

        if i < 3:
            # Horizontal connector arrow
            arr = s6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, cl + col4_w + Inches(0.08), Inches(3.1), Inches(0.22), Inches(0.18))
            arr.fill.solid()
            arr.fill.fore_color.rgb = C_TEAL
            arr.line.fill.background()

    # Sub-banner
    add_banner(s6, "SAME HELD-OUT POPULATION WITHIN EACH COMPARISON", top=Inches(5.05))

    # Note
    tb_n = s6.shapes.add_textbox(Inches(0.58), Inches(5.9), Inches(12.17), Inches(0.4))
    p = tb_n.text_frame.paragraphs[0]
    p.text = "Separate perception investigations: localization, masks, pose/anatomy and viewpoint."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT_BODY

    add_footer(s6, "Submitted report: Chapter 4, design process and methodology overview", "07 / 25")

    # --- SLIDE 7 ---
    s7 = prs.slides[7]
    clear_slide(s7)
    set_bg(s7, C_LIGHT_BG)
    add_header(s7, "02  /  DATASETS AND SPLITS", "Evaluation protects the strongest available grouping unit")

    ds_cards = [
        ("BCS", C_TEAL, "ScienceDB", "53,566 images", "5,653 repaired burst groups", "Passage / sequence-safe", "Biological cow IDs are not verified."),
        ("BEHAVIOR", C_AMBER, "CVB + Kaggle Beef", "5,274 samples", "267 source / session groups", "Source / session-disjoint", "Walking is contributed by CVB only."),
        ("RE-ID", C_PURPLE, "SideViewCows2026", "80,260 images", "110 biological identities", "41 learning / 69 evaluation cows", "Identity-disjoint Protocol A.")
    ]
    for i, (tag, tag_col, ds_title, stat1, stat2, highlight, footnote) in enumerate(ds_cards):
        cl = Inches(0.58) + i * (col_w + gap)
        add_card(s7, cl, c1_top, col_w, c1_h)
        
        # Tag
        tb = s7.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(0.35), col_w - Inches(0.7), Inches(0.25))
        p = tb.text_frame.paragraphs[0]
        p.text = tag
        p.font.name = FONT_MAIN
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = tag_col

        # Title
        tb2 = s7.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(0.85), col_w - Inches(0.7), Inches(0.6))
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = ds_title
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(21)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT_DARK

        # Stats
        tb3 = s7.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(1.65), col_w - Inches(0.7), Inches(0.9))
        tb3.text_frame.word_wrap = True
        p3_1 = tb3.text_frame.paragraphs[0]
        p3_1.text = stat1
        p3_1.font.name = FONT_MAIN
        p3_1.font.size = Pt(14)
        p3_1.font.bold = True
        p3_1.font.color.rgb = C_TEXT_DARK
        p3_1.space_after = Pt(4)

        p3_2 = tb3.text_frame.add_paragraph()
        p3_2.text = stat2
        p3_2.font.name = FONT_MAIN
        p3_2.font.size = Pt(14)
        p3_2.font.bold = True
        p3_2.font.color.rgb = C_TEXT_DARK

        # Highlight Box
        hl_box = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cl + Inches(0.35), c1_top + Inches(2.75), col_w - Inches(0.7), Inches(0.5))
        hl_box.fill.solid()
        hl_box.fill.fore_color.rgb = C_MINT_PILL
        hl_box.line.fill.background()
        tf_hl = hl_box.text_frame
        p_hl = tf_hl.paragraphs[0]
        p_hl.text = highlight
        p_hl.font.name = FONT_MAIN
        p_hl.font.size = Pt(11)
        p_hl.font.bold = True
        p_hl.font.color.rgb = C_TEAL

        # Footnote
        tb4 = s7.shapes.add_textbox(cl + Inches(0.35), c1_top + Inches(3.45), col_w - Inches(0.7), Inches(0.5))
        tb4.text_frame.word_wrap = True
        p4 = tb4.text_frame.paragraphs[0]
        p4.text = footnote
        p4.font.name = FONT_MAIN
        p4.font.size = Pt(10)
        p4.font.color.rgb = C_TEXT_MUTED

    add_banner(s7, "Different images are not automatically independent observations.")
    add_footer(s7, "Submitted report: Chapter 4, data preparation and split protection", "08 / 25")

    # --- SLIDE 8 ---
    s8 = prs.slides[8]
    clear_slide(s8)
    set_bg(s8, C_LIGHT_BG)
    add_header(s8, "03  /  REPRESENTATION DESIGN", "Cattle-centered cues were tested, not assumed useful")

    rows8 = [
        ("Localization / crop", "Identify and frame the target cow", "USED", RGBColor(220, 252, 231), RGBColor(22, 101, 52)),
        ("Binary foreground mask", "Provide explicit cow-pixel information", "USED", RGBColor(220, 252, 231), RGBColor(22, 101, 52)),
        ("Temporal frames", "Represent activity across eight time steps", "BEHAVIOR", RGBColor(254, 243, 199), RGBColor(146, 64, 14)),
        ("Viewpoint", "Classify front, side and rear views", "SEPARATE MODEL", RGBColor(243, 232, 255), RGBColor(107, 33, 168)),
        ("Pose / anatomy", "Test whether landmarks are usable", "NOT IN FINAL MTL", RGBColor(241, 245, 249), RGBColor(71, 85, 105))
    ]
    r_top = Inches(1.85)
    r_h = Inches(0.85)
    r_gap = Inches(0.18)

    for i, (cue, desc, status, bg_col, txt_col) in enumerate(rows8):
        y = r_top + i * (r_h + r_gap)
        add_card(s8, Inches(0.58), y, Inches(12.17), r_h)

        # Cue
        tb = s8.shapes.add_textbox(Inches(0.9), y + Inches(0.22), Inches(3.6), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        p.text = cue
        p.font.name = FONT_MAIN
        p.font.size = Pt(15.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK

        # Desc
        tb2 = s8.shapes.add_textbox(Inches(4.6), y + Inches(0.24), Inches(5.6), Inches(0.4))
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = desc
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(13)
        p2.font.color.rgb = C_TEXT_BODY

        # Status Badge
        badge = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.5), y + Inches(0.2), Inches(1.9), Inches(0.44))
        badge.fill.solid()
        badge.fill.fore_color.rgb = bg_col
        badge.line.fill.background()
        tf_b = badge.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = status
        p_b.alignment = PP_ALIGN.CENTER
        p_b.font.name = FONT_MAIN
        p_b.font.size = Pt(9.5)
        p_b.font.bold = True
        p_b.font.color.rgb = txt_col

    add_footer(s8, "Submitted report: Chapter 4, representation alternatives and component selection", "09 / 25")

    # --- SLIDE 9 ---
    s9 = prs.slides[9]
    clear_slide(s9)
    set_bg(s9, C_LIGHT_BG)
    add_header(s9, "03  /  CATTLE PERCEPTION PIPELINE", "The task inputs combine a cow crop and a binary mask")

    tb_sub9 = s9.shapes.add_textbox(Inches(0.58), Inches(1.8), Inches(4.0), Inches(0.25))
    p = tb_sub9.text_frame.paragraphs[0]
    p.text = "INPUT CONSTRUCTION"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    # Box 1: RGB Crop
    b1_l = Inches(0.58)
    b_top = Inches(2.2)
    b_w = Inches(2.5)
    b_h = Inches(2.4)
    add_card(s9, b1_l, b_top, b_w, b_h)
    if os.path.exists("scratch/icons/cow_rgb_crop.png"):
        s9.shapes.add_picture("scratch/icons/cow_rgb_crop.png", b1_l + Inches(0.2), b_top + Inches(0.2), b_w - Inches(0.4), Inches(1.15))
    tb_l1 = s9.shapes.add_textbox(b1_l + Inches(0.2), b_top + Inches(1.5), b_w - Inches(0.4), Inches(0.8))
    tb_l1.text_frame.word_wrap = True
    p = tb_l1.text_frame.paragraphs[0]
    p.text = "RGB cow crop"
    p.font.name = FONT_MAIN
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p2 = tb_l1.text_frame.add_paragraph()
    p2.text = "3 normalized channels"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_BODY

    # Plus sign
    tb_plus = s9.shapes.add_textbox(b1_l + b_w + Inches(0.15), b_top + Inches(0.9), Inches(0.3), Inches(0.4))
    p = tb_plus.text_frame.paragraphs[0]
    p.text = "+"
    p.font.name = FONT_MAIN
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_BODY

    # Box 2: Binary Mask
    b2_l = b1_l + b_w + Inches(0.6)
    add_card(s9, b2_l, b_top, b_w, b_h)
    if os.path.exists("scratch/icons/cow_binary_mask.png"):
        s9.shapes.add_picture("scratch/icons/cow_binary_mask.png", b2_l + Inches(0.2), b_top + Inches(0.2), b_w - Inches(0.4), Inches(1.15))
    tb_l2 = s9.shapes.add_textbox(b2_l + Inches(0.2), b_top + Inches(1.5), b_w - Inches(0.4), Inches(0.8))
    tb_l2.text_frame.word_wrap = True
    p = tb_l2.text_frame.paragraphs[0]
    p.text = "Binary cow mask"
    p.font.name = FONT_MAIN
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p2 = tb_l2.text_frame.add_paragraph()
    p2.text = "1 channel: 0 or 1"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_BODY

    # Arrow 1
    arr1 = s9.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, b2_l + b_w + Inches(0.15), b_top + Inches(1.0), Inches(0.35), Inches(0.18))
    arr1.fill.solid()
    arr1.fill.fore_color.rgb = C_TEAL
    arr1.line.fill.background()

    # Box 3: 4-Channel Input
    b3_l = b2_l + b_w + Inches(0.65)
    b3_w = Inches(2.4)
    add_card(s9, b3_l, b_top, b3_w, b_h + Inches(0.4))
    
    # Teal bar at top of box 3
    t_bar3 = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, b3_l, b_top, b3_w, Inches(0.08))
    t_bar3.fill.solid()
    t_bar3.fill.fore_color.rgb = C_TEAL
    t_bar3.line.fill.background()

    tb_l3 = s9.shapes.add_textbox(b3_l + Inches(0.2), b_top + Inches(0.2), b3_w - Inches(0.4), Inches(2.4))
    tb_l3.text_frame.word_wrap = True
    p = tb_l3.text_frame.paragraphs[0]
    p.text = "4-channel input"
    p.font.name = FONT_MAIN
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p.space_after = Pt(2)

    p2 = tb_l3.text_frame.add_paragraph()
    p2.text = "224 × 224 pixels"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_TEXT_BODY
    p2.space_after = Pt(14)

    p3 = tb_l3.text_frame.add_paragraph()
    p3.text = "RGB is not multiplied by the mask."
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(12)
    p3.font.color.rgb = C_TEXT_BODY

    # Arrow 2
    arr2 = s9.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, b3_l + b3_w + Inches(0.15), b_top + Inches(1.0), Inches(0.35), Inches(0.18))
    arr2.fill.solid()
    arr2.fill.fore_color.rgb = C_TEAL
    arr2.line.fill.background()

    # Box 4: ResNet-18
    b4_l = b3_l + b3_w + Inches(0.65)
    b4_w = Inches(2.7)
    add_card(s9, b4_l, b_top, b4_w, b_h + Inches(0.4))
    
    t_bar4 = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, b4_l, b_top, b4_w, Inches(0.08))
    t_bar4.fill.solid()
    t_bar4.fill.fore_color.rgb = C_TEAL
    t_bar4.line.fill.background()

    tb_l4 = s9.shapes.add_textbox(b4_l + Inches(0.2), b_top + Inches(0.2), b4_w - Inches(0.4), Inches(2.4))
    tb_l4.text_frame.word_wrap = True
    p = tb_l4.text_frame.paragraphs[0]
    p.text = "ResNet-18"
    p.font.name = FONT_MAIN
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p.space_after = Pt(8)

    p2 = tb_l4.text_frame.add_paragraph()
    p2.text = "ImageNet RGB filters\n+ initialized mask-channel filters"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_TEXT_BODY

    # Explanatory text
    tb_note9 = s9.shapes.add_textbox(Inches(0.58), Inches(5.35), Inches(12.17), Inches(0.4))
    p = tb_note9.text_frame.paragraphs[0]
    p.text = "BCS / Beef: detector prompts SAM 2.1.  CVB: target-tracklet box prompts SAM 2.1."
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.color.rgb = C_TEXT_BODY

    add_banner(s9, "Re-ID is different: SideView ground-truth masks provide oracle crop and mask guidance.")
    add_footer(s9, "Submitted report: Chapter 4, task-specific preprocessing; illustration is schematic", "10 / 25")

    # --- SLIDE 10 ---
    s10 = prs.slides[10]
    clear_slide(s10)
    set_bg(s10, C_LIGHT_BG)
    add_header(s10, "03  /  VIEWPOINT INVESTIGATION", "We developed a synthetic-to-real cattle viewpoint model")

    # Re-insert Cow visual in top right if exists
    cow10_path = "scratch/extracted_slides/slide_10_img_2.png"
    if os.path.exists(cow10_path):
        s10.shapes.add_picture(cow10_path, Inches(11.80), Inches(0.40), Inches(1.10), Inches(0.82))

    # Left Stack: Box 1 (MOO) + Arrow + Box 2 (Real)
    l10_w = Inches(3.8)
    add_card(s10, Inches(0.58), Inches(2.0), l10_w, Inches(1.5))
    bar_m = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(2.0), l10_w, Inches(0.08))
    bar_m.fill.solid()
    bar_m.fill.fore_color.rgb = C_PURPLE
    bar_m.line.fill.background()

    tb_m = s10.shapes.add_textbox(Inches(0.8), Inches(2.15), l10_w - Inches(0.4), Inches(1.2))
    p = tb_m.text_frame.paragraphs[0]
    p.text = "MOO synthetic data"
    p.font.name = FONT_MAIN
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p2 = tb_m.text_frame.add_paragraph()
    p2.text = "76,800 renders\n8-view ResNet-18 training"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_TEXT_BODY

    # Down arrow
    arr_d = s10.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(2.35), Inches(3.6), Inches(0.18), Inches(0.3))
    arr_d.fill.solid()
    arr_d.fill.fore_color.rgb = C_PURPLE
    arr_d.line.fill.background()

    # Box 2: Real dataset
    add_card(s10, Inches(0.58), Inches(4.0), l10_w, Inches(1.5))
    bar_r = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(4.0), l10_w, Inches(0.08))
    bar_r.fill.solid()
    bar_r.fill.fore_color.rgb = C_PURPLE
    bar_r.line.fill.background()

    tb_r = s10.shapes.add_textbox(Inches(0.8), Inches(4.15), l10_w - Inches(0.4), Inches(1.2))
    p = tb_r.text_frame.paragraphs[0]
    p.text = "Our curated real dataset"
    p.font.name = FONT_MAIN
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p2 = tb_r.text_frame.add_paragraph()
    p2.text = "879 cow crops\n616 train / 132 val / 131 test"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_TEXT_BODY

    # Horizontal Arrow to right card
    arr_rt = s10.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.55), Inches(4.65), Inches(0.4), Inches(0.18))
    arr_rt.fill.solid()
    arr_rt.fill.fore_color.rgb = C_PURPLE
    arr_rt.line.fill.background()

    # Right Card: Fine-tuned 3-class model
    r10_l = Inches(5.15)
    r10_w = Inches(7.6)
    add_card(s10, r10_l, Inches(2.0), r10_w, Inches(3.5))

    tb_ft = s10.shapes.add_textbox(r10_l + Inches(0.35), Inches(2.25), r10_w - Inches(0.7), Inches(0.25))
    p = tb_ft.text_frame.paragraphs[0]
    p.text = "FINE-TUNED 3-CLASS MODEL"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_PURPLE

    # 3 Pill Badges
    badges10 = ["FRONT", "SIDE", "REAR"]
    for j, b_name in enumerate(badges10):
        b_x = r10_l + Inches(0.35) + j * Inches(1.8)
        bdg = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b_x, Inches(2.7), Inches(1.6), Inches(0.42))
        bdg.fill.solid()
        bdg.fill.fore_color.rgb = RGBColor(243, 232, 255)
        bdg.line.fill.background()
        tf = bdg.text_frame
        p = tf.paragraphs[0]
        p.text = b_name
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_PURPLE

    # 3 Big Stats
    stat_w = Inches(2.3)
    stats10 = [
        ("86.26%", "Held-out accuracy"),
        ("84.96%", "Balanced accuracy"),
        ("0.8573", "Macro-F1")
    ]
    for j, (num, lbl) in enumerate(stats10):
        s_x = r10_l + Inches(0.35) + j * (stat_w + Inches(0.2))
        tb_s = s10.shapes.add_textbox(s_x, Inches(3.5), stat_w, Inches(1.4))
        tb_s.text_frame.word_wrap = True
        p = tb_s.text_frame.paragraphs[0]
        p.text = num
        p.font.name = FONT_MAIN
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = C_PURPLE
        p.space_after = Pt(2)

        p2 = tb_s.text_frame.add_paragraph()
        p2.text = lbl
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(12)
        p2.font.color.rgb = C_TEXT_BODY

    # Note
    tb_n10 = s10.shapes.add_textbox(Inches(0.58), Inches(5.75), Inches(12.17), Inches(0.35))
    p = tb_n10.text_frame.paragraphs[0]
    p.text = "MOO initialization was used; its isolated benefit was not tested against a same-split ImageNet control."
    p.font.name = FONT_MAIN
    p.font.size = Pt(11.5)
    p.font.color.rgb = C_TEXT_BODY

    # Banner
    add_banner(s10, "An independently evaluated model—not a fourth task or an input to the final three-task MTL.")
    add_footer(s10, "Report Chapter 4 + viewpoint transfer log, 23 Sep 2026", "11 / 25")

    # --- SLIDE 11 ---
    s11 = prs.slides[11]
    clear_slide(s11)
    set_bg(s11, C_LIGHT_BG)
    add_header(s11, "03  /  POSE AND ANATOMY FEASIBILITY", "Pose output did not guarantee useful cattle anatomy")

    # Left Dark Card
    c11_w = Inches(4.3)
    add_card(s11, Inches(0.58), Inches(1.85), c11_w, Inches(4.15), bg_color=C_DARK_BG, border_color=C_DARK_BG)
    if os.path.exists("scratch/icons/icon_bone.png"):
        s11.shapes.add_picture("scratch/icons/icon_bone.png", Inches(0.95), Inches(2.2), Inches(0.85), Inches(0.55))

    tb_c11 = s11.shapes.add_textbox(Inches(0.95), Inches(3.0), c11_w - Inches(0.7), Inches(2.6))
    tb_c11.text_frame.word_wrap = True
    p = tb_c11.text_frame.paragraphs[0]
    p.text = "WHAT WAS TESTED"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_TEAL_CYAN
    p.space_after = Pt(10)

    p2 = tb_c11.text_frame.add_paragraph()
    p2.text = "SuperAnimal\npose models"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_WHITE
    p2.space_after = Pt(14)

    p3 = tb_c11.text_frame.add_paragraph()
    p3.text = "Landmarks must be correct and useful for the task."
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(203, 213, 225)

    # Right 3 Points
    pts11 = [
        ("01", "Rear-view reliability", "ScienceDB review found unreliable rear-view predictions."),
        ("02", "BCS landmarks", "The available landmark set did not directly capture the required pelvic anatomy."),
        ("03", "Task relevance", "Pose quality and downstream usefulness needed separate validation.")
    ]
    r11_l = Inches(5.2)
    for j, (num, h_txt, b_txt) in enumerate(pts11):
        y = Inches(1.95) + j * Inches(1.3)
        # Number
        tb_num = s11.shapes.add_textbox(r11_l, y, Inches(0.6), Inches(0.4))
        p = tb_num.text_frame.paragraphs[0]
        p.text = num
        p.font.name = FONT_MAIN
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = C_TEAL

        # Heading & Body
        tb_t = s11.shapes.add_textbox(r11_l + Inches(0.7), y, Inches(6.8), Inches(1.1))
        tb_t.text_frame.word_wrap = True
        p = tb_t.text_frame.paragraphs[0]
        p.text = h_txt
        p.font.name = FONT_MAIN
        p.font.size = Pt(16.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK
        p.space_after = Pt(4)

        p2 = tb_t.text_frame.add_paragraph()
        p2.text = b_txt
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(13)
        p2.font.color.rgb = C_TEXT_BODY

    # Cow Photo in bottom right
    cow11_path = "scratch/extracted_slides/slide_11_img_2.png"
    if os.path.exists(cow11_path):
        s11.shapes.add_picture(cow11_path, Inches(10.56), Inches(4.5), Inches(1.92), Inches(1.12))

    add_banner(s11, "Decision: pose/anatomy was investigated, but excluded from the evaluated downstream models.")
    add_footer(s11, "Submitted report: Chapter 4, component selection and perception feasibility table", "12 / 25")

    # --- SLIDE 12 ---
    s12 = prs.slides[12]
    clear_slide(s12)
    set_bg(s12, C_LIGHT_BG)
    add_header(s12, "03  /  TASK-SPECIFIC MODEL DESIGN", "One cattle-centered philosophy, different task heads")

    cols12 = [
        ("BCS", C_TEAL, "One RGB + mask crop", "ResNet-18", "4 cumulative logits", "Ordered BCS prediction"),
        ("BEHAVIOR", C_AMBER, "8 RGB + mask frames", "Per-frame ResNet-18", "Residual Conv1D + pooling", "5 activity classes"),
        ("RE-ID", C_PURPLE, "One RGB + oracle mask crop", "ResNet-18", "512-D normalized embedding", "Cosine-similarity retrieval")
    ]
    for i, (tag, col, box1, box2, box3, footer_title) in enumerate(cols12):
        cl = Inches(0.58) + i * (col_w + gap)
        
        # Tag
        tb = s12.shapes.add_textbox(cl, Inches(1.75), col_w, Inches(0.25))
        p = tb.text_frame.paragraphs[0]
        p.text = tag
        p.font.name = FONT_MAIN
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = col

        # Box 1
        b_y1 = Inches(2.15)
        b_h1 = Inches(0.7)
        add_card(s12, cl, b_y1, col_w, b_h1)
        tb_b1 = s12.shapes.add_textbox(cl + Inches(0.1), b_y1 + Inches(0.15), col_w - Inches(0.2), Inches(0.4))
        p = tb_b1.text_frame.paragraphs[0]
        p.text = box1
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        # Arrow 1
        arr1 = s12.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, cl + col_w/2 - Inches(0.08), b_y1 + b_h1 + Inches(0.06), Inches(0.16), Inches(0.22))
        arr1.fill.solid()
        arr1.fill.fore_color.rgb = col
        arr1.line.fill.background()

        # Box 2
        b_y2 = b_y1 + b_h1 + Inches(0.35)
        add_card(s12, cl, b_y2, col_w, b_h1)
        tb_b2 = s12.shapes.add_textbox(cl + Inches(0.1), b_y2 + Inches(0.15), col_w - Inches(0.2), Inches(0.4))
        p = tb_b2.text_frame.paragraphs[0]
        p.text = box2
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        # Arrow 2
        arr2 = s12.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, cl + col_w/2 - Inches(0.08), b_y2 + b_h1 + Inches(0.06), Inches(0.16), Inches(0.22))
        arr2.fill.solid()
        arr2.fill.fore_color.rgb = col
        arr2.line.fill.background()

        # Box 3
        b_y3 = b_y2 + b_h1 + Inches(0.35)
        add_card(s12, cl, b_y3, col_w, b_h1, bg_color=RGBColor(240, 249, 248) if i==0 else (RGBColor(254, 249, 238) if i==1 else RGBColor(248, 245, 254)))
        tb_b3 = s12.shapes.add_textbox(cl + Inches(0.1), b_y3 + Inches(0.15), col_w - Inches(0.2), Inches(0.4))
        p = tb_b3.text_frame.paragraphs[0]
        p.text = box3
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        # Footer Title
        tb_ft = s12.shapes.add_textbox(cl, b_y3 + b_h1 + Inches(0.3), col_w, Inches(0.4))
        p = tb_ft.text_frame.paragraphs[0]
        p.text = footer_title
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK

    add_banner(s12, "Training uses task-specific datasets; an image is not assumed to carry all three labels.")
    add_footer(s12, "Submitted report: Chapter 4, model specifications", "13 / 25")

    # --- SLIDE 13 ---
    s13 = prs.slides[13]
    clear_slide(s13)
    set_bg(s13, C_LIGHT_BG)
    add_header(s13, "04  /  SINGLE-TASK RESULTS", "BCS: cattle-centered inputs reduced prediction error")

    # Left: Bar Chart Box
    bc_l = Inches(0.58)
    bc_w = Inches(5.8)
    tb_bcs_t = s13.shapes.add_textbox(bc_l, Inches(1.8), bc_w, Inches(0.3))
    p = tb_bcs_t.text_frame.paragraphs[0]
    p.text = "MAE IN PHYSICAL BCS UNITS ↓"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    # Native bar shapes for MAE
    bar_y_base = Inches(5.2)
    b_w_bcs = Inches(1.4)
    # RGB bar: 0.1929 -> height ~1.8 in
    h_rgb = Inches(1.85)
    b_rgb = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, bc_l + Inches(1.2), bar_y_base - h_rgb, b_w_bcs, h_rgb)
    b_rgb.fill.solid()
    b_rgb.fill.fore_color.rgb = C_GRAY_BAR
    b_rgb.line.fill.background()

    tb_val_rgb = s13.shapes.add_textbox(bc_l + Inches(1.2), bar_y_base - h_rgb - Inches(0.35), b_w_bcs, Inches(0.3))
    p = tb_val_rgb.text_frame.paragraphs[0]
    p.text = "0.1929"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Cattle-centered bar: 0.1709 -> height ~1.6 in
    h_cc = Inches(1.60)
    b_cc = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, bc_l + Inches(1.2) + b_w_bcs + Inches(0.1), bar_y_base - h_cc, b_w_bcs, h_cc)
    b_cc.fill.solid()
    b_cc.fill.fore_color.rgb = C_TEAL
    b_cc.line.fill.background()

    tb_val_cc = s13.shapes.add_textbox(bc_l + Inches(1.2) + b_w_bcs + Inches(0.1), bar_y_base - h_cc - Inches(0.35), b_w_bcs, Inches(0.3))
    p = tb_val_cc.text_frame.paragraphs[0]
    p.text = "0.1709"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    # Label below bars
    tb_lbl_bcs = s13.shapes.add_textbox(bc_l + Inches(1.2), bar_y_base + Inches(0.1), b_w_bcs * 2 + Inches(0.1), Inches(0.3))
    p = tb_lbl_bcs.text_frame.paragraphs[0]
    p.text = "BCS MAE"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Legend
    tb_leg = s13.shapes.add_textbox(bc_l + Inches(0.8), bar_y_base + Inches(0.55), Inches(4.5), Inches(0.3))
    p = tb_leg.text_frame.paragraphs[0]
    p.text = "■ RGB reference   ■ Cattle-centered"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_BODY

    # Right Content Card
    rc13_l = Inches(6.8)
    rc13_w = Inches(5.95)
    add_card(s13, rc13_l, Inches(1.85), rc13_w, Inches(4.15))

    tb_acc_card = s13.shapes.add_textbox(rc13_l + Inches(0.35), Inches(2.1), rc13_w - Inches(0.7), Inches(0.9))
    tb_acc_card.text_frame.word_wrap = True
    p = tb_acc_card.text_frame.paragraphs[0]
    p.text = "ACCURACY WITHIN ±0.25 BCS"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(4)

    p2 = tb_acc_card.text_frame.add_paragraph()
    p2.text = "84.95%  →  89.40%"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = C_TEAL

    # Big stat
    tb_stat13 = s13.shapes.add_textbox(rc13_l + Inches(0.35), Inches(3.2), rc13_w - Inches(0.7), Inches(1.2))
    tb_stat13.text_frame.word_wrap = True
    p = tb_stat13.text_frame.paragraphs[0]
    p.text = "-0.0220 BCS units"
    p.font.name = FONT_MAIN
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(4)

    p2 = tb_stat13.text_frame.add_paragraph()
    p2.text = "MAE improvement on the same test images"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(13)
    p2.font.color.rgb = C_TEXT_BODY

    # Note
    tb_n13 = s13.shapes.add_textbox(rc13_l + Inches(0.35), Inches(4.65), rc13_w - Inches(0.7), Inches(1.0))
    tb_n13.text_frame.word_wrap = True
    p = tb_n13.text_frame.paragraphs[0]
    p.text = "Balanced accuracy: 40.23% → 39.70%"
    p.font.name = FONT_MAIN
    p.font.size = Pt(12.5)
    p.font.color.rgb = C_TEXT_DARK
    p2 = tb_n13.text_frame.add_paragraph()
    p2.text = "Not every metric improved."
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_TEXT_MUTED

    add_banner(s13, "7,549 / 8,040 test images retained; results exclude perception failures.")
    add_footer(s13, "Submitted report: BCS comparison table; matched N = 7,549 images", "14 / 25")

    # --- SLIDE 14 ---
    s14 = prs.slides[14]
    clear_slide(s14)
    set_bg(s14, C_LIGHT_BG)
    add_header(s14, "04  /  SINGLE-TASK RESULTS", "Behavior: class-sensitive gains came with a trade-off")

    cow14_path = "scratch/extracted_slides/slide_14_img_2.png"
    if os.path.exists(cow14_path):
        s14.shapes.add_picture(cow14_path, Inches(11.78), Inches(0.40), Inches(1.08), Inches(0.74))

    # Left: Bar chart
    tb_b14_t = s14.shapes.add_textbox(bc_l, Inches(1.8), bc_w, Inches(0.3))
    p = tb_b14_t.text_frame.paragraphs[0]
    p.text = "ACCURACY MEASURES (%)"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    # Draw grouped bars for Overall and Balanced accuracy
    # Cluster 1: Overall (88.46 vs 87.44)
    c1_x = bc_l + Inches(0.8)
    b_w14 = Inches(0.85)
    h_ov_rgb = Inches(2.2)
    h_ov_tcn = Inches(2.17)
    
    b1 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1_x, bar_y_base - h_ov_rgb, b_w14, h_ov_rgb)
    b1.fill.solid()
    b1.fill.fore_color.rgb = C_GRAY_BAR
    b1.line.fill.background()

    b2 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1_x + b_w14 + Inches(0.05), bar_y_base - h_ov_tcn, b_w14, h_ov_tcn)
    b2.fill.solid()
    b2.fill.fore_color.rgb = C_AMBER
    b2.line.fill.background()

    tb_v1 = s14.shapes.add_textbox(c1_x - Inches(0.1), bar_y_base - h_ov_rgb - Inches(0.35), b_w14 * 2 + Inches(0.2), Inches(0.3))
    p = tb_v1.text_frame.paragraphs[0]
    p.text = "88.46   87.44"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    tb_lbl_ov = s14.shapes.add_textbox(c1_x, bar_y_base + Inches(0.1), b_w14 * 2 + Inches(0.05), Inches(0.3))
    p = tb_lbl_ov.text_frame.paragraphs[0]
    p.text = "Overall accuracy"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Cluster 2: Balanced (71.30 vs 74.43)
    c2_x = c1_x + Inches(2.2)
    h_ba_rgb = Inches(1.75)
    h_ba_tcn = Inches(1.85)

    b3 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2_x, bar_y_base - h_ba_rgb, b_w14, h_ba_rgb)
    b3.fill.solid()
    b3.fill.fore_color.rgb = C_GRAY_BAR
    b3.line.fill.background()

    b4 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2_x + b_w14 + Inches(0.05), bar_y_base - h_ba_tcn, b_w14, h_ba_tcn)
    b4.fill.solid()
    b4.fill.fore_color.rgb = C_AMBER
    b4.line.fill.background()

    tb_v2 = s14.shapes.add_textbox(c2_x - Inches(0.1), bar_y_base - h_ba_tcn - Inches(0.35), b_w14 * 2 + Inches(0.2), Inches(0.3))
    p = tb_v2.text_frame.paragraphs[0]
    p.text = "71.30   74.43"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    tb_lbl_ba = s14.shapes.add_textbox(c2_x, bar_y_base + Inches(0.1), b_w14 * 2 + Inches(0.05), Inches(0.3))
    p = tb_lbl_ba.text_frame.paragraphs[0]
    p.text = "Balanced accuracy"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Legend
    tb_leg14 = s14.shapes.add_textbox(bc_l + Inches(0.8), bar_y_base + Inches(0.55), Inches(4.5), Inches(0.3))
    p = tb_leg14.text_frame.paragraphs[0]
    p.text = "■ RGB / single frame   ■ Cattle-centered / 8 frames"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_BODY

    # Right Stat Cards
    add_card(s14, rc13_l, Inches(1.85), rc13_w, Inches(1.6))
    tb_wk = s14.shapes.add_textbox(rc13_l + Inches(0.35), Inches(2.0), rc13_w - Inches(0.7), Inches(1.2))
    p = tb_wk.text_frame.paragraphs[0]
    p.text = "WALKING F1"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_AMBER
    p.space_after = Pt(4)

    p2 = tb_wk.text_frame.add_paragraph()
    p2.text = "0.2174  →  0.2456"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = C_AMBER

    add_card(s14, rc13_l, Inches(3.65), rc13_w, Inches(1.6))
    tb_ls = s14.shapes.add_textbox(rc13_l + Inches(0.35), Inches(3.8), rc13_w - Inches(0.7), Inches(1.2))
    p = tb_ls.text_frame.paragraphs[0]
    p.text = "TEST LOSS"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_AMBER
    p.space_after = Pt(4)

    p2 = tb_ls.text_frame.add_paragraph()
    p2.text = "0.5505  →  0.4430"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = C_AMBER

    # Note
    tb_n14 = s14.shapes.add_textbox(Inches(0.58), Inches(5.75), Inches(12.17), Inches(0.35))
    p = tb_n14.text_frame.paragraphs[0]
    p.text = "Crop, mask and temporal processing were changed together—not isolated motion ablations."
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT_BODY

    add_footer(s14, "Submitted report: behavior comparison and per-class comparison tables; matched N = 780", "15 / 25")

    # --- SLIDE 15 ---
    s15 = prs.slides[15]
    clear_slide(s15)
    set_bg(s15, C_LIGHT_BG)
    add_header(s15, "04  /  SINGLE-TASK RESULTS", "Re-ID: the largest gain appeared in snapshot queries")

    cow15_path = "scratch/extracted_slides/slide_15_img_2.png"
    if os.path.exists(cow15_path):
        s15.shapes.add_picture(cow15_path, Inches(11.66), Inches(0.40), Inches(1.18), Inches(0.78))

    tb_reid_tag = s15.shapes.add_textbox(bc_l, Inches(1.75), bc_w, Inches(0.25))
    p = tb_reid_tag.text_frame.paragraphs[0]
    p.text = "ORACLE MASK GUIDANCE"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_PURPLE

    # Left: Grouped Bar Chart
    tb_b15_t = s15.shapes.add_textbox(bc_l, Inches(2.05), bc_w, Inches(0.25))
    p = tb_b15_t.text_frame.paragraphs[0]
    p.text = "RANK-1 RETRIEVAL (%) ↑"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_PURPLE

    # Barn to parlor: 58.64 vs 63.90
    c1_x15 = bc_l + Inches(0.8)
    h_bp_rgb = Inches(1.45)
    h_bp_ora = Inches(1.60)

    b1_15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1_x15, bar_y_base - h_bp_rgb, b_w14, h_bp_rgb)
    b1_15.fill.solid()
    b1_15.fill.fore_color.rgb = C_GRAY_BAR
    b1_15.line.fill.background()

    b2_15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1_x15 + b_w14 + Inches(0.05), bar_y_base - h_bp_ora, b_w14, h_bp_ora)
    b2_15.fill.solid()
    b2_15.fill.fore_color.rgb = C_PURPLE
    b2_15.line.fill.background()

    tb_v1_15 = s15.shapes.add_textbox(c1_x15 - Inches(0.1), bar_y_base - h_bp_ora - Inches(0.35), b_w14 * 2 + Inches(0.2), Inches(0.3))
    p = tb_v1_15.text_frame.paragraphs[0]
    p.text = "58.64   63.90"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    tb_lbl_bp = s15.shapes.add_textbox(c1_x15, bar_y_base + Inches(0.1), b_w14 * 2 + Inches(0.05), Inches(0.3))
    p = tb_lbl_bp.text_frame.paragraphs[0]
    p.text = "Barn to parlor"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Snapshots to parlor: 38.88 vs 62.93
    c2_x15 = c1_x15 + Inches(2.2)
    h_sp_rgb = Inches(0.98)
    h_sp_ora = Inches(1.58)

    b3_15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2_x15, bar_y_base - h_sp_rgb, b_w14, h_sp_rgb)
    b3_15.fill.solid()
    b3_15.fill.fore_color.rgb = C_GRAY_BAR
    b3_15.line.fill.background()

    b4_15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2_x15 + b_w14 + Inches(0.05), bar_y_base - h_sp_ora, b_w14, h_sp_ora)
    b4_15.fill.solid()
    b4_15.fill.fore_color.rgb = C_PURPLE
    b4_15.line.fill.background()

    tb_v2_15 = s15.shapes.add_textbox(c2_x15 - Inches(0.1), bar_y_base - h_sp_ora - Inches(0.35), b_w14 * 2 + Inches(0.2), Inches(0.3))
    p = tb_v2_15.text_frame.paragraphs[0]
    p.text = "38.88   62.93"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PURPLE

    tb_lbl_sp = s15.shapes.add_textbox(c2_x15, bar_y_base + Inches(0.1), b_w14 * 2 + Inches(0.05), Inches(0.3))
    p = tb_lbl_sp.text_frame.paragraphs[0]
    p.text = "Snapshots to parlor"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Legend
    tb_leg15 = s15.shapes.add_textbox(bc_l + Inches(0.8), bar_y_base + Inches(0.55), Inches(4.5), Inches(0.3))
    p = tb_leg15.text_frame.paragraphs[0]
    p.text = "■ RGB reference   ■ Cattle-centered / oracle"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_BODY

    # Right Content Cards
    add_card(s15, rc13_l, Inches(1.85), rc13_w, Inches(1.8))
    tb_gain = s15.shapes.add_textbox(rc13_l + Inches(0.35), Inches(2.1), rc13_w - Inches(0.7), Inches(1.4))
    tb_gain.text_frame.word_wrap = True
    p = tb_gain.text_frame.paragraphs[0]
    p.text = "+24.05 pp"
    p.font.name = FONT_MAIN
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = C_PURPLE
    p.space_after = Pt(4)

    p2 = tb_gain.text_frame.add_paragraph()
    p2.text = "Snapshot Rank-1 gain"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_DARK

    # mAP Card
    add_card(s15, rc13_l, Inches(3.85), rc13_w, Inches(1.5))
    tb_map = s15.shapes.add_textbox(rc13_l + Inches(0.35), Inches(4.0), rc13_w - Inches(0.7), Inches(1.2))
    p = tb_map.text_frame.paragraphs[0]
    p.text = "mAP: RGB  →  CATTLE-CENTERED"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_PURPLE
    p.space_after = Pt(4)

    p2 = tb_map.text_frame.add_paragraph()
    p2.text = "Barn:  38.32%  →  40.68%"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(13)
    p2.font.color.rgb = C_TEXT_DARK
    p2.space_after = Pt(2)

    p3 = tb_map.text_frame.add_paragraph()
    p3.text = "Snapshot:  27.05%  →  40.42%"
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = C_PURPLE

    # Note
    tb_n15 = s15.shapes.add_textbox(Inches(0.58), Inches(5.75), Inches(12.17), Inches(0.35))
    p = tb_n15.text_frame.paragraphs[0]
    p.text = "Gallery: 36,811 parlor images | Queries: 25,260 barn images + 607 snapshots"
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT_BODY

    add_footer(s15, "Submitted report: Chapter 5 Re-ID results; 69 held-out evaluation identities", "16 / 25")

    # --- SLIDE 16 (DARK SLIDE) ---
    s16 = prs.slides[16]
    clear_slide(s16)
    set_bg(s16, C_DARK_BG)
    add_header(s16, "05  /  UNIFIED MULTI-TASK FRAMEWORK", "One shared backbone learns across three task datasets", is_dark=True)

    # Diagram: 3 Input Batches (Left)
    b_in_w = Inches(3.2)
    b_in_h = Inches(0.7)
    batches16 = ["BCS batch", "8-frame behavior batch", "Re-ID batch"]
    for j, bname in enumerate(batches16):
        by = Inches(2.2) + j * Inches(1.1)
        add_card(s16, Inches(0.58), by, b_in_w, b_in_h, bg_color=C_DARK_CARD, border_color=C_DARK_BORDER)
        tb = s16.shapes.add_textbox(Inches(0.75), by + Inches(0.18), b_in_w - Inches(0.3), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        p.text = bname
        p.font.name = FONT_MAIN
        p.font.size = Pt(15.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE

    # Center Box: Shared ResNet-18
    c16_l = Inches(4.8)
    c16_w = Inches(3.6)
    c16_h = Inches(2.4)
    add_card(s16, c16_l, Inches(2.4), c16_w, c16_h, bg_color=C_DARK_CARD, border_color=C_TEAL)
    
    t_bar16 = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, c16_l, Inches(2.4), c16_w, Inches(0.08))
    t_bar16.fill.solid()
    t_bar16.fill.fore_color.rgb = C_TEAL
    t_bar16.line.fill.background()

    tb_sh = s16.shapes.add_textbox(c16_l + Inches(0.2), Inches(2.65), c16_w - Inches(0.4), Inches(1.9))
    tb_sh.text_frame.word_wrap = True
    p = tb_sh.text_frame.paragraphs[0]
    p.text = "Shared ResNet-18"
    p.font.name = FONT_MAIN
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE
    p.space_after = Pt(8)

    p2 = tb_sh.text_frame.add_paragraph()
    p2.text = "4-channel spatial input\n512-D features"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(13.5)
    p2.font.color.rgb = RGBColor(203, 213, 225)

    # Right: 3 Heads
    r16_l = Inches(9.2)
    heads16 = ["BCS ordinal head", "Temporal behavior head", "Re-ID embedding / ID head"]
    for j, hname in enumerate(heads16):
        hy = Inches(2.2) + j * Inches(1.1)
        add_card(s16, r16_l, hy, b_in_w, b_in_h, bg_color=C_DARK_CARD, border_color=C_DARK_BORDER)
        tb = s16.shapes.add_textbox(r16_l + Inches(0.2), hy + Inches(0.18), b_in_w - Inches(0.3), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        p.text = hname
        p.font.name = FONT_MAIN
        p.font.size = Pt(15.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE

    # 3 Bottom Comparison Cards
    bot_card_w = Inches(3.82)
    bot_cards16 = [
        ("E1", "Ordinary hard sharing", "11,926,706 parameters"),
        ("E3", "Task-private residual adapters", "+395,904 parameters (+3.32%)"),
        ("E4", "Same E1 model + PCGrad", "Projection on shared gradients only")
    ]
    for j, (etag, ename, edesc) in enumerate(bot_cards16):
        bx = Inches(0.58) + j * (bot_card_w + gap)
        add_card(s16, bx, Inches(5.15), bot_card_w, Inches(1.5), bg_color=C_DARK_CARD, border_color=C_DARK_BORDER)
        
        tb = s16.shapes.add_textbox(bx + Inches(0.25), Inches(5.35), bot_card_w - Inches(0.5), Inches(1.1))
        tb.text_frame.word_wrap = True
        p = tb.text_frame.paragraphs[0]
        p.text = f"{etag}   {ename}"
        p.font.name = FONT_MAIN
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = C_TEAL_CYAN
        p.space_after = Pt(6)

        p2 = tb.text_frame.add_paragraph()
        p2.text = edesc
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(12)
        p2.font.color.rgb = RGBColor(203, 213, 225)

    add_footer(s16, "Submitted report: Chapter 4, E1 / E3 / E4 architectures and super-step training", "17 / 25", is_dark=True)

    # --- SLIDE 17 ---
    s17 = prs.slides[17]
    clear_slide(s17)
    set_bg(s17, C_LIGHT_BG)
    add_header(s17, "05  /  MULTI-TASK RESULTS", "Joint learning did not match the independent task models")

    table_shape17 = s17.shapes.add_table(5, 4, Inches(0.58), Inches(1.85), Inches(12.17), Inches(3.6))
    tbl17 = table_shape17.table
    tbl17.columns[0].width = Inches(4.2)
    tbl17.columns[1].width = Inches(2.6)
    tbl17.columns[2].width = Inches(2.7)
    tbl17.columns[3].width = Inches(2.67)

    headers17 = ["Model", "BCS MAE ↓", "Behavior balanced acc. ↑", "Re-ID barn Rank-1 ↑"]
    for c_i, h_t in enumerate(headers17):
        c = tbl17.cell(0, c_i)
        c.text = h_t
        c.fill.solid()
        c.fill.fore_color.rgb = C_DARK_BG
        p = c.text_frame.paragraphs[0]
        p.font.name = FONT_MAIN
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER if c_i > 0 else PP_ALIGN.LEFT

    rows17 = [
        ("Independent single-task", "0.1709", "74.43%", "63.90%"),
        ("E1  Hard sharing", "0.1788", "67.30%", "57.38%"),
        ("E3  Private adapters", "0.1916", "66.70%", "49.08%"),
        ("E4  PCGrad", "0.1828", "69.15%", "54.97%")
    ]
    for r_i, (m_name, bcs_val, beh_val, reid_val) in enumerate(rows17):
        is_highlight = (r_i == 0)
        for c_i, val in enumerate([m_name, bcs_val, beh_val, reid_val]):
            cell = tbl17.cell(r_i + 1, c_i)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_MINT_PILL if is_highlight else C_WHITE_CARD
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_MAIN
            p.font.size = Pt(14)
            p.alignment = PP_ALIGN.CENTER if c_i > 0 else PP_ALIGN.LEFT
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = C_TEAL
            else:
                p.font.color.rgb = C_TEXT_DARK

    tb_note17 = s17.shapes.add_textbox(Inches(0.58), Inches(5.6), Inches(12.17), Inches(0.35))
    p = tb_note17.text_frame.paragraphs[0]
    p.text = "BCS: 7,549 images   |   Behavior: 780 sequences   |   Re-ID: fixed Protocol A"
    p.font.name = FONT_MAIN
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT_BODY

    add_banner(s17, "E4 improved behavior balanced accuracy over E1 by 1.85 percentage points—not all tasks.", top=Inches(6.1))
    add_footer(s17, "Submitted report: MTL BCS, behavior and Re-ID comparison tables; matched held-out sets", "18 / 25")

    # --- SLIDE 18 ---
    s18 = prs.slides[18]
    clear_slide(s18)
    set_bg(s18, C_LIGHT_BG)
    add_header(s18, "05  /  GRADIENT CONFLICT DIAGNOSTICS", "Opposing gradients were observed during E4 training")

    # Left: Horizontal Bar Chart
    tb_b18_t = s18.shapes.add_textbox(bc_l, Inches(1.8), Inches(6.0), Inches(0.3))
    p = tb_b18_t.text_frame.paragraphs[0]
    p.text = "PAIRWISE CONFLICT FREQUENCY (%)"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    # 3 Horizontal Bars
    bars18 = [
        ("Behavior / Re-ID", 47.4),
        ("BCS / Re-ID", 46.5),
        ("BCS / Behavior", 47.8)
    ]
    bar_y_start = Inches(2.4)
    bar_h18 = Inches(0.55)
    bar_gap18 = Inches(0.4)
    max_w18 = Inches(4.5)

    for j, (p_label, p_val) in enumerate(bars18):
        y = bar_y_start + j * (bar_h18 + bar_gap18)
        
        # Label
        tb_lbl = s18.shapes.add_textbox(Inches(0.58), y + Inches(0.08), Inches(2.2), Inches(0.35))
        p = tb_lbl.text_frame.paragraphs[0]
        p.text = p_label
        p.alignment = PP_ALIGN.RIGHT
        p.font.name = FONT_MAIN
        p.font.size = Pt(13)
        p.font.color.rgb = C_TEXT_DARK

        # Bar
        w = max_w18 * (p_val / 100.0)
        bar = s18.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.9), y, w, bar_h18)
        bar.fill.solid()
        bar.fill.fore_color.rgb = C_TEAL
        bar.line.fill.background()

        # Value text
        tb_v = s18.shapes.add_textbox(Inches(2.9) + w + Inches(0.12), y + Inches(0.08), Inches(0.8), Inches(0.35))
        p = tb_v.text_frame.paragraphs[0]
        p.text = str(p_val)
        p.font.name = FONT_MAIN
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK

    # Axis below
    tb_ax = s18.shapes.add_textbox(Inches(2.8), bar_y_start + 3 * (bar_h18 + bar_gap18) - Inches(0.2), max_w18 + Inches(0.4), Inches(0.3))
    p = tb_ax.text_frame.paragraphs[0]
    p.text = "0           20          40          60          80          100"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10)
    p.font.color.rgb = C_TEXT_MUTED

    # Right Big Stat Callouts
    rc18_l = Inches(8.2)
    rc18_w = Inches(4.5)
    
    # Stat 1
    tb_s1 = s18.shapes.add_textbox(rc18_l, Inches(2.1), rc18_w, Inches(1.5))
    tb_s1.text_frame.word_wrap = True
    p = tb_s1.text_frame.paragraphs[0]
    p.text = "16,140"
    p.font.name = FONT_MAIN
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p.space_after = Pt(2)

    p2 = tb_s1.text_frame.add_paragraph()
    p2.text = "Multi-task super-steps"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(14)
    p2.font.color.rgb = C_TEXT_BODY

    # Stat 2
    tb_s2 = s18.shapes.add_textbox(rc18_l, Inches(3.8), rc18_w, Inches(1.5))
    tb_s2.text_frame.word_wrap = True
    p = tb_s2.text_frame.paragraphs[0]
    p.text = "44,177"
    p.font.name = FONT_MAIN
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(2)

    p2 = tb_s2.text_frame.add_paragraph()
    p2.text = "Triggered conflict projections"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(14)
    p2.font.color.rgb = C_TEXT_BODY

    add_banner(s18, "These diagnostics do not prove that gradient conflict alone caused E1 performance loss.")
    add_footer(s18, "Submitted report: PCGrad gradient diagnostics table; measurements apply to E4 only", "19 / 25")

    # --- SLIDE 19 ---
    s19 = prs.slides[19]
    clear_slide(s19)
    set_bg(s19, C_LIGHT_BG)
    add_header(s19, "06  /  ANSWERS TO THE RESEARCH QUESTIONS", "What did the experiments establish?")

    rq_answers = [
        ("RQ1", "Cattle-centered representations were useful", "Important BCS and Re-ID metrics improved; behavior gains depended on the metric."),
        ("RQ2", "Information requirements differ by task", "Static morphology, temporal activity and individual appearance need different treatment."),
        ("RQ3", "Sharing produced measured trade-offs", "The tested MTL designs did not universally recover single-task performance.")
    ]
    for i, (rq_num, rq_h, rq_b) in enumerate(rq_answers):
        ct = rq_top_base + i * (rq_card_h + rq_gap)
        add_card(s19, Inches(0.58), ct, Inches(12.17), rq_card_h)
        
        # Badge
        badge = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), ct + Inches(0.3), Inches(0.85), Inches(0.75))
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_DARK_BG
        badge.line.fill.background()
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.text = rq_num
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_MAIN
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE

        # Heading & Body
        tb = s19.shapes.add_textbox(Inches(2.0), ct + Inches(0.25), Inches(10.0), Inches(0.9))
        tb.text_frame.word_wrap = True
        p = tb.text_frame.paragraphs[0]
        p.text = rq_h
        p.font.name = FONT_MAIN
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK
        p.space_after = Pt(4)

        p2 = tb.text_frame.add_paragraph()
        p2.text = rq_b
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(13)
        p2.font.color.rgb = C_TEXT_BODY

    add_footer(s19, "Submitted report: Chapter 6, answers to RQ1-RQ3", "20 / 25")

    # --- SLIDE 20 ---
    s20 = prs.slides[20]
    clear_slide(s20)
    set_bg(s20, C_LIGHT_BG)
    add_header(s20, "06  /  THESIS CONTRIBUTIONS", "Our contribution: a unified three-task cattle MTL framework")

    # Top Central Banner Card
    add_card(s20, Inches(0.58), Inches(1.85), Inches(12.17), Inches(1.3))
    tb_top20 = s20.shapes.add_textbox(Inches(0.8), Inches(2.05), Inches(11.7), Inches(0.9))
    tb_top20.text_frame.word_wrap = True
    p = tb_top20.text_frame.paragraphs[0]
    p.text = "CONDITION + ACTIVITY + IDENTITY"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(4)

    p2 = tb_top20.text_frame.add_paragraph()
    p2.text = "BCS, behavior and cow Re-ID trained and evaluated in one multi-task investigation"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(17)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_DARK

    # 4 Grid Cards
    grid4_w = Inches(5.9)
    grid4_h = Inches(1.45)
    c4_data = [
        ("Task-appropriate representations", "Matched RGB and cattle-centered comparisons."),
        ("Real-cattle viewpoint model", "MOO initialization + our curated real-cattle data."),
        ("Provenance-aware protocols", "Protect passage, session and identity groupings."),
        ("Multi-task conflict evidence", "E1 / E3 / E4 comparisons and E4 diagnostics.")
    ]
    for idx, (head_t, body_t) in enumerate(c4_data):
        cx = Inches(0.58) if (idx % 2 == 0) else Inches(6.85)
        cy = Inches(3.4) if (idx < 2) else Inches(5.1)
        add_card(s20, cx, cy, grid4_w, grid4_h)

        tb = s20.shapes.add_textbox(cx + Inches(0.35), cy + Inches(0.25), grid4_w - Inches(0.7), Inches(1.0))
        tb.text_frame.word_wrap = True
        p = tb.text_frame.paragraphs[0]
        p.text = head_t
        p.font.name = FONT_MAIN
        p.font.size = Pt(15.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK
        p.space_after = Pt(4)

        p2 = tb.text_frame.add_paragraph()
        p2.text = body_t
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(12.5)
        p2.font.color.rgb = C_TEXT_BODY

    add_footer(s20, "Submitted report: Chapters 1, 4 and 6; viewpoint development log", "21 / 25")

    # --- SLIDE 21 ---
    s21 = prs.slides[21]
    clear_slide(s21)
    set_bg(s21, C_LIGHT_BG)
    add_header(s21, "06  /  LIMITATIONS", "The claims stop at the evidence we evaluated")

    lim_cards = [
        ("scratch/icons/lim_icon_1.png", "Single-run estimates", "No repeated-seed distributions or significance tests."),
        ("scratch/icons/lim_icon_2.png", "Oracle Re-ID masks", "Automatic segmentation errors are not evaluated end to end."),
        ("scratch/icons/lim_icon_3.png", "Combined configurations", "Crop, mask and temporal effects are not isolated."),
        ("scratch/icons/lim_icon_4.png", "Grouping limits", "BCS and behavior do not establish unseen-cow generalization."),
        ("scratch/icons/lim_icon_5.png", "Source and coverage bias", "Walking is CVB-only; perception failures are excluded."),
        ("scratch/icons/lim_icon_6.png", "Transfer remains unverified", "External datasets and downstream viewpoint transfer were not tested.")
    ]
    lim_w = Inches(3.82)
    lim_h = Inches(2.15)
    lim_gap_x = Inches(0.35)
    lim_gap_y = Inches(0.25)

    for idx, (icon_p, title_l, desc_l) in enumerate(lim_cards):
        col_idx = idx % 3
        row_idx = idx // 3
        lx = Inches(0.58) + col_idx * (lim_w + lim_gap_x)
        ly = Inches(1.9) + row_idx * (lim_h + lim_gap_y)
        
        add_card(s21, lx, ly, lim_w, lim_h)
        if os.path.exists(icon_p):
            s21.shapes.add_picture(icon_p, lx + Inches(0.3), ly + Inches(0.25), Inches(0.35), Inches(0.35))
        
        tb = s21.shapes.add_textbox(lx + Inches(0.8), ly + Inches(0.2), lim_w - Inches(1.0), Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = title_l
        p.font.name = FONT_MAIN
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK

        tb2 = s21.shapes.add_textbox(lx + Inches(0.3), ly + Inches(0.8), lim_w - Inches(0.6), Inches(1.2))
        tb2.text_frame.word_wrap = True
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = desc_l
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(13)
        p2.font.color.rgb = C_TEXT_BODY

    add_footer(s21, "Submitted report: Chapter 6, limitations and evidence boundaries", "22 / 25")

    # --- SLIDE 22 ---
    s22 = prs.slides[22]
    clear_slide(s22)
    set_bg(s22, C_LIGHT_BG)
    add_header(s22, "06  /  FUTURE WORK", "Future work follows directly from the observed limits")

    fw_items = [
        ("01", "Repeat and quantify uncertainty", "Multiple seeds, uncertainty estimates and stronger comparisons."),
        ("02", "Use automatic Re-ID segmentation", "Replace oracle masks and measure segmentation-error impact."),
        ("03", "Evaluate external herds", "Frozen external tests and longitudinal retrieval protocols."),
        ("04", "Separate component effects", "Isolate crop, binary / soft mask and temporal contributions."),
        ("05", "Validate viewpoint and anatomy transfer", "Test downstream usefulness before model integration."),
        ("06", "Explore selective sharing", "Partial sharing and adaptive multi-task optimization.")
    ]
    col_fw_w = Inches(5.8)
    for idx, (num, h_fw, b_fw) in enumerate(fw_items):
        is_right = (idx % 2 == 1)
        row_num = idx // 2
        fx = Inches(0.58) if not is_right else Inches(6.85)
        fy = Inches(2.0) + row_num * Inches(1.5)

        tb_num = s22.shapes.add_textbox(fx, fy, Inches(0.6), Inches(0.4))
        p = tb_num.text_frame.paragraphs[0]
        p.text = num
        p.font.name = FONT_MAIN
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = C_TEAL

        tb_c = s22.shapes.add_textbox(fx + Inches(0.7), fy, col_fw_w - Inches(0.7), Inches(1.3))
        tb_c.text_frame.word_wrap = True
        p = tb_c.text_frame.paragraphs[0]
        p.text = h_fw
        p.font.name = FONT_MAIN
        p.font.size = Pt(16.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_DARK
        p.space_after = Pt(4)

        p2 = tb_c.text_frame.add_paragraph()
        p2.text = b_fw
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(13)
        p2.font.color.rgb = C_TEXT_BODY

    add_footer(s22, "Submitted report: Chapter 6, future work; these are not completed experiments", "23 / 25")

    # --- SLIDE 23 (DARK SLIDE) ---
    s23 = prs.slides[23]
    clear_slide(s23)
    set_bg(s23, C_DARK_BG)
    add_header(s23, "06  /  CONCLUSION", "Unification is possible. Its trade-offs must be measured.", is_dark=True)

    concl_items = [
        ("01", "A unified framework was developed and evaluated.", "BCS + Behavior Recognition + Cow Re-Identification"),
        ("02", "Cattle-centered representations provided useful gains.", "The gains depended on the task and the metric."),
        ("03", "Joint learning remained a performance trade-off.", "Adapters and PCGrad did not remove negative transfer universally.")
    ]
    for idx, (num, h_c, b_c) in enumerate(concl_items):
        cy = Inches(2.2) + idx * Inches(1.4)
        
        tb_num = s23.shapes.add_textbox(Inches(0.58), cy, Inches(0.6), Inches(0.4))
        p = tb_num.text_frame.paragraphs[0]
        p.text = num
        p.font.name = FONT_MAIN
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = C_TEAL_CYAN

        tb_c = s23.shapes.add_textbox(Inches(1.4), cy, Inches(11.0), Inches(1.2))
        tb_c.text_frame.word_wrap = True
        p = tb_c.text_frame.paragraphs[0]
        p.text = h_c
        p.font.name = FONT_MAIN
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(4)

        p2 = tb_c.text_frame.add_paragraph()
        p2.text = b_c
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(14)
        p2.font.color.rgb = RGBColor(175, 194, 207)

    add_footer(s23, "Submitted report: Chapter 6, final conclusion", "24 / 25", is_dark=True)

    # --- SLIDE 24 (DARK SLIDE) ---
    s24 = prs.slides[24]
    clear_slide(s24)
    set_bg(s24, C_DARK_BG)
    
    # Eyebrow
    tb_eye24 = s24.shapes.add_textbox(Inches(0.58), Inches(0.50), Inches(8.0), Inches(0.28))
    p = tb_eye24.text_frame.paragraphs[0]
    p.text = "BRAC UNIVERSITY  /  CSE THESIS DEFENSE"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL_CYAN

    # Title: Thank you.
    tb_ty = s24.shapes.add_textbox(Inches(0.58), Inches(1.8), Inches(8.0), Inches(1.1))
    p = tb_ty.text_frame.paragraphs[0]
    p.text = "Thank you."
    p.font.name = FONT_MAIN
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE

    # Subtitle: Questions?
    tb_q = s24.shapes.add_textbox(Inches(0.58), Inches(3.2), Inches(8.0), Inches(1.0))
    p = tb_q.text_frame.paragraphs[0]
    p.text = "Questions?"
    p.font.name = FONT_MAIN
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = C_TEAL_LIGHT

    # 3 Items with Icons (Condition, Activity, Identity)
    icon_row_y = Inches(5.0)
    
    # Condition
    if os.path.exists("scratch/icons/icon_bcs_scale.png"):
        s24.shapes.add_picture("scratch/icons/icon_bcs_scale.png", Inches(0.58), icon_row_y, Inches(0.5), Inches(0.5))
    tb_cnd = s24.shapes.add_textbox(Inches(1.2), icon_row_y + Inches(0.08), Inches(2.5), Inches(0.4))
    p = tb_cnd.text_frame.paragraphs[0]
    p.text = "Condition"
    p.font.name = FONT_MAIN
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE

    # Activity
    if os.path.exists("scratch/icons/icon_behavior_walk.png"):
        s24.shapes.add_picture("scratch/icons/icon_behavior_walk.png", Inches(4.5), icon_row_y, Inches(0.5), Inches(0.5))
    tb_act = s24.shapes.add_textbox(Inches(5.1), icon_row_y + Inches(0.08), Inches(2.5), Inches(0.4))
    p = tb_act.text_frame.paragraphs[0]
    p.text = "Activity"
    p.font.name = FONT_MAIN
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE

    # Identity
    if os.path.exists("scratch/icons/icon_reid_fingerprint.png"):
        s24.shapes.add_picture("scratch/icons/icon_reid_fingerprint.png", Inches(8.0), icon_row_y, Inches(0.5), Inches(0.5))
    tb_idn = s24.shapes.add_textbox(Inches(8.6), icon_row_y + Inches(0.08), Inches(2.5), Inches(0.4))
    p = tb_idn.text_frame.paragraphs[0]
    p.text = "Identity"
    p.font.name = FONT_MAIN
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE

    # Re-insert Cow visual in bottom right if exists
    cow24_path = "scratch/extracted_slides/slide_24_img_2.png"
    if os.path.exists(cow24_path):
        s24.shapes.add_picture(cow24_path, Inches(9.2), Inches(4.8), Inches(3.0), Inches(1.8))

    add_footer(s24, "Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring", "25 / 25", is_dark=True)

    # Save
    prs.save(pptx_path)
    print(f"Deck converted successfully! Saved to: {os.path.abspath(pptx_path)}")

if __name__ == "__main__":
    convert_deck()
