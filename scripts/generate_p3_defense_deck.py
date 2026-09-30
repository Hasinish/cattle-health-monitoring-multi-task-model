"""
Phase 3 Defense Presentation Generator
Generates a 16:9 widescreen, fully editable Microsoft PowerPoint (.pptx) presentation
comprising all 25 defense slides (Slides 0 to 24) with complete academic rigor,
exact benchmark numbers, clean visual card hierarchy, and custom color styling.
"""

import sys
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path="Phase3_Cattle_Health_MTL_Defense.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # --- Color Palette ---
    COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900
    COLOR_BG_LIGHT = RGBColor(248, 250, 252)   # Slate 50
    COLOR_CARD_BG = RGBColor(255, 255, 255)    # Pure White
    COLOR_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
    COLOR_CARD_BORDER_ACCENT = RGBColor(14, 165, 233) # Sky 500
    
    COLOR_TEXT_PRIMARY = RGBColor(15, 23, 42)   # Dark Charcoal
    COLOR_TEXT_SECONDARY = RGBColor(71, 85, 105) # Slate 600
    COLOR_TEXT_MUTED = RGBColor(148, 163, 184)   # Slate 400
    COLOR_TEXT_WHITE = RGBColor(255, 255, 255)
    
    COLOR_ACCENT_TEAL = RGBColor(13, 148, 136)   # Teal 600
    COLOR_ACCENT_BLUE = RGBColor(2, 132, 199)    # Sky 600
    COLOR_ACCENT_EMERALD = RGBColor(16, 185, 129) # Emerald 500
    COLOR_ACCENT_AMBER = RGBColor(217, 119, 6)   # Amber 600
    COLOR_ACCENT_ROSE = RGBColor(225, 29, 72)    # Rose 600
    COLOR_CARD_MUTED_BG = RGBColor(241, 245, 249) # Slate 100

    def add_blank_slide(bg_color=COLOR_BG_LIGHT):
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = bg_color
        bg.line.fill.background()
        return slide

    def add_header(slide, slide_num, category, title, subtitle=None, is_dark=False):
        # Category Badge
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = f"SLIDE {slide_num:02d} | {category.upper()}"
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(9.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT_BLUE if not is_dark else RGBColor(56, 189, 248)

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.733), Inches(0.6))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_PRIMARY if not is_dark else COLOR_TEXT_WHITE

        # Subtitle
        if subtitle:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.4))
            tf_sub = sub_box.text_frame
            tf_sub.word_wrap = True
            tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = COLOR_TEXT_SECONDARY if not is_dark else RGBColor(203, 213, 225)

    def add_footer(slide, slide_num, is_dark=False):
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.3))
        tf = footer_box.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = f"Multi-Task Deep Learning Framework for Cattle Health & Behavior Monitoring  •  BRAC University  •  Slide {slide_num} of 24"
        p.font.name = "Arial"
        p.font.size = Pt(9)
        p.font.color.rgb = COLOR_TEXT_MUTED if not is_dark else RGBColor(100, 116, 139)

    def add_card(slide, left, top, width, height, title=None, title_color=COLOR_TEXT_PRIMARY, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.2)
        else:
            card.line.fill.background()
        
        # If title given, put text box
        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = title_color
            return card, top + Inches(0.65)
        return card, top + Inches(0.2)

    def populate_card_text(slide, left, top, width, height, bullets, font_size=11):
        tb = slide.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = b
            p.font.name = "Arial"
            p.font.size = Pt(font_size)
            p.font.color.rgb = COLOR_TEXT_SECONDARY
            p.space_after = Pt(6)
            p.level = 0

    # =========================================================================
    # SLIDE 0: TITLE PAGE (DARK ELEGANT HERO SLIDE)
    # =========================================================================
    s0 = add_blank_slide(bg_color=COLOR_BG_DARK)
    
    # Hero Title Box
    t_box = s0.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring"
    p1.font.name = "Arial"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE
    p1.space_after = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Undergraduate Thesis Defense (Phase 3 Final Presentation)  •  Department of Computer Science & Engineering"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(56, 189, 248)

    # Authors Card (Dark Slate)
    ac = s0.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.7), Inches(5.5), Inches(2.8))
    ac.fill.solid()
    ac.fill.fore_color.rgb = RGBColor(30, 41, 59)
    ac.line.color.rgb = RGBColor(51, 65, 85)
    
    atb = s0.shapes.add_textbox(Inches(1.3), Inches(3.9), Inches(4.9), Inches(2.4))
    atf = atb.text_frame
    atf.word_wrap = True
    ap_h = atf.paragraphs[0]
    ap_h.text = "RESEARCH TEAM (AUTHORS)"
    ap_h.font.name = "Arial"
    ap_h.font.size = Pt(11)
    ap_h.font.bold = True
    ap_h.font.color.rgb = RGBColor(148, 163, 184)
    ap_h.space_after = Pt(8)

    members = [
        "Hasin Ishrak (Student ID: 22201133)",
        "Namira Abrar Haque (Student ID: 22201191)",
        "Sanjida Akter Bithi (Student ID: 22201180)",
        "Shouvik Banik (Student ID: 23101295)",
        "Nusrat Lamya Faruk (Student ID: 24241182)"
    ]
    for m in members:
        p = atf.add_paragraph()
        p.text = f"• {m}"
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_after = Pt(4)

    # Committee / Institutional Card (Dark Slate)
    sc = s0.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.7), Inches(5.5), Inches(2.8))
    sc.fill.solid()
    sc.fill.fore_color.rgb = RGBColor(30, 41, 59)
    sc.line.color.rgb = RGBColor(51, 65, 85)

    stb = s0.shapes.add_textbox(Inches(7.1), Inches(3.9), Inches(4.9), Inches(2.4))
    stf = stb.text_frame
    stf.word_wrap = True
    sp_h = stf.paragraphs[0]
    sp_h.text = "SUPERVISION & INSTITUTION"
    sp_h.font.name = "Arial"
    sp_h.font.size = Pt(11)
    sp_h.font.bold = True
    sp_h.font.color.rgb = RGBColor(148, 163, 184)
    sp_h.space_after = Pt(8)

    sup_details = [
        "Supervisor: Dr. Md. Khalilur Rahman",
        "Professor, Dept. of Computer Science & Engineering",
        "Co-Supervisor: Mehedi Hasan Emo",
        "Lecturer, Dept. of Computer Science & Engineering",
        "Institution: Brac University, Dhaka, Bangladesh",
        "Date of Defense: September 2026"
    ]
    for s_item in sup_details:
        p = stf.add_paragraph()
        p.text = s_item
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(226, 232, 240) if "Supervisor" in s_item else RGBColor(148, 163, 184)
        p.space_after = Pt(3)

    add_footer(s0, 0, is_dark=True)

    # =========================================================================
    # SLIDE 1: MOTIVATION
    # =========================================================================
    s1 = add_blank_slide()
    add_header(s1, 1, "Context & Motivation", "The Need for Unified, Non-Invasive Cattle Monitoring", 
               "Moving from fragmented physical sensors to integrated multi-dimensional vision intelligence")
    
    # 3 Column Cards
    col_w = Inches(3.68)
    gap = Inches(0.34)
    c1_left = Inches(0.8)
    c2_left = c1_left + col_w + gap
    c3_left = c2_left + col_w + gap

    # Card 1: Precision Livestock Farming
    add_card(s1, c1_left, Inches(1.85), col_w, Inches(4.9), "1. Precision Dairy Welfare", COLOR_ACCENT_BLUE)
    populate_card_text(s1, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Modern commercial dairy operations manage hundreds of cows, making individual manual checks infeasible.",
        "• Wearable Sensors Limitations: Accelerometer neck collars, ear tags, and pedometers suffer from high unit costs, mechanical detachment, battery life constraints, and animal stress during tagging.",
        "• Non-Invasive Imperative: Passive optical cameras offer continuous 24/7 observation without touching or stressing the livestock."
    ], font_size=11)

    # Card 2: Multi-Dimensional Health
    add_card(s1, c2_left, Inches(1.85), col_w, Inches(4.9), "2. The Multi-Task Reality", COLOR_ACCENT_TEAL)
    populate_card_text(s1, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Welfare cannot be assessed from a single isolated signal.",
        "• A cow may reduce feed intake (Behavior), exhibit gradual tissue catabolism (Body Condition Score), and require persistent identity linking (Re-ID) across barn chutes.",
        "• Linking Health Over Time: An observation is clinically actionable only when mapped to the correct individual cow across days and lactations.",
        "• Current Practice: Farms deploy isolated, single-purpose models resulting in severe compute redundancy."
    ], font_size=11)

    # Card 3: Multi-Task Deep Learning
    add_card(s1, c3_left, Inches(1.85), col_w, Inches(4.9), "3. Unified Vision Opportunity", COLOR_ACCENT_EMERALD)
    populate_card_text(s1, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Centralized Edge Processing: A single unified neural backbone running on farm edge hardware can process video streams for multiple analytical outputs.",
        "• Shared Animal Semantics: Low-level edge, contour, and anatomy features should theoretically be shared across tasks.",
        "• The Real Challenge: Heterogeneous visual tasks conflict. Combining them naively causes destructive interference rather than synergy.",
        "• Goal: Build a scientifically calibrated multi-task framework with verified leakage-free protocols."
    ], font_size=11)

    add_footer(s1, 1)

    # =========================================================================
    # SLIDE 2: THE MONITORING TASKS
    # =========================================================================
    s2 = add_blank_slide()
    add_header(s2, 2, "Task Definitions", "The Three Primary Cattle Monitoring Domains",
               "Three heterogeneous visual perception tasks spanning physiology, ethology, and biometrics")
    
    add_card(s2, c1_left, Inches(1.85), col_w, Inches(4.9), "Body Condition Scoring (BCS)", COLOR_ACCENT_BLUE)
    populate_card_text(s2, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Physiological Goal: Evaluates energy reserves and subcutaneous fat cover to manage nutritional health, fertility, and metabolic disease.",
        "• Anatomical Basis: Visual appraisal of skeletal landmarks: tailhead depression, pin bones (ischium), hook bones (ilium), sacral ridge, and loin.",
        "• Mathematical Formulation: 5-class quarter-point ordinal scale (2.75, 3.00, 3.25, 3.50, 3.75). Evaluated via cumulative ordinal classification.",
        "• Key Requirement: Precise anatomical contour and curvature perception; invariant to cow coat color or barn background."
    ], font_size=11)

    add_card(s2, c2_left, Inches(1.85), col_w, Inches(4.9), "Behavior Recognition", COLOR_ACCENT_TEAL)
    populate_card_text(s2, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Ethological Goal: Real-time detection of vital activities to identify estrus, lameness, digestive disorders, and feeding aggression.",
        "• 5 Monitored States: Standing, Lying, Feeding, Drinking, and Walking.",
        "• Spatio-Temporal Nature: Static posture (Standing vs Lying) can be inferred from single frames; dynamic locomotion (Walking) and ingestion (Feeding vs Drinking) demand multi-frame temporal modeling (T=8 frames).",
        "• Key Challenge: Extreme class imbalance; minority locomotion (Walking) is easily overwhelmed by static feeding and resting."
    ], font_size=11)

    add_card(s2, c3_left, Inches(1.85), col_w, Inches(4.9), "Cattle Re-Identification (Re-ID)", COLOR_ACCENT_AMBER)
    populate_card_text(s2, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Biometric Goal: Tracking individual cattle across non-overlapping surveillance cameras without ear tag readers or invasive markers.",
        "• Setting: Open-Set Biometric Retrieval. Matches unseen query cattle against a gallery of registered parlor images.",
        "• Visual Features: Distinctive Holstein coat spot patterns, facial pigmentation, and bodily aspect ratios.",
        "• Key Challenge: Robustness under severe viewpoint and acquisition shifts (stationary overhead barn CCTV vs mobile handheld pasture snapshots)."
    ], font_size=11)

    add_footer(s2, 2)

    # =========================================================================
    # SLIDE 3: RESEARCH PROBLEM
    # =========================================================================
    s3 = add_blank_slide()
    add_header(s3, 3, "Problem Statement", "The Core Dilemmas in Visual Multi-Task Cattle Monitoring",
               "Why naive vision models and conventional multi-task architectures fail in real-world livestock settings")

    c2_w = Inches(5.69)
    c_left1 = Inches(0.8)
    c_left2 = c_left1 + c2_w + Inches(0.35)

    add_card(s3, c_left1, Inches(1.85), c2_w, Inches(4.9), "1. Representational Conflict Across Tasks", COLOR_ACCENT_ROSE)
    populate_card_text(s3, c_left1 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Fundamental Visual Incompatibility:",
        "  - BCS requires fine skeletal morphology and dorsal body curvature; it MUST ignore coat patterns and color markings.",
        "  - Re-ID requires unique coat textures, spots, and surface markings; it MUST remain invariant to posture changes (standing vs lying).",
        "  - Behavior requires kinematic temporal transitions over time; it is largely independent of individual identity markings.",
        "• The Multi-Task Bottleneck: Forcing all three tasks to share a single unconstrained feature extractor creates representation tension: features optimized for Re-ID act as distracting noise for BCS!"
    ], font_size=11.5)

    add_card(s3, c_left2, Inches(1.85), c2_w, Inches(4.9), "2. Environmental Confounding & Negative Transfer", COLOR_ACCENT_AMBER)
    populate_card_text(s3, c_left2 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Background Shortcut Learning:",
        "  - Generic RGB models easily latch onto pen stanchions, feeder geometry, flooring, and lighting shadows rather than the animal.",
        "  - A model may 'predict' Feeding simply because feed bunk metal bars are visible in the background, failing when camera angles change.",
        "• Severe Negative Transfer in Hard Sharing:",
        "  - In standard monolithic multi-task networks, gradient updates from one task overwrite shared weights needed by another.",
        "  - Result: Naive multi-task models perform significantly WORSE than dedicated single-task models across all primary metrics."
    ], font_size=11.5)

    add_footer(s3, 3)

    # =========================================================================
    # SLIDE 4: RESEARCH GAP & THESIS FOCUS
    # =========================================================================
    s4 = add_blank_slide()
    add_header(s4, 4, "Research Gap & Scope", "Research Gaps in Literature and The Thesis Focus",
               "Addressing critical methodological flaws in existing precision livestock vision literature")

    add_card(s4, c_left1, Inches(1.85), c2_w, Inches(4.9), "Identified Research Gaps in Literature", COLOR_ACCENT_ROSE)
    populate_card_text(s4, c_left1 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Pervasive Data Leakage in Livestock Benchmarks:",
        "  - Literature frequently splits high-frame-rate video bursts randomly into train/test, evaluating models on near-identical frames (99%+ artificial accuracy).",
        "• Siloed Single-Task Research:",
        "  - BCS, behavior, and Re-ID are studied in strict isolation using incompatible datasets, preventing unified edge integration.",
        "• Blind Multi-Task Assumptions:",
        "  - Prior multi-task papers assume positive transfer occurs automatically, without quantifying negative transfer or gradient conflict.",
        "• Lack of Controlled Attribution:",
        "  - Existing works change architectures, augmentations, and backbones simultaneously, making it impossible to attribute gains to specific design choices."
    ], font_size=11)

    add_card(s4, c_left2, Inches(1.85), c2_w, Inches(4.9), "Thesis Focus & Methodological Rigor", COLOR_ACCENT_EMERALD)
    populate_card_text(s4, c_left2 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Strictly Leakage-Free Disjoint Evaluation:",
        "  - Burst-group-disjoint protocol for ScienceDB (BCS); session-disjoint for Behavior; open-set 69-cow unseen split for Re-ID.",
        "• Cattle-Centered Representation Learning:",
        "  - Isolating the animal via automated detection (RT-DETR-L) and foreground mask guidance (4-channel input) to strip background shortcuts.",
        "• Controlled Architectural & Optimization Progression:",
        "  - Single-Task Baselines (E0) -> Monolithic Hard-Shared (E1) -> Modular Task-Private Adapters (E3) -> PCGrad Gradient Projection (E4).",
        "• Direct Gradient Conflict Diagnostics:",
        "  - Tracking 44,177 gradient conflicts across 16,140 super-steps to directly observe parameter tension."
    ], font_size=11)

    add_footer(s4, 4)

    # =========================================================================
    # SLIDE 5: RESEARCH QUESTIONS
    # =========================================================================
    s5 = add_blank_slide()
    add_header(s5, 5, "Scientific Inquiry", "Research Questions (RQs)",
               "Three formal research questions governing representation, task-specific cues, and multi-task sharing")

    # 3 horizontal full-width cards
    h_card_h = Inches(1.48)
    gap_h = Inches(0.18)
    top_base = Inches(1.85)

    # RQ1
    add_card(s5, Inches(0.8), top_base, Inches(11.733), h_card_h, "Research Question 1 (Representation Impact)", COLOR_ACCENT_BLUE)
    populate_card_text(s5, Inches(1.05), top_base + Inches(0.55), Inches(11.2), Inches(0.8), [
        "How do cattle-centered visual representations (localized crops and foreground mask guidance) affect body condition scoring, behavior recognition, and individual re-identification relative to their generic RGB single-task references on leakage-free matched populations?"
    ], font_size=12)

    # RQ2
    add_card(s5, Inches(0.8), top_base + h_card_h + gap_h, Inches(11.733), h_card_h, "Research Question 2 (Task-Specific Cue Preservation)", COLOR_ACCENT_TEAL)
    populate_card_text(s5, Inches(1.05), top_base + h_card_h + gap_h + Inches(0.55), Inches(11.2), Inches(0.8), [
        "How can temporal information and task-specific visual cues (anatomy, pose, viewpoint) be incorporated into downstream models while preserving the mutually conflicting requirements of condition assessment, activity recognition, and identity matching?"
    ], font_size=12)

    # RQ3
    add_card(s5, Inches(0.8), top_base + (h_card_h + gap_h)*2, Inches(11.733), h_card_h, "Research Question 3 (Multi-Task Sharing & Negative Transfer)", COLOR_ACCENT_AMBER)
    populate_card_text(s5, Inches(1.05), top_base + (h_card_h + gap_h)*2 + Inches(0.55), Inches(11.2), Inches(0.8), [
        "How does task-conditioned modular sharing and gradient-projected optimization compare with monolithic hard parameter sharing in terms of task accuracy, negative transfer mitigation, and cross-domain robustness?"
    ], font_size=12)

    add_footer(s5, 5)

    # =========================================================================
    # SLIDE 6: OVERALL THESIS FRAMEWORK
    # =========================================================================
    s6 = add_blank_slide()
    add_header(s6, 6, "Research Architecture", "Overall Thesis Experimental Framework",
               "A 5-phase systematic methodology from raw data audit to unified multi-task optimization")

    step_w = Inches(2.18)
    step_gap = Inches(0.2)
    step_top = Inches(1.85)

    steps = [
        ("Phase 1: Leakage Audit", COLOR_ACCENT_BLUE, [
            "• ScienceDB: Burst-group disjoint clustering.",
            "• CVB + Beef: Session and camera isolation.",
            "• SideView: Protocol A (41 train vs 69 held-out cows)."
        ]),
        ("Phase 2: RGB Baselines", COLOR_CARD_BORDER_ACCENT, [
            "• Run 1: BCS RGB ResNet-18 (MAE: 0.1929).",
            "• Run 2: Single-frame Behavior (Acc: 88.46%).",
            "• Run 3: RGB Re-ID baseline (mAP: 27.05%)."
        ]),
        ("Phase 3: Perception", COLOR_ACCENT_TEAL, [
            "• RT-DETR-L cattle localization (COCO cls 19).",
            "• SAM 2.1-S foreground segmentation.",
            "• 4-Channel input: [B, 4, 224, 224] (RGB + Mask)."
        ]),
        ("Phase 4: Single Models", COLOR_ACCENT_AMBER, [
            "• Run 4: BCS Ordinal Head (MAE: 0.1709).",
            "• Run 5: Behavior 1D-TCN (Bal Acc: 74.43%).",
            "• Run 6: Re-ID Oracle (Rank-1: 62.93%)."
        ]),
        ("Phase 5: Unified MTL", COLOR_ACCENT_ROSE, [
            "• Run 7 (E1): Monolithic Hard Sharing.",
            "• Run 8 (E3): Modular Task Adapters.",
            "• Phase 3 (E4): PCGrad Gradient Surgery."
        ])
    ]

    for idx, (stitle, scolor, sbullets) in enumerate(steps):
        sleft = Inches(0.8) + idx * (step_w + step_gap)
        add_card(s6, sleft, step_top, step_w, Inches(4.9), stitle, scolor)
        populate_card_text(s6, sleft + Inches(0.15), step_top + Inches(0.65), step_w - Inches(0.3), Inches(4.0), sbullets, font_size=10.5)

    add_footer(s6, 6)

    # =========================================================================
    # SLIDE 7: DATASETS AND SPLITS
    # =========================================================================
    s7 = add_blank_slide()
    add_header(s7, 7, "Benchmark Data & Protocols", "Primary Datasets & Leakage-Free Split Protocols",
               "Establishing rigorous, reproducible train/val/test partitions across all three domains")

    add_card(s7, c1_left, Inches(1.85), col_w, Inches(4.9), "ScienceDB (BCS)", COLOR_ACCENT_BLUE)
    populate_card_text(s7, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Domain: Body Condition Scoring (Dairy Cows).",
        "• Acquisition: Automated overhead walk-through chute at dairy milking parlor exit.",
        "• Total Volume: 49,735 top-view depth/RGB images.",
        "• Leakage Repair Protocol: Burst-Group-Disjoint. Clusters consecutive frames within 1.5s passage bursts into atomic units.",
        "• Split Breakdown: 34,369 train / 7,817 val / 7,549 test images (successful perception subset).",
        "• Boundary: Passage-sequence safe; not biological cow-disjoint (no individual cow IDs provided)."
    ], font_size=11)

    add_card(s7, c2_left, Inches(1.85), col_w, Inches(4.9), "CVB + Kaggle Beef (Behavior)", COLOR_ACCENT_TEAL)
    populate_card_text(s7, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Domain: 5-Class Cattle Behavior Recognition.",
        "• Dual Source Composition:",
        "  - CVB: Multi-cow commercial barn CCTV video footage (Standing, Lying, Feeding, Walking).",
        "  - Kaggle Beef: Single-animal pasture & feedlot video clips (Standing, Lying, Feeding, Drinking).",
        "• Volume: 4,271 retained train/val sequences, 780 retained test sequences (T=8 frames at 224x224).",
        "• Protocol: Video-recording and session-disjoint.",
        "• Boundary: Walking behavior is exclusive to CVB."
    ], font_size=11)

    add_card(s7, c3_left, Inches(1.85), col_w, Inches(4.9), "SideViewCows2026 (Re-ID)", COLOR_ACCENT_AMBER)
    populate_card_text(s7, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Domain: Open-Set Biometric Re-Identification.",
        "• Volume: 80,260 RGB images + 80,260 binary masks across 110 biological cows.",
        "• 3 Camera Domains: Milking parlor (gallery), stationary barn CCTV (query), mobile handheld snapshots (query).",
        "• Protocol A (Strict Open-Set):",
        "  - 41 cows in training/validation (15,436 images).",
        "  - 69 held-out cows strictly reserved for evaluation against 36,811 parlor gallery images (0 identity overlap)."
    ], font_size=11)

    add_footer(s7, 7)

    # =========================================================================
    # SLIDE 8: REPRESENTATION DESIGN
    # =========================================================================
    s8 = add_blank_slide()
    add_header(s8, 8, "Cattle-Centered Representation", "Representation Design: 4-Channel Spatial Tensor",
               "Eliminating environmental shortcuts via synchronized bounding-box framing and mask fusion")

    add_card(s8, c_left1, Inches(1.85), c2_w, Inches(4.9), "The 4-Channel Input Pipeline: [B, 4, 224, 224]", COLOR_ACCENT_BLUE)
    populate_card_text(s8, c_left1 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Architectural Input Specification:",
        "  - Channels 0, 1, 2: Standard ImageNet-normalized RGB image of the cropped animal.",
        "  - Channel 3: Synchronized binary segmentation mask {0.0, 1.0} of the animal foreground.",
        "• Bounding-Box Framing with 5% Margin:",
        "  - Derived from target cow bounding box with an added deterministic 5% margin to preserve peripheral anatomical contours.",
        "• ResNet-18 Conv1 Adaptation (+3,136 Parameters):",
        "  - First convolutional layer modified from `Conv2d(3, 64)` to `Conv2d(4, 64)`.",
        "  - First 3 channel weights initialized from ImageNet pre-training; 4th channel initialized randomly, allowing the model to learn joint RGB-geometry interactions."
    ], font_size=11.5)

    add_card(s8, c_left2, Inches(1.85), c2_w, Inches(4.9), "Why 4-Channel Concatenation Outperforms Masking", COLOR_ACCENT_EMERALD)
    populate_card_text(s8, c_left2 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• The Pitfall of Hard Background Blacking (Zero-Masking):",
        "  - Setting background pixels to pure black creates sharp, artificial high-frequency edge gradients at animal boundaries that corrupt early convolution filters.",
        "• Advantages of 4-Channel Representation:",
        "  - Preserves subtle real-world boundary transitions while explicitly informing the network which pixels belong to the target animal.",
        "  - Strips Background Shortcuts: Prevents the network from associating barn stalls, feeders, or milking equipment with specific health or behavior states.",
        "  - Multi-Task Alignment: Provides a standardized, animal-centered visual coordinate system for BCS, Behavior, and Re-ID."
    ], font_size=11.5)

    add_footer(s8, 8)

    # =========================================================================
    # SLIDE 9: CATTLE PERCEPTION PIPELINE
    # =========================================================================
    s9 = add_blank_slide()
    add_header(s9, 9, "Upstream Perception Pipeline", "Cattle Localization, Segmentation & Retention Boundaries",
               "Automating cattle extraction: detection, segmentation, and sample attrition boundaries")

    add_card(s9, c1_left, Inches(1.85), col_w, Inches(4.9), "1. Detection: RT-DETR-L", COLOR_ACCENT_BLUE)
    populate_card_text(s9, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Model: Real-Time DEtection TRansformer (RT-DETR-L; 66.5M params).",
        "• Class Conditioning: COCO `class 19` (cow) with detection confidence threshold = 0.25.",
        "• Role: Generates tight bounding boxes around cattle across diverse camera resolutions and farm environments.",
        "• Execution: Batched inference on cloud GPUs with monotonic frame decoding."
    ], font_size=11)

    add_card(s9, c2_left, Inches(1.85), col_w, Inches(4.9), "2. Segmentation: SAM 2.1-S", COLOR_ACCENT_TEAL)
    populate_card_text(s9, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Model: Segment Anything Model 2.1 Small (SAM 2.1-S; 92.3M params).",
        "• Prompting Strategy: Bounding box prompt derived directly from RT-DETR-L output.",
        "• Output: High-fidelity binary foreground mask isolating animal torso, legs, and head.",
        "• Fast Caching: Monotonic video pass on NVIDIA L40S yielded 100% exact mask equality (IoU = 1.000000; 180 candidates/min)."
    ], font_size=11)

    add_card(s9, c3_left, Inches(1.85), col_w, Inches(4.9), "3. Perception Retention Boundaries", COLOR_ACCENT_AMBER)
    populate_card_text(s9, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• ScienceDB BCS Retention:",
        "  - 7,549 of 8,040 test images successfully parsed (93.89% coverage; 489 detection misses + 2 empty masks excluded).",
        "• Behavior Video Retention:",
        "  - 780 of 809 test sequences retained (96.42% coverage; 29 sequences excluded due to severe stall stanchion occlusion).",
        "• SideViewCows2026 Re-ID:",
        "  - Evaluated with released ground-truth oracle masks (100% coverage; oracle benchmark)."
    ], font_size=11)

    add_footer(s9, 9)

    # =========================================================================
    # SLIDE 10: VIEWPOINT INVESTIGATION
    # =========================================================================
    s10 = add_blank_slide()
    add_header(s10, 10, "Viewpoint Investigation", "Viewpoint Classification & Cross-Domain Shortcut Risk",
               "Why viewpoint conditioning was audited and ultimately excluded from the primary multi-task pipeline")

    add_card(s10, c_left1, Inches(1.85), c2_w, Inches(4.9), "Viewpoint Classifier: In-Domain Benchmark", COLOR_ACCENT_BLUE)
    populate_card_text(s10, c_left1 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Auxiliary Viewpoint Architecture:",
        "  - ResNet-18 classifier trained to predict 3 canonical viewpoints: Front, Side, and Rear.",
        "  - Trained on curated real-cattle image dataset (`viewpoint_resnet18_real_best.pth`).",
        "• In-Domain Test Performance:",
        "  - Achieved 86.26% Accuracy and 0.8573 Macro-F1 across 131 held-out test images.",
        "  - Proved high feasibility for viewpoint categorization under controlled single-source conditions."
    ], font_size=11.5)

    add_card(s10, c_left2, Inches(1.85), c2_w, Inches(4.9), "The Cross-Domain Shortcut Discovery (TVD = 0.7646)", COLOR_ACCENT_ROSE)
    populate_card_text(s10, c_left2 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Cross-Domain Feasibility Audit on Behavior Stack:",
        "  - Audited on 54 stratified CVB + Beef sequences (432 frames).",
        "• Severe Dataset-Camera Confounding:",
        "  - CVB (Barn CCTV) predicted: 7.9% Front, 70.8% Side, 21.2% Rear.",
        "  - Kaggle Beef (Pasture/Feedlot) predicted: 84.4% Front, 8.8% Side, 6.8% Rear.",
        "  - Total Variation Distance = 0.7646 between the two sources!",
        "• Scientific Verdict:",
        "  - Viewpoint predictions functioned as a surrogate for camera setup and dataset origin rather than true cow pose.",
        "  - High risk of shortcut learning; rightly EXCLUDED from the primary multi-task pipeline."
    ], font_size=11.5)

    add_footer(s10, 10)

    # =========================================================================
    # SLIDE 11: POSE AND ANATOMY FEASIBILITY
    # =========================================================================
    s11 = add_blank_slide()
    add_header(s11, 11, "Anatomical Feasibility Audit", "Pose & Anatomical Keypoint Feasibility Audit",
               "Forensic evaluation of DeepLabCut SuperAnimal-Quadruped on commercial livestock footage")

    add_card(s11, c_left1, Inches(1.85), c2_w, Inches(4.9), "SuperAnimal-Quadruped Model Evaluation", COLOR_ACCENT_BLUE)
    populate_card_text(s11, c_left1 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Model Tested: DeepLabCut SuperAnimal-Quadruped (ResNet-50, 39 keypoints).",
        "• Audit Setup: Evaluated across 54 representative sequences (432 frames) from the authentic CVB + Beef behavior dataset.",
        "• Severe Keypoint Missingness:",
        "  - Overall Frame Return Rate: Only 54.40% (45.60% detection failure rate).",
        "  - Sequence-Level Tracking: Only 31.48% of sequences (17/54) had all 8 frames successfully detected.",
        "  - Mean Keypoint Confidence: 0.4658 across valid returns."
    ], font_size=11.5)

    add_card(s11, c_left2, Inches(1.85), c2_w, Inches(4.9), "Catastrophic Failure Modes & Thesis Verdict", COLOR_ACCENT_ROSE)
    populate_card_text(s11, c_left2 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Catastrophic Collapse on Lying Cattle:",
        "  - Detection Failure Rate: 72.92% on recumbent/lying cattle.",
        "  - Complete Sequence Rate: 0.0% (0 of 12 lying sequences had all 8 frames).",
        "  - Keypoints completely collapsed when legs were tucked beneath the torso.",
        "• Extreme Temporal Jitter: 880 displacement spikes (>0.25 normalized jump); only 1.85% of sequences met tracking stability criteria.",
        "• BCS Incompatibility: Completely lacks rear pelvic landmarks (hooks, pins, tailhead) essential for BCS.",
        "• Verdict: Off-the-shelf animal pose is fundamentally unsuitable for real-world cattle monitoring without species retraining. EXCLUDED from primary MTL."
    ], font_size=11.5)

    add_footer(s11, 11)

    # =========================================================================
    # SLIDE 12: TASK-SPECIFIC MODEL DESIGN
    # =========================================================================
    s12 = add_blank_slide()
    add_header(s12, 12, "Task-Specific Architectures", "Downstream Prediction Heads & Loss Formulations",
               "Tailoring model heads to the specific mathematical and biological structure of each task")

    add_card(s12, c1_left, Inches(1.85), col_w, Inches(4.9), "Task 1: BCS Ordinal Head", COLOR_ACCENT_BLUE)
    populate_card_text(s12, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Frank & Hall (2001) Cumulative Ordinal Formulation:",
        "  - Predicts K-1 = 4 binary indicators: P(BCS > 2.75), P(BCS > 3.00), P(BCS > 3.25), P(BCS > 3.50).",
        "• Loss Function: Multi-label Binary Cross-Entropy.",
        "• Prediction Decoding: Final score = 2.75 + 0.25 * sum(sigmoid(logits)).",
        "• Advantage: Respects the strict ordered rank topology of body condition; penalizes distant errors more than adjacent category confusion."
    ], font_size=11)

    add_card(s12, c2_left, Inches(1.85), col_w, Inches(4.9), "Task 2: Behavior 1D-TCN Head", COLOR_ACCENT_TEAL)
    populate_card_text(s2, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• 1D Temporal Convolutional Network:",
        "  - Input: Sequence of T=8 frame embeddings (each 512-D from ResNet-18 spatial backbone).",
        "  - Architecture: 2 dilated residual TCN blocks (hidden=256, kernel=3, dropout=0.2) + Global Average Pooling + Linear(256, 5).",
        "• Loss Function: Cross-Entropy with class weighting.",
        "• Advantage: Captures multi-frame locomotion and ingestion dynamics without the vanishing gradients or sequential bottlenecks of RNNs."
    ], font_size=11)

    add_card(s12, c3_left, Inches(1.85), col_w, Inches(4.9), "Task 3: Re-ID Metric Head", COLOR_ACCENT_AMBER)
    populate_card_text(s12, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Metric Learning Classification:",
        "  - Training Head: Linear(512, 41) classifier trained with Cross-Entropy over the 41 training identities.",
        "  - Feature Representation: 512-D pooled spatial feature vector normalized to unit L2 sphere.",
        "• Open-Set Retrieval Inference:",
        "  - Extracts 512-D unit vectors for all 69 unseen evaluation cows.",
        "  - Computes Cosine Distance against the 36,811 parlor gallery images to generate cumulative match curves."
    ], font_size=11)

    add_footer(s12, 12)

    # =========================================================================
    # SLIDE 13: BCS — SINGLE-TASK RESULTS
    # =========================================================================
    s13 = add_blank_slide()
    add_header(s13, 13, "Empirical Results: BCS", "Body Condition Scoring: Single-Task Results",
               "Cattle-centered perception significantly improves physical error and tolerance accuracy")

    # Table on Left, Summary Card on Right
    tbl_w = Inches(6.8)
    tbl_h = Inches(4.8)
    
    # Add Table Shape
    rows = 7
    cols = 4
    table_shape = s13.shapes.add_table(rows, cols, Inches(0.8), Inches(1.9), tbl_w, tbl_h)
    tbl = table_shape.table
    tbl.columns[0].width = Inches(2.6)
    tbl.columns[1].width = Inches(1.4)
    tbl.columns[2].width = Inches(1.4)
    tbl.columns[3].width = Inches(1.4)

    headers = ["Evaluation Metric", "Run 1 (RGB)", "Run 4 (Percept.)", "Delta"]
    for c_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    bcs_data = [
        ("Real MAE (BCS units)", "0.1929", "0.1709", "-0.0220 (-11.4%)"),
        ("Acc@1 (+/- 0.25 units)", "84.95%", "89.40%", "+4.45 pp"),
        ("Acc@0 (Exact Match)", "40.84%", "43.57%", "+2.73 pp"),
        ("Balanced Accuracy", "40.23%", "39.70%", "-0.53 pp"),
        ("Macro-F1 Score", "0.4074", "0.4039", "-0.0035"),
        ("Test BCE Loss", "0.8224", "0.4403", "-46.5% reduction")
    ]

    for r_idx, row in enumerate(bcs_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_MUTED_BG if r_idx % 2 == 1 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10.5)
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if c_idx == 3:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT_EMERALD if "-" in val and "Loss" in row[0] or "+" in val else (COLOR_ACCENT_EMERALD if "-" in val and "MAE" in row[0] else COLOR_TEXT_SECONDARY)

    # Right Card: Scientific Analysis
    rc_left = Inches(7.85)
    rc_w = Inches(4.68)
    add_card(s13, rc_left, Inches(1.9), rc_w, tbl_h, "Key Findings & Insights", COLOR_ACCENT_BLUE)
    populate_card_text(s13, rc_left + Inches(0.25), Inches(2.6), rc_w - Inches(0.5), Inches(3.9), [
        "• Evaluated Population: Exactly matched 7,549 ScienceDB test images with verified automated perception (burst-group-disjoint).",
        "• Outstanding Physical Proximity: Real MAE dropped to 0.1709 BCS units, well within clinical tolerance thresholds.",
        "• Over 89.4% Practical Utility: Nearly 9 out of 10 cows predicted within +/- 0.25 BCS units.",
        "• Ordinal vs Balanced Split: Localized cropping and masks tightened continuous error around true boundaries without altering discrete class distribution balance.",
        "• Sample Attrition Note: Findings strictly describe the 93.89% of instances where upstream perception succeeded."
    ], font_size=11)

    add_footer(s13, 13)

    # =========================================================================
    # SLIDE 14: BEHAVIOR — SINGLE-TASK RESULTS
    # =========================================================================
    s14 = add_blank_slide()
    add_header(s14, 14, "Empirical Results: Behavior", "Behavior Recognition: Single-Task Results",
               "Temporal modeling and foreground masks elevate balanced accuracy and minority detection")

    table_shape2 = s14.shapes.add_table(8, 4, Inches(0.8), Inches(1.9), tbl_w, tbl_h)
    tbl2 = table_shape2.table
    tbl2.columns[0].width = Inches(2.6)
    tbl2.columns[1].width = Inches(1.4)
    tbl2.columns[2].width = Inches(1.4)
    tbl2.columns[3].width = Inches(1.4)

    headers2 = ["Evaluation Metric", "Run 2 (RGB 1-Fr)", "Run 5 (TCN 8-Fr)", "Delta"]
    for c_idx, h_text in enumerate(headers2):
        cell = tbl2.cell(0, c_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    beh_data = [
        ("Balanced Accuracy", "71.30%", "74.43%", "+3.13 pp [WIN]"),
        ("Test Cross-Entropy Loss", "0.5505", "0.4430", "-19.5% reduction"),
        ("Walking F1 (Minority, CVB)", "0.2174", "0.2456", "+13.0% relative"),
        ("Kaggle Beef Accuracy", "93.58%", "96.09%", "+2.51 pp [WIN]"),
        ("Kaggle Beef Macro-F1", "0.9079", "0.9414", "+0.0335 [WIN]"),
        ("Overall Accuracy", "88.46%", "87.44%", "-1.02 pp"),
        ("CVB Barn Accuracy", "84.12%", "80.09%", "-4.03 pp")
    ]

    for r_idx, row in enumerate(beh_data):
        for c_idx, val in enumerate(row):
            cell = tbl2.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_MUTED_BG if r_idx % 2 == 1 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10.5)
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if c_idx == 3:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT_EMERALD if "[WIN]" in val or "reduction" in val else COLOR_TEXT_SECONDARY

    add_card(s14, rc_left, Inches(1.9), rc_w, tbl_h, "Key Findings & Insights", COLOR_ACCENT_TEAL)
    populate_card_text(s14, rc_left + Inches(0.25), Inches(2.6), rc_w - Inches(0.5), Inches(3.9), [
        "• Matched Population: 780 retained video sequences (T=8 frames, 6,240 total frames; CVB N=422, Beef N=358).",
        "• Resolving Dynamic Ambiguity: TCN sequence modeling successfully disambiguates static standing from active locomotion.",
        "• Significant Minority Boost: Walking class F1 rose from 0.2174 to 0.2456 (+13% relative gain).",
        "• Cross-Environment Divergence: Strong gains on open pasture (Kaggle Beef: 96.09% Acc) accompanied by a slight drop on dense barn CCTV (CVB: 80.09%), highlighting domain trade-offs.",
        "• Joint Attribution Boundary: Gains reflect combined impact of foreground masks and 1D temporal convolution."
    ], font_size=11)

    add_footer(s14, 14)

    # =========================================================================
    # SLIDE 15: RE-ID — SINGLE-TASK RESULTS
    # =========================================================================
    s15 = add_blank_slide()
    add_header(s15, 15, "Empirical Results: Re-ID", "Cattle Re-Identification: Single-Task Results",
               "Target-centered foreground isolation delivers massive retrieval gains under extreme acquisition shifts")

    table_shape3 = s15.shapes.add_table(6, 4, Inches(0.8), Inches(1.9), tbl_w, tbl_h)
    tbl3 = table_shape3.table
    tbl3.columns[0].width = Inches(2.8)
    tbl3.columns[1].width = Inches(1.3)
    tbl3.columns[2].width = Inches(1.3)
    tbl3.columns[3].width = Inches(1.4)

    headers3 = ["Protocol A Setting & Metric", "Run 3 (RGB)", "Run 6 (Oracle)", "Delta"]
    for c_idx, h_text in enumerate(headers3):
        cell = tbl3.cell(0, c_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    reid_data = [
        ("Snapshots -> Parlor Rank-1", "38.88%", "62.93%", "+24.05 pp (+61.9%)"),
        ("Snapshots -> Parlor Rank-5", "57.17%", "75.29%", "+18.12 pp"),
        ("Snapshots -> Parlor mAP", "27.05%", "40.42%", "+13.37 pp (+49.4%)"),
        ("Barn -> Parlor Rank-1", "58.64%", "63.90%", "+5.26 pp"),
        ("Barn -> Parlor mAP", "38.32%", "40.68%", "+2.36 pp")
    ]

    for r_idx, row in enumerate(reid_data):
        for c_idx, val in enumerate(row):
            cell = tbl3.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_MUTED_BG if r_idx % 2 == 1 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10.5)
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if c_idx == 3:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT_EMERALD

    add_card(s15, rc_left, Inches(1.9), rc_w, tbl_h, "Key Findings & Insights", COLOR_ACCENT_AMBER)
    populate_card_text(s15, rc_left + Inches(0.25), Inches(2.6), rc_w - Inches(0.5), Inches(3.9), [
        "• Benchmark Scope: Protocol A across 69 strictly unseen cows evaluated against 36,811 parlor gallery images.",
        "• Breakthrough on Handheld Snapshots: Rank-1 retrieval surged from 38.88% to 62.93% (+24.05 pp absolute gain; +61.9% relative boost).",
        "• mAP Expansion: Mean Average Precision increased from 27.05% to 40.42% (+13.37 pp gain).",
        "• Physical Mechanism: Eliminating background barn and pasture clutter forces the metric embedding to encode true biological coat spot patterns.",
        "• Oracle Scope Boundary: Evaluated with released ground-truth masks; establishes upper-bound target isolation ceiling."
    ], font_size=11)

    add_footer(s15, 15)

    # =========================================================================
    # SLIDE 16: UNIFIED MULTI-TASK FRAMEWORK
    # =========================================================================
    s16 = add_blank_slide()
    add_header(s16, 16, "Multi-Task Framework", "Unified Multi-Task Architectures: E1, E3, and E4",
               "Comparing Monolithic Hard Sharing, Task-Private Residual Adapters, and PCGrad Optimization")

    add_card(s16, c1_left, Inches(1.85), col_w, Inches(4.9), "E1: Monolithic Hard Sharing", COLOR_ACCENT_BLUE)
    populate_card_text(s16, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Paradigm: 100% Shared Spatial Backbone.",
        "• Trunk: Exactly ONE 4-channel ResNet-18 (11,179,648 parameters; 93.74% shared capacity).",
        "• Task Heads: Dedicated BCS Ordinal (2K params), Behavior TCN (724K params), and Re-ID Linear(512, 41) (21K params).",
        "• Total Trainable Parameters: 11,926,706.",
        "• Training Schedule: Alternating super-step batches with equal task weighting (1.0, 1.0, 1.0) and unconstrained gradient summation."
    ], font_size=11)

    add_card(s16, c2_left, Inches(1.85), col_w, Inches(4.9), "E3: Modular Task Adapters", COLOR_ACCENT_TEAL)
    populate_card_text(s16, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Paradigm: Architectural Task Specialization.",
        "• Shared Trunk: Same 4-channel ResNet-18.",
        "• Task-Private Adapters: Three dedicated residual bottleneck blocks (512 -> 128 -> 512; 131,968 params each = 395,904 total; +3.32% capacity).",
        "• Initialization: Zero-weight up-projection ensures each adapter acts as identity mapping at epoch 0.",
        "• Total Trainable Parameters: 12,322,610.",
        "• Hypothesis: Private bottleneck relieves representational competition."
    ], font_size=11)

    add_card(s16, c3_left, Inches(1.85), col_w, Inches(4.9), "E4: PCGrad Optimization Control", COLOR_ACCENT_ROSE)
    populate_card_text(s16, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Paradigm: Optimization-Level Surgery.",
        "• Parameter Parity: EXACT E1 architecture match (11,926,706 params; 0 adapters, 0 gates).",
        "• Algorithm: Projecting Conflicting Gradients (PCGrad; Yu et al., 2020).",
        "• Mechanism: If cosine(g_i, g_j) < 0 on shared trunk, projects g_i orthogonal to g_j, eliminating destructive gradient cancellation.",
        "• Controlled Design: Tests whether training-time conflict resolution solves negative transfer without adding parameters."
    ], font_size=11)

    add_footer(s16, 16)

    # =========================================================================
    # SLIDE 17: MULTI-TASK RESULTS
    # =========================================================================
    s17 = add_blank_slide()
    add_header(s17, 17, "Multi-Task Empirical Evaluation", "Official Held-Out Multi-Task Results (E0 vs E1 vs E3 vs E4)",
               "Direct empirical proof of persistent negative transfer and selective metric-dependent mitigation")

    # Master Table spanning width
    mtl_tbl_w = Inches(11.733)
    mtl_table_shape = s17.shapes.add_table(11, 7, Inches(0.8), Inches(1.9), mtl_tbl_w, Inches(4.8))
    m_tbl = mtl_table_shape.table
    m_tbl.columns[0].width = Inches(1.6)
    m_tbl.columns[1].width = Inches(2.4)
    m_tbl.columns[2].width = Inches(1.5)
    m_tbl.columns[3].width = Inches(1.5)
    m_tbl.columns[4].width = Inches(1.5)
    m_tbl.columns[5].width = Inches(1.5)
    m_tbl.columns[6].width = Inches(1.733)

    m_headers = ["Domain", "Evaluation Metric", "Single (E0)", "Hard (E1)", "Modular (E3)", "PCGrad (E4)", "Synthesis Verdict"]
    for c_idx, h_text in enumerate(m_headers):
        cell = m_tbl.cell(0, c_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER if c_idx > 1 else PP_ALIGN.LEFT

    mtl_master_data = [
        ("BCS", "Real MAE (BCS units)", "0.1709", "0.1788", "0.1916", "0.1828", "Single E0 Wins (Neg. Transfer)"),
        ("BCS", "Acc@1 (+/- 0.25 units)", "89.40%", "88.44%", "86.32%", "87.84%", "Single E0 Wins"),
        ("Behavior", "Overall Accuracy", "87.44%", "85.00%", "85.77%", "86.92%", "E4 +1.92 pp over E1"),
        ("Behavior", "Balanced Accuracy", "74.43%", "67.30%", "66.70%", "69.15%", "E4 +1.85 pp over E1"),
        ("Behavior", "Macro-F1 Score", "0.7397", "0.6866", "0.6755", "0.7114", "E4 +0.0248 over E1"),
        ("Behavior", "Walking F1 (Minority)", "0.2456", "0.0408", "0.0000", "0.0909", "E4 more than doubles E1"),
        ("Behavior", "CVB Barn Accuracy", "80.09%", "76.78%", "80.33%", "81.04%", "E4 Beats Single (+0.95 pp)"),
        ("Re-ID", "Barn -> Parlor Rank-1", "63.90%", "57.38%", "49.08%", "54.97%", "Single E0 Wins (-8.93 pp)"),
        ("Re-ID", "Barn -> Parlor mAP", "40.68%", "30.37%", "28.04%", "30.23%", "Single E0 Wins (-10.45 pp)"),
        ("Re-ID", "Snapshot -> Parlor Rank-1", "62.93%", "57.17%", "53.71%", "57.00%", "Single E0 Wins (-5.93 pp)")
    ]

    for r_idx, row in enumerate(mtl_master_data):
        for c_idx, val in enumerate(row):
            cell = m_tbl.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_MUTED_BG if r_idx % 2 == 1 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(9.5)
            p.alignment = PP_ALIGN.CENTER if c_idx >= 2 and c_idx <= 5 else PP_ALIGN.LEFT
            if c_idx == 2:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT_BLUE
            elif c_idx == 5:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT_ROSE
            elif c_idx == 6:
                p.font.bold = True
                p.font.color.rgb = COLOR_TEXT_PRIMARY

    add_footer(s17, 17)

    # =========================================================================
    # SLIDE 18: GRADIENT CONFLICT DIAGNOSTICS
    # =========================================================================
    s18 = add_blank_slide()
    add_header(s18, 18, "Optimization Diagnostics", "Direct Empirical Proof of Shared-Backbone Gradient Conflict",
               "Tracking 44,177 conflict projections across 16,140 super-steps on NVIDIA L40S")

    add_card(s18, c1_left, Inches(1.85), col_w, Inches(4.9), "Diagnostic Metric Totals", COLOR_ACCENT_ROSE)
    populate_card_text(s18, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Total Optimization Scope:",
        "  - 30 complete training epochs.",
        "  - 16,140 total super-steps (538 steps/epoch).",
        "• Massive Conflict Volume:",
        "  - 44,177 pairwise gradient conflict projections executed on the shared trunk.",
        "  - Mean Rate: 2.737 conflict projections per super-step (out of theoretical max of 6).",
        "• Continuous Tension: Gradient conflict was chronic, active from epoch 1 through epoch 30."
    ], font_size=11)

    add_card(s18, c2_left, Inches(1.85), col_w, Inches(4.9), "Pairwise Conflict Frequencies", COLOR_ACCENT_AMBER)
    populate_card_text(s18, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• BCS vs Behavior Conflict Rate:",
        "  - 47.8% of super-steps exhibited opposing shared gradients (cosine < 0).",
        "• BCS vs Re-ID Conflict Rate:",
        "  - 46.5% of super-steps exhibited opposing shared gradients.",
        "• Behavior vs Re-ID Conflict Rate:",
        "  - 47.4% of super-steps exhibited opposing shared gradients.",
        "• Uniformity: Every task pair interfered with every other task pair almost exactly half the time!"
    ], font_size=11)

    add_card(s18, c3_left, Inches(1.85), col_w, Inches(4.9), "Scientific Takeaways", COLOR_ACCENT_TEAL)
    populate_card_text(s18, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Direct Proof of Interference: Demolishes the naive assumption that multi-task learning automatically enjoys positive transfer in livestock vision.",
        "• PCGrad Relief Mechanism: Projecting conflicting components prevented mutual gradient cancellation, boosting Behavior CVB accuracy to 81.04%.",
        "• Incomplete Cure: While PCGrad resolved vector conflict during updates, it could not bridge the underlying spatial representation divide.",
        "• Single-task models still maintain representational superiority."
    ], font_size=11)

    add_footer(s18, 18)

    # =========================================================================
    # SLIDE 19: ANSWERS TO RESEARCH QUESTIONS
    # =========================================================================
    s19 = add_blank_slide()
    add_header(s19, 19, "Thesis Findings Synthesis", "Direct Answers to the Research Questions",
               "Definitive, evidence-supported conclusions resolving RQ1, RQ2, and RQ3")

    add_card(s19, Inches(0.8), top_base, Inches(11.733), h_card_h, "Answer to RQ1: Impact of Cattle-Centered Representations", COLOR_ACCENT_BLUE)
    populate_card_text(s19, Inches(1.05), top_base + Inches(0.55), Inches(11.2), Inches(0.8), [
        "Cattle-centered representations (localized crops + foreground masks) yield significant, task-dependent improvements: BCS MAE dropped from 0.1929 to 0.1709 (-11.4%); Re-ID Snapshot Rank-1 surged from 38.88% to 62.93% (+24.05 pp); Behavior balanced accuracy rose from 71.30% to 74.43%. However, gains are metric-dependent: ordered tolerance improved for BCS, dynamic minority recall improved for Behavior, and view-invariance improved for Re-ID."
    ], font_size=11)

    add_card(s19, Inches(0.8), top_base + h_card_h + gap_h, Inches(11.733), h_card_h, "Answer to RQ2: Preserving Task-Specific Information", COLOR_ACCENT_TEAL)
    populate_card_text(s19, Inches(1.05), top_base + h_card_h + gap_h + Inches(0.55), Inches(11.2), Inches(0.8), [
        "Input representations must respect the underlying biology of each task: static skeletal morphology for BCS, multi-frame 1D-TCN temporal sequences (T=8) for behavior kinematics, and high-fidelity surface textures for Re-ID. Off-the-shelf quadruped pose (SuperAnimal) and viewpoint classifiers suffered high missingness and severe camera-shortcut risks, and were rightly isolated from the primary pipeline."
    ], font_size=11)

    add_card(s19, Inches(0.8), top_base + (h_card_h + gap_h)*2, Inches(11.733), h_card_h, "Answer to RQ3: Multi-Task Sharing & Negative Transfer", COLOR_ACCENT_AMBER)
    populate_card_text(s19, Inches(1.05), top_base + (h_card_h + gap_h)*2 + Inches(0.55), Inches(11.2), Inches(0.8), [
        "Monolithic hard parameter sharing (E1) causes outcome-level negative transfer across all three domains. Modular task-private adapters (E3) failed to mitigate negative transfer generally. Optimization-level gradient projection (E4 PCGrad) provided the most effective selective mitigation (lifting Behavior and recovering BCS/Re-ID vs E3) at zero parameter penalty, but dedicated single-task models remain superior overall."
    ], font_size=11)

    add_footer(s19, 19)

    # =========================================================================
    # SLIDE 20: THESIS CONTRIBUTIONS
    # =========================================================================
    s20 = add_blank_slide()
    add_header(s20, 20, "Academic Contributions", "Six Core Contributions of the Thesis",
               "Tangible methodological, empirical, and architectural contributions to precision livestock vision")

    grid_w = Inches(3.68)
    grid_h = Inches(2.35)
    row1_t = Inches(1.85)
    row2_t = row1_t + grid_h + Inches(0.2)

    conts = [
        ("1. Leakage-Free Protocols", COLOR_ACCENT_BLUE, [
            "• Established reproducible burst-group-disjoint (BCS), session-disjoint (Behavior), and open-set 69-cow identity protocols to prevent data leakage."
        ]),
        ("2. Controlled Baselines", COLOR_ACCENT_TEAL, [
            "• Built rigorous single-task baselines demonstrating true empirical gains of cattle-centered representations on identical matched test sets."
        ]),
        ("3. Task-Specific Design", COLOR_ACCENT_EMERALD, [
            "• Validated static morphology for BCS, 1D-TCN temporal convolution for behavior, and coat texture preservation for Re-ID."
        ]),
        ("4. Systematic MTL Benchmark", COLOR_ACCENT_AMBER, [
            "• First head-to-head comparison of Hard Sharing (E1), Modular Adapters (E3), and PCGrad (E4) against matched single-task reference ceilings."
        ]),
        ("5. Empirical Gradient Proof", COLOR_ACCENT_ROSE, [
            "• Directly measured 44,177 gradient conflicts across 16,140 steps, documenting chronic 47% cross-task interference in livestock vision."
        ]),
        ("6. Calibrated Insights", COLOR_CARD_BORDER_ACCENT, [
            "• Demolished the assumption of automatic multi-task synergy, framing joint MTL as an intentional engineering trade-off for edge devices."
        ])
    ]

    for idx, (ctitle, ccolor, cbullets) in enumerate(conts):
        c_col = idx % 3
        c_row = idx // 3
        c_l = Inches(0.8) + c_col * (grid_w + gap)
        c_t = row1_t if c_row == 0 else row2_t
        add_card(s20, c_l, c_t, grid_w, grid_h, ctitle, ccolor)
        populate_card_text(s20, c_l + Inches(0.15), c_t + Inches(0.55), grid_w - Inches(0.3), grid_h - Inches(0.6), cbullets, font_size=10)

    add_footer(s20, 20)

    # =========================================================================
    # SLIDE 21: LIMITATIONS
    # =========================================================================
    s21 = add_blank_slide()
    add_header(s21, 21, "Claim Boundaries", "Methodological & Empirical Limitations",
               "Preserving academic integrity by transparently stating the boundaries of this research")

    add_card(s21, c1_left, Inches(1.85), col_w, Inches(4.9), "Data & Protocol Limits", COLOR_ACCENT_ROSE)
    populate_card_text(s21, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Absence of Biological IDs in ScienceDB: Evaluated across unseen burst groups from automated passage sequences, not verified unseen biological cows.",
        "• Session-Disjoint Behavior: Behavior models evaluated across protected video sessions rather than cow-disjoint herds.",
        "• Walking Class Confounding: Walking behavior occurs exclusively in CVB barn surveillance (26 test sequences), creating source-class confounding."
    ], font_size=11)

    add_card(s21, c2_left, Inches(1.85), col_w, Inches(4.9), "Perception & Model Limits", COLOR_ACCENT_AMBER)
    populate_card_text(s21, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Perception Coverage Boundaries: Automated perception achieved 93.89% (BCS) and 96.42% (Behavior) coverage; failed perception instances were excluded.",
        "• Oracle Masks in Re-ID: Run 6 and MTL Re-ID used released ground-truth masks rather than autonomous segmentation, representing an upper-bound benchmark.",
        "• Confounded Adapter Capacity: E3 added 395,904 trainable params (+3.32%), confounding architecture with capacity."
    ], font_size=11)

    add_card(s21, c3_left, Inches(1.85), col_w, Inches(4.9), "Statistical & Scope Limits", COLOR_ACCENT_BLUE)
    populate_card_text(s21, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Point-Estimate Evaluations: All models evaluated from single trained checkpoints; computational budgets precluded repeated-seed confidence intervals.",
        "• Gradient Conflict Causality: Diagnostics show conflicts occurred during E4, but do not prove conflict was the sole causal driver of E1 degradation.",
        "• Unexecuted External Benchmarks: Planned tests on uncalibrated external herds (Ruchay, Dryad, MmCows) remain future work."
    ], font_size=11)

    add_footer(s21, 21)

    # =========================================================================
    # SLIDE 22: FUTURE WORK
    # =========================================================================
    s22 = add_blank_slide()
    add_header(s22, 22, "Future Horizons", "Promising Directions for Future Research",
               "Roadmap for transitioning multi-task cattle monitoring from benchmark to commercial barn deployment")

    add_card(s22, c_left1, Inches(1.85), c2_w, Inches(4.9), "Near-Term Technical Enhancements", COLOR_ACCENT_TEAL)
    populate_card_text(s22, c_left1 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Repeated-Seed Training & Statistical Bounds:",
        "  - Executing multi-seed runs to compute confidence intervals, variance bounds, and formal significance tests across all multi-task models.",
        "• End-to-End Autonomous Re-ID Pipeline:",
        "  - Replacing oracle ground-truth masks with an autonomous fine-tuned YOLO-seg or SAM model to quantify upstream error propagation.",
        "• Factorial Component Ablations:",
        "  - Isolating the exact marginal gain of tight bounding box cropping vs binary mask concatenation vs temporal pooling."
    ], font_size=11.5)

    add_card(s22, c_left2, Inches(1.85), c2_w, Inches(4.9), "Long-Term Deployment & Generalization", COLOR_ACCENT_BLUE)
    populate_card_text(s22, c_left2 + Inches(0.25), Inches(2.55), c2_w - Inches(0.5), Inches(4.0), [
        "• Cross-Herd External Validation:",
        "  - Deploying the frozen multi-task models on completely unseen external commercial dairy farms (Ruchay, Dryad, MmCows, BECA).",
        "• Longitudinal Biometric Tracking:",
        "  - Stress-testing Re-ID across seasonal coat shedding, lactation cycles, and substantial body weight changes over 12+ months.",
        "• Species-Specific Cattle Pose Networks:",
        "  - Fine-tuning keypoint estimators on recumbent and rear-view cattle to overcome severe occlusion in lying postures.",
        "• Advanced MTL Optimization: Exploring branched trunks and dynamic loss weighting (GradNorm, Nash-MTL)."
    ], font_size=11.5)

    add_footer(s22, 22)

    # =========================================================================
    # SLIDE 23: CONCLUSION
    # =========================================================================
    s23 = add_blank_slide()
    add_header(s23, 23, "Summary & Takeaways", "Conclusions & Takeaways for Precision Livestock Farming",
               "Synthesizing representation learning, multi-task dynamics, and edge deployment trade-offs")

    add_card(s23, c1_left, Inches(1.85), col_w, Inches(4.9), "1. Cattle Representation Wins", COLOR_ACCENT_BLUE)
    populate_card_text(s23, c1_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Isolating the target animal via localized crops and foreground segmentation is essential.",
        "• Stripping background noise unlocked major single-task gains: BCS MAE dropped to 0.1709, and Re-ID Snapshot Rank-1 surged by +24.05 pp.",
        "• Input formatting must match task biology: static contours for BCS, temporal aggregation for behavior, texture for Re-ID."
    ], font_size=11)

    add_card(s23, c2_left, Inches(1.85), col_w, Inches(4.9), "2. MTL is Not a Free Lunch", COLOR_ACCENT_ROSE)
    populate_card_text(s23, c2_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• Heterogeneous livestock tasks induce chronic gradient conflict (~47% of training steps) across shared representations.",
        "• Naive hard parameter sharing (E1) caused unambiguous negative transfer across all 3 domains.",
        "• Adding modular adapters (E3) failed to solve negative transfer generally, exacerbating error on BCS and Re-ID."
    ], font_size=11)

    add_card(s23, c3_left, Inches(1.85), col_w, Inches(4.9), "3. Calibrated Engineering Trade-Off", COLOR_ACCENT_EMERALD)
    populate_card_text(s23, c3_left + Inches(0.2), Inches(2.55), col_w - Inches(0.4), Inches(4.0), [
        "• PCGrad gradient projection (E4) provided the most effective selective mitigation at zero parameter overhead, lifting Behavior CVB accuracy to 81.04%.",
        "• Edge Inference vs Peak Accuracy: Where compute is severely constrained, PCGrad multi-tasking offers an attractive compromise. Where peak clinical accuracy is vital, single-task models remain the gold standard."
    ], font_size=11)

    add_footer(s23, 23)

    # =========================================================================
    # SLIDE 24: THANK YOU / QUESTIONS (DARK ELEGANT CLOSING SLIDE)
    # =========================================================================
    s24 = add_blank_slide(bg_color=COLOR_BG_DARK)

    # Closing Box
    q_box = s24.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.0))
    q_tf = q_box.text_frame
    q_tf.word_wrap = True
    qp1 = q_tf.paragraphs[0]
    qp1.text = "Thank You!"
    qp1.font.name = "Arial"
    qp1.font.size = Pt(40)
    qp1.font.bold = True
    qp1.font.color.rgb = COLOR_TEXT_WHITE
    qp1.space_after = Pt(12)

    qp2 = q_tf.add_paragraph()
    qp2.text = "Questions & Discussion  •  Phase 3 Undergraduate Thesis Defense"
    qp2.font.name = "Arial"
    qp2.font.size = Pt(18)
    qp2.font.color.rgb = RGBColor(56, 189, 248)

    # Info card on closing
    q_card = s24.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.8), Inches(11.333), Inches(2.7))
    q_card.fill.solid()
    q_card.fill.fore_color.rgb = RGBColor(30, 41, 59)
    q_card.line.color.rgb = RGBColor(51, 65, 85)

    q_tb = s24.shapes.add_textbox(Inches(1.4), Inches(4.0), Inches(10.5), Inches(2.3))
    q_tframe = q_tb.text_frame
    q_tframe.word_wrap = True

    qp_m = q_tframe.paragraphs[0]
    qp_m.text = "RESEARCH TEAM & CONTACT"
    qp_m.font.name = "Arial"
    qp_m.font.size = Pt(12)
    qp_m.font.bold = True
    qp_m.font.color.rgb = RGBColor(148, 163, 184)
    qp_m.space_after = Pt(10)

    p_names = q_tframe.add_paragraph()
    p_names.text = "Hasin Ishrak (22201133)  •  Namira Abrar Haque (22201191)  •  Sanjida Akter Bithi (22201180)  •  Shouvik Banik (23101295)  •  Nusrat Lamya Faruk (24241182)"
    p_names.font.name = "Arial"
    p_names.font.size = Pt(12)
    p_names.font.color.rgb = COLOR_TEXT_WHITE
    p_names.space_after = Pt(8)

    p_sup = q_tframe.add_paragraph()
    p_sup.text = "Supervisor: Dr. Md. Khalilur Rahman, Professor, Dept. of CSE  •  Co-Supervisor: Mehedi Hasan Emo, Lecturer, Dept. of CSE"
    p_sup.font.name = "Arial"
    p_sup.font.size = Pt(11.5)
    p_sup.font.color.rgb = RGBColor(203, 213, 225)
    p_sup.space_after = Pt(8)

    p_repo = q_tframe.add_paragraph()
    p_repo.text = "Project Code, Manifests & Research Logs: https://github.com/Hasinish/cattle-health-monitoring-multi-task-model"
    p_repo.font.name = "Arial"
    p_repo.font.size = Pt(11.5)
    p_repo.font.bold = True
    p_repo.font.color.rgb = RGBColor(56, 189, 248)

    add_footer(s24, 24, is_dark=True)

    # Save
    prs.save(output_path)
    print(f"Presentation successfully created and saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    out = "Phase3_Cattle_Health_MTL_Defense.pptx"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    build_presentation(out)
