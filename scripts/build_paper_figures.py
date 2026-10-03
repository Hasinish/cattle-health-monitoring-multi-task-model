import subprocess
import os

figures = {
    'fig_research_design': r'''
\documentclass[tikz,border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\begin{document}
\begin{tikzpicture}[
  box/.style={draw,rounded corners,align=center,text width=3.2cm,minimum height=1.25cm,font=\small},
  arrow/.style={->,thick}
]
\node[box] (protocols) at (0,0) {Leakage-aware protocols\\BCS, Behavior, Re-ID};
\node[box] (rgb) at (4.2,0) {RGB single-task\\reference models};
\node[box] (centered) at (8.4,0) {Task-specific\\cattle-centered models};
\node[box] (e1) at (2.1,-2.2) {Hard-shared\\multi-task control};
\node[box] (e3) at (6.3,-2.2) {Modular task-private\\multi-task model};
\node[box] (synthesis) at (4.2,-4.4) {Task-wise comparison\\and negative-transfer analysis};
\draw[arrow] (protocols) -- (rgb);
\draw[arrow] (rgb) -- (centered);
\draw[arrow] (rgb) -- (e1);
\draw[arrow] (centered) -- (e3);
\draw[arrow] (e1) -- (synthesis);
\draw[arrow] (e3) -- (synthesis);
\end{tikzpicture}
\end{document}
''',
    'fig_input_pipeline': r'''
\documentclass[tikz,border=2pt]{standalone}
\usepackage{tikz,amsmath,amssymb}
\usetikzlibrary{arrows.meta}
\begin{document}
\begin{tikzpicture}[
  tag/.style={font=\bfseries\footnotesize, align=center, text width=2.2cm},
  box/.style={draw, rectangle, rounded corners=2pt, align=center, font=\scriptsize, minimum height=1.1cm, text width=2.1cm, fill=white, inner sep=2pt},
  oraclebox/.style={draw, rectangle, rounded corners=2pt, dashed, align=center, font=\scriptsize, minimum height=1.1cm, text width=2.1cm, fill=gray!8, inner sep=2pt},
  arrow/.style={->, >=stealth, thick}
]

% Row 1: BCS Pipeline
\node[tag] (t1) at (1.1, 0.0) {Body Condition\\Scoring (BCS)};
\node[box] (bcs1) at (3.35, 0.0) {\textbf{Raw RGB Image}\\(ScienceDB)};
\node[box] (bcs2) at (5.95, 0.0) {\textbf{Cattle Detection}\\(RT-DETR-L)};
\node[box] (bcs3) at (8.55, 0.0) {\textbf{Foreground Mask}\\(SAM 2.1)};
\node[box] (bcs4) at (11.15, 0.0) {\textbf{Crop + Mask}\\($224\times224\times4$)};
\node[box] (bcs5) at (13.75, 0.0) {\textbf{4-Ch ResNet-18}\\[2pt]Ordinal Head};

\draw[arrow] (bcs1) -- (bcs2);
\draw[arrow] (bcs2) -- (bcs3);
\draw[arrow] (bcs3) -- (bcs4);
\draw[arrow] (bcs4) -- (bcs5);

% Row 2: Behavior Pipeline
\node[tag] (t2) at (1.1, -1.65) {Behavior\\Recognition};
\node[box] (beh1) at (3.35, -1.65) {\textbf{Video Clip}\\(CVB / Beef)};
\node[box] (beh2) at (5.95, -1.65) {\textbf{Sample $T=8$ Frames}\\+ Target Crop};
\node[box] (beh3) at (8.55, -1.65) {\textbf{Frame-wise Mask}\\(SAM 2.1)};
\node[box] (beh4) at (11.15, -1.65) {\textbf{Sequence Tensor}\\($8\times4\times224^2$)};
\node[box] (beh5) at (13.75, -1.65) {\textbf{ResNet-18 + TCN}\\[2pt]5 Classes};

\draw[arrow] (beh1) -- (beh2);
\draw[arrow] (beh2) -- (beh3);
\draw[arrow] (beh3) -- (beh4);
\draw[arrow] (beh4) -- (beh5);

% Row 3: Re-ID Pipeline
\node[tag] (t3) at (1.1, -3.3) {Cattle Re-ID\\(Protocol A)};
\node[box] (reid1) at (3.35, -3.3) {\textbf{SideView Image}\\(Parlor / Barn)};
\node[oraclebox] (reid2) at (5.95, -3.3) {\textbf{Target Crop}\\[2pt](Oracle / GT)};
\node[oraclebox] (reid3) at (8.55, -3.3) {\textbf{Binary Mask}\\[2pt](Oracle / GT)};
\node[box] (reid4) at (11.15, -3.3) {\textbf{Crop + Mask}\\($224\times224\times4$)};
\node[box] (reid5) at (13.75, -3.3) {\mbox{\textbf{ResNet-18}} $\to$ $L_2$\\[2pt]Cosine Ranking};

\draw[arrow] (reid1) -- (reid2);
\draw[arrow] (reid2) -- (reid3);
\draw[arrow] (reid3) -- (reid4);
\draw[arrow] (reid4) -- (reid5);

\end{tikzpicture}
\end{document}
''',
    'fig_mtl_architecture': r'''
\documentclass[tikz,border=2pt]{standalone}
\usepackage{tikz,amsmath,amssymb}
\usetikzlibrary{arrows.meta}
\begin{document}
\begin{tikzpicture}[
  title/.style={font=\bfseries\small, align=center},
  inbox/.style={draw, rectangle, rounded corners=2pt, align=center, font=\scriptsize, text width=2.15cm, minimum height=0.7cm, fill=gray!5, inner sep=2pt},
  trunk/.style={draw, rectangle, rounded corners=3pt, align=center, font=\footnotesize\bfseries, text width=7.0cm, minimum height=0.85cm, fill=gray!12, inner sep=3pt},
  feat/.style={draw, rectangle, rounded corners=2pt, align=center, font=\scriptsize, text width=7.0cm, minimum height=0.65cm, fill=gray!6, inner sep=2pt},
  adapter/.style={draw, rectangle, rounded corners=2pt, align=center, font=\scriptsize, text width=2.15cm, minimum height=0.9cm, fill=gray!15, inner sep=2pt},
  head/.style={draw, rectangle, rounded corners=2pt, align=center, font=\scriptsize, text width=2.15cm, minimum height=0.85cm, inner sep=2pt},
  outbox/.style={draw, rectangle, rounded corners=2pt, align=center, font=\scriptsize, text width=2.15cm, minimum height=0.85cm, fill=gray!10, inner sep=2pt},
  arrow/.style={->, >=stealth, thick}
]

% ==================== LEFT: Hard-Shared MTL ====================
\node[title] at (3.75, 1.1) {Monolithic Hard-Shared\\Control Architecture (E1)};

\node[inbox] (e1_in1) at (1.35, 0.0) {\textbf{BCS Input}\\[1pt]$[B, 4, 224^2]$};
\node[inbox] (e1_in2) at (3.75, 0.0) {\textbf{Beh Sequence}\\[1pt]$[B, 8, 4, 224^2]$};
\node[inbox] (e1_in3) at (6.15, 0.0) {\textbf{Re-ID Input}\\[1pt]$[B, 4, 224^2]$};

\node[trunk] (e1_trunk) at (3.75, -1.45) {Shared 4-Channel ResNet-18 Trunk\\{\scriptsize (11,179,648 parameters)}};

\draw[arrow] (e1_in1.south) -- (e1_in1 |- e1_trunk.north);
\draw[arrow] (e1_in2.south) -- (e1_trunk.north);
\draw[arrow] (e1_in3.south) -- (e1_in3 |- e1_trunk.north);

\node[feat] (e1_feat) at (3.75, -2.85) {Shared Spatial Feature: $\mathbf{h} \in \mathbb{R}^{512}$};
\draw[arrow] (e1_trunk) -- (e1_feat);

\node[head] (e1_h1) at (1.35, -4.5) {\textbf{BCS Ordinal}\\[1pt]Linear Head};
\node[head] (e1_h2) at (3.75, -4.5) {\textbf{Behavior 1D}\\[1pt]TCN Head};
\node[head] (e1_h3) at (6.15, -4.5) {\textbf{Re-ID Head \&}\\[1pt]$L_2$ Retrieval};

\draw[arrow] (e1_feat.south -| e1_h1.north) -- (e1_h1.north);
\draw[arrow] (e1_feat.south) -- (e1_h2.north);
\draw[arrow] (e1_feat.south -| e1_h3.north) -- (e1_h3.north);

\node[outbox] (e1_o1) at (1.35, -5.9) {Physical BCS\\[1pt]Score $\widehat{\mathrm{BCS}}$};
\node[outbox] (e1_o2) at (3.75, -5.9) {5-Class Beh\\[1pt]Prediction};
\node[outbox] (e1_o3) at (6.15, -5.9) {Cosine Ranking\\[1pt]mAP / Rank-$k$};

\draw[arrow] (e1_h1) -- (e1_o1);
\draw[arrow] (e1_h2) -- (e1_o2);
\draw[arrow] (e1_h3) -- (e1_o3);

% ==================== RIGHT: Modular Task-Private MTL ====================
\node[title] at (11.25, 1.1) {Modular Task-Private\\Architecture (E3)};

\node[inbox] (e3_in1) at (8.85, 0.0) {\textbf{BCS Input}\\[1pt]$[B, 4, 224^2]$};
\node[inbox] (e3_in2) at (11.25, 0.0) {\textbf{Beh Sequence}\\[1pt]$[B, 8, 4, 224^2]$};
\node[inbox] (e3_in3) at (13.65, 0.0) {\textbf{Re-ID Input}\\[1pt]$[B, 4, 224^2]$};

\node[trunk] (e3_trunk) at (11.25, -1.45) {Shared 4-Channel ResNet-18 Trunk\\{\scriptsize (Identical Trunk: 11,179,648 parameters)}};

\draw[arrow] (e3_in1.south) -- (e3_in1 |- e3_trunk.north);
\draw[arrow] (e3_in2.south) -- (e3_trunk.north);
\draw[arrow] (e3_in3.south) -- (e3_in3 |- e3_trunk.north);

\node[feat] (e3_feat) at (11.25, -2.85) {Shared Feature: $\mathbf{h}_{\mathrm{shared}} \in \mathbb{R}^{512}$};
\draw[arrow] (e3_trunk) -- (e3_feat);

\node[adapter] (ad1) at (8.85, -4.3) {\textbf{BCS Adapter}\\[1pt]$512 \to 64 \to 512$};
\node[adapter] (ad2) at (11.25, -4.3) {\textbf{Beh Adapter}\\[1pt]$512 \to 64 \to 512$};
\node[adapter] (ad3) at (13.65, -4.3) {\textbf{Re-ID Adapter}\\[1pt]$512 \to 64 \to 512$};

\draw[arrow] (e3_feat.south -| ad1.north) -- (ad1.north);
\draw[arrow] (e3_feat.south) -- (ad2.north);
\draw[arrow] (e3_feat.south -| ad3.north) -- (ad3.north);

\node[head] (e3_h1) at (8.85, -5.7) {\textbf{BCS Head}};
\node[head] (e3_h2) at (11.25, -5.7) {\textbf{Behavior Head}};
\node[head] (e3_h3) at (13.65, -5.7) {\textbf{Re-ID Head}};

\draw[arrow] (ad1) -- (e3_h1);
\draw[arrow] (ad2) -- (e3_h2);
\draw[arrow] (ad3) -- (e3_h3);

\end{tikzpicture}
\end{document}
''',
    'fig_superstep_routing': r'''
\documentclass[tikz,border=2pt]{standalone}
\usepackage{tikz,amsmath,amssymb}
\usetikzlibrary{arrows.meta}
\begin{document}
\begin{tikzpicture}[
  title/.style={font=\bfseries\small, align=center},
  stepbox/.style={draw, rectangle, rounded corners=2.5pt, align=center, font=\footnotesize, text width=4.3cm, minimum height=1.35cm, fill=gray!5, inner sep=2.5pt},
  poolbox/.style={draw, rectangle, rounded corners=2.5pt, align=center, font=\footnotesize, text width=14.3cm, minimum height=0.65cm, fill=gray!12, inner sep=2.5pt},
  branchbox/.style={draw, rectangle, rounded corners=2.5pt, align=center, font=\footnotesize, minimum height=2.40cm, fill=gray!8, inner sep=3.5pt},
  optbox/.style={draw, rectangle, rounded corners=3pt, align=center, font=\footnotesize\bfseries, text width=14.3cm, minimum height=0.85cm, fill=gray!20, inner sep=4pt},
  arrow/.style={->, >=stealth, thick}
]

\node[title] at (7.45, 1.05) {Execution Sequence of One Multi-Task Training Super-Step};

\node[stepbox] (s1) at (2.40, 0.0) {\textbf{1. BCS} ($B=64$)\\[2pt]Forward: BCE $L_{\mathrm{BCS}}$\\[2pt]Backward: $\nabla_{\boldsymbol{\theta}} L_{\mathrm{BCS}}$};
\node[stepbox] (s2) at (7.45, 0.0) {\textbf{2. Behavior} ($B=8$)\\[2pt]Forward: CE $L_{\mathrm{Beh}}$\\[2pt]Backward: $\nabla_{\boldsymbol{\theta}} L_{\mathrm{Beh}}$};
\node[stepbox] (s3) at (12.50, 0.0) {\textbf{3. Re-ID} ($B=32$)\\[2pt]Forward: CE $L_{\mathrm{ReID}}$\\[2pt]Backward: $\nabla_{\boldsymbol{\theta}} L_{\mathrm{ReID}}$};

\draw[arrow] (s1) -- (s2);
\draw[arrow] (s2) -- (s3);

\node[poolbox] (pool) at (7.45, -1.60) {\textbf{Collected Super-Step Task Gradients:}\quad $\mathbf{g}_{\mathrm{BCS}},\,\mathbf{g}_{\mathrm{Beh}},\,\mathbf{g}_{\mathrm{ReID}}$};

\draw[arrow] (s1.south) -- (s1.south |- pool.north);
\draw[arrow] (s2.south) -- (s2.south |- pool.north);
\draw[arrow] (s3.south) -- (s3.south |- pool.north);

\node[branchbox, text width=6.5cm] (branch_std) at (3.65, -3.70) {%
  \textbf{E1 \& E3: Standard Sum Accumulation}\\[3pt]
  $\mathbf{g}_{\mathrm{shared}} = \mathbf{g}_{\mathrm{BCS}} + \mathbf{g}_{\mathrm{Beh}} + \mathbf{g}_{\mathrm{ReID}}$\\[4pt]
  {\scriptsize Direct unprojected task gradient summation}
};

\node[branchbox, text width=7.2cm] (branch_pcgrad) at (11.15, -3.70) {%
  \textbf{E4: PCGrad Shared Gradient Projection}\\[2pt]
  1. Pairwise conflict test: $\mathbf{g}_i^\top \mathbf{g}_j < 0$\\[2pt]
  2. Project: $\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i^\top \mathbf{g}_j}{\lVert\mathbf{g}_j\rVert_2^2 + \epsilon}\mathbf{g}_j$\\[2pt]
  3. Sum: $\mathbf{g}_{\mathrm{shared}} = \sum_t \mathbf{g}'_t$\\[2pt]
  {\scriptsize \textit{Heads remain isolated \& unprojected}}
};

\draw[arrow] (branch_std.north |- pool.south) -- (branch_std.north);
\draw[arrow] (branch_pcgrad.north |- pool.south) -- (branch_pcgrad.north);

\node[optbox] (opt) at (7.45, -5.80) {Unified AdamW Parameter Update:\quad $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta_t \cdot \operatorname{AdamW}(\mathbf{g},\, \lambda=10^{-4})$};

\draw[arrow] (branch_std.south) -- (branch_std.south |- opt.north);
\draw[arrow] (branch_pcgrad.south) -- (branch_pcgrad.south |- opt.north);

\end{tikzpicture}
\end{document}
''',
    'fig_behavior_recall': r'''
\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=11cm,height=5.5cm,ybar,ymin=0,ymax=1.18,ytick={0,0.5,1.0},
 ylabel={Recall},symbolic x coords={Standing,Lying,Feeding,Drinking,Walking},
 xtick=data,x tick label style={rotate=20,anchor=east},enlarge x limits=0.16,
 nodes near coords,point meta=y,nodes near coords style={font=\scriptsize},
 /pgf/number format/fixed,/pgf/number format/precision=2]
\addplot coordinates {(Standing,0.6460) (Lying,0.9496) (Feeding,0.9885) (Drinking,0.8095) (Walking,0.1923)};
\end{axis}
\end{tikzpicture}
\end{document}
'''
}

out_dir = 'cattle_paper_ieee/figures'
os.makedirs(out_dir, exist_ok=True)

for name, code in figures.items():
    tex_path = os.path.join(out_dir, f'{name}.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(code.strip())
    res = subprocess.run(['pdflatex', '-interaction=nonstopmode', f'-output-directory={out_dir}', tex_path], capture_output=True, text=True)
    pdf_path = os.path.join(out_dir, f'{name}.pdf')
    exists = os.path.exists(pdf_path)
    print(f'{name}: returncode={res.returncode}, pdf_exists={exists}')
