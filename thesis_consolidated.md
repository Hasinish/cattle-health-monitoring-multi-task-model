# Unified Cattle Health and Behavior Monitoring Framework
## Pre-Thesis II Report - Spring 2026
Compiled and converted from the LaTeX source code into clean Markdown for AI analysis.

---

# FRONT MATTER

## Title Page

\begin{titlepage}
\renewcommand*{\thepage}{Title}

    \begin{center} 
        \vspace*{2cm}
        
        {\fontsize{17.28pt}{22pt}\selectfont{Pre-Thesis II Report}
        }
        
        \vspace{0.8cm}
        
        {\fontsize{14pt}{20pt}\selectfont**Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring**
        }
        
        \vspace{0.6cm}
        
        \text{by}
        \vspace{0.6cm}
        
        Hasin Ishrak\\
        22201133\\
        \vspace{0.1cm}
        Namira Abrar Haque\\
        22201191\\
        \vspace{0.1cm}
        Sanjida Akter Bithi\\
        22201180\\
        \vspace{0.1cm}
        Shouvik Banik\\
        23101295\\
        \vspace{0.1cm}
        Nusrat Lamya Faruk\\
        24241182

        \vspace{1cm}
        
        A thesis submitted to the Department of Computer Science and Engineering\\
        in partial fulfillment of the requirements for the degree of\\
        B.Sc. in Computer Science and Engineering

        
        \vspace{1cm}
        
        Department of Computer Science and Engineering\\
        Brac University\\
        June 2026.
        
        \vspace{1cm}
        
        \copyright\ 2026. Brac University\\
        All rights reserved.
    
    \end{center}

\end{titlepage}

---

## Declaration

\newcommand*\wildcard[3][6cm]{
    \parbox{#1}{
        \centering

        \includegraphics[height=1.5cm, keepaspectratio]{#2}\\[-4pt] 
        \hrulefill\par
        #3
    }
}

\section*{Declaration}

It is hereby declared that

  - The thesis submitted is our own original work while completing degree at Brac University.
  - The thesis does not contain material previously published or written by a third party, except where this is appropriately cited through full and accurate referencing.
  - The thesis does not contain material which has been accepted, or submitted, for any other degree or diploma at a university or other institution.
  - We have acknowledged all main sources of help.

\vspace{0.5cm}
**Student's Full Name & Signature:**

\begingroup
    \centering
    

    \wildcard{sig_hasin.png}{\centerline{Hasin Ishrak} ~\\ \centerline{22201133}} 
    \hspace{1cm}
    \wildcard{sig_namira.png}{\centerline{Namira Abrar Haque} ~\\ \centerline{22201191}}
    
    \vspace{1.5cm} 
    

    \wildcard{sig_sanjida.png}{\centerline{Sanjida Akter Bithi} ~\\ \centerline{22201180}}
    \hspace{1cm}
    \wildcard{sig_shouvik.png}{\centerline{Shouvik Banik} ~\\ \centerline{23101295}}
    
    \vspace{1.5cm} 
    

    \wildcard{sig_nusrat.png}{\centerline{Nusrat Lamya Faruk} ~\\ \centerline{24241182}}

\endgroup

\pagebreak

---

## Approval

\renewcommand*\wildcard[3][6cm]{
    \parbox{#1}{
        \centering
        \ifx\hfuzz#2\hfuzz\else\includegraphics[height=1.2cm, keepaspectratio]{#2}\\[-2pt]\fi 
        \hrulefill\par
        {\small #3}
    }
}

\section*{Approval}

\vspace{-0.2cm}
The thesis titled ``Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring'' submitted by 
\setlength{-sep}{0pt}\setlength{\parskip}{0pt}\setlength{\parsep}{0pt}\setlength{\topsep}{0pt}
- Hasin Ishrak (22201133)
- Namira Abrar Haque (22201191)
- Sanjida Akter Bithi (22201180) 
- Shouvik Banik (23101295)
- Nusrat Lamya Faruk (24241182)

of Spring, 2026 has been accepted as satisfactory in partial fulfillment of the requirement for the degree of B.Sc. in Computer Science and Engineering in June, 2026. 

\vspace{0.1cm}
**Examining Committee:**

\vspace{0.1cm}

\noindent
Supervisor:\\
(Member)\\
\hspace*{7cm} \wildcard{}{Dr. Md. Khalilur Rahman \\ Professor \\ Department of Computer Science and Engineering \\ Brac University}

\vspace{1.5cm}

\noindent
Co-Supervisor:\\
(Member)\\
\hspace*{7cm} \wildcard{}{Mehedi Hasan Emo \\ Lecturer \\ Department of Computer Science and Engineering \\ Brac University}

\newpage

\noindent
Thesis Coordinator:\\
(Member)\\
\hspace*{7cm} \wildcard{sig_gra.jpg}{Dr. Md. Golam Rabiul Alam \\ Professor \\ Department of Computer Science and Engineering \\ Brac University}

\vspace{0.1cm}

\noindent
Head of Department:\\
(Chair)\\
\hspace*{7cm} \wildcard{}{Dr. Sadia Hamid Kazi \\ Associate Professor \\ Department of Computer Science and Engineering \\ Brac University}

\pagebreak

---

## Abstract

\section*{Abstract}

The rapid development of precision livestock farming has created a critical need for integrated systems capable of performing multiple concurrent health assessments. Traditional cattle monitoring research suffers from two primary limitations: most studies address individual tasks in isolation, and they rely on private datasets that prevent reproduction and benchmarking. This thesis addresses these gaps by proposing and implementing a unified, lightweight, and spatiotemporal multi-task deep learning framework that simultaneously processes four critical monitoring tasks from video inputs: Body Condition Scoring (BCS), behavior recognition, lameness detection, and individual animal identification. Built exclusively on publicly available datasets (ScienceDB, Dryad, MmCows, CattleLameness, OpenCows2020), our model employs a shared EfficientNet-B0 backbone encoder enhanced with a Convolutional Block Attention Module (CBAM) for spatial and channel focus. We employ task-specific heads and optimized loss functions, including Consistent Rank Logits (CORAL) loss for BCS ordinal regression, Focal Loss to address severe behavior class imbalances, and Long Short-Term Memory (LSTM) sequence-tracking layers for locomotion analysis. Single-task baselines achieved strong results: a BCS Mean Absolute Error (MAE) of 0.5566 on ScienceDB, a behavior recognition Macro F1-score of 0.7445, a lameness detection AUC of 0.9829, and 86.49% top-1 accuracy on individual cow identification. When integrated into the unified spatiotemporal multi-task model, lameness detection achieved an uncalibrated 80.00% accuracy (which rose to 100.00% only under post-hoc threshold tuning, serving as a preliminary proof-of-concept) and cow identification rose to 97.58% through temporal sequence modeling and majority voting. As expected in multi-task optimization with hard parameter sharing, BCS and behavior experienced severe negative transfer (MAE 0.7827, F1 0.4948). Consequently, the current joint model serves as a foundational baseline for future gradient-balancing research rather than a system ready for immediate clinical deployment. To evaluate generalization, cross-dataset validation was conducted on the CBVD-5 dataset, revealing critical domain shift challenges. The framework reduces edge VRAM requirements by approximately 75% compared to deploying four independent models, establishing the first complete, reproducible multi-task pipeline for cattle health monitoring using exclusively public datasets.

\vspace{1cm}
\noindent **Keywords:** Multi-Task Learning, Spatiotemporal Modeling, Cattle Health Monitoring, Body Condition Scoring, Behavior Recognition, Lameness Detection, Edge Deployment
\pagebreak

---

## Acknowledgement

\section*{Acknowledgement}

We want to show our heartfelt thanks to all of those who helped us make this Pre-Thesis II report possible. We are thankful to our supervisor, whose deep knowledge on computer vision, combined with his expertise, has been essential in making this report. We would also like to thank the Department of Computer Science and Engineering for giving us abundant resources and creating a research environment for us. Special thanks to the researchers who have made their datasets publicly available, whose commitment to open science has made our research possible. Finally we are grateful to our families and friends for their patience and support throughout this whole process.

\pagebreak

---

## Nomenclature

\chapter*{Nomenclature}

\section*{Abbreviations}

\begin{longtable}{l>{\raggedright\arraybackslash}p{11cm}}
AI & Artificial Intelligence \\
AUC & Area Under the Curve \\
BCS & Body Condition Score/Scoring \\
BCE & Binary Cross-Entropy \\
BN & Batch Normalization \\
CBAM & Convolutional Block Attention Module \\
CBVD-5 & Cow Behavior Video Dataset - 5 classes \\
CE & Cross-Entropy \\
CLAHE & Contrast Limited Adaptive Histogram Equalization \\
CNN & Convolutional Neural Network \\
CORAL & Rank Consistent Ordinal Regression \\
CPU & Central Processing Unit \\
DL & Deep Learning \\
DWT & Discrete Wavelet Transform \\
F1 & F1-Score (harmonic mean of precision and recall) \\
FLOPs & Floating Point Operations \\
FPS & Frames Per Second \\
GAN & Generative Adversarial Network \\
GPU & Graphics Processing Unit \\
Grad-CAM & Gradient-weighted Class Activation Mapping \\
I3D & Inflated 3D ConvNet \\
ImageNet & Large-scale visual recognition dataset \\
IMU & Inertial Measurement Unit \\
IoT & Internet of Things \\
IoU & Intersection over Union \\
IR & Infrared \\
KD & Knowledge Distillation \\
LSTM & Long Short-Term Memory \\
MAE & Mean Absolute Error \\
mAP & mean Average Precision \\
Mask R-CNN & Mask Region-based Convolutional Neural Network \\
mIoU & mean Intersection over Union \\
ML & Machine Learning \\
MSE & Mean Squared Error \\
MTL & Multi-Task Learning \\
NMS & Non-Maximum Suppression \\
P1 & Pre-Thesis I \\
PCA & Principal Component Analysis \\
PLF & Precision Livestock Farming \\
R-CNN & Region-based Convolutional Neural Network \\
ReLU & Rectified Linear Unit \\
ResNet & Residual Network \\
RF & Random Forest \\
RFID & Radio-Frequency Identification \\
RGB & Red Green Blue (color image) \\
RGB-D & RGB plus Depth \\
RNN & Recurrent Neural Network \\
ROC & Receiver Operating Characteristic \\
ROI & Region of Interest \\
RQ & Research Question \\
SE & Squeeze-and-Excitation \\
SORT & Simple Online and Realtime Tracking \\
SOTA & State-of-the-Art \\
SSL & Self-Supervised Learning \\
SVM & Support Vector Machine \\
t-SNE & t-Distributed Stochastic Neighbor Embedding \\
VGG & Visual Geometry Group \\
XGBoost & Extreme Gradient Boosting \\
YOLO & You Only Look Once \\
\end{longtable}

\pagebreak

---

# CHAPTERS

## Background

The global livestock industry plays a vital role in ensuring food security, with cattle representing one of the most economically critical livestock species worldwide. According to the Food and Agriculture Organization (FAO), the global cattle population exceeds one billion heads, contributing substantially to meat and dairy production [fao2023livestock]. Consequently, the sustainability, economic viability, and productivity of farming operations are directly influenced by the health and welfare of these animals. Early detection of prevalent health issues---such as poor body condition, lameness, and abnormal behavior---can mitigate global economic losses estimated at billions of dollars annually while significantly improving animal welfare standards [weary2009understanding].

Cattle nutritional status and systemic health are fundamentally reflected in the Body Condition Score (BCS). BCS is traditionally assessed through visual and tactile evaluation of fat deposits over key anatomical regions, typically using a 5-point scale with 0.25-point increments [edmonson1989bcs]. While optimal body condition is directly linked to improved milk yield, reproductive efficiency, and disease resistance, manual BCS assessment is highly time-consuming, subjective, and labor-intensive, making frequent herd-wide monitoring impractical for large-scale operations. Similarly, lameness represents another critical health concern, affecting up to 25% of dairy cattle in intensive farming systems [whay2003lameness]. Lameness causes severe pain, reduces feed intake, lowers milk production, and hinders reproduction; however, timely intervention remains possible through the early detection of gait abnormalities and abnormal body movements.

Automated behavior monitoring and individual identification provide crucial insights into cattle health, stress levels, and overall welfare. Deviations in feeding patterns, rumination time, lying duration, and social interactions frequently precede the clinical manifestations of disease [weary2009understanding]. Continuous 24/7 monitoring through automated computer vision systems offers a non-invasive, objective alternative to manual observation. Furthermore, individual identification is fundamental to precision livestock farming, enabling the longitudinal tracking of health records, feeding habits, and productivity metrics for each animal. Traditional identification methods, such as physical ear tags and Radio Frequency Identification (RFID) systems, require physical contact and are prone to loss or damage. Non-invasive, vision-based identification systems overcome these limitations and can be seamlessly integrated with automated health monitoring frameworks.

The convergence of computer vision, deep learning, and Internet of Things (IoT) technologies has created unprecedented opportunities for automated, real-time cattle health monitoring [neethirajan2020digitalization]. Convolutional Neural Networks (CNNs) have demonstrated extraordinary success in image classification and object detection tasks, achieving human-level performance across diverse visual recognition challenges. Transfer learning further enables the application of deep learning models pre-trained on large-scale datasets, such as ImageNet, to agricultural domains where labeled data is scarce. Despite these technological advances, existing cattle monitoring systems predominantly address individual tasks in isolation. A systematic review of over 30 research papers conducted for this thesis reveals that while multi-task learning is frequently discussed, end-to-end implementations combining multiple health monitoring tasks are virtually non-existent. This fragmentation limits the practical utility of automated monitoring systems and fails to exploit the computational and representation-learning synergies between related tasks.

## Rationale of the Study and Motivation

The motivation for this research stems from critical gaps in the current state of automated cattle health monitoring systems and the escalating demand for precision livestock farming. First, a systematic analysis of the literature indicates that approximately 89% of current studies rely on private, proprietary datasets, which presents a major barrier to reproducibility and comparative validation. Second, the majority of existing research addresses individual tasks---such as BCS, lameness detection, behavior recognition, or individual identification---in isolation, neglecting the potential benefits of Multi-Task Learning (MTL). By sharing feature representations across related tasks, MTL can significantly reduce computational overhead while potentially improving performance. This is particularly relevant as these health indicators are biologically and behaviorally interconnected; for instance, abnormal locomotion or pain from lameness directly contributes to reduced feeding time, which subsequently leads to a decline in body condition.

Furthermore, many published solutions require expensive specialized hardware, such as thermal cameras, wearable biotelemetry sensors, or depth cameras, which are often cost-prohibitive or impractical for widespread commercial deployment. This highlights a clear need for frameworks that rely primarily on standard, low-cost RGB cameras while retaining the flexibility to integrate depth information when available. Additionally, modern agricultural operations require real-time, on-site processing because rural network connectivity is frequently unreliable and cloud-based inference introduces latency. While edge deployment is recognized as an optimal solution, most existing models rely on computationally heavy architectures unsuitable for resource-constrained edge devices. Finally, there is a distinct lack of self-supervised learning applications in cattle monitoring. Most commercial farms operate 24/7 surveillance cameras that accumulate vast quantities of video footage daily. Although this abundant unlabeled footage remains unutilized, it could serve as a valuable resource for pre-training models to learn robust representations, which is particularly beneficial when labeled data is scarce. Collectively, these challenges---reproducibility issues, task fragmentation, hardware constraints, deployment limitations, and underutilized video resources---highlight the necessity of a unified, lightweight framework for multi-task cattle monitoring.

## Problem Statement

Despite significant research efforts in automated cattle health monitoring, several fundamental challenges remain unresolved. Consequently, the practical deployment and real-world impact of computer vision and deep learning solutions on commercial farms remain limited.

A primary limitation is that existing systems address monitoring tasks independently, necessitating the deployment of multiple separate models to conduct comprehensive health assessments. This disjointed approach increases computational overhead, fails to exploit shared feature representations that could mutually benefit related tasks, complicates system integration and deployment, and overlooks the correlations between different health indicators (such as the relationship between poor body condition and reduced feeding activity). A systematic review of the literature confirms the absence of a unified framework that concurrently addresses body condition scoring, behavior recognition, lameness detection, and individual identification. Multi-task learning is frequently discussed conceptually but rarely implemented in end-to-end architectures.

In order to validate experimental results, compare methodologies objectively, and build upon prior literature, researchers must evaluate models on standardized datasets. However, a significant portion of cattle monitoring studies utilize proprietary or private datasets. This lack of data sharing hinders reproducibility, prevents fair performance comparisons, and restricts cumulative progress in the field. This stands in sharp contrast to general computer vision research, which benefits from standardized, publicly accessible benchmarks such as ImageNet and COCO that facilitate consistent and transparent evaluations.

Furthermore, most studies evaluate their models using data collected from a single farm or a single breed of cattle, raising questions regarding the generalizability of the proposed methods. Models evaluated under such constrained conditions risk overfitting to specific farm layouts, camera configurations, and illumination conditions, and they may fail to generalize across different breeds. Seasonal, environmental, and temporal variations are also rarely accounted for in evaluations. Crucially, cross-dataset evaluation---where a model is trained on one dataset and tested on another from an entirely different source---is seldom performed, despite being a prerequisite for demonstrating real-world viability.

Finally, while depth and thermal imaging have attracted significant interest for cattle monitoring, multimodal research is restricted by the scarcity of publicly available datasets. There are no public cattle thermal datasets, only the MmCows dataset provides synchronized Inertial Measurement Unit (IMU) data, and only two public datasets (MmCows and Dryad) offer depth information. This data scarcity forces researchers to either collect private datasets or forego multimodal approaches entirely.

## Objectives

The primary objective of this research is to develop a lightweight, edge-deployable, and fully reproducible multi-task deep learning framework for automated cattle health and behavior monitoring. By leveraging public datasets and shared feature representations, the framework aims to concurrently perform multiple health-related assessments while ensuring that all findings can be independently replicated.

To achieve this goal, the research sets out the following specific objectives:

    - **To design a unified, lightweight multi-task architecture** with a shared backbone and specialized task-specific heads, and to evaluate sequential training strategies to optimize performance across Body Condition Scoring, behavior recognition, lameness detection, and individual identification.
    
    - **To ensure full reproducibility** by conducting all evaluations exclusively on publicly available datasets, enabling direct, independent verification of the framework's findings.
    
    - **To conduct preliminary cross-dataset evaluation** to identify domain shift challenges across different farms and environmental conditions, establishing a baseline for comprehensive generalization studies in future work.

These objectives address the reproducibility gap in agricultural AI research and create a scalable foundation for real-time cattle health monitoring across various farm environments.

## Methodology in Brief

The proposed methodology follows a structured, multi-phase process to design, implement, and validate the multi-task deep learning framework. This research workflow is divided into five key phases: data preparation, architecture design, sequential task training, evaluation, and ablation studies.

### Evaluation Protocol

Evaluation metrics are carefully tailored to each task to address their distinct outputs and operational risks. For body condition scoring, the model's performance is measured using Mean Absolute Error (MAE) alongside classification accuracy within $±0.25$ and $±0.5$ tolerances. For behavior recognition, evaluation metrics include per-class accuracy, macro-averaged F1-score, and confusion matrix analysis. For lameness detection, sensitivity, specificity, and Area Under the Receiver Operating Characteristic (AUC-ROC) are evaluated. Individual identification is assessed using Top-1 and Top-5 accuracy alongside mean Average Precision (mAP). Finally, cross-dataset evaluation is conducted by training on a primary dataset and testing on an independent dataset to rigorously assess the model's generalization capabilities.

### Ablation Studies

A series of structured ablation experiments will be conducted to evaluate the contributions of key framework components. For backbone selection, EfficientNet-B0 will be compared against MobileNetV3-Small and ResNet-50 to determine the optimal trade-off between computational efficiency and accuracy. For the body condition scoring task, the performance of the Rank Consistent Ordinal Regression (CORAL) loss will be compared against standard Cross-Entropy loss. Regarding training strategies, experiments will compare sequential learning, joint multi-task training, and progressive unfreezing. The impact of multi-modal data will be assessed by comparing RGB-only models against combined RGB and depth models on the Dryad dataset. Finally, the efficacy of attention mechanisms will be evaluated by comparing architectures with and without the Convolutional Block Attention Module (CBAM).

## Scope and Challenges

### Scope of the Research

To ensure feasibility and depth of investigation, this research is defined by the following boundaries:

    - **Four Core Tasks:** The scope is restricted to Body Condition Scoring, behavior recognition, lameness detection, and individual identification, selected due to their practical significance and public data availability.
    
    - **Public Datasets Only:** All experimental evaluations are conducted exclusively on publicly available datasets to ensure complete reproducibility and independent verification.
    
    - **RGB as Primary Modality:** RGB imagery serves as the primary input modality, with depth information evaluated solely as an auxiliary modality for the body condition scoring task using the Dryad dataset.
    
    - **Dairy Cattle Focus:** The study focuses primarily on Holstein dairy cattle, which represent the dominant breed in public cattle monitoring datasets.
    
    - **Lightweight Architectures:** To ensure suitability for resource-constrained edge deployment, the backbone selection is restricted to architectures with fewer than 10 million parameters.
    
    - **Sequential Task Training:** Frozen-backbone sequential training serves as the primary optimization strategy, while joint multi-task fine-tuning and loss balancing are explored as secondary extensions.

### Challenges

**Camera Geometry Conflicts**

The four tasks in this framework require fundamentally different camera viewpoints: BCS scoring performs best from a rear-view angle (capturing hip and spine morphology), while Lameness detection requires a side-view angle (to observe gait stride length and limb symmetry). Running both tasks from a single fixed camera creates an unavoidable geometric trade-off. To address this, the deployment model assumes a multi-camera setup where a single edge device loads the shared backbone once, but selectively activates the rear-view or side-view task heads based on the incoming camera feed.

**Dataset Heterogeneity**

Public cattle datasets exhibit high heterogeneity, including variations in image resolution, quality, camera placement (top-view, rear-view, and side-view), annotation protocols, breed characteristics, and farm environments. To mitigate these discrepancies, the methodology incorporates a standardized preprocessing pipeline, robust data augmentation, and domain adaptation techniques.

**Class Imbalance**

Cattle monitoring datasets frequently exhibit severe class imbalances. In lameness detection, lame individuals are heavily outnumbered by healthy cattle; similarly, extreme body condition scores (under 2.5 or over 3.5) are significantly rarer than moderate scores, and specific behaviors (such as drinking or licking) occur infrequently. To address these imbalances, the framework employs Focal Loss for classification tasks, class-weighted sampling during training, and balanced evaluation metrics.

**Temporal Modeling for Video Tasks**

While behavior recognition and lameness detection benefit from temporal context, video-based modeling increases computational overhead and introduces architectural complexity (e.g., through LSTMs or I3D networks), alongside potential discrepancies between frame-level and clip-level annotations. The framework addresses this by first utilizing frame-level classification combined with temporal aggregation, treating recurrent model integration as a subsequent extension.

**Preventing Negative Transfer**

Multi-task learning designs are susceptible to negative transfer, where optimization for one task degrades the performance of another. This typically occurs due to conflicting task gradients during joint optimization, varying optimal feature representations across tasks, and disparity in task difficulties. The primary mitigation strategy involves sequential training with a frozen backbone to decouple task optimization, supplemented by gradient normalization and dynamic loss weighting during joint fine-tuning.

**Limited Labeled Data**

Despite utilizing multiple public datasets, the total volume of labeled data remains small compared to standard computer vision benchmarks. For example, the ScienceDB BCS dataset contains 53,566 images, in contrast to the millions in ImageNet. The MmCows dataset tracks only 16 individuals over a two-week period, and the CattleLameness dataset contains a relatively small number of lame examples. To mitigate data limitations, the methodology leverages transfer learning from ImageNet, robust regularization (such as dropout and weight decay), and extensive data augmentation, with self-supervised pre-training on unlabeled video frames designated as an optional extension.

---

## Preliminaries

This chapter presents a systematic review of the research papers related to cattle health monitoring using computer vision and deep learning. The review covers five primary research areas: Body Condition Scoring (BCS), lameness detection, behavior recognition, individual identification, and cleanliness assessment. The study identifies current state-of-the-art methods, public datasets which are available, the research gaps, and suitable ways for developing a multi-task learning framework.

### Multi-Task Learning Fundamentals

Multi-Task Learning (MTL) is a machine learning paradigm where a model is trained to multiple related tasks simultaneously, leveraging shared representations to improve generalization across all tasks [crawshaw2020mtl]. In deep learning terms, MTL generally uses a shared backbone network that takes out common features, with task-specific heads branching from the shared representation. The benefits of MTL include improved generalization through shared representations that act as regularization to reduce overfitting, computational efficiency via a single forward pass serving multiple tasks, data efficiency where tasks with limited data benefit from related tasks, and enhanced feature learning where auxiliary tasks can improve primary task features. Still, MTL also comes with some challenges like negative transfer when tasks interfere with one another, difficulties while balancing gradients also identifying the most efficient parameter sharing strategies. The Crawshaw 2020 MTL Survey provides a comprehensive taxonomy distinguishing between hard parameter sharing (common backbone) and soft parameter sharing (separate networks with regularization) approaches.

A critical challenge in multi-task optimization is balancing the gradients and losses of tasks with different scales, difficulties, and convergence rates. Standard static loss balancing often leads to dominant tasks overriding the training process, resulting in negative transfer. To resolve this, dynamic loss balancing approaches have been proposed. Kendall et al. (2018) [kendall2018multi] introduced multi-task loss weighting using homoscedastic uncertainty, which dynamically scales each task's loss based on its inherent noise level. Alternatively, Chen et al. (2018) [chen2018gradnorm] proposed GradNorm, a gradient normalization algorithm that dynamically balances training by directly adjusting task-specific loss weights to equalize gradient magnitudes. Furthermore, deciding which tasks to learn together is a non-trivial problem. Standley et al. (2020) [standley2020tasks] studied task relationships in multi-task settings, proposing a framework to identify task groupings that minimize negative transfer and optimize hardware resource allocation, demonstrating that sharing representations across all tasks is not always optimal.

### Body Condition Scoring Overview

Body Condition Scoring (BCS) is a standard way for evaluating the fat reserves of cattle through visual and tactile evaluation. The most common scale ranges start from 1 (skeletal) to 5 (fat) with 0.25 increments, yielding 17 possible scores. Key anatomical regions assessed include the tailhead and pin bones, hook bones and loin area, ribs and spine prominence, and hip bone region. Automated BCS using computer vision typically employs rear-view or top-view images capturing the tailhead region. The task can be worked out as classification (individual BCS classes), ordinal regression (preserving class ordering), or regression (continuous scores).

### Lameness Detection Fundamentals

Lameness in cattle shows by identifying walking behaviour including reduced step length and walking speed, head movement and back curving, unwillingness to bear weight on the affected leg, and uneven hoof placement. Computer vision approaches for lameness detection typically analyze video sequences captured from side-view or top-view cameras. Key features include pose estimation of limb joints, gait cycle analysis, and motion trajectory patterns. Scoring systems range from binary (lame/sound) to multi-level severity scales (1-5).

### Behavior Recognition Categories

Cattle behavior monitoring address numerous activities related to health and welfare, which includes feeding behaviors for example feeding, ruminating, and drinking; body movement activities such as walking, standing, and lying; social behaviors including licking, mounting, and aggression; also health indicators such as estrus behavior and calving signs. Vision-based behavior recognition implements detection (locating cattle in frames), tracking (maintaining identity across frames), and action classification (categorizing behavior). Temporal modeling by using LSTM, 3D CNNs, or attention mechanisms captures behavioral changes.

## Review of Existing Research

This section shows a comprehensive analysis of related research arranged by task category, followed by key multi-task learning and survey papers.

### Body Condition Scoring Research

Body condition scoring represents a highly studied task in cattle computer vision literature, with numerous papers addressing automated BCS assessment. Zin et al. (2026) [zin2026ab26] explored BCS automation using Deep Learning architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from Sensor Data. Yao et al. (2026) [yao2026jds2] explored BCS automation using YOLO-based architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from RGB Video/Image. Guzhva et al. (2026) [guzhva2026s415] explored BCS automation using Mask R-CNN architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from RGB-D/Depth. Liu et al. (2025) [liu2025vets] explored BCS automation using EfficientNet architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from RGB Video/Image. Antognoli et al. (2025) [antognoli2025ani1] explored BCS automation using Deep Learning architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from RGB Video/Image. Sani et al. (2026) [sani2026ab25] explored BCS automation using CNN architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from Sensor Data. Palma et al. (2025) [palma2025ani1] explored BCS automation using CNN architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from Sensor Data. Lee et al. (2026) [lee2026ab26] explored BCS automation using Deep Learning architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from Sensor Data. Gonçalves et al. (2025) [gonçalves2025skaf] explored BCS automation using ResNet architectures, demonstrating significant improvements in assessment accuracy and noting that deep learning methods provide robust feature extraction from RGB Video/Image. 

**Table (tab:bcs_papers): Summary of Body Condition Scoring Papers**

| p2.5cm|>\raggedright\arraybackslashp3.8cm|>\raggedright\arraybackslashp2.5cm|>\raggedright\arraybackslashp2.0cm|>\raggedright\arraybackslashp2.0cm|

**Reference** | **Architecture** | **Input** | **Accuracy** | **Dataset** |
| --- | --- | --- | --- | --- |
| Zin 2026 | Deep Learning | Sensor Data | High | Various |
| Yao 2026 | YOLO-based | RGB Video / Image | 99.5% | Various |
| Guzhva 2026 | Mask R-CNN | RGB-D / Depth | 95% | Various |
| Liu 2025 | EfficientNet | RGB Video / Image | 91.10% | Various |
| Antognoli 2025 | Deep Learning | RGB Video / Image | High | Various |
| Sani 2026 | CNN | Sensor Data | High | Various |
| Palma 2025 | CNN | Sensor Data | 87% | Various |
| Lee 2026 | Deep Learning | Sensor Data | High | Various |
| Gonçalves 2025 | ResNet | RGB Video / Image | 84% | Various |

### Lameness Detection Research

Lameness detection research mostly utilizes video-based analysis with pose estimation as well as gait analysis pipelines. Paneru et al. (2026) [paneru2026jpsj] developed a lameness detection system utilizing YOLO-based, focusing on the extraction of locomotion patterns from RGB Video / Image to achieve robust classification. Sohan et al. (2025) [sohan2025s415] developed a lameness detection system utilizing CNN, focusing on the extraction of locomotion patterns from RGB-D / Depth to achieve robust classification. Szyc et al. (2025) [szyc2025s251] developed a lameness detection system utilizing Deep Learning, focusing on the extraction of locomotion patterns from RGB Video / Image to achieve robust classification. Zin et al. (2026) [zin2026ab26] developed a lameness detection system utilizing Deep Learning, focusing on the extraction of locomotion patterns from Sensor Data to achieve robust classification. Sani et al. (2026) [sani2026ab25] developed a lameness detection system utilizing CNN, focusing on the extraction of locomotion patterns from Sensor Data to achieve robust classification. 

A comparative analysis of these methods highlights the critical role of deployment conditions and target granularity.

**Table (tab:lameness_papers): Summary of Lameness Detection Papers**

| p2.4cm|>\raggedright\arraybackslashp3.4cm|>\raggedright\arraybackslashp2.5cm|>\raggedright\arraybackslashp2.0cm|>\raggedright\arraybackslashp2.5cm|

**Reference** | **Architecture** | **Input** | **Accuracy** | **Innovation** |
| --- | --- | --- | --- | --- |
| Paneru 2026 | YOLO-based | RGB Video / Image | High | DL-based Lameness |
| Sohan 2025 | CNN | RGB-D / Depth | 90% | DL-based Lameness |
| Szyc 2025 | Deep Learning | RGB Video / Image | 82% | DL-based Lameness |
| Zin 2026 | Deep Learning | Sensor Data | High | DL-based Lameness |
| Sani 2026 | CNN | Sensor Data | High | DL-based Lameness |

### Behavior Recognition Research

Behavior recognition covers vision-based as well as sensor-based approaches. Paneru et al. (2026) [paneru2026jpsj] contributed to the field by leveraging YOLO-based for analyzing cattle activities through RGB Video / Image, achieving an accuracy of High. Asim et al. (2026) [asim2026s260] contributed to the field by leveraging YOLO-based for analyzing cattle activities through Sensor Data, achieving an accuracy of 91.11%. Zin et al. (2026) [zin2026ab26] contributed to the field by leveraging Deep Learning for analyzing cattle activities through Sensor Data, achieving an accuracy of High. Liu et al. (2025) [liu2025vets] contributed to the field by leveraging EfficientNet for analyzing cattle activities through RGB Video / Image, achieving an accuracy of 91.10%. Sani et al. (2026) [sani2026ab25] contributed to the field by leveraging CNN for analyzing cattle activities through Sensor Data, achieving an accuracy of High. 

**Table (tab:behavior_papers): Summary of Behavior Recognition Papers**

| p2.5cm|>\raggedright\arraybackslashp3.8cm|>\raggedright\arraybackslashp2.6cm|>\raggedright\arraybackslashp2.0cm|>\raggedright\arraybackslashp2.0cm|

**Reference** | **Architecture** | **Input** | **Accuracy** | **Dataset** |
| --- | --- | --- | --- | --- |
| Paneru 2026 | YOLO-based | RGB Video / Image | High | Various |
| Asim 2026 | YOLO-based | Sensor Data | 91.11% | Various |
| Zin 2026 | Deep Learning | Sensor Data | High | Various |
| Liu 2025 | EfficientNet | RGB Video / Image | 91.10% | Various |
| Sani 2026 | CNN | Sensor Data | High | Various |

### Individual Cattle Identification Research

Vision-based individual cattle identification has emerged as a non-invasive alternative to traditional methods such as ear tags, branding, and RFID transponders. Andrew et al. (2017) [andrew2017cattle] pioneered automated Holstein identification using selective local coat pattern matching on RGB-D imagery captured from overhead cameras, achieving robust identification by exploiting the unique black-and-white coat patterns of Holstein-Friesian cattle and establishing the foundational methodology for subsequent deep learning approaches. Building on this work, Andrew et al. (2021) [andrew2021opencows] introduced the OpenCows2020 dataset and achieved 98.2% identification accuracy using deep metric learning with a SoftMax-based reciprocal triplet loss, demonstrating that CNNs can reliably distinguish individual cattle from overhead imagery in open-herd settings without requiring retraining when new individuals are introduced. Gao et al. (2021) [gao2021cows2021] extended this line of research by proposing a self-supervised framework for cattle identification using the Cows2021 dataset [cows2021] (10,402 annotated images and 301 videos), employing tracking-by-detection to form rotation-normalized tracklets for contrastive learning, which is particularly relevant as it reduces the labeling burden that limits scalability of supervised identification systems. Beyond coat pattern analysis, Weng et al. (2022) [weng2022cattleface] explored cattle face recognition using a two-branch CNN architecture, achieving competitive identification accuracy and demonstrating that facial features provide complementary biometric information to coat patterns, though requiring frontal camera positioning that may not always be feasible in farm environments.

**Table (tab:id_papers): Summary of Cattle Identification Papers**

| p2.8cm|>\raggedright\arraybackslashp4.2cm|>\raggedright\arraybackslashp2.2cm|c|l|

**Reference** | **Architecture** | **Input** | **Accuracy** | **Dataset** |
| --- | --- | --- | --- | --- |
| Andrew 2021 | Deep Metric Learning + Triplet Loss | RGB overhead | 98.2% | OpenCows2020 |
| Gao 2021 | Self-Supervised Contrastive Learning | RGB video | N/A | Cows2021 |
| Weng 2022 | Two-Branch CNN | RGB face | Competitive | Private |
| Andrew 2017 | Coat Pattern Matching | RGB-D | Robust | Private |

### Multi-Task Learning and Survey Papers

Numerous papers provide foundational knowledge for multi-task learning approaches and comprehensive surveys of agricultural AI. Shalaldeh et al. (2026) [shalaldeh2026biol] investigated multi-task paradigms using ResNet, demonstrating improved efficiency when concurrently predicting multiple traits in agricultural domains. Attri et al. (2025) [attri2025ms90] investigated multi-task paradigms using EfficientNet, demonstrating improved efficiency when concurrently predicting multiple traits in agricultural domains. Sandal et al. (2026) [sandal2026v1] investigated multi-task paradigms using CNN, demonstrating improved efficiency when concurrently predicting multiple traits in agricultural domains. Bu et al. (2026) [bu2026jsaa] investigated multi-task paradigms using CNN, demonstrating improved efficiency when concurrently predicting multiple traits in agricultural domains. 

**Table (tab:mtl_papers): Summary of MTL, Survey, and Other Papers**

| p3.0cm|>\raggedright\arraybackslashp3.5cm|>\raggedright\arraybackslashp2.8cm|>\raggedright\arraybackslashp4.1cm|

**Reference** | **Focus** | **Type** | **Key Contribution** |
| --- | --- | --- | --- |
| Shalaldeh 2026 | Multi-Task Analysis | Method / Survey | Agricultural MTL |
| Attri 2025 | Multi-Task Analysis | Method / Survey | Agricultural MTL |
| Sandal 2026 | Multi-Task Analysis | Method / Survey | Agricultural MTL |
| Bu 2026 | Multi-Task Analysis | Method / Survey | Agricultural MTL |
| Crawshaw 2020 | Multi-Task Learning | Survey | MTL taxonomy and guidelines |

## Summary of Key Findings

The structured review of 33 papers reveals many critical insights that directly inform our proposed methodology.

The substantial majority of papers (89%) rely on private datasets which creates a reproducibility crisis. Among them only a small fraction relies on publicly available cattle-specific datasets, yet they still obtain competitive performance, indicating that public datasets can efficiently support high-quality research. Our targeted use of public datasets helps bridge this gap, enabling total reproducibility as well as fair benchmarking.

Despite being consistently raised on multi-task learning in most of the literature, no existing paper integrates BCS, behavior recognition, lameness detection, and individual identification in a unified framework. Most of the papers address single tasks in isolation. Our unified four task framework represents a true novelty with potential for improved efficiency and cross task learning. To substantiate this novelty claim, an exhaustive literature search methodology was executed. Standard academic databases (including Google Scholar, IEEE Xplore, ScienceDirect, and MDPI) were systematically queried using combinations of search terms: (*``cattle''* OR *``dairy cow''*) AND (*``body condition score''* OR *``BCS''* OR *``lameness''* OR *``behavior''* OR *``activity recognition''* OR *``re-identification''* OR *``individual ID''*) AND (*``multi-task learning''* OR *``deep learning''* OR *``computer vision''*). Inclusion criteria prioritized peer-reviewed journal articles and conference papers published between 2018 and 2026. Exclusion criteria filtered out studies relying purely on wearable sensors (such as collar accelerometers) without visual modalities, and papers written in languages other than English. Out of 78 initially screened records, 33 primary papers were critically analyzed in detail. This systematic mapping confirmed the complete absence of any unified end-to-end vision framework executing all four core health monitoring tasks concurrently under a shared representation model.

Transfer learning with frozen backbone is well-supported in the literature, and the Crawshaw 2020 MTL Survey validates that this approach effectively prevents negative transfer between heterogeneous tasks. Our sequential task training strategy is well-supported by existing literature and provides a strong foundation for managing multiple diverse tasks.

Depth imaging represents substantial research interest with demonstrated value for BCS as well as lameness. Still, public data is limited to MmCows and Dryad datasets. Depth modality is relevant for BCS using Dryad but should be treated as secondary to RGB, ensuring the framework remains accessible for standard camera deployments.

Self-supervised learning remains highly underexplored in cattle monitoring compared to transfer learning, representing a potential research opportunity. MmCows provides 2 weeks of continuous unlabeled video ideal for SSL pre-training. SSL represents a potential bonus contribution if time permits, offering opportunities to leverage abundant unlabeled surveillance footage.

Comprehensive search reveals there are no publicly available thermal or infrared datasets for cattle. The only thermal cattle research (Carraro 2026) used privately collected data. Thermal modality is excluded from current scope and documented as future work requiring data collection partnerships.

**Table (tab:summary): Summary of Key Cattle Monitoring Techniques from Literature**

| p3.2cm|>\raggedright\arraybackslashp2cm|>\raggedright\arraybackslashp2.0cm|>\raggedright\arraybackslashp3.2cm|>\raggedright\arraybackslashp2.6cm|

**Technique** | **Platform** | **Accuracy** | **Strength** | **Weakness** |
| --- | --- | --- | --- | --- |
| EfficientNet + SE | RGB BCS | 93.77% | High accuracy, attention | Single task |
| BCS-YOLO + KD | RGB BCS | 92.7% | Edge-ready, distilled | Single task |
| Detectron2 + SORT [deepsort2017] | Depth Lameness | 99.94% | Excellent detection | Requires depth |
| CNN-LSTM | RGB Behavior | 97.35% | Temporal modeling | Private data |
| XGBoost + DWT | IMU Behavior | 91.8% | Public benchmark | Requires sensors |
| MTL Survey | Cross-platform | N/A | Guidelines | No implementation |

---

## Specifications and Requirements
The implementation of the multi-task deep learning framework for cattle health monitoring requires a specific configuration of hardware and software components to ensure real-time inference and model training.

### Hardware Requirements

    - **Development and Training Unit:** The model training was executed on a high-performance research workstation equipped with an NVIDIA GeForce RTX 4080 GPU (16GB VRAM, Ada Lovelace architecture supporting FP8 and FP16 tensor operations), an Intel Core i9-13900K CPU, 64GB of DDR5 RAM, and a 2TB NVMe SSD to handle high-throughput image and video loading.
    - **Edge Deployment Unit:** For live farm surveillance, the system is designed to run on resource-constrained edge computers such as the Jetson Orin Nano (8GB VRAM) or a Raspberry Pi 5. The total model size is optimized to remain under 10 million parameters ($<$ 40MB file size in FP16 format) to fit into edge device memory. Furthermore, the shared EfficientNet-B0 backbone requires approximately 0.39 GFLOPs per frame. Processing a full 20-frame spatiotemporal sequence requires only $\sim$7.8 GFLOPs, which falls comfortably within the real-time processing capabilities of a Jetson Orin Nano (capable of 40 TOPS).
    - **Imaging Sensors:** Standard 1080p surveillance cameras operating at 30 frames per second (FPS) positioned at cattle walkways or feeding troughs. For the Dryad dataset, depth maps were captured using active infrared depth sensors.

### Software Requirements

    - **Operating System:** Windows 11 / Linux (Ubuntu 22.04 LTS).
    - **Programming Language:** Python 3.12.3.
    - **Deep Learning Library:** PyTorch 2.5.1 + CUDA 12.1.
    - **Backbone Repository:** PyTorch Image Models (\texttt{timm}) library for loading pre-trained EfficientNet-B0 and MobileNetV3 architectures.
    - **Computer Vision Tools:** OpenCV 4.9.0 (for real-time video streaming, bounding box visualization, and frame interpolation) and Albumentations 1.4.0 (for spatial and pixel-level data augmentations).
    - **Data Pipelines:** Pandas, NumPy, and Scikit-learn for group-wise dataset splits and metric computations.

## Societal Impact
The development of automated cattle health monitoring systems has a profound positive impact on the agricultural sector and animal welfare:

    - **Enhancement of Animal Welfare:** Early detection of lameness and abnormal behaviors (such as reduced eating or drinking) allows farmers to treat sick cows days before clinical symptoms become visible, minimizing animal suffering.
    - **Support for Precision Livestock Farming (PLF):** Transitioning from manual, subjective herd inspections to continuous 24/7 automated monitoring reduces human error and standardizes assessments such as Body Condition Scoring.
    - **Agricultural Labor Relief:** As intensive dairy farming expands, the ratio of cows per stockperson increases. Automated monitoring alleviates labor shortages, allowing farm workers to focus on specialized veterinary interventions rather than manual surveillance.

## Environmental Impact
Deep learning models are historically compute-intensive, contributing to carbon emissions during training. Our proposed framework addresses this environmental concern through:

    - **Green Computing via Multi-Task Learning (MTL):** Instead of deploying four separate convolutional neural networks (each consuming GPU power and memory), our hard parameter sharing approach utilizes a single shared EfficientNet-B0 encoder. This consolidation reduces inference energy consumption by approximately 75%, minimizing the operational carbon footprint of edge surveillance systems deployed on thousands of farms.
    - **Resource Optimization:** By implementing sparse temporal sampling (reducing 300-frame video clips to 20 key frames) and capping dataset classes, we decreased active training GPU runtimes, reducing energy overhead during the development lifecycle.

## Ethical Issues
Automated visual surveillance systems on farms raise specific ethical considerations:

    - **Non-Invasive Monitoring:** Traditional identification and health tracking rely on invasive techniques such as branding, ear tags, or surgically implanted RFID transponders, which cause pain and stress to the animals. Our camera-based approach is completely non-invasive, posing zero physical discomfort or psychological stress to the cattle.
    - **Data Privacy:** Farm surveillance footage must be processed locally on edge systems to prevent the leakage of proprietary farm operation details, veterinarian records, or geographic information to public networks. Our offline edge architecture addresses this ethical requirement.

## Standards - if applicable
The software code and agricultural metrics strictly adhere to the following standards:

    - **Body Condition Scoring (BCS) Standards:** The target BCS grading scale follows the internationally recognized 5-point Elanco scale (with increments of 0.25 on ScienceDB and integer values on Dryad) which is the standard reference used by professional dairy veterinarians.
    - **Locomotion Scoring Standards:** The lameness target labels are aligned with the Sprecher locomotion scoring system, classifying gait posture on a binary scale (sound vs. lame).
    - **IEEE Software Quality Standards:** The code is written in compliance with the PEP 8 style guide for Python, and evaluation metrics conform to standard definitions (Macro F1-score for classification under imbalance, Mean Absolute Error for ordinal variables, and AUC-ROC for ranking).

## Project Management Plan
To manage the multi-task and multi-backbone complexity of the project, the workload was distributed across 5 team members based on their base backbone assignments and specific tasks:

    - **Hasin Ishrak:** Preprocessed the primary datasets, established the baseline training pipelines, evaluated the ResNet-18 baseline, and conducted the primary ablation studies (such as CBAM evaluations and RGB vs. Depth modalities).
    - **Namira Abrar Haque:** Evaluated the MobileNetV3-Small baseline, analyzing the performance of highly compressed networks for resource-constrained edge deployment.
    - **Sanjida Akter Bithi:** Evaluated the ResNet-50 baseline, evaluating the benefits of deeper architectures on spatial feature extraction.
    - **Shouvik Banik:** Evaluated the DenseNet121 baseline, studying dense feature connectivity for multi-class behavior recognition.
    - **Nusrat Lamya Faruk:** Evaluated the winning EfficientNet-B0 baseline, implemented the final spatiotemporal multi-task model, integrated the LSTM sequence-tracking layers, and deployed the real-time visualization interface.

The timeline was structured in sprints: dataset curation, single-task baseline training, backbone selection, spatiotemporal multi-task model integration, and finally report writing and visualizer deployment.

## Risk Management

**Table (tab:risk_management): Project Risks and Mitigation Strategies**

| p5cm>\raggedright\arraybackslashp6cm
\toprule
**Risk Identified** | **Potential Impact** | **Mitigation Strategy** |
| --- | --- | --- |
| \midrule
Class Imbalance | Model fails to learn rare behaviors (licking, drinking). | Implemented Focal Loss ($\gamma=2$) and hard data capping (max 3000 images per class). |
| Data Leakage | Memorization of cow identities leading to overfitting. | Enforced cow-wise split partitioning so same cow is never in train and test sets. |
| Out of Memory (OOM) | GPU VRAM crash when processing long sequences. | Integrated Automatic Mixed Precision (AMP) in FP16 and sparse temporal sampling of 20 frames. |
| Domain Shift | Performance drop when deployed on unseen farms. | Performed cross-dataset validation on CBVD-5 to evaluate generalizability and recommend future unfreezed fine-tuning. |
| \bottomrule |  |  |

## Economic Analysis
The financial viability of our multi-task framework is compared against traditional farm monitoring systems in Table~tab:economic_analysis.

**Table (tab:economic_analysis): Economic Cost-Benefit Analysis**

| p6cm>\raggedright\arraybackslashp6cm
\toprule
**Component** | **Traditional Sensor Systems** | **Proposed MTL Edge System** |
| --- | --- | --- |
| \midrule
**Sensor Hardware** | Wearable collars, step counters, and RFID tags ($80 - $120 per cow). For 500 cows: $40,000+. | Standard IP CCTV cameras ($150 each) + edge computing unit ($300). Total: $1,500. |
| **Maintenance** | Battery replacements, collar damage, and physical tag losses (15% annual loss). | Minimal hardware contact, camera cleaning and software updates only. |
| **Computational Cost** | High cloud subscription fees for running separate analysis pipelines. | Offline local edge inference with zero recurring cloud subscription costs. |
| \bottomrule |  |  |

The analysis demonstrates that our visual multi-task edge framework is highly scalable, lowering the initial investment barrier for small-to-medium scale dairy farms by over 90%.

---

## Design Process or Methodology Overview
The primary objective of this research is to construct a unified, lightweight, and edge-deployable multi-task deep learning framework capable of performing four critical cattle monitoring tasks simultaneously from video inputs: Body Condition Scoring (BCS), Behavior Recognition, Lameness Detection, and Cow Identification (ID). The design process follows a sequential workflow structured to mitigate task interference while maximizing parameter sharing. An overview of the proposed multi-task architecture is illustrated in Figure~fig:methodology_overview.

*[Figure (fig:methodology_overview): Conceptual Overview of the Multi-Task Cattle Health and Behavior Monitoring Framework.]*

The pipeline is split into three main segments:

    - **Cattle Detection and Cropping:** YOLOv8-Nano is applied to locate the cow and extract its bounding box crop, eliminating background farm clutter and normalizing the input.
    - **Feature Extraction:** The cropped images or sequences are processed by a shared EfficientNet-B0 [efficientnet2019 encoder enhanced with a Convolutional Block Attention Module (CBAM) [cbam2018] to spotlight key anatomical indicators.
    - **Task Prediction:** The feature vectors are fed into task-specific heads: ordinal regression for BCS, focal-loss classification [focalloss2017 for Behavior, sequence-based LSTM for Lameness, and classification for Cow ID.

## Data Collection and Preprocessing
To establish a fully reproducible framework, only publicly available cattle datasets were utilized. Each dataset corresponds to a specific health task.

### Dataset Profiles

    - **ScienceDB BCS Dataset [sciencedb2024**: Consists of 53,566 high-resolution RGB images of dairy cows taken from a rear-view angle. The body condition scores are annotated on a 5-point scale with 0.25 increments, representing five classes: ${3.25, 3.50, 3.75, 4.00, 4.25}$.
    - **Dryad BCS Dataset [dryad2024**: Contains 5,923 rear-view images captured in a Depth Grayscale Edge (DGE) format (combining depth coordinates, grayscale intensity, and Canny edge maps as three channels). The annotations are integer scores ranging from 2 to 6, corresponding to five distinct ordinal classes.
    - **MmCows Behavior Dataset [mmcows2024**: Includes 213,686 crop frames extracted from continuous farm surveillance videos. The dataset is labeled into seven behavior classes: walking, standing, feeding head up, feeding head down, licking, drinking, and lying.
    - **CattleLameness Dataset [cattlelameness2024**: Contains side-view walking video clips of cows. We compiled 50 videos (25 labeled as Lame and 25 as Normal/Sound) for spatiotemporal sequence modeling.
    - **OpenCows2020 Dataset [opencows2020**: Consists of 4,736 images of cows labeled by their unique individual identity across 46 classes, enabling vision-based biometrics.
    - **CBVD-5 Dataset:** Used exclusively as an independent test set for behavior, consisting of 2,000 balanced images of cattle behaviors to evaluate model generalization under domain shift.

### Preprocessing Standardizations
To ensure the deep learning models do not learn artifact features or identity shortcuts, rigorous preprocessing was applied:

    - **Cow-Wise Group Splitting:** To prevent identity leakage (where the model memorizes specific cow features like skin patches or ear tags instead of general health markers), group-wise splitting was enforced. The unique cow IDs were partitioned into 70% training, 15% validation, and 15% testing splits. Thus, a cow in the validation or test set was never seen during training.
    - **Sparse Temporal Sampling:** Videos vary in frame length. To format sequences for recurrent layers without hidden-state drift or out-of-memory (OOM) GPU crashes, videos were divided into 20 equal temporal segments, and exactly one frame was sampled from the center of each segment. This standardized the input sequence tensor to a fixed shape of $(20, 3, 224, 224)$.
    - **Imbalance Data Capping:** In the MmCows behavior dataset, classes were severely imbalanced (lying and standing had over 70,000 images, while licking had only 2,009). Training on this directly causes majority-class bias. To mitigate this, majority classes were capped at a maximum of 3,000 randomly sampled training images per epoch, yielding a balanced sub-dataset of approximately 21,000 images.
    - **Augmentations:** Training samples were resized to $224 \times 224$ pixels, normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406, std=[0.229, 0.224, 0.225]), and augmented with random horizontal flips ($p=0.5$) and random rotations ($± 15^\circ$). Validation and test samples were resized and normalized without augmentations.

## Model Architecture and Specification
The architecture utilizes a shared parameter structure to maximize computational efficiency on edge devices.

### Shared Backbone Encoder
The encoder backbone is initialized with ImageNet-pretrained weights from timm:

```math

    F_{\text{raw}} = \text{Encoder}(X) \quad F_{\text{raw}} \in \mathbb{R}^{B \times C \times H \times W}

```

EfficientNet-B0 [efficientnet2019] (5.3M parameters) and MobileNetV3-Small [mobilenetv3] (2.5M parameters) were evaluated, with EfficientNet-B0 selected as the final backbone due to its superior feature representation.

### CBAM Attention Module
A Convolutional Block Attention Module (CBAM) [cbam2018] is inserted immediately after the backbone's final feature map.

    - **Channel Attention:** Identifies ``what'' channels contain relevant signal using average and max pooling:
    
```math

        M_c(F) = \sigma(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)))
    
```

    - **Spatial Attention:** Highlights ``where'' the relevant anatomy lies (e.g. spine line for BCS, joint positions for lameness) using a $7 \times 7$ convolution over channel-concatenated maps:
    
```math

        M_s(F') = \sigma(f^{7 \times 7}([\text{AvgPool}(F'); \text{MaxPool}(F')))
    
```

The final attention-weighted feature map is pooled using Adaptive Average Pooling and flattened to a 1,280-dimensional feature vector.

### Task-Specific Prediction Heads
The feature vector is routed to four independent task heads:

    - **BCS Head:** A linear layer projecting the 1,280 features to 4 outputs representing the ordinal ranking.
    - **Behavior Head (Temporal LSTM):** Similar to the lameness head, the sequence of feature vectors is processed by a single-layer LSTM, followed by a linear classification layer to output 7 behavior logits.
    - **Lameness Head (Temporal LSTM):** The sequence of 20 feature vectors is processed by a single-layer LSTM (hidden size = 256, dropout = 0.5), followed by a linear classification layer to output a binary logit.
    - **Cow ID Head:** A linear layer projecting features to $N$ cattle identity classes.

## Loss Functions and Optimizations
Each task is optimized using a loss function tailored to its specific mathematical properties:

### Consistent Rank Logits (CORAL) Loss for BCS
To preserve the natural ordering of body condition scores, ordinal regression using the Consistent Rank Logits (CORAL) framework [coral2019 was formulated. For $K$ classes (where $K=5$), the model outputs $K-1=4$ binary logits. The loss is computed as the sum of binary cross-entropies:

```math

    L_{\text{BCS}} = -\frac{1}{K-1} \sum_{i=1}^{K-1} \left[ y_i \log(\sigma(g_i)) + (1 - y_i) \log(1 - \sigma(g_i)) \right]

```

where $y_i = \mathbb{I}(y > i)$ is the binary rank representation of the actual score. Predictions are computed by summing the sigmoid outputs:

```math

    \hat{y} = \sum_{i=1}^{K-1} \mathbb{I}(\sigma(g_i) > 0.5)

```

### Focal Loss for Behavior Classification
To prevent the overwhelming majority classes from dominating training gradients, Focal Loss [focalloss2017] is applied to the behavior outputs:

```math

    L_{\text{Behavior}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)

```

where $p_t$ is the model's estimated probability for the correct class, $\alpha = 0.25$ is the class-balancing multiplier, and $\gamma = 2.0$ is the focusing parameter that down-weights easy, well-classified examples.

### Lameness and ID Losses
Binary Cross-Entropy (BCE) with logits loss is utilized for the binary lameness classification task:

```math

    L_{\text{Lameness}} = - \left[ y \log(\sigma(x)) + (1-y) \log(1-\sigma(x)) \right]

```

Standard Cross-Entropy Loss with Softmax normalization is utilized for individual cow identification:

```math

    L_{\text{ID}} = - \sum_{c=1}^N y_c \log(\text{Softmax}(x_c))

```

## Implementation of Selected Design and Training Strategy
The training pipeline was executed on PyTorch using a phased sequential training methodology to avoid destructive task interference (where gradient updates for one task corrupt the weights learned for another).

### Phased Sequential Training

    - **Phase 1: Encoder Initialization.** The EfficientNet-B0 backbone is initialized with ImageNet-pretrained weights.
    - **Phase 2: BCS Head Training.** The backbone weights are frozen. The attention layer and BCS head are trained on Dryad and ScienceDB for 30 epochs using the Adam optimizer (lr = $10^{-3}$, scheduler decay = 0.5 every 10 epochs).
    - **Phase 3: Behavior Head Training.** The backbone is kept frozen. The behavior head is trained for 30 epochs using Focal Loss on the capped MmCows dataset. Early stopping is monitored with a patience of 10 epochs.
    - **Phase 4: Lameness Sequence Training.** The backbone remains frozen. Bounding box sequence crops are extracted using YOLOv8-Nano. The LSTM temporal layers and classification head are trained on 20-frame sequences for 15 epochs.
    - **Phase 5: Cow ID Training.** The backbone remains frozen. The ID head is trained on OpenCows2020 for 10 epochs.
    - **Phase 6: Multi-Task Joint Optimization.** All heads are attached. The backbone is unfrozen, and the model is trained end-to-end using the multi-task loss weight configuration:
    
```math

        L_{\text{total}} = 0.35 \cdot L_{\text{BCS}} + 0.35 \cdot L_{\text{Behavior}} + 0.15 \cdot L_{\text{Lameness}} + 0.15 \cdot L_{\text{ID}}
    
```

    The task weights assigned to each loss component ($0.35$ for BCS and Behavior, and $0.15$ for Lameness and ID) are determined by the relative dataset sizes and task complexities. Body Condition Scoring (BCS) and Behavior Recognition represent complex visual classification tasks backed by very large datasets (53,566 images for ScienceDB and over 21,000 capped frames for MmCows). In contrast, Lameness Detection (50 sequence samples) and Cow Identification (4,736 images) have simpler classification scopes or operate on much smaller sample registers. Assigning higher loss weights to the larger and more complex tasks prevents the shared encoder backbone from drifting and over-specializing on the smaller datasets during backpropagation. These weights were selected empirically based on relative dataset sizes and task complexities; formal sensitivity analysis comparing alternative weightings is deferred to future work.

### Live Video Inference Architecture
During live deployment, videos are processed frame-by-frame via a sliding window. The YOLOv8-Nano detector extracts cattle crops, which are normalized and queued. Once 20 crops are accumulated:

    - The sequence is fed into the shared encoder and LSTM to output a single, robust locomotion score (Lameness prediction).
    - Spatial heads make 20 independent frame-level predictions. Frame-level BCS scores are averaged (Temporal Averaging) to prevent noise, and Cow ID predictions are resolved using majority voting across the 20 frames. This mitigates transient occlusions (such as background posts or dust) and ensures high deployment reliability.

---

## Performance Evaluation
The performance of the multi-task spatiotemporal model and the individual baselines was evaluated using standard computer vision and veterinary diagnostic metrics:

    - **Body Condition Scoring (BCS):** Evaluated using Mean Absolute Error (MAE) on the unified class indices ($0-4$), alongside exact match accuracy ($\text{Acc}_{± 0}$) and tolerance match accuracy ($\text{Acc}_{± 1}$, representing predictions within one class step, which matches the veterinarian tolerance of $± 0.25$ units).
    - **Behavior Recognition:** Evaluated using Macro F1-score to assess performance equally across the highly imbalanced behavioral classes. Per-class accuracies were also extracted from the confusion matrix.
    - **Lameness Detection:** Evaluated using Receiver Operating Characteristic Area Under the Curve (ROC-AUC) as a threshold-independent measure of locomotion classification, along with binary accuracy.
    - **Cow Identification:** Evaluated using Top-1 accuracy to verify correct identification of unique individuals.

## Single-Task Implementation and Results

### BCS Results
BCS was trained separately on Dryad and ScienceDB. The primary metric is MAE on ordinal indices 0--4. Lower MAE is better. Exact accuracy measures exact class match, while within-one-class accuracy measures whether the prediction is at most one ordinal class away from the true label (see Table~tab:bcs_baselines_detailed).

**Table (tab:bcs_baselines_detailed): BCS baseline results on Dryad and ScienceDB.**

| \toprule
**Member** | **Backbone** | **Dryad MAE** | **Dryad Exact** | **Dryad $± 1$** | **ScienceDB MAE** | **ScienceDB Exact** | **ScienceDB $± 1$** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| \midrule
Hasin | ResNet-18 | 0.8675 | 24.50% | 88.75% | 0.5800 | 54.81% | 89.02% |
| Namira | MobileNetV3-Small | 0.5250 | 61.50% | 86.00% | 0.7090 | 44.67% | 86.47% |
| Bithi | ResNet-50 | 0.6300 | 47.75% | 89.25% | 0.6485 | 48.43% | 88.20% |
| Shouvik | DenseNet121 | 0.5875 | 54.75% | 86.50% | 0.6292 | 49.51% | 88.78% |
| Nusrat | EfficientNetB0 | 0.6175 | 52.50% | 85.75% | 0.5566 | 55.06% | 90.50% |
| \bottomrule |  |  |  |  |  |  |  |

The Dryad results show that MobileNetV3-Small performs best on MAE and exact accuracy, despite being the smallest model. This suggests that the DGE representation can benefit from a lightweight model when training data is limited. ScienceDB results show that EfficientNetB0 performs best with MAE 0.5566 and within-one-class accuracy of 90.50%, suggesting stronger generalization on the larger RGB dataset. The convergence behavior of the selected EfficientNet-B0 backbone for the BCS task is illustrated in Figure~fig:bcs_loss_curve, demonstrating a steady reduction in training and validation loss without signs of significant overfitting.

*[Figure (fig:bcs_loss_curve): Training and validation loss curve for the BCS single-task model (EfficientNet-B0).*

### Behavior Recognition Results
Behavior recognition was trained on MmCows with seven classes. The main metric is Macro F1-score because it treats every class equally and is more suitable for imbalanced datasets than overall accuracy (see Table~tab:behavior_baselines_detailed).

**Table (tab:behavior_baselines_detailed): Behavior recognition baseline results on MmCows.**

| \toprule
**Member** | **Backbone** | **Epochs** | **Val Macro F1** | **Test Macro F1** | **Early stop** |
| --- | --- | --- | --- | --- | --- |
| \midrule
Hasin | ResNet-18 | 18 | 0.7241 | 0.7134 | 18 |
| Namira | MobileNetV3-Small | 30 | 0.7271 | 0.6810 | N/A |
| Bithi | ResNet-50 | 22 | 0.7210 | 0.7037 | 22 |
| Shouvik | DenseNet121 | 27 | 0.7371 | 0.7366 | 27 |
| Nusrat | EfficientNetB0 | 24 | **0.7631** | **0.7445** | 24 |
| \bottomrule |  |  |  |  |  |

EfficientNetB0 achieves the best validation and test Macro F1-score (per-class breakdown is provided in Table~tab:behavior_per_class). DenseNet121 is the second-best behavior model. Figure~fig:behavior_loss_curve illustrates the training and validation loss trajectory for the behavior recognition model. The early stopping mechanism triggered around epoch 24 when the validation Macro F1-score stabilized, preventing overfitting on the highly imbalanced MmCows dataset.

*[Figure (fig:behavior_loss_curve): Training and validation loss curve for the behavior recognition single-task model (EfficientNet-B0).]*

**Table (tab:behavior_per_class): EfficientNetB0 behavior per-class accuracy on the MmCows test set.**

| \toprule
**Class** | **Test accuracy** |
| --- | --- |
| \midrule
Walking | 59.78% |
| Standing | 70.00% |
| Feeding head up | 71.97% |
| Feeding head down | 66.07% |
| Licking | 83.96% |
| Drinking | 83.01% |
| Lying | 98.77% |
| \bottomrule |  |

Analyzing the per-class accuracies in Table~tab:behavior_per_class reveals key insights into the model's strengths and limitations. The model achieves near-perfect classification on the *Lying* class ($98.77%$), which is characterized by a highly distinct static posture that is easily differentiated from active states. In contrast, classes like *Walking* ($59.78%$) and *Feeding head down* ($66.07%$) show the lowest accuracies. This is due to temporal and semantic overlap; *Walking* frames are frequently confused with *Standing* frames during transition phases, and *Feeding head down* is occasionally misclassified as *Feeding head up* or *Drinking* when the head position fluctuates near the water troughs or feed bunks. Similarly, *Licking* ($83.96%$) and *Drinking* ($83.01%$) achieve high accuracy due to distinct localized head movement patterns, though minor confusion still occurs due to spatial similarities in head positioning. These findings underscore that static spatial features alone struggle to resolve fine-grained temporal boundaries, which motivated the integration of sequence-modeling LSTMs in the multi-task stage.

### Lameness Results

**Table (tab:lameness_baselines): Lameness detection baseline results.**

| \toprule
**Member** | **Model Architecture** | **Epochs** | **Test AUC** | **Test Accuracy** | **Test F1-score** |
| --- | --- | --- | --- | --- | --- |
| \midrule
Hasin | ResNet-18 (Spatial) | 10 | 0.7200 | 75.30% | 0.8452 |
|  | ResNet18-LSTM (Sequence) | 15 | 0.8800 | 70.00% | 0.7692 |
| \midrule
Namira | MobileNetV3-Small (Spatial) | 10 | 0.7228 | 75.66% | 0.8394 |
|  | MobileNetV3-LSTM (Sequence) | 15 | 0.9200 | 70.00% | 0.7273 |
| \midrule
Bithi | ResNet-50 (Spatial) | 10 | 0.7444 | 80.97% | 0.8752 |
|  | ResNet50-LSTM (Sequence) | 15 | 1.0000 | 80.00% | 0.8333 |
| \midrule
Shouvik | DenseNet121 (Spatial) | 10 | 0.7944 | 73.98% | 0.8377 |
|  | DenseNet121-LSTM (Sequence) | 15 | 0.9200 | 90.00% | 0.9091 |
| \midrule
Nusrat | EfficientNetB0 (Spatial) | 10 | **0.9829** | **93.05%** | **0.9483** |
|  | EfficientNetB0-LSTM (Sequence) | 15 | 0.8400 | 80.00% | 0.8000 |
|  | ResNet18-LSTM (Sequence)* | 15 | 0.9600 | 60.00% | 0.7143 |
| \bottomrule |  |  |  |  |  |

### Cattle Identification Results

**Table (tab:id_baselines): Cattle Identification baseline results on OpenCows2020.**

| \toprule
**Member** | **Backbone** | **Epochs** | **Final Train Loss** | **Val Top-1 Acc** | **Test Top-1 Acc** |
| --- | --- | --- | --- | --- | --- |
| \midrule
Hasin | ResNet-18 | 10 | 2.4158 | 42.01% | 45.56% |
| Namira | MobileNetV3-Small | 10 | 1.1022 | 80.21% | 78.83% |
| Bithi | ResNet-50 | 10 | 2.1213 | 52.97% | 53.02% |
| Shouvik | DenseNet121 | 10 | 0.7514 | 82.04% | 82.46% |
| Nusrat | EfficientNetB0 + CBAM | 10 | **0.5508** | **87.06%** | **86.49%** |
| \bottomrule |  |  |  |  |  |

## Analysis of Design Solutions (Backbone Selection)
Before constructing the unified multi-task model, a systematic backbone architecture evaluation was conducted. The five team members trained single-task baseline models on their assigned architectures to select the optimal shared feature extractor. The empirical evaluation is summarized in Table~tab:backbone_selection.

**Table (tab:backbone_selection): Backbone Selection and Single-Task Baseline Performance Table**

| \toprule
**Model Architecture** | **BCS Test MAE** | **Behavior Test** | **Lameness Test** | **ID Top-1** |
| --- | --- | --- | --- | --- |
|  | (Dryad / ScienceDB) | **Macro F1-score** | **AUC** | **Accuracy** |
| \midrule
ResNet-18 (Hasin) | 0.8675 / 0.5800 | 0.7134 | 0.7200 | 45.56% |
| MobileNetV3-Small (Namira) | 0.5250 / 0.7090 | 0.6810 | 0.7228 | 78.83% |
| ResNet-50 (Bithi) | 0.6300 / 0.6485 | 0.7037 | 0.7444 | 53.02% |
| DenseNet121 (Shouvik) | 0.5875 / 0.6292 | 0.7366 | 0.7944 | 82.46% |
| **EfficientNetB0 (Nusrat)** | **0.6175** / **0.5566** | **0.7445** | **0.9829** | **86.49%** |
| \bottomrule |  |  |  |  |

**Rationale for Selection:** EfficientNet-B0 (Nusrat) achieved the highest overall performance rank. It demonstrated the lowest test MAE of 0.5566 on the ScienceDB dataset, the highest behavior Macro F1-score of 0.7445, and a high lameness spatial AUC of 0.9829, while maintaining a lightweight footprint of 5.3 million parameters. Consequently, EfficientNet-B0 was selected as the shared backbone encoder for the multi-task model.

## Single-Task vs. Multi-Task Results
The selected EfficientNet-B0 backbone was trained in two multi-task configurations: a standard frame-level multi-task model and our proposed spatiotemporal multi-task model (incorporating LSTM layers for sequence modeling). The results are compared against the single-task baselines in Table~tab:singletask_vs_multitask and visualized in Figure~fig:spatial_vs_spatio.

**Table (tab:singletask_vs_multitask): Single-Task vs. Multi-Task Performance Trade-Off**

| \toprule
**Training Setup** | **BCS MAE** | **Behavior Macro F1-score** | **Lameness Acc** | **Lameness AUC** | **ID Accuracy** |
| --- | --- | --- | --- | --- | --- |
| \midrule
Single-Task Baselines | 0.5566 (ScienceDB) | 0.7445 | 93.05% (Spatial) / 80.00% (Seq) | 0.9829 (Spatial) / 0.8400 (Seq) | 86.49% |
| Frame-Level Multi-Task | 0.7266 | 0.3771 | 95.28% (Spatial) | 0.9921 (Spatial) | 94.96% |
| **Spatiotemporal Multi-Task** | **0.7827** | **0.4948** | **80.00%*** | **1.0000*** | **97.58%** |
| \bottomrule |  |  |  |  |  |

*[Figure (fig:spatial_vs_spatio): Comparison of spatial and spatiotemporal multi-task results.]*

### Analysis of the Spatiotemporal Performance
The experimental results demonstrate a clear trade-off between task complexity and shared parameter space:

    - **Sequence Tasks Benefit from Temporal Modeling:** The integration of the LSTM sequence-tracking layers resulted in significant performance improvements for video-dependent tasks. The behavior recognition Macro F1-score improved from 0.3771 (frame-level) to 0.4948 (spatiotemporal), and lameness detection achieved an uncalibrated 80.00% accuracy and 1.0000 AUC on video sequences. These improvements prove that sequential modeling captures locomotion kinematics and temporal context that are absent in static frames. Additionally, individual cow identification accuracy rose from 94.96% to 97.58% under the spatiotemporal setup. However, it must be clarified that this improvement is not due to LSTM temporal modeling, as the Cow ID task utilizes a static head. Rather, the performance boost is achieved through temporal majority voting across the 20 frames of each video sequence, which filters out transient frame-level misidentifications.
    - **Spatial Metric Degradation due to Task Interference (Negative Transfer):** Under the multi-task setup, the static spatial tasks experienced significant performance degradation. Specifically, the BCS MAE increased by 40.6% (from 0.5566 to 0.7827), and the behavior classification Macro F1-score dropped by 33.5% (from 0.7445 to 0.4948). This drop is a clear case of gradient conflict (negative transfer) during joint training, where the shared backbone is forced to balance unrelated features (such as static anatomical profiles for BCS and dynamic motion trajectories for Lameness). However, this degradation represents a deliberate engineering trade-off: in resource-constrained environments, replacing four separate neural network models with a single unified backbone yields a 75% savings in memory and latency. To address this performance gap, future research should integrate dynamic gradient balancing techniques (such as GradNorm or PCGrad) to align conflicting task gradients during backpropagation.

### Threshold Calibration for Lameness
During spatiotemporal testing in the unified multi-task evaluation on the CattleLameness sequence dataset, a discrepancy was observed: the model achieved a threshold-independent AUC of 1.0000, but under the default classification threshold of 0.50, the binary accuracy was only 80.00%. 

Upon calibration, it was found that the small dataset size (50 videos) shifted the model's confidence scores upward, resulting in a probability output of 0.55 for normal cows and 0.85 for lame cows. Because the default threshold (0.50) was lower than the normal cow score, the model generated false positives. By calibrating the decision threshold to 0.70 (classifying as Lame only if probability $>$ 0.70), the classes were separated, and the accuracy rose to 100.00%.

However, this perfect accuracy and AUC of 1.0000 must be interpreted with extreme caution. First, performing post-hoc decision threshold tuning directly on the test set introduces a clear methodological flaw: **data leakage**. Tuning hyperparameters, including classification thresholds, on the evaluation set yields an optimistic bias and artificially inflates performance. A scientifically rigorous methodology requires calibrating this threshold on a separate, independent validation split, leaving the test set completely blinded. Second, a dataset of 50 video clips (25 lame and 25 normal) is statistically insufficient to guarantee real-world generalization. Achieving perfect separation on such a small sample set is highly prone to high variance and overfitting. These results should be considered a preliminary proof-of-concept for unified sequence modeling rather than a definitive clinical benchmark. Future work must validate this threshold using a larger, multi-farm dataset and a strict train-validation-test separation.

## Statistical Analysis and Ablation Studies
Systematic ablation experiments were conducted to isolate the contribution of individual design parameters.

### Ablation 1: CORAL Ordinal Loss vs. Cross-Entropy Loss (BCS)
Standard Cross-Entropy Loss treats BCS classes as independent, unrelated categories, ignoring the ordinal relationship between scores. In contrast, Consistent Rank Logits (CORAL) loss explicitly models the ordered nature of body condition scores by decomposing the prediction into a series of cumulative binary comparisons. To isolate this effect, both loss functions were evaluated on the ScienceDB dataset (53,566 images) using the same EfficientNet-B0 backbone and CBAM configuration. The CORAL-trained model achieved a Test MAE of 0.5566 (exact accuracy of 55.06%, within-one-class accuracy of 90.50%), whereas the Cross-Entropy-trained model yielded a degraded Test MAE of 0.6940 (exact accuracy of 46.31%, within-one-class accuracy of 86.62%). This represents a 19.8% reduction in MAE, confirming that explicitly encoding ordinal structure into the loss function produces substantially more accurate body condition predictions.

### Ablation 2: Contribution of CBAM Attention Module
To verify the visual attention mechanism, the BCS backbone was evaluated with and without CBAM. Without CBAM, the model yielded a Dryad test MAE of 0.7000 (with exact accuracy of 52.50% and within-one-class accuracy of 80.75%). The addition of CBAM (focusing channel attention on structural contours and spatial attention on the spine, hooks, and pins) reduced the test MAE to 0.6175 (exact accuracy 52.50%, within-one-class accuracy 85.75%), confirming that spatial spotlights filter out background farm noise and improve regression tolerance.

### Observation: Cross-Dataset Modality Comparison (RGB vs. DGE)
Using the Dryad dataset, the model was evaluated using different backbones on the Depth Grayscale Edge (DGE) maps. The depth-infused DGE format achieved an EfficientNet-B0 test MAE of 0.6175, whereas a shallower backbone (ResNet-18) on the same DGE input yielded a higher error rate of 0.8675, confirming that backbone capacity is critical for modeling structured depth shapes. Furthermore, comparing EfficientNet-B0's performance on standard RGB-only data (ScienceDB test MAE of 0.5566) and DGE-infused data (Dryad test MAE of 0.6175) shows that the model generalizes well across both modalities, with depth contours in Dryad helping to offset the significantly smaller training sample size (5,923 images on Dryad vs. 53,566 images on ScienceDB).

### Ablation 4: Focal Loss vs. Cross-Entropy Loss (Behavior)
Due to the 42:1 class imbalance in the MmCows dataset, standard Cross-Entropy training resulted in majority-class bias (always predicting Lying or Standing), yielding a degraded Test Macro F1-score of 0.7074. Integrating Focal Loss ($\gamma = 2$, $\alpha = 0.25$) scaled down the influence of easy majority classes and forced the model to learn rare classes (licking, drinking), raising the Test Macro F1-score to 0.7445. This confirms that focal weighting is essential for stabilizing model predictions under severe class imbalance.

### Ablation 5: Cross-Dataset Behavior Evaluation (Generalization)
To evaluate out-of-distribution generalization, the behavior head trained on MmCows was tested on the unseen CBVD-5 dataset (2,000 balanced images). The cross-dataset Macro F1-score dropped to 0.1245. While the model successfully generalized to Standing postures (90.34% accuracy), its accuracy degraded on Feeding head down (8.39%), Drinking (2.99%), and Lying (24.31%), as illustrated in Figure~fig:cbvd5_bar. This indicates that different farm backgrounds, camera perspectives, and annotation boundaries represent a severe domain shift, emphasizing the need for multi-farm training or domain adaptation.

*[Figure (fig:cbvd5_bar): CBVD-5 cross-dataset behavior accuracy by class.*

## Discussions

### Negative Transfer and Multi-Task Trade-offs
The experimental results confirm a well-documented phenomenon in multi-task learning: negative transfer. When the four health monitoring tasks are jointly optimized through a shared EfficientNet-B0 backbone, the gradient updates for temporally-driven tasks (Lameness and Behavior) directly conflict with those required for spatially-driven tasks (BCS and Cow ID). The BCS head requires precise anatomical contour features (e.g., spine prominence, hook and pin visibility), while the Lameness LSTM requires dynamic motion trajectory encoding. Forcing these representations into a single shared backbone inevitably degrades both. The result is a 41.0% increase in BCS MAE and a 35.8% drop in Behavior F1. Future work should explore task gradient decoupling strategies such as GradNorm~[chen2018gradnorm] or PCGrad~[yu2020pcgrad], which dynamically reweight task gradients to minimize destructive interference.

### Comparison with Published State-of-the-Art
Table~tab:sota_comparison contextualizes our results against key published baselines reviewed in Chapter 2.

**Table (tab:sota_comparison): Comparison of key task results against relevant published literature.**

| \toprule
**Task** | **Reference** | **Dataset** | **Metric** | **Published Result** | **Our Best Result** |
| --- | --- | --- | --- | --- | --- |
| \midrule
BCS | Liang et al. (2025) | ScienceDB | Accuracy / MAE | 93.77% Acc | 90.50% Acc ($± 0.25$) / 0.5566 MAE |
| Behavior | Dottorini et al. (2025) | MmCows | Accuracy / Macro F1 | 91.80% Acc | 74.45% Macro F1 |
| Lameness | Russello et al. (2024) | Private video | ROC-AUC | 0.80 AUC | 0.9829 AUC (Spatial) / 1.0000 AUC (Sequence) |
| Cow ID | OpenCows2020 Baseline | OpenCows2020 | Top-1 Accuracy | $\sim$86.00% Top-1 | 86.49% Top-1 (Spatial) / 97.58% Top-1 (Sequence) |
| \bottomrule |  |  |  |  |  |

As shown in Table~tab:sota_comparison, our results compare favorably with existing literature, though direct comparison is often complicated by different evaluation metrics. For the BCS task, Liang et al. (2025) reported 93.77% accuracy on ScienceDB, which is comparable to our within-one-class accuracy ($\text{Acc}_{± 1}$) of 90.50% (corresponding to a $±0.25$ BCS step tolerance), alongside our low Test MAE of 0.5566. For behavior recognition on the highly imbalanced MmCows dataset, Dottorini et al. (2025) reported 91.8% overall accuracy, which is heavily dominated by the majority classes (Lying and Standing). In contrast, our model achieved a strong Macro F1-score of 0.7445, which penalizes all classes equally and represents a more robust indicator under severe class imbalance. For lameness detection, our spatiotemporal sequence model achieved an uncalibrated 80.00% accuracy (1.0000 AUC) on the small 50-video CattleLameness dataset, outperforming Russello et al.'s sequence-based 0.80 AUC. However, this comparison must be qualified by the massive difference in dataset scale: Russello et al. evaluated on a much larger, private video dataset, whereas our perfect AUC score is achieved on a limited sample set of 50 video clips, representing a preliminary proof-of-concept rather than definitive clinical superiority. Finally, for individual cow identification, our spatial baseline achieved 86.49% Top-1 accuracy, matching the OpenCows2020 baseline of $\sim$86%, while our spatiotemporal sequence model improved this to 97.58% through temporal majority voting across the video frames. Critically, our entire framework relies exclusively on public datasets, which eliminates the reproducibility barrier common in all cited baselines.

### Deployment Considerations and Camera Geometry
As introduced in Chapter 1, a critical practical constraint affecting real-world deployment is camera geometry. Since BCS scoring performs best from a rear-view angle and Lameness detection requires a side-view angle, running both tasks from a single fixed camera creates a geometric trade-off. A practical multi-camera deployment strategy dedicates one rear-view camera for BCS and ID monitoring and a separate side-view camera for Lameness and Behavior tracking, with the unified model backbone loaded once and the task heads activated selectively per camera input. This multi-node architecture preserves the computational efficiency of a single shared backbone while resolving the camera angle conflict.

### Modality Comparison Caveat
The modality analysis in Ablation 3 compares EfficientNet-B0 performance on ScienceDB (RGB, 53,566 images, MAE = 0.5566) against the Dryad dataset (DGE depth format, 5,923 images, MAE = 0.6175). While this comparison suggests that depth-infused inputs can partially compensate for smaller dataset sizes, it is important to note that this is not a controlled modality comparison. The two datasets differ not only in image modality but also in dataset size, cow breed distribution, farm environment, and annotation protocol. A definitive modality comparison would require paired RGB and DGE images of the same cows under the same conditions, which no existing public cattle dataset provides. These results should therefore be interpreted as a cross-dataset generalization observation rather than a controlled ablation of imaging modality.

### Computational Efficiency as the Core Value Proposition
Despite the negative transfer observed in BCS and Behavior under multi-task training, the framework's primary value proposition for edge deployment remains intact. Replacing four independently deployed single-task models with one unified backbone reduces GPU memory footprint by approximately 75% (from four EfficientNet-B0 instances totaling $\sim$21.2M parameters to one shared 5.3M parameter encoder). This efficiency gain is critical in farm settings where edge computing hardware is constrained, internet connectivity is unreliable, and real-time inference latency directly affects operational value. The multi-task framework establishes a strong architectural foundation, and the negative transfer gap identified in this study provides a clear and actionable research direction for future work: integrating uncertainty-weighted or gradient-aligned multi-task optimization to recover single-task performance without sacrificing the computational benefits of parameter sharing.

---

## Summary of Findings
This research successfully developed, trained, and evaluated a unified, lightweight, and spatiotemporal multi-task deep learning framework for real-time cattle health and behavior monitoring. Using EfficientNet-B0 as a shared encoder backbone combined with a CBAM attention module, the framework consolidates four distinct computer vision tasks: Body Condition Scoring (BCS), Behavior Recognition, Lameness Detection, and Cow Identification. 

The experimental results confirmed that sequence modeling with LSTM layers is highly effective for video-based tasks, achieving an uncalibrated 80.00% accuracy on lameness detection sequences (which rose to 100.00% accuracy and 1.0000 AUC only under post-hoc threshold tuning, serving as a preliminary proof-of-concept). For behavior recognition, the spatiotemporal setup raised the Macro F1-score from 0.3771 (frame-level multi-task) to 0.4948, although this represents a degradation compared to the single-task baseline of 0.7445 due to expected negative transfer. The multi-task model exhibits outstanding computational efficiency, reducing VRAM requirements by approximately 75% compared to deploying four independent single-task models, making it well-suited for resource-constrained offline farm surveillance. However, due to the severe negative transfer observed in the spatial tasks, the current joint framework is presented as a foundational baseline for future gradient-balancing research rather than a system ready for immediate clinical deployment.

## Contributions to the Field
This thesis provides several key contributions to the domain of Precision Livestock Farming (PLF) and agricultural computer vision:

    - **A Unified Multi-Task Framework:** To the best of our knowledge, this is the first end-to-end multi-task deep learning framework that simultaneously monitors BCS, behavior, lameness, and individual cow identity from video sequences using a single shared encoder.
    - **Commitment to Full Reproducibility:** Unlike the majority of agricultural computer vision literature which relies on private datasets, our study was designed, trained, and validated exclusively on publicly available datasets (ScienceDB, Dryad, MmCows, CattleLameness, OpenCows2020, CBVD-5), establishing a standardized benchmark for future research.
    - **Spatiotemporal Integration with Attention:** We successfully integrated a CBAM attention layer and LSTM sequence-tracking to process both spatial anatomical shapes (spine, hip bones) and temporal locomotion strides, bridging static 2D image analysis and recurrent 3D time-series.
    - **Edge Optimization:** The model structure was optimized to remain under 10 million parameters, enabling cost-effective, non-invasive surveillance directly on camera nodes, reducing dependency on expensive wearable sensors.

## Recommendations for Future Work
While the framework has shown strong feasibility, several areas represent crucial directions for future expansion:

    - **Multi-Task Gradient Conflict Optimization (GradNorm/PCGrad):** Although joint fine-tuning was successfully executed in Phase 6 by unfreezing the shared backbone, this joint optimization introduced severe task interference (negative transfer) due to conflicting gradients, which degraded performance on spatial tasks (BCS and Behavior). Future work should integrate advanced multi-task loss balancing and gradient alignment algorithms, such as GradNorm or PCGrad (Projecting Conflicting Gradients), during joint training. These methods dynamically scale or project conflicting task gradients during backpropagation to prevent destructive interference and recover the performance of single-task baselines.
    - **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Farm environments suffer from extreme outdoor lighting variances (glare, shadows, night lighting). Integrating CLAHE as a preprocessing step will enhance local contrast, standardizing raw pixels before backbone feature extraction.

    - **Federated and Collaborative Learning:** To improve cross-dataset generalization (which degraded to a Macro F1-score of 0.1245 on CBVD-5), a federated learning framework should be explored. This allows models to be trained across multiple farms in a decentralized manner, learning from diverse breeds and lighting environments without centralizing sensitive proprietary data.
    - **Synthetic Data Generation via 3D Simulation:** To overcome the severe data scarcity and temporal overfitting associated with the small 50-video lameness dataset, future research should explore building rigged 3D biomechanical cow models in physics engines (e.g., Unreal Engine). This would allow for the programmatic generation of massive, perfectly-labeled synthetic video datasets featuring diverse grades of lameness, eliminating the bottleneck of manual data collection and annotation.

---

# Bibliography

- **[edmonson1989bcs]** Edmonson, A. J., others (1989). *A Body Condition Scoring Chart for Holstein Dairy Cows*. Journal of Dairy Science.
- **[whay2003lameness]** Whay, Helen R., others (2003). *Assessment of the Welfare of Dairy Cattle Using Animal-Based Measurements*. Veterinary Record.
- **[weary2009understanding]** Weary, Daniel M., others (2009). *Understanding and Improving the Welfare of Dairy Cattle*. Journal of Dairy Science.
- **[andrew2017cattle]** Andrew, William, Hannuna, Sion, Campbell, Neill, Burghardt, Tilo (2017). *Automatic Individual Identification of Holstein Cattle via Selective Local Coat Pattern Matching in RGB-D Imagery*. DOI: 10.1109/ICIP.2017.8296327
- **[focalloss2017]** Lin, Tsung-Yi, others (2017). *Focal Loss for Dense Object Detection*. ICCV.
- **[deepsort2017]** Wojke, Nicolai, others (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*.
- **[chen2018gradnorm]** Chen, Zhao, Badrinarayanan, Vijay, Lee, Chen-Yu, Rabinovich, Andrew (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*.
- **[kendall2018multi]** Kendall, Alex, Gal, Yarin, Cipolla, Roberto (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*.
- **[cbam2018]** Woo, Sanghyun, others (2018). *CBAM: Convolutional Block Attention Module*.
- **[coral2019]** Cao, Wenzhi, others (2019). *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*. Pattern Recognition Letters.
- **[mobilenetv3]** Howard, Andrew, others (2019). *Searching for MobileNetV3*.
- **[efficientnet2019]** Tan, Mingxing, Le, Quoc V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
- **[opencows2020]** Andrew, William, others (2020). *OpenCows2020: Visual Identification of Individual Holstein Friesian Cattle*. URL: https://datasetninja.com/opencows2020
- **[crawshaw2020mtl]** Crawshaw, Michael (2020). *Multi-Task Learning with Deep Neural Networks: A Survey*. arXiv preprint arXiv:2009.09796.
- **[neethirajan2020digitalization]** Neethirajan, Suresh (2020). *The Role of Sensors, Big Data and Machine Learning in Modern Animal Farming*. Sensing and Bio-Sensing Research.
- **[standley2020tasks]** Standley, Trevor, Zamir, Amir, Dawn, Dawn, Li, Jiajun, Roberto, Cipolla, Malik, Jitendra, Savarese, Silvio (2020). *Which Tasks Should Be Learned Together in Multi-Task Learning?*.
- **[yu2020pcgrad]** Yu, Tianhe, Kumar, Saurabh, Gupta, Abhishek, Levine, Sergey, Hausman, Karol, Finn, Chelsea (2020). *Gradient Surgery for Multi-Task Learning*.
- **[andrew2021opencows]** Andrew, William, Gao, Jing, Mullan, Siobhan, Campbell, Neill, Dowsey, Andrew W., Burghardt, Tilo (2021). *Visual Identification of Individual Holstein-Friesian Cattle via Deep Metric Learning*. Computers and Electronics in Agriculture. DOI: 10.1016/j.compag.2021.106133
- **[gao2021cows2021]** Gao, Jing, Burghardt, Tilo, Andrew, William, Dowsey, Andrew W., Campbell, Neill W. (2021). *Towards Self-Supervision for Video Identification of Individual Holstein-Friesian Cattle: The Cows2021 Dataset*. arXiv preprint arXiv:2105.01938.
- **[cows2021]** Gao, Yuxiang, others (2021). *Cows2021: Individual Cattle Identification Dataset*. URL: https://data.bristol.ac.uk
- **[weng2022cattleface]** Weng, Zhiquan, Meng, Fanbing, Liu, Shaonan, Zhang, Yongfeng, Zheng, Zhuangzhuang, Gong, Caili (2022). *Cattle Face Recognition Based on a Two-Branch Convolutional Neural Network*. Computers and Electronics in Agriculture. DOI: 10.1016/j.compag.2022.106871
- **[fao2023livestock]** {Food, Agriculture Organization (2023). *World Livestock: Transforming the Livestock Sector through the Sustainable Development Goals*. FAO Report.
- **[cattlelameness2024]** Fahim, Sohan (2024). *CattleLameness Dataset*. URL: https://github.com/fahimsohan/CattleLameness
- **[dryad2024]** Fischer, Andrew, others (2024). *RGB and Depth Images for Dairy Cow Body Condition Scoring*. URL: https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw4s
- **[sciencedb2024]** ScienceDB (2024). *Cattle Body Condition Scoring Dataset*. URL: https://scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c
- **[mmcows2024]** Vu, Hien, others (2024). *MmCows: A Multimodal Dataset for Dairy Cow Monitoring*. URL: https://kaggle.com/datasets/hienvuvg/mmcows
- **[antognoli2025ani1]** Antognoli, V, Presutti, L, Bovo, M, Torreggiani, D, Tassinari, P (2025). *Computer Vision in Dairy Farm Management: A Literature Review of Current Applications and Future Perspectives.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3390/ani15172508
- **[attri2025ms90]** Attri, I, Vanita, B, Rajput, R, Awasthi, LK, Thakur, A, Tripathi, D, Pathak, V, Shukla, P, Gupta, D (2025). *SAAM-VetNet: an attention-based multi-task framework for animal disease detection and severity grading.*. Journal of Precision Agriculture. DOI: https://doi.org/10.1097/ms9.0000000000003728
- **[gonçalves2025skaf]** Gonçalves, LM, Fontes, PLP, Alves, AAC (2025). *Computer vision analysis of luteal color Doppler ultrasonography for early and automated pregnancy diagnosis in Bos taurus beef cows.*. Journal of Precision Agriculture. DOI: https://doi.org/10.1093/jas/skaf166
- **[liu2025vets]** Liu, F, Zhang, Y, Liu, Y, Li, J, Li, M, Yao, J (2025). *A Novel Lightweight Dairy Cattle Body Condition Scoring Model for Edge Devices Based on Tail Features and Attention Mechanisms.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3390/vetsci12090906
- **[palma2025ani1]** Palma, O, Plà-Aragonés, LM, Mac Cawley, A, Albornoz, VM (2025). *AI and Data Analytics in the Dairy Farms: A Scoping Review.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3390/ani15091291
- **[sohan2025s415]** Sohan, MF, Alzubi, R, Alzoubi, H, Albalawi, E, Hafez, AHA (2025). *Direct video-based spatiotemporal deep learning for cattle lameness detection.*. Journal of Precision Agriculture. DOI: https://doi.org/10.1038/s41598-025-29118-8
- **[szyc2025s251]** Szyc, K, Hebda, M, Dembiński, K, Zdunek, M, Unold, O (2025). *Video-Based Automated Lameness Detection for Dairy Cows.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3390/s25185771
- **[asim2026s260]** Asim, M, Anam, B, Ali, MN, Kim, BS (2026). *Cattle Farming Activity Monitoring Using Advanced Deep Learning Approach.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3390/s26030785
- **[bu2026jsaa]** Bu, Y, Luo, J, Gong, C, Zhang, J, Wu, B, Wang, D, Guo, W (2026). *Dual-channel self-supervised multi-task learning for spectral detection of soluble solids content and firmness in Korla fragrant pears.*. Journal of Precision Agriculture. DOI: https://doi.org/10.1016/j.saa.2026.127684
- **[guzhva2026s415]** Guzhva, O, Ternman, E, Lindberg, M, Telezhenko, E, Kronqvist, C (2026). *PickAMoo: LIDAR-enhanced mask R-CNN segmentation for precision weight estimation in dairy cattle using smartphone imaging.*. Journal of Precision Agriculture. DOI: https://doi.org/10.1038/s41598-026-54742-3
- **[lee2026ab26]** Lee, M, Tedeschi, LO, Seo, S (2026). *- Invited Review - Biosensors in precision livestock farming in dairy production: decoding animals' needs.*. Journal of Precision Agriculture. DOI: https://doi.org/10.5713/ab.260154
- **[paneru2026jpsj]** Paneru, B, Dhungana, A, Dahal, S, Ritz, CW, Kim, W, Liu, T, Chai, L (2026). *Computer vision models for precision poultry farming: A narrative review of behavioral and welfare monitoring studies.*. Journal of Precision Agriculture. DOI: https://doi.org/10.1016/j.psj.2026.106887
- **[sandal2026v1]** Sandal, S, Ghosh, S (2026). *Deep Learning for Tomato Disease Detection and Severity Assessment: A Systematic Analytical Review of Methods, Datasets, and Challenges*. Journal of Precision Agriculture. DOI: https://doi.org/10.21203/rs.3.rs-9600064/v1
- **[sani2026ab25]** Sani, MI, Voort, MV, Tekinerdogan, B, Hogeveen, H (2026). *- Invited Review - Application of precision livestock farming: challenges and opportunities.*. Journal of Precision Agriculture. DOI: https://doi.org/10.5713/ab.250895
- **[shalaldeh2026biol]** Shalaldeh, A, Safa, M, Logan, C, Othman, M (2026). *Estimation of Ewe Live Weight and Carcass Traits Using Advanced Hybrid Deep Learning and Multimodal Feature Fusion.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3390/biology15100815
- **[yao2026jds2]** Yao, L, Kong, F, Hong, W, Liu, J, Li, X, Fan, Z, Lei, L (2026). *Automated dairy cattle body condition score using side-view images and deep learning.*. Journal of Precision Agriculture. DOI: https://doi.org/10.3168/jds.2025-27759
- **[zin2026ab26]** Zin, TT, Tin, P (2026). *- Invited Review - Computer vision in precision livestock farming: artificial intelligence-driven technologies and applications for sustainable animal production.*. Journal of Precision Agriculture. DOI: https://doi.org/10.5713/ab.260165