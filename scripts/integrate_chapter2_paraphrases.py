import sys
import os

tex_path = "cattle_thesis_p3_latex/chapters/chapter_2.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# 31 Paraphrased paragraphs provided by user with LaTeX citations accurately embedded
paraphrased_list = [
    # P1
    "It is not necessary for a unified system for cattle monitoring to have exactly the same components for all the tasks. Some of the components of the system, such as cameras, preprocessing, and common features, can be shared among the tasks, while others remain specific to each individual task. Therefore, the integration is not only about merging multiple predictions. It also tries to find which features are useful for all tasks and which features should stay task-specific.",
    
    # P2
    r"BCS Automation is highly dependent on the cow's anatomy. Edmonson et al.~\cite{edmonson1989bcs} proposed an index of scoring while Ferguson et al.~\cite{ferguson1994bcs} investigated the correlation between anatomical traits and certain BCS values. From a point of view of computer vision, considering the whole body mass of the cow is not sufficient. In order for the model to work, it should detect the amount of tissue covering the body, body shape, and specific body parts. This is why the camera position and selected body area matter.",
    
    # P3
    r"Another approach uses depth images for better representation of the body shape of a cow. In particular, Rodriguez Alvarez et al.~\cite{rodriguez2018bcs} employ convolutional neural networks along with depth images and validate the prediction of BCS by means of various error thresholds. This approach relies on the combination of body geometry and features rather than the handcrafted features alone. The paper also proves that exact accuracy and tolerance-based accuracy have to be evaluated separately because of the fact that BCS scores are ordinal.",
    
    # P4
    r"Another methodology makes use of ordinary RGB images to detect visual cues associated with body condition. In the case of Liu et al.~\cite{liu2025vets}, most of the feature extraction takes place in the tail area and is done using the EfficientNet-B0 architecture with channel and spatial attention. Their work involves the selection of a specific body part and an efficient network architecture, while distillation assists in making the model size smaller. The fact that BCS prediction can be tailored through a combination of the specific body part and architecture has been illustrated.",
    
    # P5
    r"Camera view influences the information that can be obtained by the model as well. Yao et al.~\cite{yao2026jds2} select side-view pictures of the cattle using a ConvNeXt-based regression model and cow-level cross-validation. One method is focused on the tail of the animal; therefore, it observes just a small portion of the body, whereas another method selects side-view pictures and considers a bigger part of the cow's body. Both methods are different not only in their models but also in the parts of the cow they observe.",
    
    # P6
    r"The research done on cattle measurement also reveals that there is a need for distinction between image size and true physical measurements. The authors Guzhva et al.~\cite{guzhva2026s415} came up with a system called PickAMoo, which uses smartphone images, LiDAR scaling, and Mask R-CNN segmentation in order to estimate weight. Even though the purpose of estimating the weight is different from estimating BCS, this method is relevant since the estimation is done when the animal has been separated from the image, and pixels are related to physical measurements.",
    
    # P7
    r"The common factor among all these papers can be seen in three aspects that make good components for BCS prediction. First is the use of images that have proper features of the body and its orientation. Second is the use of an appropriate technique to extract such information, for example, cropping, body contour, depth map, or attention. Third is selecting an appropriate prediction technique that is suited to ordered BCS. CORAL can handle ordering of output classes~\cite{coral2020}, but it will not perform well if the input class does not have appropriate condition features. Thus, body-shape information will be considered as part of the representation, and ordinal prediction as the model choice.",
    
    # P8
    "These demonstrate two different aspects of behavioral data: temporal information and identity consistency. Video in high density helps with motion analysis in short term. Identity of the cows helps with the test being done on different cows and across time. The one does not substitute for the other. This means that a study that requires both needs to factor in both and not just choose the one with many images.",
    
    # P9
    "Contextual background information can also be useful and detrimental. While a dip or feeding barrier could aid in learning about feeding/drinking, the problem is that it might learn that one task tends to happen in one camera quadrant. While the removal of the entire background will make the shortcut less obvious, it will also take out some interaction information. It is not about whether the context can be seen as positive or negative. What matters is the amount of context that is helpful to comprehend the activity and its ability to remain relevant despite changes in the environment.",
    
    # P10
    "This is linked to the cow-focused thesis in which the cow is still the key element of the visual reference and where posture, movement and context relationship can all be used if it is necessary. It also allows for the use of a class-based assessment – a model can perform well on common rest behaviours and poorly on rare or more dynamic ones. This can be obscured by the overall accuracy if class frequency is correlated with source.",
    
    # P11
    r"The visual identification of cattle requires certain characteristics that differ from one animal to another. Andrew et al.~\cite{andrew2017cattle} use the technique of local coat pattern matching through RGB-D images. This technique provides us with an example of how the useful characteristic of identity is made explicit: local patterns are able to identify Holstein-Friesian cattle. This technique also indicates an important drawback that remains relevant for today's models. The identity-related feature must be visible in both images that are being compared.",
    
    # P12
    "Nevertheless, tracking and Re-ID are still different tasks. While tracking maintains association of an animal when it stays continuously visible, re-identification must restore this association when there is a gap in time or change in viewing conditions. Information about tracking may be used for training of the Re-ID model, but an id of the temporary tracking does not correspond to biological ID of the cow. Multi-camera and cross-setting assessment would be helpful since it examines bigger shifts than a continuous video sequence.",
    
    # P13
    r"Another challenge is long-term identification. In the dataset called BECA proposed by Zhang et al.~\cite{zhang2026beca}, there is a large population setting for beef cattle and a longitudinal setting. Unlike short-term pairing of dairy cattle with unique coat patterns, an animal may look different depending on its size, its stance, and the part of the body that is visible, and having a larger population increases visual similarity. Thus, the Re-ID model needs to be tested not just on the cows used in the training phase, but also on its ability to identify across populations and time.",
    
    # P14
    "Similarly, foreground processing and anatomy processing have to take into account the need to preserve the data needed for identification purposes. While deleting the background may force the model to concentrate on the cow, deleting too much of it by reducing it to just a silhouette or skeleton may lead to loss of critical coat features. Retaining the full environment will cause the opposite effect by teaching the model the usual position of the cow. Consequently, Re-ID will require a selective representation technique that minimizes environmental shortcuts but maintains cow-specific characteristics.",
    
    # P15
    "The localization determines the cow on which the model needs to be applied. This becomes very crucial when there are multiple cows in the same image since the predicted class or behavior belongs to that particular cow only. The bounding box provides an effective ROI, but also maintains some context within the image. Image cropping can normalize the scales and enable the encoder to focus, but alters the body-background ratio.",
    
    # P16
    r"There are several modern object detectors that can help to locate this region. RT-DETR developed by Zhao et al.~\cite{zhao2024rtdetr} uses an end-to-end transformer detector optimized for speed-accuracy trade-off. Instance segmentation approaches are richer than bounding box. Mask R-CNN extends the detection network with the mask branch and thus makes it predict not only the position but also object specific mask~\cite{he2017maskrcnn}. To be more precise, the bounding box says where the cow is, while the mask defines its pixels.",
    
    # P17
    r"Promptable segmentation decouples the process of selecting the target from creating its mask. Promptable segmentation model, Segment Anything by Kirillov et al.~\cite{kirillov2023sam}, is capable of generating masks for prompts such as points and boxes. Promptable segmentation in SAM 2, being memory-based, is applicable to both images and video~\cite{ravi2024sam2}. This implies that in a cattle pipeline, for instance, we can first detect the cow, and then prompt the segmentation model to generate the mask, without the need to train a new cattle-specific segmentation model from scratch.  The final mask will depend upon how the subject was chosen and prompted.",
    
    # P18
    "Masking could also be applied with different degrees of strength. With hard foreground masking, all content outside of the estimated boundary is eliminated. The softer form of masking will lessen the effect of the background while retaining some context along with uncertain boundary information. It is a matter of selecting the appropriate approach for the particular task. Useful body contours are required for BCS, context is needed for behavior, and coat information for Re-ID. Therefore, the quality of a mask should be assessed together with the performance of the downstream task, rather than only by how clean the mask appears.",
    
    # P19
    "Temporal modeling provides the information related to the changes through time. This information varies depending on representation choice. RGB sequences store information about the visual aspect and posture of the subject, while keypoints sequences provide a compact representation of the body configuration. Video modeling allows for learning both spatial and temporal representations at the same time. These approaches preserve different information and require different computational and labeling effort.",
    
    # P20
    r"SlowFast network explicitly separates the aspects of appearance and motion using two streams with different frame rates~\cite{feichtenhofer2019slowfast}, where the first stream concentrates on spatial information and the second captures faster changes. SlowFast network is also used as the benchmarking technique in the CVB application~\cite{zia2023cvb}, making connection between general video recognition technique and behavior of cattle. It provides a stronger motivation for temporal modeling approach other than simple assumption of more being better.",
    
    # P21
    r"A temporal convolution network may also be used. In comparing convolutional and recurrent sequence models, Bai et al.~\cite{bai2018tcn} found that temporal convolutions have shown promise in several sequence tasks. These models leverage temporal properties without needing to include a recurrent hidden state at each step. In relation to cattle behavior, this provides justification for using a light temporal network in place of image features when many visualizations must be compared.",
    
    # P22
    r"Another way to encode motion is to consider body landmark relations over time. ST-GCN by Yan et al.~\cite{yan2018stgcn} is a model that uses graphs to represent the skeleton of a moving human being with connections between nodes in space and in time. This approach is applicable to the problem of interest due to modeling relations between body parts rather than modeling coordinates separately. Still, its use in the case of cattle depends on having correct landmarks definition and accurate estimation of poses. Therefore, the pose sequence should be seen as a different representation category and not a substitution for RGB video.",
    
    # P23
    "The approaches prove that different temporal inputs address different problems. Taking the average of several frame features provides the model with more observable appearance, while a temporal model can also make use of the sequence of those frames. Pose features concentrate on body pose, while dense video contains more motion and contextual information. This choice is based on the type of action and sampling period. Hence, this thesis makes use of temporal modeling as a representation problem rather than scaling up the model.",
    
    # P24
    r"Success of the algorithm in a certain known situation does not mean that the representation learned by it will be useful in another place. Geirhos et al.~\cite{geirhos2020shortcut} demonstrate that deep learning models employ some simple correlations that cannot be applied when the environment is altered. It becomes especially crucial if the object and background are seen in many cases. For instance, in cattle pictures, the same pen, lighting, and viewpoint can occur repeatedly, this makes it easier for a model to find scene-specific shortcuts.",
    
    # P25
    "In this sense, the design of the evaluation procedure is part of the representation problem. The mask will seem relevant in one split since the model is dealing with an easy identity and localization task. The use of multiple datasets will bring variation visually while one dataset is still linked to one class. Evaluation and training should be aligned with what is claimed in this case. This brings more validity to testing whether the use of cattle information helps with transfer rather than improving performance within one known data set.",
    
    # P26
    r"The use of multitask learning (MTL) allows the combining of the three models into one system. The most basic design uses a single encoder and different heads for each task. This helps in saving computational resources and encourages the use of common features. However, the use of MTL requires more than the fact that there are three inputs containing cattle. Crawshaw~\cite{crawshaw2020mtl} notes that architecture, task relationships, and optimization all have roles in MTL performance. Thus, the sharing of the resources should have something more compelling than just the fact that the three inputs have cattle.",
    
    # P27
    r"Some of the methods used in MTL attempt to ensure the balance of task updates. In Kendall et al.~\cite{kendall2018multi}, loss weighting is done using task uncertainty. The other technique is called GradNorm, which uses gradients for modifying the task weight depending on learning pace~\cite{chen2018gradnorm}. These techniques seek to prevent dominance of a particular task during joint learning due to either loss magnitude or pace. They control the strength of each task's update to the network. However, they do not select features to be shared visually.",
    
    # P28
    r"PCGrad solves a completely different aspect of the problem through the modification of gradients during conflicting task updates~\cite{yu2020pcgrad}. Thus, gradient interaction becomes a component of the optimization process. Modification of the magnitude of an update and modification of a conflicting direction are two separate things, and therefore, PCGrad cannot be equated to loss weighting. Both these techniques remain within a selected architecture, thus allowing researchers to explore optimization as one of the causes of negative transfer.",
    
    # P29
    r"Other approaches modify how the features are shared. Cross-Stitch Networks teach how to share the activation of task-specific networks~\cite{misra2016crossstitch}. The Multi-Task Attention Network proposed by Liu et al.~\cite{liu2019mtan} uses task-specific attention to choose information from a common pool of features. Such approaches do not enforce that all tasks use the same feature route. This is especially helpful when tasks are similar but require different information, such as for cattle morphology, behavior, and individuality.",
    
    # P30
    "The research approach can be seen as a task-specific cattle-oriented approach. It seeks to preserve valuable common attributes of the cattle while being flexible enough to place different levels of attention on morphology, behavior, and identification. The comparison is mostly going to be made between generic RGB baselines and hard sharing approaches. In particular, the task performance and negative transfer will be of the main interest. The study also aims to identify which features can be shared, which ones require task-specific processing, and how these decisions influence the unified system.",
    
    # P31
    "The literature review above serves as the foundation for subsequent chapters. Chapter 3 outlines the criteria and constraints of the task. Chapter 4 explains the protocols used for the data, their representations, and model comparisons. Chapter 5 applies the results gathered to readdress the research questions."
]

prefixes = [
    "This does not mean that all three tasks should use",
    "Automated BCS has a strong anatomical basis",
    "One approach uses depth images to capture body geometry",
    "Another approach looks for useful condition cues in normal RGB images",
    "Viewpoint also changes what information is available",
    "Research on cattle body measurement also shows why image size",
    "Together, these studies show three important design choices",
    "This shows two separate properties of behavior data",
    "Background context can also help and hurt",
    "This connects behavior recognition to the cattle-centered",
    "Visual cattle identification depends on features",
    "Tracking and Re-ID are still different tasks",
    "Long-term recognition adds another challenge",
    "Foreground and anatomy processing must also protect",
    "Localization decides which cow the model should analyze",
    "Modern detectors provide different ways to find this region",
    "Promptable segmentation separates target selection",
    "Masking can also be used at different strengths",
    "Temporal modeling adds information about change over time",
    "SlowFast networks make the difference between appearance",
    "Temporal convolution is another option",
    "A more structural option is to model body landmarks across time",
    "These methods show that different temporal inputs answer different questions",
    "Good performance in one familiar setting does not guarantee",
    "This means that evaluation design is part of the representation question",
    "MTL provides a way to combine the three task-specific models",
    "Some MTL methods focus on balancing the task updates",
    "PCGrad addresses a different part of the problem",
    "Other methods change how features are shared",
    "The proposed research direction is therefore a task-conditioned",
    "This literature review provides the basis for the remaining chapters"
]

assert len(paraphrased_list) == len(prefixes) == 31

def main():
    with open(tex_path, "r", encoding="utf-8") as f:
        t = f.read()

    for idx in range(31):
        prefix = prefixes[idx]
        pos = t.find(prefix)
        if pos == -1:
            print(f"FAILED to find prefix {idx+1}: {prefix}")
            sys.exit(1)
        end_pos = t.find("\n\n", pos)
        if end_pos == -1:
            end_pos = pos + len(prefix) + 200
        
        # Replace in text
        t = t[:pos] + paraphrased_list[idx] + t[end_pos:]
        print(f"Replaced P{idx+1} successfully!")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(t)

    print(f"\nALL 31 PARAGRAPHS INTEGRATED INTO {tex_path} SUCCESSFULLY!")

if __name__ == "__main__":
    main()

