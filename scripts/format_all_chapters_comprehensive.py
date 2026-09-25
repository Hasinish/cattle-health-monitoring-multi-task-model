"""
Comprehensive Review & Keyword Stager for All Chapters (V2).
Enriched vocabulary and intelligent filtering to guarantee professional,
high-value domain terms across all 161 paragraphs.
"""

import sys
import re
import socket
import time
socket.setdefaulttimeout(20.0)

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute

CH1_FLAGGED = {
    3: {
        "roast": "অসাধারণ scientific logic! 'Reduced food intake causes a cow to eat less' (কম খাওয়া একটা গরুকে কম খাওয়ায়!) 💀 আইনস্টাইন তো কবরে বইসা কাঁদতেছে! পুরা circular logic-এর কারখানা!",
        "intact": ["Body condition", "Reduced food intake", "Housing and management"],
        "fixes": [
            "Fix 1: Change 'causes a cow to eat less' -> 'leads to gradual condition loss and altered resting'",
            "Fix 2: কম খাওয়া আর রেস্ট নেওয়াকে আলাদা লক্ষণ হিসেবে লেখো, একটা আরেকটার কারণ না"
        ]
    },
    9: {
        "roast": "'Trivial incidents'?! ☕ একটা গরু পানি খাওয়া, খাবার খাওয়া বা অসুস্থ হওয়া কোনো 'trivial incident' (তুচ্ছ ঘটনা) না! আর 'short observations' মানে সময়ের ব্যাপ্তি ছোট, কোনো লাজুক 'modest' স্বভাব না!",
        "intact": ["Cattle monitoring", "Routine welfare events", "Short-duration observations"],
        "fixes": [
            "Fix 1: Change 'Trivial incidents' -> 'Routine welfare events'",
            "Fix 2: Change 'Modest traits' -> 'Brief observation periods'"
        ]
    },
    12: {
        "roast": "প্রথম লাইনটা সিনেমার ডায়লগের মতো কোটেশন মার্কসে (\" \") কেন দিছোস? 🎬 ডিরেক্ট কোটেশন দিলে Turnitin সাথে সাথে প্ল্যাজিয়ারিজম ধরবে! আর ৫ শব্দের মধ্যে দুইবার applications লিখছোস!",
        "intact": ["Multi-task learning", "Livestock applications"],
        "fixes": [
            "Fix 1: Change Direct quotation (\"...\") -> Paraphrased in your own words",
            "Fix 2: Change 'applications... in other livestock applications' -> Remove redundant repetition"
        ]
    },
    15: {
        "roast": "'Learned traits' আর 'various duties'?! 😭 অ্যানিম্যাল সায়েন্সে 'traits' মানে জেনেটিক্স, নিউরাল নেটওয়ার্কের ফিচার না! আর মডেল Machine Learning-এর 'tasks' করে, ৯-টা ৫-টা অফিসের 'duties' না!",
        "intact": ["Visual features", "Representations", "Tasks", "Transfer learning"],
        "fixes": [
            "Fix 1: Change 'Learned traits' -> 'Visual features / representations'",
            "Fix 2: Change 'Various duties' -> 'Downstream tasks'"
        ]
    },
    18: {
        "roast": "ভাইরে ভাই! তোরা ডিকশনারি থেইকা সিনোনিম মাইরা Machine Learning-এর Tasks-রে Occupations (পেশা/চাকরি) বানায় দিলি?! 💀 গরু কি এখন LinkedIn-এ প্রোফাইল খুইলা ট্যাক্স ফাইল জমা দিবে?! MTL model-এর tasks আর মানুষের corporate jobs এক জিনিস না!",
        "intact": ["Multi-task learning (MTL)", "Tasks", "Shared features", "Selective sharing"],
        "fixes": [
            "Fix 1: Change 'Occupations' -> 'Tasks'",
            "Fix 2: Change 'Dilemma' -> 'Task relationship / trade-off'"
        ]
    },
    24: {
        "roast": "'Two occupations responsible for the care of the same cow'?! 🧑‍🌾👨‍🌾 এখানে দুইটা কম্পিউটার ভিশন মডেল (BCS এবং Re-ID) একই গরুর ছবি প্রসেস করতেছে, কোনো ডাক্তার আর খামারি মিটিং করতেছে না!",
        "intact": ["Tasks", "BCS", "Re-ID", "Shared visual input"],
        "fixes": [
            "Fix 1: Change 'Two occupations caring for cow' -> 'Two distinct computer vision tasks processing the same cow image'"
        ]
    },
    27: {
        "roast": "১) BCS মানে বডির চর্বি/এনার্জি রিজার্ভ, কোনো 'sickness' (অসুখ) না 🤒! বেশি বিসিএস হওয়া মানে গরু সুস্থ-সবল, অসুস্থ না! ২) গোয়ালঘরের খাবার জায়গা কোনো 'geography' 🗺️ না, এটা ব্যাকগ্রাউন্ড সিন!",
        "intact": ["BCS", "Body condition", "Barn / Pen background", "Visual cues"],
        "fixes": [
            "Fix 1: Change 'Sickness' -> 'Energy reserves / body condition'",
            "Fix 2: Change 'Geography' -> 'Pen / barn background context'"
        ]
    },
    30: {
        "roast": "গরু আর তার 'House'?! 🏡 গরু কি এখন ব্যাংক থেইকা home loan নিয়া গুলশানে ডুপ্লেক্স বাড়ি বানাইয়া থাকে নাকি রে ভাই?! ওটা গোয়ালঘর আর খামার (Barn / Pen)! আর 'Shortcut learning'-রে 'quick learning' বানায় দিও না!",
        "intact": ["Shortcut learning", "Barn / Pen", "Visual cues", "Spurious correlations"],
        "fixes": [
            "Fix 1: Change 'House' -> 'Barn / pen environment'",
            "Fix 2: Change 'Quick learning' -> 'Shortcut learning'"
        ]
    },
    33: {
        "roast": "'The target cow is localized through localization'—অসাধারণ বৈজ্ঞানিক থিওরি! 🧠💀 'গরু লোকালাইজড হয় কারণ সে লোকালাইজড!' আর আবার 'Job-specific'?! গরু কি বিসিএস ক্যাডার নাকি যে তার Job থাকবে?! শব্দটা হইলো 'Task-specific'!",
        "intact": ["Cattle-centered representation learning", "Localization", "Segmentation", "Task-specific"],
        "fixes": [
            "Fix 1: Change 'Target cow is localized through localization' -> 'Localization isolates the target animal'",
            "Fix 2: Change 'Job-specific' -> 'Task-specific'"
        ]
    },
    40: {
        "roast": "১) 'Movie'?! 🍿 গোয়ালঘরের CCTV ফুটেজ দেখতে দেখতে কি সিনেমা হলের ফিল পাচ্ছোস নাকি?! ২) ট্রেন-টেস্ট স্প্লিট কোনো 'characteristics মেজার' করে না—কাছাকাছি ফ্রেমের কারণে data leakage হয় যা রেজাল্ট নষ্ট করে!",
        "intact": ["CCTV surveillance footage", "Data leakage", "Train-test split", "Temporal correlation"],
        "fixes": [
            "Fix 1: Change 'Movie' -> 'Surveillance footage / video sequences'",
            "Fix 2: Change 'Measure characteristics' -> 'Evaluate generalization / prevent data leakage'"
        ]
    },
    43: {
        "roast": "কপি-পেস্ট ডিজাস্টার! 🤦‍♂️ তোরা রো ৪০-এর মুভির প্যারাগ্রাফটাই হুবহু কপি কইরা রো ৪৩-এ পেস্ট করছোস! রো ৪৩ ছিল মডেলের কম্পোনেন্ট আর single-task baseline নিয়া!",
        "intact": ["Model components", "Single-task reference baselines", "Perception cues"],
        "fixes": [
            "Fix 1: Change Copied Row 40 text -> Original Row 43 content on single-task reference baselines"
        ]
    },
    47: {
        "roast": "'Recognition of activity is activity'?! 🤖 ভাই এটা কেমন অর্থহীন বাক্য! লিখবা 'Behavior recognition focuses on temporal activity'.",
        "intact": ["Behavior recognition", "Temporal activity", "Posture and movement"],
        "fixes": [
            "Fix 1: Change 'Recognition of activity is activity' -> 'Behavior recognition targets temporal action patterns and posture changes'"
        ]
    },
    53: {
        "roast": "১) 'The input is to look into...'?! একটা থিসিসের 'contribution' থাকে, 'input' না! ২) 'Quick segmentation' না, ওটা ছিল 'promptable segmentation' (SAM model)!",
        "intact": ["Contribution", "Promptable segmentation", "SAM", "Pretrained perception models"],
        "fixes": [
            "Fix 1: Change 'The input is' -> 'The thesis contribution is'",
            "Fix 2: Change 'Quick segmentation' -> 'Promptable segmentation'"
        ]
    },
    57: {
        "roast": "'Teach and send visual representations'?! 📬 আমরা কি ডাকযোগে চিঠি পাঠাচ্ছি নাকি?! এমএল-এ বলে 'learn and share representations across tasks'!",
        "intact": ["Learn and share", "Visual representations", "Downstream tasks"],
        "fixes": [
            "Fix 1: Change 'Teach and send representations' -> 'Learn and share representations across tasks'"
        ]
    },
    63: {
        "roast": "'The volume of information to be conveyed' শুনতে ব্রডব্যান্ড ইন্টারনেটের প্যাকেজ মনে হয় 📶! আসল কথা হইলো টাস্কগুলোর মধ্যে কতটুকু প্যারামিটার 'share' করা উচিত!",
        "intact": ["Parameter sharing", "Capacity", "Task-private pathways", "Shared representations"],
        "fixes": [
            "Fix 1: Change 'Volume of information to be conveyed' -> 'Amount of shared model capacity / parameters'"
        ]
    },
    73: {
        "roast": "ভুল তথ্য! ❌ তোরা লিখছোস অরিজিনাল আইডি পাওয়া যায় না! অথচ SideViewCows2026-এ ১১০টা গরুর অরিজিনাল বায়োলজিক্যাল আইডি আছে! শুধু ScienceDB-তে নাই!",
        "intact": ["Biological identities", "SideViewCows2026", "Group-disjoint", "ScienceDB"],
        "fixes": [
            "Fix 1: Change 'No genuine identities available' -> 'SideViewCows2026 possesses 110 biological cow IDs, while ScienceDB lacks cow identities'"
        ]
    },
    79: {
        "roast": "'Geolocation'?! 🛰️📍 তোরা কি গরুর পেছনে NASA-র military satellite GPS ট্র্যাকার বসায়া ড্রোন হামলা চালাবি নাকি?! ক্যামেরার ফ্রেমে গরুর চারপাশের bounding box-রে বলে Localization, Google Maps-এর live location না!",
        "intact": ["Bounding-box localization", "Foreground masks", "Visual cues"],
        "fixes": [
            "Fix 1: Change 'Geolocation' -> 'Localization / Bounding boxes'"
        ]
    },
    82: {
        "roast": "'Without harming individual actions'?! গরুর কোনো পার্সোনাল কাজে ক্ষতি হচ্ছে না রে ভাই! ওটা ছিল 'without hurting individual tasks' (negative transfer আটকানো)!",
        "intact": ["Negative transfer", "Individual tasks", "Multi-task optimization"],
        "fixes": [
            "Fix 1: Change 'Individual actions' -> 'Individual tasks / task performance'"
        ]
    },
    92: {
        "roast": "প্যারাগ্রাফের মাঝখানে একা একা '1.' ঝুলতেছে কেন? 🔢 নাম্বার ২ কি হারিয়ে গেছে?!",
        "intact": ["Research objectives", "Multi-task framework"],
        "fixes": [
            "Fix 1: Change '1.' floating marker -> Remove rogue number and merge into smooth narrative"
        ]
    },
    95: {
        "roast": "অর্থ বদলে গেছে: তোদের লেখায় মনে হচ্ছে শুধু behavior মডেলে মাস্ক ব্যবহার করা হয়! আসলে তিনটা টাস্কেই মাস্ক ব্যবহার করা হয়, শুধু behavior-এ temporal যোগ হয়!",
        "intact": ["Segmentation masks", "Crops", "Temporal aggregation", "All three tasks"],
        "fixes": [
            "Fix 1: Change 'Only behavior uses masks' -> 'All three tasks utilize crops and masks, with temporal aggregation added for behavior'"
        ]
    },
    98: {
        "roast": "'Retrieval assignment'?! 📝 Re-ID হলো কম্পিউটার ভিশনের information retrieval task, স্কুলের হোমওয়ার্ক অ্যাসাইনমেন্ট না!",
        "intact": ["Re-ID", "Information retrieval task", "Gallery", "Rank metrics"],
        "fixes": [
            "Fix 1: Change 'Retrieval assignment' -> 'Information retrieval task'"
        ]
    },
    101: {
        "roast": "'Collected in a collaborative manner'?! 🤝 এটা ডিপ লার্নিংয়ের multi-task parameter sharing, খামারিদের দলবদ্ধ হয়ে কাজ করা না!",
        "intact": ["Multi-task learning", "Shared representation", "Parameter sharing"],
        "fixes": [
            "Fix 1: Change 'Collected collaboratively' -> 'Shared model components / multi-task representation sharing'"
        ]
    },
    105: {
        "roast": "সবচেয়ে বড় disaster তো এইখানে! তোরা thesis-এর main topic BCS (Body Condition Scoring)-ই পুরা হাওয়া কইরা দিছোস! 🚨 Individual Cow Identification আর Re-ID যে একই জিনিস, ওইটা দুইবার লেইখা আসল জিনিস BCS-ই নাই!",
        "intact": ["Body Condition Scoring (BCS)", "Behavior Recognition", "Individual Cow Identification / Re-ID"],
        "fixes": [
            "Fix 1: Change Duplicate 'Identification and Re-ID' -> Add missing 'Body Condition Scoring (BCS)' as core task"
        ]
    },
    111: {
        "roast": "১) 'Barrier' না, কম্পিউটার ভিশনে ওটাকে বলে 'occlusion' (দৃষ্টিসীমা আটকে যাওয়া)! ২) 'Validated in the real world' কোনো প্রমাণ ছাড়া ওভারক্লেম—আমরা এক্সপেরিমেন্ট করছি, খামারে ডিপ্লয় করি নাই!",
        "intact": ["Occlusion", "Viewpoint", "Benchmark experiments"],
        "fixes": [
            "Fix 1: Change 'Barrier' -> 'Occlusion'",
            "Fix 2: Change 'Validated in real world' -> 'Evaluated across empirical benchmark datasets'"
        ]
    },
    117: {
        "roast": "'Combining many activities in real life'?! 🏃‍♂️ এটা মাল্টি-টাস্ক ডিপ লার্নিং মডেল ইন্টিগ্রেশন, বাস্তব জীবনের মাল্টিটাস্কিং না!",
        "intact": ["Unified multi-task deep learning framework", "Integrated evaluation"],
        "fixes": [
            "Fix 1: Change 'Combining activities in real life' -> 'Unified multi-task deep learning framework'"
        ]
    }
}

CH2_FLAGGED = {
    3: {
        "roast": "উন্নতি হইছে! 👏 'Morphology' ফেরত আনছো এবং শেষের বাক্যটাও ঠিক করছো!",
        "intact": ["This thesis", "Morphology", "Posture and movement", "Multi-task learning"],
        "fixes": [
            "Fix 1: Change 'different task' -> 'distinct monitoring tasks (plural)'",
            "Fix 2: Change 'this paper' -> 'this thesis'"
        ]
    },
    13: {
        "roast": "হাতেনাতে ধরা খাইছোস! 🚨 এটা কোনো প্যারাফ্রেজ না, পুরা ১০০% হুবহু কপি-পেস্ট (Ctrl+C ➔ Ctrl+V)! একটা কমা বা অক্ষরও পরিবর্তন করোস নাই! Turnitin দেখলে লাল বাতি জ্বালায়া রাখবে! 🎄💀",
        "intact": ["ResNet", "EfficientNet", "Residual connections", "Transfer learning"],
        "fixes": [
            "Fix 1: Change 100% verbatim copied text -> Paraphrase sentences in your own words while retaining model names"
        ]
    },
    143: {
        "roast": "১) 'Pose estimation is the task of an animal to express their pose'?! 💀 গরু কি র‍্যাম্পে মডেলিং করতেছে?! Pose estimation গরুর কাজ না, Computer Vision মডেলের কাজ (ছবি থেকে keypoints লোকেট করা)!\n২) 'parameter values' না, 'model input' শব্দটা intact রাখতে হবে! 🚨 Pose হলো interpretable visual input, নিউরাল নেটওয়ার্কের parameters না!",
        "intact": ["Pose estimation", "DeepLabCut", "Model input", "Anatomical landmarks"],
        "fixes": [
            "Fix 1: Change 'Task of an animal to express pose' -> 'Computer vision task of estimating animal posture'",
            "Fix 2: Change 'Parameter values of model' -> 'Model input / input features'"
        ]
    }
}

CH3_FLAGGED = {
    3: {
        "roast": "হাতেনাতে ধরা খাইছোস! 🚨 সেকেন্ড হাফের পুরাটা ('Because of this, the requirements include...') অরিজিনাল লেখা থেকে ১০০% হুবহু কপি-পেস্ট করছোস! Turnitin দেখলে আগুন ধইরা যাবে 🎄!",
        "intact": ["Requirements", "Constraints", "Evaluation protocols"],
        "fixes": [
            "Fix 1: Change 100% verbatim second half -> Paraphrase the requirement sentences in your own words"
        ]
    },
    9: {
        "roast": "'Not leaking' শুনতে প্লাম্বারের পাইপ মেরামতের মতো লাগে 🚰! ওটা Machine Learning-এর data leakage prevention! আর 'Keep big checkpoints' কোনো একাডেমিক টোন না!",
        "intact": ["Data leakage prevention", "Split integrity", "Model checkpoints"],
        "fixes": [
            "Fix 1: Change 'Not leaking' -> 'Data leakage prevention'",
            "Fix 2: Change 'Keep big checkpoints' -> 'Preserve full model checkpoints for reproducible audit'"
        ]
    },
    12: {
        "roast": "'Perceptual mistakes'? মডেল গরু ডিটেক্ট করতে ফেইল করছে, কোনো দার্শনিক ভুলভ্রান্তি করে নাই 👻! লেখো 'perception failures'!",
        "intact": ["Perception failures", "Detection misses", "Segmentation quality"],
        "fixes": [
            "Fix 1: Change 'Perceptual mistakes' -> 'Perception failures / detector false negatives'"
        ]
    },
    22: {
        "roast": "১) 'Species'?! গরু নিজেই একটা একক প্রজাতি (*Bos taurus*), তোরা লিখতে চাইছোস 'breed' (জাত) 🐄! ২) 'The agri-environment is giving access' মানে কি গোয়ালঘর ওয়াইফাই পাসওয়ার্ড দিচ্ছে 📶?!",
        "intact": ["Breed", "Bos taurus", "Agricultural environment challenges"],
        "fixes": [
            "Fix 1: Change 'Species' -> 'Breed (e.g. Holstein, Angus)'",
            "Fix 2: Change 'Agri-environment giving access' -> 'Operational agricultural environment conditions'"
        ]
    },
    29: {
        "roast": "মডেলের আর্কিটেকচার উল্টায় দিছোস! 🔄 তোরা লিখছোস বড় ডিটেক্টর মডেল ছোট মডেলের ওপর নির্ভর করে! আসলে উল্টা—ছোট ডাউনস্ট্রিম মডেল ভারী আপস্ট্রিম মডেলের ওপর নির্ভর করে!",
        "intact": ["Upstream perception models", "Downstream task heads", "Sequential dependency"],
        "fixes": [
            "Fix 1: Change 'Heavy detector depends on minor models' -> 'Lightweight downstream task heads depend on heavy upstream perception backbones'"
        ]
    },
    39: {
        "roast": "মারাত্মক আবিষ্কার! 'Invisible cow assessment'?! 👻🐮 আমরা কি ফার্মের ভূতুড়ে গরুর বিচার করতেছি নাকি?! আর গরুর Behavior-রে বানাইছোস 'Evaluation of conduct'! মামা, ওটা Unseen-cow evaluation (যে গরু মডেল training-এ দেখে নাই), অদৃশ্য গরু না! 😭",
        "intact": ["Unseen-cow evaluation", "Cattle behavior", "Ground-truth masks", "Oracle evaluation"],
        "fixes": [
            "Fix 1: Change 'Invisible cow assessment' -> 'Unseen-cow evaluation'",
            "Fix 2: Change 'Evaluation of conduct' -> 'Behavior evaluation'"
        ]
    },
    45: {
        "roast": "'Measures the number of errors removed by perceptual errors'—এটা পুরা খিচুড়ি মার্কা কথা! 🥗 আসল কথা হলো কত শতাংশ ইমেজ সফলভাবে সেগমেন্ট হইছে তার কভারেজ রিপোর্ট করা!",
        "intact": ["Segmentation coverage rate", "Perception filter", "Drop rate"],
        "fixes": [
            "Fix 1: Change 'Errors removed by perceptual errors' -> 'Fraction of samples successfully processed by the perception pipeline (coverage rate)'"
        ]
    },
    68: {
        "roast": "খলিলুর রহমান স্যাররে তোরা 'Overseer' বানায় দিলি?! 💀 স্যার কি ১৯ শতকের ব্রিটিশ আমলের নীলচাষীদের ওপর চাবুক মারা ম্যানেজার নাকি?! শব্দটা হইলো Supervisors / Advisors! খলিল স্যার এই ড্রাফট দেখলে thesis defense-এর আগেই আমাদের কাঁচা চিবায়া খাবে!",
        "intact": ["Supervisors", "Advisors", "Authorship confirmation"],
        "fixes": [
            "Fix 1: Change 'Overseers' -> 'Supervisors / Advisors'"
        ]
    },
    72: {
        "roast": "১) 'Appropriate comments about train-test splits'?! আসল রিস্ক হলো DATA LEAKAGE (স্প্লিটের ভেতর ডেটা লিক হওয়া), কোনো ভালো কমেন্ট না! ২) টেবিলের নাম 'Tabletab:dangers' বানায় দিও না!",
        "intact": ["Data leakage", "Train-test split", "Table reference"],
        "fixes": [
            "Fix 1: Change 'Appropriate comments' -> 'Rigorous protocol for data leakage prevention'",
            "Fix 2: Change 'Tabletab:dangers' -> Clean citation/reference"
        ]
    },
    75: {
        "roast": "১) 'Ambulation'?! 🚶‍♂️ আমাদের ৫টা ক্লাসের অফিসিয়াল নাম 'Walking'—ল্যাটিন বানিয়ে দিও না! ২) 'Behavioral films'? ওগুলো ১০ সেকেন্ডের সিসিটিভি ক্লিপ, হলিউডের মুভি না!",
        "intact": ["Walking", "CCTV video clips", "Behavior classes"],
        "fixes": [
            "Fix 1: Change 'Ambulation' -> 'Walking'",
            "Fix 2: Change 'Behavioral films' -> 'Video clips / CCTV footage'"
        ]
    }
}

FLAGGED_BY_TAB = {
    "Chapter 1: Introduction": CH1_FLAGGED,
    "Chapter 2: Literature Review": CH2_FLAGGED,
    "Chapter 3: Requirements & Constraints": CH3_FLAGGED
}

DOMAIN_VOCAB = [
    # Core Tasks & Concepts
    "Body Condition Scoring (BCS)", "Body Condition Scoring", "BCS",
    "Behavior Recognition", "Behavior", "Re-Identification", "Re-ID",
    "Individual Cow Identification", "Multi-Task Learning (MTL)", "Multi-task learning", "MTL",
    "Single-task reference baselines", "Single-task baseline", "Single-task",
    "Task-specific", "Task-private", "Shared backbone", "Hard sharing", "Modular sharing",
    "Negative transfer", "Gradient interference", "Shortcut learning", "Spurious correlations",
    "Visual representation", "Visual representations", "Representation learning", "Visual features",
    "Deep neural networks", "Convolutional network", "Convolutional neural networks", "CNN",
    "Prediction head", "Classification head", "Residual connections", "Skip connections",
    "Hand-designed measurements", "Handcrafted features",
    
    # Computer Vision & Anatomy
    "Localization", "Bounding box", "Segmentation", "Foreground masks", "Soft mask",
    "Pose estimation", "Anatomical landmarks", "Keypoints", "Morphology", "Body shape",
    "Posture and movement", "Temporal aggregation", "Optical flow", "Surveillance footage",
    "CCTV video", "Viewpoint", "Side-view", "Top-down", "Rear-view", "Occlusion",
    "Camera angle", "Camera viewpoint", "Background scene", "Pen background",
    
    # Models & Architectures
    "ResNet", "ResNet-18", "ResNet-50", "EfficientNet", "ConvNeXt", "Vision Transformer", "ViT",
    "RT-DETR-L", "RT-DETR", "SAM 2.1", "Segment Anything (SAM)", "SAM",
    "DeepLabCut", "SuperAnimal", "TCN", "ST-GCN", "AdamW", "Cosine annealing",
    
    # Datasets
    "ScienceDB", "SideViewCows2026", "CVB", "Kaggle Beef", "CBVD-5", "MmCows", "BECA",
    "CattleEyeView", "OpenCows2020", "ImageNet", "COCO",
    
    # Experimental Rigor & Metrics
    "Unseen-cow evaluation", "Group-disjoint", "Cow-disjoint", "Data leakage",
    "Burst-group-disjoint", "Held-out test set", "Generalization",
    "Barn / Pen", "Barn", "Pen", "Drinking", "Feeding", "Lying", "Standing", "Walking",
    "Real MAE", "Acc@1", "Balanced Accuracy", "Macro-F1", "Rank-1", "mAP", "IoU",
    "Supervisors", "Advisors", "This thesis", "CSE400"
]

STOP_WORDS = {
    "older", "deep", "this", "the", "in", "such", "because", "these", "those",
    "however", "different", "several", "each", "both", "many", "most", "some",
    "first", "then", "second", "also", "into", "from", "with", "without", "which",
    "while", "when", "where", "what", "their", "there", "other", "another"
}

def extract_intact_terms(text):
    matched = []
    text_lower = text.lower()
    
    sorted_vocab = sorted(DOMAIN_VOCAB, key=len, reverse=True)
    for term in sorted_vocab:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            if not any(term.lower() in m.lower() for m in matched):
                matched.append(term)
    
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    for acr in acronyms:
        if acr not in ["AND", "THE", "FOR", "NOT", "BUT", "ALL", "ARE", "WAS"] and acr not in matched:
            matched.append(acr)
            
    ordered = []
    for m in matched:
        pos = text_lower.find(m.lower())
        ordered.append((pos, m))
    ordered.sort(key=lambda x: x[0])
    
    seen = set()
    dedup = []
    for t in ordered:
        term = t[1]
        if term.lower() not in seen and term.lower() not in STOP_WORDS:
            seen.add(term.lower())
            dedup.append(term)
    
    # Contextual fallback
    if len(dedup) < 2:
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for w in words:
            if w.lower() not in STOP_WORDS and w.lower() not in seen and len(w) > 3:
                seen.add(w.lower())
                dedup.append(w)
                
    return dedup[:5]

def format_cell(tab_name, row_num, col_b, col_c):
    flagged_dict = FLAGGED_BY_TAB.get(tab_name, {})
    
    # Check if flagged
    if row_num in flagged_dict:
        item = flagged_dict[row_num]
        roast = item["roast"]
        intact_str = ", ".join(item["intact"])
        fixes_str = "\n".join([f"• {f}" for f in item["fixes"]])
        
        return (
            f"{roast}\n\n"
            f"📌 Words/phrases to keep intact:\n• {intact_str}\n\n"
            f"🛠️ Required Fixes:\n{fixes_str}"
        )
    
    terms = extract_intact_terms(col_b)
    intact_str = ", ".join(terms) if terms else "Core task terminology and model names"
    
    if col_c and col_c.strip() and col_c.strip() not in ["Paraphrased:", "LEAVE BLANK (Do Not Paraphrase):"]:
        return (
            f"✅ অর্থ ঠিক রাখা হইছে।\n\n"
            f"📌 Words/phrases to keep intact:\n• {intact_str}"
        )
    else:
        return (
            f"📌 Words/phrases to keep intact:\n• {intact_str}"
        )

def process_all_chapters():
    service = get_service()
    
    ranges = [f"'{tab}'!A1:C265" for tab in SHEET_IDS.keys()]
    res = robust_execute(lambda: service.values().batchGet(spreadsheetId=SPREADSHEET_ID, ranges=ranges).execute())
    
    total_paras = 0
    total_cells_updated = 0
    
    for tab_name, val_range in zip(SHEET_IDS.keys(), res.get("valueRanges", [])):
        sheet_id = SHEET_IDS[tab_name]
        rows = val_range.get("values", [])
        
        updates = []
        format_requests = []
        
        for idx, r in enumerate(rows, start=1):
            row_num = idx
            col_b = r[1] if len(r) > 1 else ""
            col_c = r[2] if len(r) > 2 else ""
            
            if len(col_b.strip()) > 35 and not col_b.strip().startswith("Original (Do Paraphrase"):
                cell_content = format_cell(tab_name, row_num, col_b, col_c)
                updates.append({
                    "range": f"'{tab_name}'!D{row_num}",
                    "values": [[cell_content]]
                })
                total_paras += 1
                
                # Highlight bad paraphrases in Col C with light yellow
                if row_num in FLAGGED_BY_TAB.get(tab_name, {}):
                    format_requests.append({
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": idx - 1,
                                "endRowIndex": idx,
                                "startColumnIndex": 2,
                                "endColumnIndex": 3
                            },
                            "cell": {
                                "userEnteredFormat": {
                                    "backgroundColor": {"red": 1.0, "green": 0.98, "blue": 0.8}
                                }
                            },
                            "fields": "userEnteredFormat.backgroundColor"
                        }
                    })

        if updates:
            print(f"[*] Updating {len(updates)} paragraph cells in '{tab_name}'...")
            res_val = robust_execute(lambda: service.values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "USER_ENTERED", "data": updates}
            ).execute())
            total_cells_updated += res_val.get("totalUpdatedCells", len(updates))
            
        if format_requests:
            robust_execute(lambda: service.batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"requests": format_requests}
            ).execute())
            
    print(f"\n🏆 COMPLETED! Successfully populated all {total_paras} paragraphs across all 3 chapters!")
    print(f"Total cells updated: {total_cells_updated}")

if __name__ == "__main__":
    process_all_chapters()
