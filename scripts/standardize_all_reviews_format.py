"""
Standardize all reviews across all chapters into the exact requested format:

[Roast]

Words/phrases to keep intact: x, y, z
Fix: Change a -> b, change c -> d
"""

import sys
import socket
import time
socket.setdefaulttimeout(15.0)

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, robust_execute

ch1_reviews = {
    3: (
        "অসাধারণ scientific logic! 'Reduced food intake causes a cow to eat less' (কম খাওয়া একটা গরুকে কম খাওয়ায়!) 💀 "
        "আইনস্টাইন তো কবরে বইসা কাঁদতেছে! পুরা circular logic-এর কারখানা!\n\n"
        "Words/phrases to keep intact: Body condition, Reduced food intake, Housing and management\n"
        "Fix: Change 'causes a cow to eat less' -> 'leads to gradual condition loss and altered resting' (আলাদা লক্ষণ হিসেবে লেখো, কারণ না)"
    ),
    9: (
        "'Trivial incidents'?! ☕ একটা গরু পানি খাওয়া, খাবার খাওয়া বা অসুস্থ হওয়া কোনো 'trivial incident' (তুচ্ছ ঘটনা) না! "
        "আর 'short observations' মানে সময়ের ব্যাপ্তি ছোট, কোনো লাজুক 'modest' স্বভাব না!\n\n"
        "Words/phrases to keep intact: Cattle monitoring, Routine welfare events, Short-duration observations\n"
        "Fix: Change 'Trivial incidents' -> 'Routine welfare events', Change 'Modest traits' -> 'Brief observation periods'"
    ),
    12: (
        "প্রথম লাইনটা সিনেমার ডায়লগের মতো কোটেশন মার্কসে (\" \") কেন দিছোস? 🎬 ডিরেক্ট কোটেশন দিলে Turnitin সাথে সাথে প্ল্যাজিয়ারিজম ধরবে! "
        "আর ৫ শব্দের মধ্যে দুইবার applications লিখছোস!\n\n"
        "Words/phrases to keep intact: Multi-task learning, Livestock applications\n"
        "Fix: Change Direct quotation (\"...\") -> Paraphrased own words, Change 'applications... in other livestock applications' -> Remove word repetition"
    ),
    15: (
        "'Learned traits' আর 'various duties'?! 😭 অ্যানিম্যাল সায়েন্সে 'traits' মানে জেনেটিক্স, নিউরাল নেটওয়ার্কের ফিচার না! "
        "আর মডেল Machine Learning-এর 'tasks' করে, ৯-টা ৫-টা অফিসের 'duties' না!\n\n"
        "Words/phrases to keep intact: Visual features, Representations, Tasks, Transfer learning\n"
        "Fix: Change 'Learned traits' -> 'Visual features / representations', Change 'Various duties' -> 'Downstream tasks'"
    ),
    18: (
        "ভাইরে ভাই! তোরা ডিকশনারি থেইকা সিনোনিম মাইরা Machine Learning-এর Tasks-রে Occupations (পেশা/চাকরি) বানায় দিলি?! 💀 "
        "গরু কি এখন LinkedIn-এ প্রোফাইল খুইলা ট্যাক্স ফাইল জমা দিবে?! MTL model-এর tasks আর মানুষের corporate jobs এক জিনিস না!\n\n"
        "Words/phrases to keep intact: Multi-task learning (MTL), Tasks, Shared features, Selective sharing\n"
        "Fix: Change 'Occupations' -> 'Tasks', Change 'Dilemma' -> 'Task relationship / trade-off'"
    ),
    24: (
        "'Two occupations responsible for the care of the same cow'?! 🧑‍🌾👨‍🌾 এখানে দুইটা কম্পিউটার ভিশন মডেল (BCS এবং Re-ID) একই গরুর ছবি প্রসেস করতেছে, কোনো ডাক্তার আর খামারি মিটিং করতেছে না!\n\n"
        "Words/phrases to keep intact: Tasks, BCS, Re-ID, Shared visual input\n"
        "Fix: Change 'Two occupations caring for cow' -> 'Two distinct computer vision tasks processing the same animal'"
    ),
    27: (
        "১) BCS মানে বডির চর্বি/এনার্জি রিজার্ভ, কোনো 'sickness' (অসুখ) না 🤒! বেশি বিসিএস হওয়া মানে গরু সুস্থ-সবল, অসুস্থ না! "
        "২) গোয়ালঘরের খাবার জায়গা কোনো 'geography' 🗺️ না, এটা ব্যাকগ্রাউন্ড সিন!\n\n"
        "Words/phrases to keep intact: BCS, Body condition, Barn / Pen background, Visual cues\n"
        "Fix: Change 'Sickness' -> 'Energy reserves / body condition', Change 'Geography' -> 'Pen / barn background context'"
    ),
    30: (
        "গরু আর তার 'House'?! 🏡 গরু কি এখন ব্যাংক থেইকা home loan নিয়া গুলশানে ডুপ্লেক্স বাড়ি বানাইয়া থাকে নাকি রে ভাই?! "
        "ওটা গোয়ালঘর আর খামার (Barn / Pen)! আর 'Shortcut learning'-রে 'quick learning' বানায় দিও না!\n\n"
        "Words/phrases to keep intact: Shortcut learning, Barn / Pen, Spurious correlations\n"
        "Fix: Change 'House' -> 'Barn / pen environment', Change 'Quick learning' -> 'Shortcut learning'"
    ),
    33: (
        "'The target cow is localized through localization'—অসাধারণ বৈজ্ঞানিক থিওরি! 🧠💀 'গরু লোকালাইজড হয় কারণ সে লোকালাইজড!' "
        "আর আবার 'Job-specific'?! গরু কি বিসিএস ক্যাডার নাকি যে তার Job থাকবে?! শব্দটা হইলো 'Task-specific'!\n\n"
        "Words/phrases to keep intact: Cattle-centered representation learning, Localization, Segmentation, Task-specific\n"
        "Fix: Change 'Target cow is localized through localization' -> 'Localization isolates the target animal', Change 'Job-specific' -> 'Task-specific'"
    ),
    40: (
        "১) 'Movie'?! 🍿 গোয়ালঘরের CCTV ফুটেজ দেখতে দেখতে কি সিনেমা হলের ফিল পাচ্ছোস নাকি?! "
        "২) ট্রেন-টেস্ট স্প্লিট কোনো 'characteristics মেজার' করে না—কাছাকাছি ফ্রেমের কারণে data leakage হয় যা রেজাল্ট নষ্ট করে!\n\n"
        "Words/phrases to keep intact: CCTV surveillance footage, Data leakage, Train-test split, Temporal correlation\n"
        "Fix: Change 'Movie' -> 'Surveillance footage / video sequences', Change 'Measure characteristics' -> 'Evaluate generalization / prevent data leakage'"
    ),
    43: (
        "কপি-পেস্ট ডিজাস্টার! 🤦‍♂️ তোরা রো ৪০-এর মুভির প্যারাগ্রাফটাই হুবহু কপি কইরা রো ৪৩-এ পেস্ট করছোস! "
        "রো ৪৩ ছিল মডেলের কম্পোনেন্ট আর single-task baseline নিয়া!\n\n"
        "Words/phrases to keep intact: Model components, Single-task reference baselines, Perception cues\n"
        "Fix: Change Copied Row 40 text -> Original Row 43 content on single-task reference baselines"
    ),
    47: (
        "'Recognition of activity is activity'?! 🤖 ভাই এটা কেমন অর্থহীন বাক্য! লিখবা 'Behavior recognition focuses on temporal activity'.\n\n"
        "Words/phrases to keep intact: Behavior recognition, Temporal activity, Posture and movement\n"
        "Fix: Change 'Recognition of activity is activity' -> 'Behavior recognition targets temporal action patterns and posture changes'"
    ),
    53: (
        "১) 'The input is to look into...'?! একটা থিসিসের 'contribution' থাকে, 'input' না! "
        "২) 'Quick segmentation' না, ওটা ছিল 'promptable segmentation' (SAM model)!\n\n"
        "Words/phrases to keep intact: Contribution, Promptable segmentation, SAM, Pretrained perception models\n"
        "Fix: Change 'The input is' -> 'The thesis contribution is', Change 'Quick segmentation' -> 'Promptable segmentation'"
    ),
    57: (
        "'Teach and send visual representations'?! 📬 আমরা কি ডাকযোগে চিঠি পাঠাচ্ছি নাকি?! "
        "এমএল-এ বলে 'learn and share representations across tasks'!\n\n"
        "Words/phrases to keep intact: Learn and share, Visual representations, Downstream tasks\n"
        "Fix: Change 'Teach and send representations' -> 'Learn and share representations across tasks'"
    ),
    63: (
        "'The volume of information to be conveyed' শুনতে ব্রডব্যান্ড ইন্টারনেটের প্যাকেজ মনে হয় 📶! "
        "আসল কথা হইলো টাস্কগুলোর মধ্যে কতটুকু প্যারামিটার 'share' করা উচিত!\n\n"
        "Words/phrases to keep intact: Parameter sharing, Capacity, Task-private pathways, Shared representations\n"
        "Fix: Change 'Volume of information to be conveyed' -> 'Amount of shared model capacity / parameters'"
    ),
    73: (
        "ভুল তথ্য! ❌ তোরা লিখছোস অরিজিনাল আইডি পাওয়া যায় না! অথচ SideViewCows2026-এ ১১০টা গরুর অরিজিনাল বায়োলজিক্যাল আইডি আছে! শুধু ScienceDB-তে নাই!\n\n"
        "Words/phrases to keep intact: Biological identities, SideViewCows2026, Group-disjoint, ScienceDB\n"
        "Fix: Change 'No genuine identities available' -> 'SideViewCows2026 possesses 110 biological cow IDs, while ScienceDB lacks cow identities'"
    ),
    79: (
        "'Geolocation'?! 🛰️📍 তোরা কি গরুর পেছনে NASA-র military satellite GPS ট্র্যাকার বসায়া ড্রোন হামলা চালাবি নাকি?! "
        "ক্যামেরার ফ্রেমে গরুর চারপাশের bounding box-রে বলে Localization, Google Maps-এর live location না!\n\n"
        "Words/phrases to keep intact: Bounding-box localization, Foreground masks, Visual cues\n"
        "Fix: Change 'Geolocation' -> 'Localization / Bounding boxes'"
    ),
    82: (
        "'Without harming individual actions'?! গরুর কোনো পার্সোনাল কাজে ক্ষতি হচ্ছে না রে ভাই! "
        "ওটা ছিল 'without hurting individual tasks' (negative transfer আটকানো)!\n\n"
        "Words/phrases to keep intact: Negative transfer, Individual tasks, Multi-task optimization\n"
        "Fix: Change 'Individual actions' -> 'Individual tasks / task performance'"
    ),
    92: (
        "প্যারাগ্রাফের মাঝখানে একা একা '1.' ঝুলতেছে কেন? 🔢 নাম্বার ২ কি হারিয়ে গেছে?!\n\n"
        "Words/phrases to keep intact: Research objectives, Multi-task framework\n"
        "Fix: Change '1.' floating marker -> Remove rogue number and merge into smooth narrative"
    ),
    95: (
        "অর্থ বদলে গেছে: তোদের লেখায় মনে হচ্ছে শুধু behavior মডেলে মাস্ক ব্যবহার করা হয়! "
        "আসলে তিনটা টাস্কেই মাস্ক ব্যবহার করা হয়, শুধু behavior-এ temporal যোগ হয়!\n\n"
        "Words/phrases to keep intact: Segmentation masks, Crops, Temporal aggregation, All three tasks\n"
        "Fix: Change 'Only behavior uses masks' -> 'All three tasks utilize crops and masks, with temporal aggregation added for behavior'"
    ),
    98: (
        "'Retrieval assignment'?! 📝 Re-ID হলো কম্পিউটার ভিশনের information retrieval task, স্কুলের হোমওয়ার্ক অ্যাসাইনমেন্ট না!\n\n"
        "Words/phrases to keep intact: Re-ID, Information retrieval task, Gallery, Rank metrics\n"
        "Fix: Change 'Retrieval assignment' -> 'Information retrieval task'"
    ),
    101: (
        "'Collected in a collaborative manner'?! 🤝 এটা ডিপ লার্নিংয়ের multi-task parameter sharing, খামারিদের দলবদ্ধ হয়ে কাজ করা না!\n\n"
        "Words/phrases to keep intact: Multi-task learning, Shared representation, Parameter sharing\n"
        "Fix: Change 'Collected collaboratively' -> 'Shared model components / multi-task representation sharing'"
    ),
    105: (
        "সবচেয়ে বড় disaster তো এইখানে! তোরা thesis-এর main topic BCS (Body Condition Scoring)-ই পুরা হাওয়া কইরা দিছোস! 🚨 "
        "Individual Cow Identification আর Re-ID যে একই জিনিস, ওইটা দুইবার লেইখা আসল জিনিস BCS-ই নাই!\n\n"
        "Words/phrases to keep intact: Body Condition Scoring (BCS), Behavior Recognition, Individual Cow Identification / Re-ID\n"
        "Fix: Change Duplicate 'Identification and Re-ID' -> Add missing 'Body Condition Scoring (BCS)' as core task"
    ),
    111: (
        "১) 'Barrier' না, কম্পিউটার ভিশনে ওটাকে বলে 'occlusion' (দৃষ্টিসীমা আটকে যাওয়া)! "
        "২) 'Validated in the real world' কোনো প্রমাণ ছাড়া ওভারক্লেম—আমরা এক্সপেরিমেন্ট করছি, খামারে ডিপ্লয় করি নাই!\n\n"
        "Words/phrases to keep intact: Occlusion, Viewpoint, Benchmark experiments\n"
        "Fix: Change 'Barrier' -> 'Occlusion', Change 'Validated in real world' -> 'Evaluated across empirical benchmark datasets'"
    ),
    117: (
        "'Combining many activities in real life'?! 🏃‍♂️ এটা মাল্টি-টাস্ক ডিপ লার্নিং মডেল ইন্টিগ্রেশন, বাস্তব জীবনের মাল্টিটাস্কিং না!\n\n"
        "Words/phrases to keep intact: Unified multi-task deep learning framework, Integrated evaluation\n"
        "Fix: Change 'Combining activities in real life' -> 'Unified multi-task deep learning framework'"
    )
}

ch2_reviews = {
    3: (
        "উন্নতি হইছে! 👏 'Morphology' ফেরত আনছো এবং শেষের ভাঙা বাক্যটাও ঠিক করছো!\n\n"
        "Words/phrases to keep intact: This thesis, Morphology, Posture and movement, Multi-task learning\n"
        "Fix: Change 'different task' -> 'distinct monitoring tasks (plural)', Change 'this paper' -> 'this thesis'"
    ),
    13: (
        "হাতেনাতে ধরা খাইছোস! 🚨 এটা কোনো প্যারাফ্রেজ না, পুরা ১০০% হুবহু কপি-পেস্ট (Ctrl+C ➔ Ctrl+V)! "
        "একটা কমা বা অক্ষরও পরিবর্তন করোস নাই! Turnitin দেখলে লাল বাতি জ্বালায়া রাখবে! 🎄💀\n\n"
        "Words/phrases to keep intact: ResNet, EfficientNet, Residual connections, Transfer learning\n"
        "Fix: Change 100% verbatim copied text -> Paraphrase sentences in your own words while retaining model names"
    ),
    143: (
        "১) 'Pose estimation is the task of an animal to express their pose'?! 💀 গরু কি র‍্যাম্পে মডেলিং করতেছে?! "
        "Pose estimation গরুর কাজ না, Computer Vision মডেলের কাজ (ছবি থেকে keypoints লোকেট করা)!\n"
        "২) 'parameter values' না, 'model input' শব্দটা intact রাখতে হবে! 🚨 Pose হলো interpretable visual input, নিউরাল নেটওয়ার্কের parameters না!\n\n"
        "Words/phrases to keep intact: Pose estimation, DeepLabCut, Model input, Anatomical landmarks\n"
        "Fix: Change 'Task of an animal to express pose' -> 'Computer vision task of estimating animal posture', Change 'Parameter values of model' -> 'Model input / input features'"
    )
}

ch3_reviews = {
    3: (
        "হাতেনাতে ধরা খাইছোস! 🚨 সেকেন্ড হাফের পুরাটা ('Because of this, the requirements include...') অরিজিনাল লেখা থেকে ১০০% হুবহু কপি-পেস্ট করছোস! "
        "Turnitin দেখলে আগুন ধইরা যাবে 🎄!\n\n"
        "Words/phrases to keep intact: Requirements, Constraints, Evaluation protocols\n"
        "Fix: Change 100% verbatim second half -> Paraphrase the requirement sentences in your own words"
    ),
    9: (
        "'Not leaking' শুনতে প্লাম্বারের পাইপ মেরামতের মতো লাগে 🚰! ওটা Machine Learning-এর data leakage prevention! "
        "আর 'Keep big checkpoints' কোনো একাডেমিক টোন না!\n\n"
        "Words/phrases to keep intact: Data leakage prevention, Split integrity, Model checkpoints\n"
        "Fix: Change 'Not leaking' -> 'Data leakage prevention', Change 'Keep big checkpoints' -> 'Preserve full model checkpoints for reproducible audit'"
    ),
    12: (
        "'Perceptual mistakes'? মডেল গরু ডিটেক্ট করতে ফেইল করছে, কোনো দার্শনিক ভুলভ্রান্তি করে নাই 👻! লেখো 'perception failures'!\n\n"
        "Words/phrases to keep intact: Perception failures, Detection misses, Segmentation quality\n"
        "Fix: Change 'Perceptual mistakes' -> 'Perception failures / detector false negatives'"
    ),
    22: (
        "১) 'Species'?! গরু নিজেই একটা একক প্রজাতি (*Bos taurus*), তোরা লিখতে চাইছোস 'breed' (জাত) 🐄! "
        "২) 'The agri-environment is giving access' মানে কি গোয়ালঘর ওয়াইফাই পাসওয়ার্ড দিচ্ছে 📶?!\n\n"
        "Words/phrases to keep intact: Breed, Bos taurus, Agricultural environment challenges\n"
        "Fix: Change 'Species' -> 'Breed (e.g. Holstein, Angus)', Change 'Agri-environment giving access' -> 'Operational agricultural environment conditions'"
    ),
    29: (
        "মডেলের আর্কিটেকচার উল্টায় দিছোস! 🔄 তোরা লিখছোস বড় ডিটেক্টর মডেল ছোট মডেলের ওপর নির্ভর করে! "
        "আসলে উল্টা—ছোট ডাউনস্ট্রিম মডেল ভারী আপস্ট্রিম মডেলের ওপর নির্ভর করে!\n\n"
        "Words/phrases to keep intact: Upstream perception models, Downstream task heads, Sequential dependency\n"
        "Fix: Change 'Heavy detector depends on minor models' -> 'Lightweight downstream task heads depend on heavy upstream perception backbones'"
    ),
    39: (
        "মারাত্মক আবিষ্কার! 'Invisible cow assessment'?! 👻🐮 আমরা কি ফার্মের ভূতুড়ে গরুর বিচার করতেছি নাকি?! "
        "আর গরুর Behavior-রে বানাইছোস 'Evaluation of conduct'! মামা, ওটা Unseen-cow evaluation (যে গরু মডেল training-এ দেখে নাই), অদৃশ্য গরু না! 😭\n\n"
        "Words/phrases to keep intact: Unseen-cow evaluation, Cattle behavior, Ground-truth masks, Oracle evaluation\n"
        "Fix: Change 'Invisible cow assessment' -> 'Unseen-cow evaluation', Change 'Evaluation of conduct' -> 'Behavior evaluation'"
    ),
    45: (
        "'Measures the number of errors removed by perceptual errors'—এটা পুরা খিচুড়ি মার্কা কথা! 🥗 "
        "আসল কথা হলো কত শতাংশ ইমেজ সফলভাবে সেগমেন্ট হইছে তার কভারেজ রিপোর্ট করা!\n\n"
        "Words/phrases to keep intact: Segmentation coverage rate, Perception filter, Drop rate\n"
        "Fix: Change 'Errors removed by perceptual errors' -> 'Fraction of samples successfully processed by the perception pipeline (coverage rate)'"
    ),
    68: (
        "খলিলুর রহমান স্যাররে তোরা 'Overseer' বানায় দিলি?! 💀 স্যার কি ১৯ শতকের ব্রিটিশ আমলের নীলচাষীদের ওপর চাবুক মারা ম্যানেজার নাকি?! "
        "শব্দটা হইলো Supervisors / Advisors! খলিল স্যার এই ড্রাফট দেখলে thesis defense-এর আগেই আমাদের কাঁচা চিবায়া খাবে!\n\n"
        "Words/phrases to keep intact: Supervisors, Advisors, Authorship confirmation\n"
        "Fix: Change 'Overseers' -> 'Supervisors / Advisors'"
    ),
    72: (
        "১) 'Appropriate comments about train-test splits'?! আসল রিস্ক হলো DATA LEAKAGE (স্প্লিটের ভেতর ডেটা লিক হওয়া), কোনো ভালো কমেন্ট না! "
        "২) টেবিলের নাম 'Tabletab:dangers' বানায় দিও না!\n\n"
        "Words/phrases to keep intact: Data leakage, Train-test split, Table reference\n"
        "Fix: Change 'Appropriate comments' -> 'Rigorous protocol for data leakage prevention', Change 'Tabletab:dangers' -> Clean citation/reference"
    ),
    75: (
        "১) 'Ambulation'?! 🚶‍♂️ আমাদের ৫টা ক্লাসের অফিসিয়াল নাম 'Walking'—ল্যাটিন বানিয়ে দিও না! "
        "২) 'Behavioral films'? ওগুলো ১০ সেকেন্ডের সিসিটিভি ক্লিপ, হলিউডের মুভি না!\n\n"
        "Words/phrases to keep intact: Walking, CCTV video clips, Behavior classes\n"
        "Fix: Change 'Ambulation' -> 'Walking', Change 'Behavioral films' -> 'Video clips / CCTV footage'"
    )
}

def update_tab(tab_name, updates_dict):
    service = get_service()
    data = []
    for row_num, text in updates_dict.items():
        data.append({
            "range": f"'{tab_name}'!D{row_num}",
            "values": [[text]]
        })
    
    res = robust_execute(lambda: service.values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={"valueInputOption": "USER_ENTERED", "data": data}
    ).execute())
    print(f"[SUCCESS] Updated {res.get('totalUpdatedCells')} cells in '{tab_name}'!")

if __name__ == "__main__":
    update_tab("Chapter 1: Introduction", ch1_reviews)
    update_tab("Chapter 2: Literature Review", ch2_reviews)
    update_tab("Chapter 3: Requirements & Constraints", ch3_reviews)
    print("\n🏆 ALL 38 REVIEWS ACROSS ALL CHAPTERS STANDARDIZED WITH 'Words/phrases to keep intact' & 'Fix'!")
