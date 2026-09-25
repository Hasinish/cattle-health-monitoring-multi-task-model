"""
Add 'যা করবা:' action items after all roasts in Chapter 1 and Chapter 3.
Preserves all existing roasts, adds concrete instructions, and maintains exact formatting.
"""

import sys
import socket
import time
socket.setdefaulttimeout(15.0)

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute

ch1_actions = {
    3: (
        "অসাধারণ scientific logic! 'Reduced food intake causes a cow to eat less' (কম খাওয়া একটা গরুকে কম খাওয়ায়!) 💀 "
        "আইনস্টাইন তো কবরে বইসা কাঁদতেছে! তোদের পরের research কি 'পানি খাওয়ার কারণে গরুর পানির পিপাসা মেটে' আর 'every 60 seconds in Africa, a minute passes'?! পুরা circular logic-এর কারখানা!\n\n"
        "যা করবা: কম খাওয়া আর শরীরের কন্ডিশন কমে যাওয়া বা আলাদাভাবে রেস্ট নেওয়াকে আলাদা লক্ষণ হিসেবে লেখো, একটা আরেকটার কারণ না!"
    ),
    9: (
        "'Trivial incidents'?! ☕ একটা গরু পানি খাওয়া, খাবার খাওয়া বা অসুস্থ হওয়া কোনো 'trivial incident' (তুচ্ছ ঘটনা) না! "
        "আর 'short observations' মানে সময়ের ব্যাপ্তি ছোট, কোনো লাজুক 'modest' স্বভাব না!\n\n"
        "যা করবা: 'Trivial incidents' বাদ দিয়ে 'routine welfare events' বা 'short-duration behaviors' লেখো!"
    ),
    12: (
        "প্রথম লাইনটা সিনেমার ডায়লগের মতো কোটেশন মার্কসে (\" \") কেন দিছোস? 🎬 ডিরেক্ট কোটেশন দিলে Turnitin সাথে সাথে প্ল্যাজিয়ারিজম ধরবে! "
        "আর 'applications... in other livestock applications' ৫ শব্দের মধ্যে দুইবার applications লিখছোস!\n\n"
        "যা করবা: কোটেশন মার্কস তুলে দিয়ে নিজের ভাষায় লেখো এবং একই শব্দের পুনরাবৃত্তি বাদ দাও!"
    ),
    15: (
        "'Learned traits' আর 'various duties'?! 😭 অ্যানিম্যাল সায়েন্সে 'traits' মানে জেনেটিক্স, নিউরাল নেটওয়ার্কের ফিচার না! "
        "আর মডেল Machine Learning-এর 'tasks' করে, ৯-টা ৫-টা অফিসের 'duties' না!\n\n"
        "যা করবা: 'features / visual representations' এবং 'tasks' শব্দগুলো intact রাখো, 'traits' বা 'duties' বাদ দাও!"
    ),
    18: (
        "ভাইরে ভাই! তোরা ডিকশনারি থেইকা সিনোনিম মাইরা Machine Learning-এর Tasks-রে Occupations (পেশা/চাকরি) বানায় দিলি?! 💀 "
        "গরু কি এখন ফরমাল শার্ট পইরা ৯-টা ৫-টা অফিস করবে, নাকি LinkedIn-এ প্রোফাইল খুইলা ট্যাক্স ফাইল জমা দিবে?! MTL model-এর tasks আর মানুষের corporate jobs এক জিনিস না রে ভাই!\n\n"
        "যা করবা: 'tasks' শব্দটা intact রাখো, 'occupations' বা 'jobs' পুরোপুরি বাদ দাও!"
    ),
    24: (
        "ভাইরে ভাই! তোরা ডিকশনারি থেইকা সিনোনিম মাইরা Machine Learning-এর Tasks-রে Occupations (পেশা/চাকরি) বানায় দিলি?! 💀 "
        "'Two occupations responsible for the care of the same cow'?! 🧑‍🌾👨‍🌾 এখানে দুইটা কম্পিউটার ভিশন মডেল (BCS এবং Re-ID) একই গরুর ছবি প্রসেস করতেছে, কোনো ডাক্তার আর খামারি মিটিং করতেছে না!\n\n"
        "যা করবা: এখানে দুটি আলাদা ML task (BCS এবং Re-ID) একই গরুর ইমেজ প্রসেস করে—এটা পরিষ্কারভাবে লেখো!"
    ),
    27: (
        "১) BCS মানে বডির চর্বি/এনার্জি রিজার্ভ, কোনো 'sickness' (অসুখ) না 🤒! বেশি বিসিএস হওয়া মানে গরু সুস্থ-সবল, অসুস্থ না! "
        "২) গোয়ালঘরের খাবার জায়গা কোনো 'geography' 🗺️ না, এটা ব্যাকগ্রাউন্ড সিন!\n\n"
        "যা করবা: BCS-কে শরীরের এনার্জি রিজার্ভ/ফ্যাট কন্ডিশন হিসেবে লেখো, আর গোয়ালঘরকে 'barn / pen background' হিসেবে উল্লেখ করো!"
    ),
    30: (
        "গরু আর তার 'House'?! 🏡 গরু কি এখন ব্যাংক থেইকা home loan নিয়া গুলশানে ডুপ্লেক্স বাড়ি বানাইয়া থাকে নাকি রে ভাই?! "
        "ওটা গোয়ালঘর আর খামার (Barn / Pen), গরুর কোনো real estate বা flat নাই! আর 'Shortcut learning'-রে 'quick learning' বানায় দিও না!\n\n"
        "যা করবা: 'Shortcut learning' এবং 'Barn / Pen' শব্দগুলো intact রাখো, 'house' বা 'quick learning' বাদ দাও!"
    ),
    33: (
        "'The target cow is localized through localization'—অসাধারণ বৈজ্ঞানিক থিওরি! 🧠💀 "
        "'গরু লোকালাইজড হয় কারণ সে লোকালাইজড!' তোদের পরের পেপারের লাইন কি 'Cows eat food because they are eating'?! "
        "আর আবার 'Job-specific'?! গরু কি বিসিএস ক্যাডার নাকি কর্পোরেট ব্যাংকার যে তার আলাদা Job থাকবে?! শব্দটা হইলো 'Task-specific'!\n\n"
        "যা করবা: 'target cow is localized' পুনরাবৃত্তি বাদ দিয়ে স্বাভাবিকভাবে লেখো, এবং 'job-specific' বাদ দিয়ে 'task-specific' intact রাখো!"
    ),
    40: (
        "১) 'Movie'?! 🍿 গোয়ালঘরের CCTV ফুটেজ দেখতে দেখতে কি সিনেমা হলের ফিল পাচ্ছোস নাকি?! "
        "২) ট্রেন-টেস্ট স্প্লিট কোনো 'characteristics মেজার' করে না—কাছাকাছি ফ্রেমের কারণে data leakage হয় যা রেজাল্ট নষ্ট করে!\n\n"
        "যা করবা: 'movie' বাদ দিয়ে 'video footage / surveillance frames' লেখো, এবং ফ্রেম কোরিলেশনের কারণে 'data leakage' ঘটে তা স্পষ্টভাবে উল্লেখ করো!"
    ),
    43: (
        "কপি-পেস্ট ডিজাস্টার! 🤦‍♂️ তোরা রো ৪০-এর মুভির প্যারাগ্রাফটাই হুবহু কপি কইরা রো ৪৩-এ পেস্ট করছোস! "
        "রো ৪৩ ছিল মডেলের কম্পোনেন্ট আর single-task baseline নিয়া!\n\n"
        "যা করবা: রো ৪০-এর টেক্সট মুছে ফেলে অরিজিনাল রো ৪৩-এর মডেল কম্পোনেন্ট ও single-task reference baseline আলোচনা নিজের ভাষায় প্যারাফ্রেজ করো!"
    ),
    47: (
        "'Recognition of activity is activity'?! 🤖 ভাই এটা কেমন অর্থহীন বাক্য! লিখবা 'Behavior recognition focuses on temporal activity'.\n\n"
        "যা করবা: 'Behavior recognition focuses on temporal posture and movement' এভাবে পরিষ্কার করে লেখো!"
    ),
    53: (
        "১) 'The input is to look into...'?! একটা থিসিসের 'contribution' থাকে, 'input' না! "
        "২) 'Quick segmentation' না, ওটা ছিল 'promptable segmentation' (SAM model)!\n\n"
        "যা করবা: 'contribution' এবং 'promptable segmentation' টার্মগুলো intact রাখো!"
    ),
    57: (
        "'Teach and send visual representations'?! 📬 আমরা কি ডাকযোগে চিঠি পাঠাচ্ছি নাকি?! "
        "এমএল-এ বলে 'learn and share representations across tasks'!\n\n"
        "যা করবা: 'learn and share visual representations' শব্দগুচ্ছ ব্যবহার করো!"
    ),
    63: (
        "'The volume of information to be conveyed' শুনতে ব্রডব্যান্ড ইন্টারনেটের প্যাকেজ মনে হয় 📶! "
        "আসল কথা হইলো টাস্কগুলোর মধ্যে কতটুকু প্যারামিটার 'share' করা উচিত!\n\n"
        "যা করবা: 'parameter capacity shared between tasks' ধারণাটুকু স্পষ্টভাবে লেখো!"
    ),
    73: (
        "ভুল তথ্য! ❌ তোরা লিখছোস অরিজিনাল আইডি পাওয়া যায় না! অথচ SideViewCows2026-এ ১১০টা গরুর অরিজিনাল বায়োলজিক্যাল আইডি আছে! শুধু ScienceDB-তে নাই!\n\n"
        "যা করবা: SideViewCows2026-এ ১১০টি আসল গরুর আইডি আছে এবং কেবল ScienceDB-তে বায়োলজিক্যাল আইডি নেই—তথ্যটি শুদ্ধ করে লেখো!"
    ),
    79: (
        "'Geolocation'?! 🛰️📍 তোরা কি গরুর পেছনে NASA-র military satellite GPS ট্র্যাকার বসায়া ড্রোন হামলা চালাবি নাকি?! "
        "ক্যামেরার ফ্রেমে গরুর চারপাশের bounding box-রে বলে Localization, Google Maps-এর live location না!\n\n"
        "যা করবা: 'geolocation' বাদ দিয়ে 'bounding-box localization' টার্মটি intact রাখো!"
    ),
    82: (
        "'Without harming individual actions'?! গরুর কোনো পার্সোনাল কাজে ক্ষতি হচ্ছে না রে ভাই! "
        "ওটা ছিল 'without hurting individual tasks' (negative transfer আটকানো)!\n\n"
        "যা করবা: 'negative transfer' বা 'hurting individual tasks' কনসেপ্টটি অক্ষত রাখো!"
    ),
    92: (
        "প্যারাগ্রাফের মাঝখানে একা একা '1.' ঝুলতেছে কেন? 🔢 নাম্বার ২ কি হারিয়ে গেছে?!\n\n"
        "যা করবা: অপ্রাসঙ্গিক '1.' সরিয়ে বাক্যটি স্বাভাবিক ফ্লোতে লেখো!"
    ),
    95: (
        "অর্থ বদলে গেছে: তোদের লেখায় মনে হচ্ছে শুধু behavior মডেলে মাস্ক ব্যবহার করা হয়! "
        "আসলে তিনটা টাস্কেই মাস্ক ব্যবহার করা হয়, শুধু behavior-এ temporal যোগ হয়!\n\n"
        "যা করবা: তিনটি টাস্কেই visual crop ও mask ব্যবহার করা হয় এবং কেবল Behavior টাস্কে temporal aggregation যোগ হয়—অর্থটি ঠিক করো!"
    ),
    98: (
        "'Retrieval assignment'?! 📝 Re-ID হলো কম্পিউটার ভিশনের information retrieval task, স্কুলের হোমওয়ার্ক অ্যাসাইনমেন্ট না!\n\n"
        "যা করবা: 'information retrieval task' টার্মটি intact রাখো, 'assignment' বাদ দাও!"
    ),
    101: (
        "'Collected in a collaborative manner'?! 🤝 এটা ডিপ লার্নিংয়ের multi-task parameter sharing, খামারিদের দলবদ্ধ হয়ে কাজ করা না!\n\n"
        "যা করবা: 'multi-task parameter sharing' ধারণাটি স্পষ্ট করে লেখো!"
    ),
    105: (
        "সবচেয়ে বড় disaster তো এইখানে! তোরা thesis-এর main topic BCS (Body Condition Scoring)-ই পুরা হাওয়া কইরা দিছোস! 🚨 "
        "Individual Cow Identification আর Re-ID যে একই জিনিস, ওইটা দুইবার লেইখা আসল জিনিস BCS-ই নাই! আমরা কিসের thesis করতেছি তোরা নিজেরাও জানোস তো নাকি ভুলে গেছোস?!\n\n"
        "যা করবা: তিনটি মূল টাস্ক: 1) Body Condition Scoring (BCS), 2) Behavior Recognition, এবং 3) Individual Cow Identification / Re-ID—সঠিকভাবে যুক্ত করো!"
    ),
    111: (
        "১) 'Barrier' না, কম্পিউটার ভিশনে ওটাকে বলে 'occlusion' (দৃষ্টিসীমা আটকে যাওয়া)! "
        "২) 'Validated in the real world' কোনো প্রমাণ ছাড়া ওভারক্লেম—আমরা এক্সপেরিমেন্ট করছি, খামারে ডিপ্লয় করি নাই!\n\n"
        "যা করবা: 'occlusion' টার্মটি ব্যবহার করো এবং 'validated in benchmark experiments' লেখো, লাইভ ডিপ্লয়মেন্ট দাবি করো না!"
    ),
    117: (
        "'Combining many activities in real life'?! 🏃‍♂️ এটা মাল্টি-টাস্ক ডিপ লার্নিং মডেল ইন্টিগ্রেশন, বাস্তব জীবনের মাল্টিটাস্কিং না!\n\n"
        "যা করবা: 'unified multi-task deep learning framework' কথাটি অক্ষত রাখো!"
    )
}

ch3_actions = {
    3: (
        "হাতেনাতে ধরা খাইছোস! 🚨 সেকেন্ড হাফের পুরাটা ('Because of this, the requirements include...') অরিজিনাল লেখা থেকে ১০০% হুবহু কপি-পেস্ট করছোস! "
        "Turnitin দেখলে আগুন ধইরা যাবে 🎄!\n\n"
        "যা করবা: কপি করা দ্বিতীয় অংশটি সম্পূর্ণ নিজের ভাষায় নতুন করে রিরাইট করো!"
    ),
    9: (
        "'Not leaking' শুনতে প্লাম্বারের পাইপ মেরামতের মতো লাগে 🚰! ওটা Machine Learning-এর data leakage prevention! "
        "আর 'Keep big checkpoints' কোনো একাডেমিক টোন না!\n\n"
        "যা করবা: 'data leakage prevention' টার্মটি intact রাখো এবং ফরমাল একাডেমিক টোনে মডেল চেকপয়েন্ট সংরক্ষণের কথা লেখো!"
    ),
    12: (
        "'Perceptual mistakes'? মডেল গরু ডিটেক্ট করতে ফেইল করছে, কোনো দার্শনিক ভুলভ্রান্তি করে নাই 👻! লেখো 'perception failures'!\n\n"
        "যা করবা: 'perception failures / detection misses' শব্দগুচ্ছ ব্যবহার করো!"
    ),
    22: (
        "১) 'Species'?! গরু নিজেই একটা একক প্রজাতি (*Bos taurus*), তোরা লিখতে চাইছোস 'breed' (জাত) 🐄! "
        "২) 'The agri-environment is giving access' মানে কি গোয়ালঘর ওয়াইফাই পাসওয়ার্ড দিচ্ছে 📶?!\n\n"
        "যা করবা: 'breed' শব্দটি ব্যবহার করো এবং 'agricultural environment challenges' স্পষ্টভাবে লেখো!"
    ),
    29: (
        "মডেলের আর্কিটেকচার উল্টায় দিছোস! 🔄 তোরা লিখছোস বড় ডিটেক্টর মডেল ছোট মডেলের ওপর নির্ভর করে! "
        "আসলে উল্টা—ছোট ডাউনস্ট্রিম মডেল ভারী আপস্ট্রিম মডেলের ওপর নির্ভর করে!\n\n"
        "যা করবা: ডাউনস্ট্রিম লাইটওয়েট টাস্ক হেড ভারী আপস্ট্রিম পারসেপশন মডেলের ওপর নির্ভরশীল—এই অনুক্রমটি ঠিক করো!"
    ),
    39: (
        "মারাত্মক আবিষ্কার! 'Invisible cow assessment'?! 👻🐮 আমরা কি ফার্মের ভূতুড়ে গরুর বিচার করতেছি নাকি?! "
        "আর গরুর Behavior-রে বানাইছোস 'Evaluation of conduct'! গরু কি স্কুলের প্রিন্সিপালের কাছে character certificate নিতে আসছে?! "
        "মামা, ওটা Unseen-cow evaluation (যে গরু মডেল training-এ দেখে নাই), অদৃশ্য গরু না! 😭\n\n"
        "যা করবা: 'unseen-cow evaluation' এবং 'cattle behavior' টার্মগুলো intact রাখো!"
    ),
    45: (
        "'Measures the number of errors removed by perceptual errors'—এটা পুরা খিচুড়ি মার্কা কথা! 🥗 "
        "আসল কথা হলো কত শতাংশ ইমেজ সফলভাবে সেগমেন্ট হইছে তার কভারেজ রিপোর্ট করা!\n\n"
        "যা করবা: পারসেপশন পাইপলাইনের 'segmentation coverage rate' পরিষ্কারভাবে উল্লেখ করো!"
    ),
    68: (
        "খলিলুর রহমান স্যাররে তোরা 'Overseer' বানায় দিলি?! 💀 স্যার কি ১৯ শতকের ব্রিটিশ আমলের নীলচাষীদের ওপর চাবুক মারা ম্যানেজার নাকি?! "
        "শব্দটা হইলো Supervisors / Advisors! খলিল স্যার এই ড্রাফট দেখলে thesis defense-এর আগেই আমাদের কাঁচা চিবায়া খাবে!\n\n"
        "যা করবা: 'Supervisors' বা 'Advisors' শব্দ ব্যবহার করো, 'overseers' বাদ দাও!"
    ),
    72: (
        "১) 'Appropriate comments about train-test splits'?! আসল রিস্ক হলো DATA LEAKAGE (স্প্লিটের ভেতর ডেটা লিক হওয়া), কোনো ভালো কমেন্ট না! "
        "২) টেবিলের নাম 'Tabletab:dangers' বানায় দিও না!\n\n"
        "যা করবা: 'train-test data leakage risk' স্পষ্ট করো এবং টেবিল সাইটেশন ফরম্যাট ঠিক করো!"
    ),
    75: (
        "১) 'Ambulation'?! 🚶‍♂️ আমাদের ৫টা ক্লাসের অফিসিয়াল নাম 'Walking'—ল্যাটিন বানিয়ে দিও না! "
        "২) 'Behavioral films'? ওগুলো ১০ সেকেন্ডের সিসিটিভি ক্লিপ, হলিউডের মুভি না!\n\n"
        "যা করবা: অফিসিয়াল ক্লাস নাম 'Walking' intact রাখো এবং 'video clips / camera footage' লেখো!"
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
    print(f"[SUCCESS] Updated {res.get('totalUpdatedCells')} cells in {tab_name} with 'যা করবা:' action items!")

if __name__ == "__main__":
    update_tab("Chapter 1: Introduction", ch1_actions)
    update_tab("Chapter 3: Requirements & Constraints", ch3_actions)
    print("\n🏆 ALL ROASTS IN CHAPTER 1 & 3 NOW HAVE EXPLICIT 'যা করবা:' INSTRUCTIONS!")
