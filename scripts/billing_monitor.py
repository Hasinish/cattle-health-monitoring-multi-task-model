# -*- coding: utf-8 -*-
"""
Real-Time Modal Billing & Credit Dashboard Monitor
Syncs live billing information across all Modal profiles and updates BILLING.md.
"""
import os
import sys
import time
import json
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        tomllib = None

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILES = [
    os.path.join(ROOT_DIR, "BILLING.md"),
    r"D:\custom-antigravity\BILLING.md"
]

ACCOUNT_GRANTS = {
    "tigerwood693": 30.00,
    "hasinishrak2015": 30.00,
    "dryousufmozumder": 30.00,
    "tigerwood697": 30.00,
    "mohtasimahmedsamii": 1.00,
    "hasinishrak74001": 1.00
}

GPU_RATES = {
    "L40S": 1.95,
    "H100": 4.95,
    "L4": 0.80,
    "A100-40GB": 2.10,
    "A100-80GB": 3.70,
    "A10G": 1.10,
    "T4": 0.59
}

def discover_profiles():
    profiles = []
    modal_toml = os.path.expanduser("~/.modal.toml")
    if os.path.exists(modal_toml) and tomllib is not None:
        try:
            with open(modal_toml, "rb") as f:
                data = tomllib.load(f)
                profiles = [k for k in data.keys() if isinstance(data[k], dict)]
        except Exception:
            pass
    elif os.path.exists(modal_toml):
        # Basic line parser if no toml library installed
        try:
            with open(modal_toml, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        sec = line[1:-1].strip()
                        if sec and not sec.startswith("default"):
                            profiles.append(sec)
        except Exception:
            pass

    if not profiles:
        profiles = list(ACCOUNT_GRANTS.keys())
    return sorted(list(set(profiles)))

def get_active_profile():
    try:
        proc = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "hasinishrak74001"

def fetch_account_billing(profile):
    cmd = [sys.executable, "-m", "modal", "billing", "summary", "--profile", profile, "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            metered = float(data.get("metered_cost", 0.0))
            billed = float(data.get("billed_cost", 0.0))
            return profile, {"metered": metered, "billed": billed, "ok": True}
    except Exception:
        pass
    return profile, {"metered": 0.0, "billed": 0.0, "ok": False}

def update_dashboard():
    profiles = discover_profiles()
    active_profile = get_active_profile()

    results = {}
    with ThreadPoolExecutor(max_workers=max(len(profiles), 2)) as executor:
        for profile, data in executor.map(fetch_account_billing, profiles):
            results[profile] = data

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_grant = 0.0
    total_metered = 0.0
    total_remaining = 0.0
    total_l40s_mins = 0
    total_h100_mins = 0

    rows = []
    cards = []

    for p in profiles:
        data = results.get(p, {"metered": 0.0, "billed": 0.0, "ok": False})
        metered = data["metered"]
        billed = data["billed"]

        grant = float(os.environ.get(f"CREDIT_{p}", ACCOUNT_GRANTS.get(p, 1.00)))
        remaining = max(0.0, grant - metered)
        l40s_mins = int(remaining / (GPU_RATES["L40S"] / 60))
        h100_mins = int(remaining / (GPU_RATES["H100"] / 60))

        total_grant += grant
        total_metered += metered
        total_remaining += remaining
        total_l40s_mins += l40s_mins
        total_h100_mins += h100_mins

        is_active = (p == active_profile)
        active_badge = "🔥 **ACTIVE**" if is_active else "Standby"

        if remaining <= 0.05:
            status_badge = "🔴 Depleted"
        elif remaining < 1.00:
            status_badge = "🟡 Low"
        else:
            status_badge = "🟢 Healthy"

        rows.append(
            f"| `{p}` | {active_badge} | ${grant:.2f} | ${metered:.2f} | **${remaining:.2f}** | ~{l40s_mins}m | ~{h100_mins}m | {status_badge} | [View Ledger](https://modal.com/settings/{p}/billing) |"
        )

        cards.append(f"""### 💳 Profile: `{p}` {"(CURRENT ACTIVE)" if is_active else ""}
- **Status**: {status_badge}
- **Grant Ceiling**: `${grant:.2f}`
- **Metered Usage**: `${metered:.2f}`
- **Remaining Balance**: **`${remaining:.2f}`**
- **Est. L40S GPU Runtime**: **~{l40s_mins} minutes** ({l40s_mins // 60}h {l40s_mins % 60}m)
- **Est. H100 GPU Runtime**: **~{h100_mins} minutes** ({h100_mins // 60}h {h100_mins % 60}m)
- **Out-of-Pocket Billed**: `${billed:.2f}`
- **Direct Modal Link**: `https://modal.com/settings/{p}/billing`
""")

    l40s_str = f"~{total_l40s_mins} mins (~{total_l40s_mins / 60:.1f} hours)"
    h100_str = f"~{total_h100_mins} mins (~{total_h100_mins / 60:.1f} hours)"

    content = f"""# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `{now}`  
> **Active Target Account:** `{active_profile}`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **{len(profiles)}** (`{', '.join(profiles)}`) |
| 🎁 **Total Credit Grants** | **${total_grant:.2f}** |
| 💸 **Total Consumed** | **${total_metered:.2f}** |
| 🟢 **Total Remaining Balance** | **${total_remaining:.2f}** |
| 🚀 **Total Combined L40S Runtime** | **{l40s_str}** |
| ⚡ **Total Combined H100 Runtime** | **{h100_str}** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{"\n".join(rows)}

---

## 🔍 Detailed Account Breakdowns

{"".join(cards)}
---

## 💡 Quick Commands:
Activate any account:
```powershell
modal profile activate <PROFILE_NAME>
```
Start live auto-looping dashboard (updates every 10s until Ctrl+C):
```powershell
python monitor.py
```
Run instant single-shot terminal summary:
```powershell
python billing_monitor.py
```
"""

    for out_path in OUTPUT_FILES:
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass

    return {
        "now": now,
        "profiles": profiles,
        "active_profile": active_profile,
        "total_grant": total_grant,
        "total_metered": total_metered,
        "total_remaining": total_remaining,
        "total_l40s_mins": total_l40s_mins,
        "total_h100_mins": total_h100_mins,
        "results": results
    }

def print_terminal_summary(data):
    now = data["now"]
    active = data["active_profile"]
    remaining = data["total_remaining"]
    metered = data["total_metered"]
    grant = data["total_grant"]
    l40s = data["total_l40s_mins"]
    h100 = data["total_h100_mins"]

    print("=" * 65)
    print("📊 MODAL BILLING & CREDIT SUMMARY")
    print("=" * 65)
    print(f"🕒 Timestamp         : {now}")
    print(f"👤 Active Profile     : {active}")
    print(f"🎁 Total Grants      : ${grant:.2f}")
    print(f"💸 Total Consumed    : ${metered:.2f}")
    print(f"🟢 Total Remaining   : ${remaining:.2f}")
    print(f"🚀 Est. L40S Runtime : ~{l40s} mins (~{l40s / 60:.1f} hours)")
    print(f"⚡ Est. H100 Runtime : ~{h100} mins (~{h100 / 60:.1f} hours)")
    print("-" * 65)
    print(f"{'Profile':<20} {'Status':<10} {'Grant':<8} {'Used':<8} {'Remaining':<10} {'L40S':<8}")
    print("-" * 65)
    for p in data["profiles"]:
        res = data["results"].get(p, {})
        m = res.get("metered", 0.0)
        g = float(os.environ.get(f"CREDIT_{p}", ACCOUNT_GRANTS.get(p, 1.00)))
        rem = max(0.0, g - m)
        mins = int(rem / (GPU_RATES["L40S"] / 60))
        tag = "[ACTIVE]" if p == active else ""
        print(f"{p:<20} {tag:<10} ${g:<7.2f} ${m:<7.2f} ${rem:<9.2f} ~{mins}m")
    print("=" * 65)
    print(f"📄 Dashboard written to: {OUTPUT_FILES[0]}\n")

def run_monitor(interval_seconds=10):
    print("=" * 65)
    print("📈 Starting Real-Time Modal Billing Monitor")
    print(f"⏱️  Refresh Interval : Every {interval_seconds} seconds (Press Ctrl+C to stop)")
    print(f"📄 Output Dashboards : {OUTPUT_FILES}")
    print("=" * 65 + "\n")

    while True:
        try:
            data = update_dashboard()
            now = data["now"]
            rem = data["total_remaining"]
            act = data["active_profile"]
            print(f"[{now}] Updated BILLING.md | Total Remaining: ${rem:.2f} | Active: {act}")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n🛑 Billing monitor stopped by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(interval_seconds)

if __name__ == "__main__":
    if "--loop" in sys.argv or "monitor" in sys.argv:
        interval = 10
        for a in sys.argv[1:]:
            if a.isdigit():
                interval = int(a)
        run_monitor(interval)
    else:
        summary_data = update_dashboard()
        print_terminal_summary(summary_data)
