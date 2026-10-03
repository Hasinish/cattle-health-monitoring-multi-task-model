# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-10-04 00:41:00`  
> **Active Target Account:** `tigerwood697`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **4** (`hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$62.00** |
| 💸 **Total Consumed** | **$2.55** |
| 🟢 **Total Remaining Balance** | **$59.45** |
| 🚀 **Total Combined L40S Runtime** | **~1827 mins (~30.4 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~718 mins (~12.0 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hasinishrak74001` | Standby | $1.00 | $0.68 | **$0.32** | ~9m | ~3m | 🟡 Low | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.00 | **$1.00** | ~30m | ~12m | 🟢 Healthy | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | Standby | $30.00 | $1.32 | **$28.68** | ~882m | ~347m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | 🔥 **ACTIVE** | $30.00 | $0.55 | **$29.45** | ~906m | ~356m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

---

## 🔍 Detailed Account Breakdowns

### 💳 Profile: `hasinishrak74001` 
- **Status**: 🟡 Low
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$0.68`
- **Remaining Balance**: **`$0.32`**
- **Est. L40S GPU Runtime**: **~9 minutes** (0h 9m)
- **Est. H100 GPU Runtime**: **~3 minutes** (0h 3m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/hasinishrak74001/billing`
### 💳 Profile: `mohtasimahmedsamii` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$0.00`
- **Remaining Balance**: **`$1.00`**
- **Est. L40S GPU Runtime**: **~30 minutes** (0h 30m)
- **Est. H100 GPU Runtime**: **~12 minutes** (0h 12m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/mohtasimahmedsamii/billing`
### 💳 Profile: `tigerwood693` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$1.32`
- **Remaining Balance**: **`$28.68`**
- **Est. L40S GPU Runtime**: **~882 minutes** (14h 42m)
- **Est. H100 GPU Runtime**: **~347 minutes** (5h 47m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood693/billing`
### 💳 Profile: `tigerwood697` (CURRENT ACTIVE)
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$0.55`
- **Remaining Balance**: **`$29.45`**
- **Est. L40S GPU Runtime**: **~906 minutes** (15h 6m)
- **Est. H100 GPU Runtime**: **~356 minutes** (5h 56m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood697/billing`

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
