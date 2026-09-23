# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-09-23 19:14:41`  
> **Active Target Account:** `tigerwood693`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **6** (`dryousufmozumder, hasinishrak2015, hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$122.00** |
| 💸 **Total Consumed** | **$29.62** |
| 🟢 **Total Remaining Balance** | **$92.50** |
| 🚀 **Total Combined L40S Runtime** | **~2844 mins (~47.4 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~1118 mins (~18.6 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dryousufmozumder` | Standby | $30.00 | $0.00 | **$30.00** | ~923m | ~363m | 🟢 Healthy | [View Ledger](https://modal.com/settings/dryousufmozumder/billing) |
| `hasinishrak2015` | Standby | $30.00 | $0.00 | **$30.00** | ~923m | ~363m | 🟢 Healthy | [View Ledger](https://modal.com/settings/hasinishrak2015/billing) |
| `hasinishrak74001` | Standby | $1.00 | $1.12 | **$0.00** | ~0m | ~0m | 🔴 Depleted | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.95 | **$0.05** | ~1m | ~0m | 🟡 Low | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | 🔥 **ACTIVE** | $30.00 | $21.20 | **$8.80** | ~270m | ~106m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | Standby | $30.00 | $6.35 | **$23.65** | ~727m | ~286m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

---

## 🔍 Detailed Account Breakdowns

### 💳 Profile: `dryousufmozumder` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$0.00`
- **Remaining Balance**: **`$30.00`**
- **Est. L40S GPU Runtime**: **~923 minutes** (15h 23m)
- **Est. H100 GPU Runtime**: **~363 minutes** (6h 3m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/dryousufmozumder/billing`
### 💳 Profile: `hasinishrak2015` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$0.00`
- **Remaining Balance**: **`$30.00`**
- **Est. L40S GPU Runtime**: **~923 minutes** (15h 23m)
- **Est. H100 GPU Runtime**: **~363 minutes** (6h 3m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/hasinishrak2015/billing`
### 💳 Profile: `hasinishrak74001` 
- **Status**: 🔴 Depleted
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$1.12`
- **Remaining Balance**: **`$0.00`**
- **Est. L40S GPU Runtime**: **~0 minutes** (0h 0m)
- **Est. H100 GPU Runtime**: **~0 minutes** (0h 0m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/hasinishrak74001/billing`
### 💳 Profile: `mohtasimahmedsamii` 
- **Status**: 🟡 Low
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$0.95`
- **Remaining Balance**: **`$0.05`**
- **Est. L40S GPU Runtime**: **~1 minutes** (0h 1m)
- **Est. H100 GPU Runtime**: **~0 minutes** (0h 0m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/mohtasimahmedsamii/billing`
### 💳 Profile: `tigerwood693` (CURRENT ACTIVE)
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$21.20`
- **Remaining Balance**: **`$8.80`**
- **Est. L40S GPU Runtime**: **~270 minutes** (4h 30m)
- **Est. H100 GPU Runtime**: **~106 minutes** (1h 46m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood693/billing`
### 💳 Profile: `tigerwood697` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$6.35`
- **Remaining Balance**: **`$23.65`**
- **Est. L40S GPU Runtime**: **~727 minutes** (12h 7m)
- **Est. H100 GPU Runtime**: **~286 minutes** (4h 46m)
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
