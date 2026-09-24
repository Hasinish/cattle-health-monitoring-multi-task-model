# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-09-25 00:40:11`  
> **Active Target Account:** `hasinishrak2015`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **6** (`dryousufmozumder, hasinishrak2015, hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$122.00** |
| 💸 **Total Consumed** | **$46.55** |
| 🟢 **Total Remaining Balance** | **$75.81** |
| 🚀 **Total Combined L40S Runtime** | **~2330 mins (~38.8 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~916 mins (~15.3 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dryousufmozumder` | Standby | $30.00 | $3.49 | **$26.51** | ~815m | ~321m | 🟢 Healthy | [View Ledger](https://modal.com/settings/dryousufmozumder/billing) |
| `hasinishrak2015` | 🔥 **ACTIVE** | $30.00 | $0.88 | **$29.12** | ~896m | ~352m | 🟢 Healthy | [View Ledger](https://modal.com/settings/hasinishrak2015/billing) |
| `hasinishrak74001` | Standby | $1.00 | $1.36 | **$0.00** | ~0m | ~0m | 🔴 Depleted | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.95 | **$0.05** | ~1m | ~0m | 🟡 Low | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | Standby | $30.00 | $28.49 | **$1.51** | ~46m | ~18m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | Standby | $30.00 | $11.39 | **$18.61** | ~572m | ~225m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

---

## 🔍 Detailed Account Breakdowns

### 💳 Profile: `dryousufmozumder` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$3.49`
- **Remaining Balance**: **`$26.51`**
- **Est. L40S GPU Runtime**: **~815 minutes** (13h 35m)
- **Est. H100 GPU Runtime**: **~321 minutes** (5h 21m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/dryousufmozumder/billing`
### 💳 Profile: `hasinishrak2015` (CURRENT ACTIVE)
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$0.88`
- **Remaining Balance**: **`$29.12`**
- **Est. L40S GPU Runtime**: **~896 minutes** (14h 56m)
- **Est. H100 GPU Runtime**: **~352 minutes** (5h 52m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/hasinishrak2015/billing`
### 💳 Profile: `hasinishrak74001` 
- **Status**: 🔴 Depleted
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$1.36`
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
### 💳 Profile: `tigerwood693` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$28.49`
- **Remaining Balance**: **`$1.51`**
- **Est. L40S GPU Runtime**: **~46 minutes** (0h 46m)
- **Est. H100 GPU Runtime**: **~18 minutes** (0h 18m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood693/billing`
### 💳 Profile: `tigerwood697` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$11.39`
- **Remaining Balance**: **`$18.61`**
- **Est. L40S GPU Runtime**: **~572 minutes** (9h 32m)
- **Est. H100 GPU Runtime**: **~225 minutes** (3h 45m)
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
