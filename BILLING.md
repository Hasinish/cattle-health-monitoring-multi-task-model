# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-09-25 05:12:11`  
> **Active Target Account:** `hasinishrak2015`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **6** (`dryousufmozumder, hasinishrak2015, hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$122.00** |
| 💸 **Total Consumed** | **$76.65** |
| 🟢 **Total Remaining Balance** | **$45.71** |
| 🚀 **Total Combined L40S Runtime** | **~1405 mins (~23.4 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~551 mins (~9.2 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dryousufmozumder` | Standby | $30.00 | $9.88 | **$20.12** | ~619m | ~243m | 🟢 Healthy | [View Ledger](https://modal.com/settings/dryousufmozumder/billing) |
| `hasinishrak2015` | 🔥 **ACTIVE** | $30.00 | $11.86 | **$18.14** | ~558m | ~219m | 🟢 Healthy | [View Ledger](https://modal.com/settings/hasinishrak2015/billing) |
| `hasinishrak74001` | Standby | $1.00 | $1.36 | **$0.00** | ~0m | ~0m | 🔴 Depleted | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.95 | **$0.05** | ~1m | ~0m | 🟡 Low | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | Standby | $30.00 | $29.15 | **$0.85** | ~26m | ~10m | 🟡 Low | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | Standby | $30.00 | $23.46 | **$6.54** | ~201m | ~79m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

---

## 🔍 Detailed Account Breakdowns

### 💳 Profile: `dryousufmozumder` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$9.88`
- **Remaining Balance**: **`$20.12`**
- **Est. L40S GPU Runtime**: **~619 minutes** (10h 19m)
- **Est. H100 GPU Runtime**: **~243 minutes** (4h 3m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/dryousufmozumder/billing`
### 💳 Profile: `hasinishrak2015` (CURRENT ACTIVE)
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$11.86`
- **Remaining Balance**: **`$18.14`**
- **Est. L40S GPU Runtime**: **~558 minutes** (9h 18m)
- **Est. H100 GPU Runtime**: **~219 minutes** (3h 39m)
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
- **Status**: 🟡 Low
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$29.15`
- **Remaining Balance**: **`$0.85`**
- **Est. L40S GPU Runtime**: **~26 minutes** (0h 26m)
- **Est. H100 GPU Runtime**: **~10 minutes** (0h 10m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood693/billing`
### 💳 Profile: `tigerwood697` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$23.46`
- **Remaining Balance**: **`$6.54`**
- **Est. L40S GPU Runtime**: **~201 minutes** (3h 21m)
- **Est. H100 GPU Runtime**: **~79 minutes** (1h 19m)
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
