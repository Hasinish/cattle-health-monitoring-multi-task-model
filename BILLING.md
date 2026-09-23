# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-09-23 17:02:56`  
> **Active Target Account:** `tigerwood693`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **6** (`dryousufmozumder, hasinishrak2015, hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$122.00** |
| 💸 **Total Consumed** | **$25.76** |
| 🟢 **Total Remaining Balance** | **$96.24** |
| 🚀 **Total Combined L40S Runtime** | **~2959 mins (~49.3 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~1164 mins (~19.4 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dryousufmozumder` | Standby | $30.00 | $0.00 | **$30.00** | ~923m | ~363m | 🟢 Healthy | [View Ledger](https://modal.com/settings/dryousufmozumder/billing) |
| `hasinishrak2015` | Standby | $30.00 | $0.00 | **$30.00** | ~923m | ~363m | 🟢 Healthy | [View Ledger](https://modal.com/settings/hasinishrak2015/billing) |
| `hasinishrak74001` | Standby | $1.00 | $0.89 | **$0.11** | ~3m | ~1m | 🟡 Low | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.95 | **$0.05** | ~1m | ~0m | 🟡 Low | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | 🔥 **ACTIVE** | $30.00 | $20.76 | **$9.24** | ~284m | ~112m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | Standby | $30.00 | $3.17 | **$26.83** | ~825m | ~325m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

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
- **Status**: 🟡 Low
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$0.89`
- **Remaining Balance**: **`$0.11`**
- **Est. L40S GPU Runtime**: **~3 minutes** (0h 3m)
- **Est. H100 GPU Runtime**: **~1 minutes** (0h 1m)
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
- **Metered Usage**: `$20.76`
- **Remaining Balance**: **`$9.24`**
- **Est. L40S GPU Runtime**: **~284 minutes** (4h 44m)
- **Est. H100 GPU Runtime**: **~112 minutes** (1h 52m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood693/billing`
### 💳 Profile: `tigerwood697` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$3.17`
- **Remaining Balance**: **`$26.83`**
- **Est. L40S GPU Runtime**: **~825 minutes** (13h 45m)
- **Est. H100 GPU Runtime**: **~325 minutes** (5h 25m)
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
