# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-09-21 14:00:53`  
> **Active Target Account:** `hasinishrak74001`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **6** (`dryousufmozumder, hasinishrak2015, hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$122.00** |
| 💸 **Total Consumed** | **$16.13** |
| 🟢 **Total Remaining Balance** | **$105.87** |
| 🚀 **Total Combined L40S Runtime** | **~3255 mins (~54.2 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~1280 mins (~21.3 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dryousufmozumder` | Standby | $30.00 | $0.00 | **$30.00** | ~923m | ~363m | 🟢 Healthy | [View Ledger](https://modal.com/settings/dryousufmozumder/billing) |
| `hasinishrak2015` | Standby | $30.00 | $0.00 | **$30.00** | ~923m | ~363m | 🟢 Healthy | [View Ledger](https://modal.com/settings/hasinishrak2015/billing) |
| `hasinishrak74001` | 🔥 **ACTIVE** | $1.00 | $0.42 | **$0.58** | ~17m | ~7m | 🟡 Low | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.95 | **$0.05** | ~1m | ~0m | 🟡 Low | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | Standby | $30.00 | $14.00 | **$16.00** | ~492m | ~193m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | Standby | $30.00 | $0.76 | **$29.24** | ~899m | ~354m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

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
### 💳 Profile: `hasinishrak74001` (CURRENT ACTIVE)
- **Status**: 🟡 Low
- **Grant Ceiling**: `$1.00`
- **Metered Usage**: `$0.42`
- **Remaining Balance**: **`$0.58`**
- **Est. L40S GPU Runtime**: **~17 minutes** (0h 17m)
- **Est. H100 GPU Runtime**: **~7 minutes** (0h 7m)
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
- **Metered Usage**: `$14.00`
- **Remaining Balance**: **`$16.00`**
- **Est. L40S GPU Runtime**: **~492 minutes** (8h 12m)
- **Est. H100 GPU Runtime**: **~193 minutes** (3h 13m)
- **Out-of-Pocket Billed**: `$0.00`
- **Direct Modal Link**: `https://modal.com/settings/tigerwood693/billing`
### 💳 Profile: `tigerwood697` 
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$0.76`
- **Remaining Balance**: **`$29.24`**
- **Est. L40S GPU Runtime**: **~899 minutes** (14h 59m)
- **Est. H100 GPU Runtime**: **~354 minutes** (5h 54m)
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
