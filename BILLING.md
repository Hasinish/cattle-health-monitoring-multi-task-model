# 📊 Live Modal Billing & Credit Dashboard

> **Live Auto-Refreshed Monitor**  
> **Last Synchronized:** `2026-09-25 07:00:30`  
> **Active Target Account:** `hasinishrak2015`

---

## ⚡ Global Summary Across All Accounts

| Metric | Total |
| :--- | :--- |
| 👥 **Discovered Accounts** | **6** (`dryousufmozumder, hasinishrak2015, hasinishrak74001, mohtasimahmedsamii, tigerwood693, tigerwood697`) |
| 🎁 **Total Credit Grants** | **$122.00** |
| 💸 **Total Consumed** | **$95.13** |
| 🟢 **Total Remaining Balance** | **$29.10** |
| 🚀 **Total Combined L40S Runtime** | **~893 mins (~14.9 hours)** |
| ⚡ **Total Combined H100 Runtime** | **~351 mins (~5.8 hours)** |

---

## 📋 Accounts Scoreboard

| Account Profile | Mode | Grant | Consumed | Remaining Balance | Est. L40S | Est. H100 | Status | Direct Ledger |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dryousufmozumder` | Standby | $30.00 | $31.87 | **$0.00** | ~0m | ~0m | 🔴 Depleted | [View Ledger](https://modal.com/settings/dryousufmozumder/billing) |
| `hasinishrak2015` | 🔥 **ACTIVE** | $30.00 | $9.85 | **$20.15** | ~619m | ~244m | 🟢 Healthy | [View Ledger](https://modal.com/settings/hasinishrak2015/billing) |
| `hasinishrak74001` | Standby | $1.00 | $1.36 | **$0.00** | ~0m | ~0m | 🔴 Depleted | [View Ledger](https://modal.com/settings/hasinishrak74001/billing) |
| `mohtasimahmedsamii` | Standby | $1.00 | $0.95 | **$0.05** | ~1m | ~0m | 🟡 Low | [View Ledger](https://modal.com/settings/mohtasimahmedsamii/billing) |
| `tigerwood693` | Standby | $30.00 | $29.15 | **$0.85** | ~26m | ~10m | 🟡 Low | [View Ledger](https://modal.com/settings/tigerwood693/billing) |
| `tigerwood697` | Standby | $30.00 | $21.96 | **$8.04** | ~247m | ~97m | 🟢 Healthy | [View Ledger](https://modal.com/settings/tigerwood697/billing) |

---

## 🔍 Detailed Account Breakdowns

### 💳 Profile: `dryousufmozumder` 
- **Status**: 🔴 Depleted
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$31.87`
- **Remaining Balance**: **`$0.00`**
- **Est. L40S GPU Runtime**: **~0 minutes** (0h 0m)
- **Est. H100 GPU Runtime**: **~0 minutes** (0h 0m)
- **Out-of-Pocket Billed**: `$1.80`
- **Direct Modal Link**: `https://modal.com/settings/dryousufmozumder/billing`
### 💳 Profile: `hasinishrak2015` (CURRENT ACTIVE)
- **Status**: 🟢 Healthy
- **Grant Ceiling**: `$30.00`
- **Metered Usage**: `$9.85`
- **Remaining Balance**: **`$20.15`**
- **Est. L40S GPU Runtime**: **~619 minutes** (10h 19m)
- **Est. H100 GPU Runtime**: **~244 minutes** (4h 4m)
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
- **Metered Usage**: `$21.96`
- **Remaining Balance**: **`$8.04`**
- **Est. L40S GPU Runtime**: **~247 minutes** (4h 7m)
- **Est. H100 GPU Runtime**: **~97 minutes** (1h 37m)
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
