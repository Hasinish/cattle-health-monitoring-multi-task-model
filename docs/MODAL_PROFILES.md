# 💳 Modal Profiles & Multi-Account Cheatsheet

Below are all 6 Modal profiles configured on your machine, matching `D:\custom-antigravity\credentials\modal.toml` and `~/.modal.toml`.

---

## 📋 Accounts Overview & Roles

| # | Profile Name | Account Type | Monthly Grant | Role in This Workspace |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **tigerwood693** | Card Verified | $30.00/mo | Active (`moo-data` download) |
| 2 | **tigerwood697** | Card Verified | $30.00/mo | Active (`mmcows-data` download) |
| 3 | **hasinishrak74001** | Trial | $1.00 | Ready (`moo-data` already extracted) |
| 4 | **dryousufmozumder** | Card Verified | $30.00/mo | Standby |
| 5 | **hasinishrak2015** | Card Verified | $30.00/mo | Standby |
| 6 | **mohtasimahmedsamii** | Trial | $1.00 | Standby |

---

## ⚡ Quick Commands

### Activate any profile:
```powershell
modal profile activate <PROFILE_NAME>
```

### Check live balances across all accounts:
```powershell
python monitor.py
```

### Auto-Loaded Tokens:
Tokens are now stored in `.env` (git-ignored) and automatically read by pipeline scripts:
- `HF_TOKEN` (Hugging Face CDN throttle bypass)
- `CIVITAI_TOKEN`
