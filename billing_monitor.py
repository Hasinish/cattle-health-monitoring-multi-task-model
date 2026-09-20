# -*- coding: utf-8 -*-
"""
Modal Billing Monitor - Root Workspace Entry Point
Run:
    python billing_monitor.py          # Instant summary & update BILLING.md
    python billing_monitor.py --loop   # Live auto-refreshing background monitor
"""
import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    script_path = os.path.join(ROOT_DIR, "scripts", "billing_monitor.py")
    cmd = [sys.executable, script_path] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
