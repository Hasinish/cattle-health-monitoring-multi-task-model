# -*- coding: utf-8 -*-
"""
Live Auto-Looping Modal Billing Monitor
Run:
    python monitor.py         # Autoloops every 10s until Ctrl+C
    python monitor.py 5       # Autoloops every 5s until Ctrl+C
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

from billing_monitor import run_monitor, update_dashboard, print_terminal_summary

if __name__ == "__main__":
    interval = 10
    for arg in sys.argv[1:]:
        if arg.isdigit():
            interval = int(arg)

    # Print initial full scorecard snapshot
    print("🚀 Initializing live Modal billing monitor...\n")
    data = update_dashboard()
    print_terminal_summary(data)

    # Start live loop until Ctrl+C
    try:
        run_monitor(interval_seconds=interval)
    except KeyboardInterrupt:
        print("\n🛑 Billing monitor stopped (Ctrl+C).")
