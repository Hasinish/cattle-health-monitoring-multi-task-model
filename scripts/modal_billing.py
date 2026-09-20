# -*- coding: utf-8 -*-
"""
Quick Modal Billing Summary Runner
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from billing_monitor import update_dashboard, print_terminal_summary

if __name__ == "__main__":
    data = update_dashboard()
    print_terminal_summary(data)
