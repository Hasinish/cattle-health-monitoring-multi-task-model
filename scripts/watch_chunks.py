# -*- coding: utf-8 -*-
"""Clean, Zero-Overlap Live Dashboard for SideView Re-ID + Pose Evaluation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from typing import Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.system("")  # Enable ANSI escape sequences in Windows console

APP_ID = "ap-Zb0le0pHhi8z9ahux6fTJg"
PROFILE = "dryousufmozumder"

# Store latest status string for each chunk: {chunk_id: full_progress_string}
chunk_tracker: Dict[int, str] = {}
chunk_pcts: Dict[int, int] = {}
last_render_time = 0.0

def render_dashboard(force: bool = False):
    global last_render_time
    now = time.time()
    if not force and (now - last_render_time < 0.4):
        return
    last_render_time = now

    # Clear terminal screen cleanly using ANSI
    sys.stdout.write("\033[H\033[J")
    
    print("=" * 82)
    print(f"  SIDEVIEW RE-ID + POSE EVALUATION MONITOR  [{PROFILE} :: {APP_ID}]")
    print("=" * 82)
    
    if not chunk_tracker:
        print("  Connecting to Modal log stream... Waiting for chunk progress...")
    else:
        # Display each tracked chunk in numerical order
        for cid in sorted(chunk_tracker.keys()):
            stat = chunk_tracker[cid]
            # Highlight completed chunks
            if "100%" in stat:
                print(f"  [DONE] {stat}")
            else:
                print(f"  [RUN ] {stat}")

    completed_count = sum(1 for p in chunk_pcts.values() if p >= 100)
    active_count = len(chunk_tracker) - completed_count

    print("-" * 82)
    print(f"  Active Chunks: {active_count} | Finished Chunks: {completed_count}/42 | Total Chunks Dispatched: {len(chunk_tracker)}/42")
    print("=" * 82)
    sys.stdout.flush()

env = dict(os.environ, PYTHONIOENCODING="utf-8")
cmd = ["modal", "app", "logs", APP_ID, "--profile", PROFILE]

try:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    for raw_line in iter(proc.stdout.readline, ""):
        line = raw_line.strip()
        if not line:
            continue

        # Look for chunk progress lines
        # Example: "Chunk 01/42:  47%|████▋     | 704/1500 [04:04<04:53,  2.71it/s]"
        match = re.search(r"Chunk\s+(\d+)/(\d+):\s*(.*)", line)
        if match:
            c_idx = int(match.group(1))
            total_c = int(match.group(2))
            details = match.group(3).strip()

            pct_match = re.search(r"(\d+)%", details)
            pct_val = int(pct_match.group(1)) if pct_match else 0
            chunk_pcts[c_idx] = pct_val

            clean_entry = f"Chunk {c_idx:02d}/{total_c:02d}: {details}"
            chunk_tracker[c_idx] = clean_entry
            render_dashboard()

        # Check for final retrieval milestone
        elif any(k in line for k in ["PROTOCOL A RETRIEVAL", "Snapshots -> Parlor", "Barn -> Parlor", "Rank-1:"]):
            sys.stdout.write(f"\n[MILESTONE] {line}\n")
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\n[Stopped monitoring]")
