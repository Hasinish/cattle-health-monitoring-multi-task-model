# -*- coding: utf-8 -*-
"""Clean, Zero-Overlap Live Streaming Dashboard for SideView Re-ID + Pose Evaluation."""

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

chunk_tracker: Dict[int, str] = {}
chunk_pcts: Dict[int, int] = {}
last_render_time = 0.0

def render_dashboard(force: bool = False):
    global last_render_time
    now = time.time()
    if not force and (now - last_render_time < 0.25):
        return
    last_render_time = now

    sys.stdout.write("\033[H\033[J")
    
    print("=" * 82)
    print(f"  SIDEVIEW RE-ID + POSE EVALUATION MONITOR  [{PROFILE} :: {APP_ID}]")
    print("=" * 82)
    
    if not chunk_tracker:
        print("  Connecting to Modal log stream... Waiting for chunk progress...")
    else:
        for cid in sorted(chunk_tracker.keys()):
            stat = chunk_tracker[cid]
            if "100%" in stat or chunk_pcts.get(cid, 0) >= 100:
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

def parse_line(raw_line: str):
    line = raw_line.strip()
    if not line:
        return

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

    elif any(k in line for k in ["PROTOCOL A RETRIEVAL", "Snapshots -> Parlor", "Barn -> Parlor", "Rank-1:"]):
        sys.stdout.write(f"\n[MILESTONE] {line}\n")
        sys.stdout.flush()

# Step 1: Pre-populate from recent logs
try:
    init_res = subprocess.run(
        ["modal", "app", "logs", APP_ID, "--tail", "300", "--profile", PROFILE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=10,
    )
    for l in init_res.stdout.splitlines():
        parse_line(l)
except Exception:
    pass

render_dashboard(force=True)

# Step 2: Stream live updates continuously
cmd = ["modal", "app", "logs", APP_ID, "-f", "--profile", PROFILE]

while True:
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
            parse_line(raw_line)

        proc.wait()
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[Stopped monitoring]")
        break
    except Exception:
        time.sleep(2.0)
