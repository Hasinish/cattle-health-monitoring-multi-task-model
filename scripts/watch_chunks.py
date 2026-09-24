# -*- coding: utf-8 -*-
"""Live chunk progress monitor for SideView Re-ID + Pose Protocol A evaluation."""

import os
import subprocess
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

APP_ID = "ap-Zb0le0pHhi8z9ahux6fTJg"
PROFILE = "dryousufmozumder"

print("\n" + "=" * 76)
print(f"  STREAMING LIVE CHUNK PROGRESS ({PROFILE} :: {APP_ID})")
print("=" * 76 + "\n")

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
    for line in iter(proc.stdout.readline, ""):
        line_clean = line.strip()
        if not line_clean:
            continue
        # Filter for chunk progress or key milestone lines
        if any(keyword in line_clean for keyword in ["Chunk", "PROTOCOL A", "Retrieval", "Rank-", "mAP", "Loaded"]):
            sys.stdout.write(line_clean + "\n")
            sys.stdout.flush()
except KeyboardInterrupt:
    print("\n[Stopped monitoring]")
