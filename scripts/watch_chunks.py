# -*- coding: utf-8 -*-
"""Rock-Solid, Zero-Glitch, Flicker-Free Live Monitor using Rich."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Set

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.system("")

APP_ID = "ap-Zb0le0pHhi8z9ahux6fTJg"
PROFILE = "dryousufmozumder"
TOTAL_CHUNKS = 42

console = Console()

class ChunkInfo:
    def __init__(self, cid: int):
        self.cid = cid
        self.pct: int = 0
        self.done: int = 0
        self.total: int = 1500
        self.elapsed: str = "00:00"
        self.eta: str = "--:--"
        self.rate: str = "0.0 it/s"
        self.last_update = time.time()

active_chunks: Dict[int, ChunkInfo] = {}
completed_chunks: Set[int] = set()
milestones: List[str] = []

def parse_line(line: str):
    line = line.strip()
    if not line:
        return

    m = re.search(r"Chunk\s+(\d+)/(\d+):\s*(\d+)%.*?(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([^\]]+)\]", line)
    if m:
        c_idx = int(m.group(1))
        pct = int(m.group(3))
        done = int(m.group(4))
        tot = int(m.group(5))
        elapsed = m.group(6).strip()
        eta = m.group(7).strip()
        rate = m.group(8).strip()

        if pct >= 100 or done >= tot:
            completed_chunks.add(c_idx)
            if c_idx in active_chunks:
                del active_chunks[c_idx]
        else:
            if c_idx not in active_chunks:
                active_chunks[c_idx] = ChunkInfo(c_idx)
            info = active_chunks[c_idx]
            info.pct = pct
            info.done = done
            info.total = tot
            info.elapsed = elapsed
            info.eta = eta
            info.rate = rate
            info.last_update = time.time()
        return

    # Check for simple 100% completion notice
    m_done = re.search(r"Chunk\s+(\d+)/(\d+):\s*100%", line)
    if m_done:
        c_idx = int(m_done.group(1))
        completed_chunks.add(c_idx)
        if c_idx in active_chunks:
            del active_chunks[c_idx]
        return

    # Milestone checks
    if any(k in line for k in ["PROTOCOL A RETRIEVAL", "Snapshots -> Parlor", "Barn -> Parlor", "Rank-1:", "RESULTS SAVED"]):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if clean not in milestones:
            milestones.append(clean)

def make_layout() -> Group:
    n_done = len(completed_chunks)
    n_active = len(active_chunks)
    overall_pct = (n_done / TOTAL_CHUNKS) * 100.0

    # 1. Header / Overall Progress
    overall_pbar = ProgressBar(total=TOTAL_CHUNKS, completed=n_done, width=40)
    done_list_str = ", ".join(f"{c:02d}" for c in sorted(completed_chunks)) if completed_chunks else "None yet"

    summary_text = Text()
    summary_text.append(f"Modal App: ", style="bold")
    summary_text.append(f"{APP_ID} ({PROFILE})\n", style="cyan")
    summary_text.append(f"Progress:  ", style="bold")
    summary_text.append(f"{n_done}/{TOTAL_CHUNKS} Finished ({overall_pct:.1f}%)  |  {n_active} Active T4 GPUs  |  {TOTAL_CHUNKS - n_done - n_active} Queued\n", style="yellow")
    summary_text.append(f"Done IDs:  ", style="bold")
    summary_text.append(f"[{done_list_str}]", style="green")

    header_panel = Panel(
        Group(overall_pbar, summary_text),
        title="[bold cyan]SIDEVIEW RE-ID + POSE RETRIEVAL EVALUATION[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )

    # 2. Active Workers Table
    table = Table(expand=True, box=None, show_header=True, header_style="bold magenta")
    table.add_column("Worker", justify="center", width=12)
    table.add_column("Progress Bar", justify="left")
    table.add_column("Processed", justify="center", width=14)
    table.add_column("Speed", justify="center", width=12)
    table.add_column("ETA", justify="center", width=10)

    if not active_chunks:
        if n_done == TOTAL_CHUNKS:
            table.add_row("", "[bold green]All 42 chunks complete! Finalizing retrieval...[/bold green]", "", "", "")
        else:
            table.add_row("", "[dim]Connecting or waiting for next chunk...[/dim]", "", "", "")
    else:
        for cid in sorted(active_chunks.keys()):
            info = active_chunks[cid]
            pbar = ProgressBar(total=info.total, completed=info.done, width=28)
            table.add_row(
                f"[cyan]Chunk {cid:02d}/{TOTAL_CHUNKS:02d}[/cyan]",
                pbar,
                f"{info.done}/{info.total} ({info.pct}%)",
                f"[green]{info.rate}[/green]",
                f"[yellow]{info.eta}[/yellow]",
            )

    active_panel = Panel(
        table,
        title="[bold green]ACTIVE T4 WORKERS (PARALLEL CONTAINER STREAM)[/bold green]",
        border_style="green",
        padding=(0, 1),
    )

    items = [header_panel, active_panel]

    # 3. Milestones panel if any exist
    if milestones:
        m_text = Text("\n".join(milestones[-4:]), style="bold yellow")
        m_panel = Panel(m_text, title="[bold gold1]RETRIEVAL METRICS / MILESTONES[/bold gold1]", border_style="gold1")
        items.append(m_panel)

    return Group(*items)

def main():
    # 1. Warm start from recent history
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"

    try:
        init_res = subprocess.run(
            ["modal", "app", "logs", APP_ID, "--tail", "600", "--profile", PROFILE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=sub_env,
            timeout=10,
        )
        for line in init_res.stdout.splitlines():
            parse_line(line)
    except Exception:
        pass

    # 2. Live refresh loop
    cmd = ["modal", "app", "logs", APP_ID, "-f", "--profile", PROFILE]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sub_env,
    )
    fd = proc.stdout.fileno()
    buf = b""

    with Live(make_layout(), console=console, refresh_per_second=3, screen=False) as live:
        try:
            while True:
                raw = os.read(fd, 2048)
                if not raw:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue

                buf += raw
                parts = re.split(b"[\r\n]+", buf)
                buf = parts[-1]
                updated = False
                for part in parts[:-1]:
                    decoded = part.decode("utf-8", errors="replace")
                    parse_line(decoded)
                    updated = True

                if updated:
                    live.update(make_layout())

        except KeyboardInterrupt:
            pass

    console.print("\n[bold green]Evaluation monitoring session ended.[/bold green]")

if __name__ == "__main__":
    main()
