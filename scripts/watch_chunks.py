# -*- coding: utf-8 -*-
"""Rock-Solid, Live-Ticking, Zero-Glitch Live Monitor using Rich & Background Thread."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Set

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

APP_ID = sys.argv[1] if len(sys.argv) > 1 else "ap-5z7WOwmTfEGYZGVVqBvwQ2"
PROFILE = "dryousufmozumder"
TOTAL_CHUNKS = 42
TOTAL_IMAGES = 61678

console = Console(force_terminal=True)

def parse_time_str(s: str) -> int:
    try:
        parts = [int(p) for p in s.strip().split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 1:
            return parts[0]
    except Exception:
        pass
    return 0

def format_seconds(sec: int) -> str:
    if sec < 0:
        sec = 0
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

class ChunkInfo:
    def __init__(self, cid: int):
        self.cid = cid
        self.pct: int = 0
        self.done: int = 0
        self.total: int = 1500
        self.base_elapsed_sec: int = 0
        self.base_eta_sec: int = 0
        self.rate_str: str = "0.0 it/s"
        self.rate_float: float = 2.7
        self.last_packet_time: float = time.time()

    def update_from_packet(
        self,
        pct: int,
        done: int,
        tot: int,
        elapsed_str: str,
        eta_str: str,
        rate_str: str,
    ):
        self.pct = pct
        self.done = done
        self.total = tot
        self.base_elapsed_sec = parse_time_str(elapsed_str)
        self.base_eta_sec = parse_time_str(eta_str)
        self.rate_str = rate_str
        m_rate = re.search(r"([\d\.]+)", rate_str)
        self.rate_float = float(m_rate.group(1)) if m_rate else 2.7
        self.last_packet_time = time.time()

    def current_state(self, now: float) -> Dict[str, Any]:
        dt = max(0.0, now - self.last_packet_time)
        cur_done = min(self.total, int(self.done + dt * self.rate_float))
        cur_pct = int((cur_done / self.total) * 100) if self.total else 0
        cur_elapsed_sec = int(self.base_elapsed_sec + dt)
        cur_eta_sec = max(0, int(self.base_eta_sec - dt))
        return {
            "pct": cur_pct,
            "done": cur_done,
            "total": self.total,
            "elapsed": format_seconds(cur_elapsed_sec),
            "eta": format_seconds(cur_eta_sec),
            "rate": self.rate_str,
            "rate_float": self.rate_float,
        }

active_chunks: Dict[int, ChunkInfo] = {}
completed_chunks: Set[int] = set()
milestones: List[str] = []

def parse_line(line: str):
    line = line.strip()
    if not line:
        return

    m = re.search(
        r"Chunk\s+(\d+)/(\d+):\s*(\d+)%.*?(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([^\]]+)\]",
        line,
    )
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
            active_chunks[c_idx].update_from_packet(
                pct=pct,
                done=done,
                tot=tot,
                elapsed_str=elapsed,
                eta_str=eta,
                rate_str=rate,
            )
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
    if any(
        k in line
        for k in [
            "PROTOCOL A RETRIEVAL",
            "Snapshots -> Parlor",
            "Barn -> Parlor",
            "Rank-1:",
            "RESULTS SAVED",
        ]
    ):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if clean not in milestones:
            milestones.append(clean)

def make_layout() -> Group:
    now = time.time()
    n_done = len(completed_chunks)
    n_active = len(active_chunks)
    overall_pct = (n_done / TOTAL_CHUNKS) * 100.0

    # Collect live interpolated states for all active chunks
    active_states = {cid: active_chunks[cid].current_state(now) for cid in sorted(active_chunks.keys())}

    total_done_imgs = (n_done * 1500) + sum(s["done"] for s in active_states.values())
    total_remaining_imgs = max(0, TOTAL_IMAGES - total_done_imgs)
    fleet_speed = sum(s["rate_float"] for s in active_states.values())
    overall_eta_sec = int(total_remaining_imgs / max(1.0, fleet_speed)) if fleet_speed > 0 else 0
    overall_eta_str = format_seconds(overall_eta_sec)

    # 1. Header / Overall Progress
    overall_pbar = ProgressBar(total=TOTAL_CHUNKS, completed=n_done, width=38)
    done_list_str = ", ".join(f"{c:02d}" for c in sorted(completed_chunks)) if completed_chunks else "None yet"

    summary_text = Text()
    summary_text.append(f"Modal App:   ", style="bold")
    summary_text.append(f"{APP_ID} ({PROFILE})\n", style="cyan")
    summary_text.append(f"Fleet State: ", style="bold")
    summary_text.append(
        f"{n_done}/{TOTAL_CHUNKS} Finished ({overall_pct:.1f}%)  |  {n_active} Active T4s  |  Fleet: {fleet_speed:.1f} it/s\n",
        style="yellow",
    )
    summary_text.append(f"Overall ETA: ", style="bold")
    summary_text.append(f"~{overall_eta_str} remaining  ", style="bold green")
    summary_text.append(f"({total_done_imgs:,}/{TOTAL_IMAGES:,} images)\n", style="dim")
    summary_text.append(f"Done IDs:    ", style="bold")
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
    table.add_column("Speed", justify="center", width=11)
    table.add_column("Elapsed", justify="center", width=10)
    table.add_column("Live ETA", justify="center", width=10)

    if not active_states:
        if n_done == TOTAL_CHUNKS:
            table.add_row("", "[bold green]All 42 chunks complete! Finalizing retrieval...[/bold green]", "", "", "", "")
        else:
            table.add_row("", "[dim]Connecting or waiting for next chunk...[/dim]", "", "", "", "")
    else:
        for cid, state in active_states.items():
            pbar = ProgressBar(total=state["total"], completed=state["done"], width=24)
            table.add_row(
                f"[cyan]Chunk {cid:02d}/{TOTAL_CHUNKS:02d}[/cyan]",
                pbar,
                f"{state['done']}/{state['total']} ({state['pct']}%)",
                f"[green]{state['rate']}[/green]",
                f"[white]{state['elapsed']}[/white]",
                f"[bold yellow]{state['eta']}[/bold yellow]",
            )

    active_panel = Panel(
        table,
        title="[bold green]ACTIVE T4 WORKERS (REAL-TIME LIVE COUNTDOWN)[/bold green]",
        border_style="green",
        padding=(0, 1),
    )

    items = [header_panel, active_panel]

    # 3. Milestones panel if any exist
    if milestones:
        m_text = Text("\n".join(milestones[-4:]), style="bold yellow")
        m_panel = Panel(
            m_text,
            title="[bold gold1]RETRIEVAL METRICS / MILESTONES[/bold gold1]",
            border_style="gold1",
        )
        items.append(m_panel)

    return Group(*items)

def log_reader_thread(proc: subprocess.Popen, line_queue: queue.Queue):
    fd = proc.stdout.fileno()
    buf = b""
    while proc.poll() is None:
        try:
            raw = os.read(fd, 2048)
            if not raw:
                break
            buf += raw
            parts = re.split(b"[\r\n]+", buf)
            buf = parts[-1]
            for part in parts[:-1]:
                decoded = part.decode("utf-8", errors="replace")
                line_queue.put(decoded)
        except Exception:
            break

def main():
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"

    # 1. Warm start from recent history
    try:
        init_res = subprocess.run(
            ["modal", "app", "logs", APP_ID, "--tail", "1500", "--profile", PROFILE],
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

    # 2. Start streaming subprocess & non-blocking background thread
    cmd = ["modal", "app", "logs", APP_ID, "-f", "--profile", PROFILE]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sub_env,
    )

    line_queue: queue.Queue = queue.Queue()
    t = threading.Thread(target=log_reader_thread, args=(proc, line_queue), daemon=True)
    t.start()

    # 3. Fast UI Tick Loop (2 updates per second for smooth countdown clock)
    with Live(make_layout(), console=console, refresh_per_second=2, screen=False) as live:
        try:
            while proc.poll() is None or not line_queue.empty():
                # Drain incoming log lines without blocking
                while not line_queue.empty():
                    try:
                        line = line_queue.get_nowait()
                        parse_line(line)
                    except queue.Empty:
                        break

                live.update(make_layout())
                time.sleep(0.5)

        except KeyboardInterrupt:
            pass

    console.print("\n[bold green]Evaluation monitoring session ended.[/bold green]")

if __name__ == "__main__":
    main()
