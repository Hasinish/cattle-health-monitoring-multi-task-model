"""
High-Speed Multi-Threaded Resumable Downloader for BECA Dataset (Figshare file 63928608).

Dataset: BECA: A Computer Vision Dataset for Long-Term Recognition In Beef Cattle
Article: https://doi.org/10.6084/m9.figshare.32070171
File: BECA.zip (20,309,712,996 bytes / ~18.91 GB)
Sub-datasets included:
  - BECA-D: 16,889 images across 5,661 beef cattle (diversity benchmark)
  - BECA-L: 12,172 images across 103 beef cattle tracked over 5 months (long-term benchmark)

Features:
- Multi-threaded parallel chunk downloading via HTTP Range headers
- Automatic resume capability for partial downloads (.part files)
- Uses ndownloader.figshare.com endpoint with automatic S3 presigned URL renewal
- Single-line carriage-return (\\r) live progress bar with speed (MB/s) and ETA
- Optional automatic extraction into datasets/id/external/beca/
"""

import os
import sys
import time
import zipfile
import argparse
import threading
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = REPO_ROOT / "datasets" / "id" / "external" / "beca"

DOWNLOAD_ENDPOINT = "https://ndownloader.figshare.com/files/63928608"
EXPECTED_SIZE = 20309712996  # 18.91 GB
FILENAME = "BECA.zip"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


class CleanProgressBar:
    """
    Ultra-clean, single-line in-place carriage-return (\\r) progress bar.
    Thread-safe, throttled rendering (~8 fps) with rolling MB/s and ETA.
    """
    def __init__(self, total_bytes: int, initial_bytes: int = 0, desc: str = "", bar_len: int = 25):
        self.total = total_bytes
        self.current = initial_bytes
        self.desc = desc
        self.bar_len = bar_len
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_render_time = 0.0
        self.last_bytes = initial_bytes
        self.speed = 0.0
        self._render(force=True)

    def update(self, n_bytes: int):
        with self.lock:
            self.current += n_bytes
            now = time.time()
            dt = now - self.last_render_time
            if dt >= 0.12 or self.current >= self.total:
                if dt > 0:
                    self.speed = (self.current - self.last_bytes) / dt
                    self.last_render_time = now
                    self.last_bytes = self.current
                self._render(force=False)

    def write(self, msg: str):
        with self.lock:
            sys.stdout.write(f"\r{' ' * 95}\r{msg}\n")
            sys.stdout.flush()
            self._render(force=True)

    def _render(self, force: bool = False):
        pct = (self.current / self.total * 100) if self.total > 0 else 100.0
        filled = int(self.bar_len * self.current / self.total) if self.total > 0 else self.bar_len
        filled = min(self.bar_len, max(0, filled))
        arrow = ">" if (filled < self.bar_len and self.current < self.total) else "="
        bar = "=" * max(0, filled - (1 if arrow == ">" else 0)) + (arrow if filled > 0 else "")
        bar = bar.ljust(self.bar_len)

        def fmt_size(b: int) -> str:
            if b >= 1024**3:
                return f"{b / (1024**3):.2f} GB"
            if b >= 1024**2:
                return f"{b / (1024**2):.1f} MB"
            if b >= 1024:
                return f"{b / 1024:.1f} KB"
            return f"{b} B"

        curr_str = fmt_size(self.current)
        total_str = fmt_size(self.total)
        speed_str = f"{self.speed / (1024**2):.2f} MB/s" if self.speed > 0 else "-- MB/s"

        rem_bytes = max(0, self.total - self.current)
        if self.speed > 0 and rem_bytes > 0:
            eta_sec = int(rem_bytes / self.speed)
            m, s = divmod(eta_sec, 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
        else:
            eta_str = "--:--"

        line = f"\r{self.desc} [{bar}] {pct:5.1f}% | {curr_str} / {total_str} | {speed_str} | ETA: {eta_str}"
        sys.stdout.write(line.ljust(95))
        sys.stdout.flush()

    def close(self):
        with self.lock:
            self._render(force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()


def get_s3_url() -> str:
    """Resolve fresh presigned Amazon S3 URL from Figshare ndownloader endpoint."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = opener.open(urllib.request.Request(DOWNLOAD_ENDPOINT, headers=headers))
        return resp.geturl()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers.get("Location")
        raise


def download_chunk_with_retry(
    endpoint: str,
    part_path: Path,
    start_byte: int,
    end_byte: int,
    pbar: CleanProgressBar = None,
    max_retries: int = 5
) -> int:
    existing_size = part_path.stat().st_size if part_path.exists() else 0
    actual_start = start_byte + existing_size

    if actual_start > end_byte:
        return existing_size

    for attempt in range(1, max_retries + 1):
        try:
            s3_url = get_s3_url()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Range": f"bytes={actual_start}-{end_byte}",
            }
            req = urllib.request.Request(s3_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")
                with open(part_path, "ab") as f:
                    while chunk := resp.read(256 * 1024):  # 256KB buffer
                        f.write(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))
            return part_path.stat().st_size
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed chunk {actual_start}-{end_byte} after {max_retries} attempts: {e}")
            wait_time = attempt * 2
            if pbar:
                pbar.write(f"[RETRY] Chunk {actual_start}-{end_byte}: attempt {attempt} failed ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            if part_path.exists():
                actual_start = start_byte + part_path.stat().st_size

    return part_path.stat().st_size if part_path.exists() else 0


def download_beca(dest_dir: Path, num_threads: int = 8, extract: bool = False):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / FILENAME

    if dest_file.exists() and dest_file.stat().st_size == EXPECTED_SIZE:
        print(f"[SKIP] {FILENAME} already fully downloaded ({EXPECTED_SIZE / (1024**3):.2f} GB).")
        if extract:
            extract_zip(dest_file, dest_dir)
        return

    chunk_size = EXPECTED_SIZE // num_threads
    ranges = []
    initial_bytes = 0
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < num_threads - 1 else (EXPECTED_SIZE - 1)
        ranges.append((start, end, i))
        part = dest_dir / f"{FILENAME}.part{i}"
        if part.exists():
            initial_bytes += part.stat().st_size

    print("=" * 70)
    print("  BECA DATASET FAST PARALLEL DOWNLOADER (FIGSHARE: 63928608)")
    print(f"  Target File: {dest_file}")
    print(f"  Total Size:  {EXPECTED_SIZE / (1024**3):.2f} GB ({EXPECTED_SIZE:,} bytes)")
    print(f"  Parallel Threads: {num_threads}")
    print("=" * 70)

    pbar = CleanProgressBar(
        total_bytes=EXPECTED_SIZE,
        initial_bytes=initial_bytes,
        desc=f"📥 {FILENAME}"
    )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                download_chunk_with_retry,
                DOWNLOAD_ENDPOINT,
                dest_dir / f"{FILENAME}.part{idx}",
                start,
                end,
                pbar
            ): idx
            for start, end, idx in ranges
        }
        for future in as_completed(futures):
            future.result()

    pbar.close()

    # Assemble parts
    print(f"[ASSEMBLING] Combining {num_threads} parts into {FILENAME}...")
    with open(dest_file, "wb") as outfile:
        for i in range(num_threads):
            part_path = dest_dir / f"{FILENAME}.part{i}"
            if part_path.exists():
                with open(part_path, "rb") as infile:
                    while chunk := infile.read(1024 * 1024 * 8):  # 8MB buffer
                        outfile.write(chunk)
                part_path.unlink()

    actual_sz = dest_file.stat().st_size
    if actual_sz != EXPECTED_SIZE:
        raise RuntimeError(f"Size mismatch: expected {EXPECTED_SIZE}, got {actual_sz}")

    print(f"[COMPLETE] {FILENAME} ({actual_sz / (1024**3):.2f} GB) successfully assembled!\n")

    if extract:
        extract_zip(dest_file, dest_dir)


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"[EXTRACTED] {zip_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Fast Downloader for BECA Dataset (Figshare file 63928608)")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_TARGET_DIR), help="Target directory")
    parser.add_argument("--threads", type=int, default=8, help="Number of parallel download threads")
    parser.add_argument("--extract", action="store_true", help="Extract zip archive after downloading")
    args = parser.parse_args()

    download_beca(Path(args.dest), num_threads=args.threads, extract=args.extract)


if __name__ == "__main__":
    main()
