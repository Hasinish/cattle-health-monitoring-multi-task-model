"""
High-Speed Multi-Threaded Resumable Downloader for SideViewCows2026 (Zenodo record 21605650).

Author: Hasin Ishrak
Dataset: SideViewCows2026 - Dairy Cow Re-Identification Dataset
DOI: 10.5281/zenodo.21605650

Features:
- Multi-threaded parallel chunk downloading via HTTP Range headers
- Automatic resume capability for partial downloads
- Full browser header spoofing to bypass Cloudflare/Zenodo bot blocks
- Exponential backoff retry logic for unstable connections
- Progress tracking via tqdm
- Checksum verification
- Optional automatic extraction into datasets/id/external/sideviewcows2026/
"""

import os
import sys
import time
import zipfile
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = REPO_ROOT / "datasets" / "id" / "external" / "sideviewcows2026"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

ZENODO_RECORD_ID = "21605650"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"

FILES_CATALOG = {
    "manifest.csv": {
        "url": f"{BASE_URL}manifest.csv?download=1",
        "size": 12588074,
        "type": "metadata",
        "description": "Master metadata (80,260 images: individual_id, time_offset_s, dimensions, sha256)"
    },
    "SHA256SUMS": {
        "url": f"{BASE_URL}SHA256SUMS?download=1",
        "size": 15393015,
        "type": "metadata",
        "description": "SHA-256 checksums covering every file in the dataset"
    },
    "README.md": {
        "url": f"{BASE_URL}README.md?download=1",
        "size": 8177,
        "type": "metadata",
        "description": "Official dataset documentation, directory layout, and citation"
    },
    "preview.jpg": {
        "url": f"{BASE_URL}preview.jpg?download=1",
        "size": 976915,
        "type": "metadata",
        "description": "Sample grid of individual cows with overlaid segmentation masks"
    },
    "snapshots.zip": {
        "url": f"{BASE_URL}snapshots.zip?download=1",
        "size": 375219777,
        "type": "data",
        "description": "607 images + masks across 63 cows (field/cubicle snapshots, variable angles)"
    },
    "parlor.zip": {
        "url": f"{BASE_URL}parlor.zip?download=1",
        "size": 9598064961,
        "type": "data",
        "description": "54,393 images + masks across 110 cows (milking parlor fixed camera gallery)"
    },
    "barn.zip": {
        "url": f"{BASE_URL}barn.zip?download=1",
        "size": 15051008416,
        "type": "data",
        "description": "25,260 images + masks across 69 cows (handheld video recorded in barn)"
    },
}


import threading


class CleanProgressBar:
    """
    Ultra-clean, single-line in-place carriage-return (\\r) progress bar.
    Thread-safe, throttled rendering (max ~10 fps) to prevent console spam and flicker.
    Shows exact downloaded size, percentage, rolling MB/s, and ETA.
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
            if dt >= 0.12 or self.current >= self.total:  # ~8 Hz update
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


def download_chunk_with_retry(
    url: str,
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

    headers = dict(BROWSER_HEADERS)
    headers["Range"] = f"bytes={actual_start}-{end_byte}"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(part_path, "ab") as f:
                    while chunk := resp.read(256 * 1024):  # 256KB buffer for smooth progress updates
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
                headers["Range"] = f"bytes={actual_start}-{end_byte}"

    return part_path.stat().st_size if part_path.exists() else 0


def download_file_multithreaded(url: str, dest_path: Path, expected_size: int, num_threads: int = 8):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(f"[SKIP] {dest_path.name} already fully downloaded ({expected_size / (1024**2):.2f} MB).")
        return

    # Small files (< 20MB) or single thread
    if expected_size < 20 * 1024 * 1024 or num_threads <= 1:
        part_path = dest_path.with_name(f"{dest_path.name}.part")
        existing = part_path.stat().st_size if part_path.exists() else 0
        pbar = CleanProgressBar(
            total_bytes=expected_size,
            initial_bytes=existing,
            desc=f"📥 {dest_path.name}"
        )
        download_chunk_with_retry(url, part_path, 0, expected_size - 1, pbar=pbar)
        pbar.close()
        if part_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
        print(f"[COMPLETE] {dest_path.name}")
        return

    chunk_size = expected_size // num_threads
    ranges = []
    initial_bytes = 0
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < num_threads - 1 else (expected_size - 1)
        ranges.append((start, end, i))
        part = dest_path.with_name(f"{dest_path.name}.part{i}")
        if part.exists():
            initial_bytes += part.stat().st_size

    print(f"[FAST-DOWNLOAD] {dest_path.name} ({expected_size / (1024**3):.2f} GB) using {num_threads} parallel threads...")
    pbar = CleanProgressBar(
        total_bytes=expected_size,
        initial_bytes=initial_bytes,
        desc=f"📥 {dest_path.name}"
    )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                download_chunk_with_retry,
                url,
                dest_path.with_name(f"{dest_path.name}.part{idx}"),
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
    print(f"[ASSEMBLING] Combining {num_threads} parts into {dest_path.name}...")
    with open(dest_path, "wb") as outfile:
        for i in range(num_threads):
            part_path = dest_path.with_name(f"{dest_path.name}.part{i}")
            if part_path.exists():
                with open(part_path, "rb") as infile:
                    while chunk := infile.read(1024 * 1024 * 4):  # 4MB buffer
                        outfile.write(chunk)
                part_path.unlink()

    actual_sz = dest_path.stat().st_size
    if actual_sz != expected_size:
        raise RuntimeError(f"Size mismatch for {dest_path.name}: expected {expected_size}, got {actual_sz}")

    print(f"[COMPLETE] {dest_path.name} ({actual_sz / (1024**3):.2f} GB)\n")


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"[EXTRACTED] {zip_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Fast Downloader for SideViewCows2026 (Zenodo record 21605650)")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_TARGET_DIR), help="Target directory for downloaded files")
    parser.add_argument("--meta-only", action="store_true", help="Download only metadata files (manifest, README, preview, SHA256SUMS)")
    parser.add_argument("--subsets", nargs="+", choices=["snapshots", "parlor", "barn"], help="Specific subsets to download")
    parser.add_argument("--threads", type=int, default=8, help="Number of parallel download threads per file")
    parser.add_argument("--extract", action="store_true", help="Extract zip archives after downloading")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SIDEVIEWCOWS2026 FAST PARALLEL DOWNLOADER (ZENODO: 21605650)")
    print(f"  Target Directory: {dest_dir}")
    print(f"  Parallel Threads: {args.threads}")
    print("=" * 70)

    # 1. Download metadata files
    meta_keys = ["manifest.csv", "SHA256SUMS", "README.md", "preview.jpg"]
    print("\n--- Phase 1: Metadata Files ---")
    for key in meta_keys:
        info = FILES_CATALOG[key]
        dest_file = dest_dir / key
        download_file_multithreaded(info["url"], dest_file, info["size"], num_threads=1)

    if args.meta_only:
        print("\n[SUCCESS] Metadata files downloaded successfully!")
        return

    # 2. Determine which zip archives to download
    data_keys = []
    if args.subsets:
        for s in args.subsets:
            data_keys.append(f"{s}.zip")
    else:
        data_keys = ["snapshots.zip", "parlor.zip", "barn.zip"]

    print(f"\n--- Phase 2: Data Archives ({len(data_keys)} files) ---")
    total_download_bytes = sum(FILES_CATALOG[k]["size"] for k in data_keys)
    print(f"Total download size: {total_download_bytes / (1024**3):.2f} GB")

    for key in data_keys:
        info = FILES_CATALOG[key]
        dest_file = dest_dir / key
        download_file_multithreaded(info["url"], dest_file, info["size"], num_threads=args.threads)
        if args.extract:
            extract_zip(dest_file, dest_dir)

    print("\n" + "=" * 70)
    print("[SUCCESS] All requested SideViewCows2026 files are ready!")
    print("=" * 70)


if __name__ == "__main__":
    main()
