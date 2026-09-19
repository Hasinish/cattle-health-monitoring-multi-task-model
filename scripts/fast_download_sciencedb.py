"""
High-Speed Multi-Threaded Chunk Downloader for ScienceDB Cattle BCS Dataset.
Splits dataset.rar into N parallel byte-range streams to bypass single-thread throttling.
Supports automatic resuming, retry backoff, chunk stitching, and extraction.
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
BCS_DIR = DATASETS_DIR / "bcs"
SCIENCEDB_DIR = BCS_DIR / "sciencedb_bcs"
CHUNKS_DIR = SCIENCEDB_DIR / ".chunks"
ARCHIVE_PATH = SCIENCEDB_DIR / "dataset.rar"
EXPECTED_DATASET_DIR = SCIENCEDB_DIR / "dataset"

DEFAULT_URL = (
    "https://china.scidb.cn/download?fileId=4516a6866954512e08446bc3504b32c9&traceId=ae1cb76e-9f3b-48b0-9c51-2750c6266d25"
)

UNRAR_CANDIDATES = [
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files\WinRAR\WinRAR.exe",
    r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "unrar",
    "7z",
    "tar",
]


def find_extractor():
    for candidate in UNRAR_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        elif shutil.which(candidate):
            return candidate
    return None


def get_remote_file_info(url, max_retries=5):
    """Query remote headers to get total file size and verify byte-range support."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    }
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                total_size = resp.headers.get("Content-Length")
                accept_ranges = resp.headers.get("Accept-Ranges", "")
                if total_size:
                    return int(total_size), "bytes" in accept_ranges.lower()
        except Exception:
            # Fallback to GET with 1-byte range if HEAD is blocked
            try:
                test_headers = dict(headers)
                test_headers["Range"] = "bytes=0-0"
                req = urllib.request.Request(url, headers=test_headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    cr = resp.headers.get("Content-Range", "")
                    # Content-Range: bytes 0-0/4407548599
                    if "/" in cr:
                        total_size = cr.split("/")[-1].strip()
                        return int(total_size), True
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to query remote file size after {max_retries} attempts: {e}")
                time.sleep(2 * attempt)
    raise RuntimeError("Unable to determine remote file size.")


def download_chunk(url, chunk_idx, start_byte, end_byte, part_file, pbar, max_retries=10):
    """Download a single chunk with byte-range resume and retry logic."""
    expected_len = end_byte - start_byte + 1

    for attempt in range(1, max_retries + 1):
        existing_len = 0
        if part_file.exists():
            existing_len = part_file.stat().st_size
            if existing_len == expected_len:
                # Already complete
                return True
            elif existing_len > expected_len:
                # Corrupt, re-download chunk
                part_file.unlink()
                existing_len = 0

        current_start = start_byte + existing_len
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Range": f"bytes={current_start}-{end_byte}",
            "Accept": "*/*",
        }

        req = urllib.request.Request(url, headers=headers)
        try:
            mode = "ab" if existing_len > 0 else "wb"
            with urllib.request.urlopen(req, timeout=20) as resp:
                with open(part_file, mode) as f:
                    while True:
                        buffer = resp.read(512 * 1024)  # 512 KB blocks
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))

            # Validate chunk final size
            if part_file.stat().st_size == expected_len:
                return True
            else:
                # Partial read, retry
                time.sleep(1)
        except Exception:
            if attempt >= max_retries:
                raise RuntimeError(f"Chunk {chunk_idx} failed after {max_retries} attempts.")
            time.sleep(2 * attempt)

    return False


def stitch_chunks(part_files, output_path, total_size):
    """Concatenate chunk part files into the final archive."""
    print(f"\n[STITCH] Assembling {len(part_files)} chunks into {output_path.name}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(".tmp_stitch")
    if temp_output.exists():
        temp_output.unlink()

    with open(temp_output, "wb") as outfile:
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Assembling") as pbar:
            for part in part_files:
                with open(part, "rb") as infile:
                    while True:
                        data = infile.read(16 * 1024 * 1024)  # 16 MB buffer
                        if not data:
                            break
                        outfile.write(data)
                        pbar.update(len(data))

    final_size = temp_output.stat().st_size
    if final_size != total_size:
        raise ValueError(f"Stitched archive size mismatch: expected {total_size}, got {final_size}")

    if output_path.exists():
        output_path.unlink()
    temp_output.rename(output_path)
    print(f"[SUCCESS] Stitched archive verified: {output_path} ({final_size / (1024**3):.2f} GB)")

    # Clean up part files
    for part in part_files:
        try:
            part.unlink()
        except OSError:
            pass
    try:
        CHUNKS_DIR.rmdir()
    except OSError:
        pass


def extract_rar(archive_path, extract_dir):
    extractor = find_extractor()
    if not extractor:
        print("[WARNING] No RAR extractor found (WinRAR, 7-Zip, tar).")
        print(f"Archive is intact at: {archive_path}")
        print("Install 7-Zip or WinRAR to extract the images.")
        return False

    print(f"\n[EXTRACT] Extracting {archive_path.name} using {extractor}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if "unrar.exe" in extractor.lower():
        cmd = [extractor, "x", "-y", "-o+", str(archive_path), str(extract_dir)]
    elif "winrar.exe" in extractor.lower():
        cmd = [extractor, "x", "-y", "-ibck", str(archive_path), str(extract_dir)]
    elif "7z" in extractor.lower():
        cmd = [extractor, "x", str(archive_path), f"-o{extract_dir}", "-y"]
    elif "tar" in extractor.lower():
        cmd = [extractor, "-xf", str(archive_path), "-C", str(extract_dir)]
    else:
        cmd = [extractor, "x", str(archive_path), str(extract_dir)]

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Extraction returned non-zero code {res.returncode}")
        return False

    print("[SUCCESS] Extraction complete!")
    return True


def verify_classes():
    expected_classes = ["3.25", "3.5", "3.75", "4.0", "4.25"]
    # Check if dataset/ subdirectory exists
    if not EXPECTED_DATASET_DIR.exists():
        found_in_root = all((SCIENCEDB_DIR / c).exists() for c in expected_classes)
        if found_in_root:
            EXPECTED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
            for c in expected_classes:
                src = SCIENCEDB_DIR / c
                dst = EXPECTED_DATASET_DIR / c
                if src.exists():
                    shutil.move(str(src), str(dst))

    total = 0
    for c in expected_classes:
        c_dir = EXPECTED_DATASET_DIR / c
        if c_dir.exists():
            cnt = len(list(c_dir.glob("*.jpg")))
            print(f"  Class {c}: {cnt} images")
            total += cnt
        else:
            print(f"  [MISSING] Class {c} not found in {EXPECTED_DATASET_DIR}")

    print(f"[INFO] Total verified ScienceDB images: {total}")
    return total > 1000


def run_preprocessor():
    script = REPO_ROOT / "context" / "preprocess_sciencedb_bcs.py"
    if script.exists():
        print(f"\nRunning ScienceDB preprocessor: {script.name}...")
        subprocess.check_call([sys.executable, str(script)])
        print("[SUCCESS] ScienceDB index generated!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Threaded ScienceDB BCS Dataset Downloader")
    parser.add_argument("--url", default=DEFAULT_URL, help="Direct download URL")
    parser.add_argument("--threads", type=int, default=16, help="Number of parallel chunk threads (default: 16)")
    parser.add_argument("--clean-archive", action="store_true", help="Delete dataset.rar after successful extraction")
    parser.add_argument("--no-extract", action="store_true", help="Only download and stitch dataset.rar, skip extraction")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  ScienceDB High-Speed Chunk Downloader ({args.threads} parallel threads)")
    print("=" * 70)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    SCIENCEDB_DIR.mkdir(parents=True, exist_ok=True)

    # If archive already exists and is full size, skip download
    if ARCHIVE_PATH.exists() and not args.no_extract:
        print(f"[FOUND] Existing archive found at {ARCHIVE_PATH}")
    else:
        print(f"[INFO] Querying server file metadata...")
        total_size, range_supported = get_remote_file_info(args.url)
        print(f"[INFO] File size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
        print(f"[INFO] Server Range Support: {'YES' if range_supported else 'NO'}")

        if not range_supported:
            print("[WARN] Server does not report byte-range support. Falling back to single-stream.")
            num_threads = 1
        else:
            num_threads = max(1, args.threads)

        chunk_size = (total_size + num_threads - 1) // num_threads
        chunks = []
        part_files = []
        already_downloaded_bytes = 0

        for i in range(num_threads):
            start = i * chunk_size
            end = min((i + 1) * chunk_size - 1, total_size - 1)
            part_path = CHUNKS_DIR / f"chunk_{i:03d}.part"
            chunks.append((i, start, end, part_path))
            part_files.append(part_path)

            if part_path.exists():
                already_downloaded_bytes += min(part_path.stat().st_size, end - start + 1)

        print(f"[INFO] Partitioned into {num_threads} chunks (~{chunk_size / (1024**2):.1f} MB each).")
        if already_downloaded_bytes > 0:
            print(f"[RESUME] Resuming from {already_downloaded_bytes / (1024**2):.1f} MB already on disk!")

        with tqdm(
            total=total_size,
            initial=already_downloaded_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
        ) as pbar:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(download_chunk, args.url, idx, s, e, part, pbar): idx
                    for idx, s, e, part in chunks
                }
                for f in as_completed(futures):
                    chunk_idx = futures[f]
                    try:
                        f.result()
                    except Exception as err:
                        print(f"\n[FATAL] Error downloading chunk {chunk_idx}: {err}")
                        sys.exit(1)

        # Stitch chunks into final rar
        stitch_chunks(part_files, ARCHIVE_PATH, total_size)

    # Extraction step
    if not args.no_extract and ARCHIVE_PATH.exists():
        extracted = extract_rar(ARCHIVE_PATH, SCIENCEDB_DIR)
        if extracted:
            verify_classes()
            run_preprocessor()
            if args.clean_archive and ARCHIVE_PATH.exists():
                print(f"[CLEANUP] Removing {ARCHIVE_PATH.name} to reclaim disk space...")
                ARCHIVE_PATH.unlink()

    print("\n" + "=" * 70)
    print("  ScienceDB BCS Dataset is Ready!")
    print("=" * 70)


if __name__ == "__main__":
    main()
