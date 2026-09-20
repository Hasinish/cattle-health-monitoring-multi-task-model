"""
Comprehensive Exact-Duplicate and Perceptual Near-Duplicate Leakage Audit.

Audits split-bearing datasets in Phase 3:
1. ScienceDB BCS (passage-disjoint train/val/test)
2. MmCows Behavior (cow-grouped canonical split and 4-fold GroupKFold)
3. OpenCows2020 (legacy train/val/official-test protocol)

Detection Methods:
- Exact duplicates: SHA-256 content hashing of raw image bytes.
- Perceptual near-duplicates: 64-bit dHash (Difference Hash, 9x8 gradient comparison)
  and 64-bit aHash (Average Hash, 8x8 mean comparison), searched via Multi-Index Hashing (MIH)
  for Hamming distance d <= 6.
- Verification: For near-duplicate suspects, computes normalized pixel MAE on 64x64 grayscale.

Outputs detailed CSV reports of suspicious cross-partition pairs.
"""

import os
import sys
import time
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_image_fingerprints(img_path: str):
    """
    Compute SHA-256, 64-bit dHash, and 64-bit aHash for a given image.
    Returns (sha256, dhash_int, ahash_int, (h, w), error_msg)
    """
    try:
        p = Path(img_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        
        with open(p, "rb") as f:
            raw = f.read()
        sha256 = hashlib.sha256(raw).hexdigest()

        # Load image via cv2
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return sha256, None, None, None, "Unreadable image"

        orig_shape = img.shape  # (h, w)

        # 1. 64-bit dHash (resize to 9x8, compare adjacent columns)
        d_resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        d_diff = d_resized[:, 1:] > d_resized[:, :-1]
        dhash_val = 0
        for bit in d_diff.flatten():
            dhash_val = (dhash_val << 1) | int(bit)

        # 2. 64-bit aHash (resize to 8x8, compare with mean)
        a_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        a_diff = a_resized > a_resized.mean()
        ahash_val = 0
        for bit in a_diff.flatten():
            ahash_val = (ahash_val << 1) | int(bit)

        return sha256, dhash_val, ahash_val, orig_shape, None
    except Exception as e:
        return None, None, None, None, str(e)


class MultiIndexHash64:
    """
    Multi-Index Hash for 64-bit integers.
    Divides 64 bits into 8 blocks of 8 bits each.
    Guarantees finding all pairs with Hamming distance <= 7 by pigeonhole principle.
    """
    def __init__(self):
        self.tables = [defaultdict(list) for _ in range(8)]
        self.hashes = []
        self.metadata = []

    def add(self, h: int, meta: dict):
        idx = len(self.hashes)
        self.hashes.append(h)
        self.metadata.append(meta)
        for b in range(8):
            byte_val = (h >> (b * 8)) & 0xFF
            self.tables[b][byte_val].append(idx)

    def query(self, q_h: int, max_dist: int = 6):
        """
        Find all indexed hashes with Hamming distance <= max_dist.
        Returns list of (indexed_meta, dist).
        """
        candidate_indices = set()
        for b in range(8):
            byte_val = (q_h >> (b * 8)) & 0xFF
            candidate_indices.update(self.tables[b].get(byte_val, []))

        results = []
        for idx in candidate_indices:
            h = self.hashes[idx]
            dist = bin(q_h ^ h).count("1")
            if dist <= max_dist:
                results.append((self.metadata[idx], dist))
        return results


def audit_dataset_splits(
    dataset_name: str,
    split_files: dict[str, str],
    id_col: str = "cow_id",
    max_hamming_dist: int = 6,
    max_workers: int = 8,
):
    print("\n" + "=" * 75)
    print(f"  AUDITING DATASET: {dataset_name}")
    print("=" * 75)

    # 1. Load dataframes
    dfs = {}
    total_images = 0
    for split_name, path in split_files.items():
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        df = pd.read_csv(p)
        df["split"] = split_name
        dfs[split_name] = df
        total_images += len(df)
        print(f"  Split '{split_name}': {len(df):,} images ({p.name})")
    print(f"  Total images to audit: {total_images:,}")

    # 2. Extract fingerprints in parallel
    all_rows = []
    for split_name, df in dfs.items():
        for _, row in df.iterrows():
            item = dict(row)
            item["split_name"] = split_name
            all_rows.append(item)

    print(f"\nComputing fingerprints (SHA-256 + 64-bit dHash + aHash) across {len(all_rows):,} samples...")
    t0 = time.time()
    paths = [r["image_path"] for r in all_rows]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fingerprints = list(executor.map(compute_image_fingerprints, paths))
    t1 = time.time()
    print(f"Fingerprinting completed in {t1 - t0:.2f}s ({(t1 - t0)/len(all_rows)*1000:.2f} ms/image)")

    # Attach results
    valid_samples = []
    unreadable = 0
    for row, (sha, dh, ah, shape, err) in zip(all_rows, fingerprints):
        if err or dh is None:
            unreadable += 1
            continue
        row["sha256"] = sha
        row["dhash"] = dh
        row["ahash"] = ah
        row["img_shape"] = shape
        valid_samples.append(row)

    if unreadable > 0:
        print(f"[WARNING] {unreadable} unreadable/corrupt images encountered!")

    # 3. Exact Duplicate Audit (SHA-256)
    print("\n--- PHASE 1: Exact Duplicate Audit (SHA-256) ---")
    sha_map = defaultdict(list)
    for sample in valid_samples:
        sha_map[sample["sha256"]].append(sample)

    exact_within_partition = 0
    exact_cross_partition = []

    for sha, group in sha_map.items():
        if len(group) > 1:
            splits = {s["split_name"] for s in group}
            if len(splits) > 1:
                exact_cross_partition.append(group)
            else:
                exact_within_partition += (len(group) - 1)

    print(f"Exact duplicates within same partition: {exact_within_partition:,} pairs/occurrences")
    print(f"Exact duplicates CROSSING partitions:   {len(exact_cross_partition):,} groups")

    if exact_cross_partition:
        print("[ALERT] Cross-partition exact duplicate groups found:")
        for grp in exact_cross_partition[:5]:
            splits = [g["split_name"] for g in grp]
            paths = [g["image_path"] for g in grp]
            cows = [str(g.get(id_col, "NA")) for g in grp]
            print(f"  - SHA256: {grp[0]['sha256'][:16]}... Splits: {splits} Cows: {cows}")

    # 4. Perceptual Near-Duplicate Audit (Multi-Index Hashing on dHash)
    print(f"\n--- PHASE 2: Perceptual Near-Duplicate Audit (dHash Hamming <= {max_hamming_dist}) ---")
    # Separate samples by split
    samples_by_split = defaultdict(list)
    for s in valid_samples:
        samples_by_split[s["split_name"]].append(s)

    split_names = list(samples_by_split.keys())
    cross_suspects = []

    # Pairwise cross-split search
    for i in range(len(split_names)):
        split_a = split_names[i]
        index_a = MultiIndexHash64()
        for s in samples_by_split[split_a]:
            index_a.add(s["dhash"], s)

        for j in range(i + 1, len(split_names)):
            split_b = split_names[j]
            print(f"  Searching cross-partition: '{split_a}' ({len(samples_by_split[split_a]):,}) vs '{split_b}' ({len(samples_by_split[split_b]):,})...")
            
            t_search0 = time.time()
            pair_count = 0
            for s_b in samples_by_split[split_b]:
                matches = index_a.query(s_b["dhash"], max_dist=max_hamming_dist)
                for s_a, dist in matches:
                    # Skip if exact same file path (sanity check)
                    if s_a["image_path"] == s_b["image_path"]:
                        continue
                    
                    # Compute aHash distance as second confirmation
                    ahash_dist = bin(s_a["ahash"] ^ s_b["ahash"]).count("1")
                    
                    cross_suspects.append({
                        "dataset": dataset_name,
                        "split_a": split_a,
                        "split_b": split_b,
                        "image_path_a": s_a["image_path"],
                        "image_path_b": s_b["image_path"],
                        "cow_id_a": s_a.get(id_col, "NA"),
                        "cow_id_b": s_b.get(id_col, "NA"),
                        "dhash_dist": dist,
                        "ahash_dist": ahash_dist,
                        "exact_sha256": s_a["sha256"] == s_b["sha256"],
                    })
                    pair_count += 1
            t_search1 = time.time()
            print(f"    Found {pair_count:,} pairs with dHash dist <= {max_hamming_dist} in {t_search1 - t_search0:.2f}s")

    print(f"\nTotal cross-partition near-duplicate suspects (dHash <= {max_hamming_dist}): {len(cross_suspects):,}")

    # Compute pixel MAE on top suspicious pairs (up to 50 strongest)
    if cross_suspects:
        # Sort by dhash_dist ascending, then ahash_dist ascending
        cross_suspects.sort(key=lambda x: (x["dhash_dist"], x["ahash_dist"]))
        print(f"\nComputing pixel MAE verification for top {min(len(cross_suspects), 50)} candidates...")
        for pair in cross_suspects[:50]:
            try:
                pa = Path(pair["image_path_a"])
                pb = Path(pair["image_path_b"])
                if not pa.is_absolute(): pa = REPO_ROOT / pa
                if not pb.is_absolute(): pb = REPO_ROOT / pb
                
                im_a = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
                im_b = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
                if im_a is not None and im_b is not None:
                    # Resize to common 64x64
                    ra = cv2.resize(im_a, (64, 64), interpolation=cv2.INTER_AREA).astype(float)
                    rb = cv2.resize(im_b, (64, 64), interpolation=cv2.INTER_AREA).astype(float)
                    mae = np.mean(np.abs(ra - rb))
                    pair["pixel_mae_64x64"] = round(mae, 2)
                else:
                    pair["pixel_mae_64x64"] = -1.0
            except Exception:
                pair["pixel_mae_64x64"] = -1.0

        # Print top 10 suspects
        print("\nTop 10 Most Suspicious Cross-Partition Pairs:")
        for idx, p in enumerate(cross_suspects[:10], 1):
            print(
                f"  [{idx:02d}] {p['split_a']} vs {p['split_b']} | "
                f"Cows: {p['cow_id_a']} vs {p['cow_id_b']} | "
                f"dHash dist: {p['dhash_dist']} | aHash dist: {p['ahash_dist']} | "
                f"Pixel MAE: {p.get('pixel_mae_64x64', 'NA')} | "
                f"\n       A: {Path(p['image_path_a']).name}"
                f"\n       B: {Path(p['image_path_b']).name}"
            )

    return {
        "dataset": dataset_name,
        "total_images": total_images,
        "valid_images": len(valid_samples),
        "exact_within": exact_within_partition,
        "exact_cross": len(exact_cross_partition),
        "near_cross": len(cross_suspects),
        "suspects": cross_suspects,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit split-bearing datasets for exact & near-duplicate leakage")
    parser.add_argument("--dataset", choices=["all", "sciencedb", "mmcows", "opencows"], default="all", help="Dataset to audit")
    parser.add_argument("--hamming-dist", type=int, default=6, help="Maximum dHash Hamming distance for near-duplicates")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads for image decoding")
    args = parser.parse_args()

    results = []

    # 1. OpenCows2020 (Legacy Baseline)
    if args.dataset in ("all", "opencows"):
        res = audit_dataset_splits(
            dataset_name="OpenCows2020 (Legacy)",
            split_files={
                "train": "datasets/id/opencow2020/train.csv",
                "val": "datasets/id/opencow2020/val.csv",
                "test": "datasets/id/opencow2020/test.csv",
            },
            id_col="cow_id",
            max_hamming_dist=args.hamming_dist,
            max_workers=args.workers,
        )
        results.append(res)

    # 2. ScienceDB BCS
    if args.dataset in ("all", "sciencedb"):
        res = audit_dataset_splits(
            dataset_name="ScienceDB Cattle BCS",
            split_files={
                "train": "datasets/bcs/sciencedb/train.csv",
                "val": "datasets/bcs/sciencedb/val.csv",
                "test": "datasets/bcs/sciencedb/test.csv",
            },
            id_col="cow_id",
            max_hamming_dist=args.hamming_dist,
            max_workers=args.workers,
        )
        results.append(res)

    # 3. MmCows Behavior (Canonical Split)
    if args.dataset in ("all", "mmcows"):
        res = audit_dataset_splits(
            dataset_name="MmCows Behavior (Canonical)",
            split_files={
                "train": "datasets/behavior/mmcows/train.csv",
                "val": "datasets/behavior/mmcows/val.csv",
                "test": "datasets/behavior/mmcows/test.csv",
            },
            id_col="cow_id",
            max_hamming_dist=args.hamming_dist,
            max_workers=args.workers,
        )
        results.append(res)

        # Also audit 4-Fold GroupKFold
        res_folds = audit_dataset_splits(
            dataset_name="MmCows Behavior (4-Fold GroupKFold)",
            split_files={
                "fold_0": "datasets/behavior/mmcows/folds/fold_0.csv",
                "fold_1": "datasets/behavior/mmcows/folds/fold_1.csv",
                "fold_2": "datasets/behavior/mmcows/folds/fold_2.csv",
                "fold_3": "datasets/behavior/mmcows/folds/fold_3.csv",
            },
            id_col="cow_id",
            max_hamming_dist=args.hamming_dist,
            max_workers=args.workers,
        )
        results.append(res_folds)

    # Export combined suspects to CSV
    all_suspects = []
    for r in results:
        all_suspects.extend(r["suspects"])

    if all_suspects:
        out_csv = REPO_ROOT / "docs" / "audits" / "phase3_near_duplicate_suspects.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_suspects).to_csv(out_csv, index=False)
        print(f"\n[EXPORT] Detailed suspect pairs exported to {out_csv} ({len(all_suspects):,} rows)")

    # Print summary table
    print("\n" + "=" * 80)
    print("  FINAL LEAKAGE AUDIT SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<35} | {'Total Imgs':<10} | {'Exact Cross':<12} | {f'Near Cross (d<={args.hamming_dist})':<18}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<35} | {r['total_images']:<10,d} | {r['exact_cross']:<12,d} | {r['near_cross']:<18,d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
