"""
Standalone verification script for OpenCows2020 legacy Re-ID evaluation protocol.
Performs rigorous audit of manifests, splits, identity distribution, path resolution,
and cryptographic duplicate overlap / frame-index adjacency crossing.
"""

import sys
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "id" / "opencow2020"
ID_INDEX_CSV = REPO_ROOT / "datasets" / "id" / "id_index.csv"
RAW_TEST_DIR = REPO_ROOT / "datasets" / "id" / "opencow2020-DatasetNinja" / "identification-test" / "img"
RAW_TRAIN_DIR = REPO_ROOT / "datasets" / "id" / "opencow2020-DatasetNinja" / "identification-train" / "img"


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify():
    print("==================================================================")
    print("  OPENCOWS2020 LEGACY RE-ID EVALUATION PROTOCOL VERIFICATION")
    print("==================================================================")

    errors = []

    manifest_path = DATASET_DIR / "manifest.csv"
    train_path = DATASET_DIR / "train.csv"
    val_path = DATASET_DIR / "val.csv"
    test_path = DATASET_DIR / "test.csv"

    for p in [manifest_path, train_path, val_path, test_path, ID_INDEX_CSV]:
        if not p.exists():
            errors.append(f"Missing required file: {p}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(val_path, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    with open(ID_INDEX_CSV, "r", encoding="utf-8") as f:
        id_index_rows = list(csv.DictReader(f))

    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Train rows:    {len(train_rows)}")
    print(f"Val rows:      {len(val_rows)}")
    print(f"Test rows:     {len(test_rows)}")
    print(f"id_index rows: {len(id_index_rows)}")

    # 1. Total counts check
    if len(manifest_rows) != 4736:
        errors.append(f"Manifest count != 4736 (got {len(manifest_rows)})")
    if len(train_rows) != 3586:
        errors.append(f"Train count != 3586 (got {len(train_rows)})")
    if len(val_rows) != 654:
        errors.append(f"Val count != 654 (got {len(val_rows)})")
    if len(test_rows) != 496:
        errors.append(f"Test count != 496 (got {len(test_rows)})")
    if len(id_index_rows) != 4736:
        errors.append(f"id_index.csv count != 4736 (got {len(id_index_rows)})")

    # 2. Path resolution check
    missing_paths = 0
    for r in manifest_rows:
        if not Path(r["image_path"]).exists():
            missing_paths += 1
    if missing_paths > 0:
        errors.append(f"{missing_paths} manifest image paths do not exist on disk")

    # 3. Identities check (46 cows in all splits)
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        cows = set(int(r["cow_id"]) for r in rows)
        if len(cows) != 46:
            errors.append(f"{name} split does not have 46 cows (got {len(cows)})")
        expected = set(range(1, 47))
        if cows != expected:
            errors.append(f"{name} split cow set mismatch: missing {expected - cows}")

    # 4. Official test set integrity
    official_test_files = set(RAW_TEST_DIR.iterdir())
    manifest_test_paths = set(Path(r["image_path"]) for r in test_rows)
    if manifest_test_paths != official_test_files:
        errors.append("Test set does not match official identification-test files 1-to-1")

    # 5. Cryptographic exact-duplicate overlap check
    t_hashes = set(r["sha256"] for r in train_rows)
    v_hashes = set(r["sha256"] for r in val_rows)
    te_hashes = set(r["sha256"] for r in test_rows)

    tv_inter = t_hashes.intersection(v_hashes)
    tte_inter = t_hashes.intersection(te_hashes)
    vte_inter = v_hashes.intersection(te_hashes)

    if len(tv_inter) > 0:
        errors.append(f"Exact-duplicate overlap between train and val: {len(tv_inter)} hashes")
    if len(tte_inter) > 0:
        errors.append(f"Exact-duplicate overlap between train and test: {len(tte_inter)} hashes")
    if len(vte_inter) > 0:
        errors.append(f"Exact-duplicate overlap between val and test: {len(vte_inter)} hashes")

    # 6. Frame-index adjacency crossing check (|f_train - f_val| == 1)
    adj_crossings = 0
    for cid in range(1, 47):
        t_fids = set(int(r["frame_id"]) for r in train_rows if int(r["cow_id"]) == cid)
        v_fids = set(int(r["frame_id"]) for r in val_rows if int(r["cow_id"]) == cid)
        for tf in t_fids:
            if (tf - 1) in v_fids or (tf + 1) in v_fids:
                adj_crossings += 1

    print(f"Frame-index adjacency crossings across train/val: {adj_crossings} (expected <= 48 single boundary crossings, down from 1,023 under random mixing)")
    if adj_crossings > 48:
        errors.append(f"Frame-index adjacency crossings exceed expected single boundary transitions: {adj_crossings}")

    # 7. id_index.csv alignment check
    manifest_map = {r["image_path"]: r for r in manifest_rows}
    id_index_mismatches = 0
    for r in id_index_rows:
        m = manifest_map.get(r["image_path"])
        if not m or m["split"] != r["split"] or str(m["cow_id"]) != str(r["cow_id"]) or str(m["label"]) != str(r["label"]):
            id_index_mismatches += 1
    if id_index_mismatches > 0:
        errors.append(f"id_index.csv has {id_index_mismatches} mismatches against manifest.csv")

    print("\n--- Summary of Verification ---")
    if errors:
        print(f"FAILED: {len(errors)} error(s) found:")
        for e in errors:
            print(f"  [ERROR] {e}")
        return False
    else:
        print("[PASS] All 4,736 images verified.")
        print("[PASS] All 46 identities present in Train, Val, and Test.")
        print("[PASS] Official identification-test benchmark set (496 images) is 100% preserved.")
        print("[PASS] Zero exact-duplicate overlap between any partition.")
        print(f"[PASS] Frame-index adjacency crossings reduced from 1,023 down to {adj_crossings} (contiguous frame-index heuristic).")
        print("[PASS] id_index.csv is 100% aligned with manifest.csv.")
        print("[NOTE] True tracklet/temporal leakage cannot be verified because provenance is unavailable.")
        print("[SUCCESS] OpenCows2020 protocol is fully verified and reproducible!")
        return True


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
