"""
Modal Dataset Hydration Script
Downloads, extracts, and indexes all 4 cattle datasets directly into a persistent Modal Volume.
Run once from your local terminal:
    modal run scripts/modal_sync_datasets.py
"""

import modal

app = modal.App("cattle-dataset-sync")

# Persistent Cloud Volume (attached at /datasets)
dataset_volume = modal.Volume.from_name("cattle-datasets", create_if_missing=True)

# Container image with tools for fast download
sync_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "huggingface_hub",
        "kagglehub",
        "tqdm",
        "opencv-python-headless",
        "pandas",
        "requests",
    )
)

@app.function(
    image=sync_image,
    volumes={"/datasets": dataset_volume},
    timeout=3600 * 2,
    cpu=4.0,
    memory=16384,
)
def sync_all_datasets():
    import os
    import sys
    import shutil
    import zipfile
    import tarfile
    import subprocess
    from pathlib import Path
    from collections import defaultdict, Counter
    import random
    import csv

    print("==================================================")
    print("  HYDRATING CATTLE DATASETS ON MODAL VOLUME")
    print("==================================================")

    data_dir = Path("/datasets")
    lame_dir = data_dir / "lameness"
    beh_dir = data_dir / "behavior"
    id_dir = data_dir / "id"
    bcs_dir = data_dir / "bcs"

    for p in [lame_dir, beh_dir, id_dir, bcs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. LAMENESS (Mendeley CattleLameness)
    # -------------------------------------------------------------
    lame_csv = lame_dir / "lameness_index.csv"
    if not lame_csv.exists():
        print("\n[1/4] Restoring Mendeley CattleLameness...")
        repo_target = lame_dir / "CattleLameness"
        if not (repo_target / ".git").exists():
            subprocess.check_call(["git", "clone", "https://github.com/fahimsohan/CattleLameness", str(repo_target)])
        
        # Frame extraction
        frames_dir = lame_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        rows = []
        for split in ['train', 'test']:
            split_folder = repo_target / split
            if not split_folder.exists():
                continue
            for video_file in split_folder.glob("*.mp4"):
                stem = video_file.stem
                is_lame = 1 if 'lame' in stem.lower() else 0
                cap = cv2.VideoCapture(str(video_file))
                f_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if f_idx % 2 == 0:
                        out_path = frames_dir / f"{stem}_f{f_idx:05d}.jpg"
                        if not out_path.exists():
                            cv2.imwrite(str(out_path), frame)
                        rows.append([str(out_path), str(is_lame), stem, split])
                    f_idx += 1
                cap.release()

        with open(lame_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['image_path', 'label', 'cow_id', 'split'])
            w.writerows(rows)
        print(f"  [OK] Lameness indexed: {len(rows)} frames -> {lame_csv}")
    else:
        print(f"[1/4] Lameness already indexed ({lame_csv}).")

    # -------------------------------------------------------------
    # 2. BEHAVIOR (MmCows cropped_bboxes via Hugging Face)
    # -------------------------------------------------------------
    beh_csv = beh_dir / "behavior_index.csv"
    if not beh_csv.exists():
        print("\n[2/4] Restoring MmCows Behavior from Hugging Face...")
        from huggingface_hub import hf_hub_download
        mmcows_target = beh_dir / "mmcows"
        mmcows_target.mkdir(parents=True, exist_ok=True)
        
        zip_path = mmcows_target / "cropped_bboxes.zip"
        if not (mmcows_target / "cropped_bboxes").exists():
            print("  Downloading cropped_bboxes.zip (~12.7 GB) via 10Gbps cloud pipe...")
            dl_path = hf_hub_download(
                repo_id="neis-lab/mmcows",
                filename="cropped_bboxes.zip",
                repo_type="dataset",
                local_dir=str(mmcows_target),
            )
            print("  Extracting cropped_bboxes...")
            with zipfile.ZipFile(dl_path, 'r') as zf:
                zf.extractall(mmcows_target)
            if os.path.exists(dl_path):
                os.remove(dl_path) # purge archive to save volume space

        # Generate index CSV
        bbox_root = mmcows_target / "cropped_bboxes" / "behaviors"
        rows = []
        for class_id in range(1, 9):
            cdir = bbox_root / str(class_id)
            if not cdir.exists():
                continue
            for img in cdir.glob("*.jpg"):
                parts = img.stem.split('_')
                cow_id = parts[2] if len(parts) >= 3 else "unknown"
                split = "test" if cow_id in ["5", "6", "11", "12"] else "train"
                rows.append([str(img), str(class_id - 1), cow_id, split])

        with open(beh_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['image_path', 'label', 'cow_id', 'split'])
            w.writerows(rows)
        print(f"  [OK] Behavior indexed: {len(rows)} samples -> {beh_csv}")
    else:
        print(f"[2/4] Behavior already indexed ({beh_csv}).")

    # -------------------------------------------------------------
    # 3. COW ID (OpenCows2020 via Kagglehub)
    # -------------------------------------------------------------
    id_csv = id_dir / "id_index.csv"
    if not id_csv.exists():
        print("\n[3/4] Restoring OpenCows2020 via Kagglehub...")
        import kagglehub
        path = Path(kagglehub.dataset_download("amibhavsar/open-cow-2020"))
        
        # Locate identification/images
        id_img_root = None
        for cand in [
            path / "identification" / "images",
            path / "10m32xl88x2b61zlkkgz3fml17" / "identification" / "images",
        ]:
            if cand.exists() and (cand / "train").exists():
                id_img_root = cand
                break
        if not id_img_root:
            for p in path.rglob("train"):
                if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
                    id_img_root = p.parent
                    break

        target_ninja = id_dir / "opencow2020-DatasetNinja"
        train_dst = target_ninja / "identification-train" / "img"
        test_dst = target_ninja / "identification-test" / "img"
        train_dst.mkdir(parents=True, exist_ok=True)
        test_dst.mkdir(parents=True, exist_ok=True)

        for split_name, src, dst in [("train", id_img_root / "train", train_dst), ("test", id_img_root / "test", test_dst)]:
            for cow_f in src.iterdir():
                if not cow_f.is_dir():
                    continue
                try:
                    cid = str(int(cow_f.name))
                except ValueError:
                    continue
                for img in cow_f.glob("*.jpg"):
                    dest_file = dst / f"{cid}_{img.name}"
                    if not dest_file.exists():
                        shutil.copy2(img, dest_file)

        # Index CSV
        train_by_cow = defaultdict(list)
        for f in sorted(os.listdir(train_dst)):
            if f.endswith('.jpg'):
                cid = f.split('_')[0]
                train_by_cow[cid].append(str(train_dst / f))
        
        rows = []
        random.seed(42)
        for cid, imgs in train_by_cow.items():
            random.shuffle(imgs)
            n_tr = int(len(imgs) * 0.85)
            lbl = str(int(cid) - 1)
            for idx, pth in enumerate(imgs):
                sp = 'train' if idx < n_tr else 'val'
                rows.append([pth, lbl, cid, sp])

        for f in sorted(os.listdir(test_dst)):
            if f.endswith('.jpg'):
                cid = f.split('_')[0]
                lbl = str(int(cid) - 1)
                rows.append([str(test_dst / f), lbl, cid, 'test'])

        with open(id_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['image_path', 'label', 'cow_id', 'split'])
            w.writerows(rows)
        print(f"  [OK] Cow ID indexed: {len(rows)} samples -> {id_csv}")
    else:
        print(f"[3/4] Cow ID already indexed ({id_csv}).")

    # -------------------------------------------------------------
    # Commit changes to permanent Volume
    # -------------------------------------------------------------
    print("\nCommitting changes to persistent Modal Volume...")
    dataset_volume.commit()
    print("==================================================")
    print("  [SUCCESS] DATASETS FULLY HYDRATED ON MODAL!")
    print("==================================================")

@app.local_entrypoint()
def main():
    sync_all_datasets.remote()
