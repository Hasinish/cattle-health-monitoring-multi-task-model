# -*- coding: utf-8 -*-
"""
Modal Cloud Pipeline for MOO (Multi-view Oriented Observations)
Handles:
  1. Ultra-fast cloud download of MOO.zip into a persistent Modal Volume
  2. Extraction of metadata.json and data.hdf5
  3. Fast inspection of viewpoint/azimuth/elevation distributions
  4. T4 GPU smoke training of cattle viewpoint classifier on synthetic crops

Usage:
  # 1. Download MOO into Modal Volume:
  modal run --profile mohtasimahmedsamii scripts/modal_moo_pipeline.py::download_moo

  # 2. Inspect metadata & verify coordinate system:
  modal run --profile mohtasimahmedsamii scripts/modal_moo_pipeline.py::inspect_moo
"""
import os
import sys

# Guard against Windows cross-drive ValueError in ntpath.commonpath for Modal
if sys.platform == "win32":
    _orig_commonpath = os.path.commonpath

    def _safe_commonpath(paths):
        try:
            return _orig_commonpath(paths)
        except ValueError:
            return ""

    os.path.commonpath = _safe_commonpath

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

from pathlib import Path

# Persistent storage volume for MOO dataset
volume = modal.Volume.from_name("moo-data", create_if_missing=True)
VOLUME_DIR = "/data"

# Canonical split directory on host
SPLITS_DIR_LOCAL = Path(__file__).resolve().parent.parent / "datasets" / "viewpoint" / "moo"

# Container image with aria2, h5py, PyTorch, and imaging tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2", "unzip")
    .pip_install(
        "requests",
        "tqdm",
        "h5py",
        "numpy",
        "pandas",
        "pillow",
        "torch",
        "torchvision",
        "scikit-learn"
    )
)

if SPLITS_DIR_LOCAL.exists():
    image = image.add_local_dir(str(SPLITS_DIR_LOCAL), remote_path="/root/moo_splits")

app = modal.App("moo-viewpoint-pipeline", image=image)

MOO_URL = "https://kalisteo.cea.fr/index.php/download/moo-dataset/?wpdmdl=4365"


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=3600,  # 1 hour max
    cpu=2.0,
    memory=4096,  # 4 GB RAM for fast disk I/O throughput
)
def download_moo():
    """Download MOO.zip into persistent Modal volume and extract it."""
    import subprocess
    import time

    os.makedirs(VOLUME_DIR, exist_ok=True)
    zip_path = os.path.join(VOLUME_DIR, "MOO.zip")
    hdf5_path = os.path.join(VOLUME_DIR, "data.hdf5")
    meta_path = os.path.join(VOLUME_DIR, "metadata.json")

    print(f"=== MOO Cloud Download & Setup ===")
    print(f"Volume directory: {VOLUME_DIR}")

    # Check if already extracted
    if os.path.exists(hdf5_path) and os.path.exists(meta_path):
        h5_size_gb = os.path.getsize(hdf5_path) / (1024 ** 3)
        meta_size_mb = os.path.getsize(meta_path) / (1024 ** 2)
        print(f"✓ MOO is already extracted and ready in volume!")
        print(f"  data.hdf5:     {h5_size_gb:.2f} GB")
        print(f"  metadata.json: {meta_size_mb:.2f} MB")
        return {"status": "already_extracted", "hdf5_gb": h5_size_gb}

    # Step 1: Download MOO.zip if not present or incomplete
    import zipfile
    aria2_control = zip_path + ".aria2"
    is_complete_zip = (
        os.path.exists(zip_path)
        and not os.path.exists(aria2_control)
        and zipfile.is_zipfile(zip_path)
    )

    if not is_complete_zip:
        print(f"\n[1/2] Downloading MOO.zip using aria2c (16 parallel connections)...")
        if os.path.exists(aria2_control):
            print(f"  Found existing aria2 session file ({aria2_control}). Resuming download...")
        elif os.path.exists(zip_path):
            print(f"  Existing MOO.zip is incomplete/corrupt. Resuming or completing chunks...")
        start_time = time.time()
        import threading
        stop_commit_thread = threading.Event()

        def periodic_commit_worker():
            while not stop_commit_thread.wait(60):
                try:
                    volume.commit()
                    print("\n[Volume Checkpoint] 60s commit complete. Progress saved!", flush=True)
                except Exception as ce:
                    print(f"\n[Volume Checkpoint] Commit warning: {ce}", flush=True)

        commit_thread = threading.Thread(target=periodic_commit_worker, daemon=True)
        commit_thread.start()

        cmd = [
            "aria2c",
            "-c",
            "-s", "16",
            "-x", "16",
            "-j", "16",
            "-k", "1M",
            "--disk-cache=64M",
            "--file-allocation=falloc",
            "--stream-piece-selector=geom",
            "--max-tries=0",
            "--retry-wait=1",
            "--check-certificate=false",
            "-d", VOLUME_DIR,
            "-o", "MOO.zip",
            "--summary-interval=2",
            MOO_URL
        ]

        try:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise RuntimeError(f"aria2c download failed with exit code {res.returncode}")
        except (KeyboardInterrupt, SystemExit, BaseException) as e:
            print(f"\n[Interrupt] Caught signal ({type(e).__name__}). Flushing final commit to volume...", flush=True)
            stop_commit_thread.set()
            try:
                volume.commit()
                print("✓ Volume committed on interrupt! Partial download is safe and resumable.", flush=True)
            except Exception as ce:
                print(f"Failed to commit volume on interrupt: {ce}", flush=True)
            raise
        finally:
            stop_commit_thread.set()
            commit_thread.join(timeout=5)
            try:
                volume.commit()
            except Exception:
                pass

        elapsed = time.time() - start_time
        downloaded_gb = os.path.getsize(zip_path) / (1024 ** 3)
        speed_mb = (downloaded_gb * 1024) / max(1, elapsed)
        print(f"✓ Download complete in {elapsed:.1f}s ({speed_mb:.1f} MB/s avg)!")
        volume.commit()
    else:
        print(f"✓ MOO.zip already downloaded and verified ({os.path.getsize(zip_path) / (1024**3):.2f} GB)")

    # Step 2: Extract MOO.zip
    print(f"\n[2/2] Extracting MOO.zip into {VOLUME_DIR}...")
    start_unzip = time.time()
    unzip_cmd = ["unzip", "-o", zip_path, "-d", VOLUME_DIR]
    res_unzip = subprocess.run(unzip_cmd)
    if res_unzip.returncode != 0:
        raise RuntimeError(f"unzip failed with exit code {res_unzip.returncode}")

    unzip_elapsed = time.time() - start_unzip
    print(f"✓ Extraction complete in {unzip_elapsed:.1f}s!")

    # Optionally remove MOO.zip to save volume space
    print("Cleaning up MOO.zip to preserve volume quota...")
    try:
        os.remove(zip_path)
        print("✓ Removed MOO.zip archive (extracted data preserved).")
    except Exception as e:
        print(f"Warning: could not remove MOO.zip: {e}")

    # Commit volume changes so data persists across containers
    volume.commit()
    print("✓ Volume committed successfully!")

    h5_size_gb = os.path.getsize(hdf5_path) / (1024 ** 3)
    meta_size_mb = os.path.getsize(meta_path) / (1024 ** 2)
    return {
        "status": "success",
        "hdf5_gb": h5_size_gb,
        "meta_mb": meta_size_mb
    }


@app.function(
    volumes={VOLUME_DIR: volume},
    cpu=1.0,
    memory=2048,  # 2 GB RAM (minimal cost)
)
def inspect_moo():
    """Inspect metadata.json and data.hdf5 inside the volume."""
    import json
    import h5py

    hdf5_path = os.path.join(VOLUME_DIR, "data.hdf5")
    meta_path = os.path.join(VOLUME_DIR, "metadata.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {VOLUME_DIR}. Run download_moo first!")

    print("=== Inspecting MOO Metadata & Structure ===")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    total_samples = len(metadata)
    print(f"Total samples in metadata: {total_samples}")

    # Inspect first few samples to see keys and coordinate conventions
    first_key = list(metadata.keys())[0]
    sample_entry = metadata[first_key]
    print(f"\nSample entry [{first_key}]:")
    for k, v in sample_entry.items():
        print(f"  {k}: {v}")

    # Check HDF5 structure
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, "r") as hf:
            print(f"\nHDF5 Keys: {list(hf.keys())}")
            for k in list(hf.keys())[:3]:
                ds = hf[k]
                print(f"  Dataset '{k}': shape={getattr(ds, 'shape', 'N/A')}, dtype={getattr(ds, 'dtype', 'N/A')}")

    # Analyze Azimuth and Elevation distributions
    azimuths = []
    elevations = []
    for k, entry in metadata.items():
        # Keys might be 'azimuth', 'phi', 'elevation', 'theta'
        az = entry.get("azimuth", entry.get("phi", None))
        el = entry.get("elevation", entry.get("theta", None))
        if az is not None:
            azimuths.append(float(az))
        if el is not None:
            elevations.append(float(el))

    if azimuths:
        print(f"\nAzimuth Distribution ({len(azimuths)} samples):")
        print(f"  Min: {min(azimuths):.2f}°, Max: {max(azimuths):.2f}°")
    if elevations:
        print(f"\nElevation Distribution ({len(elevations)} samples):")
        print(f"  Min: {min(elevations):.2f}°, Max: {max(elevations):.2f}°")

    return {
        "total_samples": total_samples,
        "sample_keys": list(sample_entry.keys())
    }


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=600,
    cpu=4.0,
    memory=4096,
)
def train_smoke_classifier(samples_per_class: int = 500, epochs: int = 15):
    """Train a fast 5-class linear viewpoint head on ResNet-18 using MOO synthetic crops on CPU."""
    import io
    import time
    import json
    import random
    import numpy as np
    import h5py
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torchvision.transforms as T
    import torchvision.models as models
    from PIL import Image

    print("=== MOO Viewpoint Synthetic-to-Real Smoke Test (CPU Optimized) ===")
    print(f"Hardware: 4 vCPUs, Samples/Class: {samples_per_class}, Epochs: {epochs}")

    meta_path = os.path.join(VOLUME_DIR, "metadata.json")
    hdf5_path = os.path.join(VOLUME_DIR, "data.hdf5")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Class mapping
    VIEW_MAP = {
        "front": 0,
        "front-left": 1,
        "front-right": 1,
        "left": 2,
        "right": 2,
        "back-left": 3,
        "back-right": 3,
        "back": 4,
    }
    CLASS_NAMES = ["front", "front-oblique", "side", "rear-oblique", "rear"]

    # Filter and group samples by class
    class_pools = {i: [] for i in range(5)}
    for cow_id, cow_imgs in meta.items():
        for img_id, entry in cow_imgs.items():
            view = entry.get("view", "")
            if view.startswith("top-"):
                continue
            el = float(entry.get("elevation", 0))
            if el > 50 or el < -25:
                continue
            if view in VIEW_MAP:
                cls = VIEW_MAP[view]
                class_pools[cls].append((cow_id, img_id))

    print("\nAvailable filtered samples per class:")
    for cls_idx, name in enumerate(CLASS_NAMES):
        print(f"  Class {cls_idx} ({name:14s}): {len(class_pools[cls_idx])} samples")

    # Sample balanced dataset
    random.seed(2026)
    train_samples = []
    val_samples = []
    val_per_class = min(100, samples_per_class // 5)

    for cls_idx in range(5):
        pool = class_pools[cls_idx]
        random.shuffle(pool)
        needed = samples_per_class + val_per_class
        selected = pool[:needed]
        train_part = selected[:samples_per_class]
        val_part = selected[samples_per_class:]
        for cow_id, img_id in train_part:
            train_samples.append((cow_id, img_id, cls_idx))
        for cow_id, img_id in val_part:
            val_samples.append((cow_id, img_id, cls_idx))

    random.shuffle(train_samples)
    print(f"\nBalanced Dataset: {len(train_samples)} train, {len(val_samples)} val")

    # Dataset definition
    class MOODataset(Dataset):
        def __init__(self, samples, hdf5_path, transform=None):
            self.samples = samples
            self.hdf5_path = hdf5_path
            self.transform = transform
            self.hf = None

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            if self.hf is None:
                self.hf = h5py.File(self.hdf5_path, "r")

            cow_id, img_id, label = self.samples[idx]
            grp = self.hf[cow_id]
            if img_id in grp:
                obj = grp[img_id]
            elif f"{cow_id}_{img_id}" in grp:
                obj = grp[f"{cow_id}_{img_id}"]
            else:
                obj = grp[list(grp.keys())[0]]

            # Handle if obj is a sub-group (e.g. obj['image']) or a direct dataset
            if isinstance(obj, h5py.Group):
                if "image" in obj:
                    raw = obj["image"][()]
                elif "rgb" in obj:
                    raw = obj["rgb"][()]
                elif "data" in obj:
                    raw = obj["data"][()]
                else:
                    raw = obj[list(obj.keys())[0]][()]
            else:
                raw = obj[()]

            if isinstance(raw, (bytes, bytearray, np.void)) or (hasattr(raw, 'dtype') and raw.dtype == np.uint8 and raw.ndim == 1):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                img = Image.fromarray(raw).convert("RGB")

            if self.transform:
                img = self.transform(img)

            return img, label

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MOODataset(train_samples, hdf5_path, transform)
    val_ds = MOODataset(val_samples, hdf5_path, transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    # 1. Feature Extraction on CPU using frozen ResNet-18
    print("\n--- Pre-extracting 512-dim features with frozen ResNet-18 ---")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.eval()

    start_feat = time.time()
    def extract_features(loader):
        feats, targets = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                out = resnet(imgs)
                feats.append(out)
                targets.append(labels)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    X_train, y_train = extract_features(train_loader)
    X_val, y_val = extract_features(val_loader)
    feat_time = time.time() - start_feat
    print(f"✓ Feature extraction complete in {feat_time:.1f}s! (Train: {X_train.shape}, Val: {X_val.shape})")

    # 2. Fast Linear Head Training
    head = nn.Linear(512, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)

    fast_train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    print(f"\n--- Training Linear Head ({epochs} Epochs) ---")
    start_train = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for fts, lbls in fast_train_loader:
            optimizer.zero_grad()
            outs = head(fts)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * fts.size(0)
            preds = outs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        head.eval()
        with torch.no_grad():
            v_outs = head(X_val)
            v_preds = v_outs.argmax(dim=1)
            val_acc = (v_preds == y_val).float().mean().item()

        print(f"Epoch [{epoch:2d}/{epochs:2d}] Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    train_time = time.time() - start_train
    print(f"\n✓ Linear head trained in {train_time:.2f}s!")

    # 3. Assemble and save full checkpoint
    export_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    export_model.fc = head

    save_path = os.path.join(VOLUME_DIR, "moo_resnet18_viewpoint.pth")
    torch.save({
        "model_state_dict": export_model.state_dict(),
        "head_state_dict": head.state_dict(),
        "class_names": CLASS_NAMES,
        "view_map": VIEW_MAP,
        "samples_per_class": samples_per_class,
        "val_acc": val_acc
    }, save_path)
    volume.commit()
    print(f"✓ Saved checkpoint to {save_path} and committed volume!")

    return {
        "status": "success",
        "feat_time_s": feat_time,
        "train_time_s": train_time,
        "final_val_acc": val_acc,
        "checkpoint_path": save_path
    }


@app.function(
    volumes={VOLUME_DIR: volume},
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=7200,
)
def train_full_directional_classifier(
    git_commit_sha: str = "",
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 2026,
    splits_dir: str = "/root/moo_splits",
    checkpoint_name: str = "moo_resnet18_viewpoint8_full.pth",
):
    """Full fine-tuning of ImageNet-pretrained ResNet-18 on the canonical 8-direction MOO synthetic viewpoint split."""
    if not git_commit_sha or not git_commit_sha.strip():
        raise ValueError(
            "Explicit 'git_commit_sha' is required (e.g. --git-commit-sha <SHA>) "
            "to guarantee scientific provenance. Empty or missing SHA is rejected."
        )
    git_commit_sha = git_commit_sha.strip()
    print(f"Git Commit SHA: {git_commit_sha}")
    import io
    import time
    import copy
    import hashlib
    import random
    import numpy as np
    import pandas as pd
    import h5py
    from PIL import Image
    from tqdm import tqdm
    from sklearn.metrics import f1_score

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    import torchvision.models as models
    from torch.cuda.amp import GradScaler, autocast

    CANONICAL_VIEWPOINT_CLASSES = [
        "front",
        "front-left",
        "left",
        "back-left",
        "back",
        "back-right",
        "right",
        "front-right",
    ]
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CANONICAL_VIEWPOINT_CLASSES)}

    EXPECTED_SPLIT_HASHES = {
        "train.csv": "596a1a49c985223202bdf04b6d2b20d6544ab01d7683827fe65b70f6fc521e61",
        "val.csv": "7995c1e357cc33ccc17c9d70f0a210d1035839bfdd4ff26d7179201fdf42b261",
        "test.csv": "276603b9589f66d6d66f19140889d2e5ca369e69493815a23b47e13fb0ab4a8d",
    }

    # Locate splits directory
    resolved_splits_dir = None
    candidate_dirs = [
        splits_dir,
        "/root/moo_splits",
        os.path.join(VOLUME_DIR, "moo_splits"),
        os.path.join(VOLUME_DIR, "splits"),
        str(SPLITS_DIR_LOCAL),
    ]
    for c in candidate_dirs:
        if c and os.path.exists(os.path.join(c, "train.csv")):
            resolved_splits_dir = c
            break

    if resolved_splits_dir is None:
        raise FileNotFoundError(
            f"Could not locate canonical MOO splits directory. Checked: {candidate_dirs}"
        )

    print(f"=== MOO Full 8-Direction Viewpoint Training (ResNet-18) ===")
    print(f"Using splits directory: {resolved_splits_dir}")

    # 1. Cryptographic hash validation
    def compute_sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    computed_hashes = {}
    for fname, expected_hash in EXPECTED_SPLIT_HASHES.items():
        fpath = os.path.join(resolved_splits_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required split file {fname} not found in {resolved_splits_dir}")
        chash = compute_sha256(fpath)
        computed_hashes[fname] = chash
        if chash != expected_hash:
            raise ValueError(
                f"Provenance validation failure: SHA256 mismatch for {fname} in {resolved_splits_dir}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Computed: {chash}\n"
                f"Aborting training to prevent unvalidated data execution."
            )
        print(f"  ✓ {fname}: SHA256 verified ({chash[:16]}...)")

    # 2. Parse and validate split contents
    train_df = pd.read_csv(os.path.join(resolved_splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(resolved_splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(resolved_splits_dir, "test.csv"))

    if len(train_df) != 76800:
        raise ValueError(f"Expected 76,800 train rows, found {len(train_df)}")
    if len(val_df) != 9600:
        raise ValueError(f"Expected 9,600 val rows, found {len(val_df)}")
    if len(test_df) != 9600:
        raise ValueError(f"Expected 9,600 test rows, found {len(test_df)}")

    train_cows = set(train_df["cow_id"])
    val_cows = set(val_df["cow_id"])
    test_cows = set(test_df["cow_id"])

    if len(train_cows) != 800:
        raise ValueError(f"Expected 800 train cows, found {len(train_cows)}")
    if len(val_cows) != 100:
        raise ValueError(f"Expected 100 val cows, found {len(val_cows)}")
    if len(test_cows) != 100:
        raise ValueError(f"Expected 100 test cows, found {len(test_cows)}")

    # Leakage checks
    train_val_cow_overlap = len(train_cows & val_cows)
    train_test_cow_overlap = len(train_cows & test_cows)
    val_test_cow_overlap = len(val_cows & test_cows)
    if train_val_cow_overlap > 0 or train_test_cow_overlap > 0 or val_test_cow_overlap > 0:
        raise ValueError(
            f"Cow identity leakage detected! "
            f"train∩val={train_val_cow_overlap}, train∩test={train_test_cow_overlap}, val∩test={val_test_cow_overlap}"
        )

    train_imgs = set(zip(train_df["cow_id"], train_df["image_id"]))
    val_imgs = set(zip(val_df["cow_id"], val_df["image_id"]))
    test_imgs = set(zip(test_df["cow_id"], test_df["image_id"]))

    train_val_img_overlap = len(train_imgs & val_imgs)
    train_test_img_overlap = len(train_imgs & test_imgs)
    val_test_img_overlap = len(val_imgs & test_imgs)
    if train_val_img_overlap > 0 or train_test_img_overlap > 0 or val_test_img_overlap > 0:
        raise ValueError(
            f"Image leakage detected! "
            f"train∩val={train_val_img_overlap}, train∩test={train_test_img_overlap}, val∩test={val_test_img_overlap}"
        )

    # Class label validation
    allowed_labels = set(CANONICAL_VIEWPOINT_CLASSES)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        present_labels = set(df["view"].unique())
        invalid_labels = present_labels - allowed_labels
        if invalid_labels:
            raise ValueError(f"Invalid / unmapped labels in {split_name}: {invalid_labels}")

    print(f"  ✓ Split integrity verified: 800 train cows (76,800 imgs), 100 val cows (9,600 imgs), 100 test cows (9,600 imgs)")
    print(f"  ✓ Zero cow identity overlap, zero image overlap, all 8 canonical classes verified.")

    # 3. Deterministic seed
    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    hdf5_path = os.path.join(VOLUME_DIR, "data.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"data.hdf5 not found in {VOLUME_DIR}. Run download_moo first!")

    # 4. PyTorch Dataset Definition
    class MOODataset(Dataset):
        def __init__(self, df, h5_path, cls_to_idx, transform=None):
            self.cow_ids = df["cow_id"].values
            self.image_ids = df["image_id"].values
            self.labels = [cls_to_idx[v] for v in df["view"].values]
            self.h5_path = h5_path
            self.transform = transform
            self.hf = None

        def __len__(self):
            return len(self.cow_ids)

        def __getitem__(self, idx):
            if self.hf is None:
                self.hf = h5py.File(self.h5_path, "r")
            cow_id = self.cow_ids[idx]
            img_id = self.image_ids[idx]
            label = self.labels[idx]

            if cow_id not in self.hf:
                raise KeyError(f"Cow ID '{cow_id}' not found in HDF5!")

            grp = self.hf[cow_id]
            if img_id in grp:
                obj = grp[img_id]
            elif f"{cow_id}_{img_id}" in grp:
                obj = grp[f"{cow_id}_{img_id}"]
            elif "_" in img_id and img_id.split("_")[-1] in grp:
                obj = grp[img_id.split("_")[-1]]
            else:
                available_sample = list(grp.keys())[:5]
                raise KeyError(
                    f"Could not resolve image '{img_id}' for cow '{cow_id}' in HDF5! "
                    f"Tried: '{img_id}', '{cow_id}_{img_id}', '{img_id.split('_')[-1]}'. "
                    f"Available keys (first 5 of {len(grp)}): {available_sample}"
                )

            if isinstance(obj, h5py.Group):
                if "colors" in obj:
                    raw = obj["colors"][()]
                elif "image" in obj:
                    raw = obj["image"][()]
                elif "rgb" in obj:
                    raw = obj["rgb"][()]
                elif "data" in obj:
                    raw = obj["data"][()]
                else:
                    available_fields = list(obj.keys())
                    raise KeyError(
                        f"Could not find valid image dataset ('colors', 'image', 'rgb', 'data') "
                        f"in HDF5 group for cow '{cow_id}', image '{img_id}'! Available fields: {available_fields}"
                    )
            else:
                raw = obj[()]

            if isinstance(raw, (bytes, bytearray, np.void)) or (hasattr(raw, "dtype") and raw.dtype == np.uint8 and raw.ndim == 1):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                img = Image.fromarray(raw).convert("RGB")

            if self.transform:
                img = self.transform(img)

            return img, label

        def __del__(self):
            if hasattr(self, "hf") and self.hf is not None:
                try:
                    self.hf.close()
                except Exception:
                    pass

    # Viewpoint-preserving transforms (Strictly NO horizontal flip)
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MOODataset(train_df, hdf5_path, CLASS_TO_IDX, transform=train_transform)
    val_ds = MOODataset(val_df, hdf5_path, CLASS_TO_IDX, transform=eval_transform)
    test_ds = MOODataset(test_df, hdf5_path, CLASS_TO_IDX, transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 5. Model Architecture & Optimizer Setup
    print("\n--- Initializing ResNet-18 (ImageNet-1K Pretrained) for Full Fine-Tuning ---")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CANONICAL_VIEWPOINT_CLASSES))

    # Full fine-tuning: backbone and classification head both trainable
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # 6. Training Loop
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state_dict = None

    start_train_time = time.time()
    print(f"\n{'='*75}")
    print(f"Starting MOO Full 8-Direction Viewpoint Training on {device}")
    print(f"Config: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, WeightDecay={weight_decay}")
    print(f"Dataset: Train={len(train_ds):,} | Val={len(val_ds):,} | Test={len(test_ds):,}")
    print(f"{'='*75}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs:2d} [Train]", leave=True)
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            running_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total_train_samples += bs

            cur_loss = running_loss / total_train_samples
            cur_acc = running_correct / total_train_samples
            pbar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "acc": f"{cur_acc*100:.1f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        train_loss = running_loss / total_train_samples
        train_acc = running_correct / total_train_samples

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_samples_total = 0
        val_preds_all = []
        val_targets_all = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{epochs:2d} [Val]  ", leave=False)
        with torch.no_grad():
            for imgs, targets in val_pbar:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    v_loss = criterion(outputs, targets)

                val_loss_total += v_loss.item() * targets.size(0)
                val_samples_total += targets.size(0)
                preds = outputs.argmax(dim=1)

                val_preds_all.extend(preds.cpu().numpy())
                val_targets_all.extend(targets.cpu().numpy())

        val_loss = val_loss_total / val_samples_total
        val_acc = float((np.array(val_preds_all) == np.array(val_targets_all)).mean())
        val_f1 = float(f1_score(val_targets_all, val_preds_all, average="macro"))
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f} | LR: {current_lr:.2e}"
        )

        # Primary checkpoint selection metric: synthetic validation macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  --> [BEST CHECKPOINT] Updated best val Macro-F1: {best_val_f1:.4f} (Acc: {best_val_acc*100:.2f}%) at epoch {epoch}")

    # 7. Final Synthetic Test Evaluation (strictly once on best validation checkpoint)
    print(f"\n--- Evaluating Best Validation Checkpoint on Synthetic TEST Split ---")
    if best_model_state_dict is None:
        raise RuntimeError("No best checkpoint was captured during training.")

    print(f"Loading weights from Best Epoch {best_epoch} (Val F1: {best_val_f1:.4f}, Val Acc: {best_val_acc*100:.2f}%)...")
    model.load_state_dict(best_model_state_dict)
    model.eval()

    test_loss_total = 0.0
    test_samples_total = 0
    test_preds_all = []
    test_targets_all = []

    test_pbar = tqdm(test_loader, desc="Testing Best Model [Test]", leave=True)
    with torch.no_grad():
        for imgs, targets in test_pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                t_loss = criterion(outputs, targets)

            test_loss_total += t_loss.item() * targets.size(0)
            test_samples_total += targets.size(0)
            preds = outputs.argmax(dim=1)

            test_preds_all.extend(preds.cpu().numpy())
            test_targets_all.extend(targets.cpu().numpy())

    test_loss = test_loss_total / test_samples_total
    test_acc = float((np.array(test_preds_all) == np.array(test_targets_all)).mean())
    test_f1 = float(f1_score(test_targets_all, test_preds_all, average="macro"))

    total_runtime_s = time.time() - start_train_time

    print(f"\n{'='*75}")
    print(f"MOO Full 8-Direction Viewpoint Training Run Complete!")
    print(f"Total Runtime: {total_runtime_s/60:.2f} minutes ({total_runtime_s:.1f}s)")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"Synthetic Test Accuracy: {test_acc*100:.2f}%")
    print(f"Synthetic Test Macro-F1: {test_f1:.4f}")
    print(f"{'='*75}\n")

    # 8. Save Canonical Full Checkpoint to Modal Volume
    save_path = os.path.join(VOLUME_DIR, checkpoint_name)
    print(f"Saving canonical full checkpoint to {save_path}...")
    torch.save({
        "model_state_dict": best_model_state_dict,
        "class_names": CANONICAL_VIEWPOINT_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "seed": seed,
        "epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "optimizer_configuration": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "scheduler_configuration": {
            "type": "CosineAnnealingLR",
            "T_max": epochs,
            "eta_min": 1e-6,
        },
        "train_val_test_csv_sha256": computed_hashes,
        "git_commit_sha": git_commit_sha,
    }, save_path)
    volume.commit()
    print(f"✓ Full checkpoint committed to volume: {save_path}")

    return {
        "status": "success",
        "checkpoint_path": save_path,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "total_runtime_s": total_runtime_s,
    }


if __name__ == "__main__":
    print("This script is a Modal App.")
    print("Run via Modal CLI:")
    print("  modal run --profile tigerwood693 scripts/modal_moo_pipeline.py::download_moo")
    print("  modal run --profile tigerwood693 scripts/modal_moo_pipeline.py::inspect_moo")
    print("  modal run --profile tigerwood693 scripts/modal_moo_pipeline.py::train_smoke_classifier")
    print("  modal run --profile tigerwood693 scripts/modal_moo_pipeline.py::train_full_directional_classifier --git-commit-sha <SHA>")
