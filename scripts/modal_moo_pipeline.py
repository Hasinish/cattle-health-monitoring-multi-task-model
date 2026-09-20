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

try:
    import modal
except ImportError:
    print("Error: modal package not found. Run: pip install modal")
    sys.exit(1)

# Persistent storage volume for MOO dataset
volume = modal.Volume.from_name("moo-data", create_if_missing=True)
VOLUME_DIR = "/data"

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

app = modal.App("moo-viewpoint-pipeline", image=image)

MOO_URL = "https://kalisteo.cea.fr/index.php/download/moo-dataset/?wpdmdl=4365"


@app.function(
    volumes={VOLUME_DIR: volume},
    timeout=3600,  # 1 hour max
    cpu=1.0,
    memory=2048,  # 2 GB RAM (minimal cost)
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
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 36_000_000_000:
        print(f"\n[1/2] Downloading MOO.zip using aria2c (16 parallel connections)...")
        start_time = time.time()
        cmd = [
            "aria2c",
            "-c",
            "-s", "16",
            "-x", "16",
            "-j", "16",
            "-k", "2M",
            "--check-certificate=false",
            "-d", VOLUME_DIR,
            "-o", "MOO.zip",
            "--summary-interval=5",
            MOO_URL
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"aria2c download failed with exit code {res.returncode}")
        
        elapsed = time.time() - start_time
        downloaded_gb = os.path.getsize(zip_path) / (1024 ** 3)
        speed_mb = (downloaded_gb * 1024) / max(1, elapsed)
        print(f"✓ Download complete in {elapsed:.1f}s ({speed_mb:.1f} MB/s avg)!")
        volume.commit()
    else:
        print(f"✓ MOO.zip already downloaded ({os.path.getsize(zip_path) / (1024**3):.2f} GB)")

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


if __name__ == "__main__":
    print("This script is a Modal App.")
    print("Run via Modal CLI:")
    print("  modal run --profile mohtasimahmedsamii scripts/modal_moo_pipeline.py::download_moo")
    print("  modal run --profile mohtasimahmedsamii scripts/modal_moo_pipeline.py::inspect_moo")
    print("  modal run --profile mohtasimahmedsamii scripts/modal_moo_pipeline.py::train_smoke_classifier")
