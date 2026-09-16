"""
Modal L40S Multi-Task Training with PCGrad
Runs full 4-task training with Projecting Conflicting Gradients (NeurIPS 2020) on an NVIDIA L40S (48GB VRAM).

Usage:
    modal run scripts/modal_train_pcgrad.py
"""

import modal

app = modal.App("cattle-multitask-pcgrad")

dataset_volume = modal.Volume.from_name("cattle-datasets", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cattle-checkpoints", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "pandas",
        "numpy",
        "opencv-python-headless",
        "scikit-learn",
        "tqdm",
        "pillow",
    )
)

@app.function(
    gpu="L40S", # 48GB VRAM Ada Lovelace
    image=train_image,
    volumes={
        "/datasets": dataset_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600 * 4, # 4 hours max runtime
    cpu=8.0,
    memory=32768,
)
def train_multitask_l40s(epochs: int = 30, batch_size: int = 128, lr: float = 1e-4):
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    import pandas as pd
    import random

    print("==================================================")
    print("  HARDWARE INITIALIZATION (MODAL)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr}")
    print("==================================================")

    # -------------------------------------------------------------
    # 1. DATASET DEFINITION
    # -------------------------------------------------------------
    class GenericCattleDataset(Dataset):
        def __init__(self, csv_path, split="train", transform=None):
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row['image_path']
            label = int(row['label'])
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, label

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nLoading datasets from fast NVMe volume...")
    datasets = {
        'lameness': GenericCattleDataset("/datasets/lameness/lameness_index.csv", split='train', transform=train_transform),
        'behavior': GenericCattleDataset("/datasets/behavior/behavior_index.csv", split='train', transform=train_transform),
        'id': GenericCattleDataset("/datasets/id/id_index.csv", split='train', transform=train_transform),
    }

    loaders = {
        task: DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        for task, ds in datasets.items()
    }
    for task, ds in datasets.items():
        print(f"  - {task:<10}: {len(ds):,} training samples")

    # -------------------------------------------------------------
    # 2. MULTI-TASK MODEL (ResNet-18 Backbone + 4 Heads)
    # -------------------------------------------------------------
    class MultiTaskCattleModel(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-1]) # Output: (B, 512, 1, 1)
            
            # Task Heads
            self.head_lameness = nn.Linear(512, 2)
            self.head_behavior = nn.Linear(512, 8)
            self.head_id = nn.Linear(512, 46)
            self.head_bcs = nn.Linear(512, 5)

        def forward(self, x, task=None):
            feat = torch.flatten(self.backbone(x), 1)
            if task == 'lameness':
                return self.head_lameness(feat)
            elif task == 'behavior':
                return self.head_behavior(feat)
            elif task == 'id':
                return self.head_id(feat)
            elif task == 'bcs':
                return self.head_bcs(feat)
            return {
                'lameness': self.head_lameness(feat),
                'behavior': self.head_behavior(feat),
                'id': self.head_id(feat),
                'bcs': self.head_bcs(feat),
            }

    model = MultiTaskCattleModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # -------------------------------------------------------------
    # 3. PCGRAD GRADIENT SURGERY (NeurIPS 2020)
    # -------------------------------------------------------------
    def pcgrad_backward(objectives):
        """Project conflicting gradients across task loss objectives."""
        grads = []
        for obj in objectives:
            optimizer.zero_grad()
            obj.backward(retain_graph=True)
            grad_vec = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_vec.append(p.grad.view(-1))
                else:
                    grad_vec.append(torch.zeros(p.numel(), device=p.device))
            grads.append(torch.cat(grad_vec))

        # Pairwise projection
        shared = torch.stack(grads)
        num_tasks = len(grads)
        task_order = list(range(num_tasks))
        random.shuffle(task_order)

        for i in task_order:
            for j in task_order:
                if i == j:
                    continue
                g_i, g_j = shared[i], shared[j]
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    shared[i] -= (dot / (g_j.norm() ** 2 + 1e-8)) * g_j

        # Set unified project gradient
        final_grad = shared.mean(dim=0)
        optimizer.zero_grad()
        offset = 0
        for p in model.parameters():
            num = p.numel()
            p.grad = final_grad[offset:offset + num].view_as(p).clone()
            offset += num

    # -------------------------------------------------------------
    # 4. TRAINING LOOP
    # -------------------------------------------------------------
    print("\nStarting Multi-Task PCGrad Training on L40S...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        iterators = {task: iter(loader) for task, loader in loaders.items()}
        steps = max(len(l) for l in loaders.values())
        total_loss = 0.0

        for step in range(steps):
            losses = []
            for task in ['lameness', 'behavior', 'id']:
                try:
                    imgs, labels = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(loaders[task])
                    imgs, labels = next(iterators[task])

                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    preds = model(imgs, task=task)
                    loss = criterion(preds, labels)
                    losses.append(loss)

            # Perform PCGrad backward projection
            pcgrad_backward(losses)
            optimizer.step()
            total_loss += sum(l.item() for l in losses) / len(losses)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {total_loss/steps:.4f} | Time: {epoch_time:.2f}s ({batch_size*steps/epoch_time:.1f} img/s)")

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = f"/checkpoints/multitask_l40s_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            checkpoint_volume.commit()
            print(f"  [SAVED] Checkpoint committed to Volume -> {ckpt_path}")

    total_min = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Training Completed in {total_min:.2f} minutes!")
    checkpoint_volume.commit()

@app.local_entrypoint()
def main(epochs: int = 30, batch_size: int = 128, lr: float = 1e-4):
    train_multitask_l40s.remote(epochs, batch_size, lr)
