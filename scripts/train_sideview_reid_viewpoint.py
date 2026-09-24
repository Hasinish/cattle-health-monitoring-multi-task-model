# -*- coding: utf-8 -*-
"""SideViewCows2026 Re-ID + Viewpoint Ablation Training Pipeline.

Scientific Ablation:
  Tests whether frozen certified real-cattle viewpoint priors improve
  the Run 6 GT-mask perception representation:

  Run 6 (Baseline):  4-channel ResNet-18 [R, G, B, Mask] (512-D) -> Linear(512, 41)
                     Total Trainable Parameters: 11,200,681

  Run 6 + Viewpoint: 4-channel ResNet-18 (512-D) + Frozen Viewpoint ResNet-18 (3-class probs)
                     -> Viewpoint MLP (3 -> 16-D) -> Fused Embedding (528-D) -> Linear(528, 41)
                     Total Trainable Parameters: 11,201,737 (+1,056 / +0.0094%)
                     Frozen Viewpoint Parameters: 11,178,051 (Zero Gradients)

Representations:
  - Visual: identical to Run 6 (GT-mask cow crop + 5% margin -> 224x224 -> [R, G, B, Mask]).
  - Viewpoint: frozen ResNet-18 (viewpoint_resnet18_real_best.pth) evaluated on the RGB channels
    of the exact same Run 6 crop -> softmax [p_front, p_side, p_rear] -> small MLP (3 -> 16).
  - Fusion: [visual_512, vp_16] -> 528-D -> unit L2 normalized embedding for retrieval.
  - Zero metadata leakage: no cow IDs, camera IDs, filenames, recording IDs, or subset IDs.

Protocols (Strict Isolation):
  - Training/Val: 41 Protocol-A representation-learning cows (Protocol D: 12,753 train / 2,683 val).
  - Held-out Evaluation: 69 Protocol-A cows (Parlor gallery, Barn queries, Snapshot queries).
    Strictly isolated and NEVER loaded during training or smoke testing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from scripts.train_sideview_reid_baseline import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        evaluate_retrieval_chunked,
        extract_dataset_embeddings,
        get_git_commit_sha,
        resolve_sideview_image_path,
    )
    from scripts.train_sideview_reid_perception import (
        ResNet18ReIDPerception,
        SideViewGTMaskReIDDataset,
        _mask_bbox_with_margin,
        _validate_pair_paths,
        _integrity_audit,
        build_contact_sheet,
    )
except ModuleNotFoundError:
    from train_sideview_reid_baseline import (  # type: ignore
        IMAGENET_MEAN,
        IMAGENET_STD,
        evaluate_retrieval_chunked,
        extract_dataset_embeddings,
        get_git_commit_sha,
        resolve_sideview_image_path,
    )
    from train_sideview_reid_perception import (  # type: ignore
        ResNet18ReIDPerception,
        SideViewGTMaskReIDDataset,
        _mask_bbox_with_margin,
        _validate_pair_paths,
        _integrity_audit,
        build_contact_sheet,
    )

VIEWPOINT_CLASSES = ["front", "side", "rear"]
VIEWPOINT_PROB_DIM = 3
VIEWPOINT_EMB_DIM = 16
VISUAL_DIM = 512
FUSED_DIM = VISUAL_DIM + VIEWPOINT_EMB_DIM  # 528


# ==============================================================================
# 1. VIEWPOINT MLP
# ==============================================================================
class ViewpointMLP(nn.Module):
    """Maps 3-class probability vector into a compact 16-D viewpoint embedding."""

    def __init__(self, in_features: int = VIEWPOINT_PROB_DIM, out_features: int = VIEWPOINT_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 2. RE-ID + VIEWPOINT ABLATION MODEL
# ==============================================================================
class ResNet18ReIDViewpointAblation(nn.Module):
    """Run 6 Perception Visual Trunk fused with frozen viewpoint representation."""

    def __init__(
        self,
        num_classes: int = 41,
        viewpoint_checkpoint_path: Optional[str] = None,
        viewpoint_emb_dim: int = VIEWPOINT_EMB_DIM,
        pretrained: bool = True,
    ):
        super().__init__()
        # 1. Visual Spatial Trunk (4-channel ResNet-18 identical to Run 6)
        self.visual_backbone = ResNet18ReIDPerception(num_classes=num_classes, pretrained=pretrained)
        self.visual_backbone.classifier = nn.Identity()  # Discard unused classifier
        self.visual_dim = VISUAL_DIM
        self.viewpoint_dim = viewpoint_emb_dim
        self.fused_dim = self.visual_dim + self.viewpoint_dim  # 512 + 16 = 528

        # 2. Frozen Certified Viewpoint Classifier (3-channel ResNet-18)
        self.viewpoint_model = models.resnet18()
        self.viewpoint_model.fc = nn.Linear(512, VIEWPOINT_PROB_DIM)

        if viewpoint_checkpoint_path is not None and Path(viewpoint_checkpoint_path).exists():
            ckpt = torch.load(viewpoint_checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.viewpoint_model.load_state_dict(state_dict)
            print(f"[OK] Loaded certified frozen viewpoint weights from {viewpoint_checkpoint_path}")
        else:
            print("[WARN] No viewpoint checkpoint provided at init (weights uninitialized)")

        # Strictly freeze viewpoint network
        self.viewpoint_model.eval()
        for param in self.viewpoint_model.parameters():
            param.requires_grad = False

        # 3. Viewpoint MLP
        self.viewpoint_mlp = ViewpointMLP(in_features=VIEWPOINT_PROB_DIM, out_features=viewpoint_emb_dim)

        # 4. Identity Classifier Head
        self.classifier = nn.Linear(self.fused_dim, num_classes)

    def train(self, mode: bool = True):
        """Keep viewpoint model permanently in eval mode to freeze BatchNorm."""
        super().train(mode)
        self.viewpoint_model.eval()
        return self

    def assert_frozen_viewpoint(self):
        """Programmatically assert zero gradients and requires_grad=False on viewpoint model."""
        for name, param in self.viewpoint_model.named_parameters():
            assert not param.requires_grad, f"Viewpoint param {name} has requires_grad=True!"
            assert param.grad is None, f"Viewpoint param {name} accumulated gradient: {param.grad}!"

    def extract_visual_feature(self, visual_input: torch.Tensor) -> torch.Tensor:
        """Extract 512-D spatial feature from 4-channel visual backbone."""
        x = self.visual_backbone.conv1(visual_input)
        x = self.visual_backbone.bn1(x)
        x = self.visual_backbone.relu(x)
        x = self.visual_backbone.maxpool(x)

        x = self.visual_backbone.layer1(x)
        x = self.visual_backbone.layer2(x)
        x = self.visual_backbone.layer3(x)
        x = self.visual_backbone.layer4(x)

        x = self.visual_backbone.avgpool(x)
        return torch.flatten(x, 1)  # [B, 512]

    def extract_features(self, visual_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract raw fused 528-D features and unit L2-normalized 528-D embeddings.

        Args:
            visual_input: [B, 4, 224, 224] (ImageNet-normalized RGB + GT binary mask)

        Returns:
            fused_raw: [B, 528]
            fused_norm: [B, 528] (unit L2-normalized)
        """
        # 1. Run frozen viewpoint classifier on RGB channels only (torch.no_grad)
        with torch.no_grad():
            rgb_input = visual_input[:, :3, :, :]
            vp_logits = self.viewpoint_model(rgb_input)
            vp_probs = F.softmax(vp_logits, dim=-1)  # [B, 3]

        # 2. Extract visual representation & viewpoint embedding
        f_vis = self.extract_visual_feature(visual_input)  # [B, 512]
        f_vp = self.viewpoint_mlp(vp_probs)               # [B, 16]

        # 3. Fuse & normalize
        fused_raw = torch.cat([f_vis, f_vp], dim=1)        # [B, 528]
        fused_norm = F.normalize(fused_raw, p=2, dim=1)    # [B, 528]
        return fused_raw, fused_norm

    def forward(self, visual_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits:     [B, 41] classification logits
            embeddings: [B, 528] unit L2-normalized embeddings for retrieval
        """
        fused_raw, fused_norm = self.extract_features(visual_input)
        logits = self.classifier(fused_raw)
        return logits, fused_norm


def get_parameter_counts() -> Dict[str, Any]:
    """Calculate exact parameter counts for Run 6 vs Run 6 + Viewpoint."""
    model = ResNet18ReIDViewpointAblation(num_classes=41, pretrained=False)
    vis_trainable = sum(p.numel() for p in model.visual_backbone.parameters() if p.requires_grad)
    vp_mlp_trainable = sum(p.numel() for p in model.viewpoint_mlp.parameters() if p.requires_grad)
    clf_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    vp_model_frozen = sum(p.numel() for p in model.viewpoint_model.parameters())

    total_trainable = vis_trainable + vp_mlp_trainable + clf_trainable
    run6_trainable = 11200681  # Run 6 perception baseline

    return {
        "run6_trainable_parameters": run6_trainable,
        "visual_backbone_parameters": vis_trainable,
        "viewpoint_mlp_parameters": vp_mlp_trainable,
        "identity_classifier_parameters": clf_trainable,
        "total_trainable_parameters": total_trainable,
        "added_trainable_parameters_vs_run6": total_trainable - run6_trainable,
        "added_trainable_percentage": round((total_trainable - run6_trainable) / run6_trainable * 100, 4),
        "frozen_viewpoint_parameters": vp_model_frozen,
        "total_model_parameters": total_trainable + vp_model_frozen,
    }


# ==============================================================================
# 3. VALIDATION EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate_validation(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for samples, targets, _ in loader:
        samples = samples.to(device)
        targets = targets.to(device)
        logits, _ = model(samples)
        loss = criterion(logits, targets)

        batch_count = samples.size(0)
        val_count += batch_count
        val_loss_sum += float(loss.item()) * batch_count
        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_predictions.extend(predictions)
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)
    return {
        "val_loss": round(val_loss_sum / val_count, 4),
        "val_top1_acc": round(100.0 * float(accuracy_score(y_true, y_pred)), 2),
        "val_balanced_acc": round(100.0 * float(balanced_accuracy_score(y_true, y_pred)), 2),
        "val_macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


# ==============================================================================
# 4. MAIN TRAINING & SMOKE ROUTINE
# ==============================================================================
def train_sideview_reid_viewpoint(
    data_root: Path,
    protocols_dir: Path,
    viewpoint_checkpoint_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 2026,
    smoke: bool = False,
    smoke_samples: int = 64,
    device_str: str = "cuda",
    num_workers: int = 4,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    start_wall_time = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SIDEVIEWCOWS2026 RE-ID + VIEWPOINT ABLATION PIPELINE")
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Viewpoint Checkpoint : {viewpoint_checkpoint_path}")
    print(f"Mode                 : {'SMOKE TEST' if smoke else 'FULL ABLATION TRAINING'}")
    print(f"Epochs               : {epochs} | Batch Size: {batch_size} | Seed: {seed}")
    print("=" * 80)

    # 1. Protocols & Isolation Check
    proto_a_path = protocols_dir / "protocol_cross_setting.csv"
    proto_d_path = protocols_dir / "protocol_closed_set.csv"
    if not proto_a_path.exists() or not proto_d_path.exists():
        raise FileNotFoundError(f"Missing protocol files: {proto_a_path} or {proto_d_path}")

    df_a = pd.read_csv(proto_a_path)
    df_d = pd.read_csv(proto_d_path)

    train_cows = sorted(df_a[df_a["setting_role"].eq("train")]["individual_id"].astype(str).unique())
    evaluation_cows = sorted(df_a[df_a["setting_role"].ne("train")]["individual_id"].astype(str).unique())

    if len(train_cows) != 41 or len(evaluation_cows) != 69:
        raise AssertionError(f"Expected 41 train and 69 eval cows, got {len(train_cows)} and {len(evaluation_cows)}")
    if set(train_cows).intersection(set(evaluation_cows)):
        raise AssertionError("Train and held-out evaluation cows overlap detected!")

    df_d_41 = df_d[df_d["individual_id"].astype(str).isin(train_cows)].copy()
    df_train_full = df_d_41[df_d_41["closed_set_split"].eq("train")].reset_index(drop=True)
    df_val_full = df_d_41[df_d_41["closed_set_split"].eq("val")].reset_index(drop=True)

    if len(df_train_full) != 12753 or len(df_val_full) != 2683:
        raise AssertionError(f"Protocol D split counts mismatch: train={len(df_train_full)}, val={len(df_val_full)}")

    if smoke:
        df_train = df_train_full.sample(n=min(smoke_samples, len(df_train_full)), random_state=seed).reset_index(drop=True)
        df_val = df_val_full.sample(n=min(smoke_samples, len(df_val_full)), random_state=seed).reset_index(drop=True)
    else:
        df_train = df_train_full
        df_val = df_val_full

    accessed_cows = set(df_train["individual_id"].astype(str)).union(df_val["individual_id"].astype(str))
    if not accessed_cows.issubset(set(train_cows)):
        raise AssertionError("Dataset contains a held-out Protocol A evaluation cow!")

    cow_to_label = {cow: idx for idx, cow in enumerate(train_cows)}
    train_dataset = SideViewGTMaskReIDDataset(df_train, data_root, cow_to_label, augment=True)
    val_dataset = SideViewGTMaskReIDDataset(df_val, data_root, cow_to_label, augment=False)

    print(f"[*] Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    loader_kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "drop_last": False}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # 2. Model Initialization
    model = ResNet18ReIDViewpointAblation(
        num_classes=41,
        viewpoint_checkpoint_path=str(viewpoint_checkpoint_path),
        pretrained=True,
    ).to(device)

    counts = get_parameter_counts()
    print("\n" + "=" * 60)
    print("PARAMETER COUNT VERIFICATION:")
    print(f"  Visual Backbone (Trainable)    : {counts['visual_backbone_parameters']:,}")
    print(f"  Viewpoint MLP (Trainable)      : {counts['viewpoint_mlp_parameters']:,}")
    print(f"  Identity Head (Trainable)      : {counts['identity_classifier_parameters']:,}")
    print(f"  Total Trainable Parameters     : {counts['total_trainable_parameters']:,}")
    print(f"  Run 6 Trainable Parameters     : {counts['run6_trainable_parameters']:,}")
    print(f"  Added Parameters vs Run 6      : +{counts['added_trainable_parameters_vs_run6']:,} ({counts['added_trainable_percentage']}%)")
    print(f"  Frozen Viewpoint Network       : {counts['frozen_viewpoint_parameters']:,} (requires_grad=False)")
    print(f"  Total Model Parameters         : {counts['total_model_parameters']:,}")
    print("=" * 60 + "\n")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 3. Shape & Gradient Checks
    shape_checks = {}
    history: List[Dict[str, Any]] = []
    best_val_metric = -1.0
    best_epoch = -1

    # Grab fixed batch for determinism test
    fixed_samples, fixed_targets, _ = next(iter(val_loader))
    fixed_samples = fixed_samples.to(device)

    print("[*] Launching training loop...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", file=sys.stdout, mininterval=1.0)
        for samples, targets, _ in pbar:
            samples = samples.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            fused_raw, embeddings = model.extract_features(samples)
            logits = model.classifier(fused_raw)

            # Tensor shape and norm checks on first batch
            if not shape_checks:
                norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
                assert samples.shape[1:] == (4, 224, 224), f"Bad sample shape: {samples.shape}"
                assert fused_raw.shape == (samples.shape[0], FUSED_DIM), f"Bad raw shape: {fused_raw.shape}"
                assert embeddings.shape == (samples.shape[0], FUSED_DIM), f"Bad emb shape: {embeddings.shape}"
                assert logits.shape == (samples.shape[0], 41), f"Bad logit shape: {logits.shape}"
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "L2 norms != 1.0"
                shape_checks = {
                    "input_shape": list(samples.shape),
                    "fused_raw_shape": list(fused_raw.shape),
                    "embedding_shape": list(embeddings.shape),
                    "logits_shape": list(logits.shape),
                    "embedding_norm_min": float(norms.min().item()),
                    "embedding_norm_max": float(norms.max().item()),
                }

            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")

            loss.backward()

            # ASSERT ZERO GRADIENTS IN VIEWPOINT MODEL
            model.assert_frozen_viewpoint()

            # Assert trainable parameters have finite gradients
            for p in trainable_params:
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), "Non-finite gradient in trainable param!"

            optimizer.step()

            batch_count = samples.size(0)
            train_count += batch_count
            train_loss_sum += float(loss.item()) * batch_count
            train_correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
            pbar.set_postfix(
                loss=f"{train_loss_sum / train_count:.4f}",
                acc=f"{100 * train_correct / train_count:.1f}%",
            )

        scheduler.step()
        val_metrics = evaluate_validation(model, val_loader, criterion, device)
        epoch_rec = {
            "epoch": epoch,
            "train_loss": round(train_loss_sum / train_count, 4),
            "train_top1_acc": round(100.0 * train_correct / train_count, 2),
            **val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
            "duration_sec": round(time.perf_counter() - epoch_start, 2),
        }
        history.append(epoch_rec)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_rec['train_loss']:.4f} | "
              f"Val Loss: {epoch_rec['val_loss']:.4f} | Val Top-1: {epoch_rec['val_top1_acc']:.2f}% | "
              f"Val Macro-F1: {epoch_rec['val_macro_f1']:.4f}")

        # Save Best
        if val_metrics["val_top1_acc"] > best_val_metric:
            best_val_metric = val_metrics["val_top1_acc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "parameter_counts": counts,
                    "seed": seed,
                },
                output_dir / "reid_viewpoint_best.pth",
            )

        # Save Latest
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val_metric": best_val_metric,
                "parameter_counts": counts,
                "seed": seed,
            },
            output_dir / "reid_viewpoint_latest.pth",
        )

    # 4. Checkpoint Reload Determinism Test
    print("\n[*] Verifying bit-identical checkpoint reload...")
    model.eval()
    with torch.no_grad():
        logits_before, _ = model(fixed_samples)

    reloaded_model = ResNet18ReIDViewpointAblation(
        num_classes=41,
        viewpoint_checkpoint_path=str(viewpoint_checkpoint_path),
        pretrained=False,
    ).to(device)
    latest_ckpt = torch.load(output_dir / "reid_viewpoint_latest.pth", map_location=device, weights_only=False)
    reloaded_model.load_state_dict(latest_ckpt["model_state_dict"])
    reloaded_model.eval()
    with torch.no_grad():
        logits_after, _ = reloaded_model(fixed_samples)

    max_logit_diff = float((logits_before - logits_after).abs().max().item())
    bit_identical = bool(torch.equal(logits_before, logits_after))
    print(f"  Max Logit Difference : {max_logit_diff:.8f}")
    print(f"  Bit-Identical Reload  : {bit_identical}")
    assert max_logit_diff == 0.0, f"Checkpoint reload produced non-zero logit difference: {max_logit_diff}"

    # 5. Protocol A Held-Out Isolation Check
    held_out_images_loaded = 0
    retrieval_results: Dict[str, Any]
    if smoke:
        retrieval_results = {
            "status": "NOT_LOADED_OR_EVALUATED_IN_SMOKE_MODE",
            "protocol_a_held_out_evaluated": False,
            "held_out_images_loaded": 0,
        }
        print("[*] Smoke mode: Held-out Protocol A evaluation cows STRICTLY UNTOUCHED (0 images loaded).")
    else:
        # Full evaluation on held-out 69 cows
        print("\n[*] Running canonical Protocol A retrieval evaluation on 69 held-out cows...")
        best_ckpt = torch.load(output_dir / "reid_viewpoint_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        gallery_df = df_a[df_a["setting_role"].eq("gallery")].reset_index(drop=True)
        barn_df = df_a[df_a["setting_role"].eq("query_barn")].reset_index(drop=True)
        snapshots_df = df_a[df_a["setting_role"].eq("query_snapshots")].reset_index(drop=True)
        held_out_images_loaded = len(gallery_df) + len(barn_df) + len(snapshots_df)

        eval_loaders = []
        for frame in (gallery_df, barn_df, snapshots_df):
            ds = SideViewGTMaskReIDDataset(frame, data_root, {c: 0 for c in evaluation_cows}, augment=False)
            eval_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers))

        gallery_feats, gallery_ids = extract_dataset_embeddings(model, eval_loaders[0], device, desc="Gallery Parlor")
        barn_feats, barn_ids = extract_dataset_embeddings(model, eval_loaders[1], device, desc="Query Barn")
        snap_feats, snap_ids = extract_dataset_embeddings(model, eval_loaders[2], device, desc="Query Snapshots")

        retrieval_results = {
            "query_barn": evaluate_retrieval_chunked(barn_feats, barn_ids, gallery_feats, gallery_ids),
            "query_snapshots": evaluate_retrieval_chunked(snap_feats, snap_ids, gallery_feats, gallery_ids, chunk_size=607),
        }

    # 6. Save Final Metrics
    metrics = {
        "task": "sideviewcows2026_reid_viewpoint_ablation",
        "smoke": smoke,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_top1_acc": best_val_metric,
        "history": history,
        "shape_checks": shape_checks,
        "parameter_counts": counts,
        "checkpoint_reload": {
            "max_logit_difference": max_logit_diff,
            "bit_identical": bit_identical,
        },
        "retrieval_results": retrieval_results,
        "held_out_protocol_a_images_loaded": held_out_images_loaded,
        "total_elapsed_sec": round(time.perf_counter() - start_wall_time, 2),
    }

    metrics_path = output_dir / ("reid_viewpoint_smoke_metrics.json" if smoke else "reid_viewpoint_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Saved metrics to {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SideViewCows2026 Re-ID + Viewpoint Ablation")
    parser.add_argument("--data-root", type=str, default="/data/sideviewcows2026")
    parser.add_argument("--protocols-dir", type=str, default="/root/datasets/id/sideviewcows2026")
    parser.add_argument("--viewpoint-checkpoint-path", type=str, default="/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth")
    parser.add_argument("--output-dir", type=str, default="/checkpoints/sideview_reid_viewpoint_ablation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_sideview_reid_viewpoint(
        data_root=Path(args.data_root),
        protocols_dir=Path(args.protocols_dir),
        viewpoint_checkpoint_path=Path(args.viewpoint_checkpoint_path),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        smoke=args.smoke,
        smoke_samples=args.smoke_samples,
        device_str=args.device,
        num_workers=args.num_workers,
    )
