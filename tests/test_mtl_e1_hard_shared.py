# -*- coding: utf-8 -*-
"""
tests/test_mtl_e1_hard_shared.py — Unit & Smoke Verification for Run 7 E1 Hard-Shared MTL
========================================================================================
Verifies:
  1. Exact E1 Hard-Shared Model Architecture & Parameter Counts (11,926,706 params).
  2. Proof that exactly ONE ResNet-18 visual trunk exists and is shared across all 3 tasks.
  3. Conv1 4th mask-channel initialization (mean of ImageNet RGB weights).
  4. Task forward shapes:
     - BCS      : [B, 4, 224, 224]   -> [B, 4] ordinal logits
     - Behavior : [B, 8, 4, 224, 224]-> [B, 5] behavior logits + [B, 8, 512] frame features
     - Re-ID    : [B, 4, 224, 224]   -> [B, 41] identity logits + [B, 512] L2 embeddings
  5. Gradient accumulation: proves all 3 tasks backpropagate into the SAME backbone weights.
  6. Checkpoint save & reload bit-identity.
  7. Protocol disjointness: asserts 0 overlap between 41 train cows and 69 held-out eval cows.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_mtl_e1_hard_shared import (
    MTLE1HardSharedModel,
    ResNet18SharedBackbone,
    BCSOrdinalHead,
    BehaviorTemporalTCN,
    ReIDHead,
    ordinal_targets_from_class_indices,
    logits_to_bcs_predictions,
    evaluate_bcs_metrics,
    evaluate_behavior_metrics,
    evaluate_reid_metrics,
    NUM_BCS_CLASSES,
    NUM_BEHAVIOR_CLASSES,
    NUM_REID_TRAIN_COWS,
)


class TestMTLE1HardShared(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_parameter_counts_and_structure(self):
        """Verifies certified parameter counts and absence of duplicate trunks."""
        model = MTLE1HardSharedModel(pretrained=False)
        counts = model.count_parameters()

        self.assertEqual(counts["shared_backbone_parameters"], 11179648)
        self.assertEqual(counts["bcs_head_parameters"], 2052)
        self.assertEqual(counts["behavior_tcn_parameters"], 723973)
        self.assertEqual(counts["reid_head_parameters"], 21033)
        self.assertEqual(counts["total_trainable_parameters"], 11926706)

        # Programmatic assertion
        self.assertTrue(model.assert_hard_sharing())

    def test_02_conv1_mask_initialization(self):
        """Verifies that 4th channel is initialized from the mean of RGB weights."""
        model = MTLE1HardSharedModel(pretrained=True)
        conv1_w = model.backbone.conv1.weight.data

        rgb_mean = conv1_w[:, :3, :, :].mean(dim=1, keepdim=True)
        mask_w = conv1_w[:, 3:4, :, :]

        max_diff = (rgb_mean - mask_w).abs().max().item()
        self.assertAlmostEqual(max_diff, 0.0, places=5)

    def test_03_forward_shapes(self):
        """Verifies exact tensor shapes for all 3 tasks."""
        model = MTLE1HardSharedModel(pretrained=False).to(self.device)
        model.eval()

        B = 4
        # 1. BCS
        x_bcs = torch.randn(B, 4, 224, 224, device=self.device)
        logits_bcs = model.forward_bcs(x_bcs)
        self.assertEqual(logits_bcs.shape, (B, 4))

        # 2. Behavior
        x_beh = torch.randn(B, 8, 4, 224, 224, device=self.device)
        logits_beh, feats_beh = model.forward_behavior(x_beh)
        self.assertEqual(logits_beh.shape, (B, 5))
        self.assertEqual(feats_beh.shape, (B, 8, 512))

        # 3. Re-ID
        x_reid = torch.randn(B, 4, 224, 224, device=self.device)
        logits_reid, emb_reid = model.forward_reid(x_reid)
        self.assertEqual(logits_reid.shape, (B, 41))
        self.assertEqual(emb_reid.shape, (B, 512))

        # L2 norm of embedding
        norms = torch.norm(emb_reid, p=2, dim=1)
        for n in norms.detach().cpu().numpy():
            self.assertAlmostEqual(float(n), 1.0, places=5)

    def test_04_gradient_accumulation_into_shared_backbone(self):
        """Proves that all three tasks backpropagate into the EXACT same ResNet-18 weights."""
        model = MTLE1HardSharedModel(pretrained=False).to(self.device)
        model.train()

        crit_bcs = nn.BCEWithLogitsLoss()
        crit_beh = nn.CrossEntropyLoss()
        crit_reid = nn.CrossEntropyLoss()

        B = 2
        x_bcs = torch.randn(B, 4, 224, 224, device=self.device)
        t_bcs = torch.randint(0, 5, (B,), device=self.device)
        t_bcs_ord = ordinal_targets_from_class_indices(t_bcs, num_classes=5, device=self.device)

        x_beh = torch.randn(B, 8, 4, 224, 224, device=self.device)
        t_beh = torch.randint(0, 5, (B,), device=self.device)

        x_reid = torch.randn(B, 4, 224, 224, device=self.device)
        t_reid = torch.randint(0, 41, (B,), device=self.device)

        # Step 1: BCS backward
        model.zero_grad()
        loss_bcs = crit_bcs(model.forward_bcs(x_bcs), t_bcs_ord)
        loss_bcs.backward()
        grad_bcs = model.backbone.conv1.weight.grad.clone()
        self.assertTrue(torch.isfinite(grad_bcs).all())
        self.assertTrue((grad_bcs.abs() > 0).any())

        # Step 2: Behavior backward (accumulating)
        loss_beh = crit_beh(model.forward_behavior(x_beh)[0], t_beh)
        loss_beh.backward()
        grad_bcs_beh = model.backbone.conv1.weight.grad.clone()
        self.assertFalse(torch.allclose(grad_bcs, grad_bcs_beh))

        # Step 3: Re-ID backward (accumulating)
        loss_reid = crit_reid(model.forward_reid(x_reid)[0], t_reid)
        loss_reid.backward()
        grad_all = model.backbone.conv1.weight.grad.clone()
        self.assertFalse(torch.allclose(grad_bcs_beh, grad_all))

        # Verify optimizer step works
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.step()
        self.assertTrue(True)

    def test_05_checkpoint_save_reload_bit_identity(self):
        """Verifies checkpoint save/reload produces identical logits."""
        model = MTLE1HardSharedModel(pretrained=False).to(self.device)
        model.eval()

        probe_x = torch.randn(2, 4, 224, 224, device=self.device)
        with torch.no_grad():
            orig_logits = model.forward_bcs(probe_x)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            torch.save({"model_state_dict": model.state_dict()}, tmp_path)

            reload_model = MTLE1HardSharedModel(pretrained=False).to(self.device)
            ckpt = torch.load(tmp_path, map_location=self.device, weights_only=False)
            reload_model.load_state_dict(ckpt["model_state_dict"])
            reload_model.eval()

            with torch.no_grad():
                reload_logits = reload_model.forward_bcs(probe_x)

            diff = (orig_logits - reload_logits).abs().max().item()
            self.assertEqual(diff, 0.0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_06_protocol_disjointness(self):
        """Verifies 41 train cows and 69 held-out evaluation cows have 0 overlap."""
        protocols_dir = REPO_ROOT / "datasets" / "id" / "sideviewcows2026"
        if not (protocols_dir / "protocol_cross_setting.csv").exists():
            self.skipTest("SideView protocol CSVs not present locally.")

        df_a = pd.read_csv(protocols_dir / "protocol_cross_setting.csv")
        train_cows = set(df_a.loc[df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())
        eval_cows = set(df_a.loc[~df_a["setting_role"].eq("train"), "individual_id"].astype(str).unique())

        self.assertEqual(len(train_cows), 41)
        self.assertEqual(len(eval_cows), 69)
        overlap = train_cows.intersection(eval_cows)
        self.assertEqual(len(overlap), 0, f"Overlapping cows found: {overlap}")


if __name__ == "__main__":
    unittest.main()
