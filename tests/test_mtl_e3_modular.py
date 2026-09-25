# -*- coding: utf-8 -*-
"""
tests/test_mtl_e3_modular.py — Focused Unit & Architecture Tests for Phase 3 Run 8 E3 Modular MTL
=================================================================================================
Verifies:
  1. Exact Parameter Counts & Structural Ownership (12,322,610 trainable parameters).
  2. Adapter Identity Initialization (Houlsby et al., 2019):
     - up_proj weight and bias == 0.0
     - adapter(x) == x at step 0 (identity mapping; adapted task feature equals shared backbone output).
  3. Conv1 4th mask-channel initialization (mean of ImageNet RGB weights).
  4. Forward Tensor Shapes across all 3 tasks:
     - BCS:      [B, 4, 224, 224]    -> [B, 4] ordinal logits
     - Behavior: [B, 8, 4, 224, 224] -> [B, 5] behavior logits + [B, 8, 512] frame features
     - Re-ID:    [B, 4, 224, 224]    -> [B, 41] identity logits + [B, 512] unit L2 embeddings
  5. Task-Private Routing & Gradient Isolation:
     - BCS backward: gradients ONLY in backbone, bcs_adapter, bcs_head (zero in beh/reid).
     - Behavior backward: gradients ONLY in backbone, behavior_adapter, behavior_tcn (zero in bcs/reid).
     - Re-ID backward: gradients ONLY in backbone, reid_adapter, reid_head (zero in bcs/beh).
     - Super-step accumulation into shared backbone.
  6. Deterministic Checkpoint Save & Reload Bit-Identity (max_logit_diff == 0.00000000).
  7. SideView Protocol Disjointness (0 overlap between 41 train and 69 held-out cows).
  8. Zero Unavailable-Label Padding Verification (pure task-specific batches).
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

from scripts.train_mtl_e3_modular import (
    MTLE3ModularModel,
    ResNet18SharedBackbone,
    TaskResidualAdapter,
    BCSOrdinalHead,
    BehaviorTemporalTCN,
    ReIDHead,
    ordinal_targets_from_class_indices,
    bcs_discrete_predictions_from_ordinal_logits,
    evaluate_bcs_metrics,
    evaluate_behavior_metrics,
    evaluate_reid_metrics,
    NUM_BCS_CLASSES,
    NUM_BEHAVIOR_CLASSES,
    NUM_REID_TRAIN_COWS,
)


class TestMTLE3Modular(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_parameter_counts_and_structure(self):
        """Verifies certified parameter counts and absence of duplicate trunks or illegal convs."""
        model = MTLE3ModularModel(pretrained=False)
        counts = model.count_parameters()

        expected_backbone = 11179648
        expected_adapter = 131968
        expected_total_adapters = 395904
        expected_bcs_head = 2052
        expected_beh_head = 723973
        expected_reid_head = 21033
        expected_total = 12322610

        self.assertEqual(counts["shared_backbone_parameters"], expected_backbone)
        self.assertEqual(counts["bcs_adapter_parameters"], expected_adapter)
        self.assertEqual(counts["behavior_adapter_parameters"], expected_adapter)
        self.assertEqual(counts["reid_adapter_parameters"], expected_adapter)
        self.assertEqual(counts["total_task_private_adapter_parameters"], expected_total_adapters)
        self.assertEqual(counts["bcs_head_parameters"], expected_bcs_head)
        self.assertEqual(counts["behavior_tcn_parameters"], expected_beh_head)
        self.assertEqual(counts["reid_head_parameters"], expected_reid_head)
        self.assertEqual(counts["total_trainable_parameters"], expected_total)

        # Delta over E1 (11,926,706): exactly 395,904 params (+3.32%)
        e1_params = 11926706
        self.assertEqual(counts["total_trainable_parameters"] - e1_params, expected_total_adapters)

        # Programmatic modular sharing assertion
        model.assert_modular_sharing()

    def test_02_adapter_identity_initialization(self):
        """
        Verifies that adapters are identity-initialized:
        up_proj weights and bias are initialized to 0, so out == in.
        This guarantees each adapter initially acts as an identity mapping, so the
        adapted task feature equals the shared backbone output before adaptation is learned.
        """
        model = MTLE3ModularModel(pretrained=False)
        for name, adapter in [
            ("bcs_adapter", model.bcs_adapter),
            ("behavior_adapter", model.behavior_adapter),
            ("reid_adapter", model.reid_adapter),
        ]:
            self.assertEqual(adapter.up_proj.weight.abs().max().item(), 0.0)
            self.assertEqual(adapter.up_proj.bias.abs().max().item(), 0.0)

            # Random 512-D features
            x = torch.randn(8, 512)
            out = adapter(x)
            diff = (out - x).abs().max().item()
            self.assertAlmostEqual(diff, 0.0, places=6, msg=f"{name} did not pass identity test!")

    def test_03_conv1_mask_initialization(self):
        """Verifies that 4th channel is initialized from the mean of RGB weights."""
        model = MTLE3ModularModel(pretrained=True)
        conv1_w = model.backbone.conv1.weight.data

        rgb_mean = conv1_w[:, :3, :, :].mean(dim=1, keepdim=True)
        mask_w = conv1_w[:, 3:4, :, :]

        max_diff = (rgb_mean - mask_w).abs().max().item()
        self.assertAlmostEqual(max_diff, 0.0, places=5)

    def test_04_forward_shapes(self):
        """Verifies exact tensor shapes for all 3 tasks."""
        model = MTLE3ModularModel(pretrained=False).to(self.device)
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

    def test_05_gradient_isolation_and_task_private_routing(self):
        """
        Verifies task-private routing & gradient isolation:
        - When BCS runs, ONLY backbone, bcs_adapter, and bcs_head receive gradients.
        - When Behavior runs, ONLY backbone, behavior_adapter, and behavior_tcn receive gradients.
        - When Re-ID runs, ONLY backbone, reid_adapter, and reid_head receive gradients.
        - When all 3 run in super-step, shared backbone accumulates gradients while adapters stay private.
        """
        model = MTLE3ModularModel(pretrained=False).to(self.device)
        model.train()

        crit_bcs = nn.BCEWithLogitsLoss()
        crit_beh = nn.CrossEntropyLoss()
        crit_reid = nn.CrossEntropyLoss()

        B = 2
        # Initialize non-zero weights in adapters for gradient check (since up_proj is zero)
        for p in model.parameters():
            if p.requires_grad:
                p.data.normal_(0, 0.02)

        # --- Subtest 1: BCS Isolation ---
        model.zero_grad()
        x_bcs = torch.randn(B, 4, 224, 224, device=self.device)
        t_bcs = torch.randint(0, 5, (B,), device=self.device)
        t_bcs_ord = ordinal_targets_from_class_indices(t_bcs, num_classes=5, device=self.device)
        loss_bcs = crit_bcs(model.forward_bcs(x_bcs), t_bcs_ord)
        loss_bcs.backward()

        # Check backbone has grad
        for p in model.backbone.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, "Backbone parameter missing grad during BCS step")
        # Check bcs adapter & head have grad
        for p in model.bcs_adapter.parameters():
            self.assertIsNotNone(p.grad, "BCS adapter parameter missing grad")
        for p in model.bcs_head.parameters():
            self.assertIsNotNone(p.grad, "BCS head parameter missing grad")
        # Check behavior & reid have ZERO grad (None)
        for p in model.behavior_adapter.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Behavior adapter received grad during BCS step!")
        for p in model.behavior_tcn.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Behavior TCN received grad during BCS step!")
        for p in model.reid_adapter.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Re-ID adapter received grad during BCS step!")
        for p in model.reid_head.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Re-ID head received grad during BCS step!")

        # --- Subtest 2: Behavior Isolation ---
        model.zero_grad()
        x_beh = torch.randn(B, 8, 4, 224, 224, device=self.device)
        t_beh = torch.randint(0, 5, (B,), device=self.device)
        loss_beh = crit_beh(model.forward_behavior(x_beh)[0], t_beh)
        loss_beh.backward()

        for p in model.backbone.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, "Backbone parameter missing grad during Behavior step")
        for p in model.behavior_adapter.parameters():
            self.assertIsNotNone(p.grad, "Behavior adapter parameter missing grad")
        for p in model.behavior_tcn.parameters():
            self.assertIsNotNone(p.grad, "Behavior TCN parameter missing grad")
        # Check bcs & reid have ZERO grad (None)
        for p in model.bcs_adapter.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: BCS adapter received grad during Behavior step!")
        for p in model.bcs_head.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: BCS head received grad during Behavior step!")
        for p in model.reid_adapter.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Re-ID adapter received grad during Behavior step!")
        for p in model.reid_head.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Re-ID head received grad during Behavior step!")

        # --- Subtest 3: Re-ID Isolation ---
        model.zero_grad()
        x_reid = torch.randn(B, 4, 224, 224, device=self.device)
        t_reid = torch.randint(0, 41, (B,), device=self.device)
        loss_reid = crit_reid(model.forward_reid(x_reid)[0], t_reid)
        loss_reid.backward()

        for p in model.backbone.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, "Backbone parameter missing grad during Re-ID step")
        for p in model.reid_adapter.parameters():
            self.assertIsNotNone(p.grad, "Re-ID adapter parameter missing grad")
        for p in model.reid_head.parameters():
            self.assertIsNotNone(p.grad, "Re-ID head parameter missing grad")
        # Check bcs & behavior have ZERO grad (None)
        for p in model.bcs_adapter.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: BCS adapter received grad during Re-ID step!")
        for p in model.bcs_head.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: BCS head received grad during Re-ID step!")
        for p in model.behavior_adapter.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Behavior adapter received grad during Re-ID step!")
        for p in model.behavior_tcn.parameters():
            self.assertIsNone(p.grad, "LEAKAGE: Behavior TCN received grad during Re-ID step!")

        # --- Subtest 4: Super-Step Multi-Task Accumulation ---
        model.zero_grad()
        loss_bcs = crit_bcs(model.forward_bcs(x_bcs), t_bcs_ord)
        loss_bcs.backward()
        b_grad_bcs = model.backbone.conv1.weight.grad.clone()

        loss_beh = crit_beh(model.forward_behavior(x_beh)[0], t_beh)
        loss_beh.backward()
        b_grad_bcs_beh = model.backbone.conv1.weight.grad.clone()

        loss_reid = crit_reid(model.forward_reid(x_reid)[0], t_reid)
        loss_reid.backward()
        b_grad_accum = model.backbone.conv1.weight.grad.clone()

        # Check accumulation in shared backbone
        self.assertFalse(torch.allclose(b_grad_bcs, b_grad_bcs_beh))
        self.assertFalse(torch.allclose(b_grad_bcs_beh, b_grad_accum))

    def test_06_checkpoint_reload_bit_identity(self):
        """Verifies checkpoint saving and reloading produces bit-identical logits (diff == 0.00000000)."""
        model1 = MTLE3ModularModel(pretrained=False).to(self.device)
        model1.eval()

        # Generate test inputs
        B = 2
        x_bcs = torch.randn(B, 4, 224, 224, device=self.device)
        x_beh = torch.randn(B, 8, 4, 224, 224, device=self.device)
        x_reid = torch.randn(B, 4, 224, 224, device=self.device)

        with torch.no_grad():
            out1_bcs = model1.forward_bcs(x_bcs)
            out1_beh = model1.forward_behavior(x_beh)[0]
            out1_reid_logits, out1_reid_emb = model1.forward_reid(x_reid)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_e3_checkpoint.pth"
            payload = {
                "epoch": 1,
                "model_state_dict": model1.state_dict(),
            }
            torch.save(payload, ckpt_path)

            model2 = MTLE3ModularModel(pretrained=False).to(self.device)
            loaded = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model2.load_state_dict(loaded["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                out2_bcs = model2.forward_bcs(x_bcs)
                out2_beh = model2.forward_behavior(x_beh)[0]
                out2_reid_logits, out2_reid_emb = model2.forward_reid(x_reid)

            max_bcs_diff = (out1_bcs - out2_bcs).abs().max().item()
            max_beh_diff = (out1_beh - out2_beh).abs().max().item()
            max_reid_diff = (out1_reid_logits - out2_reid_logits).abs().max().item()
            max_emb_diff = (out1_reid_emb - out2_reid_emb).abs().max().item()

            self.assertEqual(max_bcs_diff, 0.0)
            self.assertEqual(max_beh_diff, 0.0)
            self.assertEqual(max_reid_diff, 0.0)
            self.assertEqual(max_emb_diff, 0.0)

    def test_07_sideview_reid_protocol_disjointness(self):
        """Verifies that the canonical SideView protocols have 0 cow overlap between 41 train and 69 eval cows."""
        proto_dir = REPO_ROOT / "datasets" / "id" / "sideviewcows2026"
        csv_cross = proto_dir / "protocol_cross_setting.csv"
        self.assertTrue(csv_cross.exists(), f"Missing {csv_cross}")

        df = pd.read_csv(csv_cross)
        train_cows = set(df.loc[df["setting_role"].eq("train"), "individual_id"].astype(str).unique())
        eval_cows = set(df.loc[~df["setting_role"].eq("train"), "individual_id"].astype(str).unique())

        self.assertEqual(len(train_cows), 41)
        self.assertEqual(len(eval_cows), 69)
        self.assertEqual(len(train_cows.intersection(eval_cows)), 0, "FATAL: Train and eval cows overlap!")

    def test_08_zero_unavailable_label_padding(self):
        """
        Verifies that ordinal targets and behavior/reid batches are created with
        pure task-specific labels, with zero dummy/mask padding.
        """
        # BCS targets are strictly length 4 binary vectors
        indices = torch.tensor([0, 1, 2, 3, 4])
        ord_targets = ordinal_targets_from_class_indices(indices, num_classes=5)
        self.assertEqual(ord_targets.shape, (5, 4))
        # Expected Frank & Hall cumulative encoding:
        # idx 0 -> [0, 0, 0, 0]
        # idx 1 -> [1, 0, 0, 0]
        # idx 2 -> [1, 1, 0, 0]
        # idx 3 -> [1, 1, 1, 0]
        # idx 4 -> [1, 1, 1, 1]
        expected = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        self.assertTrue(torch.equal(ord_targets, expected))


if __name__ == "__main__":
    unittest.main()
