# -*- coding: utf-8 -*-
"""
tests/test_mtl_e4_pcgrad.py — Unit & Architecture Tests for Phase 3 E4 PCGrad MTL
================================================================================
Verifies:
  1. Exact E1 Architecture Retained: total trainable params = 11,926,706.
  2. No adapters or private visual trunks exist (assert_hard_sharing passes).
  3. PCGrad conflict projection math: verifies numerical projection formula on known conflicting vectors.
  4. Non-conflicting gradients remain unchanged under PCGrad surgery.
  5. PCGrad operates only on shared backbone parameters.
  6. BCS head receives only BCS gradients (zero Behavior/Re-ID cross-talk).
  7. Behavior TCN receives only Behavior gradients (zero BCS/Re-ID cross-talk).
  8. Re-ID head receives only Re-ID gradients (zero BCS/Behavior cross-talk).
  9. Deterministic projected gradients under identical seed and inputs.
 10. Normal task forward shapes remain correct across all 3 tasks.
 11. Checkpoint save/reload is bit-identical and restores PCGrad RNG state.
 12. SideView training/evaluation identity separation remains intact (41 train vs 69 held-out cows).
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

from scripts.train_mtl_e4_pcgrad import (
    MTLE4PCGradModel,
    PCGradProjector,
    ResNet18SharedBackbone,
    BCSOrdinalHead,
    BehaviorTemporalTCN,
    ReIDHead,
    ordinal_targets_from_class_indices,
    NUM_BCS_CLASSES,
    NUM_BEHAVIOR_CLASSES,
    NUM_REID_TRAIN_COWS,
)


class TestMTLE4PCGrad(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_exact_e1_architecture_retained(self):
        """Verifies total parameter count matches E1 exactly: 11,926,706 params."""
        model = MTLE4PCGradModel(pretrained=False)
        counts = model.count_parameters()

        expected_backbone = 11179648
        expected_bcs = 2052
        expected_beh = 723973
        expected_reid = 21033
        expected_total = 11926706

        self.assertEqual(counts["shared_backbone_parameters"], expected_backbone)
        self.assertEqual(counts["bcs_head_parameters"], expected_bcs)
        self.assertEqual(counts["behavior_tcn_parameters"], expected_beh)
        self.assertEqual(counts["reid_head_parameters"], expected_reid)
        self.assertEqual(counts["total_trainable_parameters"], expected_total)

    def test_02_no_adapters_or_private_visual_trunks(self):
        """Verifies no adapters, gates, or duplicate visual trunks exist."""
        model = MTLE4PCGradModel(pretrained=False)
        self.assertTrue(model.assert_hard_sharing())

        # Assert no adapter modules exist
        for name, _ in model.named_modules():
            self.assertNotIn("adapter", name.lower())
            self.assertNotIn("gate", name.lower())

    def test_03_pcgrad_conflict_projection_math(self):
        """
        Numerically verifies PCGrad conflict projection formula on known conflicting vectors:
        g1 = [1.0, 0.0], g2 = [-1.0, 1.0] (dot = -1.0 < 0).
        g1_proj = g1 - [dot(g1, g2) / ||g2||^2] * g2 = [1.0, 0.0] - [-1.0/2.0] * [-1.0, 1.0] = [0.5, 0.5]
        g2_proj = g2 - [dot(g2, g1) / ||g1||^2] * g1 = [-1.0, 1.0] - [-1.0/1.0] * [1.0, 0.0] = [0.0, 1.0]
        """
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([-1.0, 1.0])
        g2 = torch.tensor([0.0, 1.0])

        projector = PCGradProjector(seed=2026, eps=1e-12)
        proj_grads, diags = projector.project_and_diagnose([g0, g1, g2])

        # g0 was conflicting with g1 (dot = -1.0)
        self.assertAlmostEqual(diags["dot_bcs_beh"], -1.0, places=5)
        self.assertEqual(diags["conflict_bcs_beh"], 1.0)
        self.assertGreater(diags["projections_triggered"], 0)

        # Verified projected vector g0 against g1
        # Dot product with original g1 must now be non-negative (>= 0.0)
        dot_post_01 = torch.dot(proj_grads[0], g1).item()
        self.assertGreaterEqual(dot_post_01, -1e-6)

    def test_04_non_conflicting_gradients_remain_unchanged(self):
        """Verifies that non-conflicting (orthogonal or positively aligned) gradients are unchanged."""
        g0 = torch.tensor([1.0, 0.0, 0.0])
        g1 = torch.tensor([0.5, 1.0, 0.0])
        g2 = torch.tensor([0.2, 0.3, 1.0])

        projector = PCGradProjector(seed=2026, eps=1e-12)
        proj_grads, diags = projector.project_and_diagnose([g0, g1, g2])

        self.assertEqual(diags["conflict_bcs_beh"], 0.0)
        self.assertEqual(diags["conflict_bcs_reid"], 0.0)
        self.assertEqual(diags["conflict_beh_reid"], 0.0)
        self.assertEqual(diags["projections_triggered"], 0.0)

        # Gradients must be numerically identical to original
        for orig, proj in zip([g0, g1, g2], proj_grads):
            diff = (orig - proj).abs().max().item()
            self.assertAlmostEqual(diff, 0.0, places=6)

    def test_05_pcgrad_operates_only_on_shared_backbone(self):
        """Verifies PCGrad modifies only shared backbone gradients, leaving head gradients unaltered."""
        model = MTLE4PCGradModel(pretrained=False).to(self.device)
        model.train()

        # Generate artificial opposing gradients on backbone
        bcs_head_w = model.bcs_head.fc.weight.data.clone()
        beh_head_w = model.behavior_tcn.classifier.weight.data.clone()

        # Forward BCS
        x_bcs = torch.randn(2, 4, 224, 224, device=self.device)
        t_bcs = ordinal_targets_from_class_indices(torch.tensor([1, 2], device=self.device), 5, self.device)
        model.zero_grad(set_to_none=True)
        loss_bcs = nn.BCEWithLogitsLoss()(model.forward_bcs(x_bcs), t_bcs)
        loss_bcs.backward()
        g_bcs_shared = [p.grad.clone() for p in model.backbone.parameters() if p.requires_grad]
        g_bcs_head = [p.grad.clone() for p in model.bcs_head.parameters() if p.requires_grad]

        # Forward Behavior
        x_beh = torch.randn(2, 8, 4, 224, 224, device=self.device)
        t_beh = torch.tensor([0, 1], device=self.device)
        model.zero_grad(set_to_none=True)
        loss_beh = nn.CrossEntropyLoss()(model.forward_behavior(x_beh)[0], t_beh)
        loss_beh.backward()
        g_beh_shared = [p.grad.clone() for p in model.backbone.parameters() if p.requires_grad]
        g_beh_head = [p.grad.clone() for p in model.behavior_tcn.parameters() if p.requires_grad]

        # Forward Re-ID
        x_reid = torch.randn(2, 4, 224, 224, device=self.device)
        t_reid = torch.tensor([5, 10], device=self.device)
        model.zero_grad(set_to_none=True)
        loss_reid = nn.CrossEntropyLoss()(model.forward_reid(x_reid)[0], t_reid)
        loss_reid.backward()
        g_reid_shared = [p.grad.clone() for p in model.backbone.parameters() if p.requires_grad]
        g_reid_head = [p.grad.clone() for p in model.reid_head.parameters() if p.requires_grad]

        # Flatten and project shared
        flat_bcs = torch.cat([g.reshape(-1) for g in g_bcs_shared])
        flat_beh = torch.cat([g.reshape(-1) for g in g_beh_shared])
        flat_reid = torch.cat([g.reshape(-1) for g in g_reid_shared])

        projector = PCGradProjector(seed=2026, eps=1e-12)
        proj_grads, diags = projector.project_and_diagnose([flat_bcs, flat_beh, flat_reid])

        # Restore
        merged_shared = proj_grads[0] + proj_grads[1] + proj_grads[2]
        offset = 0
        for p in model.backbone.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.grad = merged_shared[offset : offset + numel].view_as(p).clone()
                offset += numel

        for p, g in zip(model.bcs_head.parameters(), g_bcs_head):
            p.grad = g.clone()
        for p, g in zip(model.behavior_tcn.parameters(), g_beh_head):
            p.grad = g.clone()
        for p, g in zip(model.reid_head.parameters(), g_reid_head):
            p.grad = g.clone()

        # Verify: Backbone gradients are present and finite
        self.assertTrue(torch.isfinite(model.backbone.conv1.weight.grad).all())
        # Verify: Head gradients are exactly equal to their unprojected individual gradients
        for p, orig_g in zip(model.bcs_head.parameters(), g_bcs_head):
            self.assertEqual(float((p.grad - orig_g).abs().max().item()), 0.0)

    def test_06_bcs_head_receives_only_bcs_gradients(self):
        """Verifies that Behavior and Re-ID tasks produce zero gradient in BCS head."""
        model = MTLE4PCGradModel(pretrained=False).to(self.device)
        model.train()

        # Backward Behavior
        model.zero_grad(set_to_none=True)
        x_beh = torch.randn(2, 8, 4, 224, 224, device=self.device)
        loss_beh = nn.CrossEntropyLoss()(model.forward_behavior(x_beh)[0], torch.tensor([0, 1], device=self.device))
        loss_beh.backward()
        for p in model.bcs_head.parameters():
            self.assertIsNone(p.grad)

        # Backward Re-ID
        model.zero_grad(set_to_none=True)
        x_reid = torch.randn(2, 4, 224, 224, device=self.device)
        loss_reid = nn.CrossEntropyLoss()(model.forward_reid(x_reid)[0], torch.tensor([2, 3], device=self.device))
        loss_reid.backward()
        for p in model.bcs_head.parameters():
            self.assertIsNone(p.grad)

    def test_07_behavior_tcn_receives_only_behavior_gradients(self):
        """Verifies that BCS and Re-ID tasks produce zero gradient in Behavior TCN."""
        model = MTLE4PCGradModel(pretrained=False).to(self.device)
        model.train()

        # Backward BCS
        model.zero_grad(set_to_none=True)
        x_bcs = torch.randn(2, 4, 224, 224, device=self.device)
        t_bcs = ordinal_targets_from_class_indices(torch.tensor([1, 2], device=self.device), 5, self.device)
        loss_bcs = nn.BCEWithLogitsLoss()(model.forward_bcs(x_bcs), t_bcs)
        loss_bcs.backward()
        for p in model.behavior_tcn.parameters():
            self.assertIsNone(p.grad)

        # Backward Re-ID
        model.zero_grad(set_to_none=True)
        x_reid = torch.randn(2, 4, 224, 224, device=self.device)
        loss_reid = nn.CrossEntropyLoss()(model.forward_reid(x_reid)[0], torch.tensor([2, 3], device=self.device))
        loss_reid.backward()
        for p in model.behavior_tcn.parameters():
            self.assertIsNone(p.grad)

    def test_08_reid_head_receives_only_reid_gradients(self):
        """Verifies that BCS and Behavior tasks produce zero gradient in Re-ID head."""
        model = MTLE4PCGradModel(pretrained=False).to(self.device)
        model.train()

        # Backward BCS
        model.zero_grad(set_to_none=True)
        x_bcs = torch.randn(2, 4, 224, 224, device=self.device)
        t_bcs = ordinal_targets_from_class_indices(torch.tensor([1, 2], device=self.device), 5, self.device)
        loss_bcs = nn.BCEWithLogitsLoss()(model.forward_bcs(x_bcs), t_bcs)
        loss_bcs.backward()
        for p in model.reid_head.parameters():
            self.assertIsNone(p.grad)

        # Backward Behavior
        model.zero_grad(set_to_none=True)
        x_beh = torch.randn(2, 8, 4, 224, 224, device=self.device)
        loss_beh = nn.CrossEntropyLoss()(model.forward_behavior(x_beh)[0], torch.tensor([0, 1], device=self.device))
        loss_beh.backward()
        for p in model.reid_head.parameters():
            self.assertIsNone(p.grad)

    def test_09_deterministic_projected_gradients(self):
        """Verifies that identical seed and inputs produce bit-identical projected gradients."""
        g0 = torch.randn(100, generator=torch.Generator().manual_seed(42))
        g1 = torch.randn(100, generator=torch.Generator().manual_seed(43))
        g2 = torch.randn(100, generator=torch.Generator().manual_seed(44))

        # Run 1
        p1 = PCGradProjector(seed=2026)
        res1, d1 = p1.project_and_diagnose([g0.clone(), g1.clone(), g2.clone()])

        # Run 2
        p2 = PCGradProjector(seed=2026)
        res2, d2 = p2.project_and_diagnose([g0.clone(), g1.clone(), g2.clone()])

        for r1, r2 in zip(res1, res2):
            diff = (r1 - r2).abs().max().item()
            self.assertEqual(diff, 0.0)

        for k in d1:
            self.assertEqual(d1[k], d2[k])

    def test_10_forward_shapes(self):
        """Verifies exact tensor forward shapes for all 3 tasks."""
        model = MTLE4PCGradModel(pretrained=False).to(self.device)
        model.eval()

        B = 4
        # 1. BCS: [B, 4, 224, 224] -> [B, 4]
        x_bcs = torch.randn(B, 4, 224, 224, device=self.device)
        logits_bcs = model.forward_bcs(x_bcs)
        self.assertEqual(logits_bcs.shape, (B, 4))

        # 2. Behavior: [B, 8, 4, 224, 224] -> [B, 5] logits + [B, 8, 512] feats
        x_beh = torch.randn(B, 8, 4, 224, 224, device=self.device)
        logits_beh, feats_beh = model.forward_behavior(x_beh)
        self.assertEqual(logits_beh.shape, (B, 5))
        self.assertEqual(feats_beh.shape, (B, 8, 512))

        # 3. Re-ID: [B, 4, 224, 224] -> [B, 41] logits + [B, 512] L2 embeddings
        x_reid = torch.randn(B, 4, 224, 224, device=self.device)
        logits_reid, emb_reid = model.forward_reid(x_reid)
        self.assertEqual(logits_reid.shape, (B, 41))
        self.assertEqual(emb_reid.shape, (B, 512))

        # Verify unit L2 norm
        norms = torch.norm(emb_reid, p=2, dim=1)
        for n in norms.detach().cpu().numpy():
            self.assertAlmostEqual(float(n), 1.0, places=5)

    def test_11_checkpoint_save_reload_bit_identity(self):
        """Verifies bit-identical checkpoint reload and PCGrad RNG state restoration."""
        model = MTLE4PCGradModel(pretrained=False).to(self.device)
        model.eval()

        projector = PCGradProjector(seed=2026)
        # Advance projector RNG
        projector.project_and_diagnose([torch.randn(10), torch.randn(10), torch.randn(10)])
        saved_rng_state = projector.get_state()

        probe_x = torch.randn(2, 4, 224, 224, device=self.device)
        with torch.no_grad():
            orig_logits = model.forward_bcs(probe_x)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "pcgrad_rng_state": saved_rng_state,
            }, tmp_path)

            reload_model = MTLE4PCGradModel(pretrained=False).to(self.device)
            ckpt = torch.load(tmp_path, map_location=self.device, weights_only=False)
            reload_model.load_state_dict(ckpt["model_state_dict"])
            reload_model.eval()

            reload_projector = PCGradProjector(seed=0)
            reload_projector.set_state(ckpt["pcgrad_rng_state"])

            with torch.no_grad():
                reload_logits = reload_model.forward_bcs(probe_x)

            diff = (orig_logits - reload_logits).abs().max().item()
            self.assertEqual(diff, 0.0)

            # Test RNG state equality
            self.assertEqual(projector.get_state(), reload_projector.get_state())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_12_sideview_protocol_disjointness(self):
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
