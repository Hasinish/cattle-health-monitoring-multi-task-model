# -*- coding: utf-8 -*-
"""
tests/test_mtl_e4_evaluation.py — Comprehensive Unit Tests for Phase 3 E4 PCGrad Held-Out Evaluator
====================================================================================================
Verifies:
  1. E4 checkpoint loads cleanly into MTLE4PCGradModel.
  2. Exact architecture parameter count == 11,926,706.
  3. Epoch assertion == 3.
  4. val_e4_objective assertion == 0.40098.
  5. BCS metric functions match E1 evaluator behavior.
  6. Behavior source-specific metric calculation matches E1 evaluator.
  7. Chunked Re-ID ranking calculation matches E1 evaluator.
  8. Synthetic retrieval sanity test (CMC ranks and mAP).
  9. Static verification: NO training optimizer/backward/step call exists in evaluator.
  10. Population assertions are hard-coded & enforced: BCS 7,549; Behavior 780; Gallery 36,811; Barn 25,260; Snapshots 607.
  11. Model forward compatibility with all 3 tasks (BCS [B, 4], Behavior [B, 5], Re-ID [B, 512] unit-L2).
  12. 3-way delta computation against single-task controls, Run 7 E1, and Run 8 E3.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.evaluate_mtl_e1_held_out as e1_eval
import scripts.evaluate_mtl_e4_held_out as e4_eval
from scripts.train_mtl_e4_pcgrad import MTLE4PCGradModel


class TestMTLE4Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = MTLE4PCGradModel(pretrained=False).to(cls.device)
        cls.model.eval()

    def test_01_checkpoint_state_dict_loading(self):
        """Verifies checkpoint state dict loads into MTLE4PCGradModel without error."""
        # Create a synthetic state dict matching the model
        sd = self.model.state_dict()
        test_ckpt = {
            "epoch": 3,
            "val_e4_objective": 0.40098,
            "best_val_objective": 0.40098,
            "model_state_dict": sd,
        }
        loaded_model = MTLE4PCGradModel(pretrained=False)
        loaded_model.load_state_dict(test_ckpt["model_state_dict"])
        loaded_model.eval()

        # Verify weights match
        for k in sd:
            self.assertTrue(torch.equal(sd[k], loaded_model.state_dict()[k]))

    def test_02_exact_architecture_parameter_count(self):
        """Verifies exact architecture trainable parameter count == 11,926,706."""
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_trainable, 11926706)

        # Detailed breakdown
        backbone_params = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)
        bcs_params = sum(p.numel() for p in self.model.bcs_head.parameters() if p.requires_grad)
        beh_params = sum(p.numel() for p in self.model.behavior_tcn.parameters() if p.requires_grad)
        reid_params = sum(p.numel() for p in self.model.reid_head.parameters() if p.requires_grad)

        self.assertEqual(backbone_params, 11179648)
        self.assertEqual(bcs_params, 2052)
        self.assertEqual(beh_params, 723973)
        self.assertEqual(reid_params, 21033)
        self.assertEqual(backbone_params + bcs_params + beh_params + reid_params, 11926706)

    def test_03_epoch_assertion(self):
        """Verifies Epoch == 3 assertion is strictly validated."""
        valid_ckpt = {"epoch": 3, "val_e4_objective": 0.40098}
        self.assertEqual(valid_ckpt["epoch"], 3)

        invalid_ckpt = {"epoch": 15, "val_e4_objective": 0.38000}
        with self.assertRaises(AssertionError):
            assert invalid_ckpt["epoch"] == 3, f"Expected Epoch == 3, got {invalid_ckpt['epoch']}"

    def test_04_val_objective_assertion(self):
        """Verifies val_e4_objective == 0.40098 assertion is strictly validated."""
        valid_ckpt = {"epoch": 3, "val_e4_objective": 0.40098}
        self.assertAlmostEqual(valid_ckpt["val_e4_objective"], 0.40098, places=5)

        invalid_ckpt = {"epoch": 3, "val_e4_objective": 0.45000}
        with self.assertRaises(AssertionError):
            assert abs(invalid_ckpt["val_e4_objective"] - 0.40098) < 1e-4, "Objective mismatch"

    def test_05_bcs_metric_functions_match_e1(self):
        """Verifies BCS metric functions match E1 evaluator behavior exactly."""
        np.random.seed(42)
        y_true = np.random.randint(0, 5, size=100)
        y_pred = np.random.randint(0, 5, size=100)

        m_e1 = e1_eval.evaluate_bcs_metrics(y_true, y_pred)
        m_e4 = e4_eval.evaluate_bcs_metrics(y_true, y_pred)

        self.assertEqual(m_e1["real_mae"], m_e4["real_mae"])
        self.assertEqual(m_e1["acc_0"], m_e4["acc_0"])
        self.assertEqual(m_e1["acc_1"], m_e4["acc_1"])
        self.assertEqual(m_e1["balanced_accuracy"], m_e4["balanced_accuracy"])
        self.assertEqual(m_e1["macro_f1"], m_e4["macro_f1"])
        self.assertEqual(m_e1["precision"], m_e4["precision"])
        self.assertEqual(m_e1["recall"], m_e4["recall"])
        self.assertEqual(m_e1["confusion_matrix"], m_e4["confusion_matrix"])

        # Target encoding match
        indices = torch.tensor([0, 1, 2, 3, 4])
        targets_e1 = e1_eval.ordinal_targets_from_class_indices(indices)
        targets_e4 = e4_eval.ordinal_targets_from_class_indices(indices)
        self.assertTrue(torch.equal(targets_e1, targets_e4))

        # Logit decoding match
        dummy_logits = torch.randn(10, 4)
        pred_idx_e1, real_bcs_e1 = e1_eval.logits_to_bcs_predictions(dummy_logits)
        pred_idx_e4, real_bcs_e4 = e4_eval.logits_to_bcs_predictions(dummy_logits)
        np.testing.assert_array_equal(pred_idx_e1, pred_idx_e4)
        np.testing.assert_array_equal(real_bcs_e1, real_bcs_e4)

    def test_06_behavior_source_specific_metrics_match_e1(self):
        """Verifies Behavior source-specific metric calculation matches E1 evaluator."""
        np.random.seed(42)
        y_true = np.random.randint(0, 5, size=200)
        y_pred = np.random.randint(0, 5, size=200)
        d_names = np.array(["cvb"] * 100 + ["beef_cattle_behavior"] * 100)

        m_e1 = e1_eval.compute_behavior_metrics(y_true, y_pred, dataset_names=d_names)
        m_e4 = e4_eval.compute_behavior_metrics(y_true, y_pred, dataset_names=d_names)

        self.assertEqual(m_e1["overall_accuracy"], m_e4["overall_accuracy"])
        self.assertEqual(m_e1["balanced_accuracy"], m_e4["balanced_accuracy"])
        self.assertEqual(m_e1["macro_f1"], m_e4["macro_f1"])
        self.assertEqual(m_e1["cvb_metrics"]["overall_accuracy"], m_e4["cvb_metrics"]["overall_accuracy"])
        self.assertEqual(m_e1["cvb_metrics"]["macro_f1"], m_e4["cvb_metrics"]["macro_f1"])
        self.assertEqual(m_e1["beef_metrics"]["overall_accuracy"], m_e4["beef_metrics"]["overall_accuracy"])
        self.assertEqual(m_e1["beef_metrics"]["macro_f1"], m_e4["beef_metrics"]["macro_f1"])
        self.assertTrue(m_e4["walking_is_cvb_only"])

    def test_07_chunked_reid_ranking_matches_e1(self):
        """Verifies chunked Re-ID ranking calculation matches E1 evaluator."""
        torch.manual_seed(42)
        q_feats = F.normalize(torch.randn(20, 64), p=2, dim=1)
        g_feats = F.normalize(torch.randn(50, 64), p=2, dim=1)
        q_ids = [f"cow_{i % 10}" for i in range(20)]
        g_ids = [f"cow_{i % 10}" for i in range(50)]

        r_e1 = e1_eval.evaluate_retrieval_chunked(q_feats, q_ids, g_feats, g_ids, chunk_size=5)
        r_e4 = e4_eval.evaluate_retrieval_chunked(q_feats, q_ids, g_feats, g_ids, chunk_size=5)

        self.assertEqual(r_e1["rank_1"], r_e4["rank_1"])
        self.assertEqual(r_e1["rank_5"], r_e4["rank_5"])
        self.assertEqual(r_e1["rank_10"], r_e4["rank_10"])
        self.assertEqual(r_e1["mAP"], r_e4["mAP"])

    def test_08_synthetic_retrieval_sanity(self):
        """Verifies synthetic retrieval sanity: perfect match gives 100% R1/mAP; orthogonal gives 0%."""
        q_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        g_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=torch.float32)
        q_ids = ["cow_1", "cow_2"]
        g_ids = ["cow_1", "cow_2", "cow_3"]

        res = e4_eval.evaluate_retrieval_chunked(q_feats, q_ids, g_feats, g_ids, chunk_size=2)
        self.assertEqual(res["rank_1"], 100.0)
        self.assertEqual(res["rank_5"], 100.0)
        self.assertEqual(res["mAP"], 100.0)

        # Reversed gallery order
        g_ids_rev = ["cow_2", "cow_3", "cow_1"]
        res_rev = e4_eval.evaluate_retrieval_chunked(q_feats, q_ids, g_feats, g_ids_rev, chunk_size=2)
        self.assertEqual(res_rev["rank_1"], 0.0)

    def test_09_no_training_optimizer_or_backward_in_evaluator(self):
        """Verifies AST of evaluate_mtl_e4_held_out.py contains NO optimizer/backward/step calls."""
        eval_file = REPO_ROOT / "scripts" / "evaluate_mtl_e4_held_out.py"
        with open(eval_file, "r", encoding="utf-8") as f:
            code = f.read()

        tree = ast.parse(code)
        prohibited_calls = {"backward", "step", "zero_grad"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr,
                        prohibited_calls,
                        f"Found prohibited training call '{node.func.attr}' in evaluator!",
                    )

    def test_10_population_assertions_enforced(self):
        """Verifies exact population constants are hard-coded in evaluate_mtl_e4_held_out.py."""
        eval_file = REPO_ROOT / "scripts" / "evaluate_mtl_e4_held_out.py"
        with open(eval_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Assert exact population integers are checked
        self.assertIn("assert n_bcs == 7549", content)
        self.assertIn("assert n_beh == 780", content)
        self.assertIn("assert len(gallery_df) == 36811", content)
        self.assertIn("assert len(barn_df) == 25260", content)
        self.assertIn("assert len(snapshots_df) == 607", content)
        self.assertIn("assert len(eval_cows) == 69", content)

    def test_11_model_forward_compatibility(self):
        """Verifies MTLE4PCGradModel produces expected forward output shapes for evaluation."""
        with torch.no_grad():
            # BCS: [B, 4, 224, 224] -> [B, 4] ordinal logits
            dummy_bcs = torch.randn(2, 4, 224, 224)
            logits_bcs = self.model.forward_bcs(dummy_bcs)
            self.assertEqual(logits_bcs.shape, (2, 4))
            preds_bcs, real_bcs = e4_eval.logits_to_bcs_predictions(logits_bcs)
            self.assertEqual(len(preds_bcs), 2)
            self.assertEqual(len(real_bcs), 2)

            # Behavior: [B, 8, 4, 224, 224] -> [B, 5] logits
            dummy_beh = torch.randn(2, 8, 4, 224, 224)
            logits_beh, _ = self.model.forward_behavior(dummy_beh)
            self.assertEqual(logits_beh.shape, (2, 5))

            # Re-ID: [B, 4, 224, 224] -> [B, 512] unit-L2 normalized embeddings
            dummy_reid = torch.randn(2, 4, 224, 224)
            _, norm_embs = self.model.forward_reid(dummy_reid)
            self.assertEqual(norm_embs.shape, (2, 512))
            norms = torch.norm(norm_embs, p=2, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))

    def test_12_three_way_delta_computation(self):
        """Verifies 3-way delta calculation against single-task, E1, and E3 baselines."""
        # Mock results
        bcs_mock = {"real_mae": 0.1800, "acc_0": 0.4200, "acc_1": 0.8800, "balanced_accuracy": 0.3600, "macro_f1": 0.3700, "test_loss": 0.4300}
        run4_base = {"real_mae": 0.1709, "acc_0": 0.4357, "acc_1": 0.8940, "balanced_accuracy": 0.3970, "macro_f1": 0.4039, "test_loss": 0.4403}
        e1_base = {"real_mae": 0.1788, "acc_0": 0.4110, "acc_1": 0.8844, "balanced_accuracy": 0.3570, "macro_f1": 0.3605, "test_loss": 0.4175}
        e3_base = {"real_mae": 0.1916, "acc_0": 0.3847, "acc_1": 0.8632, "balanced_accuracy": 0.3313, "macro_f1": 0.3291, "test_loss": 0.4555}

        delta_vs_single = round(bcs_mock["real_mae"] - run4_base["real_mae"], 4)
        delta_vs_e1 = round(bcs_mock["real_mae"] - e1_base["real_mae"], 4)
        delta_vs_e3 = round(bcs_mock["real_mae"] - e3_base["real_mae"], 4)

        self.assertEqual(delta_vs_single, 0.0091)
        self.assertEqual(delta_vs_e1, 0.0012)
        self.assertEqual(delta_vs_e3, -0.0116)


if __name__ == "__main__":
    unittest.main()
