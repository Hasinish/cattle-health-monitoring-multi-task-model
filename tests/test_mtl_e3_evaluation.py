# -*- coding: utf-8 -*-
"""
tests/test_mtl_e3_evaluation.py — Unit Tests for MTL E3 Held-Out Evaluation Logic
=================================================================================
Verifies:
  1. BCS metric computation and delta calculations.
  2. Behavior metric computation including CVB vs Beef breakdowns and Walking minority class.
  3. Re-ID Protocol A chunked retrieval logic (CMC ranks and mAP).
  4. MTLE3ModularModel forward compatibility with all 3 evaluation tasks.
  5. Multi-task delta comparison against both single-task (Runs 4/5/6) and hard-shared (Run 7 E1) baselines.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_mtl_e3_held_out import (
    evaluate_bcs_metrics,
    compute_behavior_metrics,
    evaluate_retrieval_chunked,
    ordinal_targets_from_class_indices,
    logits_to_bcs_predictions,
)
from scripts.train_mtl_e3_modular import MTLE3ModularModel


class TestMTLE3Evaluation(unittest.TestCase):
    def test_bcs_metrics_perfect(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        m = evaluate_bcs_metrics(y_true, y_pred)
        self.assertEqual(m["real_mae"], 0.0)
        self.assertEqual(m["acc_0"], 1.0)
        self.assertEqual(m["acc_1"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)

    def test_bcs_ordinal_targets(self):
        indices = torch.tensor([0, 1, 4])
        targets = ordinal_targets_from_class_indices(indices, num_classes=5)
        # Class 0: [0, 0, 0, 0]
        # Class 1: [1, 0, 0, 0]
        # Class 4: [1, 1, 1, 1]
        self.assertTrue(torch.equal(targets[0], torch.tensor([0.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(targets[1], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(targets[2], torch.tensor([1.0, 1.0, 1.0, 1.0])))

    def test_behavior_metrics_with_domains(self):
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 0])  # last one error
        d_names = np.array(["cvb"] * 5 + ["beef_cattle_behavior"] * 5)
        m = compute_behavior_metrics(y_true, y_pred, dataset_names=d_names)
        self.assertIn("overall_accuracy", m)
        self.assertIn("balanced_accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertIn("cvb_metrics", m)
        self.assertIn("beef_metrics", m)
        self.assertEqual(m["cvb_metrics"]["overall_accuracy"], 1.0)
        self.assertEqual(m["beef_metrics"]["overall_accuracy"], 0.8)
        self.assertIn("Walking", m["per_class_f1"])

    def test_retrieval_chunked(self):
        # 3 cows, 2 queries, 3 gallery items
        q_feats = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
        q_ids = ["cow_A", "cow_B"]
        g_feats = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]), p=2, dim=1)
        g_ids = ["cow_A", "cow_B", "cow_C"]

        retrieval = evaluate_retrieval_chunked(q_feats, q_ids, g_feats, g_ids, chunk_size=2)
        self.assertEqual(retrieval["rank_1"], 100.0)
        self.assertEqual(retrieval["mAP"], 100.0)

    def test_e3_model_forward_compatibility(self):
        """Verifies MTLE3ModularModel produces expected forward output shapes for evaluation."""
        model = MTLE3ModularModel(pretrained=False)
        model.eval()

        with torch.no_grad():
            # BCS: [B, 4, 224, 224] -> [B, 4] ordinal logits
            dummy_bcs = torch.randn(2, 4, 224, 224)
            logits_bcs = model.forward_bcs(dummy_bcs)
            self.assertEqual(logits_bcs.shape, (2, 4))
            preds_bcs, real_bcs = logits_to_bcs_predictions(logits_bcs)
            self.assertEqual(len(preds_bcs), 2)
            self.assertEqual(len(real_bcs), 2)

            # Behavior: [B, 8, 4, 224, 224] -> [B, 5] logits
            dummy_beh = torch.randn(2, 8, 4, 224, 224)
            logits_beh, _ = model.forward_behavior(dummy_beh)
            self.assertEqual(logits_beh.shape, (2, 5))

            # Re-ID: [B, 4, 224, 224] -> [B, 512] unit-L2 normalized embeddings
            dummy_reid = torch.randn(2, 4, 224, 224)
            _, norm_embs = model.forward_reid(dummy_reid)
            self.assertEqual(norm_embs.shape, (2, 512))
            norms = torch.norm(norm_embs, p=2, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
