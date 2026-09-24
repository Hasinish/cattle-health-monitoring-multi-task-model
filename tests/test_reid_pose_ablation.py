# -*- coding: utf-8 -*-
"""Unit tests for SideViewCows2026 Re-ID + Pose Ablation Pipeline.

Tests:
1. Double-flip mathematical involution: flip(flip(v)) == v.
2. Anatomical left/right keypoint swapping and confidence/validity preservation.
3. Midline keypoint isolation (inversion of x only, no index swap).
4. Invalid keypoint stability (is_valid=0 keeps x=0, not 1.0).
5. Exact parameter counts (+31,360 params over Run 6).
6. Forward pass tensor shapes and unit-L2 embedding normalization.
7. Best-to-fresh checkpoint reload bit-identity determinism.
8. Full-mode Protocol A retrieval evaluation functions.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_sideview_reid_pose import (
    NUM_SUPERANIMAL_KEYPOINTS,
    POSE_EMBEDDING_DIM,
    POSE_FEATURE_DIM,
    FUSED_EMBEDDING_DIM,
    SUPERANIMAL_FLIP_PAIRS,
    flip_pose_vector,
    get_parameter_counts,
    ResNet18ReIDPoseAblation,
    extract_dataset_embeddings_pose,
)
from scripts.train_sideview_reid_baseline import evaluate_retrieval_chunked


class TestReIDPoseAblation(unittest.TestCase):
    """Test suite for Re-ID + SuperAnimal Pose ablation fixes."""

    def test_superanimal_flip_pairs_coverage(self):
        """Verify that exactly 13 pairs (26 keypoints) and 13 midline keypoints exist."""
        paired_indices = set()
        for idx_a, idx_b in SUPERANIMAL_FLIP_PAIRS:
            self.assertNotIn(idx_a, paired_indices, f"Duplicate index {idx_a}")
            self.assertNotIn(idx_b, paired_indices, f"Duplicate index {idx_b}")
            paired_indices.add(idx_a)
            paired_indices.add(idx_b)

        self.assertEqual(len(SUPERANIMAL_FLIP_PAIRS), 13)
        self.assertEqual(len(paired_indices), 26)

        midline_indices = set(range(NUM_SUPERANIMAL_KEYPOINTS)) - paired_indices
        self.assertEqual(len(midline_indices), 13)
        self.assertEqual(len(paired_indices.union(midline_indices)), 39)

    def test_pose_double_flip_involution(self):
        """Test that applying flip_pose_vector twice returns the exact original vector."""
        rng = np.random.RandomState(42)
        vec = rng.uniform(0.01, 0.99, size=(POSE_FEATURE_DIM,)).astype(np.float32)

        # Set some points as invalid
        kpts = vec.reshape(NUM_SUPERANIMAL_KEYPOINTS, 4)
        for i in [0, 4, 10, 24, 33]:
            kpts[i] = [0.0, 0.0, 0.0, 0.0]
        vec = kpts.reshape(POSE_FEATURE_DIM)

        flipped = flip_pose_vector(vec)
        double_flipped = flip_pose_vector(flipped)

        max_diff = np.max(np.abs(vec - double_flipped))
        self.assertLess(max_diff, 1e-6, f"Double-flip error too large: {max_diff}")
        self.assertTrue(np.allclose(vec, double_flipped, atol=1e-6))

    def test_pose_flip_anatomical_swap(self):
        """Verify left/right coordinates and confidences swap correctly."""
        vec = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        kpts = vec.reshape(NUM_SUPERANIMAL_KEYPOINTS, 4)

        # Keypoint 24 (front_left_thai) valid, Keypoint 27 (front_right_thai) invalid
        kpts[24] = [0.25, 0.60, 0.90, 1.0]
        kpts[27] = [0.0, 0.0, 0.0, 0.0]

        flipped_kpts = flip_pose_vector(vec).reshape(NUM_SUPERANIMAL_KEYPOINTS, 4)

        # 24 should now be zeroed
        self.assertEqual(flipped_kpts[24, 3], 0.0)
        self.assertEqual(flipped_kpts[24, 0], 0.0)

        # 27 should now have the flipped x and original y, conf, is_valid
        self.assertAlmostEqual(flipped_kpts[27, 0], 1.0 - 0.25, places=5)
        self.assertAlmostEqual(flipped_kpts[27, 1], 0.60, places=5)
        self.assertAlmostEqual(flipped_kpts[27, 2], 0.90, places=5)
        self.assertEqual(flipped_kpts[27, 3], 1.0)

    def test_pose_flip_midline_preservation(self):
        """Verify midline keypoints keep their index and only invert x."""
        vec = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        kpts = vec.reshape(NUM_SUPERANIMAL_KEYPOINTS, 4)

        # Keypoint 0 (nose) is midline
        kpts[0] = [0.35, 0.45, 0.85, 1.0]
        flipped_kpts = flip_pose_vector(vec).reshape(NUM_SUPERANIMAL_KEYPOINTS, 4)

        self.assertAlmostEqual(flipped_kpts[0, 0], 1.0 - 0.35, places=5)
        self.assertAlmostEqual(flipped_kpts[0, 1], 0.45, places=5)
        self.assertAlmostEqual(flipped_kpts[0, 2], 0.85, places=5)
        self.assertEqual(flipped_kpts[0, 3], 1.0)

    def test_pose_flip_invalid_points_remain_zero(self):
        """Verify invalid points (is_valid=0) are NOT inverted to x=1.0."""
        vec = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        flipped = flip_pose_vector(vec)
        self.assertTrue(np.all(flipped == 0.0))

    def test_parameter_counts(self):
        """Verify exact parameter counts match the specification (+31,360 params vs Run 6)."""
        counts = get_parameter_counts()
        self.assertEqual(counts["visual_backbone_trainable_parameters"], 11_179_648)
        self.assertEqual(counts["pose_mlp_trainable_parameters"], 28_736)
        self.assertEqual(counts["classifier_trainable_parameters"], 23_657)
        self.assertEqual(counts["total_trainable_parameters"], 11_232_041)

    def test_model_forward_shapes_and_norm(self):
        """Verify forward pass outputs correct tensor shapes and unit-L2 normalized embeddings."""
        model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False)
        model.eval()

        dummy_vis = torch.randn(4, 4, 224, 224)
        dummy_pose = torch.randn(4, POSE_FEATURE_DIM)

        logits, embs = model(dummy_vis, dummy_pose)
        self.assertEqual(logits.shape, (4, 41))
        self.assertEqual(embs.shape, (4, FUSED_EMBEDDING_DIM))

        # Check unit L2 norm
        norms = torch.norm(embs, p=2, dim=1).detach().numpy()
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_checkpoint_reload_determinism(self):
        """Verify that loading state_dict into reference and fresh models gives bit-identical outputs."""
        model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False)
        fresh_model = ResNet18ReIDPoseAblation(num_classes=41, pretrained=False)

        state = model.state_dict()
        model.load_state_dict(state)
        fresh_model.load_state_dict(state)
        model.eval()
        fresh_model.eval()

        dummy_vis = torch.randn(2, 4, 224, 224)
        dummy_pose = torch.randn(2, POSE_FEATURE_DIM)

        with torch.no_grad():
            out1, emb1 = model(dummy_vis, dummy_pose)
            out2, emb2 = fresh_model(dummy_vis, dummy_pose)

        self.assertTrue(torch.equal(out1, out2), "Logits mismatch between reloaded models")
        self.assertTrue(torch.equal(emb1, emb2), "Embeddings mismatch between reloaded models")

    def test_protocol_a_retrieval_chunked_functions(self):
        """Verify that retrieval evaluation function operates properly on 576-D embeddings."""
        rng = np.random.RandomState(42)
        gallery_embs = torch.from_numpy(rng.randn(20, FUSED_EMBEDDING_DIM).astype(np.float32))
        gallery_embs = torch.nn.functional.normalize(gallery_embs, p=2, dim=1)
        gallery_ids = [f"cow_{i % 5}" for i in range(20)]

        query_embs = torch.from_numpy(rng.randn(10, FUSED_EMBEDDING_DIM).astype(np.float32))
        query_embs = torch.nn.functional.normalize(query_embs, p=2, dim=1)
        query_ids = [f"cow_{i % 5}" for i in range(10)]

        results = evaluate_retrieval_chunked(query_embs, query_ids, gallery_embs, gallery_ids, chunk_size=5)
        for key in ["rank_1", "rank_5", "rank_10", "mAP"]:
            self.assertIn(key, results)
            self.assertGreaterEqual(results[key], 0.0)
            self.assertLessEqual(results[key], 100.0)


if __name__ == "__main__":
    unittest.main()
