"""Behavioral checks for representation statistics and readout isolation."""

import unittest

import numpy as np

from jcim.analysis import (
    balanced_accuracy,
    fit_ridge_readout,
    layerwise_similarity,
    linear_cka,
    orthogonal_procrustes_similarity,
)


class SimilarityTests(unittest.TestCase):
    def test_invariant_to_rotation_shift_and_isotropic_scale(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(240, 13))
        rotation, _ = np.linalg.qr(rng.normal(size=(13, 13)))
        y = 7.0 * x @ rotation + rng.normal(size=13)
        self.assertAlmostEqual(linear_cka(x, y), 1.0, places=12)
        self.assertAlmostEqual(orthogonal_procrustes_similarity(x, y), 1.0, places=12)

    def test_row_correspondence_matters(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(700, 9))
        y = x[rng.permutation(len(x))]
        self.assertLess(linear_cka(x, y), 0.1)
        self.assertLess(orthogonal_procrustes_similarity(x, y), 0.25)

    def test_constant_representation_has_no_similarity_evidence(self):
        x = np.ones((12, 4))
        y = np.arange(60).reshape(12, 5)
        self.assertEqual(linear_cka(x, y), 0.0)
        self.assertEqual(orthogonal_procrustes_similarity(x, y), 0.0)

    def test_unequal_feature_widths_are_supported(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(30, 4))
        y = np.column_stack([x, np.zeros((30, 3))])
        self.assertAlmostEqual(linear_cka(x, y), 1.0, places=12)
        self.assertAlmostEqual(orthogonal_procrustes_similarity(x, y), 1.0, places=12)
        result = layerwise_similarity([x, x], [y, y])
        self.assertEqual([row["layer"] for row in result], [0, 1])

    def test_rejects_invalid_correspondence_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            linear_cka(np.ones((3, 2)), np.ones((4, 2)))
        with self.assertRaises(ValueError):
            linear_cka(np.array([[1.0], [np.nan]]), np.ones((2, 1)))


class ReadoutTests(unittest.TestCase):
    @staticmethod
    def make_problem(seed=11):
        rng = np.random.default_rng(seed)
        train_y = np.tile(np.arange(8), 40)
        test_y = np.tile(np.arange(8), 80)
        prototypes = np.eye(8)
        train_x = prototypes[train_y] + rng.normal(scale=0.12, size=(train_y.size, 8))
        test_x = prototypes[test_y] + rng.normal(scale=0.12, size=(test_y.size, 8))
        return train_x, train_y, test_x, test_y

    def test_balanced_accuracy_exposes_majority_class_failure(self):
        y = np.array([0, 0, 0, 0, 1])
        self.assertEqual(balanced_accuracy(y, np.zeros(5, dtype=int)), 0.5)

    def test_readout_recovers_reusable_linear_information(self):
        train_x, train_y, test_x, test_y = self.make_problem()
        result = fit_ridge_readout(train_x, train_y, test_x, test_y)
        self.assertGreater(result["balanced_accuracy"], 0.99)
        self.assertEqual(result["classes_seen"], list(range(8)))

    def test_test_labels_cannot_change_predictions(self):
        train_x, train_y, test_x, test_y = self.make_problem()
        first = fit_ridge_readout(train_x, train_y, test_x, test_y)
        relabeled = fit_ridge_readout(train_x, train_y, test_x, (test_y + 1) % 8)
        self.assertEqual(first["predictions"], relabeled["predictions"])
        self.assertLess(relabeled["accuracy"], 0.01)

    def test_test_batch_cannot_change_standardization(self):
        train_x, train_y, test_x, test_y = self.make_problem()
        first = fit_ridge_readout(train_x, train_y, test_x[:20], test_y[:20])
        extended_x = np.concatenate([test_x[:20], np.full((20, 8), 1e6)])
        extended_y = np.concatenate([test_y[:20], test_y[:20]])
        extended = fit_ridge_readout(train_x, train_y, extended_x, extended_y)
        self.assertEqual(first["predictions"], extended["predictions"][:20])

    def test_shuffled_training_labels_do_not_transfer(self):
        train_x, train_y, test_x, test_y = self.make_problem()
        shuffled = np.random.default_rng(19).permutation(train_y)
        result = fit_ridge_readout(train_x, shuffled, test_x, test_y)
        self.assertLess(abs(result["balanced_accuracy"] - 1 / 8), 0.1)

    def test_wide_features_and_constant_columns_are_supported(self):
        train_x, train_y, test_x, test_y = self.make_problem()
        # More columns than training rows exercises the smaller dual solve.
        train_x = np.column_stack([np.tile(train_x, 50), np.ones(len(train_x))])
        test_x = np.column_stack([np.tile(test_x, 50), np.ones(len(test_x))])
        result = fit_ridge_readout(train_x, train_y, test_x, test_y)
        self.assertGreater(result["balanced_accuracy"], 0.99)


if __name__ == "__main__":
    unittest.main()
