"""Negative controls and analytic checks for the independent scale audit."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from maintenance.audit_scale_v4 import (
    Checks, array_digest, encoder_layers, independent_pool, independent_ridge, main,
    paired_measurements, receptive_region, scope_masks, verify_aggregated_summary, verify_direct_supports,
)


class IndependentScaleAuditTests(unittest.TestCase):
    def test_graph_distance_expansion_respects_boundaries_and_padding(self):
        adjacency = np.zeros((7, 7), dtype=np.float32)
        for left, right in ((0, 1), (1, 2), (1, 3), (3, 4)):
            adjacency[left, right] = adjacency[right, left] = 1
        masks = scope_masks(adjacency, 5, 0, [0, 1])
        self.assertEqual(np.flatnonzero(masks["root"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(masks["target"]).tolist(), [0, 1])
        self.assertEqual(np.flatnonzero(masks["target_plus1"]).tolist(), [0, 1, 2, 3])
        self.assertEqual(np.flatnonzero(masks["target_plus2"]).tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(np.array_equal(masks["target_plus3"], masks["whole_graph"]))
        self.assertEqual(np.flatnonzero(receptive_region(adjacency, masks["target"], 1)).tolist(), [0, 1, 2, 3])
        self.assertEqual(np.flatnonzero(receptive_region(adjacency, masks["root"], 2)).tolist(), [0, 1, 2, 3])

    def test_invalid_oracle_membership_and_directed_graph_are_rejected(self):
        adjacency = np.zeros((3, 3))
        for target in ([1], [0, 0], [0, 3]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                scope_masks(adjacency, 3, 0, target)
        adjacency[0, 1] = 1
        with self.assertRaisesRegex(ValueError, "undirected"):
            scope_masks(adjacency, 3, 0, [0, 1])

    def test_ridge_matches_analytic_one_dimensional_solution(self):
        predictions, scores = independent_ridge([[-1], [1]], np.array([0, 1]), [[2], [0]], alpha=2, classes=2)
        np.testing.assert_allclose(scores, [[0, 1], [0.5, 0.5]], atol=1e-15)
        self.assertEqual(predictions.tolist(), [1, 0])
        wide = np.pad([[-1.0], [1.0]], ((0, 0), (0, 3)))
        wide_test = np.pad([[2.0], [0.0]], ((0, 0), (0, 3)))
        dual_predictions, dual_scores = independent_ridge(wide, np.array([0, 1]), wide_test, alpha=2, classes=2)
        np.testing.assert_allclose(dual_scores, scores, atol=1e-15)
        np.testing.assert_array_equal(dual_predictions, predictions)

    def test_ridge_standardization_is_support_only_and_rejects_invalid_inputs(self):
        _, original_scores = independent_ridge([[-1], [1]], np.array([0, 1]), [[2]], classes=2)
        _, augmented_scores = independent_ridge([[-1], [1]], np.array([0, 1]), [[2], [100000]], classes=2)
        np.testing.assert_array_equal(original_scores, augmented_scores[:1])
        for alpha in (0, -1, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                independent_ridge([[-1], [1]], np.array([0, 1]), [[2]], alpha=alpha, classes=2)
        with self.assertRaises(ValueError):
            independent_ridge([[-1], [1]], np.array([0, 2]), [[2]], classes=2)

    def test_checkpoint_tensor_forward_has_expected_normalization_and_padding(self):
        state = {"embedding.weight": torch.eye(2), "embedding.bias": torch.zeros(2),
                 "layers.0.update.0.weight": torch.zeros((2, 4)),
                 "layers.0.update.0.bias": torch.zeros(2),
                 "layers.0.update.2.weight": torch.zeros((2, 2)),
                 "layers.0.update.2.bias": torch.zeros(2),
                 "layers.0.norm.weight": torch.ones(2), "layers.0.norm.bias": torch.zeros(2)}
        x = np.array([[[1, 0], [0, 1], [100, 100]]], dtype=np.float32)
        adjacency = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.float32)
        layers = encoder_layers(state, x, adjacency, mask)
        np.testing.assert_array_equal(layers[0], [[[1, 0], [0, 1], [0, 0]]])
        expected = 0.5 / np.sqrt(0.25 + 1e-5)
        np.testing.assert_allclose(layers[1], [[[expected, 0], [0, expected], [0, 0]]], atol=1e-7)

    def test_pooling_reference_uses_queries_and_excludes_padding(self):
        hidden = np.array([[[1, 2], [3, 4], [100, 100]]], dtype=np.float32)
        x = np.zeros((1, 3, 18), dtype=np.float32)
        x[0, 0, 15] = 1
        x[0, 1, 13] = 1
        x[0, 0, 14] = 1
        x[0, 1, 16] = 1
        x[0, 0, 17] = 1
        atom_mask = np.array([[1, 1, 0]], dtype=np.float32)
        masks = {"root": np.array([[True, False, False]]), "whole_graph": np.array([[True, True, False]])}
        pools = independent_pool(hidden, x, atom_mask, masks)
        np.testing.assert_array_equal(pools["root"], [[1, 2]])
        np.testing.assert_array_equal(pools["whole_graph"], [[2, 3]])
        self.assertEqual(pools["legacy_graph"].shape, (1, 14))
        np.testing.assert_array_equal(pools["legacy_graph"][0, 4:], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
        with self.assertRaisesRegex(ValueError, "valid atoms"):
            independent_pool(hidden, x, atom_mask, {"bad": np.array([[True, True, True]])})

    def test_array_fingerprints_include_dtype_shape_and_signed_zero(self):
        values = np.array([0.0, 1.0], dtype=np.float32)
        altered = values.copy()
        altered[0] = -0.0
        self.assertNotEqual(array_digest(values), array_digest(altered))
        self.assertNotEqual(array_digest(values), array_digest(values.reshape(1, 2)))
        self.assertNotEqual(array_digest(values), array_digest(values.astype(np.float64)))

    def test_direct_support_audit_detects_wrong_position_and_unrecorded_draw(self):
        records = [{"id": f"molecule_{index:04d}", "y_family": index % 8, "y_position": index // 320}
                   for index in range(960)]
        splits = {"analysis": list(range(960))}
        plan = {"sets": []}
        for repeat in range(3):
            for position in range(3):
                rng = np.random.default_rng(50000 + 100 * repeat + position)
                indices = []
                for label in range(8):
                    candidates = [index for index in splits["analysis"]
                                  if records[index]["y_family"] == label and records[index]["y_position"] == position]
                    indices.extend(rng.choice(candidates, size=32, replace=False).tolist())
                indices.sort(key=lambda index: records[index]["id"])
                plan["sets"].append({"repeat": repeat, "source_position": position, "indices": indices,
                                     "ids": [records[index]["id"] for index in indices]})
        checks = Checks()
        verify_direct_supports(plan, records, splits, checks)
        self.assertTrue(checks.success)
        changed = copy.deepcopy(plan)
        changed["sets"][0]["indices"].reverse()
        checks = Checks()
        verify_direct_supports(changed, records, splits, checks)
        self.assertEqual(checks.results["direct_support_exact_predeclared_sampling_rule"]["failed"], 1)
        changed = copy.deepcopy(plan)
        changed["sets"][0]["indices"][0] = 500
        checks = Checks()
        verify_direct_supports(changed, records, splits, checks)
        self.assertGreater(checks.results["direct_support_has_256_unique_source_position_IDs"]["failed"], 0)

    def test_nested_summary_rejects_nine_independent_encoder_claim(self):
        raw = [{"experience": "a", "seed": seed, "repeat": repeat,
                "accuracy": 0.2 + 0.1 * seed + 0.03 * repeat,
                "balanced_accuracy": 0.2 + 0.1 * seed + 0.03 * repeat}
               for seed in range(3) for repeat in range(3)]
        statistic = {"mean": 0.33, "std": 0.1, "n": 3, "per_seed": {"0": 0.23, "1": 0.33, "2": 0.43}}
        summary = [{"experience": "a", "accuracy": copy.deepcopy(statistic), "balanced_accuracy": copy.deepcopy(statistic)}]
        checks = Checks()
        verify_aggregated_summary(raw, summary, ("experience",), checks, "example")
        self.assertTrue(checks.success)
        summary[0]["accuracy"]["n"] = 9
        checks = Checks()
        verify_aggregated_summary(raw, summary, ("experience",), checks, "example")
        self.assertEqual(checks.results["example_correct_seed_unit_mean_sample_SD"]["failed"], 1)

    def test_paired_summary_rejects_mismatched_support_budget(self):
        rows = [{"condition": condition, "seed": "0", "repeat": "0", "experience": "a",
                 "n_fit": "64", "n_test": "100", "accuracy": accuracy, "balanced_accuracy": accuracy}
                for condition, accuracy in (("trained", 0.9), ("random", 0.6))]
        checks = Checks()
        delta = paired_measurements(rows, ("experience",), checks)
        self.assertTrue(checks.success)
        self.assertAlmostEqual(delta[0]["accuracy"], 0.3)
        rows[1]["n_fit"] = "32"
        checks = Checks()
        paired_measurements(rows, ("experience",), checks)
        self.assertFalse(checks.success)

    def test_cli_refuses_overwrite_and_outputs_inside_frozen_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            existing = directory / "existing.json"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--record", str(directory / "absent"), "--output", str(existing)])
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")
            with self.assertRaisesRegex(ValueError, "outside"):
                main(["--record", str(directory / "record"), "--output", str(directory / "record" / "audit.json")])


if __name__ == "__main__":
    unittest.main()
