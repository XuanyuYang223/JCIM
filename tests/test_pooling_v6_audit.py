"""Negative controls for independent branch and frozen-pooling auditing."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from maintenance.audit_pooling_v6 import (
    Checks, branch_effects, branch_summary, classification_scores,
    first_layer_pools, independent_ridge, main, verify_seed_panels,
)


class PoolingV6AuditTests(unittest.TestCase):
    def test_wrong_to_wrong_flip_does_not_masquerade_as_accuracy_change(self):
        effect = branch_effects([[0, 3, 1], [3, 0, 1]], [[0, 1, 3], [2, 0, 1]], [0, 0])
        np.testing.assert_array_equal(effect["prediction_changed"], [True, False])
        np.testing.assert_array_equal(effect["both_wrong_changed"], [True, False])
        np.testing.assert_array_equal(effect["wrong_to_correct"], [False, False])
        np.testing.assert_array_equal(effect["correct_to_wrong"], [False, False])
        self.assertEqual(effect["intact_correct"].mean(), effect["altered_correct"].mean())

    def test_common_logit_shift_has_zero_margin_and_centered_effect(self):
        before = np.array([[1., 3., 2.], [4., 1., 2.]])
        effect = branch_effects(before, before + 7, [2, 0])
        np.testing.assert_array_equal(effect["margin_delta"], [0, 0])
        np.testing.assert_array_equal(effect["centered_logit_l2"], [0, 0])
        self.assertTrue((effect["raw_logit_l2"] > 0).all())
        self.assertFalse(effect["logits_exactly_unchanged"].any())
        self.assertTrue(effect["centered_logits_exactly_unchanged"].all())

    def test_signed_margin_cancellation_retains_absolute_change_and_sample_SD(self):
        effect = branch_effects([[0., 2.], [0., 2.]], [[1., 2.], [-1., 2.]], [0, 0])
        summary = branch_summary(effect, [0, 1])
        self.assertEqual(summary["margin_delta_mean"], 0)
        self.assertEqual(summary["absolute_margin_delta_mean"], 1)
        self.assertAlmostEqual(summary["margin_delta_sd"], np.sqrt(2))
        self.assertEqual(summary["prediction_changed_rate"], 0)
        with self.assertRaises(ValueError):
            branch_summary(effect, [])

    def test_tied_true_margin_keeps_argmax_first_class_distinct(self):
        effect = branch_effects([[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [0, 1])
        np.testing.assert_array_equal(effect["intact_margin_tie"], [True, True])
        np.testing.assert_array_equal(effect["intact_correct"], [True, False])

    def test_ridge_only_uses_fit_support_and_keeps_constant_scale_finite(self):
        x = np.array([[0., 8.], [1., 8.], [2., 8.], [3., 8.]])
        state = independent_ridge(x, [0, 0, 1, 1])
        np.testing.assert_array_equal(state["mean"], [1.5, 8])
        self.assertEqual(state["scale"][1], 1)
        self.assertTrue(np.isfinite(state["weights"]).all())
        before = {name: value.copy() for name, value in state.items()}
        _ = ((np.array([[100., 100.]]) - state["mean"]) / state["scale"]) @ state["weights"] + state["bias"]
        for name, value in before.items():
            np.testing.assert_array_equal(state[name], value)
        with self.assertRaises(ValueError):
            independent_ridge(x, [0, 1, 2, 8])

    def test_independent_layer1_pooling_excludes_padding_and_respects_oracle_mask(self):
        # Zero updates and identity embedding make each valid node's normalized
        # embedding directly calculable, without an experiment-model import.
        state = {"embedding.weight": torch.eye(2), "embedding.bias": torch.zeros(2),
                 "layers.0.update.0.weight": torch.zeros(2, 4), "layers.0.update.0.bias": torch.zeros(2),
                 "layers.0.update.2.weight": torch.zeros(2, 2), "layers.0.update.2.bias": torch.zeros(2),
                 "layers.0.norm.weight": torch.ones(2), "layers.0.norm.bias": torch.zeros(2)}
        x = torch.tensor([[[3., 1.], [1., 3.], [999., 1.]]])
        mask = torch.tensor([[1., 1., 0.]])
        adjacency = torch.zeros(1, 3, 3)
        target = torch.tensor([[True, False, False]])
        values = first_layer_pools(state, x, adjacency, mask, target)
        self.assertAlmostEqual(float(values["whole_graph"][0, 0]), float(values["whole_graph"][0, 1]), places=7)
        self.assertGreater(values["target"][0, 0], .99)
        self.assertEqual(values["target"][0, 1], 0)
        with self.assertRaisesRegex(ValueError, "padding"):
            first_layer_pools(state, x, adjacency, mask, torch.tensor([[True, False, True]]))

    def test_seed_panel_rejects_population_SD_and_duplicate_seeds(self):
        rows = [{"regime": "r", "seed": seed, "accuracy": value, "balanced_accuracy": value}
                for seed, value in enumerate((.1, .2, .3))]
        metric = {"mean": .2, "std": .1, "n_seeds": 3, "per_seed": {"0": .1, "1": .2, "2": .3}}
        panels = [{"regime": "r", "accuracy": metric, "balanced_accuracy": metric}]
        checks = Checks()
        verify_seed_panels(rows, panels, ("regime",), checks, "test")
        self.assertTrue(checks.success)
        altered = copy.deepcopy(panels)
        altered[0]["accuracy"]["std"] = float(np.std([.1, .2, .3], ddof=0))
        checks = Checks()
        verify_seed_panels(rows, altered, ("regime",), checks, "test")
        self.assertFalse(checks.success)
        checks = Checks()
        verify_seed_panels(rows + [rows[0]], panels, ("regime",), checks, "test")
        self.assertEqual(checks.results["test_one_value_per_seed"]["failed"], 1)

    def test_metrics_keep_balanced_accuracy_distinct_from_ordinary(self):
        scores = classification_scores([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(scores["accuracy"], .75)
        self.assertEqual(scores["balanced_accuracy"], .5)

    def test_audit_cannot_overwrite_evidence_or_write_inside_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "existing.json"
            output.write_text("frozen", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "frozen")
            archive = directory / "record"
            with self.assertRaisesRegex(ValueError, "outside"):
                main(["--record", str(archive), "--output", str(archive / "audit.json")])


if __name__ == "__main__":
    unittest.main()
