"""Negative controls for independent factorial and frozen-head verification."""

import itertools
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from maintenance.audit_readout_v7 import (
    Checks, classification_scores, expected_initial_heads, factorial_effects,
    independent_ridge, independent_scaler, ridge_logits, scoped_values,
    main, validate_clip_row, verify_seed_panels,
)


class ReadoutV7AuditTests(unittest.TestCase):
    def test_scaler_uses_fit_features_and_keeps_masked_coordinates_zero(self):
        x = np.zeros((4, 512), dtype=np.float32)
        x[:, 0], x[:, 448] = [100, 200, 300, 400], [1, 2, 3, 4]
        masked = scoped_values(x, "shallow_only")
        mean, scale = independent_scaler(masked, "train_z")
        self.assertEqual(mean[0], 0)
        self.assertEqual(scale[0], 1)
        self.assertEqual(mean[448], 2.5)
        np.testing.assert_array_equal(((masked - mean) / scale)[:, :448], 0)
        np.testing.assert_array_equal(x[:, 0], [100, 200, 300, 400])
        raw_mean, raw_scale = independent_scaler(x, "raw")
        np.testing.assert_array_equal(raw_mean, np.zeros(512))
        np.testing.assert_array_equal(raw_scale, np.ones(512))
        with self.assertRaises(ValueError):
            scoped_values(x[:, :64], "full512")

    def test_raw_ridge_intercept_is_unpenalized_and_translation_invariant(self):
        x = np.zeros((8, 512))
        x[:, 448] = np.arange(8)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        state = independent_ridge(x, y, "raw", "shallow_only")
        shifted = x.copy()
        shifted[:, 448] += 1000
        translated = independent_ridge(shifted, y, "raw", "shallow_only")
        np.testing.assert_allclose(state["weights"], translated["weights"], atol=1e-14)
        np.testing.assert_allclose(ridge_logits(state, x, "shallow_only"),
                                   ridge_logits(translated, shifted, "shallow_only"), atol=1e-13)
        np.testing.assert_array_equal(state["active_columns"], np.arange(448, 512))
        np.testing.assert_allclose(ridge_logits(state, x, "shallow_only").mean(0),
                                   np.eye(8)[y].mean(0), atol=1e-14)

    def test_standardized_shallow_ridge_ignores_deep_features_and_retains_constant_scale(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(12, 512))
        x[:, 449] = 19
        state = independent_ridge(x, np.arange(12) % 8, "train_z", "shallow_only")
        changed = x.copy()
        changed[:, :448] += 1e6
        actual = ridge_logits(state, x, "shallow_only")
        np.testing.assert_array_equal(actual, ridge_logits(state, changed, "shallow_only"))
        self.assertEqual(state["scale"][1], 1)
        with self.assertRaises(ValueError):
            independent_ridge(x, np.repeat(8, 12), "train_z", "shallow_only")

    def test_expected_initialization_copies_original_heads_and_zeros_added_columns(self):
        source = {"embedding.weight": torch.ones(2, 2),
                  "heads.family.0.weight": torch.arange(896).reshape(2, 448).float(),
                  "heads.family.0.bias": torch.tensor([1., 2.]),
                  "heads.family.2.weight": torch.ones(8, 2),
                  "heads.family.2.bias": torch.ones(8)}
        initial = expected_initial_heads(source)
        self.assertNotIn("embedding.weight", initial)
        self.assertEqual(tuple(initial["family.0.weight"].shape), (2, 512))
        self.assertTrue(torch.equal(initial["family.0.weight"][:, :448], source["heads.family.0.weight"]))
        self.assertEqual(initial["family.0.weight"][:, 448:].count_nonzero().item(), 0)
        initial["family.0.bias"][0] = -9
        self.assertEqual(source["heads.family.0.bias"][0].item(), 1)

    def test_clip_diagnostics_reject_wrong_denominator_nonfinite_and_false_flags(self):
        row = {"total_gradient_norm": 10, "family_gradient_norm": 4,
               "clip_coefficient": 5 / (10 + 1e-6), "clipped": True}
        self.assertTrue(validate_clip_row(row))
        for alteration in ({"clip_coefficient": .8}, {"family_gradient_norm": 20},
                           {"total_gradient_norm": float("nan")}, {"clipped": False}):
            with self.subTest(alteration=alteration):
                self.assertFalse(validate_clip_row({**row, **alteration}))

    def test_factorial_effects_preserve_interactions_and_conditional_cancellation(self):
        cells = {levels: levels[0] + 2 * levels[1] - levels[0] * levels[1]
                 + 4 * levels[0] * levels[1] * levels[2]
                 for levels in itertools.product((0, 1), repeat=3)}
        effects = factorial_effects(cells)
        self.assertEqual(effects[(0,)]["average"], 1.5)
        self.assertEqual(effects[(0, 1)]["conditional"], [((0,), -1.0), ((1,), 3.0)])
        self.assertEqual(effects[(0, 1, 2)]["average"], 4)
        with self.assertRaises(ValueError):
            factorial_effects({key: value for key, value in cells.items() if key != (1, 1, 1)})

    def test_frozen_disjoint_heads_have_identical_family_gradient_before_shared_clipping(self):
        family = torch.nn.Linear(2, 2)
        position = torch.nn.Linear(2, 2)
        x, labels = torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([0, 1])
        motif_loss = F.cross_entropy(family(x), labels)
        motif_grad = torch.autograd.grad(motif_loss, tuple(family.parameters()), retain_graph=True)
        joint_loss = motif_loss + 100 * F.cross_entropy(position(x), 1 - labels)
        joint_grad = torch.autograd.grad(joint_loss, tuple(family.parameters()), retain_graph=True)
        for expected, actual in zip(motif_grad, joint_grad):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        family.zero_grad(set_to_none=True)
        position.zero_grad(set_to_none=True)
        joint_loss.backward()
        before = family.weight.grad.clone()
        norm = torch.nn.utils.clip_grad_norm_([*family.parameters(), *position.parameters()], 5)
        self.assertGreater(norm.item(), 5)
        self.assertFalse(torch.equal(before, family.weight.grad))

    def test_metrics_do_not_equate_ordinary_and_balanced_accuracy(self):
        scores = classification_scores([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(scores["accuracy"], .75)
        self.assertEqual(scores["balanced_accuracy"], .5)

    def test_seed_panels_reject_population_SD_and_duplicate_seed_rows(self):
        rows = [{"regime": "test", "seed": seed, "accuracy": value, "balanced_accuracy": value}
                for seed, value in enumerate((.1, .2, .3))]
        metric = {"mean": .2, "std": .1, "n_seeds": 3,
                  "per_seed": {"0": .1, "1": .2, "2": .3}}
        summary = [{"regime": "test", "accuracy": metric, "balanced_accuracy": metric}]
        checks = Checks()
        verify_seed_panels(rows, summary, ("regime",), [0, 1, 2], checks, "test")
        self.assertTrue(checks.success)
        altered = copy.deepcopy(summary)
        altered[0]["accuracy"]["std"] = float(np.std([.1, .2, .3]))
        checks = Checks()
        verify_seed_panels(rows, altered, ("regime",), [0, 1, 2], checks, "test")
        self.assertFalse(checks.success)
        checks = Checks()
        verify_seed_panels(rows + [rows[0]], summary, ("regime",), [0, 1, 2], checks, "test")
        self.assertFalse(checks.success)

    def test_ridge_two_factor_effects_keep_unscaled_difference_in_differences(self):
        cells = {(a, b): 2 * a + b + 3 * a * b for a, b in itertools.product((0, 1), repeat=2)}
        effects = factorial_effects(cells)
        self.assertEqual(effects[(0,)]["average"], 3.5)
        self.assertEqual(effects[(1,)]["average"], 2.5)
        self.assertEqual(effects[(0, 1)]["average"], 3)

    def test_audit_refuses_existing_outputs_and_paths_inside_every_prior_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            existing = directory / "existing.json"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--output", str(existing)])
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")
            hard = directory / "hard_v2"
            with self.assertRaisesRegex(ValueError, "outside"):
                main(["--hard", str(hard), "--output", str(directory / "pilot_v1/audit.json")])


if __name__ == "__main__":
    unittest.main()
