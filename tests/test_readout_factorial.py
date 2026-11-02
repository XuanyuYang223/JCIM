"""Meaningful isolation, continuity, and gradient-coupling checks for readouts."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jcim.hard_model import HardMolecularGNN
from jcim.pooling_diagnostics import ridge_fit as previous_ridge_fit
from jcim.readout_factorial import (
    build_plan, encoder_cases, exact_array_check, fit_mlp, fit_ridge, fit_scaler,
    gradient_diagnostics, head_scores, original_heads, ridge_scores,
    scoped_features, transform, validate_config, validate_output,
)
from jcim.scale_run import state_sha256


def config_fixture():
    return {"hidden_dim": 64, "layers": 8, "shallow_layer": 1,
        "batch_size": 128, "learning_rate": 0.0015, "weight_decay": 0.0001,
        "gradient_clip": 5.0, "order_seed_offset": 1701, "alpha": 1.0,
        "seeds": [0], "experiences": ["a", "b"],
        "regimes": ["end_to_end", "frozen_readout"],
        "scopes": ["full512", "shallow_only"], "normalizations": ["raw", "train_z"],
        "objectives": ["family", "joint"], "epochs": 2, "cpu_threads": 2, "device": "cpu"}


def heads_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(32)
        return nn.ModuleDict({name: nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, count))
                              for name, count in (("family", 8), ("position", 3), ("distance", 5))})


def plan_fixture():
    records, splits = [], {}
    for name, size, cell in (("train_a", 1386, 0), ("train_b", 1386, 1),
                              ("val_a", 8, 0), ("val_b", 8, 1),
                              ("iid_a", 8, 0), ("iid_b", 8, 1),
                              ("ood_common", 24, 2), ("adaptation_pool", 8, 2)):
        chosen = []
        for offset in range(size):
            index = len(records)
            family = offset % 8
            chosen.append(index)
            records.append({"id": f"m{index}", "y_family": family, "y_position": (cell - family) % 3})
        splits[name] = chosen
    splits["analysis"] = splits["iid_a"] + splits["iid_b"] + splits["ood_common"]
    orders = {}
    for experience in ("a", "b"):
        rng = np.random.default_rng(1701)
        orders[f"{experience}_seed0"] = np.asarray([rng.permutation(splits[f"train_{experience}"]) for _ in range(120)], dtype=np.int32)
    return records, splits, orders


class FactorialProtocolTests(unittest.TestCase):
    def test_exact_factor_grid_and_hyperparameters(self):
        config = config_fixture()
        validate_config(config)
        for name, value in (("scopes", ["full512"]), ("objectives", ["family"]),
                            ("normalizations", ["raw"]), ("epochs", True),
                            ("order_seed_offset", 99), ("gradient_clip", 1),
                            ("seeds", [0, 0]), ("alpha", 2)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_config({**config, name: value})

    def test_output_guard_refuses_old_records_and_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for destination in (source, source / "nested", root):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    validate_output(destination, [source])
            output = root / "new"
            self.assertEqual(validate_output(output, [source]), output.resolve())
            output.mkdir()
            (output / "result.txt").write_text("frozen")
            with self.assertRaises(FileExistsError):
                validate_output(output, [source])

    def test_original_source_labels_and_orders_are_preserved(self):
        records, splits, orders = plan_fixture()
        plan = build_plan(records, splits, orders, config_fixture())
        self.assertEqual(plan["source_only"]["a"]["train"]["indices"], splits["train_a"])
        self.assertEqual(plan["tests"]["analysis"]["ids"], [records[index]["id"] for index in splits["analysis"]])
        self.assertIn("clipping", plan["objective"])

    def test_source_leakage_and_changed_orders_fail(self):
        for failure in ("identity", "forbidden", "order", "duplicate_identity", "short_train"):
            with self.subTest(failure=failure):
                records, splits, orders = plan_fixture()
                if failure == "identity":
                    splits["train_a"][0] = splits["ood_common"][0]
                elif failure == "forbidden":
                    records[0]["y_position"] = 2
                elif failure == "order":
                    orders["a_seed0"][0, :2] = orders["a_seed0"][0, 1::-1]
                elif failure == "duplicate_identity":
                    records[1]["id"] = records[0]["id"]
                else:
                    splits["train_a"].pop()
                with self.assertRaises(ValueError):
                    build_plan(records, splits, orders, config_fixture())

    def test_random_encoder_is_shared_across_experiences(self):
        cases = encoder_cases(Path("hard"), Path("skip"), {**config_fixture(), "seeds": [0, 1, 2]})
        self.assertEqual(len(cases), 15)
        self.assertEqual(sum(row["condition"] == "random" for row in cases), 3)
        self.assertEqual(len({row["model"] for row in cases}), 15)
        self.assertTrue(all(row["experience"] == "shared" for row in cases if row["condition"] == "random"))


class FactorialFeatureTests(unittest.TestCase):
    def test_scope_keeps_architecture_width_and_original_inputs_unchanged(self):
        values = np.random.default_rng(8).normal(size=(9, 512)).astype(np.float32)
        original = values.copy()
        shallow = scoped_features(values, "shallow_only")
        self.assertEqual(shallow.shape, values.shape)
        np.testing.assert_array_equal(shallow[:, :448], 0)
        np.testing.assert_array_equal(shallow[:, 448:], values[:, 448:])
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(scoped_features(values, "full512"), original)

    def test_training_only_standardization_and_inactive_zeros(self):
        train = np.random.default_rng(9).normal(size=(12, 512)).astype(np.float32)
        train[:, 448] = 7
        scoped = scoped_features(train, "shallow_only")
        mean, scale = fit_scaler(scoped, "train_z")
        np.testing.assert_array_equal(mean[:448], 0)
        np.testing.assert_array_equal(scale[:449], 1)
        normalized = transform(scoped, mean, scale)
        np.testing.assert_array_equal(normalized[:, :449], 0)
        np.testing.assert_allclose(normalized[:, 449:].mean(0), 0, atol=6e-8)
        old_mean, old_scale = mean.copy(), scale.copy()
        transform(np.full((4, 512), 1e6), mean, scale)
        np.testing.assert_array_equal(mean, old_mean)
        np.testing.assert_array_equal(scale, old_scale)

    def test_raw_input_is_bitwise_unchanged(self):
        values = np.random.default_rng(4).normal(size=(7, 512)).astype(np.float32)
        mean, scale = fit_scaler(values, "raw")
        np.testing.assert_array_equal(transform(values, mean, scale), values)

    def test_archived_head_extension_is_identical_and_unshared(self):
        base = HardMolecularGNN(64, 8)
        state = copy.deepcopy(base.state_dict())
        with patch("jcim.readout_factorial.checkpoint_state", return_value=state):
            first, second = original_heads(Path("unused"), 0), original_heads(Path("unused"), 0)
        self.assertEqual(state_sha256(first.state_dict()), state_sha256(second.state_dict()))
        for name, head in first.items():
            torch.testing.assert_close(head[0].weight[:, :448], base.heads[name][0].weight, rtol=0, atol=0)
            self.assertFalse(torch.any(head[0].weight[:, 448:]))
            self.assertNotEqual(head[0].weight.data_ptr(), second[name][0].weight.data_ptr())

    def test_exact_feature_checks_fail_without_tolerance(self):
        original = np.zeros((4, 512), dtype=np.float32)
        self.assertTrue(exact_array_check("m", "reference", original, original.copy(), "test")["exact_match"])
        changed = original.copy()
        changed[0, 0] = np.nextafter(np.float32(0), np.float32(1))
        with self.assertRaises(AssertionError):
            exact_array_check("m", "reference", changed, original, "test")

    def test_invalid_dimensions_and_normalization_fail(self):
        for values in (np.ones((4, 64)), np.full((4, 512), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                scoped_features(values, "full512")
        with self.assertRaises(ValueError):
            fit_scaler(np.ones((4, 512)), "test_fitted")


class FactorialGradientTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_joint_objective_has_identical_unclipped_family_gradients(self):
        family_only, joint = heads_fixture(), heads_fixture()
        generator = torch.Generator().manual_seed(6)
        values = torch.randn(12, 512, generator=generator)
        labels = {"family": torch.arange(12) % 8, "position": torch.arange(12) % 3, "distance": torch.arange(12) % 5}
        F.cross_entropy(family_only["family"](values), labels["family"]).backward()
        sum(F.cross_entropy(head(values), labels[task]) for task, head in joint.items()).backward()
        for first, second in zip(family_only["family"].parameters(), joint["family"].parameters()):
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for task in ("position", "distance") for parameter in family_only[task].parameters()))

    def test_global_clipping_is_a_real_route_for_objective_coupling(self):
        family_only, joint = heads_fixture(), heads_fixture()
        values = torch.randn(12, 512, generator=torch.Generator().manual_seed(7))
        F.cross_entropy(family_only["family"](values), torch.arange(12) % 8).backward()
        sum(F.cross_entropy(head(values), torch.arange(12) % count)
            for head, count in zip(joint.values(), (8, 3, 5))).backward()
        first = gradient_diagnostics(family_only, .1)
        second = gradient_diagnostics(joint, .1)
        self.assertEqual(first["family_gradient_norm"], second["family_gradient_norm"])
        self.assertGreater(second["total_gradient_norm"], first["total_gradient_norm"])
        self.assertLess(second["clip_coefficient"], first["clip_coefficient"])
        self.assertTrue(first["clipped"] and second["clipped"])
        self.assertFalse(torch.equal(family_only["family"][0].weight.grad, joint["family"][0].weight.grad))

    def test_fit_clones_initializer_and_uses_exact_epoch_order(self):
        heads = heads_fixture()
        before = state_sha256(heads.state_dict())
        values = torch.randn(9, 512, generator=torch.Generator().manual_seed(8))
        labels = {"family": torch.arange(9) % 8, "position": torch.arange(9) % 3, "distance": torch.arange(9) % 5}
        train = [30, 10, 20, 40, 50]
        orders = np.asarray([train[::-1], train], dtype=np.int32)
        result, fitted = fit_mlp(heads, values, labels, train, orders, [5, 6], config_fixture(), "family", "cpu", "fit_test")
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(len(result["clipping"]), 2)
        self.assertEqual(state_sha256(heads.state_dict()), before)
        self.assertEqual(state_sha256(result["initial_state_dict"]), before)
        for name in ("position", "distance"):
            for key, value in heads[name].state_dict().items():
                torch.testing.assert_close(value, fitted[name].state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            fit_mlp(heads, values, labels, train, np.asarray([[30] * 5]), [5], config_fixture(), "joint", "cpu", "bad")

    def test_joint_fit_matches_historical_optimizer_and_global_clip_exactly(self):
        heads = heads_fixture()
        reference = copy.deepcopy(heads)
        values = torch.randn(8, 512, generator=torch.Generator().manual_seed(15))
        labels = {"family": torch.arange(8) % 8, "position": torch.arange(8) % 3, "distance": torch.arange(8) % 5}
        config = {**config_fixture(), "gradient_clip": .1}
        train = [6, 2, 9, 12, 22]
        orders = np.asarray([train, train[::-1]], dtype=np.int32)
        fitted, _ = fit_mlp(heads, values, labels, train, orders, [5, 6], config, "joint", "cpu", "same")
        optimizer = torch.optim.AdamW(reference.parameters(), lr=.0015, weight_decay=.0001)
        lookup = {index: row for row, index in enumerate(train)}
        for order in orders:
            rows = [lookup[int(index)] for index in order]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(F.cross_entropy(head(values[rows]), labels[task][rows]) for task, head in reference.items())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
            optimizer.step()
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, fitted["state_dict"][key], rtol=0, atol=0)

    def test_scores_repeat_valid_tail_rows_without_changing_predictions(self):
        heads = heads_fixture()
        values = torch.randn(129, 512, generator=torch.Generator().manual_seed(16))
        scores = head_scores(heads, values)
        with torch.no_grad():
            for task, head in heads.items():
                expected = torch.cat((head(values[:128]), head(values[-1:].repeat(128, 1))[:1])).numpy()
                np.testing.assert_array_equal(scores[task], expected)


class FactorialRidgeTests(unittest.TestCase):
    def test_standardized_shallow_ridge_exactly_reproduces_v6(self):
        rng = np.random.default_rng(8)
        values = rng.normal(size=(80, 512)).astype(np.float32)
        labels = np.arange(80) % 8
        previous = previous_ridge_fit(values[:, 448:].copy(), labels)
        fitted = fit_ridge(scoped_features(values, "shallow_only"), labels, "train_z", "shallow_only")
        for key in ("mean", "scale", "weights", "bias"):
            np.testing.assert_array_equal(fitted[key], previous[key])
        expected = ((values[:, 448:].astype(np.float64) - previous["mean"]) / previous["scale"]) @ previous["weights"] + previous["bias"]
        np.testing.assert_array_equal(ridge_scores(fitted, values), expected)

    def test_raw_ridge_has_unpenalized_intercept_and_is_translation_invariant(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(40, 512)).astype(np.float64)
        values[:, :448] = 0
        labels = np.arange(40) % 8
        first = fit_ridge(values, labels, "raw", "shallow_only")
        shifted = values.copy()
        shifted[:, 448:] += 5
        second = fit_ridge(shifted, labels, "raw", "shallow_only")
        np.testing.assert_allclose(ridge_scores(first, values), ridge_scores(second, shifted), atol=1e-12)
        np.testing.assert_array_equal(first["mean"], 0)
        np.testing.assert_array_equal(first["scale"], 1)

    def test_zero_padding_does_not_invent_additional_shallow_ridge_features(self):
        rng = np.random.default_rng(14)
        values = scoped_features(rng.normal(size=(30, 512)).astype(np.float32), "shallow_only")
        labels = np.arange(30) % 8
        fitted = fit_ridge(values, labels, "train_z", "shallow_only")
        self.assertEqual(fitted["weights"].shape, (64, 8))
        changed = values.copy()
        changed[:, :448] = 1e6
        np.testing.assert_array_equal(ridge_scores(fitted, values), ridge_scores(fitted, changed))


if __name__ == "__main__":
    unittest.main()
