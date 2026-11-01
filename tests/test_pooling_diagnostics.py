"""Identity, oracle-boundary, and matched-readout controls for pooling diagnosis."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from jcim.analysis import fit_ridge_readout
from jcim.hard_model import HardMolecularGNN
from jcim.mechanism import encoder_state_sha256
from jcim.pooling_diagnostics import (
    build_plan, checked_indices, exact_feature_check, extract_first_layer,
    mlp_fit, mlp_scores, new_head, ridge_fit, standardizer, target_masks, validate_config,
)
from jcim.scale_run import state_sha256
from jcim.skip_model import SkipMolecularGNN


def config_fixture():
    return {"hidden_dim": 64, "layers": 8, "shallow_layer": 1, "batch_size": 128,
            "learning_rate": 0.0015, "weight_decay": 0.0001, "gradient_clip": 5.0,
            "alpha": 1.0, "seeds": [0], "support_repeats": [0],
            "regimes": ["end_to_end", "frozen_readout"], "experiences": ["a", "b"],
            "epochs": 2, "mlp_seed_offset": 61000, "cpu_threads": 2, "device": "cpu"}


def plan_fixture():
    records, splits = [], {}

    def add(name, count, cell):
        indices = []
        for number in range(count):
            label = number % 8
            indices.append(len(records))
            records.append({"id": f"m{len(records):05d}", "y_family": label,
                            "y_position": (cell - label) % 3})
        splits[name] = indices

    for name, count, cell in (("train_a", 1386, 0), ("train_b", 1386, 1),
                              ("val_a", 8, 0), ("val_b", 8, 1),
                              ("iid_a", 8, 0), ("iid_b", 8, 1),
                              ("ood_common", 24, 2), ("adaptation_pool", 8, 2)):
        add(name, count, cell)
    supports = []
    diagnostic_indices = []
    for position in range(3):
        indices = []
        for label in range(8):
            for _ in range(32):
                indices.append(len(records))
                records.append({"id": f"m{len(records):05d}", "y_family": label, "y_position": position})
        diagnostic_indices.extend(indices)
        supports.append({"repeat": 0, "source_position": position, "indices": indices,
                         "ids": [records[index]["id"] for index in indices]})
    splits["analysis"] = splits["iid_a"] + splits["iid_b"] + splits["ood_common"] + diagnostic_indices
    orders = {}
    for experience in ("a", "b"):
        rng = np.random.default_rng(1701)
        orders[f"{experience}_seed0"] = np.asarray([rng.permutation(splits[f"train_{experience}"]) for _ in range(120)], dtype=np.int32)
    return records, splits, {"sets": supports}, orders, config_fixture()


class PoolingPlanTests(unittest.TestCase):
    def test_original_labels_and_diagnostic_labels_are_separate(self):
        records, splits, direct, orders, config = plan_fixture()
        plan = build_plan(records, splits, direct, orders, config)
        self.assertEqual(plan["source_only"]["a"]["train"]["indices"], splits["train_a"])
        self.assertEqual(plan["direct_support"], direct["sets"])
        self.assertFalse(set(splits["train_a"]) & set(direct["sets"][0]["indices"]))
        self.assertIn("unlike", plan["mlp_loss"])

    def test_holdout_identity_and_combination_leakage_are_rejected(self):
        for failure in ("test_identity", "reserved_identity", "forbidden_combination"):
            with self.subTest(failure=failure):
                records, splits, direct, orders, config = plan_fixture()
                if failure == "test_identity":
                    splits["train_a"][0] = splits["ood_common"][0]
                elif failure == "reserved_identity":
                    splits["train_a"][0] = splits["adaptation_pool"][0]
                else:
                    records[splits["train_a"][0]]["y_position"] = 2
                with self.assertRaises(ValueError):
                    build_plan(records, splits, direct, orders, config)

    def test_exact_orders_and_balanced_direct_support_are_enforced(self):
        for failure in ("order", "repeated_support", "identity", "position", "balance"):
            with self.subTest(failure=failure):
                records, splits, direct, orders, config = plan_fixture()
                row = direct["sets"][0]
                if failure == "order":
                    orders["a_seed0"][0, :2] = orders["a_seed0"][0, 1::-1]
                elif failure == "repeated_support":
                    row["indices"][0] = row["indices"][1]
                elif failure == "identity":
                    row["ids"][0] = "wrong"
                elif failure == "position":
                    records[row["indices"][0]]["y_position"] = 1
                else:
                    records[row["indices"][0]]["y_family"] = 1
                with self.assertRaises(ValueError):
                    build_plan(records, splits, direct, orders, config)

    def test_protocol_rejects_changed_hyperparameters(self):
        config = config_fixture()
        validate_config(config)
        for key, value in (("hidden_dim", 32), ("alpha", 0.1), ("batch_size", 64), ("epochs", 121)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**config, key: value})


class PoolingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    @staticmethod
    def feature_fixture():
        torch.manual_seed(75)
        count, atoms = 3, 6
        x = torch.randn(count, atoms, 18)
        x[:, :, 13:] = 0
        for channel, atom in ((13, 0), (14, 1), (15, 2), (16, 3), (17, 4)):
            x[:, atom, channel] = 1
        mask = torch.ones(count, atoms)
        mask[:, -1] = 0
        adjacency = torch.zeros(count, atoms, atoms)
        for atom in range(atoms - 2):
            adjacency[:, atom, atom + 1] = adjacency[:, atom + 1, atom] = 1
        records = [{"id": f"m{index}", "target_fragment_indices": [2, 3], "target_attachment_idx": 2}
                   for index in range(count)]
        return records, {"x": x, "adj": adjacency, "mask": mask}

    def test_full_input_branch_exact_and_encoder_unchanged(self):
        records, tensors = self.feature_fixture()
        model = SkipMolecularGNN.from_base(HardMolecularGNN(64, 8), "shallow_skip")
        model.train()
        before = encoder_state_sha256(model.state_dict())
        membership = target_masks(records, tensors["mask"], [0, 1, 2])
        features = extract_first_layer(model, tensors, [0, 1, 2], membership)
        chosen = np.resize(np.arange(3), 128)
        with torch.no_grad():
            expected = model({key: value[chosen] for key, value in tensors.items()}, representations=True)["appended_branch"][:3]
        np.testing.assert_array_equal(features["whole_graph"], expected.numpy())
        self.assertEqual(before, encoder_state_sha256(model.state_dict()))
        self.assertTrue(model.training)
        for key in tensors:
            np.testing.assert_array_equal(tensors[key].numpy(), self.feature_fixture()[1][key].numpy())

    def test_membership_changes_pool_only_and_keeps_remote_message_context(self):
        records, tensors = self.feature_fixture()
        model = HardMolecularGNN(64, 8)
        selected = [0, 1, 2]
        membership = target_masks(records, tensors["mask"], selected)
        first = extract_first_layer(model, tensors, selected, membership)
        changed = membership.copy()
        changed[:, 0] = True
        second = extract_first_layer(model, tensors, selected, changed)
        np.testing.assert_array_equal(first["whole_graph"], second["whole_graph"])
        self.assertGreater(np.max(np.abs(first["target"] - second["target"])), 0)
        distant_changed = {key: value.clone() for key, value in tensors.items()}
        distant_changed["x"][:, 1, :13] += 3
        third = extract_first_layer(model, distant_changed, selected, membership)
        self.assertGreater(np.max(np.abs(first["target"] - third["target"])), 0)

    def test_oracle_pool_rejects_padding_duplicates_and_missing_root(self):
        records, tensors = self.feature_fixture()
        for fragment in ([2, 2], [3], [2, 5], [2, 6]):
            with self.subTest(fragment=fragment), self.assertRaises(ValueError):
                changed = copy.deepcopy(records)
                changed[0]["target_fragment_indices"] = fragment
                target_masks(changed, tensors["mask"], [0, 1, 2])

    def test_exact_archive_validation_does_not_tolerate_small_changes(self):
        records = [{"id": "m0"}]
        values = np.zeros((1, 64), dtype=np.float32)
        self.assertTrue(exact_feature_check("m", "r", values, values, [0], records)["exact_match"])
        altered = values.copy()
        altered[0, 0] = 1e-8
        with self.assertRaises(AssertionError):
            exact_feature_check("m", "r", values, altered, [0], records)


class PoolingReadoutTests(unittest.TestCase):
    def test_ridge_matches_archived_algorithm_and_support_only_scaling(self):
        rng = np.random.default_rng(45)
        train, test = rng.normal(size=(48, 64)), rng.normal(size=(12, 64))
        train[:, 0] = test[:, 0] = 3
        labels, truth = np.arange(48) % 8, np.arange(12) % 8
        fitted = ridge_fit(train, labels)
        expected = fit_ridge_readout(train, labels, test, truth)
        predicted = (((test - fitted["mean"]) / fitted["scale"]) @ fitted["weights"] + fitted["bias"]).argmax(1)
        np.testing.assert_array_equal(predicted, expected["predictions"])
        np.testing.assert_array_equal(fitted["mean"], train.mean(0))
        self.assertEqual(fitted["scale"][0], 1)
        self.assertFalse(np.array_equal(fitted["mean"], np.concatenate((train, test + 100)).mean(0)))

    def test_initializer_is_identical_across_conditions_without_global_rng_change(self):
        torch.manual_seed(56)
        before = torch.random.get_rng_state().clone()
        first, second = new_head(1), new_head(1)
        self.assertEqual(state_sha256(first.state_dict()), state_sha256(second.state_dict()))
        self.assertNotEqual(state_sha256(first.state_dict()), state_sha256(new_head(2).state_dict()))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(sum(parameter.numel() for parameter in first.parameters()), 4680)

    def test_mlp_training_matches_one_independent_standardized_update(self):
        config = config_fixture()
        rng = np.random.default_rng(34)
        values = rng.normal(size=(6, 64)).astype(np.float32)
        labels, indices = np.arange(6), [13, 17, 21, 25, 29, 33]
        order = np.asarray([[33, 29, 25, 21, 17, 13]], dtype=np.int32)
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            fitted, head = mlp_fit(values, labels, indices, order, 0, config, torch.device("cpu"))
            reference = new_head(0)
            mean, scale = standardizer(values)
            optimizer = torch.optim.AdamW(reference.parameters(), lr=.0015, weight_decay=.0001)
            rows = [indices.index(int(index)) for index in order[0]]
            z = torch.from_numpy(((values.astype(np.float64) - mean) / scale).astype(np.float32))
            loss = nn.functional.cross_entropy(reference(z[rows]), torch.tensor(labels[rows]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 5)
            optimizer.step()
            self.assertEqual(fitted["optimizer_updates"], 1)
            self.assertEqual(state_sha256(head.state_dict()), state_sha256(reference.state_dict()))
            self.assertEqual(fitted["initial_state_dict"].keys(), new_head(0).state_dict().keys())
            self.assertEqual(mlp_scores(head, values, mean, scale).shape, (6, 8))
            with self.assertRaises(ValueError):
                mlp_fit(values, labels, indices, np.asarray([[33, 29, 25, 21, 17, 999]]), 0, config, torch.device("cpu"))
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
