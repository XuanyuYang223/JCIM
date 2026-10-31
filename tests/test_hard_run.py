"""Check the harder model and full-model adaptation against scientific shortcuts."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcim import hard_run
from jcim.hard_data import N_DISTANCE_CLASSES, generate_hard_dataset, split_hard_dataset
from jcim.hard_model import HardMolecularGNN, tensorize_hard
from jcim.model import pool_nodes


def cloned_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class HardRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.records = generate_hard_dataset((0,))
        cls.splits = split_hard_dataset(cls.records)
        cls.cpu = torch.device("cpu")
        cls.tensors = tensorize_hard(cls.records, cls.cpu)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def assert_states_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0,
                                       msg=lambda message, key=key: f"{key}: {message}")

    def adaptation_config(self):
        return {"hidden_dim": 12, "layers": 2, "batch_size": 128,
                "adaptation_shots": [4, 8], "adaptation_repeats": 2,
                "adaptation_steps": 3, "adaptation_batch_size": 8,
                "adaptation_learning_rate": 0.01, "weight_decay": 0.0}

    def test_finetuning_updates_encoder_and_every_head_with_sixteen_distance_classes(self):
        torch.manual_seed(41)
        model = HardMolecularGNN(hidden_dim=12, layers=2)
        before = cloned_state(model)
        indices = np.linspace(0, len(self.records) - 1, 32, dtype=int)
        batch = {key: value[indices].clone() for key, value in self.tensors.items()}
        # Exercise the hard head's upper classes without pretending they are molecular labels.
        batch["y_distance"] = torch.arange(len(indices)) % N_DISTANCE_CLASSES
        self.assertEqual(model(batch)["distance"].shape, (32, 16))
        config = self.adaptation_config()
        config["adaptation_batch_size"] = 32
        hard_run.finetune_model(model, batch, list(range(32)), config, repeat=0)
        after = model.state_dict()
        for key in before:
            self.assertFalse(torch.equal(before[key], after[key]),
                             f"Full-model adaptation left parameter tensor {key} unchanged")

    def test_local_pool_uses_one_query_root_and_is_atom_order_and_padding_invariant(self):
        record = self.records[73]
        perm = np.random.default_rng(91).permutation(len(record["x"]))
        permuted = dict(record, x=record["x"][perm], adj=record["adj"][np.ix_(perm, perm)])
        batch = tensorize_hard([record], self.cpu)
        self.assertEqual(float(batch["x"][0, :, 15].sum()), 1.0)
        # Pooling is a selection of the marked atom, even if other atoms have large values.
        h = torch.arange(batch["x"].shape[1] * 3, dtype=torch.float32).reshape(1, -1, 3)
        local, _ = pool_nodes(h, batch["x"], batch["mask"])
        torch.testing.assert_close(local[0], h[0, record["target_attachment_idx"]])
        torch.manual_seed(12)
        model = HardMolecularGNN(hidden_dim=12, layers=3).eval()
        with torch.no_grad():
            original = model(batch, representations=True)
            shuffled = model(tensorize_hard([permuted], self.cpu), representations=True)
            padded = model(tensorize_hard([record, max(self.records, key=lambda r: len(r["x"]))],
                                          self.cpu), representations=True)
        for task in ("family", "position", "distance"):
            torch.testing.assert_close(original[task], shuffled[task], atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(original[task][0], padded[task][0], atol=3e-6, rtol=3e-6)
        for before, shuffled_layer, padded_layer in zip(original["local"], shuffled["local"],
                                                        padded["local"]):
            torch.testing.assert_close(before, shuffled_layer, atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(before[0], padded_layer[0], atol=3e-6, rtol=3e-6)

    def test_metric_joint_requires_all_three_tasks_on_the_same_molecule(self):
        records = [{"y_family": f, "y_position": p, "y_distance": d}
                   for f, p, d in [(0, 0, 13), (0, 1, 14), (1, 2, 15), (1, 0, 13)]]
        indices = [3, 0, 2, 1]
        predictions = {"family": np.array([1, 0, 1, 1]),
                       "position": np.array([0, 0, 2, 2]),
                       "distance": np.array([12, 13, 14, 14])}
        row = hard_run.metric_row("m", "a", 0, "ood_common", predictions, indices, records)
        self.assertEqual(row["motif_accuracy"], 0.75)
        self.assertEqual(row["motif_balanced_accuracy"], 0.75)
        self.assertEqual(row["position_accuracy"], 0.75)
        self.assertEqual(row["distance_accuracy"], 0.5)
        self.assertEqual(row["distance_mae"], 0.5)
        self.assertEqual(row["joint_accuracy"], 0.25)

    def test_adaptation_restarts_inherited_weights_and_preserves_nested_common_support(self):
        torch.manual_seed(7)
        model = HardMolecularGNN(hidden_dim=12, layers=2)
        state = cloned_state(model)
        untouched = {key: value.clone() for key, value in state.items()}
        config = self.adaptation_config()
        actual_finetune = hard_run.finetune_model
        seen_support = []

        def checked_finetune(adapted, tensors, support, used_config, repeat):
            self.assert_states_equal(state, adapted.state_dict())
            seen_support.append((repeat, tuple(support)))
            actual_finetune(adapted, tensors, support, used_config, repeat)
            self.assertTrue(any(not torch.equal(state[key], value)
                                for key, value in adapted.state_dict().items()))

        def metric_only_infer(adapted, tensors, indices, batch_size):
            return {task: tensors[f"y_{task}"][indices].numpy()
                    for task in ("family", "position", "distance")}, None

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoints").mkdir()
            (output / "adaptation_support").mkdir()
            with patch.object(hard_run, "finetune_model", side_effect=checked_finetune), \
                    patch.object(hard_run, "infer", side_effect=metric_only_infer):
                rows_a = hard_run.run_adaptation("a_seed7", "a", 7, state, self.tensors,
                                                 self.records, self.splits, config, output)
                rows_b = hard_run.run_adaptation("b_seed7", "b", 7, state, self.tensors,
                                                 self.records, self.splits, config, output)
            self.assertEqual(len(rows_a), 2 * 2 * 3)
            self.assertEqual(len(rows_a), len(rows_b))
            self.assertEqual(seen_support[:4], seen_support[4:])
            forbidden = set(self.splits["train_a"]) | set(self.splits["train_b"]) \
                | set(self.splits["analysis"]) | set(self.splits["val"])
            for repeat in range(config["adaptation_repeats"]):
                small = seen_support[repeat * 2][1]
                large = seen_support[repeat * 2 + 1][1]
                self.assertEqual(small, large[:len(small)])
                self.assertFalse(set(large) & forbidden)
                self.assertTrue(set(large) <= set(self.splits["adaptation_pool"]))
                for shots in config["adaptation_shots"]:
                    saved = json.loads((output / "adaptation_support" /
                                        f"repeat{repeat}_shots{shots}.json").read_text())
                    self.assertEqual(saved["molecule_ids"],
                                     [self.records[i]["id"] for i in large[:shots]])
            self.assertEqual(len(list((output / "checkpoints").glob("*.pt"))), 8)
        self.assert_states_equal(untouched, state)

    def test_main_starts_random_a_and_b_from_one_seed_snapshot(self):
        config = self.adaptation_config()
        config.update({"split_seed": 20261030, "seeds": [23], "epochs": 1,
                       "device": "cpu", "cpu_threads": 2, "geometry_examples": 16,
                       "probe_alpha": 1.0})
        observed_starts, inherited = {}, {}

        def training_spy(model, tensors, train, val, used_config, seed, output, name):
            observed_starts[name] = cloned_state(model)
            condition = name.split("_")[0]
            self.assertEqual(train, self.splits[f"train_{condition}"])
            self.assertEqual(val, self.splits[f"val_{condition}"])
            with torch.no_grad():
                next(model.parameters()).add_(0.01 if condition == "a" else 0.02)
            return 0.0

        def adaptation_spy(name, condition, seed, state, *args):
            inherited[name] = {key: value.clone() for key, value in state.items()}
            return []

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = Path(directory) / "run"
            argv = ["hard_run", "--config", str(config_path), "--output", str(output)]
            with patch("sys.argv", argv), \
                    patch.object(hard_run, "generate_hard_dataset", return_value=self.records), \
                    patch.object(hard_run, "archive_dataset"), \
                    patch.object(hard_run, "make_probes", return_value=[]), \
                    patch.object(hard_run, "train_model", side_effect=training_spy), \
                    patch.object(hard_run, "run_adaptation", side_effect=adaptation_spy), \
                    patch("jcim.hard_report.generate_hard_report"), \
                    contextlib.redirect_stdout(io.StringIO()):
                hard_run.main()
            self.assertEqual(set(inherited), {"random_seed23", "a_seed23", "b_seed23"})
            self.assert_states_equal(inherited["random_seed23"], observed_starts["a_seed23"])
            self.assert_states_equal(inherited["random_seed23"], observed_starts["b_seed23"])
            self.assertFalse(torch.equal(inherited["a_seed23"]["embedding.weight"],
                                         inherited["b_seed23"]["embedding.weight"]))
            for name in inherited:
                saved = torch.load(output / "checkpoints" / f"{name}.pt", weights_only=True)
                self.assert_states_equal(inherited[name], saved["state_dict"])


if __name__ == "__main__":
    unittest.main()
