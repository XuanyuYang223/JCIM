"""Integration checks for matched experiences and identity-aligned evaluation."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcim import run
from jcim.data import audit_splits, generate_dataset, split_dataset
from jcim.model import MolecularGNN, tensorize


class RunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The small chemistry grid is sufficient to exercise every split type.
        cls.records = generate_dataset(spacer_lengths=(0, 1))
        cls.splits = split_dataset(cls.records)

    def test_matching_preserves_shared_molecules_and_balances_exclusives(self):
        splits = deepcopy(self.splits)
        before = {name: set(splits[name]) for name in ("train_a", "train_b")}
        unchanged = {name: deepcopy(value) for name, value in splits.items()
                     if name not in before}
        matching = run.match_training_experiences(self.records, splits)
        after = {name: set(splits[name]) for name in before}
        shared = before["train_a"] & before["train_b"]
        self.assertEqual(after["train_a"] & after["train_b"], shared)
        self.assertEqual(len(after["train_a"]), len(after["train_b"]))
        for name in before:
            self.assertTrue(after[name] <= before[name])
            self.assertEqual(len(after[name] - shared), matching["exclusive_per_experience"])
            self.assertEqual(len(splits[name]), len(after[name]))
            self.assertEqual(unchanged, {key: splits[key] for key in unchanged})
            removed_ids = {self.records[i]["id"] for i in before[name] - after[name]}
            self.assertEqual(removed_ids, set(matching[f"dropped_{name}_ids"]))
        self.assertFalse(audit_splits(self.records, splits)["split_identity_leakage"])

        reversed_records = list(reversed(self.records))
        reversed_splits = split_dataset(reversed_records)
        run.match_training_experiences(reversed_records, reversed_splits)
        for name in before:
            self.assertEqual({self.records[i]["id"] for i in splits[name]},
                             {reversed_records[i]["id"] for i in reversed_splits[name]})

    def test_post_matching_audit_rejects_partition_and_support_mutations(self):
        splits = deepcopy(self.splits)
        run.match_training_experiences(self.records, splits)
        audit_splits(self.records, splits)
        with self.subTest("training overlaps common test"):
            changed = deepcopy(splits)
            changed["train_a"].append(changed["analysis"][0])
            with self.assertRaises(ValueError):
                audit_splits(self.records, changed)
        with self.subTest("adaptation reuses test identity"):
            changed = deepcopy(splits)
            changed["adaptation_pool"].append(changed["adaptation_test"][0])
            with self.assertRaisesRegex(ValueError, "Adaptation training and test"):
                audit_splits(self.records, changed)

    def test_adaptation_uses_nested_supports_with_aligned_disjoint_test_rows(self):
        records = [{"y_distance": i % 12} for i in range(10)]
        indices = [8, 0, 7, 2, 9, 1, 6, 5, 3, 4]
        features = np.column_stack([indices, np.array(indices) + 100]).astype(float)
        splits = {"adaptation_pool": [6, 0, 5, 2, 4, 1], "adaptation_test": [9, 7, 8]}
        config = {"adaptation_repeats": 2, "adaptation_shots": [2, 4, 6], "probe_alpha": 1.0}
        calls = []

        def checked_fit(train_x, train_y, test_x, test_y, **kwargs):
            support_ids = train_x[:, 0].astype(int)
            test_ids = test_x[:, 0].astype(int)
            self.assertEqual(test_ids.tolist(), splits["adaptation_test"])
            self.assertFalse(set(support_ids) & set(test_ids))
            np.testing.assert_array_equal(train_y, [records[i]["y_distance"] for i in support_ids])
            np.testing.assert_array_equal(test_y, [records[i]["y_distance"] for i in test_ids])
            calls.append(support_ids.tolist())
            return {"accuracy": 0.5, "balanced_accuracy": 0.5}

        reps = {"graph": [features]}
        with patch.object(run, "fit_readout", side_effect=checked_fit):
            rows = run.adaptation_rows("a", "a", 0, reps, indices, splits, records, config)
        self.assertEqual(len(rows), 6)
        for first in (0, 3):
            self.assertEqual(calls[first], calls[first + 1][:2])
            self.assertEqual(calls[first + 1], calls[first + 2][:4])
        previous = list(calls)
        calls.clear()
        with patch.object(run, "fit_readout", side_effect=checked_fit):
            run.adaptation_rows("random", "random", 19, reps, indices, splits, records, config)
        self.assertEqual(calls, previous)
        with patch.object(run, "fit_readout", side_effect=checked_fit):
            with self.assertRaisesRegex(ValueError, "budget exceeds"):
                run.adaptation_rows("a", "a", 0, reps, indices, splits, records,
                                    {**config, "adaptation_shots": [7]})

    def test_inference_keeps_requested_molecule_order_across_batches(self):
        records = self.records[:5]
        tensors = tensorize(records, torch.device("cpu"))
        torch.manual_seed(83)
        model = MolecularGNN(hidden_dim=8, layers=1).eval()
        indices = [4, 1, 3, 0]
        predicted, representations = run.infer(model, tensors, indices, batch_size=2,
                                               representations=True)
        for row, index in enumerate(indices):
            single_predicted, single_reps = run.infer(model, tensors, [index], representations=True)
            for task in predicted:
                self.assertEqual(predicted[task][row], single_predicted[task][0])
            for pooling in ("local", "graph"):
                for layer, values in enumerate(representations[pooling]):
                    np.testing.assert_allclose(values[row], single_reps[pooling][layer][0],
                                               rtol=2e-5, atol=2e-6)

    def test_main_uses_identical_initial_states_and_shared_evaluation_rows(self):
        # Exercise orchestration without optimizer steps or actual model training.
        records = self.records
        initial_states = {}
        checked_pairs = []
        checked_probe_models = []

        def fake_train(model, tensors, train_indices, val_indices, config, seed, output, name):
            initial_states[name] = {key: value.clone() for key, value in model.state_dict().items()}
            # A mutation exposes accidental state sharing between experiences.
            with torch.no_grad():
                next(model.parameters()).add_(3)
            return 0.0

        def fake_infer(model, tensors, indices, batch_size=128, representations=False):
            self.assertTrue(representations)
            predicted = {task: np.array([records[i][f"y_{task}"] for i in indices])
                         for task in ("family", "position", "distance")}
            marker = np.column_stack([indices, np.array(indices) + 1000]).astype(float)
            return predicted, {"local": [marker], "graph": [marker]}

        def checked_probes(name, condition, seed, reps, indices, records_arg, alpha):
            if condition != "raw":
                for pooling in ("local", "graph"):
                    np.testing.assert_array_equal(reps[pooling][0][:, 0], indices)
                checked_probe_models.append(name)
            return []

        def checked_similarity(first, second):
            np.testing.assert_array_equal(first[0], second[0])
            checked_pairs.append(first[0].copy())
            return [{"layer": 0, "cka": 1.0, "procrustes_similarity": 1.0}]

        previous_threads = torch.get_num_threads()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"seeds": [17], "hidden_dim": 8, "layers": 1, "epochs": 1,
                      "batch_size": 16, "device": "cpu", "cpu_threads": 1,
                      "split_seed": 20261030, "probe_alpha": 1.0,
                      "adaptation_shots": [2], "adaptation_repeats": 1}
            config_path, output = root / "config.json", root / "result"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = ["jcim.run", "--config", str(config_path), "--output", str(output)]
            try:
                with patch.object(sys, "argv", args), \
                     patch.object(run, "generate_dataset", return_value=records), \
                     patch.object(run, "train_model", side_effect=fake_train), \
                     patch.object(run, "infer", side_effect=fake_infer), \
                     patch.object(run, "make_probes", side_effect=checked_probes), \
                     patch.object(run, "adaptation_rows", return_value=[]), \
                     patch.object(run, "layerwise_similarity", side_effect=checked_similarity), \
                     patch("jcim.report.generate_report"), redirect_stdout(io.StringIO()):
                    run.main()
            finally:
                torch.set_num_threads(previous_threads)
            random_state = torch.load(output / "checkpoints/random_seed17.pt",
                                      map_location="cpu", weights_only=True)["state_dict"]
            self.assertEqual(set(initial_states), {"a_seed17", "b_seed17"})
            for state in initial_states.values():
                for key, values in random_state.items():
                    torch.testing.assert_close(state[key], values, rtol=0, atol=0)
            self.assertEqual(checked_probe_models, ["random_seed17", "a_seed17", "b_seed17"])
            self.assertEqual(len(checked_pairs), 3)
            archived_splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            for pair in checked_pairs:
                np.testing.assert_array_equal(pair[:, 0], archived_splits["analysis"])
            with (output / "behavior.csv").open(newline="", encoding="utf-8") as handle:
                behavior = list(csv.DictReader(handle))
            self.assertTrue(behavior)
            for row in behavior:
                for task in ("family", "position", "distance"):
                    self.assertEqual(float(row[f"{task}_accuracy"]), 1.0)
                self.assertEqual(float(row["distance_mae"]), 0.0)
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                prediction_rows = list(csv.DictReader(handle))
            self.assertEqual(len(prediction_rows), 3 * len(archived_splits["analysis"]))
            by_id = {record["id"]: record for record in records}
            for row in prediction_rows:
                for task in ("family", "position", "distance"):
                    expected = by_id[row["id"]][f"y_{task}"]
                    self.assertEqual(int(row[f"true_{task}"]), expected)
                    self.assertEqual(int(row[f"pred_{task}"]), expected)
            representation_indices = archived_splits["analysis"] + archived_splits["adaptation_pool"]
            with np.load(output / "representations/random_seed17.npz") as archived:
                self.assertEqual(archived["ids"].tolist(),
                                 [records[i]["id"] for i in representation_indices])
                np.testing.assert_array_equal(archived["local_0"][:, 0], representation_indices)
            self.assertTrue((output / "MANIFEST.json").exists())


if __name__ == "__main__":
    unittest.main()
