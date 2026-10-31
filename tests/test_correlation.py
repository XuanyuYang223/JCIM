"""Guard intervention isolation, matched initialization, and archived zero controls."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcim import correlation
from jcim.hard_data import heldout_cell
from jcim.hard_model import HardMolecularGNN
from jcim.run import sha256, write_csv, write_json


def synthetic_design():
    """Eight labels and three position cells with distinct identity partitions."""
    records = []
    splits = {name: [] for name in ("candidate_train", "train_a", "train_b", "val", "val_a", "val_b",
                                   "discarded_val_ood", "adaptation_reserve", "analysis", "ood_common",
                                   "iid_a", "iid_b")}
    for motif in range(8):
        for position in range(3):
            cell = (motif + position) % 3
            for partition, count in (("candidate_train", 20), ("validation", 3),
                                     ("adaptation_reserve", 2), ("analysis", 4)):
                for replicate in range(count):
                    index = len(records)
                    records.append({"id": f"{motif}_{position}_{partition}_{replicate:02d}",
                                    "y_family": motif, "y_position": position, "y_distance": 4})
                    if partition == "candidate_train":
                        splits[partition].append(index)
                        if cell < 2:
                            splits[("train_a", "train_b")[cell]].append(index)
                    elif partition == "validation":
                        if cell < 2:
                            splits["val"].append(index)
                            splits[("val_a", "val_b")[cell]].append(index)
                        else:
                            splits["discarded_val_ood"].append(index)
                    elif partition == "analysis":
                        splits["analysis"].append(index)
                        splits[("iid_a", "iid_b", "ood_common")[cell]].append(index)
                    else:
                        splits[partition].append(index)
    return records, splits


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.records, self.splits = synthetic_design()

    def test_nested_same_motif_replacements_preserve_counts_and_odd_identities(self):
        plan = correlation.build_intervention_plan(self.records, self.splits)
        self.assertEqual([item["replaced_count"] for item in plan["conditions"]["a"]], [0, 2, 8, 32])
        for condition, selections in plan["conditions"].items():
            base = set(self.splits[f"train_{condition}"])
            base_odd = {i for i in base if self.records[i]["y_family"] % 2}
            prior_added, prior_removed = [], []
            for item in selections:
                count = item["replaced_count"]
                self.assertEqual(item["added_indices"][:len(prior_added)], prior_added)
                self.assertEqual(item["removed_indices"][:len(prior_removed)], prior_removed)
                self.assertEqual(len(set(item["train_indices"])), len(base))
                self.assertEqual(len(set(item["train_indices"]) - base), count)
                self.assertEqual(len(base - set(item["train_indices"])), count)
                self.assertEqual(item["motif_counts"], selections[0]["motif_counts"])
                self.assertTrue(base_odd <= set(item["train_indices"]))
                for old, new in zip(item["removed_indices"], item["added_indices"]):
                    self.assertEqual(self.records[old]["y_family"], self.records[new]["y_family"])
                    self.assertEqual(self.records[new]["y_family"] % 2, 0)
                    self.assertEqual(heldout_cell(self.records[new]), 2)
                for motif in correlation.EXPOSED_MOTIFS:
                    self.assertLessEqual(abs(sum(self.records[i]["y_family"] == motif for i in item["added_indices"])
                                             - count / 4), 1)
                prior_added, prior_removed = item["added_indices"], item["removed_indices"]
        for left, right in zip(plan["conditions"]["a"], plan["conditions"]["b"]):
            self.assertEqual(left["added_ids"], right["added_ids"])

    def test_interventions_never_use_protected_validation_reserve_or_test(self):
        plan = correlation.build_intervention_plan(self.records, self.splits)
        forbidden = set(self.splits["analysis"]) | set(self.splits["val"]) \
            | set(self.splits["adaptation_reserve"]) | set(self.splits["discarded_val_ood"])
        protected = set(plan["protected_all_indices"])
        for selections in plan["conditions"].values():
            for item in selections:
                self.assertFalse(set(item["train_indices"]) & (forbidden | protected))
                self.assertTrue(set(item["added_indices"]) <= set(self.splits["candidate_train"]))
        protected_test = plan["evaluation"]["protected_common"]
        self.assertEqual(set(protected_test["motif_counts"]), {"1", "3", "5", "7"})
        self.assertEqual(set(protected_test["position_counts"]), {"0", "1", "2"})
        self.assertEqual(plan["prediction_classes"], list(range(8)))

    def test_real_hard_v2_plan_has_exact_predeclared_replacement_counts(self):
        source = correlation.ROOT / "experiments/hard_v2"
        records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
        splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
        plan = correlation.build_intervention_plan(records, splits, expected_count=1386)
        self.assertEqual(len(records), 6912)
        for selections in plan["conditions"].values():
            self.assertEqual([item["replaced_count"] for item in selections], [0, 14, 69, 277])
            self.assertTrue(all(item["n_train"] == 1386 for item in selections))

    def test_corrupted_identity_partitions_and_wrong_size_are_rejected(self):
        corrupted = copy.deepcopy(self.splits)
        corrupted["candidate_train"].append(corrupted["analysis"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            correlation.build_intervention_plan(self.records, corrupted)
        with self.assertRaisesRegex(ValueError, "equal training size"):
            correlation.build_intervention_plan(self.records, self.splits, expected_count=1386)
        corrupted = copy.deepcopy(self.splits)
        protected = next(i for i in corrupted["ood_common"] if self.records[i]["y_family"] % 2)
        corrupted["val_a"].append(protected)
        with self.assertRaisesRegex(ValueError, "validation"):
            correlation.build_intervention_plan(self.records, corrupted)

    def test_invalid_rates_and_insufficient_candidates_are_rejected(self):
        for rates in ([0.01], [0, 0.2, 0.1], [0, float("nan")], [0, 0.2, 0.2], [0, 1.1]):
            with self.assertRaises(ValueError):
                correlation.build_intervention_plan(self.records, self.splits, rates)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            correlation.build_intervention_plan(self.records, self.splits, [0, 0.9])

    def test_four_true_classes_do_not_restrict_the_eight_class_chance(self):
        plan = correlation.build_intervention_plan(self.records, self.splits)
        rows = correlation.baseline_rows(self.records, self.splits["train_a"],
                                         {"protected": plan["evaluation"]["protected_common"]["indices"]}, "a", 0)
        row = rows[0]
        self.assertEqual(row["n_true_classes"], 4)
        self.assertEqual(row["n_prediction_classes"], 8)
        self.assertEqual(row["uniform_accuracy"], 0.125)
        self.assertEqual(row["uniform_macro_recall"], 0.125)
        self.assertEqual(row["training_prior_expected_accuracy"], 0.125)
        self.assertEqual(row["training_majority_label"], 0)
        self.assertEqual(row["training_majority_accuracy"], 0)

    def test_summary_uses_sample_standard_deviation_across_seeds(self):
        rows = []
        for seed, value in enumerate([0.0, 0.5, 1.0]):
            row = {"condition": "a", "rate": 0.2, "split": "protected_common", "seed": seed,
                   "replaced_count": 277, "n": 10}
            row.update({key: value for key in ("motif_accuracy", "motif_balanced_accuracy",
                                               "position_accuracy", "distance_accuracy", "joint_accuracy")})
            rows.append(row)
        summary = correlation.summarize_behavior(rows)[0]
        self.assertEqual(summary["n_seeds"], 3)
        self.assertEqual(summary["motif_accuracy_mean"], 0.5)
        self.assertEqual(summary["motif_accuracy_std"], 0.5)

    def test_run_refuses_overwrite_before_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                correlation.run_correlation(Path(directory) / "missing", output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_run_reuses_zero_bytes_and_matched_archived_initialization(self):
        config = {"seeds": [3], "hidden_dim": 4, "layers": 1, "epochs": 1,
                  "batch_size": 128, "learning_rate": 0.0015, "weight_decay": 0.0001,
                  "cpu_threads": 2, "device": "cpu", "correlation_rates": [0, 0.05, 0.2],
                  "correlation_train_count": len(self.splits["train_a"])}
        torch.manual_seed(91)
        initial = HardMolecularGNN(4, 1)
        initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
        calls = []

        def train_spy(model, tensors, train, val, used_config, seed, output, name):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, initial_state[key], rtol=0, atol=0)
            self.assertEqual(len(train), config["correlation_train_count"])
            self.assertEqual(seed, 3)
            self.assertTrue(all(heldout_cell(self.records[i]) != 2 or self.records[i]["y_family"] % 2 == 0
                                for i in train))
            self.assertFalse(set(train) & set(self.splits["analysis"]))
            condition = name.split("_")[0]
            self.assertEqual(val, self.splits[f"val_{condition}"])
            calls.append((name, list(train)))
            with torch.no_grad():
                next(model.parameters()).add_(0.3)
            return 0.0

        def infer_spy(model, tensors, indices, batch_size, representations):
            predicted = {task: np.array([self.records[i][f"y_{task}"] for i in indices])
                         for task in ("family", "position", "distance")}
            values = np.ones((len(indices), 4), dtype=np.float32)
            return predicted, {"local": [values], "graph": [values]}

        original_threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as directory:
                source, output = Path(directory) / "source", Path(directory) / "output"
                (source / "checkpoints").mkdir(parents=True)
                (source / "representations").mkdir()
                write_json(source / "config.json", config)
                write_json(source / "splits.json", self.splits)
                write_json(source / "MANIFEST.json", {"sha256": {}})
                with (source / "molecules.jsonl").open("w", encoding="utf-8") as handle:
                    for record in self.records:
                        handle.write(json.dumps(record) + "\n")
                np.savez_compressed(source / "graphs.npz", x=np.zeros((len(self.records), 1, 18), dtype=np.float32))
                torch.save({"state_dict": initial_state}, source / "checkpoints/random_seed3.pt")
                prediction_rows = []
                for condition in ("a", "b"):
                    name = f"{condition}_seed3"
                    torch.save({"state_dict": initial_state, "frozen_source": name}, source / "checkpoints" / f"{name}.pt")
                    np.savez_compressed(source / "representations" / f"{name}.npz", local_0=np.array([2, 5, 7]))
                    for i in self.splits["analysis"]:
                        prediction_rows.append({"model": name, "id": self.records[i]["id"],
                                                **{f"pred_{task}": self.records[i][f"y_{task}"]
                                                   for task in ("family", "position", "distance")}})
                write_csv(source / "predictions.csv", prediction_rows)
                hashes = {"checkpoints/random_seed3.pt": sha256(source / "checkpoints/random_seed3.pt")}
                with patch.object(correlation, "_verify_source_files", return_value=({}, hashes)), \
                        patch.object(correlation, "train_model", side_effect=train_spy), \
                        patch.object(correlation, "infer", side_effect=infer_spy), \
                        patch.object(correlation, "generate_correlation_report"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    rows = correlation.run_correlation(source, output)
                self.assertEqual(len(calls), 4)
                self.assertEqual(len(rows), 18)
                self.assertTrue(all(row["motif_accuracy"] == 1 for row in rows))
                for condition in ("a", "b"):
                    for subdirectory, suffix in (("checkpoints", "pt"), ("representations", "npz")):
                        self.assertEqual((source / subdirectory / f"{condition}_seed3.{suffix}").read_bytes(),
                                         (output / subdirectory / f"{condition}_p0_seed3.{suffix}").read_bytes())
                self.assertTrue((output / "MANIFEST.json").is_file())
        finally:
            torch.set_num_threads(original_threads)

    def test_report_has_unrestricted_chance_and_every_seed(self):
        plan = correlation.build_intervention_plan(self.records, self.splits, [0, 0.2])
        rows = []
        for condition in ("a", "b"):
            for seed in (0, 1, 2):
                for rate in (0, 0.2):
                    for split in ("iid", "exposed_common", "protected_common"):
                        row = {"condition": condition, "seed": seed, "rate": rate, "split": split,
                               "replaced_count": int(160 * rate), "n": 16}
                        row.update({metric: 0.5 for metric in ("motif_accuracy", "motif_balanced_accuracy",
                                                              "position_accuracy", "distance_accuracy", "joint_accuracy")})
                        rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_csv(output / "behavior.csv", rows)
            write_json(output / "config.json", {"seeds": [0, 1, 2], "epochs": 120})
            write_json(output / "selection_audit.json", plan)
            correlation.generate_correlation_report(output)
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("12.5%, not 25%", report)
            self.assertIn("## Every seed", report)
            self.assertIn("Protected", (output / "figures/correlation_summary.svg").read_text(encoding="utf-8"))
            self.assertTrue((output / "figures/correlation_summary.png").stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()
