"""Output-effect diagnostics must distinguish label flips from accuracy changes."""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from jcim.branch_diagnostics import (TASKS, aggregate_seeds, logits_from_readouts,
                                     paired_effects, run_branch_diagnostics, summarize_effects)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_manifest(directory):
    manifest = {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    write_json(directory / "MANIFEST.json", {"sha256": manifest})


def source_fixture(directory):
    """Tiny complete panel with predictions made through independent nn heads."""
    skip, hard = directory / "skip", directory / "hard"
    skip.mkdir()
    hard.mkdir()
    (skip / "representations").mkdir()
    (skip / "checkpoints").mkdir()
    ids = np.asarray([f"m{index}" for index in range(4)])
    records = [{"id": identity, "y_family": index, "y_position": index % 3, "y_distance": index + 4}
               for index, identity in enumerate(ids.tolist())]
    (hard / "molecules.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    splits = {"analysis": [0, 1, 2, 3], "iid_a": [0], "iid_b": [1], "ood_common": [2, 3]}
    write_json(hard / "splits.json", splits)
    write_json(skip / "config.json", {"batch_size": 128, "seeds": [0], "epochs": 120})
    permutation = [2, 0, 3, 1]
    write_json(skip / "plan.json", {"tests": {"analysis": {"indices": [0, 1, 2, 3], "ids": ids.tolist()}},
                                    "branch_permutations": {"0": {"indices": permutation}}})
    inventory, predictions = [], []
    torch.manual_seed(217)
    for regime, variant, experience in itertools.product(("end_to_end", "frozen_readout"),
                                                        ("deep_control", "shallow_skip"), ("a", "b")):
        metadata = {"regime": regime, "variant": variant, "experience": experience, "seed": 0}
        inventory.append(metadata)
        name = f"{regime}_{variant}_{experience}_seed0"
        readout = torch.randn(4, 512)
        np.savez_compressed(skip / "representations" / f"{name}.npz", indices=np.arange(4), ids=ids,
                            readout=readout.numpy())
        heads = {task: nn.Sequential(nn.Linear(512, 5), nn.ReLU(), nn.Linear(5, classes))
                 for task, classes in zip(TASKS, (8, 3, 16))}
        state = {f"heads.{task}.{key}": value for task, head in heads.items() for key, value in head.state_dict().items()}
        torch.save({"state_dict": state}, skip / "checkpoints" / f"{name}.pt")
        for intervention in ("intact", "zero_branch", "permuted_branch"):
            values = readout.clone()
            if intervention != "intact":
                values[:, 448:] = 0 if intervention == "zero_branch" else readout[permutation, 448:]
            with torch.no_grad():
                padded = values[torch.arange(128) % 4]
                prediction = {task: head(padded)[:4].argmax(-1).numpy() for task, head in heads.items()}
            for index in range(4):
                predictions.append({**metadata, "intervention": intervention, "index": index, "id": str(ids[index]),
                                    **{f"pred_{task}": int(value[index]) for task, value in prediction.items()}})
    write_csv(skip / "inventory.csv", inventory)
    write_csv(skip / "predictions.csv", predictions)
    archive_manifest(skip)
    archive_manifest(hard)
    return skip, hard


class BranchDiagnosticsTests(unittest.TestCase):
    def test_equal_accuracy_can_hide_three_distinct_prediction_changes(self):
        intact = np.array([[4, 1, 0], [3, 2, 0], [2, 3, 0], [3, 2, 0]])
        altered = np.array([[0, 1, 4], [1, 3, 0], [0, 1, 4], [4, 2, 0]])
        effects = paired_effects(intact, altered, [0, 1, 0, 0])
        np.testing.assert_array_equal(effects["margin_intact"], [3, -1, -1, 1])
        np.testing.assert_array_equal(effects["margin_altered"], [-4, 2, -4, 2])
        np.testing.assert_array_equal(effects["margin_delta"], [-7, 3, -3, 1])
        summary = summarize_effects(effects, range(4))
        self.assertEqual(summary["intact_correct_rate"], .5)
        self.assertEqual(summary["altered_correct_rate"], .5)
        self.assertEqual(summary["prediction_changed_rate"], .75)
        for category in ("wrong_to_correct", "correct_to_wrong", "both_wrong_changed"):
            self.assertEqual(summary[f"{category}_count"], 1)
        self.assertEqual(summary["margin_delta_mean"], -1.5)
        self.assertEqual(summary["absolute_margin_delta_mean"], 3.5)
        self.assertAlmostEqual(summary["margin_delta_sd"], np.std([-7, 3, -3, 1], ddof=1))
        self.assertEqual(summary["margin_negative_to_positive_rate"], .25)
        self.assertEqual(summary["margin_positive_to_negative_rate"], .25)

    def test_common_logit_shift_is_removed_without_concealing_raw_shift(self):
        intact = np.array([[4., 1., 0.], [3., 2., 0.]])
        effects = paired_effects(intact, intact + 7, [0, 1])
        np.testing.assert_array_equal(effects["margin_delta"], [0, 0])
        np.testing.assert_array_equal(effects["centered_logit_l2"], [0, 0])
        np.testing.assert_allclose(effects["raw_logit_l2"], np.sqrt(147))
        self.assertTrue(effects["centered_logits_exactly_unchanged"].all())
        self.assertFalse(effects["logits_exactly_unchanged"].any())

    def test_ties_remain_distinct_from_positive_margins_and_argmax_correctness(self):
        effects = paired_effects([[2, 2, 0], [2, 2, 0]], [[2, 2, 0], [2, 2, 0]], [0, 1])
        np.testing.assert_array_equal(effects["intact_correct"], [True, False])
        self.assertTrue(effects["intact_margin_tie"].all())
        self.assertFalse(effects["margin_negative_to_positive"].any())

    def test_invalid_logits_and_labels_fail(self):
        for logits, altered, truth in (([[1, 2]], [[1, 2]], [2]), ([[1, np.nan]], [[1, 2]], [0]),
                                       ([[1, 2]], [[1, 2, 3]], [0]), ([[1]], [[1]], [0])):
            with self.subTest(logits=logits, altered=altered, truth=truth):
                with self.assertRaises(ValueError):
                    paired_effects(logits, altered, truth)

    def test_seed_aggregation_uses_seed_means_and_sample_sd(self):
        rows = [{"regime": "end_to_end", "variant": "shallow_skip", "experience": "a", "split": "ood_common",
                 "task": "family", "intervention": "zero_branch", "seed": seed, "n": 5,
                 "margin_delta_mean": value, "prediction_changed_count": seed, "prediction_changed_rate": seed / 5}
                for seed, value in enumerate((1., 3., 5.))]
        aggregate = aggregate_seeds(rows)[0]
        self.assertEqual(aggregate["margin_delta_mean_seed_mean"], 3)
        self.assertEqual(aggregate["margin_delta_mean_seed_sd"], 2)
        self.assertEqual(aggregate["n_seeds"], 3)
        self.assertNotIn("prediction_changed_count_seed_mean", aggregate)
        with self.assertRaises(ValueError):
            aggregate_seeds([rows[0], rows[0]])

    def test_fixed_shape_tail_matches_original_sequential_heads(self):
        torch.manual_seed(92)
        features = torch.randn(129, 512)
        heads = {task: nn.Sequential(nn.Linear(512, 4), nn.ReLU(), nn.Linear(4, count))
                 for task, count in zip(TASKS, (8, 3, 16))}
        state = {f"heads.{task}.{key}": value for task, head in heads.items() for key, value in head.state_dict().items()}
        actual = logits_from_readouts(state, features)
        with torch.no_grad():
            for task, head in heads.items():
                expected = torch.cat((head(features[:128]), head(features[128:].expand(128, -1))[:1])).numpy()
                np.testing.assert_array_equal(actual[task], expected)

    def test_complete_tiny_record_checks_every_saved_prediction_and_preserves_inputs(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                skip, hard = source_fixture(root)
                before = (skip / "MANIFEST.json").read_bytes()
                (root / "result").mkdir()
                result = run_branch_diagnostics(skip, hard, root / "result/branch")
                validation = result["validation"]
                self.assertEqual(validation["status"], "PASS")
                self.assertEqual(validation["case_count"], 8)
                self.assertEqual(validation["compared_rows"], 192)
                self.assertEqual(validation["prediction_rows_checked"], 288)
                self.assertEqual(validation["summary_rows"], 192)
                self.assertTrue(validation["pipeline_only"])
                with np.load(root / "result/branch/representations/branch_logits.npz") as dense:
                    self.assertEqual(len(dense.files), 74)
                    np.testing.assert_array_equal(dense["indices"], range(4))
                self.assertEqual((skip / "MANIFEST.json").read_bytes(), before)
                with self.assertRaises(FileExistsError):
                    run_branch_diagnostics(skip, hard, root / "result/branch")
                with self.assertRaises(ValueError):
                    run_branch_diagnostics(skip, hard, skip / "new_diagnostics")
        finally:
            torch.set_num_threads(previous)

    def test_manifest_failure_and_prediction_mismatch_are_distinct_failures(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                skip, hard = source_fixture(root)
                with (skip / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                rows[0]["pred_family"] = str((int(rows[0]["pred_family"]) + 1) % 8)
                write_csv(skip / "predictions.csv", rows)
                with self.assertRaisesRegex(ValueError, "manifest"):
                    run_branch_diagnostics(skip, hard, root / "invalid_hash")
                archive_manifest(skip)
                with self.assertRaisesRegex(AssertionError, "disagrees with frozen v5"):
                    run_branch_diagnostics(skip, hard, root / "invalid_prediction")
                subset = run_branch_diagnostics(skip, hard, root / "subset",
                                                {"experiences": ["b"], "regimes": ["frozen_readout"]})
                self.assertEqual(subset["validation"]["case_count"], 2)
                self.assertEqual(subset["validation"]["prediction_rows_checked"], 72)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
