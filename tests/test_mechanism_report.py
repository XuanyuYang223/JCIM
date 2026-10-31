"""Check nested replication, paired comparisons, and mechanism-report safeguards."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jcim.mechanism_report import ARM_NAMES, _validate_trajectory, generate_mechanism_report


def _write_json(directory: Path, filename: str, value: dict) -> None:
    (directory / filename).write_text(json.dumps(value), encoding="utf-8")


def _write_csv(directory: Path, filename: str, rows: list[dict]) -> None:
    with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MechanismReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="jcim_mechanism_report_test_")
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.output = Path(cls.temporary_directory.name)
        cls.config = {"seeds": [0, 1, 2], "support_repeats": [0, 1, 2], "primary_repeat": 0,
                      "evaluation_steps": [0, 80], "mechanism_steps": 80,
                      "adaptation_learning_rate": 0.0005, "new_count": 32, "old_count": 32}
        _write_json(cls.output, "config.json", cls.config)
        _write_json(cls.output, "environment.json", {"device": "cpu", "parameter_count": 189531})
        _write_json(cls.output, "audit.json", {"support_evaluation_overlap": False})
        _write_json(cls.output, "support_sets.json", {"sets": [
            {"experience": experience, "repeat": repeat,
             "new_ids": [f"new_{repeat}_{index}" for index in range(32)],
             "old_ids": [f"old_{experience}_{repeat}_{index}" for index in range(32)]}
            for experience in ("a", "b") for repeat in range(3)]})
        trajectory, drift, components, ridge = [], [], [], []
        bonuses = {"frozen_new": 0, "full_new": 0.01, "frozen_replay": 0.02, "full_replay": 0.03, "scratch_replay": -0.1}
        for experience in ("a", "b"):
            for seed in range(3):
                for repeat in range(3):
                    for arm, bonus in bonuses.items():
                        for step in (0, 80):
                            for split in ("iid_a", "iid_b", "ood_common"):
                                if step == 0:
                                    accuracy = 0.1 + 0.01 * seed if split == "ood_common" else 0.9 - 0.01 * seed
                                else:
                                    accuracy = (0.4 + 0.05 * seed + 0.03 * repeat if split == "ood_common"
                                                else 0.8 - 0.02 * seed - 0.04 * repeat) + bonus
                                trajectory.append({"experience": experience, "seed": seed, "repeat": repeat,
                                    "arm": arm, "step": step, "split": split, "n": 100,
                                    "motif_accuracy": accuracy, "position_accuracy": 0.95,
                                    "distance_accuracy": 0.7, "joint_accuracy": accuracy * 0.5,
                                    "distance_mae": 0.6})
                            frozen = arm.startswith("frozen_")
                            drift.append({"experience": experience, "seed": seed, "repeat": repeat,
                                "arm": arm, "step": step, "encoder_changed": step > 0 and not frozen,
                                "encoder_max_abs_delta": 0 if frozen or step == 0 else 0.02,
                                "head_max_abs_delta": 0 if step == 0 else 0.03,
                                "graph_cka": 1.0 if frozen or step == 0 else 0.9,
                                "local_cka": 1.0 if frozen or step == 0 else 0.8})
                        if arm != "scratch_replay":
                            for composition, accuracy in (("E0_H0", 0.1), ("E0_H1", 0.3), ("E1_H0", 0.2), ("E1_H1", 0.6)):
                                for split in ("iid_a", "iid_b", "ood_common"):
                                    components.append({"experience": experience, "seed": seed, "repeat": repeat,
                                        "arm": arm, "composition": composition, "split": split, "n": 100,
                                        "motif_accuracy": accuracy + 0.01 * seed,
                                        "position_accuracy": 0.8, "distance_accuracy": 0.7,
                                        "joint_accuracy": 0.05, "distance_mae": 0.6})
                    for readout in ("local", "graph"):
                        for support in ("new", "replay"):
                            for split in ("iid_a", "iid_b", "ood_common"):
                                ridge.append({"experience": experience, "seed": seed, "repeat": repeat,
                                    "readout": readout, "support": support, "split": split, "n": 100,
                                    "motif_accuracy": 0.7 + 0.01 * seed,
                                    "position_accuracy": 0.9, "distance_accuracy": 0.8,
                                    "joint_accuracy": 0.5, "distance_mae": 0.6})
        cls.trajectory = trajectory
        for filename, rows in (("trajectory.csv", trajectory), ("drift.csv", drift),
                               ("component_eval.csv", components), ("ridge_diagnostic.csv", ridge)):
            _write_csv(cls.output, filename, rows)
        cls.input_digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.output.iterdir()}
        generate_mechanism_report(cls.output)
        cls.summary = json.loads((cls.output / "report_summary.json").read_text(encoding="utf-8"))
        cls.report = (cls.output / "REPORT.md").read_text(encoding="utf-8")

    def test_support_draws_are_nested_within_initialization(self):
        stat = next(row["motif_accuracy"] for row in self.summary["final"]
                    if row["experience"] == "a" and row["arm"] == "frozen_new" and row["split"] == "ood_common")
        self.assertEqual(stat["n"], 3)
        self.assertEqual(stat["per_seed"], {"0": 0.43, "1": 0.48, "2": 0.53})
        self.assertAlmostEqual(stat["mean"], 0.48)
        self.assertAlmostEqual(stat["std"], 0.05)

    def test_primary_fixed_support_is_separate_from_robustness_mean(self):
        primary = next(row["motif_accuracy"] for row in self.summary["primary_final"]
                       if row["experience"] == "a" and row["arm"] == "frozen_new" and row["split"] == "ood_common")
        self.assertEqual(self.summary["primary_repeat"], "0")
        self.assertAlmostEqual(primary["mean"], 0.45)
        self.assertAlmostEqual(primary["std"], 0.05)
        self.assertIn("Primary comparison, using only the predeclared fixed support repetition 0", self.report)
        self.assertIn("Supplemental summary, averaging all support repetitions within each seed", self.report)
        self.assertIn("45.0 +/- 5.0% (n=3)", self.report)
        self.assertIn("48.0 +/- 5.0% (n=3)", self.report)

    def test_paired_arm_contrast_cancels_seed_and_repeat_variation(self):
        contrast = next(row["motif_accuracy"] for row in self.summary["paired_arm_contrasts"]
                        if row["experience"] == "b" and row["current_arm"] == "frozen_replay"
                        and row["reference_arm"] == "frozen_new" and row["split"] == "ood_common")
        self.assertEqual(contrast["n"], 3)
        self.assertAlmostEqual(contrast["mean"], 0.02)
        self.assertAlmostEqual(contrast["std"], 0)
        self.assertIn("2.0 +/- 0.0 percentage points (n=3)", self.report)

    def test_old_distribution_uses_each_experiences_own_iid(self):
        changes = self.summary["changes_from_initial"]
        self.assertTrue(any(row["experience"] == "a" and row["split"] == "iid_a" for row in changes))
        self.assertTrue(any(row["experience"] == "b" and row["split"] == "iid_b" for row in changes))
        self.assertFalse(any(row["experience"] == "a" and row["split"] == "iid_b" for row in changes))
        change = next(row["motif_accuracy"] for row in changes
                      if row["experience"] == "a" and row["arm"] == "frozen_new" and row["split"] == "iid_a")
        self.assertAlmostEqual(change["mean"], -0.15)
        self.assertAlmostEqual(change["std"], 0.01)
        self.assertIn("scratch arm starts without pretrained performance", self.report)

    def test_all_task_joint_measurement_is_retained(self):
        stat = next(row["joint_accuracy"] for row in self.summary["final"]
                    if row["experience"] == "a" and row["arm"] == "frozen_new" and row["split"] == "ood_common")
        self.assertAlmostEqual(stat["mean"], 0.24)
        self.assertIn("computed per molecule before aggregation", self.report)

    def test_seed_csv_retains_full_precision_and_three_splits(self):
        with (self.output / "final_seed_metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2 * len(ARM_NAMES) * 3 * 3)
        selected = next(row for row in rows if row["experience"] == "a" and row["arm"] == "frozen_new"
                        and row["seed"] == "0" and row["split"] == "ood_common")
        self.assertAlmostEqual(float(selected["motif_accuracy"]), 0.43)
        self.assertEqual(selected["support_repeats"], "3")
        self.assertEqual({row["split"] for row in rows}, {"iid_a", "iid_b", "ood_common"})

    def test_frozen_scope_budget_and_readout_limit_are_explicit(self):
        self.assertIn("embedding and all message-passing layers", self.report)
        self.assertIn("all three inherited MLP task heads", self.report)
        self.assertIn("32 distinct labels per adaptation run", self.report)
        self.assertIn("64 distinct labels", self.report)
        self.assertIn("equal numbers of optimizer updates", self.report)
        self.assertIn("not prove that no successful frozen readout exists", self.report)
        self.assertIn("not a replication", self.report)
        self.assertEqual(self.summary["motif_uniform_random_chance"], 0.125)
        self.assertIn("37.1% balanced accuracy is not a theoretical chance level", self.report)

    def test_drift_and_component_swaps_keep_conditional_interpretation(self):
        checks = self.summary["frozen_encoder_checks"]
        self.assertEqual(checks["changed_rows"], 0)
        self.assertEqual(checks["max_recorded_abs_delta"], 0)
        self.assertIn("bitwise", self.report)
        self.assertIn("does not determine which change caused the loss", self.report)
        self.assertIn("not give an additive causal decomposition", self.report)
        self.assertIn("altered feature basis", self.report)
        self.assertTrue(self.summary["primary_component_eval"])
        self.assertEqual({row["composition"] for row in self.summary["component_eval"]}, {"E0_H0", "E0_H1", "E1_H0", "E1_H1"})

    def test_fresh_ridge_readout_is_a_separate_optimization_diagnostic(self):
        self.assertTrue(self.summary["primary_ridge_diagnostic"])
        self.assertEqual({row["readout"] for row in self.summary["ridge_diagnostic"]}, {"graph", "local"})
        self.assertIn("normalization", self.report.lower())
        self.assertIn("support data only", self.report)
        self.assertIn("not by itself establish local information", self.report)

    def test_incomplete_or_duplicate_panels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete trajectory"):
            _validate_trajectory(self.trajectory[1:], self.config)
        with self.assertRaisesRegex(ValueError, "Duplicate trajectory"):
            _validate_trajectory([*self.trajectory, self.trajectory[0]], self.config)
        with self.assertRaisesRegex(ValueError, "configured evaluation"):
            _validate_trajectory(self.trajectory, {**self.config, "evaluation_steps": [0, 1, 80]})

    def test_nonfinite_and_out_of_range_metrics_are_rejected(self):
        for value in (float("nan"), float("inf"), 1.1, -0.1):
            rows = [{**self.trajectory[0], "motif_accuracy": value}, *self.trajectory[1:]]
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_trajectory(rows, self.config)

    def test_report_preserves_inputs_and_creates_english_figures(self):
        for filename, digest in self.input_digests.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), digest)
        for stem in ("mechanism_summary", "mechanism_support_robustness"):
            self.assertEqual((self.output / "figures" / f"{stem}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn("<svg", (self.output / "figures" / f"{stem}.svg").read_text(encoding="utf-8"))
        self.assertFalse(any("\u4e00" <= character <= "\u9fff" for character in self.report))
        self.assertIn("preserves recorded measurements", self.report)


if __name__ == "__main__":
    unittest.main()
