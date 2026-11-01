"""Check exact-seed inference, source-only reporting, and matched head controls."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jcim.skip_report import (INTERVENTIONS, METRICS, REGIMES, VARIANTS, _histories, _inventory,
                              _paired_rows, _validate, generate_skip_report)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(output, *, seeds=(0, 1, 2), epochs=120):
    config = {"seeds": list(seeds), "experiences": ["a", "b"], "regimes": list(REGIMES),
              "variants": list(VARIANTS), "epochs": epochs, "reuse_original_deep_baseline": epochs == 120}
    (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (output / "histories").mkdir()
    inventory, measurements = [], []
    for regime in REGIMES:
        for variant in VARIANTS:
            for experience in ("a", "b"):
                for seed in seeds:
                    common = {"regime": regime, "variant": variant, "experience": experience, "seed": seed}
                    inventory.append({**common, "readout_dim": 448 if variant == "deep" else 512,
                                      "parameter_count": 189531 if variant == "deep" else 201819,
                                      "trainable_parameter_count": (87963 if variant == "deep" else 100251)
                                      if regime == "frozen_readout" else (189531 if variant == "deep" else 201819),
                                      "initial_encoder_sha256": "initial", "final_encoder_sha256": "initial" if regime == "frozen_readout" else "updated",
                                      "encoder_source": "source", "head_source": "random", "reused_source": config["reuse_original_deep_baseline"] and regime == "end_to_end" and variant == "deep"})
                    name = f"{regime}_{variant}_{experience}_seed{seed}"
                    history = [{"epoch": epoch, "train_loss": 1.0 / epoch + seed * 0.01,
                                "val_family_accuracy": 0.7 + 0.01 * seed,
                                "val_position_accuracy": 0.8 + 0.01 * seed,
                                "val_distance_accuracy": 0.6 + 0.01 * seed}
                               for epoch in sorted({1, epochs})]
                    (output / "histories" / f"history_{name}.json").write_text(json.dumps(history), encoding="utf-8")
                    for intervention in (("intact",) if variant == "deep" else INTERVENTIONS):
                        for split in ("iid_a", "iid_b", "ood_common"):
                            score = 0.8 + seed * 0.01 if split != "ood_common" else 0.2 + seed * 0.03
                            if variant == "shallow_skip":
                                score += 0.02 if split != "ood_common" else 0.12 + seed * 0.02
                            if intervention != "intact":
                                score -= 0.04
                            measurements.append({**common, "intervention": intervention, "split": split, "n": 80,
                                                 **{metric: score if metric != "distance_mae" else 1.2 - score for metric in METRICS}})
    _write_csv(output / "inventory.csv", inventory)
    _write_csv(output / "behavior.csv", measurements)
    return config, measurements, inventory


class SkipReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="jcim_skip_report_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name)
        cls.config, cls.rows, cls.inventory = _fixture(cls.output)
        cls.input_hashes = {str(path.relative_to(cls.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in cls.output.rglob("*") if path.is_file()}
        generate_skip_report(cls.output)
        cls.summary = json.loads((cls.output / "report_summary.json").read_text(encoding="utf-8"))
        cls.report = (cls.output / "REPORT.md").read_text(encoding="utf-8")

    def cell(self, name, **filters):
        return next(row for row in self.summary[name] if all(str(row[key]) == str(value) for key, value in filters.items()))

    def test_primary_contrast_is_paired_with_exact_encoder_seed(self):
        cell = self.cell("paired_skip_minus_control", regime="end_to_end", experience="a", intervention="intact", split="ood_common")["motif_accuracy"]
        self.assertEqual(cell["n"], 3)
        self.assertAlmostEqual(cell["mean"], 0.14)
        self.assertAlmostEqual(cell["std"], 0.02)
        self.assertAlmostEqual(cell["per_seed"]["0"], 0.12)
        self.assertEqual(self.summary["primary_metric"], "motif_accuracy")
        self.assertEqual(self.summary["primary_regime"], "end_to_end")
        self.assertFalse(self.summary["pipeline_only"])

    def test_own_iid_gap_and_experiences_are_kept_separate(self):
        gap = self.cell("generalization_gap", regime="frozen_readout", variant="shallow_skip", experience="b")["motif_accuracy"]
        self.assertAlmostEqual(gap["mean"], 0.46)
        self.assertAlmostEqual(gap["std"], 0.04)
        self.assertEqual(gap["n"], 3)
        self.assertEqual(self.summary["encoder_seed_count"], 3)
        self.assertEqual(len(self.summary["behavior"]), 84)

    def test_all_tasks_and_secondary_interventions_are_retained(self):
        cell = self.cell("behavior", regime="end_to_end", variant="shallow_skip", experience="a", intervention="zero_branch", split="iid_b")
        self.assertTrue(set(METRICS).issubset(cell))
        self.assertAlmostEqual(cell["distance_mae"]["mean"], 0.41)
        self.assertIn("functional compatibility", self.report)
        self.assertIn("do not measure information destruction", self.report)
        with (self.output / "paired_seed_deltas.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 108)
        self.assertEqual({row["contrast"] for row in rows}, {"shallow_skip_minus_deep_control"})

    def test_missing_duplicate_and_nonfinite_endpoints_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            _validate(self.rows[1:], self.config)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _validate([*self.rows, self.rows[0]], self.config)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid motif_accuracy"):
                _validate([{**self.rows[0], "motif_accuracy": value}, *self.rows[1:]], self.config)

    def test_unmatched_pairs_or_different_test_counts_are_rejected(self):
        removed = False
        unmatched = []
        for row in self.rows:
            if row["variant"] == "shallow_skip" and not removed:
                removed = True
            else:
                unmatched.append(row)
        with self.assertRaisesRegex(ValueError, "exactly matched"):
            _paired_rows(unmatched)
        changed = [{**row, "n": 79} if row["variant"] == "shallow_skip" else row for row in self.rows]
        with self.assertRaisesRegex(ValueError, "different test counts"):
            _paired_rows(changed)

    def test_width_and_parameter_matching_are_checked(self):
        changed = copy.deepcopy(self.inventory)
        current = next(row for row in changed if row["variant"] == "shallow_skip")
        current["parameter_count"] += 1
        with self.assertRaisesRegex(ValueError, "identical parameter counts"):
            _inventory(changed, self.config)
        changed = copy.deepcopy(self.inventory)
        changed[0]["readout_dim"] = 512
        with self.assertRaisesRegex(ValueError, "448D"):
            _inventory(changed, self.config)

    def test_frozen_encoder_hash_changes_are_rejected(self):
        changed = copy.deepcopy(self.inventory)
        next(row for row in changed if row["regime"] == "frozen_readout")["final_encoder_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "frozen encoder changed"):
            _inventory(changed, self.config)

    def test_training_histories_cannot_include_ood_or_stop_early(self):
        with tempfile.TemporaryDirectory(prefix="jcim_skip_history_") as directory:
            output = Path(directory)
            config, _, inventory = _fixture(output, seeds=(0,))
            path = output / "histories" / "history_end_to_end_deep_a_seed0.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history[-1]["ood_motif_accuracy"] = 0.9
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exclude OOD"):
                _histories(output, inventory, config)
            del history[-1]["ood_motif_accuracy"]
            history[-1]["epoch"] = 119
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predeclared final epoch"):
                _histories(output, inventory, config)

    def test_initialization_and_scope_are_described_without_overclaiming(self):
        for required in ("reset the heads", "do not continue the inherited trained classifier", "exact completed hard_v2 checkpoints",
                         "deterministic duplicate", "No target-combination labels", "boundary-free rather than query-free",
                         "not that all useful shallow information is absent", "A and B are not pooled", "12.5%", "not significance tests"):
            with self.subTest(required=required):
                self.assertIn(required, self.report)

    def test_derived_outputs_preserve_source_inputs_and_make_english_figures(self):
        for filename, digest in self.input_hashes.items():
            self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), digest)
        for stem in ("skip_primary", "skip_training"):
            self.assertEqual((self.output / "figures" / f"{stem}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn("<svg", (self.output / "figures" / f"{stem}.svg").read_text(encoding="utf-8"))
        self.assertFalse(any("\u4e00" <= character <= "\u9fff" for character in self.report))

    def test_reduced_pipeline_report_reports_actual_seed_count(self):
        with tempfile.TemporaryDirectory(prefix="jcim_skip_smoke_report_") as directory:
            output = Path(directory)
            _fixture(output, seeds=(0,), epochs=1)
            with patch("jcim.skip_report._make_figures"):
                generate_skip_report(output)
            summary = json.loads((output / "report_summary.json").read_text(encoding="utf-8"))
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertTrue(summary["pipeline_only"])
            self.assertEqual(summary["encoder_seed_count"], 1)
            self.assertIsNone(summary["behavior"][0]["motif_accuracy"]["std"])
            self.assertIn("across 1 initialization seeds", report)
            self.assertIn("Pipeline check only", report)
            self.assertIn("All arms are freshly trained for this run", report)
            self.assertIn("no original deep end-to-end checkpoint or history is reused", report)
            self.assertNotIn("baselines reuse the exact completed hard_v2 checkpoints", report)


if __name__ == "__main__":
    unittest.main()
