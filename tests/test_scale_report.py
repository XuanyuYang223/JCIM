"""Protect paired comparisons, nested support draws, and spatial interpretation."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jcim.scale_report import SCALES, _paired_rows, _validate, generate_scale_report, PAIR_ADAPT_KEYS


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ScaleReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="jcim_scale_report_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name)
        cls.config = {"seeds": [0, 1, 2], "support_repeats": [0, 1, 2], "primary_repeat": 0,
                      "layers": [0, 8], "primary_layer": 8, "alpha": 1.0}
        (cls.output / "config.json").write_text(json.dumps(cls.config), encoding="utf-8")
        adaptation, direct = [], []
        for experience in ("a", "b"):
            for condition in ("trained", "random"):
                for seed in range(3):
                    for repeat in range(3):
                        for layer in cls.config["layers"]:
                            for scale_index, scale in enumerate(SCALES):
                                common = {"experience": experience, "condition": condition,
                                          "model": f"{experience if condition == 'trained' else 'random'}_seed{seed}",
                                          "seed": seed, "repeat": repeat, "layer": layer, "scale": scale,
                                          "feature_dim": 448 if scale == "legacy_graph" else 64}
                                bonus = 0.20 + 0.02 * seed + 0.03 * repeat if condition == "trained" else 0
                                for support in ("new", "replay"):
                                    for split in ("iid_a", "iid_b", "ood_common"):
                                        score = 0.15 + 0.01 * seed + 0.02 * repeat + 0.002 * scale_index + 0.001 * layer + bonus
                                        adaptation.append({**common, "support": support, "split": split,
                                                           "n_fit": 32 if support == "new" else 64, "n_test": 80,
                                                           "accuracy": score + 0.01, "balanced_accuracy": score,
                                                           "prediction_key": f"adapt_{len(adaptation)}"})
                                for source in range(3):
                                    for target in range(3):
                                        if source == target:
                                            continue
                                        for split, stratum_shift in (("target_all", 0), ("target_common_ood", 0.07), ("target_own_iid", 0.12)):
                                            score = 0.10 + 0.01 * seed + 0.02 * repeat + 0.002 * scale_index + 0.001 * layer + 0.005 * source + 0.004 * target + bonus + stratum_shift
                                            direct.append({**common, "source_position": source, "target_position": target, "split": split,
                                                           "n_fit": 256, "n_test": 80 if split == "target_all" else 24,
                                                           "accuracy": score + 0.01, "balanced_accuracy": score,
                                                           "prediction_key": f"direct_{len(direct)}"})
        _write_csv(cls.output / "adaptation.csv", adaptation)
        _write_csv(cls.output / "direct_transfer.csv", direct)
        locality = []
        for scale in SCALES:
            for layer in cls.config["layers"]:
                for index in range(2):
                    pool_count = 1 if scale == "root" else 9 if scale == "target" else 20
                    receptive_count = pool_count if layer == 0 else 12 if scale == "root" else 20
                    locality.append({"index": index, "id": f"mol{index}", "region": scale, "layer": layer,
                                     "pool_atom_count": pool_count, "target_atom_count": 9, "molecule_atom_count": 20,
                                     "pool_outside_target_count": max(0, pool_count - 9),
                                     "input_receptive_atom_count": receptive_count,
                                     "receptive_outside_target_count": max(0, receptive_count - 9),
                                     "receptive_outside_pool_count": receptive_count - pool_count,
                                     "receptive_covers_full_graph": receptive_count == 20})
        _write_csv(cls.output / "locality.csv", locality)
        cls.input_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.output.iterdir()}
        with (cls.output / "adaptation.csv").open(encoding="utf-8", newline="") as stream:
            cls.adaptation = list(csv.DictReader(stream))
        with (cls.output / "direct_transfer.csv").open(encoding="utf-8", newline="") as stream:
            cls.direct = list(csv.DictReader(stream))
        generate_scale_report(cls.output)
        cls.summary = json.loads((cls.output / "report_summary.json").read_text(encoding="utf-8"))
        cls.report = (cls.output / "REPORT.md").read_text(encoding="utf-8")

    def cell(self, name, **filters):
        return next(row for row in self.summary[name] if all(str(row[key]) == str(value) for key, value in filters.items()))

    def test_support_draws_are_averaged_inside_three_encoder_seeds(self):
        stat = self.cell("adaptation", experience="a", condition="trained", layer=8, scale="root", support="replay", split="ood_common")["balanced_accuracy"]
        self.assertEqual(stat["n"], 3)
        self.assertAlmostEqual(stat["mean"], 0.438)
        self.assertAlmostEqual(stat["std"], 0.03)
        self.assertAlmostEqual(stat["per_seed"]["0"], 0.408)

    def test_primary_fixed_support_is_separate_from_repeat_robustness(self):
        stat = self.cell("primary_adaptation", experience="a", condition="trained", layer=8, scale="root", support="replay", split="ood_common")["balanced_accuracy"]
        self.assertAlmostEqual(stat["mean"], 0.388)
        self.assertAlmostEqual(stat["std"], 0.03)
        self.assertIn("Only support repetition 0 and layer 8", self.report)
        self.assertIn("38.8 +/- 3.0% (n=3)", self.report)

    def test_paired_advantages_subtract_each_matched_support_first(self):
        filters = {"experience": "b", "layer": 8, "scale": "target", "support": "replay", "split": "ood_common"}
        primary = self.cell("primary_paired_adaptation", **filters)["balanced_accuracy"]
        robust = self.cell("paired_adaptation", **filters)["balanced_accuracy"]
        self.assertAlmostEqual(primary["mean"], 0.22)
        self.assertAlmostEqual(robust["mean"], 0.25)
        self.assertAlmostEqual(robust["std"], 0.02)
        self.assertEqual(robust["n"], 3)

    def test_all_six_directions_are_averaged_within_seed_and_strata_stay_separate(self):
        filters = {"experience": "a", "condition": "trained", "layer": 8, "scale": "root"}
        all_targets = self.cell("direct_direction_average", **filters, split="target_all")["balanced_accuracy"]
        protected = self.cell("direct_direction_average", **filters, split="target_common_ood")["balanced_accuracy"]
        self.assertAlmostEqual(all_targets["mean"], 0.397)
        self.assertAlmostEqual(protected["mean"], 0.467)
        self.assertEqual(all_targets["n"], 3)
        self.assertAlmostEqual(all_targets["std"], 0.03)

    def test_incomplete_duplicate_and_nonfinite_panels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            _validate(self.adaptation[1:], self.config, direct=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _validate([*self.direct, self.direct[0]], self.config, direct=True)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid balanced_accuracy"):
                _validate([{**self.adaptation[0], "balanced_accuracy": value}, *self.adaptation[1:]], self.config, direct=False)

    def test_unmatched_seeds_and_fit_counts_cannot_enter_paired_statistics(self):
        with self.assertRaisesRegex(ValueError, "exactly matched"):
            _paired_rows(self.adaptation[1:], PAIR_ADAPT_KEYS)
        changed = [{**row, "n_fit": "63"} if row["condition"] == "random" else row for row in self.adaptation]
        with self.assertRaisesRegex(ValueError, "different fit or test"):
            _paired_rows(changed, PAIR_ADAPT_KEYS)

    def test_feature_width_and_exact_initialization_names_are_validated(self):
        for field, value, message in (("feature_dim", "448", "single-pool"), ("model", "random_seed99", "initialization")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                _validate([{**self.adaptation[0], field: value}, *self.adaptation[1:]], self.config, direct=False)

    def test_seed_tables_preserve_precision_and_statistical_unit(self):
        with (self.output / "seed_metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row = next(row for row in rows if row["experiment"] == "direct_direction_average" and row["repeat_scope"] == "all_repeats_within_seed"
                   and row["experience"] == "a" and row["condition"] == "trained" and row["layer"] == "8"
                   and row["scale"] == "root" and row["seed"] == "0" and row["split"] == "target_all")
        self.assertEqual(row["support_repeats"], "3")
        self.assertEqual(row["measurements_averaged"], "18")
        self.assertAlmostEqual(float(row["balanced_accuracy"]), 0.367)
        with (self.output / "paired_seed_deltas.csv").open(encoding="utf-8", newline="") as stream:
            paired = list(csv.DictReader(stream))
        self.assertTrue(all(row["condition"] == "trained_minus_random" for row in paired))

    def test_locality_and_target_labels_are_not_overinterpreted(self):
        for required in ("32 new plus 32 old", "256 labels", "zero target-position labels", "label budgets and fit distributions differ",
                         "not a strictly local input", "up to eight graph hops", "oracle metadata", "448 features", "64 features",
                         "not the inherited chemical task head's zero-shot", "not count nine repetitions as nine independent models",
                         "12.5%", "no target-test hyperparameter or layer selection"):
            with self.subTest(required=required):
                self.assertIn(required, self.report)
        self.assertEqual(self.summary["motif_uniform_random_chance"], 0.125)

    def test_locality_quantifies_receptive_reach_separately_from_pool_size(self):
        cell = self.cell("locality", layer=8, scale="root")
        self.assertEqual(cell["n"], 2)
        self.assertEqual(cell["pool_atom_count"], {"mean": 1.0, "min": 1, "max": 1})
        self.assertEqual(cell["input_receptive_atom_count"]["mean"], 12)
        self.assertEqual(cell["mean_receptive_fraction"], 0.6)
        self.assertEqual(cell["fraction_covering_full_graph"], 0)
        self.assertEqual(self.cell("locality", layer=8, scale="target")["fraction_covering_full_graph"], 1)
        self.assertIn("architectural upper bound", self.report)
        self.assertIn("1.0 [1, 1]", self.report)

    def test_reporting_preserves_inputs_and_generates_english_figures(self):
        for filename, digest in self.input_hashes.items():
            self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), digest)
        for stem in ("scale_primary", "scale_paired_layers"):
            self.assertEqual((self.output / "figures" / f"{stem}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn("<svg", (self.output / "figures" / f"{stem}.svg").read_text(encoding="utf-8"))
        self.assertFalse(any("\u4e00" <= character <= "\u9fff" for character in self.report))

    def test_reduced_smoke_report_uses_actual_replication_counts(self):
        with tempfile.TemporaryDirectory(prefix="jcim_scale_smoke_report_") as directory:
            output = Path(directory)
            config = {**self.config, "seeds": [0], "support_repeats": [0]}
            (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for name, rows in (("adaptation", self.adaptation), ("direct_transfer", self.direct)):
                _write_csv(output / f"{name}.csv", [row for row in rows if row["seed"] == "0" and row["repeat"] == "0"])
            with patch("jcim.scale_report._make_figures"):
                generate_scale_report(output)
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            summary = json.loads((output / "report_summary.json").read_text(encoding="utf-8"))
            self.assertIn("1 support repetition is nested", report)
            self.assertIn("sample SD across 1 encoder seed;", report)
            self.assertIn("Pipeline check only", report)
            self.assertTrue(summary["pipeline_only"])
            self.assertEqual(summary["encoder_seed_count"], 1)
            self.assertEqual(summary["support_repeat_count"], 1)


if __name__ == "__main__":
    unittest.main()
