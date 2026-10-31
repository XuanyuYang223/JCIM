"""Regression checks for uncertainty reporting and scientific interpretation."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jcim.report import generate_report


def _write_json(directory, filename, value):
    (directory / filename).write_text(json.dumps(value), encoding="utf-8")


def _write_csv(directory, filename, rows):
    with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReportRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="jcim_report_test_")
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.output = Path(cls.temporary_directory.name)
        _write_json(cls.output, "config.json", {
            "seeds": [0, 1, 2], "layers": 2,
            "adaptation_shots": [8, 24], "adaptation_repeats": 5,
        })
        _write_json(cls.output, "audit.json", {
            "n_molecules": 120,
            "unique_canonical_identities": 120,
            "split_identity_leakage": False,
            "split_counts": {"train_a": 60, "train_b": 60, "ood_f_meta": 20},
            "effective_train_counts": {"train_a": 60, "train_b": 60},
            "distance_range": [3, 11],
        })
        _write_json(cls.output, "environment.json", {"device": "cpu", "parameter_count": 1000})
        _write_json(cls.output, "splits.json", {
            "train_a": [0, 1], "train_b": [0, 2, 3], "adaptation_pool": [2, 3],
        })

        behavior, probes, adaptation, similarities = [], [], [], []
        for condition in ("a", "b", "random"):
            for seed in range(3):
                model = f"{condition}_seed{seed}"
                for split in ("iid_test", "ood_f_meta", "ood_cl_meta", "ood_f_meta_direct_f"):
                    direct = split == "ood_f_meta_direct_f"
                    behavior.append({
                        "model": model, "condition": condition, "seed": seed,
                        "split": split, "n": 4 if direct else 20,
                        "family_accuracy": 0.95,
                        "position_accuracy": (0.2 if direct else 0.6) + seed * 0.01,
                        "distance_accuracy": 0.5, "distance_mae": 1.2,
                    })
                for probe in ("cross_position_family", "cross_family_position", "cross_family_local_position"):
                    for layer in range(3):
                        if probe == "cross_position_family":
                            accuracy = {"a": 0.75, "b": 0.8, "random": 0.9}[condition]
                        else:
                            accuracy = {"a": 0.8, "b": 0.82, "random": 0.4}[condition]
                        probes.append({
                            "model": model, "condition": condition, "seed": seed,
                            "probe": probe, "layer": layer,
                            "accuracy": accuracy, "balanced_accuracy": accuracy,
                            "shuffled_accuracy": 0.125,
                        })
                for shots in (8, 24):
                    for repeat in range(5):
                        adaptation.append({
                            "model": model, "condition": condition, "seed": seed,
                            "shots": shots, "repeat": repeat,
                            "accuracy": 0.5,
                            "balanced_accuracy": 0.4 + seed * 0.02 + repeat * 0.01,
                        })
        for probe in ("cross_position_family", "cross_family_position", "cross_family_local_position"):
            accuracy = 1.0 if probe == "cross_position_family" else 0.34
            probes.append({
                "model": "raw", "condition": "raw", "seed": -1,
                "probe": probe, "layer": -1,
                "accuracy": accuracy, "balanced_accuracy": accuracy,
                "shuffled_accuracy": 0.125,
            })
        for shots in (8, 24):
            for repeat in range(5):
                adaptation.append({
                    "model": "raw", "condition": "raw", "seed": -1,
                    "shots": shots, "repeat": repeat,
                    "accuracy": 0.4, "balanced_accuracy": 0.3 + repeat * 0.01,
                })
        for pair_kind in ("random_random", "trained_same_data", "trained_different_data", "random_trained"):
            for pair in range(3):
                for layer in range(3):
                    similarities.append({
                        "pair_kind": pair_kind,
                        "model_a": f"model_{pair}", "model_b": f"model_{pair + 1}",
                        "layer": layer, "cka": 0.8 + pair * 0.01,
                        "procrustes_similarity": 0.9,
                    })
        for filename, rows in (("behavior.csv", behavior), ("probes.csv", probes),
                               ("adaptation.csv", adaptation), ("similarities.csv", similarities)):
            _write_csv(cls.output, filename, rows)
        cls.input_digests = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in cls.output.iterdir()
        }
        generate_report(cls.output)
        cls.summary = json.loads((cls.output / "report_summary.json").read_text(encoding="utf-8"))
        cls.report = (cls.output / "REPORT.md").read_text(encoding="utf-8")

    def test_adaptation_repeats_do_not_inflate_encoder_sample_count(self):
        stat = next(
            row["balanced_accuracy"] for row in self.summary["adaptation"]
            if row["condition"] == "a" and row["shots"] == "8"
        )
        self.assertEqual(stat["n"], 3)
        self.assertEqual(len(stat["per_seed"]), 3)
        self.assertAlmostEqual(stat["mean"], 0.44)
        self.assertAlmostEqual(stat["std"], 0.02)
        self.assertIn("44.0 +/- 2.0% (n=3)", self.report)

    def test_raw_feature_repeats_have_one_deterministic_encoder(self):
        stat = next(
            row["balanced_accuracy"] for row in self.summary["adaptation"]
            if row["condition"] == "raw" and row["shots"] == "8"
        )
        self.assertEqual(stat["n"], 1)
        self.assertIsNone(stat["std"])
        self.assertAlmostEqual(stat["mean"], 0.32)
        self.assertIn("32.0% (n=1; SD undefined)", self.report)

    def test_direct_aryl_f_does_not_contaminate_main_ood_aggregate(self):
        main = next(
            row["position_accuracy"] for row in self.summary["behavior"]
            if row["condition"] == "a" and row["split"] == "ood_f_meta"
        )
        direct = next(
            row["position_accuracy"] for row in self.summary["behavior"]
            if row["condition"] == "a" and row["split"] == "ood_f_meta_direct_f"
        )
        self.assertAlmostEqual(main["mean"], 0.61)
        self.assertAlmostEqual(direct["mean"], 0.21)
        main_section, direct_section = self.report.split("### Direct aryl-F subset", 1)
        self.assertIn("| Experience A | ood_f_meta | 20 |", main_section)
        self.assertNotIn("| Experience A | ood_f_meta_direct_f |", main_section)
        self.assertIn("| Experience A | ood_f_meta_direct_f | 4 |", direct_section)

    def test_cross_family_local_position_is_reported_separately(self):
        members = [
            row for row in self.summary["probes"]
            if row["probe"] == "cross_family_local_position" and row["layer"] == "2"
        ]
        self.assertEqual({row["probe_condition"] for row in members}, {"a", "b", "random"})
        self.assertIn("| cross_family_local_position | Experience A | 2 | 80.0", self.report)
        self.assertIn("| cross_family_local_position | Raw input features | Raw | 34.0", self.report)

    def test_saturated_raw_family_probe_does_not_become_a_learning_claim(self):
        raw = next(
            row["balanced_accuracy"] for row in self.summary["probes"]
            if row["probe"] == "cross_position_family" and row["probe_condition"] == "raw_input"
        )
        self.assertEqual(raw["mean"], 1.0)
        self.assertIn("raw-input family readout reaches 100.0%", self.report)
        self.assertIn("mean cross-position family balanced-accuracy difference of -15.0 percentage points", self.report)
        self.assertIn("High family readout accuracy alone therefore does not show that training created a reusable representation", self.report)
        self.assertNotRegex(
            self.report,
            r"(?i)(?:these results|the results|this run)\s+(?:demonstrate|demonstrates|prove|proves|establish|establishes)\s+(?:that\s+)?(?:learned|functional|reusable)",
        )

    def test_generation_preserves_recorded_inputs_and_writes_valid_figures(self):
        for filename, digest in self.input_digests.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), digest)
        png = self.output / "figures" / "pilot_summary.png"
        self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("<svg", (self.output / "figures" / "pilot_summary.svg").read_text(encoding="utf-8"))
        self.assertIn("Experience B pretrained on 2 of the 2 adaptation-pool molecular identities", self.report)


if __name__ == "__main__":
    unittest.main()
