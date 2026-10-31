"""Check scientific aggregation and record preservation in the harder report."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jcim.hard_report import PAIR_NAMES, generate_hard_report


def _write_json(directory: Path, filename: str, value: dict) -> None:
    (directory / filename).write_text(json.dumps(value), encoding="utf-8")


def _write_csv(directory: Path, filename: str, rows: list[dict]) -> None:
    with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HardReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="jcim_hard_report_test_")
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.output = Path(cls.temporary_directory.name)
        _write_json(cls.output, "config.json", {"seeds": [0, 1, 2], "adaptation_shots": [8, 32], "adaptation_repeats": 3})
        _write_json(cls.output, "audit.json", {
            "n_molecules": 240, "unique_canonical_identities": 240,
            "split_identity_leakage": False,
            "split_counts": {"train_a": 40, "train_b": 40, "ood_common": 20},
            "effective_train_counts": {"train_a": 40, "train_b": 40},
        })
        _write_json(cls.output, "environment.json", {"device": "cpu", "parameter_count": 12000})
        _write_json(cls.output, "splits.json", {"train_a": [1, 2], "train_b": [3, 4], "adaptation_pool": [5, 6]})
        fragments = ("CCOC(CCC)OC", "CCOC(CC)OCC", "CCOC(CC)COC", "COCC(CCC)OC",
            "COCC(CC)OCC", "COCC(CC)COC", "CCCC(OC)OCC", "CCCC(COC)OC")
        (cls.output / "molecules.jsonl").write_text("".join(json.dumps({"y_motif": index,
            "target_family": f"motif_{index}", "motif_smiles": fragment}) + "\n"
            for index, fragment in enumerate(fragments)), encoding="utf-8")
        behavior, probes, adaptation, similarities = [], [], [], []
        for condition in ("a", "b", "random"):
            for seed in range(3):
                model = f"{condition}_seed{seed}"
                for split in ("iid_a", "iid_b", "ood_common"):
                    accuracy = {"a": 0.6, "b": 0.65, "random": 0.2}[condition]
                    behavior.append({"model": model, "condition": condition, "seed": seed, "split": split, "n": 20,
                        "motif_accuracy": accuracy + 0.01 * seed, "position_accuracy": 0.6,
                        "distance_accuracy": 0.5, "distance_mae": 1.2})
                for layer in range(3):
                    accuracy = {"a": 0.6, "b": 0.65, "random": 0.7}[condition]
                    probes.append({"model": model, "condition": condition, "seed": seed, "probe": "cross_position_motif", "layer": layer,
                        "accuracy": accuracy, "balanced_accuracy": accuracy, "shuffled_accuracy": 0.125})
                for shots in (0, 8, 32):
                    for repeat in range(1 if shots == 0 else 3):
                        for split in ("ood_common", "iid_a", "iid_b"):
                            # Three support repetitions per seed must still yield n=3.
                            accuracy = 0.4 + 0.02 * seed + 0.01 * repeat if split == "ood_common" else 0.2
                            adaptation.append({"model": model, "condition": condition, "seed": seed, "shots": shots,
                                "repeat": repeat, "split": split, "motif_accuracy": accuracy, "position_accuracy": 0.6,
                                "distance_accuracy": 0.3, "distance_mae": 1.4})
        probes.append({"model": "raw", "condition": "raw", "seed": -1, "probe": "cross_position_motif", "layer": -1,
            "accuracy": 0.125, "balanced_accuracy": 0.125, "shuffled_accuracy": 0.125})
        for kind in PAIR_NAMES:
            for pooling in ("local", "graph"):
                for pair in range(3):
                    for layer in range(3):
                        similarities.append({"pair_kind": kind, "model_a": f"model_{pair}", "model_b": f"model_{pair + 1}",
                            "layer": layer, "pooling": pooling, "cka": (0.95 if pooling == "local" else 0.75) + 0.01 * pair,
                            "procrustes_similarity": 0.9})
        for filename, rows in (("behavior.csv", behavior), ("probes.csv", probes),
            ("adaptation.csv", adaptation), ("similarities.csv", similarities)):
            _write_csv(cls.output, filename, rows)
        cls.input_digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.output.iterdir()}
        generate_hard_report(cls.output)
        cls.summary = json.loads((cls.output / "report_summary.json").read_text(encoding="utf-8"))
        cls.report = (cls.output / "REPORT.md").read_text(encoding="utf-8")

    def test_support_repeats_do_not_inflate_seed_count(self):
        stat = next(row["motif_accuracy"] for row in self.summary["adaptation"]
            if row["condition"] == "a" and row["shots"] == "8" and row["split"] == "ood_common")
        self.assertEqual(stat["n"], 3)
        self.assertEqual(len(stat["per_seed"]), 3)
        self.assertAlmostEqual(stat["mean"], 0.43)
        self.assertAlmostEqual(stat["std"], 0.02)
        self.assertIn("43.0 +/- 2.0% (n=3)", self.report)

    def test_shared_ood_and_each_iid_guard_are_kept_separate(self):
        at_eight = [row for row in self.summary["adaptation"] if row["condition"] == "a" and row["shots"] == "8"]
        self.assertEqual({row["split"] for row in at_eight}, {"ood_common", "iid_a", "iid_b"})
        iid = next(row["motif_accuracy"] for row in at_eight if row["split"] == "iid_a")
        self.assertAlmostEqual(iid["mean"], 0.2)
        self.assertIn("specialization or forgetting", self.report)
        self.assertIn("combination types are unseen by both experiences", self.report)
        self.assertNotIn("Experience B sees OOD", self.report)

    def test_adaptation_gain_and_iid_retention_use_paired_zero_shot_baseline(self):
        change = next(row for row in self.summary["adaptation_changes"] if row["condition"] == "a" and row["shots"] == 8)
        self.assertEqual(change["ood_motif_accuracy_change"]["n"], 3)
        self.assertAlmostEqual(change["ood_motif_accuracy_change"]["mean"], 0.01)
        self.assertAlmostEqual(change["own_iid_motif_accuracy_change"]["mean"], 0.0)
        self.assertIn("do not identify the fraction of molecules correct on both tasks", self.report)

    def test_paired_seed_contrasts_preserve_direction_and_cancel_seed_noise(self):
        stat = next(row for row in self.summary["paired_contrasts"]
            if row["comparison"] == "a_minus_b" and row["split"] == "ood_common" and row["metric"] == "motif_accuracy")
        self.assertEqual(stat["n"], 3)
        self.assertAlmostEqual(stat["mean"], -0.05)
        self.assertAlmostEqual(stat["std"], 0.0)
        self.assertIn("-5.0 +/- 0.0 percentage points (n=3)", self.report)

    def test_geometry_separates_pooling_and_initialization_controls(self):
        final_rows = [row for row in self.summary["similarities"] if row["layer"] == 2]
        self.assertEqual(len(final_rows), 2 * len(PAIR_NAMES))
        local = next(row for row in final_rows if row["pooling"] == "local" and row["pair_kind"] == "random_random")
        graph = next(row for row in final_rows if row["pooling"] == "graph" and row["pair_kind"] == "random_random")
        self.assertAlmostEqual(local["cka"]["mean"], 0.96)
        self.assertAlmostEqual(graph["cka"]["mean"], 0.76)
        self.assertIn("Pairs reuse encoders", self.report)

    def test_zero_shot_full_model_protocol_and_no_automatic_reuse_claim(self):
        self.assertTrue(any(row["shots"] == "0" for row in self.summary["adaptation"]))
        self.assertIn("Zero shots reuses each model's original task heads", self.report)
        self.assertIn("fine-tune the encoder and those same task heads", self.report)
        self.assertIn("does not replay pretraining data", self.report)
        self.assertIn("-10.0 +/- 0.0 percentage points", self.report)
        self.assertIn("High CKA or Procrustes similarity alone does not establish functional transfer", self.report)
        self.assertNotRegex(self.report, r"(?i)(?:these results|the results|this run)\s+(?:demonstrate|demonstrates|prove|proves|establish|establishes)\s+(?:that\s+)?(?:learned|functional|reusable)")

    def test_raw_root_control_is_deterministic_and_support_is_unexposed(self):
        raw = next(row["balanced_accuracy"] for row in self.summary["probes"] if row["probe_condition"] == "raw")
        self.assertEqual(raw["n"], 1)
        self.assertIsNone(raw["std"])
        self.assertEqual(self.summary["support_audit"]["pretraining_identity_overlap"], {"a": 0, "b": 0})
        self.assertIn("Exact support-pool identity overlap with pretraining is A = 0, B = 0", self.report)

    def test_report_preserves_measurements_and_writes_valid_figures(self):
        for filename, digest in self.input_digests.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), digest)
        self.assertEqual((self.output / "figures" / "hard_summary.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("<svg", (self.output / "figures" / "hard_summary.svg").read_text(encoding="utf-8"))
        self.assertEqual((self.output / "figures" / "motif_grid.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("![Recorded target motifs]", self.report)
        self.assertFalse(any("\u4e00" <= character <= "\u9fff" for character in self.report))


if __name__ == "__main__":
    unittest.main()
