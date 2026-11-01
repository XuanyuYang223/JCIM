"""Paired seeds and nested diagnostic replicates retain their proper units."""

import copy
import unittest

from jcim.pooling_report import pair_source, summarize_direct, summarize_pairs


class PoolingReportTests(unittest.TestCase):
    def source_rows(self):
        rows = []
        for seed, whole, target in ((0, .1, .8), (1, .4, .9), (2, .8, .7)):
            for pool, value in (("whole_graph", whole), ("target", target)):
                rows.append({"regime": "end_to_end", "experience": "a", "classifier": "ridge",
                    "split": "ood_common", "seed": seed, "pool": pool, "n_test": 443,
                    "accuracy": value, "balanced_accuracy": value})
        return rows

    def test_pooling_effects_are_paired_before_seed_uncertainty(self):
        paired = pair_source(self.source_rows())
        values = summarize_pairs(paired)[0]["accuracy"]
        self.assertAlmostEqual(values["mean"], 1.1 / 3)
        self.assertEqual(values["n_seeds"], 3)
        self.assertAlmostEqual(values["per_seed"]["0"], .7)
        self.assertAlmostEqual(values["per_seed"]["1"], .5)
        self.assertAlmostEqual(values["per_seed"]["2"], -.1)
        self.assertAlmostEqual(values["std"], (.17333333333333334) ** .5)
        for altered in (self.source_rows()[:-1], self.source_rows() + [self.source_rows()[0]]):
            with self.assertRaises(ValueError):
                pair_source(altered)

    def direct_rows(self):
        rows = []
        for seed in (0, 1, 2):
            for experience in ("a", "b"):
                for repeat in (0, 1, 2):
                    for source in range(3):
                        for target in range(3):
                            if source != target:
                                value = .2 + seed * .1 + repeat * .03
                                rows.append({"regime": "random", "experience": experience,
                                    "pool": "whole_graph", "split": "target_all", "seed": seed,
                                    "repeat": repeat, "source_position": source, "target_position": target,
                                    "n_test": 100, "accuracy": value, "balanced_accuracy": value})
        return rows

    def test_directions_and_draws_are_averaged_within_seed(self):
        seeds, panels = summarize_direct(self.direct_rows())
        self.assertEqual(len(seeds), 3)
        self.assertTrue(all(row["n_directions"] == 6 and row["n_support_draws"] == 3 for row in seeds))
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0]["experience"], "shared")
        stats = panels[0]["balanced_accuracy"]
        self.assertEqual(stats["n_seeds"], 3)
        self.assertAlmostEqual(stats["mean"], .33)
        self.assertAlmostEqual(stats["std"], .1)

    def test_missing_direction_and_inconsistent_shared_reference_fail(self):
        missing = [row for row in self.direct_rows() if not (row["source_position"] == 0 and row["target_position"] == 1)]
        with self.assertRaises(ValueError):
            summarize_direct(missing)
        changed = copy.deepcopy(self.direct_rows())
        next(row for row in changed if row["experience"] == "b")["accuracy"] = .99
        with self.assertRaises(ValueError):
            summarize_direct(changed)


if __name__ == "__main__":
    unittest.main()
