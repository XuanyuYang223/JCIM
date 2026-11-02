"""Factorial contrasts retain conditioning, interaction signs, and paired seeds."""

from itertools import product
import unittest

from jcim.readout_report import FACTORS, aggregate, factorial_contrasts, matched_random


def cube(seed=0, regime="end_to_end", classifier="mlp"):
    rows = []
    objectives = ("joint", "family") if classifier == "mlp" else ("ridge",)
    for scope, norm, objective in product(FACTORS["scope"], FACTORS["normalization"], objectives):
        s, n, o = scope == "shallow_only", norm == "train_z", objective == "family"
        value = .1 + .03*s + .05*n + .07*o + .11*s*n + .13*s*o + .17*n*o + .19*s*n*o
        rows.append({"regime": regime, "experience": "a", "classifier": classifier, "seed": seed,
            "scope": scope, "normalization": norm, "objective": objective, "split": "ood_common",
            "n_test": 443, "accuracy": value, "balanced_accuracy": value})
    return rows


class ReadoutReportTests(unittest.TestCase):
    def test_main_and_interaction_effects_are_finite_differences(self):
        edges, effects = factorial_contrasts(cube())
        self.assertEqual(len(edges), 12)
        self.assertEqual(len(effects), 7)
        effect = {row["factors"]: row["accuracy"] for row in effects}
        self.assertAlmostEqual(effect["scope"], .03 + .11/2 + .13/2 + .19/4)
        self.assertAlmostEqual(effect["normalization"], .05 + .11/2 + .17/2 + .19/4)
        self.assertAlmostEqual(effect["objective"], .07 + .13/2 + .17/2 + .19/4)
        self.assertAlmostEqual(effect["scope+normalization"], .11 + .19/2)
        self.assertAlmostEqual(effect["scope+objective"], .13 + .19/2)
        self.assertAlmostEqual(effect["normalization+objective"], .17 + .19/2)
        self.assertAlmostEqual(effect["scope+normalization+objective"], .19)
        scopes = [row for row in edges if row["factor"] == "scope"]
        at_origin = next(row for row in scopes if row["normalization"] == "raw" and row["objective"] == "joint")
        self.assertAlmostEqual(at_origin["accuracy"], .03)
        self.assertEqual(at_origin["contrast"], "shallow_only_minus_full512")

    def test_ridge_panel_has_two_factors_without_fabricated_objective_levels(self):
        edges, effects = factorial_contrasts(cube(classifier="ridge"))
        self.assertEqual(len(edges), 4)
        self.assertEqual(len(effects), 3)
        self.assertTrue(all(row["objective"] == "ridge" for row in edges))
        self.assertTrue(all("objective" not in row["factors"] for row in effects))

    def test_missing_or_duplicate_factor_cells_fail(self):
        for rows in (cube()[:-1], cube() + [cube()[0]]):
            with self.subTest(n=len(rows)), self.assertRaises(ValueError):
                factorial_contrasts(rows)

    def test_random_pairing_holds_every_readout_factor_fixed(self):
        trained, random = cube(), cube(regime="random")
        for row in trained:
            row["accuracy"] += .02
        pairs = matched_random(trained + random)
        self.assertEqual(len(pairs), 8)
        self.assertTrue(all(abs(row["accuracy"] - .02) < 1e-12 for row in pairs))
        for broken in (trained + random[:-1], trained + random + [random[0]]):
            with self.assertRaises(ValueError):
                matched_random(broken)

    def test_seed_aggregation_does_not_count_cells_as_replicates(self):
        keys = ("regime", "experience", "classifier", "factors", "order", "split")
        rows = [{"regime": "end_to_end", "experience": "a", "classifier": "mlp",
                 "factors": "scope", "order": 1, "split": "ood_common", "seed": seed,
                 "accuracy": value, "balanced_accuracy": value} for seed, value in enumerate((.1, .2, .3))]
        result = aggregate(rows, keys)[0]["accuracy"]
        self.assertEqual(result["n_seeds"], 3)
        self.assertAlmostEqual(result["std"], .1)
        self.assertAlmostEqual(result["mean"], .2)
        with self.assertRaises(ValueError):
            aggregate([rows[0], rows[0]], keys)


if __name__ == "__main__":
    unittest.main()
