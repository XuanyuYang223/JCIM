"""The readout factorial cannot silently change its causal comparison."""

import json
import unittest

from jcim.readout_run import ROOT, validate_config


class ReadoutRunTests(unittest.TestCase):
    def test_complete_grid_and_smoke_budget_are_explicit(self):
        config = json.loads((ROOT / "configs/readout_v7.json").read_text())
        validate_config(config)
        validate_config({**config, "seeds": [0], "epochs": 2})
        for key, value in (("scopes", ["full512"]), ("objectives", ["joint"]),
                           ("normalizations", ["raw"]), ("alpha", .1),
                           ("epochs", 0), ("epochs", True), ("seeds", [0, 0])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**config, key: value})


if __name__ == "__main__":
    unittest.main()
