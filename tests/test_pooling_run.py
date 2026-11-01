"""Frozen-source and fixed-protocol safeguards for the pooling diagnostic."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from jcim.pooling_run import ROOT, validate_config, validate_output


class PoolingRunTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/pooling_v6.json").read_text())

    def test_primary_protocol_and_explicit_smoke_budget(self):
        validate_config(self.config)
        smoke = {**self.config, "epochs": 2, "seeds": [0]}
        validate_config(smoke)
        for key, value in (("shallow_layer", 2), ("alpha", 0), ("seeds", [0, 0]),
                           ("epochs", True), ("epochs", 121), ("batch_size", 64)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.config)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_output_cannot_contain_or_modify_archived_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for output in (source, source / "analysis", root):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    validate_output(output, [source])
            fresh = root / "fresh"
            self.assertEqual(validate_output(fresh, [source]), fresh.resolve())
            fresh.mkdir()
            (fresh / "existing.txt").write_text("Frozen evidence")
            with self.assertRaises(FileExistsError):
                validate_output(fresh, [source])


if __name__ == "__main__":
    unittest.main()
