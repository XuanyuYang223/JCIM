"""Structural checks against atom-order and padding shortcuts."""

import unittest

import numpy as np
import torch

from jcim.data import generate_dataset
from jcim.model import MolecularGNN, tensorize


class ModelTests(unittest.TestCase):
    def test_atom_permutation_and_padding_invariance(self):
        torch.manual_seed(8)
        records = generate_dataset(spacer_lengths=(0, 3))
        record = records[37]
        perm = np.random.default_rng(9).permutation(len(record["x"]))
        permuted = dict(record, x=record["x"][perm], adj=record["adj"][np.ix_(perm, perm)])
        model = MolecularGNN(hidden_dim=16, layers=3).eval()
        with torch.no_grad():
            original = model(tensorize([record], torch.device("cpu")), representations=True)
            shuffled = model(tensorize([permuted], torch.device("cpu")), representations=True)
            padded = model(tensorize([record, max(records, key=lambda r: len(r["x"]))],
                                     torch.device("cpu")), representations=True)
        for task in ("family", "position", "distance"):
            torch.testing.assert_close(original[task], shuffled[task], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(original[task][0], padded[task][0], atol=2e-6, rtol=2e-6)
        for before, after in zip(original["local"], shuffled["local"]):
            torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
