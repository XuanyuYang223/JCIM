"""Scientific checks for molecular identity, graph queries, and held-out cells."""

import unittest

import numpy as np
from rdkit import Chem

from jcim.data import (
    FEATURE_NAMES, N_FEATURES, audit_splits, canonical_identity,
    generate_dataset, split_dataset,
)


class MolecularDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_dataset()
        cls.splits = split_dataset(cls.records)

    def test_full_factorial_grid_has_unique_molecular_identities(self):
        self.assertEqual(len(self.records), 8 * 10 * 4 * 4 * 3)
        self.assertEqual(len({r["smiles"] for r in self.records}), len(self.records))
        self.assertEqual(len({r["id"] for r in self.records}), len(self.records))
        counts = {}
        for record in self.records:
            cell = (record["target_family"], record["position"])
            counts[cell] = counts.get(cell, 0) + 1
        self.assertEqual(set(counts.values()), {160})

    def test_identity_is_invariant_to_random_smiles_and_atom_maps(self):
        for record in self.records[::97]:
            molecule = Chem.MolFromSmiles(record["smiles"])
            random_smiles = Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
            self.assertEqual(canonical_identity(random_smiles), record["smiles"])
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            self.assertEqual(canonical_identity(Chem.MolToSmiles(molecule)), record["smiles"])

    def test_graph_queries_and_path_labels_are_consistent(self):
        for record in self.records:
            x, adj, mask = record["x"], record["adj"], record["local_mask"]
            self.assertEqual(x.shape, (len(mask), N_FEATURES))
            self.assertEqual(N_FEATURES, 18)
            np.testing.assert_array_equal(adj, adj.T)
            np.testing.assert_array_equal(np.diag(adj), 0)
            for feature, index_key in (
                ("query_anchor_ring_atom", "anchor_ring_idx"),
                ("query_target_ring_atom", "target_ring_idx"),
                ("query_anchor_functional_root", "anchor_root_idx"),
                ("query_target_functional_root", "target_root_idx"),
            ):
                query = x[:, FEATURE_NAMES.index(feature)]
                self.assertEqual(query.sum(), 1)
                self.assertEqual(query[record[index_key]], 1)
            np.testing.assert_array_equal(mask, x[:, FEATURE_NAMES.index("query_target_fragment")])
            self.assertFalse(any(mask[i] for i in record["ring_indices"]))
            start, goal = record["anchor_root_idx"], record["target_root_idx"]
            frontier, visited, distance = {start}, {start}, 0
            while goal not in frontier:
                frontier = {int(j) for i in frontier for j in np.flatnonzero(adj[i])} - visited
                self.assertTrue(frontier)
                visited.update(frontier)
                distance += 1
            self.assertEqual(distance, record["y_distance"])
            self.assertEqual(distance, record["anchor_spacer"] + record["target_spacer"]
                             + record["y_position"] + 3)
        self.assertEqual({r["y_distance"] for r in self.records}, set(range(3, 12)))

    def test_all_evaluation_identities_are_absent_from_pretraining(self):
        analysis = set(self.splits["analysis"])
        validation = set(self.splits["val"])
        for train_name in ("train_a", "train_b"):
            training = set(self.splits[train_name])
            self.assertFalse(training & analysis)
            self.assertFalse(training & validation)
        for train_name, family in (("train_a", "fluoro"), ("train_b", "chloro")):
            self.assertFalse(any(self.records[i]["target_family"] == family
                                 and self.records[i]["position"] == "meta"
                                 for i in self.splits[train_name]))
        self.assertFalse(set(self.splits["adaptation_pool"]) & set(self.splits["train_a"]))
        self.assertFalse(set(self.splits["adaptation_pool"]) & set(self.splits["adaptation_test"]))
        self.assertEqual(self.splits["adaptation_test"], self.splits["ood_f_meta"])
        self.assertFalse(set(self.splits["probe_source"]) & set(self.splits["probe_target"]))
        self.assertFalse(audit_splits(self.records, self.splits)["split_identity_leakage"])

    def test_split_assignment_is_independent_of_record_order(self):
        reversed_records = list(reversed(self.records))
        reversed_splits = split_dataset(reversed_records)
        for name, indices in self.splits.items():
            original_ids = {self.records[i]["id"] for i in indices}
            reversed_ids = {reversed_records[i]["id"] for i in reversed_splits[name]}
            self.assertEqual(original_ids, reversed_ids)

    def test_split_assignment_is_invariant_to_alternative_smiles(self):
        rewritten = [dict(record) for record in self.records]
        for record in rewritten:
            molecule = Chem.MolFromSmiles(record["smiles"])
            record["smiles"] = Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
        self.assertEqual(split_dataset(rewritten), self.splits)

    def test_repeated_molecule_is_rejected_before_partitioning(self):
        with self.assertRaisesRegex(ValueError, "deduplicated"):
            split_dataset(self.records + [self.records[0]])


if __name__ == "__main__":
    unittest.main()
