"""Scientific controls for connectivity motifs and isolated composition holdouts."""

import unittest
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from jcim.data import canonical_identity
from jcim.hard_data import (
    FEATURE_NAMES, MOTIF_FRAGMENTS, N_FEATURES, audit_hard_splits,
    generate_hard_dataset, heldout_cell, split_hard_dataset,
)


class HardMolecularDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_hard_dataset()
        cls.splits = split_hard_dataset(cls.records)

    def test_connectivity_motifs_share_formula_and_ordinary_atom_feature_bags(self):
        feature_bags, formulas, rooted_identities = [], [], []
        for _, smiles in MOTIF_FRAGMENTS:
            molecule = Chem.MolFromSmiles(smiles)
            formulas.append(rdMolDescriptors.CalcMolFormula(molecule))
            feature_bags.append(Counter((a.GetSymbol(), a.GetDegree(), a.GetTotalNumHs())
                                        for a in molecule.GetAtoms()))
            rooted = Chem.Mol(molecule)
            rooted.GetAtomWithIdx(0).SetAtomMapNum(1)
            rooted_identities.append(Chem.MolToSmiles(rooted))
            root = molecule.GetAtomWithIdx(0)
            self.assertEqual((root.GetSymbol(), root.GetDegree(), root.GetTotalNumHs()), ("C", 1, 3))
        self.assertEqual(len(set(formulas)), 1)
        self.assertTrue(all(bag == feature_bags[0] for bag in feature_bags))
        self.assertEqual(len(set(rooted_identities)), 8)

    def test_full_grid_is_unique_and_balanced(self):
        self.assertEqual(len(self.records), 8 * 3 * 8 * 3 * 3 * 4)
        self.assertEqual(len({r["smiles"] for r in self.records}), len(self.records))
        cells = Counter((r["target_family"], r["position"]) for r in self.records)
        self.assertEqual(set(cells.values()), {288})
        self.assertEqual({r["endpoint_kind"] for r in self.records}, {0, 1})

    def test_raw_local_root_does_not_expose_motif_label(self):
        roots = [tuple(r["x"][r["target_attachment_idx"]]) for r in self.records]
        self.assertEqual(len(set(roots)), 1)
        self.assertEqual(N_FEATURES, 18)
        self.assertNotIn("query_target_fragment", FEATURE_NAMES)
        for r in self.records[::19]:
            self.assertEqual(r["local_mask"].sum(), 1)
            self.assertEqual(r["y_target_atom"].sum(), 9)
            np.testing.assert_array_equal(r["local_mask"], r["x"][:, 15])

    def test_path_query_is_correct_under_graph_reference_order(self):
        for r in self.records[::13]:
            mapped = Chem.MolFromSmiles(r["atom_map_reference_smiles"])
            map_to_index = {a.GetAtomMapNum() - 1: a.GetIdx() for a in mapped.GetAtoms()}
            start, goal = r["anchor_root_idx"], r["path_endpoint_idx"]
            distance = len(Chem.GetShortestPath(mapped, map_to_index[start], map_to_index[goal])) - 1
            self.assertEqual(distance, r["y_distance"])
            for feature, key in ((13, "anchor_ring_idx"), (14, "target_ring_idx"),
                                 (15, "target_attachment_idx"), (16, "anchor_root_idx"),
                                 (17, "path_endpoint_idx")):
                self.assertEqual(r["x"][:, feature].sum(), 1)
                self.assertEqual(r["x"][r[key], feature], 1)
            np.testing.assert_array_equal(r["adj"], r["adj"].T)
            self.assertFalse(np.diag(r["adj"]).any())

    def test_support_and_test_are_unseen_by_both_pretrained_experiences(self):
        for train_name, included in (("train_a", 0), ("train_b", 1)):
            train = set(self.splits[train_name])
            for name in ("val", "analysis", "adaptation_reserve"):
                self.assertFalse(train & set(self.splits[name]))
            self.assertTrue(all(heldout_cell(self.records[i]) == included for i in train))
        self.assertFalse(set(self.splits["adaptation_pool"]) & set(self.splits["adaptation_test"]))
        self.assertEqual(self.splits["adaptation_test"], self.splits["ood_common"])
        self.assertFalse(audit_hard_splits(self.records, self.splits)["split_identity_leakage"])

    def test_training_experiences_have_matched_motif_counts_and_deterministic_selection(self):
        audit = audit_hard_splits(self.records, self.splits)
        self.assertEqual(audit["train_motif_counts"]["train_a"], audit["train_motif_counts"]["train_b"])
        self.assertEqual(audit["effective_train_counts"]["train_a"], audit["effective_train_counts"]["train_b"])
        for family in range(8):
            available = {
                name: sorted((i for i in self.splits["candidate_train"]
                              if self.records[i]["y_family"] == family
                              and heldout_cell(self.records[i]) == cell),
                             key=lambda i: self.records[i]["id"])
                for name, cell in (("train_a", 0), ("train_b", 1))
            }
            count = min(map(len, available.values()))
            for name in available:
                selected = {i for i in self.splits[name] if self.records[i]["y_family"] == family}
                self.assertEqual(selected, set(available[name][:count]))
        self.assertEqual(sum(audit["train_motif_counts"]["train_a"].values()),
                         audit["effective_train_counts"]["train_a"])

    def test_split_is_invariant_to_order_and_smiles_rewriting(self):
        reversed_records = list(reversed(self.records))
        reversed_splits = split_hard_dataset(reversed_records)
        for name, indices in self.splits.items():
            self.assertEqual({self.records[i]["id"] for i in indices},
                             {reversed_records[i]["id"] for i in reversed_splits[name]})
        rewritten = [dict(r) for r in self.records]
        for r in rewritten:
            molecule = Chem.MolFromSmiles(r["smiles"])
            r["smiles"] = Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
            self.assertEqual(canonical_identity(r["smiles"]), canonical_identity(r["atom_map_reference_smiles"]))
        self.assertEqual(split_hard_dataset(rewritten), self.splits)

    def test_repeated_molecular_identity_and_mutated_split_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "deduplicated"):
            split_hard_dataset(self.records + [self.records[0]])
        broken = {name: list(indices) for name, indices in self.splits.items()}
        broken["train_a"].append(broken["adaptation_pool"][0])
        with self.assertRaisesRegex(ValueError, "checkerboard"):
            audit_hard_splits(self.records, broken)
        missing = {name: list(indices) for name, indices in self.splits.items()}
        missing["train_a"].pop()
        with self.assertRaisesRegex(ValueError, "matched checkerboard"):
            audit_hard_splits(self.records, missing)


if __name__ == "__main__":
    unittest.main()
