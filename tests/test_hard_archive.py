"""Negative controls for the independent hard-record archival auditor."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from maintenance.audit_hard_archive import (
    ROOT, audit_record, bfs_distances, digest, main, mapped_molecule,
    reference_identity, rooted_component_identity,
)


class IndependentHardArchiveTests(unittest.TestCase):
    def test_rooted_topology_identity_is_invariant_to_atom_order(self):
        molecule = Chem.MolFromSmiles("CCOC(CC)COC")
        root, indices = 0, list(range(molecule.GetNumAtoms()))
        expected = rooted_component_identity(molecule, indices, root)
        order = list(reversed(indices))
        renumbered = Chem.RenumberAtoms(molecule, order)
        self.assertEqual(expected, rooted_component_identity(renumbered, indices, order.index(root)))
        self.assertNotEqual(expected, reference_identity("CCOC(CCC)OC"))

    def test_mapped_reference_recovers_original_order_and_rejects_missing_maps(self):
        molecule = Chem.MolFromSmiles("CCOC(CC)COC")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        alternate = Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
        recovered = mapped_molecule(alternate)
        np.testing.assert_array_equal(Chem.GetAdjacencyMatrix(molecule), Chem.GetAdjacencyMatrix(recovered))
        with self.assertRaisesRegex(ValueError, "bijection"):
            mapped_molecule("CCO")

    def test_bfs_excludes_the_cut_attachment_edge(self):
        adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 1],
                              [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.assertEqual(bfs_distances(adjacency, 0), {0: 0, 1: 1, 2: 2, 3: 2})
        self.assertEqual(bfs_distances(adjacency, 2, (1, 2)), {2: 0})

    def test_cli_refuses_to_overwrite_before_reading_an_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--record", str(Path(temporary) / "absent"), "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_independent_audit_detects_feature_and_training_selection_corruption(self):
        source = ROOT / "experiments" / "smoke_hard_v2"
        if not source.exists():
            self.skipTest("The independent smoke archival fixture is not available.")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            for name in ("molecules.jsonl", "graphs.npz", "splits.json", "config.json",
                         "environment.json", "audit.json"):
                shutil.copyfile(source / name, archive / name)
            shutil.copytree(source / "adaptation_support", archive / "adaptation_support")
            inputs = sorted(path for path in archive.rglob("*") if path.is_file())
            manifest = {"sha256": {str(path.relative_to(archive)): digest(path) for path in inputs}}
            (archive / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            with np.load(archive / "graphs.npz", allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
            arrays["x"][0, 0, 0] = 1 - arrays["x"][0, 0, 0]
            np.savez_compressed(archive / "graphs.npz", **arrays)
            splits = json.loads((archive / "splits.json").read_text(encoding="utf-8"))
            splits["train_a"].pop()
            (archive / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
            result = audit_record(archive)
            self.assertEqual(result["status"], "FAIL")
            for name in ("every_frozen_manifest_file_matches_sha256",
                         "all_13_ordinary_features_from_chemistry",
                         "exact_per_motif_matched_lowest_ID_training_selection"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)


if __name__ == "__main__":
    unittest.main()
