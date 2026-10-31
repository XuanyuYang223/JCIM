"""Controlled benzene chemistry and molecule-identity-safe evaluation splits.

The graphs carry only ordinary atom features and explicit task-query masks.
Position, spacer lengths, and family labels are metadata, never input features.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

import numpy as np
from rdkit import Chem


TARGET_FRAGMENTS = (
    ("fluoro", "F"),
    ("chloro", "Cl"),
    ("bromo", "Br"),
    ("hydroxy", "O"),
    ("amino", "N"),
    ("thiol", "S"),
    ("cyano", "C#N"),
    ("nitro", "[N+](=O)[O-]"),
)
ANCHOR_FRAGMENTS = (
    ("amide", "C(=O)N"),
    ("carboxyl", "C(=O)O"),
    ("methyl_ester", "C(=O)OC"),
    ("acetyl", "C(=O)C"),
    ("methylsulfonyl", "S(=O)(=O)C"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("trifluoromethyl", "C(F)(F)F"),
    ("isopropyl", "C(C)C"),
    ("ethyl", "CC"),
    ("dimethylamide", "C(=O)N(C)C"),
)
TARGET_FAMILIES = tuple(name for name, _ in TARGET_FRAGMENTS)
ANCHOR_FAMILIES = tuple(name for name, _ in ANCHOR_FRAGMENTS)
POSITION_NAMES = ("ortho", "meta", "para")
ELEMENTS = ("C", "N", "O", "F", "S", "Cl", "Br")
FEATURE_NAMES = tuple(f"element_{element}" for element in ELEMENTS) + (
    "element_other",
    "aromatic",
    "formal_charge_div_2",
    "degree_div_4",
    "hydrogen_count_div_4",
    "in_ring",
    "query_anchor_ring_atom",
    "query_target_ring_atom",
    "query_target_fragment",
    "query_anchor_functional_root",
    "query_target_functional_root",
)
N_FEATURES = len(FEATURE_NAMES)
N_DISTANCE_CLASSES = 12

Record = dict[str, Any]


def canonical_identity(smiles: str) -> str:
    """Return a map-free canonical identity for a valid molecular SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _construct_molecule(
    anchor_smiles: str,
    target_smiles: str,
    anchor_spacer: int,
    target_spacer: int,
    ring_distance: int,
) -> tuple[Chem.Mol, dict[str, Any]]:
    ring = Chem.MolFromSmiles("c1ccccc1")
    anchor = Chem.MolFromSmiles("C" * anchor_spacer + anchor_smiles)
    target = Chem.MolFromSmiles("C" * target_spacer + target_smiles)
    if ring is None or anchor is None or target is None:
        raise ValueError("A configured molecular fragment could not be parsed.")
    anchor_offset = ring.GetNumAtoms()
    target_offset = anchor_offset + anchor.GetNumAtoms()
    molecule = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(ring, anchor), target))
    molecule.AddBond(0, anchor_offset, Chem.BondType.SINGLE)
    molecule.AddBond(ring_distance, target_offset, Chem.BondType.SINGLE)
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)
    indices = {
        "ring_indices": list(range(6)),
        "anchor_ring_idx": 0,
        "target_ring_idx": ring_distance,
        "anchor_attachment_idx": anchor_offset,
        "target_attachment_idx": target_offset,
        "anchor_root_idx": anchor_offset + anchor_spacer,
        "target_root_idx": target_offset + target_spacer,
        "target_fragment_indices": list(range(target_offset, molecule.GetNumAtoms())),
    }
    return molecule, indices


def _graph_arrays(molecule: Chem.Mol, indices: dict[str, Any]) -> dict[str, np.ndarray]:
    """Featurize an atom graph without using generated scientific labels."""
    n_atoms = molecule.GetNumAtoms()
    x = np.zeros((n_atoms, N_FEATURES), dtype=np.float32)
    target_atoms = frozenset(indices["target_fragment_indices"])
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        element = atom.GetSymbol()
        x[i, ELEMENTS.index(element) if element in ELEMENTS else len(ELEMENTS)] = 1.0
        x[i, 8:] = (
            float(atom.GetIsAromatic()),
            atom.GetFormalCharge() / 2.0,
            atom.GetDegree() / 4.0,
            atom.GetTotalNumHs() / 4.0,
            float(atom.IsInRing()),
            float(i == indices["anchor_ring_idx"]),
            float(i == indices["target_ring_idx"]),
            float(i in target_atoms),
            float(i == indices["anchor_root_idx"]),
            float(i == indices["target_root_idx"]),
        )
    adj = Chem.GetAdjacencyMatrix(molecule).astype(np.float32)
    local_mask = np.zeros(n_atoms, dtype=np.float32)
    local_mask[indices["target_fragment_indices"]] = 1.0
    return {"x": x, "adj": adj, "local_mask": local_mask}


def generate_dataset(spacer_lengths: Iterable[int] = range(4)) -> list[Record]:
    """Enumerate and deduplicate all controlled disubstituted benzene molecules.

    Spacer length counts methylene carbons between the ring and each named
    terminal group. The distance task queries the root atom of each terminal
    group, so its exact answer ranges from 3 to 11 for the default grid.
    Ring atoms and functional-root atoms are explicitly marked query endpoints.
    """
    spacer_lengths = tuple(sorted(set(spacer_lengths)))
    if not spacer_lengths or any(not isinstance(n, int) or n < 0 for n in spacer_lengths):
        raise ValueError("Spacer lengths must be a nonempty collection of nonnegative integers.")
    identities: dict[str, Record] = {}
    for family_idx, (target_family, target_smiles) in enumerate(TARGET_FRAGMENTS):
        for anchor_family, anchor_smiles in ANCHOR_FRAGMENTS:
            for target_spacer in spacer_lengths:
                for anchor_spacer in spacer_lengths:
                    for position_idx, position in enumerate(POSITION_NAMES):
                        molecule, indices = _construct_molecule(
                            anchor_smiles,
                            target_smiles,
                            anchor_spacer,
                            target_spacer,
                            position_idx + 1,
                        )
                        smiles = canonical_identity(Chem.MolToSmiles(molecule))
                        distance = len(Chem.GetShortestPath(
                            molecule, indices["anchor_root_idx"], indices["target_root_idx"]
                        )) - 1
                        mapped_molecule = Chem.Mol(molecule)
                        for atom in mapped_molecule.GetAtoms():
                            atom.SetAtomMapNum(atom.GetIdx() + 1)
                        record = {
                            "id": hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20],
                            "smiles": smiles,
                            "atom_map_reference_smiles": Chem.MolToSmiles(mapped_molecule),
                            "target_family": target_family,
                            "anchor_family": anchor_family,
                            "target_spacer": target_spacer,
                            "anchor_spacer": anchor_spacer,
                            "position": position,
                            "y_position": position_idx,
                            "y_family": family_idx,
                            "y_distance": distance,
                            **indices,
                            **_graph_arrays(molecule, indices),
                        }
                        if smiles in identities:
                            existing = identities[smiles]
                            label_keys = ("y_position", "y_family", "y_distance")
                            if any(existing[key] != record[key] for key in label_keys):
                                raise ValueError(f"Conflicting queries for canonical molecule: {smiles}")
                        else:
                            identities[smiles] = record
    return [identities[smiles] for smiles in sorted(identities)]


def _partition_score(canonical_smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{canonical_smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_dataset(records: list[Record], seed: int = 20261030) -> dict[str, list[int]]:
    """Create a shared identity partition before applying experience exclusions.

    A stable hash assigns approximately 65% to candidate training, 10% to
    validation, and 25% to testing. Experience A never trains on fluoro-meta;
    experience B never trains on chloro-meta. Validation excludes both cells.

    Analysis and probe sets intentionally reuse the untouched test partition.
    Adaptation test aliases the fluoro-meta OOD test, while its training pool
    contains only candidate-training fluoro-meta identities. Those pool molecules
    are unseen by experience A but can be seen during experience B pretraining.
    """
    smiles = [canonical_identity(record["smiles"]) for record in records]
    if len(set(smiles)) != len(smiles):
        raise ValueError("Canonical molecular identities must be deduplicated before splitting.")
    splits: dict[str, list[int]] = {name: [] for name in (
        "candidate_train", "train_a", "train_b", "val", "iid_test",
        "ood_f_meta", "ood_cl_meta", "analysis", "adaptation_pool",
        "adaptation_test", "probe_source", "probe_target", "discarded_val_ood",
    )}
    for index, record in enumerate(records):
        score = _partition_score(smiles[index], seed)
        f_meta = record["target_family"] == "fluoro" and record["position"] == "meta"
        cl_meta = record["target_family"] == "chloro" and record["position"] == "meta"
        if score < 0.65:
            splits["candidate_train"].append(index)
            if not f_meta:
                splits["train_a"].append(index)
            if not cl_meta:
                splits["train_b"].append(index)
            if f_meta:
                splits["adaptation_pool"].append(index)
        elif score < 0.75:
            splits["discarded_val_ood" if f_meta or cl_meta else "val"].append(index)
        else:
            splits["analysis"].append(index)
            if f_meta:
                splits["ood_f_meta"].append(index)
                splits["adaptation_test"].append(index)
            elif cl_meta:
                splits["ood_cl_meta"].append(index)
            else:
                splits["iid_test"].append(index)
            if record["position"] == "ortho":
                splits["probe_source"].append(index)
            elif record["position"] == "meta":
                splits["probe_target"].append(index)
    audit_splits(records, splits)
    return splits


def audit_splits(records: list[Record], splits: dict[str, list[int]]) -> dict[str, Any]:
    """Raise on identity leakage; return compact, JSON-serializable split counts."""
    indexed = {name: set(indices) for name, indices in splits.items()}
    for name, indices in splits.items():
        if len(indices) != len(indexed[name]):
            raise ValueError(f"Repeated record index in split {name!r}.")
        if any(i < 0 or i >= len(records) for i in indices):
            raise ValueError(f"Out-of-bounds record index in split {name!r}.")
    canonical = [canonical_identity(record["smiles"]) for record in records]
    if len(set(canonical)) != len(canonical):
        raise ValueError("The dataset contains repeated canonical molecular identities.")
    validation = indexed["val"] | indexed["discarded_val_ood"]
    if indexed["candidate_train"] & validation or indexed["candidate_train"] & indexed["analysis"]:
        raise ValueError("Candidate training overlaps validation or testing.")
    if validation & indexed["analysis"]:
        raise ValueError("Validation overlaps testing.")
    if indexed["candidate_train"] | validation | indexed["analysis"] != set(range(len(records))):
        raise ValueError("The identity partition does not cover the full dataset.")
    for train_name, heldout_family in (("train_a", "fluoro"), ("train_b", "chloro")):
        if not indexed[train_name] <= indexed["candidate_train"]:
            raise ValueError(f"{train_name} extends beyond the candidate training partition.")
        if any(records[i]["target_family"] == heldout_family and records[i]["position"] == "meta"
               for i in indexed[train_name]):
            raise ValueError(f"Held-out {heldout_family}-meta molecules leaked into {train_name}.")
    if indexed["adaptation_pool"] & indexed["train_a"]:
        raise ValueError("Adaptation pool molecules leaked into experience A pretraining.")
    if indexed["adaptation_pool"] & indexed["adaptation_test"]:
        raise ValueError("Adaptation training and test identities overlap.")
    if indexed["probe_source"] & indexed["probe_target"]:
        raise ValueError("Probe source and target identities overlap.")
    if indexed["adaptation_test"] != indexed["ood_f_meta"]:
        raise ValueError("Adaptation test must alias the held-out fluoro-meta test.")
    if not indexed["adaptation_pool"] <= indexed["candidate_train"]:
        raise ValueError("Adaptation support must be in the candidate-training partition.")
    for name, expected_family in (("ood_f_meta", "fluoro"), ("ood_cl_meta", "chloro"),
                                  ("adaptation_pool", "fluoro")):
        if any(records[i]["target_family"] != expected_family or records[i]["position"] != "meta"
               for i in indexed[name]):
            raise ValueError(f"Incorrect chemistry in split {name!r}.")
    test_cells = [indexed[name] for name in ("iid_test", "ood_f_meta", "ood_cl_meta")]
    if any(a & b for j, a in enumerate(test_cells) for b in test_cells[j + 1:]):
        raise ValueError("Behavioral test cells overlap.")
    if set().union(*test_cells) != indexed["analysis"]:
        raise ValueError("Behavioral test cells must partition the shared analysis set.")
    for name, expected_position in (("probe_source", "ortho"), ("probe_target", "meta")):
        expected = {i for i in indexed["analysis"] if records[i]["position"] == expected_position}
        if indexed[name] != expected:
            raise ValueError(f"Incorrect membership in {name!r}.")
    return {
        "n_molecules": len(records),
        "unique_canonical_identities": len(set(canonical)),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "position_counts": dict(Counter(record["position"] for record in records)),
        "target_family_counts": dict(Counter(record["target_family"] for record in records)),
        "feature_names": list(FEATURE_NAMES),
        "feature_dimension": N_FEATURES,
        "distance_range": [min(record["y_distance"] for record in records),
                           max(record["y_distance"] for record in records)],
        "query_endpoints_explicitly_marked": True,
        "split_identity_leakage": False,
    }
