"""Connectivity-controlled molecular benchmark with identity-safe holdouts.

The eight rooted ether fragments have equal elemental, degree, and hydrogen
feature multisets. Only one attachment atom is used for local readout; the
target fragment membership is a supervision label and never an input feature.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

import numpy as np
from rdkit import Chem

from .data import ANCHOR_FRAGMENTS as _ANCHORS
from .data import ELEMENTS, POSITION_NAMES, canonical_identity


MOTIF_FRAGMENTS = tuple((f"motif_{i}", smiles) for i, smiles in enumerate((
    "CCOC(CCC)OC", "CCOC(CC)OCC", "CCOC(CC)COC", "COCC(CCC)OC",
    "COCC(CC)OCC", "COCC(CC)COC", "CCCC(OC)OCC", "CCCC(COC)OC",
)))
TARGET_FAMILIES = tuple(name for name, _ in MOTIF_FRAGMENTS)
ANCHOR_FRAGMENTS = _ANCHORS[:8]
ANCHOR_FAMILIES = tuple(name for name, _ in ANCHOR_FRAGMENTS)
SCAFFOLD_FRAGMENTS = (
    ("benzene", "c1ccccc1"),
    ("pyridine_n4", "c1cccnc1"),
    ("pyridine_n5", "c1ccccn1"),
)
DISTRACTOR_FRAGMENTS = (
    ("methyl", "C"), ("fluoro", "F"), ("hydroxy", "O"), ("cyano", "C#N"),
)
FEATURE_NAMES = tuple(f"element_{element}" for element in ELEMENTS) + (
    "element_other", "aromatic", "formal_charge_div_2", "degree_div_4",
    "hydrogen_count_div_4", "in_ring", "query_anchor_ring_atom",
    "query_target_ring_atom", "query_target_attachment_atom",
    "query_anchor_functional_root", "query_path_endpoint",
)
N_FEATURES = len(FEATURE_NAMES)
N_DISTANCE_CLASSES = 16
Record = dict[str, Any]


def _partition_score(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def heldout_cell(record: Record) -> int:
    """Return the predeclared motif-by-position checkerboard cell class."""
    return (int(record["y_family"]) + int(record["y_position"])) % 3


def _matched_training_indices(records: list[Record], candidate_indices: Iterable[int]) -> dict[str, list[int]]:
    """Match the number of identities per motif across the two experiences."""
    candidates = list(candidate_indices)
    matched: dict[str, list[int]] = {"train_a": [], "train_b": []}
    for family in sorted({int(records[i]["y_family"]) for i in candidates}):
        available = {
            name: sorted((i for i in candidates if int(records[i]["y_family"]) == family
                          and heldout_cell(records[i]) == cell), key=lambda i: records[i]["id"])
            for name, cell in (("train_a", 0), ("train_b", 1))
        }
        count = min(len(indices) for indices in available.values())
        for name in matched:
            matched[name].extend(available[name][:count])
    return matched


def _leaf_endpoint(motif: Chem.Mol, endpoint_kind: int) -> int:
    leaves = [atom.GetIdx() for atom in motif.GetAtoms()
              if atom.GetIdx() != 0 and atom.GetDegree() == 1
              and atom.GetSymbol() == "C"]
    ordered = sorted(leaves, key=lambda i: (len(Chem.GetShortestPath(motif, 0, i)), i))
    return ordered[0] if endpoint_kind == 0 else ordered[-1]


def _construct_molecule(
    motif_smiles: str, anchor_smiles: str, scaffold_smiles: str,
    distractor_smiles: str, anchor_spacer: int, ring_distance: int,
    distractor_choice: int, endpoint_kind: int,
) -> tuple[Chem.Mol, dict[str, Any]]:
    ring = Chem.MolFromSmiles(scaffold_smiles)
    anchor = Chem.MolFromSmiles("C" * anchor_spacer + anchor_smiles)
    motif = Chem.MolFromSmiles(motif_smiles)
    distractor = Chem.MolFromSmiles(distractor_smiles)
    if any(fragment is None for fragment in (ring, anchor, motif, distractor)):
        raise ValueError("A hard-benchmark molecular fragment could not be parsed.")
    available = [atom.GetIdx() for atom in ring.GetAtoms()
                 if atom.GetSymbol() == "C" and atom.GetIdx() not in (0, ring_distance)]
    distractor_ring_idx = available[distractor_choice % len(available)]
    anchor_offset = ring.GetNumAtoms()
    target_offset = anchor_offset + anchor.GetNumAtoms()
    distractor_offset = target_offset + motif.GetNumAtoms()
    molecule = Chem.RWMol(Chem.CombineMols(
        Chem.CombineMols(Chem.CombineMols(ring, anchor), motif), distractor))
    molecule.AddBond(0, anchor_offset, Chem.BondType.SINGLE)
    molecule.AddBond(ring_distance, target_offset, Chem.BondType.SINGLE)
    molecule.AddBond(distractor_ring_idx, distractor_offset, Chem.BondType.SINGLE)
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)
    endpoint = target_offset + _leaf_endpoint(motif, endpoint_kind)
    return molecule, {
        "ring_indices": list(range(6)), "anchor_ring_idx": 0,
        "target_ring_idx": ring_distance, "distractor_ring_idx": distractor_ring_idx,
        "anchor_attachment_idx": anchor_offset, "anchor_root_idx": anchor_offset + anchor_spacer,
        "target_attachment_idx": target_offset, "target_root_idx": target_offset,
        "path_endpoint_idx": endpoint,
        "target_fragment_indices": list(range(target_offset, distractor_offset)),
    }


def _graph_arrays(molecule: Chem.Mol, indices: dict[str, Any]) -> dict[str, np.ndarray]:
    n_atoms = molecule.GetNumAtoms()
    x = np.zeros((n_atoms, N_FEATURES), dtype=np.float32)
    for atom in molecule.GetAtoms():
        i, element = atom.GetIdx(), atom.GetSymbol()
        x[i, ELEMENTS.index(element) if element in ELEMENTS else len(ELEMENTS)] = 1.0
        x[i, 8:] = (
            float(atom.GetIsAromatic()), atom.GetFormalCharge() / 2.0,
            atom.GetDegree() / 4.0, atom.GetTotalNumHs() / 4.0,
            float(atom.IsInRing()), float(i == indices["anchor_ring_idx"]),
            float(i == indices["target_ring_idx"]),
            float(i == indices["target_attachment_idx"]),
            float(i == indices["anchor_root_idx"]),
            float(i == indices["path_endpoint_idx"]),
        )
    local_mask = np.zeros(n_atoms, dtype=np.float32)
    local_mask[indices["target_attachment_idx"]] = 1.0
    y_target_atom = np.zeros(n_atoms, dtype=np.float32)
    y_target_atom[indices["target_fragment_indices"]] = 1.0
    return {"x": x, "adj": Chem.GetAdjacencyMatrix(molecule).astype(np.float32),
            "local_mask": local_mask, "y_target_atom": y_target_atom}


def generate_hard_dataset(spacer_lengths: Iterable[int] = range(3)) -> list[Record]:
    """Enumerate 6,912 molecules by default, with one fixed query per identity.

    Eight rooted C7H15O2 ether motifs cross three ring positions, eight anchor
    families, three anchor spacers, three aromatic scaffolds, and four
    distractors. Endpoint kind is fixed by the canonical molecular identity;
    distractor placement depends on background factors only. No motif,
    position, or spacer labels enter atom features.
    """
    spacer_lengths = tuple(sorted(set(spacer_lengths)))
    if not spacer_lengths or any(not isinstance(n, int) or n < 0 for n in spacer_lengths):
        raise ValueError("Spacer lengths must be a nonempty collection of nonnegative integers.")
    identities: dict[str, Record] = {}
    for family_idx, (target_family, motif_smiles) in enumerate(MOTIF_FRAGMENTS):
        for position_idx, position in enumerate(POSITION_NAMES):
            for anchor_idx, (anchor_family, anchor_smiles) in enumerate(ANCHOR_FRAGMENTS):
                for anchor_spacer in spacer_lengths:
                    for scaffold_idx, (scaffold, scaffold_smiles) in enumerate(SCAFFOLD_FRAGMENTS):
                        for distractor_idx, (distractor_family, distractor_smiles) in enumerate(DISTRACTOR_FRAGMENTS):
                            molecule, indices = _construct_molecule(
                                motif_smiles, anchor_smiles, scaffold_smiles,
                                distractor_smiles, anchor_spacer, position_idx + 1,
                                anchor_idx + anchor_spacer + scaffold_idx, 0,
                            )
                            smiles = canonical_identity(Chem.MolToSmiles(molecule))
                            endpoint_kind = hashlib.sha256(smiles.encode("utf-8")).digest()[0] % 2
                            motif = Chem.MolFromSmiles(motif_smiles)
                            indices["path_endpoint_idx"] = indices["target_attachment_idx"] + _leaf_endpoint(motif, endpoint_kind)
                            distance = len(Chem.GetShortestPath(
                                molecule, indices["anchor_root_idx"], indices["path_endpoint_idx"]
                            )) - 1
                            if distance >= N_DISTANCE_CLASSES:
                                raise ValueError("Configured spacer exceeds the distance-class vocabulary.")
                            mapped = Chem.Mol(molecule)
                            for atom in mapped.GetAtoms():
                                atom.SetAtomMapNum(atom.GetIdx() + 1)
                            record = {
                                "id": hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20],
                                "smiles": smiles, "atom_map_reference_smiles": Chem.MolToSmiles(mapped),
                                "target_family": target_family, "motif_smiles": motif_smiles,
                                "anchor_family": anchor_family, "anchor_spacer": anchor_spacer,
                                "target_spacer": 0, "scaffold": scaffold,
                                "distractor_family": distractor_family, "endpoint_kind": endpoint_kind,
                                "position": position, "y_position": position_idx,
                                "y_family": family_idx, "y_motif": family_idx, "y_distance": distance,
                                **indices, **_graph_arrays(molecule, indices),
                            }
                            if smiles in identities:
                                raise ValueError(f"Duplicate hard-benchmark canonical molecule: {smiles}")
                            identities[smiles] = record
    return [identities[smiles] for smiles in sorted(identities)]


def split_hard_dataset(records: list[Record], seed: int = 20261030) -> dict[str, list[int]]:
    """Partition identity before checkerboard exclusions: 60/10/10/20 percent.

    The partitions are candidate pretraining, validation, adaptation reserve,
    and testing. A trains only on cell class 0 and B only on class 1; their
    validation and IID tests use the respective familiar cell classes. Cell
    class 2 supplies a common OOD test and adaptation support from the separate
    reserve, unseen by both pretrained experiences. For each motif, the two
    training experiences are matched to their smaller available count, choosing
    canonical identity hashes in deterministic order. Remaining candidates are
    retained in the base partition but unused during training.
    Probe source/target are untouched test ortho/meta identities respectively.
    """
    canonical = [canonical_identity(record["smiles"]) for record in records]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Canonical molecular identities must be deduplicated before splitting.")
    splits = {name: [] for name in (
        "candidate_train", "train_a", "train_b", "val", "val_a", "val_b", "discarded_val_ood",
        "adaptation_reserve", "adaptation_pool", "adaptation_test",
        "analysis", "iid_a", "iid_b", "ood_common", "probe_source", "probe_target",
    )}
    for i, (record, smiles) in enumerate(zip(records, canonical)):
        score, cell = _partition_score(smiles, seed), heldout_cell(record)
        if score < 0.6:
            splits["candidate_train"].append(i)
            if cell == 0:
                splits["train_a"].append(i)
            if cell == 1:
                splits["train_b"].append(i)
        elif score < 0.7:
            if cell == 2:
                splits["discarded_val_ood"].append(i)
            else:
                splits["val"].append(i)
                splits["val_a" if cell == 0 else "val_b"].append(i)
        elif score < 0.8:
            splits["adaptation_reserve"].append(i)
            if cell == 2:
                splits["adaptation_pool"].append(i)
        else:
            splits["analysis"].append(i)
            splits[("iid_a", "iid_b", "ood_common")[cell]].append(i)
            if cell == 2:
                splits["adaptation_test"].append(i)
            if record["position"] == "ortho":
                splits["probe_source"].append(i)
            elif record["position"] == "meta":
                splits["probe_target"].append(i)
    splits.update(_matched_training_indices(records, splits["candidate_train"]))
    audit_hard_splits(records, splits)
    return splits


def audit_hard_splits(records: list[Record], splits: dict[str, list[int]]) -> dict[str, Any]:
    """Validate molecular identities, compositional cells, and adaptation isolation."""
    indexed = {name: set(indices) for name, indices in splits.items()}
    for name, indices in splits.items():
        if len(indices) != len(indexed[name]):
            raise ValueError(f"Repeated record index in split {name!r}.")
        if any(i < 0 or i >= len(records) for i in indices):
            raise ValueError(f"Out-of-bounds record index in split {name!r}.")
    canonical = [canonical_identity(record["smiles"]) for record in records]
    if len(set(canonical)) != len(canonical):
        raise ValueError("The dataset contains repeated canonical molecular identities.")
    partitions = [indexed["candidate_train"], indexed["val"] | indexed["discarded_val_ood"],
                  indexed["adaptation_reserve"], indexed["analysis"]]
    if any(a & b for j, a in enumerate(partitions) for b in partitions[j + 1:]):
        raise ValueError("Base identity partitions overlap.")
    if set().union(*partitions) != set(range(len(records))):
        raise ValueError("The identity partitions do not cover the full dataset.")
    matched = _matched_training_indices(records, splits["candidate_train"])
    for train_name in ("train_a", "train_b"):
        if indexed[train_name] != set(matched[train_name]):
            raise ValueError(f"Incorrect matched checkerboard membership in {train_name!r}.")
        if indexed[train_name] & indexed["adaptation_reserve"]:
            raise ValueError("Adaptation support leaked into pretraining.")
    for name, cell, partition in (("val_a", 0, partitions[1]), ("val_b", 1, partitions[1]),
                                  ("iid_a", 0, indexed["analysis"]),
                                  ("iid_b", 1, indexed["analysis"]),
                                  ("ood_common", 2, indexed["analysis"]),
                                  ("adaptation_pool", 2, indexed["adaptation_reserve"])):
        expected = {i for i in partition if heldout_cell(records[i]) == cell}
        if indexed[name] != expected:
            raise ValueError(f"Incorrect chemistry membership in {name!r}.")
    if indexed["val"] != indexed["val_a"] | indexed["val_b"]:
        raise ValueError("Validation must be the union of experience-specific familiar cells.")
    if indexed["adaptation_test"] != indexed["ood_common"]:
        raise ValueError("Adaptation test must alias the common held-out test.")
    for name, position in (("probe_source", "ortho"), ("probe_target", "meta")):
        expected = {i for i in indexed["analysis"] if records[i]["position"] == position}
        if indexed[name] != expected:
            raise ValueError(f"Incorrect membership in {name!r}.")
    return {
        "n_molecules": len(records), "unique_canonical_identities": len(set(canonical)),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "effective_train_counts": {name: len(indexed[name]) for name in matched},
        "train_motif_counts": {
            name: dict(Counter(records[i]["target_family"] for i in splits[name])) for name in matched
        },
        "train_position_counts": {
            name: dict(Counter(records[i]["position"] for i in splits[name])) for name in matched
        },
        "discarded_train_counts": {
            name: sum(heldout_cell(records[i]) == cell for i in indexed["candidate_train"])
            - len(indexed[name]) for name, cell in (("train_a", 0), ("train_b", 1))
        },
        "position_counts": dict(Counter(r["position"] for r in records)),
        "target_family_counts": dict(Counter(r["target_family"] for r in records)),
        "scaffold_counts": dict(Counter(r["scaffold"] for r in records)),
        "motif_position_cell_counts": dict(Counter(f"{r['target_family']}:{r['position']}" for r in records)),
        "feature_names": list(FEATURE_NAMES), "feature_dimension": N_FEATURES,
        "distance_range": [min(r["y_distance"] for r in records), max(r["y_distance"] for r in records)],
        "query_endpoints_explicitly_marked": True, "local_readout_atoms": 1,
        "target_fragment_mask_used_as_input": False,
        "adaptation_support_unseen_by_both_experiences": True, "split_identity_leakage": False,
    }
