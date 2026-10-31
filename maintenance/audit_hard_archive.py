"""Independently audit a frozen hard molecular archive using CPU chemistry only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
ELEMENTS = ("C", "N", "O", "F", "S", "Cl", "Br")
POSITIONS = ("ortho", "meta", "para")
MOTIFS = (
    "CCOC(CCC)OC", "CCOC(CC)OCC", "CCOC(CC)COC", "COCC(CCC)OC",
    "COCC(CC)OCC", "COCC(CC)COC", "CCCC(OC)OCC", "CCCC(COC)OC",
)
ANCHORS = {
    "amide": "C(=O)N", "carboxyl": "C(=O)O", "methyl_ester": "C(=O)OC",
    "acetyl": "C(=O)C", "methylsulfonyl": "S(=O)(=O)C",
    "sulfonamide": "S(=O)(=O)N", "trifluoromethyl": "C(F)(F)F", "isopropyl": "C(C)C",
}
DISTRACTORS = {"methyl": "C", "fluoro": "F", "hydroxy": "O", "cyano": "C#N"}
FEATURE_NAMES = tuple(f"element_{element}" for element in ELEMENTS) + (
    "element_other", "aromatic", "formal_charge_div_2", "degree_div_4",
    "hydrogen_count_div_4", "in_ring", "query_anchor_ring_atom",
    "query_target_ring_atom", "query_target_attachment_atom",
    "query_anchor_functional_root", "query_path_endpoint",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def identity(molecule: Chem.Mol) -> str:
    clean = Chem.Mol(molecule)
    for atom in clean.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(clean, canonical=True, isomericSmiles=True)


def mapped_molecule(smiles: str) -> Chem.Mol:
    """Restore original array order using the complete atom-map bijection."""
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError("Mapped molecular reference could not be parsed.")
    index_by_map = {a.GetAtomMapNum() - 1: a.GetIdx() for a in parsed.GetAtoms()}
    if set(index_by_map) != set(range(parsed.GetNumAtoms())):
        raise ValueError("Atom maps are not a bijection over the original graph indices.")
    return Chem.RenumberAtoms(parsed, [index_by_map[i] for i in range(parsed.GetNumAtoms())])


def bfs_distances(adjacency: np.ndarray, start: int, blocked_edge=None) -> dict[int, int]:
    distances, queue = {start: 0}, deque([start])
    while queue:
        atom = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[atom]):
            neighbor = int(neighbor)
            if blocked_edge is not None and frozenset((atom, neighbor)) == frozenset(blocked_edge):
                continue
            if neighbor not in distances:
                distances[neighbor] = distances[atom] + 1
                queue.append(neighbor)
    return distances


def rooted_component_identity(molecule: Chem.Mol, indices, root: int) -> str:
    """Canonicalize an induced component with a dummy attachment at its root.

    Rebuilding atoms avoids retaining explicit hydrogen counts from mapped
    SMILES after cutting a bond. The dummy preserves the original attachment
    valence and identifies the rooted chemical topology without atom ordering.
    """
    indices = sorted(indices)
    old_to_new = {old: new for new, old in enumerate(indices)}
    result = Chem.RWMol()
    for index in indices:
        original = molecule.GetAtomWithIdx(index)
        atom = Chem.Atom(original.GetAtomicNum())
        atom.SetFormalCharge(original.GetFormalCharge())
        atom.SetIsAromatic(original.GetIsAromatic())
        result.AddAtom(atom)
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in old_to_new and b in old_to_new:
            result.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())
    dummy = result.AddAtom(Chem.Atom(0))
    result.AddBond(old_to_new[root], dummy, Chem.BondType.SINGLE)
    result = result.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, canonical=True, isomericSmiles=True)


def reference_identity(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return rooted_component_identity(molecule, range(molecule.GetNumAtoms()), 0)


def ordinary_features(molecule: Chem.Mol) -> np.ndarray:
    x = np.zeros((molecule.GetNumAtoms(), 13), dtype=np.float32)
    for atom in molecule.GetAtoms():
        i, symbol = atom.GetIdx(), atom.GetSymbol()
        x[i, ELEMENTS.index(symbol) if symbol in ELEMENTS else 7] = 1.0
        x[i, 8:] = (float(atom.GetIsAromatic()), atom.GetFormalCharge() / 2,
                    atom.GetDegree() / 4, atom.GetTotalNumHs() / 4, float(atom.IsInRing()))
    return x


def archive_path(root: Path, relative: str) -> Path:
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Archived path leaves its declared root: {relative}")
    return path


def audit_record(record_dir: Path, repository_root: Path = ROOT) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    record_dir, repository_root = record_dir.resolve(), repository_root.resolve()
    source_digest = digest(Path(__file__))
    manifest = json.loads((record_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    input_names = sorted(set(manifest["sha256"]) | {"MANIFEST.json"})
    input_hashes = {name: digest(archive_path(record_dir, name)) for name in input_names}
    records = [json.loads(line) for line in (record_dir / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((record_dir / "splits.json").read_text(encoding="utf-8"))
    config = json.loads((record_dir / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((record_dir / "environment.json").read_text(encoding="utf-8"))
    recorded_audit = json.loads((record_dir / "audit.json").read_text(encoding="utf-8"))
    with np.load(record_dir / "graphs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    checks, failures = {}, []

    def check(name, condition, molecule_id=None):
        counts = checks.setdefault(name, {"checked": 0, "passed": 0, "failed": 0})
        counts["checked"] += 1
        result = bool(condition)
        counts["passed" if result else "failed"] += 1
        if not result and len(failures) < 40:
            failures.append({"check": name, "molecule_id": molecule_id})

    for name, expected in manifest["sha256"].items():
        check("every_frozen_manifest_file_matches_sha256", input_hashes[name] == expected, name)
    count = len(records)
    check("full_factorial_molecule_count", count == 6912)
    check("unique_molecular_ids", len({r["id"] for r in records}) == count)
    check("atom_feature_shape", arrays["x"].shape == (count, 23, 18))
    check("adjacency_shape", arrays["adj"].shape == (count, 23, 23))
    check("padding_mask_shape", arrays["mask"].shape == (count, 23))
    for name in ("y_family", "y_position", "y_distance"):
        check("label_array_shapes", arrays[name].shape == (count,))
    check("all_arrays_finite", all(np.isfinite(array).all() for array in arrays.values()))
    references = [reference_identity(smiles) for smiles in MOTIFS]
    check("eight_distinct_rooted_topology_references", len(set(references)) == 8)
    canonical_identities, atom_counts, root_features, motif_bags, radii, distances = [], [], set(), set(), [], []
    formula_counts = Counter()

    for row, record in enumerate(records):
        record_id = record["id"]
        try:
            molecule = mapped_molecule(record["atom_map_reference_smiles"])
        except ValueError:
            check("mapped_reference_and_bijection", False, record_id)
            continue
        check("mapped_reference_and_bijection", True, record_id)
        n_atoms = molecule.GetNumAtoms()
        atom_counts.append(n_atoms)
        canonical = identity(molecule)
        canonical_identities.append(canonical)
        check("canonical_identity_and_sha256_id", canonical == record["smiles"] and
              hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20] == record_id, record_id)
        expected_adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        for bond in molecule.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            expected_adj[a, b] = expected_adj[b, a] = 1
        check("all_adjacency_entries_from_mapped_bonds",
              np.array_equal(expected_adj, arrays["adj"][row, :n_atoms, :n_atoms]), record_id)
        check("zero_adjacency_and_feature_padding", not arrays["adj"][row, n_atoms:, :].any()
              and not arrays["adj"][row, :, n_atoms:].any() and not arrays["x"][row, n_atoms:, :].any(), record_id)
        check("atom_count_padding_mask", np.array_equal(arrays["mask"][row],
              np.arange(arrays["mask"].shape[1]) < n_atoms), record_id)
        ordinary = ordinary_features(molecule)
        check("all_13_ordinary_features_from_chemistry",
              np.array_equal(ordinary, arrays["x"][row, :n_atoms, :13]), record_id)
        query_indices = [record[key] for key in ("anchor_ring_idx", "target_ring_idx",
                         "target_attachment_idx", "anchor_root_idx", "path_endpoint_idx")]
        expected_queries = np.zeros((n_atoms, 5), dtype=np.float32)
        expected_queries[query_indices, np.arange(5)] = 1
        check("all_five_query_markers_from_recorded_indices",
              np.array_equal(expected_queries, arrays["x"][row, :n_atoms, 13:]), record_id)
        check("local_readout_is_single_attachment_atom", arrays["x"][row, :, 15].sum() == 1
              and arrays["x"][row, record["target_attachment_idx"], 15] == 1, record_id)
        root_features.add(tuple(arrays["x"][row, record["target_attachment_idx"]]))
        ring = set(record["ring_indices"])
        check("ring_indices_match_six_aromatic_ring_atoms", len(ring) == 6 and ring == {
            a.GetIdx() for a in molecule.GetAtoms() if a.GetIsAromatic() and a.IsInRing()}, record_id)
        expected_n = {"benzene": set(), "pyridine_n4": {4}, "pyridine_n5": {5}}[record["scaffold"]]
        check("scaffold_label_from_ring_nitrogen_positions", expected_n == {
            i for i in ring if molecule.GetAtomWithIdx(i).GetSymbol() == "N"}, record_id)
        target_root, target_ring = record["target_attachment_idx"], record["target_ring_idx"]
        check("target_attachment_bond_exists", expected_adj[target_root, target_ring] == 1, record_id)
        component_dist = bfs_distances(expected_adj, target_root, (target_root, target_ring))
        target = set(component_dist)
        check("target_membership_equals_cut_component", target == set(record["target_fragment_indices"])
              and len(target) == 9, record_id)
        inferred = rooted_component_identity(molecule, target, target_root)
        inferred_family = references.index(inferred) if inferred in references else -1
        check("motif_label_from_independent_rooted_topology", inferred_family == record["y_family"]
              == record["y_motif"] == int(arrays["y_family"][row]) and
              record["target_family"] == f"motif_{inferred_family}", record_id)
        elements = Counter(molecule.GetAtomWithIdx(i).GetSymbol() for i in target)
        hydrogens = sum(molecule.GetAtomWithIdx(i).GetTotalNumHs() for i in target)
        formula = f"C{elements['C']}H{hydrogens}O{elements['O']}"
        formula_counts[formula] += 1
        check("attached_target_formula_C7H15O2", elements == {"C": 7, "O": 2} and hydrogens == 15, record_id)
        motif_bags.add(tuple(sorted(Counter(tuple(ordinary[i]) for i in target).items())))
        radii.append(max(component_dist.values()))
        check("target_motif_fits_radius_six", max(component_dist.values()) <= 6, record_id)
        leaves = sorted((i for i in target if i != target_root and expected_adj[i].sum() == 1
                         and molecule.GetAtomWithIdx(i).GetSymbol() == "C"),
                        key=lambda i: (component_dist[i], i))
        endpoint_kind = hashlib.sha256(canonical.encode("utf-8")).digest()[0] % 2
        expected_endpoint = leaves[0] if endpoint_kind == 0 else leaves[-1]
        check("path_endpoint_determined_by_identity_hash", record["endpoint_kind"] == endpoint_kind
              and record["path_endpoint_idx"] == expected_endpoint, record_id)
        path_distance = bfs_distances(expected_adj, record["anchor_root_idx"])[record["path_endpoint_idx"]]
        distances.append(path_distance)
        check("distance_label_from_independent_BFS", path_distance == record["y_distance"]
              == int(arrays["y_distance"][row]), record_id)
        ring_distance = bfs_distances(expected_adj, record["anchor_ring_idx"])[target_ring]
        check("position_label_from_independent_BFS", ring_distance - 1 == record["y_position"]
              == int(arrays["y_position"][row]) and record["position"] == POSITIONS[ring_distance - 1], record_id)
        anchor_root, anchor_ring = record["anchor_attachment_idx"], record["anchor_ring_idx"]
        anchor_component = set(bfs_distances(expected_adj, anchor_root, (anchor_root, anchor_ring)))
        anchor_expected = reference_identity("C" * record["anchor_spacer"] + ANCHORS[record["anchor_family"]])
        check("anchor_family_from_rooted_component", rooted_component_identity(molecule, anchor_component, anchor_root)
              == anchor_expected, record_id)
        check("anchor_spacer_from_query_depth", bfs_distances(expected_adj, anchor_ring)[record["anchor_root_idx"]]
              == record["anchor_spacer"] + 1 and record["target_spacer"] == 0 and record["target_root_idx"] == target_root, record_id)
        distractor_ring = record["distractor_ring_idx"]
        external = [i for i in np.flatnonzero(expected_adj[distractor_ring]) if int(i) not in ring]
        distractor_root = int(external[0]) if len(external) == 1 else -1
        if distractor_root >= 0:
            distractor_component = set(bfs_distances(expected_adj, distractor_root, (distractor_root, distractor_ring)))
            observed = rooted_component_identity(molecule, distractor_component, distractor_root)
        else:
            observed = None
        check("distractor_label_from_rooted_component", observed == reference_identity(DISTRACTORS[record["distractor_family"]]), record_id)

    check("reconstructed_canonical_identities_unique", len(set(canonical_identities)) == count)
    check("single_raw_local_root_feature_vector", len(root_features) == 1)
    check("identical_target_ordinary_feature_multisets", len(motif_bags) == 1)
    check("observed_maximum_motif_radius_six", max(radii) == 6)
    check("observed_distance_range_eight_to_thirteen", [min(distances), max(distances)] == [8, 13])
    check("maximum_atom_count_matches_padding", max(atom_counts) == arrays["x"].shape[1] == 23)
    check("balanced_full_factorial_motif_position_cells", set(Counter(
        (r["y_family"], r["y_position"]) for r in records).values()) == {288})

    sets = {name: set(indices) for name, indices in splits.items()}
    for name, indices in splits.items():
        check("split_indices_unique_and_in_bounds", len(indices) == len(sets[name])
              and all(isinstance(i, int) and 0 <= i < count for i in indices), name)
    expected_partition = [set() for _ in range(4)]
    for i, record in enumerate(records):
        score = int.from_bytes(hashlib.sha256(f"{config['split_seed']}|{record['smiles']}".encode("utf-8")).digest()[:8], "big") / 2**64
        expected_partition[0 if score < 0.6 else 1 if score < 0.7 else 2 if score < 0.8 else 3].add(i)
    candidate, validation, reserve, test = expected_partition
    check("global_identity_partitions_match_hash_thresholds", candidate == sets["candidate_train"]
          and validation == sets["val"] | sets["discarded_val_ood"]
          and reserve == sets["adaptation_reserve"] and test == sets["analysis"])
    cells = [{i for i, r in enumerate(records) if (r["y_family"] + r["y_position"]) % 3 == cell} for cell in range(3)]
    for name, expected in (("val_a", validation & cells[0]), ("val_b", validation & cells[1]),
                           ("discarded_val_ood", validation & cells[2]),
                           ("iid_a", test & cells[0]), ("iid_b", test & cells[1]),
                           ("ood_common", test & cells[2]), ("adaptation_pool", reserve & cells[2]),
                           ("adaptation_test", test & cells[2])):
        check("experience_and_common_holdout_cells_exact", sets[name] == expected, name)
    check("validation_is_experience_specific_union", sets["val"] == sets["val_a"] | sets["val_b"])
    matched, discarded = {"train_a": set(), "train_b": set()}, Counter()
    for family in range(8):
        available = [sorted((i for i in candidate & cells[cell] if records[i]["y_family"] == family),
                            key=lambda i: records[i]["id"]) for cell in (0, 1)]
        keep = min(map(len, available))
        for name, indices in zip(("train_a", "train_b"), available):
            matched[name].update(indices[:keep])
            discarded[name] += len(indices) - keep
    for name in matched:
        check("exact_per_motif_matched_lowest_ID_training_selection", sets[name] == matched[name], name)
    check("training_experiences_have_equal_motif_counts", Counter(records[i]["y_family"] for i in sets["train_a"])
          == Counter(records[i]["y_family"] for i in sets["train_b"]))
    training = sets["train_a"] | sets["train_b"]
    check("adaptation_support_and_all_evaluation_unseen_during_both_pretrainings", not training & (validation | reserve | test))
    check("support_and_adaptation_test_disjoint", not sets["adaptation_pool"] & sets["adaptation_test"])
    for name, position in (("probe_source", "ortho"), ("probe_target", "meta")):
        check("probe_membership_uses_disjoint_heldout_positions", sets[name] == {
            i for i in test if records[i]["position"] == position}, name)
    support_ids = {records[i]["id"] for i in sets["adaptation_pool"]}
    support_count = 0
    for repeat in range(config["adaptation_repeats"]):
        ordering = np.random.default_rng(6000 + repeat).permutation(splits["adaptation_pool"])
        previous = []
        for shots in sorted(config["adaptation_shots"]):
            support_path = record_dir / "adaptation_support" / f"repeat{repeat}_shots{shots}.json"
            support = json.loads(support_path.read_text(encoding="utf-8"))
            ids = support["molecule_ids"]
            expected_ids = [records[i]["id"] for i in ordering[:shots]]
            check("adaptation_support_exact_repeat_permutation_and_size", ids == expected_ids
                  and len(ids) == len(set(ids)) == shots and set(ids) <= support_ids, support_path.name)
            check("adaptation_support_nested_across_shot_counts", ids[:len(previous)] == previous, support_path.name)
            previous = ids
            support_count += 1
    check("recorded_audit_counts_match_independent_results", recorded_audit["split_counts"] == {
        name: len(indices) for name, indices in splits.items()} and recorded_audit["feature_names"] == list(FEATURE_NAMES)
          and recorded_audit["distance_range"] == [min(distances), max(distances)]
          and recorded_audit["local_readout_atoms"] == 1 and recorded_audit["target_fragment_mask_used_as_input"] is False)
    source_hashes = {}
    for relative, expected in environment["source_sha256"].items():
        observed = digest(archive_path(repository_root, relative))
        source_hashes[relative] = {"expected_at_run_start": expected, "observed_during_audit": observed}
        check("source_hash_matches_run_start_snapshot", observed == expected, relative)
    check("all_archive_inputs_unchanged_during_audit", input_hashes == {
        name: digest(archive_path(record_dir, name)) for name in input_names})
    check("auditor_source_unchanged_during_audit", source_digest == digest(Path(__file__)))
    failed = sum(value["failed"] for value in checks.values())
    return {
        "audit_name": "Independent hard molecular archive audit", "status": "PASS" if failed == 0 else "FAIL",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "archive_directory": str(record_dir), "training_source_commit_at_start": environment["git_head_at_start"],
        "scope": "All archived molecules, graph entries, query and class labels, identity partitions, support selections, source hashes, and frozen manifest entries; no generator or model imported; no training or GPU use.",
        "method": "Reorder mapped SMILES by atom map, reconstruct chemistry and binary graph arrays, infer rooted topology with an external dummy attachment, and recompute paths and partitions independently.",
        "audit_implementation_sha256": source_digest,
        "independent_dependencies": {"python": sys.version, "numpy": importlib.metadata.version("numpy"),
                                     "rdkit": importlib.metadata.version("rdkit"), "platform": platform.platform()},
        "counts": {"molecules_audited": count, "atoms_audited": sum(atom_counts),
                   "adjacency_entries_audited": sum(n * n for n in atom_counts),
                   "ordinary_feature_entries_audited": sum(atom_counts) * 13,
                   "query_feature_entries_audited": sum(atom_counts) * 5,
                   "unique_canonical_identities": len(set(canonical_identities)),
                   "maximum_atoms": max(atom_counts), "maximum_motif_root_radius": max(radii),
                   "support_files_audited": support_count, "check_categories": len(checks),
                   "individual_checks": sum(v["checked"] for v in checks.values()), "failed_checks": failed},
        "checks": checks, "failures_first_40": failures,
        "archived_input_sha256": input_hashes, "training_source_sha256_verification": source_hashes,
        "array_shapes_and_dtypes": {name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                                    for name, array in arrays.items()},
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "training_matching": {"effective_train_counts": {name: len(indices) for name, indices in matched.items()},
                              "discarded_train_counts": dict(discarded)},
        "target_formula_counts": dict(formula_counts), "rooted_motif_reference_identities": references,
        "raw_local_root_feature_vectors": len(root_features), "target_ordinary_feature_bags": len(motif_bags),
        "distance_range": [min(distances), max(distances)], "frozen_record_modified": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit: {args.output}")
    report = audit_record(args.record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "output": str(args.output), "failures": report["failures_first_40"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
