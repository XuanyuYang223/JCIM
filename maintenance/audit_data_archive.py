"""Independent audit of archived graph arrays using mapped molecular references."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(r'F:\JCIM')
ARCHIVE = ROOT / 'experiments' / 'pilot_v1'
OUTPUT = ROOT / 'maintenance' / 'DATA_ARCHIVE_AUDIT.json'
INPUT_NAMES = ('molecules.jsonl', 'graphs.npz', 'splits.json', 'config.json', 'audit.json')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


started = datetime.now(timezone.utc).isoformat()
input_hashes = {name: digest(ARCHIVE / name) for name in INPUT_NAMES}
records = [json.loads(line) for line in (ARCHIVE / 'molecules.jsonl').read_text(encoding='utf-8').splitlines()]
splits = json.loads((ARCHIVE / 'splits.json').read_text(encoding='utf-8'))
config = json.loads((ARCHIVE / 'config.json').read_text(encoding='utf-8'))
environment = json.loads((ARCHIVE / 'environment.json').read_text(encoding='utf-8'))
arrays = dict(np.load(ARCHIVE / 'graphs.npz', allow_pickle=False))
checks = {}
failures = []


def check(name, condition, molecule_id=None):
    counter = checks.setdefault(name, {'checked': 0, 'passed': 0, 'failed': 0})
    counter['checked'] += 1
    result = bool(condition)
    counter['passed' if result else 'failed'] += 1
    if not result and len(failures) < 30:
        failures.append({'check': name, 'molecule_id': molecule_id})


def identity(molecule):
    clean = Chem.Mol(molecule)
    for atom in clean.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(clean, canonical=True, isomericSmiles=True)


n_records = len(records)
check('expected_full_factorial_molecule_count', n_records == 3840)
check('unique_record_ids', len({record['id'] for record in records}) == n_records)
check('unique_canonical_smiles', len({record['smiles'] for record in records}) == n_records)
check('atom_feature_array_shape', arrays['x'].shape == (n_records, 20, 18))
check('adjacency_array_shape', arrays['adj'].shape == (n_records, 20, 20))
check('atom_mask_array_shape', arrays['mask'].shape == (n_records, 20))
for name in ('y_family', 'y_position', 'y_distance'):
    check('label_array_shapes', arrays[name].shape == (n_records,))
check('all_archived_arrays_finite', all(np.isfinite(value).all() for value in arrays.values()))

ELEMENTS = ('C', 'N', 'O', 'F', 'S', 'Cl', 'Br')
FAMILIES = ('fluoro', 'chloro', 'bromo', 'hydroxy', 'amino', 'thiol', 'cyano', 'nitro')
POSITIONS = ('ortho', 'meta', 'para')
atom_counts = []
canonical_identities = []

for row, record in enumerate(records):
    record_id = record['id']
    parsed = Chem.MolFromSmiles(record['atom_map_reference_smiles'])
    check('mapped_smiles_parse', parsed is not None, record_id)
    if parsed is None:
        continue
    n_atoms = parsed.GetNumAtoms()
    atom_counts.append(n_atoms)
    map_numbers = [atom.GetAtomMapNum() for atom in parsed.GetAtoms()]
    map_is_bijection = sorted(map_numbers) == list(range(1, n_atoms + 1))
    check('atom_map_bijection_one_through_atom_count', map_is_bijection, record_id)
    if not map_is_bijection:
        continue
    original_to_parsed = {atom.GetAtomMapNum() - 1: atom.GetIdx() for atom in parsed.GetAtoms()}
    molecule = Chem.RenumberAtoms(parsed, [original_to_parsed[i] for i in range(n_atoms)])
    canonical = identity(molecule)
    canonical_identities.append(canonical)
    check('mapped_reference_canonical_identity', canonical == record['smiles'], record_id)
    check('canonical_smiles_id_sha256_prefix', hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20] == record_id, record_id)

    expected_adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        expected_adj[a, b] = expected_adj[b, a] = 1.0
    check('every_binary_adjacency_entry_from_mapped_bonds', np.array_equal(expected_adj, arrays['adj'][row, :n_atoms, :n_atoms]), record_id)
    check('zero_adjacency_padding_rows_and_columns', not arrays['adj'][row, n_atoms:, :].any() and not arrays['adj'][row, :, n_atoms:].any(), record_id)
    expected_mask = np.zeros(arrays['mask'].shape[1], dtype=np.float32)
    expected_mask[:n_atoms] = 1.0
    check('atom_count_and_padding_mask', np.array_equal(expected_mask, arrays['mask'][row]), record_id)
    check('zero_atom_feature_padding', not arrays['x'][row, n_atoms:, :].any(), record_id)

    ordinary_features = np.zeros((n_atoms, 13), dtype=np.float32)
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        symbol = atom.GetSymbol()
        ordinary_features[i, ELEMENTS.index(symbol) if symbol in ELEMENTS else 7] = 1.0
        ordinary_features[i, 8:] = (float(atom.GetIsAromatic()), atom.GetFormalCharge() / 2.0,
                                     atom.GetDegree() / 4.0, atom.GetTotalNumHs() / 4.0, float(atom.IsInRing()))
    check('all_13_ordinary_atom_features_from_mapped_chemistry', np.array_equal(ordinary_features, arrays['x'][row, :n_atoms, :13]), record_id)

    expected_queries = np.zeros((n_atoms, 5), dtype=np.float32)
    expected_queries[record['anchor_ring_idx'], 0] = 1.0
    expected_queries[record['target_ring_idx'], 1] = 1.0
    expected_queries[record['target_fragment_indices'], 2] = 1.0
    expected_queries[record['anchor_root_idx'], 3] = 1.0
    expected_queries[record['target_root_idx'], 4] = 1.0
    check('all_5_query_features_match_recorded_atom_indices', np.array_equal(expected_queries, arrays['x'][row, :n_atoms, 13:18]), record_id)
    check('four_endpoint_markers_unique', np.array_equal(expected_queries[:, [0, 1, 3, 4]].sum(0), np.ones(4)), record_id)
    check('recorded_ring_atoms_match_aromatic_ring', set(record['ring_indices']) == {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetIsAromatic() and atom.IsInRing()}, record_id)

    target_attachment = record['target_attachment_idx']
    target_ring = record['target_ring_idx']
    check('target_attachment_is_recorded_ring_bond', molecule.GetBondBetweenAtoms(target_ring, target_attachment) is not None, record_id)
    visited, frontier = {target_attachment}, {target_attachment}
    while frontier:
        next_frontier = set()
        for atom_index in frontier:
            for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if {atom_index, neighbor_index} == {target_ring, target_attachment}:
                    continue
                if neighbor_index not in visited:
                    next_frontier.add(neighbor_index)
        visited.update(next_frontier)
        frontier = next_frontier
    check('fragment_membership_is_component_after_cutting_target_ring_bond', visited == set(record['target_fragment_indices']), record_id)

    root_distance = len(Chem.GetShortestPath(molecule, record['anchor_root_idx'], record['target_root_idx'])) - 1
    ring_distance = len(Chem.GetShortestPath(molecule, record['anchor_ring_idx'], record['target_ring_idx'])) - 1
    check('distance_label_from_independent_graph_shortest_path', root_distance == record['y_distance'] == int(arrays['y_distance'][row]), record_id)
    check('position_label_from_independent_ring_shortest_path', ring_distance - 1 == record['y_position'] == int(arrays['y_position'][row]) and record['position'] == POSITIONS[ring_distance - 1], record_id)
    anchor_depth = len(Chem.GetShortestPath(molecule, record['anchor_ring_idx'], record['anchor_root_idx'])) - 1
    target_depth = len(Chem.GetShortestPath(molecule, record['target_ring_idx'], record['target_root_idx'])) - 1
    check('spacer_lengths_match_both_queried_functional_root_depths', anchor_depth == record['anchor_spacer'] + 1 and target_depth == record['target_spacer'] + 1, record_id)
    check('path_label_equals_ring_plus_both_spacer_depths', root_distance == ring_distance + anchor_depth + target_depth, record_id)

    root = molecule.GetAtomWithIdx(record['target_root_idx'])
    symbol = root.GetSymbol()
    inferred_family = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'O': 'hydroxy', 'S': 'thiol'}.get(symbol)
    if symbol == 'N':
        if root.GetFormalCharge() == 0:
            inferred_family = 'amino'
        elif root.GetFormalCharge() == 1 and sum(neighbor.GetSymbol() == 'O' for neighbor in root.GetNeighbors()) == 2:
            inferred_family = 'nitro'
    elif symbol == 'C' and any(bond.GetBondType() == Chem.BondType.TRIPLE and bond.GetOtherAtom(root).GetSymbol() == 'N' for bond in root.GetBonds()):
        inferred_family = 'cyano'
    check('target_family_label_from_functional_root_chemistry', inferred_family == record['target_family'] and FAMILIES.index(inferred_family) == record['y_family'] == int(arrays['y_family'][row]), record_id)

check('reconstructed_canonical_identities_all_unique', len(set(canonical_identities)) == n_records)
check('balanced_full_factorial_family_position_cells', set(Counter((record['target_family'], record['position']) for record in records).values()) == {160})
check('maximum_atom_count_matches_padding_width', max(atom_counts) == arrays['x'].shape[1] == 20)

sets = {name: set(indices) for name, indices in splits.items()}
for name, indices in splits.items():
    check('split_lists_contain_unique_in_bounds_indices', len(indices) == len(sets[name]) and all(0 <= i < n_records for i in indices))

candidate, validation, test = sets['candidate_train'], sets['val'] | sets['discarded_val_ood'], sets['analysis']
check('global_partition_full_coverage', candidate | validation | test == set(range(n_records)))
check('global_partition_pairwise_disjoint', not candidate & validation and not candidate & test and not validation & test)
for index, record in enumerate(records):
    score = int.from_bytes(hashlib.sha256(f"{config['split_seed']}|{record['smiles']}".encode('utf-8')).digest()[:8], 'big') / 2**64
    expected_partition = candidate if score < 0.65 else validation if score < 0.75 else test
    check('global_partition_matches_independent_identity_hash', index in expected_partition, record['id'])

f_meta = {i for i, record in enumerate(records) if record['target_family'] == 'fluoro' and record['position'] == 'meta'}
cl_meta = {i for i, record in enumerate(records) if record['target_family'] == 'chloro' and record['position'] == 'meta'}
check('experience_a_excludes_all_fluoro_meta', not sets['train_a'] & f_meta)
check('experience_b_excludes_all_chloro_meta', not sets['train_b'] & cl_meta)
check('both_pretraining_sets_exclude_all_validation_and_test', not (sets['train_a'] | sets['train_b']) & (validation | test))
check('validation_excludes_both_combination_holdouts', not sets['val'] & (f_meta | cl_meta))
check('ood_fluoro_meta_test_exactly_matches_global_test_cell', sets['ood_f_meta'] == test & f_meta)
check('ood_chloro_meta_test_exactly_matches_global_test_cell', sets['ood_cl_meta'] == test & cl_meta)
check('iid_test_excludes_both_ood_cells_and_covers_rest', sets['iid_test'] == test - f_meta - cl_meta)
check('adaptation_pool_is_candidate_fluoro_meta', sets['adaptation_pool'] == candidate & f_meta)
check('adaptation_pool_unseen_by_a_pretraining', not sets['adaptation_pool'] & sets['train_a'])
check('adaptation_pool_seen_by_b_reference_pretraining', sets['adaptation_pool'] <= sets['train_b'])
check('adaptation_test_equals_ood_fluoro_meta', sets['adaptation_test'] == sets['ood_f_meta'])
check('adaptation_support_and_test_disjoint', not sets['adaptation_pool'] & sets['adaptation_test'])
check('probe_source_is_ortho_test_and_target_is_meta_test', sets['probe_source'] == {i for i in test if records[i]['position'] == 'ortho'} and sets['probe_target'] == {i for i in test if records[i]['position'] == 'meta'})
check('probe_fit_and_test_molecular_identities_disjoint', not sets['probe_source'] & sets['probe_target'])

full_a, full_b = candidate - f_meta, candidate - cl_meta
shared = full_a & full_b
unique_a = sorted(full_a - full_b, key=lambda i: records[i]['id'])
unique_b = sorted(full_b - full_a, key=lambda i: records[i]['id'])
matched_count = min(len(unique_a), len(unique_b))
check('matched_training_a_matches_shared_plus_deterministic_exclusive_subset', sets['train_a'] == shared | set(unique_a[:matched_count]))
check('matched_training_b_matches_shared_plus_deterministic_exclusive_subset', sets['train_b'] == shared | set(unique_b[:matched_count]))
check('effective_training_sample_budgets_equal', len(sets['train_a']) == len(sets['train_b']) == 2371)

source_hashes = {}
for relative, expected in environment['source_sha256'].items():
    observed = digest(ROOT / relative)
    source_hashes[relative] = {'expected_at_run_start': expected, 'observed_during_audit': observed}
    check('training_source_hash_matches_run_environment', observed == expected)
end_input_hashes = {name: digest(ARCHIVE / name) for name in INPUT_NAMES}
check('all_audited_archive_inputs_unchanged_during_audit', input_hashes == end_input_hashes)

failed_count = sum(value['failed'] for value in checks.values())
report = {
    'audit_name': 'Independent mapped-molecule archive audit',
    'status': 'PASS' if failed_count == 0 else 'FAIL',
    'started_utc': started,
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'archive_directory': str(ARCHIVE.relative_to(ROOT)),
    'training_source_commit_at_start': environment['git_head_at_start'],
    'scope': 'All archived molecules and graph entries; no generator or model code imported; no training performed.',
    'method': 'Parse atom-map reference SMILES, renumber atoms by map_number minus one, independently reconstruct binary bonds, 13 ordinary chemical features and 5 query features, and recompute graph shortest paths and chemical class labels.',
    'audit_implementation_sha256': digest(Path(__file__)),
    'independent_dependencies': {'python': sys.version, 'numpy': importlib.metadata.version('numpy'), 'rdkit': importlib.metadata.version('rdkit'), 'platform': platform.platform()},
    'counts': {'molecules_audited': n_records, 'atoms_audited': sum(atom_counts), 'adjacency_entries_audited': sum(count * count for count in atom_counts), 'ordinary_feature_entries_audited': sum(atom_counts) * 13, 'query_feature_entries_audited': sum(atom_counts) * 5, 'unique_canonical_identities': len(set(canonical_identities)), 'maximum_atoms': max(atom_counts), 'check_categories': len(checks), 'individual_checks': sum(value['checked'] for value in checks.values()), 'failed_checks': failed_count},
    'checks': checks,
    'failures_first_30': failures,
    'archived_input_sha256': input_hashes,
    'training_source_sha256_verification': source_hashes,
    'array_shapes_and_dtypes': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in arrays.items()},
    'split_counts': {name: len(value) for name, value in splits.items()},
    'training_matching': {'shared_molecular_identities': len(shared), 'exclusive_molecular_identities_per_experience': matched_count, 'dropped_from_experience_a': len(unique_a) - matched_count, 'dropped_from_experience_b': len(unique_b) - matched_count},
    'interpretation_notes': ['The fluoro-meta adaptation support pool is unseen during experience A pretraining and seen during experience B pretraining; B is a seen-combination reference.', 'Atom-map references are audit-only metadata. Every archived atom feature was verified against ordinary chemistry or the five explicit query markers.'],
    'frozen_record_modified': False,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'counts': report['counts'], 'output': str(OUTPUT), 'failures': failures}, indent=2))
raise SystemExit(0 if failed_count == 0 else 1)
