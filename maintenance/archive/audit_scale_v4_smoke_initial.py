"""Independently audit frozen feature-scale readouts without experiment imports.

The audit reconstructs spatial masks, sampled encoder computations and ridge
predictions from immutable source inputs. No training routine is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maintenance.audit_mechanism_v3 import (
    Checks, digest, encoder_digest, indexed, read_csv, read_json,
    verify_executed_source, verify_manifest, verify_supports,
)


ROOT = Path(__file__).resolve().parents[1]
SCALES = ("root", "target", "target_plus1", "target_plus2", "target_plus3", "whole_graph", "legacy_graph")
POSITIONS = (0, 1, 2)
SPLITS = ("iid_a", "iid_b", "ood_common")


def array_digest(array: np.ndarray) -> str:
    """Hash values together with shape and dtype, including signed zero bits."""
    array = np.ascontiguousarray(array)
    checksum = hashlib.sha256()
    checksum.update(str((str(array.dtype), tuple(array.shape))).encode("utf-8"))
    checksum.update(array.tobytes())
    return checksum.hexdigest()


def scope_masks(adjacency: np.ndarray, count: int, root: int, target: list[int],
                radii: tuple[int, ...] = (1, 2, 3)) -> dict[str, np.ndarray]:
    """Reconstruct oracle post-hoc pooling regions by undirected graph distance."""
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square.")
    if not 0 < count <= adjacency.shape[0] or not 0 <= root < count:
        raise ValueError("Root and atom count must identify an existing atom.")
    if not target or len(set(target)) != len(target) or any(not 0 <= atom < count for atom in target):
        raise ValueError("Target atoms must be unique existing atoms.")
    if root not in target:
        raise ValueError("The target root must belong to the target fragment.")
    if any(not isinstance(radius, int) or radius < 1 for radius in radii):
        raise ValueError("Expansion radii must be positive integers.")
    active = adjacency[:count, :count] != 0
    if not np.array_equal(active, active.T):
        raise ValueError("Spatial masks require an undirected graph.")
    width = adjacency.shape[0]
    masks = {}
    root_mask = np.zeros(width, dtype=bool)
    root_mask[root] = True
    target_mask = np.zeros(width, dtype=bool)
    target_mask[target] = True
    masks.update(root=root_mask, target=target_mask)
    expanded = target_mask[:count].copy()
    for radius in range(1, max(radii, default=0) + 1):
        expanded |= np.any(active[expanded], axis=0)
        if radius in radii:
            mask = np.zeros(width, dtype=bool)
            mask[:count] = expanded
            masks[f"target_plus{radius}"] = mask
    graph = np.zeros(width, dtype=bool)
    graph[:count] = True
    masks["whole_graph"] = graph
    return masks


def receptive_region(adjacency: np.ndarray, pooled_atoms: np.ndarray, depth: int) -> np.ndarray:
    """Return the graph-distance union that can influence a pooled node set."""
    adjacency, pooled_atoms = np.asarray(adjacency), np.asarray(pooled_atoms, dtype=bool)
    if depth < 0 or int(depth) != depth or adjacency.shape != (pooled_atoms.size, pooled_atoms.size):
        raise ValueError("Depth and pooled membership must define a graph region.")
    reached = pooled_atoms.copy()
    for _ in range(depth):
        reached |= np.any(adjacency[reached] != 0, axis=0)
    return reached


def independent_ridge(train_x, train_y, test_x, alpha: float = 1.0,
                      classes: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Fit the declared support-standardized one-hot ridge independently.

    The smaller dual system is used for wide features. Test labels are absent
    from this function and cannot affect scaling, coefficients or predictions.
    """
    train_x, test_x = np.asarray(train_x, dtype=np.float64), np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and testing features must be compatible matrices.")
    if not train_x.shape[0] or train_y.shape != (train_x.shape[0],):
        raise ValueError("Training labels must match nonempty support rows.")
    if (not np.issubdtype(train_y.dtype, np.integer) or np.any(train_y < 0)
            or np.any(train_y >= classes) or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("Class labels and fixed positive regularization are required.")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("All features must be finite.")
    center = np.average(train_x, axis=0)
    dispersion = np.sqrt(np.average(np.square(train_x - center), axis=0))
    dispersion[dispersion <= np.finfo(np.float64).eps] = 1.0
    standardized = (train_x - center) / dispersion
    test_standardized = (test_x - center) / dispersion
    response = (train_y[:, None] == np.arange(classes)[None, :]).astype(np.float64)
    bias = response.mean(axis=0)
    centered_response = response - bias
    n, dimensions = standardized.shape
    if dimensions <= n:
        regularized = standardized.T @ standardized + alpha * np.eye(dimensions)
        coefficients = np.linalg.solve(regularized, standardized.T @ centered_response)
    else:
        regularized = standardized @ standardized.T + alpha * np.eye(n)
        coefficients = standardized.T @ np.linalg.solve(regularized, centered_response)
    scores = test_standardized @ coefficients + bias
    return scores.argmax(axis=1), scores


def encoder_layers(state: dict, x: np.ndarray, adjacency: np.ndarray,
                   atom_mask: np.ndarray) -> list[np.ndarray]:
    """Evaluate checkpoint tensors directly, independently of model classes."""
    source = {key: value.detach().cpu() for key, value in state.items()}
    features = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    edges = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
    valid = torch.as_tensor(np.asarray(atom_mask), dtype=torch.float32).unsqueeze(-1)
    layer_indices = sorted({int(name.split(".")[1]) for name in source if name.startswith("layers.")})
    if layer_indices != list(range(len(layer_indices))):
        raise ValueError("Encoder message-layer indices must be contiguous.")
    with torch.no_grad():
        hidden = F.relu(F.linear(features, source["embedding.weight"], source["embedding.bias"])) * valid
        result = [hidden.numpy().copy()]
        for index in layer_indices:
            prefix = f"layers.{index}."
            concatenated = torch.cat((hidden, torch.bmm(edges, hidden)), dim=-1)
            update = F.linear(F.relu(F.linear(concatenated, source[prefix + "update.0.weight"],
                                             source[prefix + "update.0.bias"])),
                              source[prefix + "update.2.weight"], source[prefix + "update.2.bias"])
            normalized = F.layer_norm(hidden + update, (hidden.shape[-1],),
                                      source[prefix + "norm.weight"], source[prefix + "norm.bias"], 1e-5)
            hidden = F.relu(normalized) * valid
            result.append(hidden.numpy().copy())
    return result


def independent_pool(hidden: np.ndarray, x: np.ndarray, atom_mask: np.ndarray,
                     masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Recompute equal-width means plus the separately named 448-D reference."""
    hidden, x, atom_mask = np.asarray(hidden), np.asarray(x), np.asarray(atom_mask)
    result = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.shape != atom_mask.shape or np.any(mask & (atom_mask == 0)):
            raise ValueError("Pooling masks must include only valid atoms.")
        counts = mask.sum(axis=1, keepdims=True)
        if np.any(counts == 0):
            raise ValueError("Every pooling region must contain an atom.")
        result[name] = ((hidden.astype(np.float64) * mask[:, :, None]).sum(axis=1) / counts).astype(np.float32)
    graph_sum = (hidden * atom_mask[:, :, None]).sum(axis=1)
    graph_count = atom_mask.sum(axis=1, keepdims=True)
    query = [(hidden * x[:, :, feature, None]).sum(axis=1) for feature in (13, 14, 16, 17)]
    root_mask = x[:, :, 15]
    root = (hidden * root_mask[:, :, None]).sum(axis=1) / root_mask.sum(axis=1, keepdims=True).clip(min=1)
    result["legacy_graph"] = np.concatenate((graph_sum / graph_count, graph_sum / np.sqrt(graph_count), root, *query), axis=1)
    return result


def balanced(truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute macro recall over classes that occur in the declared test split."""
    return float(statistics.fmean(float(np.mean(predictions[truth == label] == label))
                                 for label in np.unique(truth)))


def verify_direct_supports(plan: dict, records: list[dict], splits: dict,
                           checks: Checks, repeats=(0, 1, 2)) -> dict:
    """Verify that each source-only readout has all eight classes equally often."""
    entries = plan["sets"]
    lookup = indexed(entries, ("repeat", "source_position"), checks, "direct_support")
    expected = set(itertools.product(map(str, repeats), ("0", "1", "2")))
    checks.check("direct_support_complete_three_draws_three_source_positions", set(lookup) == expected)
    for key, entry in lookup.items():
        indices = entry["indices"]
        source_position = int(entry["source_position"])
        counts = Counter(records[index]["y_family"] for index in indices)
        checks.check("direct_support_has_256_unique_source_position_IDs", len(indices) == len(set(indices)) == 256
                     and set(indices) <= set(splits["analysis"])
                     and all(records[index]["y_position"] == source_position for index in indices), key)
        checks.check("direct_support_32_per_each_of_eight_classes", counts == Counter({label: 32 for label in range(8)}), key)
        checks.check("direct_support_IDs_match_indices", entry["ids"] == [records[index]["id"] for index in indices], key)
        rng = np.random.default_rng(50000 + 100 * int(entry["repeat"]) + source_position)
        reconstructed = []
        for label in range(8):
            candidates = sorted((index for index in splits["analysis"]
                                 if records[index]["y_position"] == source_position and records[index]["y_family"] == label),
                                key=lambda index: records[index]["id"])
            reconstructed.extend(rng.choice(candidates, size=32, replace=False).tolist())
        reconstructed.sort(key=lambda index: records[index]["id"])
        checks.check("direct_support_exact_predeclared_sampling_rule", indices == reconstructed, key)
    return lookup


def verify_regions(record: Path, records: list[dict], source_arrays: dict, indices: list[int],
                   layers: list[int], checks: Checks) -> dict[str, np.ndarray]:
    """Reconstruct every oracle pooling mask and every locality CSV row."""
    with np.load(record / "region_masks.npz", allow_pickle=False) as archive:
        masks = {name: archive[name].copy() for name in archive.files}
    checks.check("six_equal_width_masks_present", set(masks) == set(SCALES[:-1]))
    width = source_arrays["mask"].shape[1]
    for name, mask in masks.items():
        checks.check("pool_masks_boolean_and_correct_shape", mask.dtype == np.bool_
                     and mask.shape == (len(indices), width), name)
    locality = indexed(read_csv(record / "locality.csv"), ("index", "region", "layer"), checks, "locality")
    expected_locality = set(itertools.product(map(str, indices), SCALES, map(str, layers)))
    checks.check("locality_rows_cover_every_molecule_region_depth", set(locality) == expected_locality)
    for row, index in enumerate(indices):
        molecule = records[index]
        valid = source_arrays["mask"][index].astype(bool)
        count = int(valid.sum())
        target = molecule["target_fragment_indices"]
        expected = scope_masks(source_arrays["adj"][index], count, molecule["target_root_idx"], target)
        for name in SCALES[:-1]:
            checks.check("all_pool_masks_match_independent_BFS", np.array_equal(masks[name][row], expected[name]), (index, name))
        expected["legacy_graph"] = expected["whole_graph"]
        target_mask = expected["target"]
        for name, pooled in expected.items():
            for layer in layers:
                key = (str(index), name, str(layer))
                saved = locality.get(key)
                if saved is None:
                    continue
                receptive = receptive_region(source_arrays["adj"][index], pooled, layer)
                values = {"pool_atom_count": int(pooled.sum()), "target_atom_count": len(target),
                          "molecule_atom_count": count, "pool_outside_target_count": int(np.sum(pooled & ~target_mask)),
                          "input_receptive_atom_count": int(receptive.sum()),
                          "receptive_outside_target_count": int(np.sum(receptive & ~target_mask)),
                          "receptive_outside_pool_count": int(np.sum(receptive & ~pooled))}
                exact = saved["id"] == molecule["id"] and all(int(saved[field]) == value for field, value in values.items())
                exact &= (saved["receptive_covers_full_graph"].lower() == "true") == np.array_equal(receptive, valid)
                checks.check("all_locality_counts_recomputed_independently", exact, key)
    return masks


def verify_feature_archives(record: Path, source: Path, records: list[dict], source_arrays: dict,
                            indices: list[int], ids: list[str], layers: list[int],
                            masks: dict[str, np.ndarray], checks: Checks, seeds=(0, 1, 2)) -> tuple[dict, dict]:
    """Validate all feature arrays and independently recalculate CPU sentinels."""
    features = {}
    checkpoints = {}
    selected = np.array(sorted({0, len(indices) // 3, 2 * len(indices) // 3, len(indices) - 1}))
    selected_indices = [indices[row] for row in selected]
    sample_x = source_arrays["x"][selected_indices]
    sample_adj = source_arrays["adj"][selected_indices]
    sample_valid = source_arrays["mask"][selected_indices]
    for condition, seed in itertools.product(("a", "b", "random"), seeds):
        name = f"{condition}_seed{seed}"
        with np.load(record / "representations" / f"{name}.npz", allow_pickle=False) as archive:
            checks.check("feature_archives_have_exact_keys", set(archive.files) == {"ids", "indices", "layers", *SCALES}, name)
            checks.check("feature_rows_follow_one_shared_identity_order", archive["ids"].tolist() == ids
                         and archive["indices"].tolist() == indices and archive["layers"].tolist() == layers, name)
            arrays = {scale: archive[scale].copy() for scale in SCALES}
        checkpoint = torch.load(source / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        checkpoints[name] = {"checkpoint_sha256": digest(source / "checkpoints" / f"{name}.pt"),
                             "encoder_sha256": encoder_digest(state)}
        independently_encoded = encoder_layers(state, sample_x, sample_adj, sample_valid)
        for scale, array in arrays.items():
            dimension = 448 if scale == "legacy_graph" else 64
            checks.check("all_features_finite_float32_declared_dimensions", array.dtype == np.float32
                         and array.shape == (len(layers), len(indices), dimension) and np.isfinite(array).all(), (name, scale))
        for layer_row, layer in enumerate(layers):
            pools = independent_pool(independently_encoded[layer], sample_x, sample_valid,
                                     {scale: mask[selected] for scale, mask in masks.items()})
            for scale, expected in pools.items():
                observed = arrays[scale][layer_row, selected]
                maximum = float(np.max(np.abs(observed - expected)))
                checks.check("sampled_CPU_tensor_forward_reproduces_all_scales_and_layers",
                             np.allclose(observed, expected, atol=5e-5, rtol=2e-5), (name, scale, layer, maximum))
            checks.check("whole_graph_mean_agrees_with_legacy_first_block",
                         np.allclose(arrays["whole_graph"][layer_row], arrays["legacy_graph"][layer_row, :, :64], atol=2e-6, rtol=1e-6),
                         (name, layer))
        checks.check("layer_zero_root_has_one_constant_vector", np.array_equal(arrays["root"][0],
                     np.repeat(arrays["root"][0, :1], len(indices), axis=0)), name)
        checks.check("layer_zero_target_bag_has_one_constant_vector", np.array_equal(arrays["target"][0],
                     np.repeat(arrays["target"][0, :1], len(indices), axis=0)), name)
        features[name] = arrays
    return features, checkpoints


def verify_prediction_metrics(record: Path, records: list[dict], splits: dict, indices: list[int],
                              layers: list[int], checks: Checks, supports: dict, direct_supports: dict,
                              seeds=(0, 1, 2), repeats=(0, 1, 2)) -> tuple[list[dict], list[dict], dict]:
    """Check every metric against its stored per-molecule prediction vector."""
    with np.load(record / "predictions.npz", allow_pickle=False) as archive:
        predictions = {name: archive[name].copy() for name in archive.files}
    analysis = splits["analysis"]
    analysis_row = {index: row for row, index in enumerate(analysis)}
    truth = np.array([records[index]["y_family"] for index in analysis])
    test_rows = {}
    for experience in ("a", "b"):
        for split in SPLITS:
            test_rows[(experience, split, None)] = np.array([analysis_row[index] for index in splits[split]])
        for position in POSITIONS:
            position_indices = [index for index in analysis if records[index]["y_position"] == position]
            for split, allowed in (("target_all", set(analysis)), ("target_common_ood", set(splits["ood_common"])),
                                   ("target_own_iid", set(splits[f"iid_{experience}"]))):
                test_rows[(experience, split, str(position))] = np.array([analysis_row[index] for index in position_indices if index in allowed])
    adaptation = read_csv(record / "adaptation.csv")
    direct = read_csv(record / "direct_transfer.csv")
    common = ("experience", "condition", "seed", "repeat", "layer", "scale")
    adaptation_index = indexed(adaptation, (*common, "support", "split"), checks, "adaptation")
    direct_index = indexed(direct, (*common, "source_position", "target_position", "split"), checks, "direct_transfer")
    base = list(itertools.product(("a", "b"), ("trained", "random"), map(str, seeds), map(str, repeats), map(str, layers), SCALES))
    expected_adaptation = {(*case, support, split) for case in base for support, split in itertools.product(("new", "replay"), SPLITS)}
    expected_direct = {(*case, str(source), str(target), split) for case in base
                       for source, target in itertools.permutations(POSITIONS, 2)
                       for split in ("target_all", "target_common_ood", "target_own_iid")}
    checks.check("adaptation_complete_fixed_factorial_grid", set(adaptation_index) == expected_adaptation)
    checks.check("direct_complete_fixed_factorial_grid", set(direct_index) == expected_direct)
    prediction_usage = defaultdict(list)
    for kind, rows in (("adaptation", adaptation), ("direct", direct)):
        for row in rows:
            key = row["prediction_key"]
            prediction_usage[key].append((kind, row))
            dimension = 448 if row["scale"] == "legacy_graph" else 64
            expected_model = f"{row['experience'] if row['condition'] == 'trained' else 'random'}_seed{row['seed']}"
            checks.check("row_metadata_matches_model_feature_dimensions_and_budget", row["model"] == expected_model
                         and int(row["feature_dim"]) == dimension
                         and int(row["n_fit"]) == (256 if kind == "direct" else 32 if row["support"] == "new" else 64), row)
            if kind == "adaptation":
                selected = test_rows[(row["experience"], row["split"], None)]
            else:
                selected = test_rows[(row["experience"], row["split"], row["target_position"])]
            saved = predictions.get(key)
            valid = saved is not None and saved.shape == (len(analysis),) and saved.dtype == np.int8
            if valid:
                valid &= bool(np.all((saved >= 0) & (saved < 8)))
                actual = saved[selected]
                expected_accuracy = float(np.mean(actual == truth[selected]))
                expected_balanced = balanced(truth[selected], actual)
                valid &= (int(row["n_test"]) == len(selected)
                          and math.isclose(float(row["accuracy"]), expected_accuracy, abs_tol=1e-12, rel_tol=0)
                          and math.isclose(float(row["balanced_accuracy"]), expected_balanced, abs_tol=1e-12, rel_tol=0))
            checks.check("every_prediction_metric_recomputed", valid, (kind, key, row.get("split", row.get("target_position"))))
    checks.check("prediction_archive_covers_exactly_declared_readouts", set(predictions) == set(prediction_usage))
    identity_lookup = {row["id"]: index for index, row in enumerate(records)}
    logical_fits = {}
    for key, uses in prediction_usage.items():
        kinds = {kind for kind, _ in uses}
        experience_count = len({row["experience"] for _, row in uses})
        checks.check("each_fit_evaluated_on_all_fixed_test_splits", kinds == {"adaptation"} and len(uses) == 3 * experience_count
                     or kinds == {"direct"} and len(uses) == 6 * experience_count, key)
        fit_tags = set()
        fit_indices = None
        for kind, row in uses:
            if kind == "adaptation":
                support = supports[(row["experience"], row["repeat"])]
                fit_ids = support["new_ids"] + (support["old_ids"] if row["support"] == "replay" else [])
                fit_indices = tuple(identity_lookup[identifier] for identifier in fit_ids)
            else:
                fit_indices = tuple(direct_supports[(row["repeat"], row["source_position"])]["indices"])
            fit_tags.add((row["model"], row["layer"], row["scale"], fit_indices))
        checks.check("shared_prediction_keys_do_not_merge_distinct_logical_fits", len(fit_tags) == 1, key)
        logical_fits[key] = next(iter(fit_tags))
    checks.check("identical_logical_fits_are_cached_once", len(set(logical_fits.values())) == len(logical_fits))
    fits = indexed(read_csv(record / "fits.csv"), ("prediction_key",), checks, "fit_records")
    checks.check("fit_metadata_covers_all_unique_prediction_arrays", set(fits) == {(key,) for key in predictions})
    for key, (model, layer, scale, fit_indices) in logical_fits.items():
        saved = fits.get((key,))
        if saved is None:
            continue
        identifiers = [records[index]["id"] for index in fit_indices]
        support_hash = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
        prediction_hash = hashlib.sha256(predictions[key].tobytes()).hexdigest()
        valid = (saved["model"] == model and saved["layer"] == layer and saved["scale"] == scale
                 and int(saved["n_fit"]) == len(fit_indices)
                 and saved["support_ids_sha256"] == support_hash and saved["predictions_sha256"] == prediction_hash
                 and json.loads(saved["classes_seen"]) == sorted({records[index]["y_family"] for index in fit_indices}))
        checks.check("all_fit_ID_prediction_hashes_and_classes_recomputed", valid, key)
    for key, row in direct_index.items():
        if row["condition"] == "random" and row["experience"] == "a":
            paired_key = ("b", *key[1:])
            paired = direct_index.get(paired_key)
            checks.check("random_source_only_predictions_identical_across_experience_labels", paired is not None
                         and np.array_equal(predictions[row["prediction_key"]], predictions[paired["prediction_key"]]), key)
    return adaptation, direct, predictions


def verify_ridge_sentinels(adaptation: list[dict], direct: list[dict], predictions: dict,
                           features: dict, indices: list[int], records: list[dict], splits: dict,
                           supports: dict, direct_supports: dict, layers: list[int],
                           alpha: float, checks: Checks) -> dict:
    """Recompute every final-layer fit plus deterministic lower-depth sentinels."""
    feature_row = {index: row for row, index in enumerate(indices)}
    id_index = {record["id"]: index for index, record in enumerate(records)}
    analysis_rows = [feature_row[index] for index in splits["analysis"]]
    verified = set()
    counts = Counter()
    for kind, rows in (("adaptation", adaptation), ("direct", direct)):
        for row in rows:
            prediction_key = row["prediction_key"]
            if prediction_key in verified:
                continue
            primary_layer = int(row["layer"]) == 8
            lower_sentinel = row["seed"] == "0" and row["repeat"] == "0"
            if not (primary_layer or lower_sentinel):
                continue
            if kind == "adaptation":
                support = supports[(row["experience"], row["repeat"])]
                identifiers = support["new_ids"] + (support["old_ids"] if row["support"] == "replay" else [])
                fit = [id_index[identifier] for identifier in identifiers]
            else:
                fit = direct_supports[(row["repeat"], row["source_position"])]["indices"]
            layer_row = layers.index(int(row["layer"]))
            matrix = features[row["model"]][row["scale"]][layer_row]
            independently_predicted, _ = independent_ridge(matrix[[feature_row[index] for index in fit]],
                np.array([records[index]["y_family"] for index in fit]), matrix[analysis_rows], alpha=alpha)
            checks.check("independently_recomputed_ridge_predictions_exactly_match", np.array_equal(independently_predicted,
                         predictions[prediction_key]), (kind, row["model"], row["layer"], row["scale"], prediction_key))
            verified.add(prediction_key)
            counts[f"{kind}_{'primary_layer' if primary_layer else 'lower_layer_sentinel'}"] += 1
    return dict(counts)


def paired_measurements(rows: list[dict], keys: tuple[str, ...], checks: Checks) -> list[dict]:
    """Subtract exact matched random controls before any seed-level averaging."""
    controls = {}
    trained = {}
    for row in rows:
        key = tuple(row[field] for field in (*keys, "seed", "repeat"))
        destination = trained if row["condition"] == "trained" else controls
        checks.check("paired_measurements_have_unique_matched_keys", key not in destination, key)
        destination[key] = row
    checks.check("trained_random_measurements_have_identical_fit_test_panel", set(trained) == set(controls))
    result = []
    for key, current in trained.items():
        reference = controls.get(key)
        if reference is None:
            continue
        checks.check("paired_fits_have_same_label_budget_and_test_count", current["n_fit"] == reference["n_fit"]
                     and current["n_test"] == reference["n_test"], key)
        result.append({**dict(zip((*keys, "seed", "repeat"), key)),
                       "accuracy": float(current["accuracy"]) - float(reference["accuracy"]),
                       "balanced_accuracy": float(current["balanced_accuracy"]) - float(reference["balanced_accuracy"])})
    return result


def verify_aggregated_summary(rows: list[dict], saved: list[dict], keys: tuple[str, ...],
                              checks: Checks, label: str, expected_seeds=(0, 1, 2)) -> None:
    """Independently recompute support means within seed, then sample seed SD."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in keys)].append(row)
    indexed_saved = indexed(saved, keys, checks, label)
    checks.check(f"{label}_all_summary_groups_present", set(indexed_saved) == set(groups))
    for key, members in groups.items():
        saved_row = indexed_saved.get(key)
        if saved_row is None:
            continue
        for metric in ("accuracy", "balanced_accuracy"):
            by_seed = defaultdict(list)
            for member in members:
                by_seed[str(member["seed"])].append(float(member[metric]))
            per_seed = {seed: statistics.fmean(values) for seed, values in by_seed.items()}
            statistic = saved_row[metric]
            valid = (set(per_seed) == set(map(str, expected_seeds)) and statistic["n"] == len(expected_seeds)
                     and set(statistic["per_seed"]) == set(per_seed))
            if valid:
                valid &= all(math.isclose(float(statistic["per_seed"][seed]), value, rel_tol=0, abs_tol=1e-12)
                             for seed, value in per_seed.items())
                valid &= math.isclose(float(statistic["mean"]), statistics.fmean(per_seed.values()), rel_tol=0, abs_tol=1e-12)
                valid &= (statistic["std"] is None if len(expected_seeds) == 1 else
                          math.isclose(float(statistic["std"]), statistics.stdev(per_seed.values()), rel_tol=0, abs_tol=1e-12))
            checks.check(f"{label}_correct_seed_unit_mean_sample_SD", valid, (key, metric))


def verify_seed_tables(record: Path, panels: list[tuple], checks: Checks) -> None:
    """Check every derived seed mean, direction count and support-draw count."""
    fields = ("experiment", "repeat_scope", "experience", "condition", "seed", "layer", "scale", "feature_dim",
              "support", "split", "source_position", "target_position")
    for filename, paired in (("seed_metrics.csv", False), ("paired_seed_deltas.csv", True)):
        actual = indexed(read_csv(record / filename), fields, checks, filename)
        expected = {}
        for experiment, rows, keys, is_paired in panels:
            if paired != is_paired:
                continue
            for scope in ("all_repeats_within_seed", "primary_fixed_repeat"):
                selected = [row for row in rows if scope == "all_repeats_within_seed" or str(row["repeat"]) == "0"]
                groups = defaultdict(list)
                for row in selected:
                    groups[tuple(str(row[field]) for field in (*keys, "seed"))].append(row)
                for values, members in groups.items():
                    metadata = dict(zip((*keys, "seed"), values))
                    expected_row = {"experiment": experiment, "repeat_scope": scope,
                                    **{field: metadata.get(field, "") for field in fields[2:]}}
                    if paired:
                        expected_row["condition"] = "trained_minus_random"
                    key = tuple(expected_row[field] for field in fields)
                    expected[key] = {"support_repeats": len({str(row["repeat"]) for row in members}),
                                     "measurements_averaged": len(members),
                                     **{metric: statistics.fmean(float(row[metric]) for row in members)
                                        for metric in ("accuracy", "balanced_accuracy")}}
        checks.check(f"{filename}_complete_independent_seed_panel", set(actual) == set(expected))
        for key, values in expected.items():
            saved = actual.get(key)
            valid = saved is not None
            if valid:
                valid &= all(int(saved[field]) == values[field] for field in ("support_repeats", "measurements_averaged"))
                valid &= all(math.isclose(float(saved[field]), values[field], abs_tol=1e-12, rel_tol=0)
                             for field in ("accuracy", "balanced_accuracy"))
            checks.check(f"{filename}_means_and_nested_counts_recomputed", valid, key)


def verify_reporting(record: Path, adaptation: list[dict], direct: list[dict], checks: Checks, seeds=(0, 1, 2)) -> None:
    """Validate all raw, paired, primary, robustness and six-direction summaries."""
    common = ("experience", "condition", "layer", "scale", "feature_dim")
    adaptation_keys = (*common, "support", "split")
    direct_keys = (*common, "source_position", "target_position", "split")
    direction_keys = (*common, "split")
    paired_adapt_keys = tuple(field for field in adaptation_keys if field != "condition")
    paired_direct_keys = tuple(field for field in direct_keys if field != "condition")
    paired_direction_keys = tuple(field for field in direction_keys if field != "condition")
    paired_adapt = paired_measurements(adaptation, paired_adapt_keys, checks)
    paired_direct = paired_measurements(direct, paired_direct_keys, checks)
    summary = read_json(record / "report_summary.json")
    checks.check("primary_scope_and_uniform_chance_recorded", str(summary["primary_repeat"]) == "0"
                 and int(summary["primary_layer"]) == 8 and summary["motif_uniform_random_chance"] == 0.125)
    panels = [("adaptation", adaptation, adaptation_keys, False), ("direct_transfer", direct, direct_keys, False),
              ("direct_direction_average", direct, direction_keys, False),
              ("adaptation", paired_adapt, paired_adapt_keys, True), ("direct_transfer", paired_direct, paired_direct_keys, True),
              ("direct_direction_average", paired_direct, paired_direction_keys, True)]
    for experiment, rows, keys, paired in panels:
        name = ("paired_" if paired else "") + experiment
        verify_aggregated_summary(rows, summary[name], keys, checks, name, seeds)
        verify_aggregated_summary([row for row in rows if str(row["repeat"]) == "0"], summary["primary_" + name],
                                  keys, checks, "primary_" + name, seeds)
    verify_seed_tables(record, panels, checks)


def audit_record(record: Path, source: Path) -> dict:
    checks = Checks()
    config, environment = read_json(record / "config.json"), read_json(record / "environment.json")
    smoke = record.name.startswith("smoke_")
    checks.check("predeclared_matched_configuration_or_explicit_smoke_subset", (smoke or config["seeds"] == [0, 1, 2]
                 and config["support_repeats"] == [0, 1, 2] and config["layers"] == list(range(9)))
                 and bool(config["seeds"]) and set(config["seeds"]) <= {0, 1, 2}
                 and bool(config["support_repeats"]) and set(config["support_repeats"]) <= {0, 1, 2}
                 and set(config["layers"]) <= set(range(9)) and config["primary_repeat"] == 0 and config["primary_layer"] == 8
                 and config["alpha"] == 1.0 and config["direct_sampling_base"] == 50000 and config["direct_per_class"] == 32)
    torch.set_num_threads(int(config["cpu_threads"]))
    manifests = {"record": verify_manifest(record, checks, "record")}
    source_reference = read_json(record / "source_reference.json")
    for name, reference in source_reference["records"].items():
        path = Path(reference["path"])
        verified = verify_manifest(path, checks, f"source_{name}")
        manifests[name] = verified
        checks.check("every_prior_record_manifest_and_file_count_unchanged", verified["manifest_sha256"] == reference["manifest_sha256"]
                     and verified["files"] == reference["files"], name)
    provenance = verify_executed_source(ROOT, environment, checks)
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = read_json(source / "splits.json")
    supports = read_json(record / "support_sets.json")
    mechanism_source = Path(config["mechanism_source"])
    if not mechanism_source.is_absolute():
        mechanism_source = ROOT / mechanism_source
    checks.check("adaptation_support_file_is_exact_v3_copy", digest(record / "support_sets.json") == digest(mechanism_source / "support_sets.json"))
    support_index = verify_supports(source, records, splits, supports, checks)
    direct_supports = verify_direct_supports(read_json(record / "direct_support.json"), records, splits, checks, config["support_repeats"])
    order = read_json(record / "feature_indices.json")
    indices, ids = order["indices"], order["ids"]
    lookup = {row["id"]: index for index, row in enumerate(records)}
    expected_indices = set(splits["analysis"])
    for support in supports["sets"]:
        if support["repeat"] in config["support_repeats"]:
            expected_indices.update(lookup[identifier] for identifier in support["new_ids"] + support["old_ids"])
    checks.check("one_identity_safe_feature_union_covers_exactly_test_and_supports", len(indices) == len(set(indices))
                 and set(indices) == expected_indices and ids == [records[index]["id"] for index in indices])
    with np.load(source / "graphs.npz", allow_pickle=False) as archive:
        source_arrays = {key: archive[key].copy() for key in ("x", "adj", "mask")}
    masks = verify_regions(record, records, source_arrays, indices, config["layers"], checks)
    features, checkpoints = verify_feature_archives(record, source, records, source_arrays, indices, ids,
                                                   config["layers"], masks, checks, config["seeds"])
    adaptation, direct, predictions = verify_prediction_metrics(record, records, splits, indices, config["layers"], checks,
        support_index, direct_supports, config["seeds"], config["support_repeats"])
    recomputed = verify_ridge_sentinels(adaptation, direct, predictions, features, indices, records, splits,
                                      support_index, direct_supports, config["layers"], config["alpha"], checks)
    verify_reporting(record, adaptation, direct, checks, config["seeds"])
    return {"status": "PASS" if checks.success else "FAIL", "explicit_smoke_subset": smoke, "audited_utc": datetime.now(timezone.utc).isoformat(),
            "auditor_sha256": digest(Path(__file__)), "record": str(record), "source": str(source),
            "platform": platform.platform(), "manifests": manifests, "executed_source_provenance": provenance,
            "source_checkpoints": checkpoints,
            "method": "Independent CPU-only source and record hashes, exact support identities, BFS masks/receptive counts, sampled checkpoint-tensor forward features, all stored prediction metrics, selected full ridge refits, and all nested/paired seed summaries; no jcim imports and no training.",
            "limits": "Tensor-forward validation uses four deterministic molecule sentinels per encoder. Full ridge recomputation covers all layer-eight fits plus seed-zero/repeat-zero fits at every lower layer; other stored predictions have all metrics recomputed but not all coefficients reconstructed. Pooling scopes remain full-input receptive fields, and oracle atom membership does not test localization.",
            "counts": {"feature_molecules": len(indices), "encoders": len(features), "adaptation_rows": len(adaptation),
                       "direct_rows": len(direct), "prediction_vectors": len(predictions), "ridge_refits": recomputed},
            "checks": checks.results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/scale_v4")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/SCALE_V4_AUDIT.json")
    args = parser.parse_args(argv)
    record, source, output = args.record.resolve(), args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit record: {output}")
    if output.is_relative_to(record) or output.is_relative_to(source):
        raise ValueError("The audit output must remain outside immutable experiment records.")
    result = audit_record(record, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
