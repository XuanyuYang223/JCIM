"""Independent audit of the frozen-readout factorial experiment.

No v7 experiment module is imported. The audit reconstructs fitting transforms,
closed-form fits, classifier evaluations, source identities, and paired results.
Recorded states and clipping logs do not prove every historical optimizer step.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from maintenance.audit_mechanism_v3 import (
    Checks, digest, encoder_bytes, encoder_digest, verify_executed_source,
    verify_manifest,
)
from maintenance.audit_skip_v5 import functional_features

PRIOR_RECORDS = ("pilot_v1", "hard_v2", "mechanism_diagnostics_v3",
                 "readout_replay_v3", "correlation_v3", "scale_v4", "skip_v5",
                 "pooling_v6")
TASKS = ("family", "position", "distance")
SPLITS = ("iid_a", "iid_b", "ood_common")
SCOPES = ("full512", "shallow_only")
NORMALIZATIONS = ("raw", "train_z")
OBJECTIVES = ("family", "joint")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, tolerance=1e-11):
    return math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), rel_tol=0, abs_tol=tolerance)


def state_digest(state):
    checksum = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        checksum.update(name.encode("utf-8"))
        checksum.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def scoped_values(values, scope):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 512 or not np.isfinite(values).all():
        raise ValueError("Finite 512-coordinate features are required.")
    if scope not in SCOPES:
        raise ValueError("Unknown scope.")
    result = values.copy()
    if scope == "shallow_only":
        result[:, :448] = 0
    return result


def independent_scaler(values, normalization):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Nonempty finite training features are required.")
    if normalization == "raw":
        return np.zeros(values.shape[1]), np.ones(values.shape[1])
    if normalization != "train_z":
        raise ValueError("Unknown normalization.")
    mean, scale = values.mean(0), values.std(0)
    scale[scale <= np.finfo(np.float64).eps] = 1
    return mean, scale


def independent_ridge(values, labels, normalization, scope):
    values = scoped_values(values, scope).astype(np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(values),) or np.any((labels < 0) | (labels >= 8)):
        raise ValueError("Valid eight-class labels are required.")
    active = np.arange(448, 512) if scope == "shallow_only" else np.arange(512)
    selected = np.ascontiguousarray(values[:, active], dtype=np.float64)
    mean, scale = independent_scaler(selected, normalization)
    z = (selected - mean) / scale
    target = np.eye(8, dtype=np.float64)[labels]
    bias = target.mean(0)
    center = z.mean(0) if normalization == "raw" else np.zeros(len(active))
    centered = z - center
    covariance = centered.T @ centered
    covariance.flat[::len(covariance) + 1] += 1
    weights = np.linalg.solve(covariance, centered.T @ (target - bias))
    bias = bias - center @ weights
    return {"active_columns": active, "mean": mean, "scale": scale,
            "weights": weights, "bias": bias}


def ridge_logits(state, values, scope):
    values = scoped_values(values, scope).astype(np.float64)
    selected = np.ascontiguousarray(values[:, state["active_columns"]], dtype=np.float64)
    return ((selected - state["mean"]) / state["scale"] @ state["weights"] + state["bias"])


@torch.no_grad()
def independent_head_logits(state, values, mean, scale, task):
    z = ((np.asarray(values, dtype=np.float64) - np.asarray(mean))
         / np.asarray(scale)).astype(np.float32)
    output = []
    for offset in range(0, len(z), 128):
        current = z[offset:offset + 128]
        n = len(current)
        tensor = torch.from_numpy(current[np.arange(128) % n])
        hidden = F.relu(F.linear(tensor, state[f"{task}.0.weight"], state[f"{task}.0.bias"]))
        logits = F.linear(hidden, state[f"{task}.2.weight"], state[f"{task}.2.bias"])
        output.append(logits[:n].numpy())
    return np.concatenate(output)


def expected_initial_heads(random_state):
    output = {}
    for name, value in random_state.items():
        if not name.startswith("heads."):
            continue
        name = name[len("heads."):]
        if name.endswith(".0.weight"):
            value = torch.cat((value, value.new_zeros((value.shape[0], 64))), dim=1)
        output[name] = value.clone()
    return output


def classification_scores(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or not len(labels):
        raise ValueError("Matched nonempty class arrays are required.")
    correct = labels == predictions
    return {"accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean([correct[labels == value].mean()
                                                for value in np.unique(labels)]))}


def validate_clip_row(row, threshold=5):
    total, family, coefficient = (float(row[key]) for key in
                                 ("total_gradient_norm", "family_gradient_norm", "clip_coefficient"))
    expected = min(1, threshold / (total + 1e-6))
    return (all(math.isfinite(value) for value in (total, family, coefficient))
            and total >= 0 and family >= 0 and family <= total + 1e-5
            and 0 < coefficient <= 1 and close(coefficient, expected, 2e-7)
            and str(row["clipped"]).lower() == str(coefficient < 1).lower())


def factorial_effects(cells):
    """Return saturated within-seed effects without an additive attribution claim."""
    dimensions = len(next(iter(cells))) if cells else 0
    expected = set(itertools.product((0, 1), repeat=dimensions))
    if dimensions not in (2, 3):
        raise ValueError("Two or three factorial dimensions are required.")
    if set(cells) != expected or not all(math.isfinite(float(x)) for x in cells.values()):
        raise ValueError("One finite outcome in every factorial cell is required.")
    effects = {}
    for size in range(1, dimensions + 1):
        for factors in itertools.combinations(range(dimensions), size):
            companion = tuple(index for index in range(dimensions) if index not in factors)
            conditional = []
            for settings in itertools.product((0, 1), repeat=len(companion)):
                value = 0.0
                for levels in itertools.product((0, 1), repeat=len(factors)):
                    cell = [0] * dimensions
                    for index, level in zip(companion, settings):
                        cell[index] = level
                    for index, level in zip(factors, levels):
                        cell[index] = level
                    value += (-1) ** (len(factors) - sum(levels)) * cells[tuple(cell)]
                conditional.append((settings, value))
            effects[factors] = {"conditional": conditional,
                                "average": statistics.mean(value for _, value in conditional)}
    return effects


def check_identity(entry, expected, records):
    return (entry["indices"] == expected
            and entry["ids"] == [records[index]["id"] for index in expected])


def audit_plan(panel, skip, records, splits, config, checks):
    plan = read_json(panel / "plan.json")
    excluded = set(splits["analysis"] + splits["adaptation_pool"])
    for experience in ("a", "b"):
        for name, original in (("train", f"train_{experience}"),
                               ("validation", f"val_{experience}")):
            selected = plan["source_only"][experience][name]
            checks.check("source_train_validation_exact_original_order",
                         check_identity(selected, splits[original], records), (experience, name))
            checks.check("no_test_identity_reserve_or_target_combination_used_for_fitting",
                         not set(selected["indices"]) & excluded and all(
                             (records[index]["y_family"] + records[index]["y_position"]) % 3 != 2
                             for index in selected["indices"]), (experience, name))
        checks.check("source_training_count_and_train_validation_disjoint",
                     len(splits[f"train_{experience}"]) == 1386
                     and not set(splits[f"train_{experience}"]) & set(splits[f"val_{experience}"]))
    for name in (*SPLITS, "analysis"):
        checks.check("all_test_identities_exact_original_order",
                     check_identity(plan["tests"][name], splits[name], records), name)
    orders = {}
    with np.load(panel / "training_orders.npz") as archive, np.load(skip / "training_orders.npz") as old:
        expected_keys = {f"{experience}_seed{seed}" for experience in ("a", "b")
                         for seed in config["seeds"]}
        checks.check("exact_order_panel_without_duplicate_random_regimes", set(archive.files) == expected_keys)
        for key in expected_keys:
            experience, seed_text = key.rsplit("_seed", 1)
            seed = int(seed_text)
            values = archive[key].copy()
            orders[key] = values
            checks.check("orders_are_exact_archived_v5_prefix",
                         values.dtype == np.int32 and np.array_equal(values, old[key][:config["epochs"]]), key)
            rng = np.random.default_rng(1701 + seed)
            independent = np.asarray([rng.permutation(splits[f"train_{experience}"])
                                      for _ in range(config["epochs"])], dtype=np.int32)
            checks.check("orders_independently_reconstructed_from_seed_and_source_ids",
                         np.array_equal(values, independent), key)
    return plan, orders


def expected_encoder_names(seeds):
    result = {f"{regime}_shallow_skip_{experience}_seed{seed}"
              for regime in ("end_to_end", "frozen_readout") for experience in ("a", "b")
              for seed in seeds}
    result.update(f"random_seed{seed}" for seed in seeds)
    return result


def audit_encoders(panel, hard, skip, pooling, records, splits, config, checks):
    inventory = read_csv(panel / "encoder_inventory.csv")
    encoders = {row["model"]: row for row in inventory}
    checks.check("fifteen_unique_sources_without_random_duplicates",
                 len(encoders) == len(inventory)
                 and set(encoders) == expected_encoder_names(config["seeds"]))
    indices = sorted(set(splits["train_a"] + splits["train_b"] + splits["val_a"]
                         + splits["val_b"] + splits["analysis"]))
    lookup = {index: row for row, index in enumerate(indices)}
    analysis_rows = [lookup[index] for index in splits["analysis"]]
    sample_positions = [0, len(indices) // 3, len(indices) // 2, len(indices) - 1]
    sample_indices = [indices[position] for position in sample_positions]
    with np.load(hard / "graphs.npz") as archive:
        x, adjacency, mask = (torch.from_numpy(archive[name][sample_indices].copy())
                              for name in ("x", "adj", "mask"))
    features, source_states = {}, {}
    feature_checks = read_csv(panel / "feature_checks.csv")
    expected_feature_checks = set()
    for name, row in encoders.items():
        seed = int(row["seed"])
        random = name.startswith("random_")
        expected_checkpoint = (hard if random else skip) / "checkpoints" / f"{name}.pt"
        checks.check("encoder_source_paths_and_hashes_match_frozen_records",
                     Path(row["checkpoint"]).resolve() == expected_checkpoint.resolve()
                     and row["checkpoint_sha256"] == digest(expected_checkpoint), name)
        state = torch.load(expected_checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        source_states[name] = state
        source_hash = encoder_digest(state)
        checks.check("all_encoder_hashes_before_after_equal_original_frozen_source",
                     row["encoder_before_sha256"] == row["encoder_after_sha256"] == source_hash
                     and row["encoder_bitwise_unchanged"].lower() == "true", name)
        checks.check("encoder_condition_regime_and_identity_metadata_consistent",
                     (random and row["condition"] == row["regime"] == "random" and row["experience"] == "shared")
                     or (not random and row["condition"] == "trained"
                         and name == f"{row['regime']}_shallow_skip_{row['experience']}_seed{seed}"), name)
        if row["regime"] == "frozen_readout":
            old = torch.load(hard / f"checkpoints/{row['experience']}_seed{seed}.pt",
                             map_location="cpu", weights_only=True)["state_dict"]
            checks.check("v5_frozen_regime_is_original_task_trained_encoder",
                         encoder_bytes(state) == encoder_bytes(old), name)
        path = panel / row["features"]
        checks.check("feature_archive_hash_and_dimensions_match_inventory",
                     digest(path) == row["features_sha256"] and int(row["n_features"]) == len(indices)
                     and int(row["feature_dim"]) == 512, name)
        with np.load(path) as archive:
            values = archive["readout"].copy()
            checks.check("features_use_exact_source_train_val_analysis_union_without_reserve",
                         archive["indices"].tolist() == indices
                         and archive["ids"].tolist() == [records[index]["id"] for index in indices], name)
        checks.check("all_frozen_features_finite_float32_width512",
                     values.shape == (len(indices), 512) and values.dtype == np.float32
                     and np.isfinite(values).all(), name)
        features[name] = values
        graph, shallow = functional_features(state, x, adjacency, mask)
        independent = torch.cat((graph, shallow), dim=1).numpy()
        checks.check("selected_full_encoder_features_independently_reconstructed_on_CPU",
                     np.allclose(independent, values[sample_positions], rtol=3e-5, atol=3e-5),
                     (name, float(np.max(np.abs(independent - values[sample_positions])))))
        with np.load(pooling / f"pooling/representations/pooling_{name}.npz") as old:
            checks.check("all_shallow_features_bitexact_v6_actual_branch",
                         old["indices"].tolist() == indices and np.array_equal(values[:, 448:], old["whole_graph"]), name)
        expected_feature_checks.add((name, "v6_shallow64"))
        if not random:
            with np.load(skip / f"representations/{name}.npz") as old:
                checks.check("all_trained_analysis512_features_bitexact_v5",
                             old["indices"].tolist() == splits["analysis"]
                             and np.array_equal(values[analysis_rows], old["readout"]), name)
            expected_feature_checks.add((name, "v5_analysis512"))
            if row["regime"] == "frozen_readout":
                with np.load(skip / f"representations/frozen_{row['experience']}_seed{seed}.npz") as old:
                    selected = [lookup[int(index)] for index in old["indices"]]
                    expected = np.concatenate((old["deep_graph"], old["shallow_mean"]), axis=1)
                    checks.check("all_original_frozen_cache_features_bitexact_v5",
                                 np.array_equal(values[selected], expected), name)
                expected_feature_checks.add((name, "v5_frozen_cache512"))
        random_state = torch.load(hard / f"checkpoints/random_seed{seed}.pt",
                                  map_location="cpu", weights_only=True)["state_dict"]
        checks.check("encoder_inventory_random_head_initializer_is_archived_same_seed",
                     row["initial_head_sha256"] == state_digest(expected_initial_heads(random_state)), name)
    actual_feature_checks = {(row["model"], row["kind"]) for row in feature_checks}
    checks.check("feature_continuity_check_panel_complete_and_exact",
                 actual_feature_checks == expected_feature_checks and len(feature_checks) == len(actual_feature_checks)
                 and all(row["exact_match"].lower() == "true" and float(row["max_abs_delta"]) == 0
                         for row in feature_checks))
    return encoders, source_states, features, indices


def initial_gradient_norms(state, values, labels, objective):
    """Check the first recorded gradient without using training implementation."""
    parameters = {name: value.clone().requires_grad_() for name, value in state.items()}
    x = torch.from_numpy(np.asarray(values, dtype=np.float32))
    losses = []
    for task in TASKS if objective == "joint" else ("family",):
        hidden = F.relu(F.linear(x, parameters[f"{task}.0.weight"], parameters[f"{task}.0.bias"]))
        logits = F.linear(hidden, parameters[f"{task}.2.weight"], parameters[f"{task}.2.bias"])
        losses.append(F.cross_entropy(logits, torch.as_tensor(labels[task], dtype=torch.int64)))
    sum(losses).backward()
    gradients = {name: parameter.grad for name, parameter in parameters.items() if parameter.grad is not None}
    family = [torch.linalg.vector_norm(value, 2) for name, value in gradients.items() if name.startswith("family.")]
    total = [torch.linalg.vector_norm(value, 2) for value in gradients.values()]
    return float(torch.linalg.vector_norm(torch.stack(total), 2)), float(torch.linalg.vector_norm(torch.stack(family), 2))


def audit_clipping(panel, fits, orders, features, feature_indices, records, config, checks):
    clipping = defaultdict(list)
    with (panel / "gradient_clipping.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            clipping[row["fit_key"]].append(row)
    expected_keys = {key for key, row in fits.items() if row["classifier"] == "mlp"}
    checks.check("gradient_log_contains_exact_MLP_panel", set(clipping) == expected_keys)
    lookup = {index: row for row, index in enumerate(feature_indices)}
    total_rows = 0
    for key in expected_keys:
        fit = fits[key]
        rows = clipping[key]
        updates = config["epochs"] * 11
        total_rows += len(rows)
        valid = len(rows) == updates
        valid = valid and all(int(row["step"]) == step + 1 and int(row["epoch"]) == step // 11 + 1
                              and validate_clip_row(row) for step, row in enumerate(rows))
        checks.check("every_update_has_valid_norm_global_clip_and_sequential_budget", valid, key)
        clip_count = sum(str(row["clipped"]).lower() == "true" for row in rows)
        checks.check("clipping_inventory_matches_per_update_evidence",
                     int(fit["clip_count"]) == clip_count and close(fit["minimum_clip_coefficient"],
                                                                   min(float(row["clip_coefficient"]) for row in rows)), key)
        if fit["objective"] == "family":
            checks.check("family_only_total_gradient_contains_only_family_head",
                         all(close(row["total_gradient_norm"], float(row["family_gradient_norm"]), 1e-6) for row in rows), key)
        checkpoint = torch.load(panel / fit["checkpoint"], map_location="cpu", weights_only=True)
        first = orders[f"{fit['experience']}_seed{fit['seed']}"][0, :128]
        selected = [lookup[int(index)] for index in first]
        values = scoped_values(features[fit["model"]], fit["scope"])[selected]
        values = ((values.astype(np.float64) - checkpoint["mean"].numpy()) / checkpoint["scale"].numpy()).astype(np.float32)
        labels = {task: np.asarray([records[int(index)][f"y_{task}"] for index in first]) for task in TASKS}
        total, family = initial_gradient_norms(checkpoint["initial_state_dict"], values, labels, fit["objective"])
        checks.check("first_update_gradient_norms_independently_reconstructed_on_CPU",
                     math.isclose(total, float(rows[0]["total_gradient_norm"]), rel_tol=1e-4, abs_tol=1e-4)
                     and math.isclose(family, float(rows[0]["family_gradient_norm"]), rel_tol=1e-4, abs_tol=1e-4),
                     (key, total, family))
    return total_rows


def audit_fits(panel, hard, skip, pooling, encoders, source_states, features, indices,
               records, splits, config, plan, orders, checks):
    rows = read_csv(panel / "fit_inventory.csv")
    fits = {row["fit_key"]: row for row in rows}
    supports = {row["fit_key"]: row for row in plan["fits"]}
    checks.check("fit_inventory_and_plan_keys_unique_and_complete",
                 len(fits) == len(rows) == len(supports) == len(plan["fits"])
                 and set(fits) == set(supports) == {f"fit_{index:04d}" for index in range(len(fits))})
    lookup = {index: row for row, index in enumerate(indices)}
    expected_metrics, actual_fit_cells, continuity_expected = {}, set(), {}
    mlp_count, ridge_count, head_mismatches = 0, 0, 0
    initializer_hashes = defaultdict(set)
    previous_ridges = {(row["model"], row["experience"]): row
                      for row in read_csv(pooling / "pooling/pooling_fit_inventory.csv")
                      if row["classifier"] == "ridge" and row["protocol"] == "source_only"
                      and row["pool"] == "whole_graph"}
    v5_predictions = {}
    for row in read_csv(skip / "predictions.csv"):
        if (row["regime"] == "frozen_readout" and row["variant"] == "shallow_skip"
                and row["intervention"] == "intact"):
            v5_predictions[(row["experience"], int(row["seed"]), int(row["index"]))] = {
                task: int(row[f"pred_{task}"]) for task in TASKS}
    for key, row in fits.items():
        model, experience, seed = row["model"], row["experience"], int(row["seed"])
        classifier, scope, normalization, objective = (row[field] for field in
                                                       ("classifier", "scope", "normalization", "objective"))
        encoder = encoders.get(model, {})
        cell = (model, experience, classifier, scope, normalization, objective)
        valid = bool(encoder) and cell not in actual_fit_cells and scope in SCOPES and normalization in NORMALIZATIONS
        valid = valid and classifier in ("ridge", "mlp") and (
            objective in OBJECTIVES if classifier == "mlp" else objective == "ridge")
        valid = valid and all(row[field] == encoder.get(field) for field in ("condition", "regime", "seed"))
        valid = valid and (row["condition"] == "random" or experience == encoder.get("experience"))
        checks.check("every_fit_factor_encoder_and_experience_metadata_exact", valid, (key, cell))
        actual_fit_cells.add(cell)
        support = supports[key]["support"]
        train = splits[f"train_{experience}"]
        labels = np.asarray([records[index]["y_family"] for index in train], dtype=np.int64)
        checks.check("every_fit_uses_exact_source_labels_and_only_source_support",
                     check_identity(support, train, records)
                     and all(str(supports[key][field]) == row[field] for field in
                             ("model", "condition", "regime", "experience", "seed", "classifier", "scope", "normalization", "objective"))
                     and int(row["n_fit"]) == 1386
                     and row["support_ids_sha256"] == hashlib.sha256("\n".join(support["ids"]).encode()).hexdigest()
                     and row["support_labels_sha256"] == hashlib.sha256(labels.tobytes()).hexdigest()
                     and row["encoder_sha256"] == encoder["encoder_before_sha256"], key)
        values = scoped_values(features[model], scope)
        train_values = values[[lookup[index] for index in train]]
        mean, scale = independent_scaler(train_values, normalization)
        checkpoint_path, prediction_path = panel / row["checkpoint"], panel / row["predictions"]
        checks.check("fit_state_and_prediction_hashes_match_inventory",
                     digest(checkpoint_path) == row["checkpoint_sha256"]
                     and digest(prediction_path) == row["predictions_sha256"], key)
        with np.load(prediction_path) as archive:
            expected_tasks = TASKS if classifier == "mlp" else ("family",)
            expected_keys = {"indices", "ids"} | {f"{prefix}_{task}" for prefix in ("logits", "pred") for task in expected_tasks}
            scores = {task: archive[f"logits_{task}"].copy() for task in expected_tasks}
            predictions = {task: archive[f"pred_{task}"].copy() for task in expected_tasks}
            checks.check("prediction_archive_keys_and_identity_order_exact",
                         set(archive.files) == expected_keys and archive["indices"].tolist() == indices
                         and archive["ids"].tolist() == [records[index]["id"] for index in indices], key)
        checks.check("all_predictions_are_argmax_of_finite_task_logits",
                     all(scores[task].ndim == 2 and scores[task].shape[0] == len(indices)
                         and np.isfinite(scores[task]).all() and np.array_equal(predictions[task], scores[task].argmax(1))
                         for task in expected_tasks) and scores["family"].shape[1] == 8, key)
        if classifier == "ridge":
            ridge_count += 1
            expected = independent_ridge(features[model][[lookup[index] for index in train]], labels, normalization, scope)
            with np.load(checkpoint_path) as archive:
                checks.check("all_ridge_coefficients_independently_refitted_exactly",
                             set(archive.files) == set(expected)
                             and all(np.array_equal(archive[field], value) for field, value in expected.items()), key)
            actual = ridge_logits(expected, features[model], scope)
            checks.check("all_ridge_logits_and_predictions_recomputed_exactly",
                         np.array_equal(actual, scores["family"])
                         and np.array_equal(actual.argmax(1), predictions["family"]), key)
            checks.check("ridge_fixed_dimensions_and_zero_optimizer_budget",
                         int(row["optimizer_updates"]) == 0 and row["training_order_sha256"] == ""
                         and row["initial_head_sha256"] == row["final_head_sha256"] == ""
                         and int(row["feature_dim"]) == int(row["active_feature_dim"]) == len(expected["active_columns"])
                         and int(row["classifier_parameter_count"]) == int(row["family_parameter_count"])
                         == expected["weights"].size + 8 and row["history"] == "", key)
            if scope == "shallow_only" and normalization == "train_z":
                previous = previous_ridges[(model, experience)]
                with np.load(pooling / "pooling" / previous["checkpoint"]) as archive:
                    exact_state = all(np.array_equal(expected[field], archive[field]) for field in ("mean", "scale", "weights", "bias"))
                with np.load(pooling / "pooling" / previous["predictions"]) as archive:
                    exact_scores = np.array_equal(scores["family"], archive["logits"])
                    exact_predictions = np.array_equal(predictions["family"], archive["prediction"])
                checks.check("v6_standardized_shallow_ridge_states_logits_predictions_exact",
                             exact_state and exact_scores and exact_predictions, key)
                continuity_expected[(key, "v6_shallow_standardized_ridge")] = True
        else:
            mlp_count += 1
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            random_state = torch.load(hard / f"checkpoints/random_seed{seed}.pt",
                                      map_location="cpu", weights_only=True)["state_dict"]
            initial = expected_initial_heads(random_state)
            digest_initial = state_digest(initial)
            initializer_hashes[seed].add(row["initial_head_sha256"])
            checks.check("MLP_initialization_exact_archived_random_heads_zero_appended_columns",
                         digest_initial == state_digest(state["initial_state_dict"]) == row["initial_head_sha256"], key)
            checks.check("MLP_transform_uses_only_source_training_statistics",
                         isinstance(state["mean"], torch.Tensor) and state["mean"].dtype == torch.float64
                         and isinstance(state["scale"], torch.Tensor) and state["scale"].dtype == torch.float64
                         and np.array_equal(state["mean"].numpy(), mean) and np.array_equal(state["scale"].numpy(), scale), key)
            parameter_count = sum(value.numel() for value in initial.values())
            family_count = sum(value.numel() for name, value in initial.items() if name.startswith("family."))
            checks.check("every_MLP_has_same512_input_parameters_and_exact_update_budget",
                         int(row["feature_dim"]) == 512 and int(row["active_feature_dim"]) == (512 if scope == "full512" else 64)
                         and int(row["classifier_parameter_count"]) == parameter_count
                         and int(row["family_parameter_count"]) == family_count
                         and int(row["optimizer_updates"]) == 11 * config["epochs"]
                         and row["training_order_sha256"] == hashlib.sha256(orders[f"{experience}_seed{seed}"].tobytes()).hexdigest(), key)
            checks.check("MLP_saved_final_state_digest_and_metadata_match_inventory",
                         state_digest(state["state_dict"]) == row["final_head_sha256"]
                         and all(str(state["metadata"][field]) == row[field] for field in state["metadata"]), key)
            if objective == "family":
                checks.check("untrained_other_heads_remain_exactly_at_initialization",
                             all(torch.equal(value, state["state_dict"][name]) for name, value in initial.items()
                                 if not name.startswith("family.")), key)
            for task in TASKS:
                actual = independent_head_logits(state["state_dict"], values, mean, scale, task)
                mismatches = int(np.count_nonzero(actual.argmax(1) != predictions[task]))
                head_mismatches += mismatches
                checks.check("every_saved_MLP_head_prediction_independently_reconstructed_on_CPU",
                             mismatches == 0, (key, task, mismatches))
                checks.check("MLP_logits_match_independent_function_with_fixed_tolerance",
                             np.allclose(actual, scores[task], rtol=3e-5, atol=3e-5),
                             (key, task, float(np.max(np.abs(actual - scores[task])))))
            history = read_json(panel / row["history"])
            epochs = sorted(set([1, *range(10, config["epochs"] + 1, 10), config["epochs"]]))
            checks.check("histories_only_source_loss_familiar_validation_and_fixed_final_epoch",
                         [entry["epoch"] for entry in history] == epochs
                         and all(set(entry) == {"epoch", "train_loss", *{f"val_{task}_accuracy" for task in TASKS}}
                                 and math.isfinite(float(entry["train_loss"])) for entry in history), key)
            validation_rows = [lookup[index] for index in splits[f"val_{experience}"]]
            checks.check("final_familiar_validation_history_matches_saved_predictions",
                         all(close(history[-1][f"val_{task}_accuracy"], np.mean(
                             predictions[task][validation_rows] == [records[index][f"y_{task}"]
                                                                     for index in splits[f"val_{experience}"]])) for task in TASKS), key)
            if (config["epochs"] == 120 and row["regime"] == "frozen_readout"
                    and scope == "full512" and normalization == "raw" and objective == "joint"):
                original_heads = {name[len("heads."):]: value for name, value in source_states[model].items()
                                  if name.startswith("heads.")}
                identical = state_digest(original_heads) == state_digest(state["state_dict"])
                identical_predictions = all(int(predictions[task][lookup[index]]) == v5_predictions[(experience, seed, index)][task]
                                            for index in splits["analysis"] for task in TASKS)
                checks.check("v5_frozen_raw_full_joint_head_states_and_all_predictions_exact",
                             identical and identical_predictions, key)
                continuity_expected[(key, "v5_raw_full_joint")] = True
        for split in ("train", *SPLITS):
            selected = train if split == "train" else splits[split]
            selected_rows = [lookup[index] for index in selected]
            for task in expected_tasks:
                truth = [records[index][f"y_{task}"] for index in selected]
                expected_metrics[(key, split, task)] = {"n_test": len(selected),
                    "task_trained": task == "family" or objective == "joint",
                    **classification_scores(truth, predictions[task][selected_rows])}
    expected_cells = set()
    for model, encoder in encoders.items():
        experiences = ("a", "b") if encoder["condition"] == "random" else (encoder["experience"],)
        for experience in experiences:
            for scope, normalization in itertools.product(SCOPES, NORMALIZATIONS):
                expected_cells.add((model, experience, "ridge", scope, normalization, "ridge"))
                expected_cells.update((model, experience, "mlp", scope, normalization, objective) for objective in OBJECTIVES)
    checks.check("exact144_MLP72_ridge_factorial_without_random_fit_duplication", actual_fit_cells == expected_cells and len(rows) == len(expected_cells))
    checks.check("one_archived_initializer_per_seed_across_every_cell_and_encoder_source",
                 all(len(values) == 1 for values in initializer_hashes.values()))
    metric_rows = read_csv(panel / "source_metrics.csv")
    seen = set()
    metadata_fields = ("model", "condition", "regime", "seed", "experience", "classifier", "scope", "normalization", "objective")
    for saved in metric_rows:
        key = (saved["fit_key"], saved["split"], saved["task"])
        expected = expected_metrics.get(key)
        valid = key not in seen and expected is not None
        valid = valid and all(saved[field] == fits[saved["fit_key"]][field] for field in metadata_fields)
        if expected:
            valid = valid and int(saved["n_test"]) == expected["n_test"]
            valid = valid and saved["task_trained"].lower() == str(expected["task_trained"]).lower()
            valid = valid and all(close(saved[field], expected[field]) for field in ("accuracy", "balanced_accuracy"))
        seen.add(key)
        checks.check("every_source_metric_recomputed_from_per_molecule_predictions", valid, key)
    checks.check("source_metric_panel_complete_without_duplicate_rows", seen == set(expected_metrics))
    continuity = read_csv(panel / "continuity_checks.csv")
    checks.check("continuity_anchor_log_exact_panel_states_and_predictions",
                 {(row["fit_key"], row["kind"]) for row in continuity} == set(continuity_expected)
                 and len(continuity) == len(continuity_expected)
                 and all(row["state_exact_match"].lower() == row["predictions_exact_match"].lower() == "true"
                         and float(row["max_abs_delta"]) == 0 for row in continuity))
    gradient_rows = audit_clipping(panel, fits, orders, features, indices, records, config, checks)
    return metric_rows, {"encoder_count": len(encoders), "fit_count": len(fits),
        "MLP_saved_states_evaluated": mlp_count, "ridge_fits_independently_resolved": ridge_count,
        "MLP_CPU_prediction_mismatches": head_mismatches, "source_metric_rows": len(metric_rows),
        "clipping_rows": gradient_rows, "continuity_anchors": len(continuity_expected)}


def verify_seed_panels(rows, panels, fields, seeds, checks, label):
    grouped = defaultdict(dict)
    for row in rows:
        key, seed = tuple(str(row[field]) for field in fields), str(row["seed"])
        checks.check(f"{label}_one_value_per_seed", seed not in grouped[key], (key, seed))
        grouped[key][seed] = row
    saved = {tuple(str(row[field]) for field in fields): row for row in panels}
    checks.check(f"{label}_panel_exact", len(saved) == len(panels) and set(saved) == set(grouped))
    expected_seeds = {str(seed) for seed in seeds}
    for key, members in grouped.items():
        panel = saved.get(key, {})
        valid = set(members) == expected_seeds
        for metric in ("accuracy", "balanced_accuracy"):
            recorded = panel.get(metric, {})
            values = {seed: float(row[metric]) for seed, row in members.items()}
            valid = valid and set(recorded.get("per_seed", {})) == set(values)
            valid = valid and int(recorded.get("n_seeds", -1)) == len(values)
            if valid:
                valid = all(close(recorded["per_seed"][seed], value) for seed, value in values.items())
                valid = valid and close(recorded["mean"], statistics.mean(values.values()))
                valid = valid and (recorded["std"] is None if len(values) == 1 else
                                   recorded["std"] is not None and close(recorded["std"], statistics.stdev(values.values())))
        checks.check(f"{label}_paired_seed_mean_sample_SD_and_exact_seed_values", valid, key)


def audit_statistics(record, metric_rows, config, checks):
    family = [row for row in metric_rows if row["task"] == "family"]
    summary = read_json(record / "report_summary.json")
    cell_fields = ("regime", "experience", "classifier", "scope", "normalization", "objective", "split")
    base_fields = ("regime", "experience", "classifier", "split", "seed")
    levels = {"scope": ("full512", "shallow_only"), "normalization": ("raw", "train_z"),
              "objective": ("joint", "family")}
    verify_seed_panels(family, summary["cells"], cell_fields, config["seeds"], checks, "cell_summary")
    grouped = defaultdict(list)
    for row in family:
        grouped[tuple(row[field] for field in base_fields)].append(row)
    expected_edges, expected_effects = {}, {}
    for key, rows in grouped.items():
        names = ("scope", "normalization", "objective") if key[2] == "mlp" else ("scope", "normalization")
        cells = {tuple(levels[name].index(row[name]) for name in names): row for row in rows}
        checks.check("statistics_input_has_exact_factorial_cells",
                     len(cells) == len(rows) and set(cells) == set(itertools.product((0, 1), repeat=len(names))), key)
        per_metric = {metric: factorial_effects({bits: float(row[metric]) for bits, row in cells.items()})
                      for metric in ("accuracy", "balanced_accuracy")}
        for factor_tuple, data in per_metric["accuracy"].items():
            factor_names = "+".join(names[index] for index in factor_tuple)
            effect_key = (*key, factor_names, len(factor_tuple))
            expected_effects[effect_key] = {metric: values[factor_tuple]["average"] for metric, values in per_metric.items()}
            if len(factor_tuple) != 1:
                continue
            factor = factor_tuple[0]
            companion = [index for index in range(len(names)) if index != factor]
            for ordinal, (settings, _) in enumerate(data["conditional"]):
                tags = {names[factor]: "*"}
                tags.update({names[index]: levels[names[index]][value] for index, value in zip(companion, settings)})
                tags.setdefault("objective", "ridge")
                edge_key = (*key, names[factor], tags["scope"], tags["normalization"], tags["objective"],
                            f"{levels[names[factor]][1]}_minus_{levels[names[factor]][0]}")
                expected_edges[edge_key] = {metric: values[factor_tuple]["conditional"][ordinal][1]
                                            for metric, values in per_metric.items()}
    edge_rows = read_csv(record / "conditional_factor_edges.csv")
    effect_rows = read_csv(record / "factorial_effects.csv")
    edge_fields = (*base_fields, "factor", "scope", "normalization", "objective", "contrast")
    effect_fields = (*base_fields, "factors", "order")
    seen = set()
    for row in edge_rows:
        key = tuple(row[field] for field in edge_fields)
        expected = expected_edges.get(key)
        checks.check("all_conditional_edges_recomputed_with_companion_factors_fixed",
                     key not in seen and expected is not None
                     and all(close(row[field], value) for field, value in (expected or {}).items()), key)
        seen.add(key)
    checks.check("conditional_edge_panel_complete", seen == set(expected_edges) and len(edge_rows) == len(seen))
    seen = set()
    for row in effect_rows:
        key = tuple(row[field] for field in effect_fields[:-1]) + (int(row["order"]),)
        expected = expected_effects.get(key)
        checks.check("all_averaged_main_and_interaction_effects_independently_recomputed",
                     key not in seen and expected is not None
                     and all(close(row[field], value) for field, value in (expected or {}).items()), key)
        seen.add(key)
    checks.check("factor_effect_panel_complete", seen == set(expected_effects) and len(effect_rows) == len(seen))
    random_fields = ("experience", "classifier", "scope", "normalization", "objective", "split", "seed")
    random = {tuple(row[field] for field in random_fields): row for row in family if row["regime"] == "random"}
    expected_random = {}
    for row in family:
        if row["regime"] == "random":
            continue
        key = tuple(row[field] for field in random_fields)
        reference = random.get(key)
        checks.check("every_trained_endpoint_has_same_protocol_seed_matched_random",
                     reference is not None and reference["n_test"] == row["n_test"], key)
        if reference:
            expected_random[(*key, row["regime"])] = {
                metric: float(row[metric]) - float(reference[metric]) for metric in ("accuracy", "balanced_accuracy")}
    random_rows = read_csv(record / "matched_random_deltas.csv")
    seen = set()
    for row in random_rows:
        key = tuple(row[field] for field in (*random_fields, "regime"))
        expected = expected_random.get(key)
        checks.check("all_trained_random_deltas_recomputed_without_random_regime_duplication",
                     key not in seen and expected is not None and row["contrast"] == "trained_minus_matched_random"
                     and all(close(row[field], value) for field, value in (expected or {}).items()), key)
        seen.add(key)
    checks.check("matched_random_delta_panel_complete", seen == set(expected_random) and len(random_rows) == len(seen))
    verify_seed_panels(edge_rows, summary["conditional_edges"],
                      (*base_fields[:-1], "factor", "scope", "normalization", "objective", "contrast"),
                      config["seeds"], checks, "edge_summary")
    verify_seed_panels(effect_rows, summary["effects"], (*base_fields[:-1], "factors", "order"),
                      config["seeds"], checks, "effect_summary")
    verify_seed_panels(random_rows, summary["random_pairs"], cell_fields, config["seeds"], checks, "random_summary")
    return {"cell_panels": len(summary["cells"]), "conditional_edges": len(edge_rows),
            "factor_effects": len(effect_rows), "random_pairs": len(random_rows)}


def audit_record(record, hard, skip, pooling):
    record, hard, skip, pooling = (Path(path).resolve() for path in (record, hard, skip, pooling))
    checks = Checks()
    manifests = {"record": verify_manifest(record, checks, "record")}
    references = read_json(record / "source_reference.json")
    checks.check("all_eight_prior_records_preserved_in_source_provenance",
                 set(references["records"]) == set(PRIOR_RECORDS))
    for name, reference in references["records"].items():
        current = verify_manifest(Path(reference["path"]), checks, "prior_" + name)
        checks.check("all_prior_record_counts_and_manifest_hashes_match_start_reference",
                     current["files"] == reference["files"]
                     and current["manifest_sha256"] == reference["manifest_sha256"], name)
        manifests[name] = current
    checks.check("all1260_prior_manifest_files_unchanged",
                 sum(row["files"] for name, row in manifests.items() if name != "record") == 1260)
    config, environment = read_json(record / "config.json"), read_json(record / "environment.json")
    panel = record / "panel"
    checks.check("panel_and_archive_configs_identical", config == read_json(panel / "panel_config.json"))
    for field, expected in (("hard_source", hard), ("skip_source", skip), ("pooling_source", pooling)):
        checks.check("explicit_source_locations_match_provenance", Path(references[field]).resolve() == expected, field)
    panel_references = read_json(panel / "source_reference.json")["records"]
    checks.check("panel_input_provenance_exact_parent_records",
                 set(panel_references) == {"hard_v2", "skip_v5", "pooling_v6"}
                 and all(panel_references[name] == references["records"][name] for name in panel_references))
    torch.set_num_threads(config["cpu_threads"])
    executed = verify_executed_source(ROOT, environment, checks)
    checks.check("predeclared_protocol_exact_recorded_hash",
                 digest(ROOT / "configs/READOUT_V7_PROTOCOL.md") == environment["protocol_sha256"])
    fixed = {"hidden_dim": 64, "layers": 8, "shallow_layer": 1, "batch_size": 128,
             "learning_rate": .0015, "weight_decay": .0001, "gradient_clip": 5.,
             "alpha": 1., "order_seed_offset": 1701,
             "experiences": ["a", "b"], "regimes": ["end_to_end", "frozen_readout"],
             "scopes": list(SCOPES), "normalizations": list(NORMALIZATIONS), "objectives": list(OBJECTIVES)}
    checks.check("architecture_factors_and_optimization_match_predeclared_protocol",
                 all(config[field] == value for field, value in fixed.items()))
    checks.check("full_primary_panel_or_explicit_pipeline_only",
                 environment["pipeline_only"] or (config["epochs"] == 120 and config["seeds"] == [0, 1, 2]))
    checks.check("no_encoder_updates_new_labels_or_target_boundaries_and_old_records_unchanged",
                 environment["encoder_updates"] == environment["new_combination_labels"] == 0
                 and environment["target_membership_supplied"] is False
                 and environment["previous_records_unchanged"] is True
                 and bool(environment.get("finished_utc")))
    checks.check("deterministic_single_thread_BLAS_recorded",
                 environment["blas_threads"] == {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    records = [json.loads(line) for line in (hard / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = read_json(hard / "splits.json")
    checks.check("molecule_identities_unique", len(records) == len({row["id"] for row in records}))
    plan, orders = audit_plan(panel, skip, records, splits, config, checks)
    encoders, states, features, indices = audit_encoders(panel, hard, skip, pooling, records, splits, config, checks)
    metric_rows, counts = audit_fits(panel, hard, skip, pooling, encoders, states, features,
                                    indices, records, splits, config, plan, orders, checks)
    counts.update(audit_statistics(record, metric_rows, config, checks))
    panel_counts = environment["panel"]
    checks.check("coordinator_counts_match_independent_reconstruction",
                 panel_counts["encoder_count"] == counts["encoder_count"]
                 and panel_counts["fit_count"] == counts["fit_count"]
                 and panel_counts["mlp_fit_count"] == counts["MLP_saved_states_evaluated"]
                 and panel_counts["ridge_fit_count"] == counts["ridge_fits_independently_resolved"]
                 and panel_counts["metric_rows"] == counts["source_metric_rows"]
                 and panel_counts["gradient_clipping_rows"] == counts["clipping_rows"]
                 and panel_counts["continuity_checks"] == counts["continuity_anchors"])
    return {"status": "PASS" if checks.success else "FAIL",
        "audited_utc": datetime.now(timezone.utc).isoformat(), "record": str(record),
        "auditor_sha256": digest(Path(__file__)),
        "audit_helper_sha256": {name: digest(ROOT / "maintenance" / name)
                                for name in ("audit_mechanism_v3.py", "audit_skip_v5.py")},
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "method": "Independent frozen-source identity/hash checks and selected functional encoder reconstruction; every scaler reconstructed from original source labels; every MLP head prediction recomputed; every ridge refit; all metrics, conditional and averaged factorial contrasts, interactions, random pairs, and sample-SD summaries independently recomputed; every gradient-clip record checked and first-step norms reconstructed; exact v5/v6 anchors; all input/output manifests.",
        "limits": "Stored states and logs cannot observe every historical optimizer action or exclude external selection. CPU features and logits use fixed numerical tolerances across kernels, while all class predictions and continuity anchors are exact. Independent frozen task heads limit objective attribution to global clipping and numerical execution. Conditional contrasts do not uniquely decompose historical v5-to-v6 changes, and three seeds do not establish general population-level effects.",
        "manifests": manifests, "executed_source_provenance": executed,
        "counts": counts, "checks": checks.results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/readout_v7")
    parser.add_argument("--hard", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/skip_v5")
    parser.add_argument("--pooling", type=Path, default=ROOT / "experiments/pooling_v6")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/READOUT_V7_AUDIT.json")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit record: {output}")
    protected = [args.record, args.hard, args.source, args.pooling,
                 *(args.hard.parent / name for name in PRIOR_RECORDS)]
    if any(output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("The audit output must remain outside every immutable scientific record.")
    result = audit_record(args.record, args.hard, args.source, args.pooling)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
