"""Independently audit frozen branch sensitivity and pooling diagnostics.

This auditor imports no experiment implementation. It checks immutable records,
reconstructs heads and selected layer-one message passing on the CPU, refits
fixed ridge classifiers, and derives metrics from stored per-molecule outputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
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
from maintenance.audit_mechanism_v3 import Checks, digest, encoder_bytes, encoder_digest, verify_executed_source, verify_manifest

CASE_FIELDS = ("regime", "variant", "experience", "seed")
TASKS = ("family", "position", "distance")
INTERVENTIONS = ("intact", "zero_branch", "permuted_branch")
TEST_SPLITS = ("iid_a", "iid_b", "ood_common")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, tolerance=1e-11):
    return math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance)


def tensor_digest(state):
    checksum = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        checksum.update(name.encode("utf-8"))
        checksum.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def case_tuple(row):
    return tuple(int(row[field]) if field == "seed" else row[field] for field in CASE_FIELDS)


def case_name(case):
    return f"{case[0]}_{case[1]}_{case[2]}_seed{case[3]}"


def classification_scores(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    correct = truth == prediction
    return {"accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean([correct[truth == value].mean() for value in np.unique(truth)]))}


def branch_effects(intact, altered, truth):
    """Independently derive effects, retaining wrong-to-wrong label changes."""
    intact, altered = np.asarray(intact, dtype=np.float64), np.asarray(altered, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if (intact.ndim != 2 or intact.shape != altered.shape or intact.shape[1] < 2
            or truth.shape != (len(intact),) or np.any(truth < 0) or np.any(truth >= intact.shape[1])
            or not np.isfinite(intact).all() or not np.isfinite(altered).all()):
        raise ValueError("Finite paired class logits and valid labels are required.")
    row = np.arange(len(truth))
    margin = []
    for logits in (intact, altered):
        competitors = logits.copy()
        competitors[row, truth] = -np.inf
        margin.append(logits[row, truth] - competitors.max(1))
    initial, final = intact.argmax(1), altered.argmax(1)
    initial_correct, final_correct = initial == truth, final == truth
    changed = initial != final
    delta = altered - intact
    centered_delta = delta - delta.mean(1, keepdims=True)
    centered_initial = intact - intact.mean(1, keepdims=True)
    centered_l2 = np.linalg.norm(centered_delta, axis=1)
    return {
        "truth": truth, "intact_prediction": initial, "altered_prediction": final,
        "intact_correct": initial_correct, "altered_correct": final_correct,
        "prediction_changed": changed,
        "wrong_to_correct": ~initial_correct & final_correct,
        "correct_to_wrong": initial_correct & ~final_correct,
        "both_wrong_changed": ~initial_correct & ~final_correct & changed,
        "margin_intact": margin[0], "margin_altered": margin[1],
        "margin_delta": margin[1] - margin[0], "absolute_margin_delta": np.abs(margin[1] - margin[0]),
        "raw_logit_l2": np.linalg.norm(delta, axis=1), "centered_logit_l2": centered_l2,
        "relative_centered_logit_l2": centered_l2 / np.maximum(np.linalg.norm(centered_initial, axis=1), 1e-12),
        "max_abs_logit_delta": np.abs(delta).max(1),
        "logits_exactly_unchanged": np.all(delta == 0, axis=1),
        "centered_logits_exactly_unchanged": np.all(centered_delta == 0, axis=1),
        "margin_improved": margin[1] > margin[0], "margin_worsened": margin[1] < margin[0],
        "margin_exactly_unchanged": margin[1] == margin[0],
        "margin_negative_to_positive": (margin[0] < 0) & (margin[1] > 0),
        "margin_positive_to_negative": (margin[0] > 0) & (margin[1] < 0),
        "intact_margin_tie": margin[0] == 0, "altered_margin_tie": margin[1] == 0,
    }


def branch_summary(effects, selected):
    selected = np.asarray(selected, dtype=np.int64)
    if not len(selected):
        raise ValueError("Cannot summarize an empty split.")
    result = {"n": len(selected)}
    for field, values in effects.items():
        if field in ("truth", "intact_prediction", "altered_prediction"):
            continue
        values = np.asarray(values)[selected]
        if values.dtype == bool:
            result[field + "_count"] = int(values.sum())
            result[field + "_rate"] = float(values.mean())
        else:
            result[field + "_mean"] = float(values.mean())
            result[field + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            for label, probability in (("q05", .05), ("q25", .25), ("q50", .5), ("q75", .75), ("q95", .95)):
                result[field + "_" + label] = float(np.quantile(values, probability))
    return result


@torch.no_grad()
def head_logits(state, readouts, task, batch_size=128):
    """Reconstruct the saved v5 head, preserving its fixed inference shape."""
    values = []
    for offset in range(0, len(readouts), batch_size):
        batch = torch.as_tensor(readouts[offset:offset + batch_size], dtype=torch.float32)
        genuine = len(batch)
        batch = batch[torch.arange(batch_size) % genuine]
        first = F.linear(batch, state[f"heads.{task}.0.weight"], state[f"heads.{task}.0.bias"])
        logits = F.linear(F.relu(first), state[f"heads.{task}.2.weight"], state[f"heads.{task}.2.bias"])
        values.append(logits[:genuine].numpy())
    return np.concatenate(values)


def audit_branch(record, source, hard, config, records, splits, checks):
    directory = record / "branch"
    inventory = [row for row in read_csv(source / "inventory.csv") if row["variant"] != "deep"
                 and int(row["seed"]) in config["seeds"]]
    plan = read_json(source / "plan.json")
    analysis = splits["analysis"]
    lookup = {index: row for row, index in enumerate(analysis)}
    summary_lookup = {}
    for row in read_csv(directory / "branch_summary.csv"):
        key = (*case_tuple(row), row["split"], row["task"], row["intervention"])
        checks.check("branch_summary_keys_unique", key not in summary_lookup, key)
        summary_lookup[key] = row
    frozen_predictions = {}
    for row in read_csv(source / "predictions.csv"):
        if row["variant"] != "deep":
            frozen_predictions[(*case_tuple(row), row["intervention"], int(row["index"]))] = row
    effects_lookup = {}
    expected_summaries = {}
    raw_logit_mismatch_count = 0
    archived_prediction_mismatch_count = 0
    with np.load(directory / "representations/branch_logits.npz") as archive:
        checks.check("branch_logits_exact_analysis_identity_order", archive["indices"].tolist() == analysis
                     and archive["ids"].tolist() == [records[i]["id"] for i in analysis])
        for metadata in inventory:
            case = case_tuple(metadata)
            name = case_name(case)
            state = torch.load(source / f"checkpoints/{name}.pt", map_location="cpu", weights_only=True)["state_dict"]
            with np.load(source / f"representations/{name}.npz") as cached:
                readout = cached["readout"].copy()
            intact = {}
            for intervention in INTERVENTIONS:
                values = readout.copy()
                if intervention == "zero_branch":
                    values[:, 448:] = 0
                elif intervention == "permuted_branch":
                    values[:, 448:] = readout[plan["branch_permutations"][str(case[3])]["indices"], 448:]
                for task in TASKS:
                    logits = archive[f"{name}__{intervention}__{task}"]
                    actual = head_logits(state, values, task)
                    mismatch = not np.array_equal(actual, logits)
                    raw_logit_mismatch_count += mismatch
                    checks.check("every_archived_branch_logit_is_independently_recomputed", not mismatch,
                                 (case, intervention, task, float(np.max(np.abs(actual - logits)))))
                    predictions = logits.argmax(1)
                    for row, index in enumerate(analysis):
                        previous = frozen_predictions[(*case, intervention, index)]
                        archived_prediction_mismatch_count += int(predictions[row] != int(previous[f"pred_{task}"]))
                    if intervention == "intact":
                        intact[task] = logits.copy()
                    else:
                        truth = np.array([records[i][f"y_{task}"] for i in analysis])
                        effects = branch_effects(intact[task], logits, truth)
                        effects_lookup[(*case, task, intervention)] = effects
                        for split in (*TEST_SPLITS, "analysis"):
                            selected = list(range(len(analysis))) if split == "analysis" else [lookup[i] for i in splits[split]]
                            expected = branch_summary(effects, selected)
                            key = (*case, split, task, intervention)
                            expected_summaries[key] = expected
                            saved = summary_lookup.get(key)
                            checks.check("branch_summary_all_metrics_from_independent_logit_effects",
                                         saved is not None and all(close(saved[field], value) for field, value in expected.items()), key)
        expected_array_count = 2 + len(inventory) * 3 * 3
        checks.check("branch_logit_array_panel_exact", len(archive.files) == expected_array_count)
    checks.check("all_branch_predictions_match_frozen_v5_per_molecule_results", archived_prediction_mismatch_count == 0,
                 archived_prediction_mismatch_count)
    checks.check("branch_summary_panel_exact", set(summary_lookup) == set(expected_summaries))
    row_keys = set()
    row_errors = []
    with (directory / "branch_rows.csv").open(encoding="utf-8", newline="") as stream:
        for saved in csv.DictReader(stream):
            case = case_tuple(saved)
            index = int(saved["index"])
            key = (*case, saved["task"], saved["intervention"])
            row_key = (*key, index)
            duplicate = row_key in row_keys
            row_keys.add(row_key)
            effect = effects_lookup.get(key)
            position = lookup.get(index)
            correct = not duplicate and effect is not None and position is not None
            if correct:
                expected_split = next(split for split in TEST_SPLITS if index in splits[split])
                correct = saved["id"] == records[index]["id"] and saved["split"] == expected_split
                for field, array in effect.items():
                    value = array[position]
                    correct = correct and ((saved[field].lower() == str(bool(value)).lower()) if array.dtype == bool
                                            else close(saved[field], value))
            if not correct and len(row_errors) < 16:
                row_errors.append(row_key)
    expected_rows = len(inventory) * len(analysis) * 3 * 2
    checks.check("all_branch_per_example_values_and_identity_keys_recomputed",
                 not row_errors and len(row_keys) == expected_rows, row_errors)
    aggregate = read_csv(directory / "branch_seed_aggregate.csv")
    groups = defaultdict(dict)
    for key, summary in expected_summaries.items():
        regime, variant, experience, seed, split, task, intervention = key
        groups[(regime, variant, experience, split, task, intervention)][seed] = summary
    aggregate_keys = set()
    for saved in aggregate:
        key = tuple(saved[field] for field in ("regime", "variant", "experience", "split", "task", "intervention"))
        aggregate_keys.add(key)
        expected = groups.get(key, {})
        valid = len(expected) == len(config["seeds"]) and int(saved["n_seeds"]) == len(expected)
        if valid:
            valid = int(saved["n_molecules_per_seed"]) == next(iter(expected.values()))["n"]
            for field in next(iter(expected.values())):
                if field == "n" or field.endswith("_count"):
                    continue
                values = [row[field] for row in expected.values()]
                valid = valid and close(saved[field + "_seed_mean"], statistics.mean(values))
                valid = valid and close(saved[field + "_seed_sd"], statistics.stdev(values) if len(values) > 1 else 0)
        checks.check("branch_seed_aggregates_use_three_seed_units_and_sample_SD", valid, key)
    checks.check("branch_seed_aggregate_panel_exact", aggregate_keys == set(groups) and len(aggregate) == len(groups))
    reference = read_json(directory / "source_reference.json")
    checks.check("branch_pre_and_post_input_source_hashes_current",
                 all(Path(path).is_file() and digest(Path(path)) == expected for path, expected in reference["input_sha256"].items()))
    return {"branch_model_cases": len(inventory), "branch_logit_arrays": expected_array_count - 2,
            "branch_per_example_comparisons": len(row_keys), "branch_summary_rows": len(summary_lookup),
            "branch_logit_mismatches": raw_logit_mismatch_count,
            "branch_v5_prediction_mismatches": archived_prediction_mismatch_count}


def independent_ridge(train_x, labels):
    """Solve the declared support-only standardized eight-output linear fit."""
    values = np.asarray(train_x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (values.ndim != 2 or not len(values) or labels.shape != (len(values),)
            or not np.isfinite(values).all() or np.any((labels < 0) | (labels >= 8))):
        raise ValueError("Finite fit features and eight-class labels are required.")
    mean = values.mean(0)
    scale = values.std(0)
    scale[scale <= np.finfo(np.float64).eps] = 1
    z = (values - mean) / scale
    targets = np.eye(8, dtype=np.float64)[labels]
    bias = targets.mean(0)
    covariance = z.T @ z
    covariance.flat[::len(covariance) + 1] += 1
    weights = np.linalg.solve(covariance, z.T @ (targets - bias))
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


@torch.no_grad()
def first_layer_pools(state, x, adjacency, mask, target):
    """Independently evaluate one message layer and two matched-width means."""
    h = F.relu(F.linear(x, state["embedding.weight"], state["embedding.bias"])) * mask.unsqueeze(-1)
    neighbor = torch.bmm(adjacency, h)
    updated = F.linear(F.relu(F.linear(torch.cat((h, neighbor), dim=-1),
                                      state["layers.0.update.0.weight"], state["layers.0.update.0.bias"])),
                       state["layers.0.update.2.weight"], state["layers.0.update.2.bias"])
    h = F.relu(F.layer_norm(h + updated, (h.shape[-1],), state["layers.0.norm.weight"],
                           state["layers.0.norm.bias"], eps=1e-5)) * mask.unsqueeze(-1)
    result = {}
    for name, region in (("whole_graph", mask), ("target", target)):
        region = region.double()
        if torch.any(region.sum(1) <= 0) or torch.any(region.bool() & ~mask.bool()):
            raise ValueError("Pooling regions must be nonempty and exclude padding.")
        result[name] = ((h.double() * region.unsqueeze(-1)).sum(1)
                        / region.sum(1, keepdim=True)).float().numpy()
    return result


@torch.no_grad()
def mlp_predictions(state, values, mean, scale):
    z = ((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32)
    logits = []
    for offset in range(0, len(z), 128):
        current = z[offset:offset + 128]
        n = len(current)
        tensor = torch.from_numpy(current[np.arange(128) % n])
        hidden = F.relu(F.linear(tensor, state["0.weight"], state["0.bias"]))
        logits.append(F.linear(hidden, state["2.weight"], state["2.bias"])[:n].numpy())
    return np.concatenate(logits)


def check_identity_entry(entry, expected, records):
    return entry["indices"] == expected and entry["ids"] == [records[index]["id"] for index in expected]


def audit_pooling_plan(directory, source, scale, records, splits, config, checks):
    plan = read_json(directory / "pooling_plan.json")
    for experience in ("a", "b"):
        for name, original in (("train", f"train_{experience}"),
                               ("validation_not_used_for_selection", f"val_{experience}")):
            entry = plan["source_only"][experience][name]
            expected = splits[original]
            checks.check("pooling_source_train_validation_exact_original_order",
                         check_identity_entry(entry, expected, records), (experience, name))
            checks.check("pooling_source_train_validation_excludes_heldout_combinations_and_reserve",
                         not set(entry["indices"]) & set(splits["analysis"] + splits["adaptation_pool"])
                         and all((records[index]["y_family"] + records[index]["y_position"]) % 3 != 2
                                 for index in entry["indices"]), (experience, name))
    for name in (*TEST_SPLITS, "analysis"):
        checks.check("pooling_tests_exact_original_order", check_identity_entry(plan["tests"][name], splits[name], records), name)
    direct = [row for row in read_json(scale / "direct_support.json")["sets"] if row["repeat"] in config["support_repeats"]]
    checks.check("direct_support_is_exact_archived_scale_v4_labels_and_order", plan["direct_support"] == direct)
    for row in direct:
        indices = row["indices"]
        valid = len(indices) == 256 and len(set(indices)) == 256 and set(indices) <= set(splits["analysis"])
        valid = valid and row["ids"] == [records[index]["id"] for index in indices]
        valid = valid and all(records[index]["y_position"] == row["source_position"] for index in indices)
        valid = valid and all(sum(records[index]["y_family"] == label for index in indices) == 32 for label in range(8))
        checks.check("direct_support_balanced256_source_position_only", valid, (row["repeat"], row["source_position"]))
    with np.load(directory / "pooling_training_orders.npz") as current, np.load(source / "training_orders.npz") as original:
        expected_keys = {f"{experience}_seed{seed}" for experience in ("a", "b") for seed in config["seeds"]}
        checks.check("pooling_epoch_order_keys_exact", set(current.files) == expected_keys)
        for key in expected_keys:
            experience, seed_text = key.rsplit("_seed", 1)
            seed = int(seed_text)
            values = current[key]
            checks.check("pooling_orders_exact_v5_prefix", np.array_equal(values, original[key][:config["epochs"]]), key)
            rng = np.random.default_rng(1701 + seed)
            expected = np.asarray([rng.permutation(splits[f"train_{experience}"]) for _ in range(config["epochs"])], dtype=np.int32)
            checks.check("pooling_orders_independently_reconstructed_from_original_curriculum",
                         values.dtype == np.int32 and np.array_equal(values, expected), key)
    return plan


def audit_pooling(record, source, hard, scale, config, records, splits, checks):
    directory = record / "pooling"
    plan = audit_pooling_plan(directory, source, scale, records, splits, config, checks)
    encoder_rows = read_csv(directory / "pooling_encoders.csv")
    encoders = {row["model"]: row for row in encoder_rows}
    expected_encoders = {f"{regime}_shallow_skip_{experience}_seed{seed}"
                         for regime in ("end_to_end", "frozen_readout") for experience in ("a", "b") for seed in config["seeds"]}
    expected_encoders.update(f"random_seed{seed}" for seed in config["seeds"])
    checks.check("pooling_exact_unique_encoder_panel", len(encoders) == len(encoder_rows) and set(encoders) == expected_encoders)
    feature_indices = sorted(set(splits["train_a"] + splits["train_b"] + splits["val_a"] + splits["val_b"] + splits["analysis"]))
    lookup = {index: row for row, index in enumerate(feature_indices)}
    analysis_rows = [lookup[index] for index in splits["analysis"]]
    feature_cache = {}
    sample_positions = [0, len(feature_indices) // 3, len(feature_indices) // 2, len(feature_indices) - 1]
    sample_indices = [feature_indices[position] for position in sample_positions]
    with np.load(hard / "graphs.npz") as archive:
        x, adjacency, mask = (torch.from_numpy(archive[key][sample_indices].copy()) for key in ("x", "adj", "mask"))
    membership = np.zeros(tuple(mask.shape), dtype=bool)
    for row, index in enumerate(sample_indices):
        target = records[index]["target_fragment_indices"]
        checks.check("oracle_target_membership_has_nine_real_atoms_and_root", len(target) == 9
                     and len(set(target)) == 9 and records[index]["target_attachment_idx"] in target
                     and mask[row, target].bool().all().item(), records[index]["id"])
        membership[row, target] = True
    for name, inventory in encoders.items():
        is_random = name.startswith("random_")
        expected_checkpoint = (hard if is_random else source) / f"checkpoints/{name}.pt"
        checks.check("pooling_encoder_checkpoint_path_and_hash_exact_archived_source",
                     Path(inventory["checkpoint"]).resolve() == expected_checkpoint.resolve()
                     and inventory["checkpoint_sha256"] == digest(expected_checkpoint), name)
        state = torch.load(expected_checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        expected_hash = encoder_digest(state)
        checks.check("frozen_encoder_tensor_hashes_before_after_equal_exact_source",
                     inventory["encoder_before_sha256"] == inventory["encoder_after_sha256"] == expected_hash
                     and inventory["encoder_bitwise_unchanged"].lower() == "true", name)
        if inventory["regime"] == "frozen_readout":
            previous = torch.load(hard / f"checkpoints/{inventory['experience']}_seed{inventory['seed']}.pt",
                                  map_location="cpu", weights_only=True)["state_dict"]
            checks.check("frozen_readout_encoder_is_exact_original_pretrained_encoder",
                         encoder_bytes(state) == encoder_bytes(previous), name)
        feature_path = directory / inventory["features"]
        checks.check("pooling_features_hash_and_row_count_match_inventory",
                     digest(feature_path) == inventory["features_sha256"] and int(inventory["n_features"]) == len(feature_indices), name)
        with np.load(feature_path) as cached:
            checks.check("pooling_features_use_exact_train_val_analysis_union_without_reserve",
                         cached["indices"].tolist() == feature_indices and cached["ids"].tolist() == [records[i]["id"] for i in feature_indices], name)
            features = {pool: cached[pool].copy() for pool in ("whole_graph", "target")}
        feature_cache[name] = features
        checks.check("both_pooling_feature_matrices_are_finite_float32_width64",
                     all(values.shape == (len(feature_indices), 64) and values.dtype == np.float32 and np.isfinite(values).all()
                         for values in features.values()), name)
        independent = first_layer_pools(state, x, adjacency, mask, torch.from_numpy(membership))
        for pool, values in independent.items():
            delta = float(np.max(np.abs(values - features[pool][sample_positions])))
            checks.check("fixed_sample_layer1_pools_independently_reconstructed_on_CPU",
                         np.allclose(values, features[pool][sample_positions], rtol=3e-5, atol=3e-5), (name, pool, delta))
        if not is_random:
            with np.load(source / f"representations/{name}.npz") as archived:
                checks.check("whole_graph_probe_is_bitexact_actual_v5_classification_branch",
                             np.array_equal(features["whole_graph"][analysis_rows], archived["readout"][:, 448:]), name)
            if inventory["regime"] == "frozen_readout":
                with np.load(source / f"representations/frozen_{inventory['experience']}_seed{inventory['seed']}.npz") as archived:
                    selected = [lookup[index] for index in archived["indices"].tolist()]
                    checks.check("whole_graph_frozen_source_cache_is_bitexact_actual_v5_cache",
                                 np.array_equal(features["whole_graph"][selected], archived["shallow_mean"]), name)
    return _audit_pooling_fits(directory, source, config, records, splits, plan, encoders, feature_indices,
                               feature_cache, sample_positions, checks)


def _audit_pooling_fits(directory, source, config, records, splits, plan, encoders, feature_indices,
                        feature_cache, sample_positions, checks):
    fits = read_csv(directory / "pooling_fit_inventory.csv")
    fit_lookup = {row["fit_key"]: row for row in fits}
    support_lookup = {row["fit_key"]: row for row in plan["fits"]}
    checks.check("pooling_fit_inventory_and_support_keys_unique_and_equal", len(fit_lookup) == len(fits)
                 and len(support_lookup) == len(plan["fits"]) and set(fit_lookup) == set(support_lookup))
    feature_lookup = {index: row for row, index in enumerate(feature_indices)}
    expected_metrics = {"source_only": {}, "direct_diagnostic": {}}
    initializers = {}
    ridge_count, mlp_count, head_mismatches = 0, 0, 0
    with np.load(directory / "pooling_training_orders.npz") as archive:
        orders = {key: archive[key].copy() for key in archive.files}
    source_fit_keys, direct_fit_keys = set(), set()
    for key, inventory in fit_lookup.items():
        model, experience, pool = inventory["model"], inventory["experience"], inventory["pool"]
        classifier, protocol = inventory["classifier"], inventory["protocol"]
        seed = int(inventory["seed"])
        encoder = encoders.get(model, {})
        checks.check("fit_metadata_identifies_the_exact_encoder_and_fixed_readout_dimensions",
                     bool(encoder) and inventory["condition"] == encoder["condition"]
                     and inventory["regime"] == encoder["regime"] and int(encoder["seed"]) == seed
                     and pool in ("whole_graph", "target") and classifier in ("ridge", "mlp")
                     and protocol in ("source_only", "direct_diagnostic") and int(inventory["feature_dim"]) == 64
                     and int(inventory["classifier_parameter_count"]) == (4680 if classifier == "mlp" else 520), key)
        support = support_lookup[key]["support"]
        indices = support["indices"]
        labels = np.array([records[index]["y_family"] for index in indices], dtype=np.int64)
        checks.check("every_fit_support_id_and_label_hash_exact", support["ids"] == [records[i]["id"] for i in indices]
                     and support_lookup[key]["protocol"] == protocol
                     and int(inventory["n_fit"]) == len(indices)
                     and inventory["support_ids_sha256"] == hashlib.sha256("\n".join(support["ids"]).encode()).hexdigest()
                     and inventory["support_labels_sha256"] == hashlib.sha256(labels.tobytes()).hexdigest(), key)
        if protocol == "source_only":
            expected_indices = splits[f"train_{experience}"]
            checks.check("every_source_only_fit_uses_exact_original_curriculum_and_no_reserved_labels",
                         indices == expected_indices and not set(indices) & set(splits["analysis"] + splits["adaptation_pool"]), key)
            source_fit_keys.add((model, experience, pool, classifier))
            selected = feature_indices
        else:
            sample = next(row for row in plan["direct_support"] if row["repeat"] == int(inventory["repeat"])
                          and row["source_position"] == int(inventory["source_position"]))
            checks.check("every_direct_fit_uses_exact_archived256_support_only", indices == sample["indices"]
                         and classifier == "ridge" and experience == "shared", key)
            direct_fit_keys.add((model, pool, int(inventory["repeat"]), int(inventory["source_position"])))
            selected = splits["analysis"]
        values = feature_cache[model][pool]
        train_values = values[[feature_lookup[index] for index in indices]]
        eval_values = values[[feature_lookup[index] for index in selected]]
        mean = train_values.astype(np.float64).mean(0)
        scale = train_values.astype(np.float64).std(0)
        scale[scale <= np.finfo(np.float64).eps] = 1
        checkpoint_path, prediction_path = directory / inventory["checkpoint"], directory / inventory["predictions"]
        checks.check("every_readout_state_and_prediction_file_hash_matches_inventory",
                     digest(checkpoint_path) == inventory["checkpoint_sha256"]
                     and digest(prediction_path) == inventory["predictions_sha256"], key)
        with np.load(prediction_path) as archive:
            logits, predictions = archive["logits"].copy(), archive["prediction"].copy()
            checks.check("readout_predictions_have_exact_expected_identity_order",
                         archive["indices"].tolist() == selected and archive["ids"].tolist() == [records[i]["id"] for i in selected], key)
        checks.check("every_readout_prediction_is_argmax_of_finite_eight_class_logits",
                     logits.shape == (len(selected), 8) and np.isfinite(logits).all()
                     and np.array_equal(predictions, logits.argmax(1)), key)
        if classifier == "ridge":
            ridge_count += 1
            expected = independent_ridge(train_values, labels)
            with np.load(checkpoint_path) as state:
                checks.check("all_ridge_parameters_independently_refit_from_support_only",
                             set(state.files) == set(expected)
                             and all(np.array_equal(state[field], value) for field, value in expected.items()), key)
            independent_logits = ((eval_values.astype(np.float64) - expected["mean"]) / expected["scale"]) @ expected["weights"] + expected["bias"]
            checks.check("all_ridge_logits_and_argmax_predictions_independently_recomputed",
                         np.array_equal(logits, independent_logits) and np.array_equal(predictions, independent_logits.argmax(1)), key)
            checks.check("ridge_has_no_optimizer_updates", int(inventory["optimizer_updates"]) == 0, key)
        else:
            mlp_count += 1
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checks.check("mlp_standardization_uses_only_original_fit_support",
                         np.array_equal(state["mean"], mean) and np.array_equal(state["scale"], scale), key)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(61000 + seed)
                first, second = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)
                expected_initial = {"0.weight": first.weight, "0.bias": first.bias,
                                    "2.weight": second.weight, "2.bias": second.bias}
            checks.check("mlp_initializer_exact_seed_matched_across_scopes_and_conditions",
                         tensor_digest(expected_initial) == tensor_digest(state["initial_state_dict"])
                         == inventory["initial_head_sha256"], key)
            initializers.setdefault(seed, set()).add(inventory["initial_head_sha256"])
            order = orders[f"{experience}_seed{seed}"]
            expected_updates = config["epochs"] * math.ceil(1386 / 128)
            checks.check("mlp_fixed_epoch_order_hash_and_optimizer_budget_match_original",
                         inventory["training_order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest()
                         and int(inventory["optimizer_updates"]) == state["optimizer_updates"] == expected_updates
                         and state["history"][-1]["epoch"] == config["epochs"]
                         and all(set(row) == {"epoch", "train_loss"} for row in state["history"]), key)
            checks.check("mlp_final_head_digest_matches_inventory", tensor_digest(state["state_dict"]) == inventory["final_head_sha256"], key)
            actual = mlp_predictions(state["state_dict"], eval_values, mean, scale)
            mismatches = int(np.count_nonzero(actual.argmax(1) != predictions))
            head_mismatches += mismatches
            checks.check("every_MLP_saved_head_prediction_independently_recomputed_on_CPU", mismatches == 0, (key, mismatches))
            checks.check("MLP_saved_logits_match_independent_CPU_function_with_declared_tolerance",
                         np.allclose(actual, logits, rtol=3e-5, atol=3e-5), (key, float(np.max(np.abs(actual - logits)))))
        selected_lookup = {index: row for row, index in enumerate(selected)}
        if protocol == "source_only":
            for split in ("train", *TEST_SPLITS):
                test = splits[f"train_{experience}"] if split == "train" else splits[split]
                truth = np.array([records[index]["y_family"] for index in test])
                expected_metrics[protocol][(key, split)] = {"n_test": len(test), **classification_scores(truth, predictions[[selected_lookup[i] for i in test]])}
        else:
            experiences = ("a", "b") if inventory["condition"] == "random" else (encoders[model]["experience"],)
            for scored_experience in experiences:
                for target_position in range(3):
                    if target_position == int(inventory["source_position"]):
                        continue
                    target = [i for i in splits["analysis"] if records[i]["y_position"] == target_position]
                    own, common = set(splits[f"iid_{scored_experience}"]), set(splits["ood_common"])
                    for split, test in (("target_all", target), ("target_common_ood", [i for i in target if i in common]),
                                        ("target_own_iid", [i for i in target if i in own])):
                        checks.check("direct_target_position_tests_have_no_support_identity_overlap", not set(test) & set(indices), (key, scored_experience, target_position, split))
                        truth = np.array([records[index]["y_family"] for index in test])
                        expected_metrics[protocol][(key, scored_experience, target_position, split)] = {
                            "n_test": len(test), **classification_scores(truth, predictions[[selected_lookup[i] for i in test]])}
    expected_source = {(model, experience, pool, classifier)
                       for model, encoder in encoders.items()
                       for experience in (("a", "b") if encoder["condition"] == "random" else (encoder["experience"],))
                       for pool in ("whole_graph", "target") for classifier in ("ridge", "mlp")}
    expected_direct = {(model, pool, repeat, position) for model in encoders for pool in ("whole_graph", "target")
                       for repeat in config["support_repeats"] for position in range(3)}
    checks.check("complete_source_readout_and_unique_direct_fit_panels_without_random_duplicates",
                 source_fit_keys == expected_source and direct_fit_keys == expected_direct
                 and len(fits) == len(expected_source) + len(expected_direct))
    checks.check("one_exact_MLP_initializer_per_seed_across_all_conditions", all(len(values) == 1 for values in initializers.values()))
    for protocol, filename in (("source_only", "pooling_source_metrics.csv"), ("direct_diagnostic", "pooling_direct_metrics.csv")):
        saved_rows = read_csv(directory / filename)
        seen = set()
        for saved in saved_rows:
            key = ((saved["fit_key"], saved["split"]) if protocol == "source_only" else
                   (saved["fit_key"], saved["experience"], int(saved["target_position"]), saved["split"]))
            expected = expected_metrics[protocol].get(key)
            valid = key not in seen and expected is not None and all(close(saved[field], value) for field, value in expected.items())
            fit = fit_lookup.get(saved["fit_key"], {})
            metadata_fields = ("model", "condition", "regime", "seed", "pool", "classifier", "protocol", "repeat", "source_position")
            if protocol == "source_only":
                metadata_fields += ("experience",)
            valid = valid and all(saved[field] == fit.get(field) for field in metadata_fields)
            seen.add(key)
            checks.check("all_pooling_metrics_recomputed_from_per_molecule_predictions", valid, (protocol, key))
        checks.check("pooling_metric_panels_complete_without_duplicate_rows", seen == set(expected_metrics[protocol]), protocol)
    return {"pooling_unique_encoders": len(encoders), "pooling_unique_fits": len(fits),
            "ridge_fits_independently_resolved": ridge_count, "MLP_readout_states_reconstructed": mlp_count,
            "MLP_CPU_prediction_mismatches": head_mismatches, "feature_sample_positions": sample_positions,
            "source_metric_rows": len(expected_metrics["source_only"]), "direct_metric_rows": len(expected_metrics["direct_diagnostic"])}


def verify_seed_panels(rows, panels, fields, checks, label):
    groups = defaultdict(dict)
    duplicate = False
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        seed = str(row["seed"])
        duplicate = duplicate or seed in groups[key]
        groups[key][seed] = row
    checks.check(label + "_one_value_per_seed", not duplicate)
    saved = {tuple(str(panel[field]) for field in fields): panel for panel in panels}
    checks.check(label + "_panel_keys_exact", len(saved) == len(panels) and set(saved) == set(groups))
    for key, values in groups.items():
        valid = key in saved
        if valid:
            for metric in ("accuracy", "balanced_accuracy"):
                expected = {seed: float(row[metric]) for seed, row in values.items()}
                actual = saved[key][metric]
                valid = valid and actual["n_seeds"] == len(expected) and set(actual["per_seed"]) == set(expected)
                valid = valid and all(close(actual["per_seed"][seed], value) for seed, value in expected.items())
                valid = valid and close(actual["mean"], statistics.mean(expected.values()))
                valid = valid and ((actual["std"] is None) if len(expected) == 1 else close(actual["std"], statistics.stdev(expected.values())))
        checks.check(label + "_retained_seed_values_mean_sample_SD", valid, key)


def audit_statistics(record, checks):
    summary = read_json(record / "report_summary.json")
    source = read_csv(record / "pooling/pooling_source_metrics.csv")
    verify_seed_panels(source, summary["source"], ("regime", "experience", "classifier", "pool", "split"),
                       checks, "source_summary")
    grouped = defaultdict(dict)
    pair_fields = ("regime", "experience", "classifier", "split", "seed")
    for row in source:
        grouped[tuple(row[field] for field in pair_fields)][row["pool"]] = row
    expected_pairs = []
    for key, pools in grouped.items():
        expected_pairs.append({**dict(zip(pair_fields, key)), "contrast": "oracle_target_minus_whole_graph",
                               **{metric: float(pools["target"][metric]) - float(pools["whole_graph"][metric])
                                  for metric in ("accuracy", "balanced_accuracy")}})
    actual_pairs = read_csv(record / "pooling_source_paired.csv")
    lookup = {tuple(row[field] for field in pair_fields): row for row in actual_pairs}
    checks.check("paired_source_table_exact_keys_without_duplicates", len(lookup) == len(actual_pairs) == len(expected_pairs)
                 and set(lookup) == set(grouped))
    for row in expected_pairs:
        key = tuple(row[field] for field in pair_fields)
        actual = lookup.get(key)
        checks.check("source_scope_contrasts_are_paired_within_seed", actual is not None
                     and actual["contrast"] == row["contrast"]
                     and all(close(actual[metric], row[metric]) for metric in ("accuracy", "balanced_accuracy")), key)
    verify_seed_panels(expected_pairs, summary["source_pairs"], pair_fields[:-1], checks, "source_pair_summary")
    direct = read_csv(record / "pooling/pooling_direct_metrics.csv")
    direct_groups = defaultdict(list)
    fields = ("regime", "experience", "pool", "split", "seed")
    for original in direct:
        row = dict(original)
        if row["regime"] == "random" and row["split"] != "target_own_iid":
            if row["experience"] != "a":
                continue
            row["experience"] = "shared"
        direct_groups[tuple(row[field] for field in fields)].append(row)
    expected_direct = []
    for key, rows in direct_groups.items():
        directions = {(row["source_position"], row["target_position"]) for row in rows}
        repeats = {row["repeat"] for row in rows}
        checks.check("direct_summaries_average_exactly_six_directions_per_support_within_seed",
                     len(directions) == 6 and len(rows) == 6 * len(repeats), key)
        expected_direct.append({**dict(zip(fields, key)), "n_directions": 6, "n_support_draws": len(repeats),
                                **{metric: statistics.mean(float(row[metric]) for row in rows)
                                   for metric in ("accuracy", "balanced_accuracy")}})
    actual_direct = read_csv(record / "pooling_direct_seed.csv")
    saved_lookup = {tuple(row[field] for field in fields): row for row in actual_direct}
    checks.check("direct_within_seed_table_exact_keys_without_duplicate_random_encoder",
                 len(saved_lookup) == len(actual_direct) == len(expected_direct) and set(saved_lookup) == set(direct_groups))
    for row in expected_direct:
        key = tuple(row[field] for field in fields)
        actual = saved_lookup.get(key)
        checks.check("direct_within_seed_means_recomputed_from_directions_and_support_draws", actual is not None
                     and all(close(actual[field], row[field]) for field in ("accuracy", "balanced_accuracy", "n_directions", "n_support_draws")), key)
    verify_seed_panels(expected_direct, summary["direct"], fields[:-1], checks, "direct_summary")
    checks.check("report_branch_summary_is_exact_verified_seed_aggregate",
                 summary["branch"] == read_csv(record / "branch/branch_seed_aggregate.csv"))
    return {"source_paired_seed_rows": len(expected_pairs), "direct_within_seed_rows": len(expected_direct)}


def audit_membership(directory, hard, records, splits, checks):
    indices = sorted(set(splits["train_a"] + splits["train_b"] + splits["val_a"] + splits["val_b"] + splits["analysis"]))
    with np.load(hard / "graphs.npz") as graphs:
        valid = graphs["mask"][indices].astype(bool)
    membership = np.zeros_like(valid)
    for row, index in enumerate(indices):
        target = records[index]["target_fragment_indices"]
        checks.check("every_target_oracle_mask_has_exactly_nine_valid_atoms_and_attachment_root",
                     len(set(target)) == len(target) == 9 and records[index]["target_attachment_idx"] in target
                     and valid[row, target].all(), records[index]["id"])
        membership[row, target] = True
    with np.load(directory / "pooling_target_membership.npz") as archive:
        checks.check("archived_oracle_pooling_masks_reconstructed_from_frozen_structural_annotation",
                     archive["indices"].tolist() == indices and archive["ids"].tolist() == [records[index]["id"] for index in indices]
                     and archive["target_membership"].dtype == bool and np.array_equal(archive["target_membership"], membership))


def audit_record(record, hard, source, scale):
    record, hard, source, scale = map(lambda path: Path(path).resolve(), (record, hard, source, scale))
    checks = Checks()
    manifests = {"record": verify_manifest(record, checks, "record")}
    references = read_json(record / "source_reference.json")
    expected_prior = {"pilot_v1", "hard_v2", "mechanism_diagnostics_v3", "readout_replay_v3",
                      "correlation_v3", "scale_v4", "skip_v5"}
    checks.check("all_seven_completed_source_records_preserved_in_provenance", set(references["records"]) == expected_prior)
    for name, reference in references["records"].items():
        current = verify_manifest(Path(reference["path"]), checks, "prior_" + name)
        checks.check("every_prior_record_manifest_hash_and_count_matches_start_reference",
                     current["manifest_sha256"] == reference["manifest_sha256"] and current["files"] == reference["files"], name)
        manifests[name] = current
    checks.check("all536_prior_manifest_files_unchanged", sum(row["files"] for name, row in manifests.items() if name != "record") == 536)
    config, environment = read_json(record / "config.json"), read_json(record / "environment.json")
    torch.set_num_threads(config["cpu_threads"])
    executed = verify_executed_source(ROOT, environment, checks)
    protocol = ROOT / "configs/POOLING_V6_PROTOCOL.md"
    checks.check("predeclared_protocol_bytes_match_record", digest(protocol) == environment["protocol_sha256"])
    fixed = {"hidden_dim": 64, "layers": 8, "shallow_layer": 1, "batch_size": 128,
             "learning_rate": .0015, "weight_decay": .0001, "gradient_clip": 5., "alpha": 1., "mlp_seed_offset": 61000}
    checks.check("diagnostic_architecture_optimization_and_regularization_predeclared",
                 all(config[field] == value for field, value in fixed.items()))
    checks.check("full_primary_panel_or_explicit_pipeline_only", environment["pipeline_only"]
                 or (config["epochs"] == 120 and config["seeds"] == [0, 1, 2] and config["support_repeats"] == [0, 1, 2]
                     and config["regimes"] == ["end_to_end", "frozen_readout"] and config["experiences"] == ["a", "b"]))
    checks.check("no_encoder_optimization_and_all_old_records_unchanged", environment["encoder_updates"] == 0
                 and environment["previous_records_unchanged"] is True)
    checks.check("deterministic_single_thread_BLAS_recorded", environment["blas_threads"] == {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    records = [json.loads(line) for line in (hard / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = read_json(hard / "splits.json")
    audit_membership(record / "pooling", hard, records, splits, checks)
    counts = audit_branch(record, source, hard, config, records, splits, checks)
    counts.update(audit_pooling(record, source, hard, scale, config, records, splits, checks))
    counts.update(audit_statistics(record, checks))
    return {"status": "PASS" if checks.success else "FAIL", "audited_utc": datetime.now(timezone.utc).isoformat(),
            "auditor_sha256": digest(Path(__file__)), "audit_helper_sha256": digest(ROOT / "maintenance/audit_mechanism_v3.py"),
            "record": str(record), "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
            "method": "Independent CPU head/message-passing reconstruction; exact archived branch logits and v5 predictions; all per-example effects, quantiles, seed means and sample SD; full oracle-mask reconstruction; all fixed ridge fits; every saved MLP head prediction; source/support identity isolation and optimizer-order metadata; all input and output manifests.",
            "limits": "Stored states and metadata cannot independently observe every historical optimizer step or rule out external selection. CPU feature/logit comparison uses declared numerical tolerance across device kernels; all class predictions are exact. Oracle localization is supplied information. Source-only motif-head fitting differs from original joint-task training. Branch replacements can be off distribution; neither null effects nor readout success imply universal absence or causal localization.",
            "manifests": manifests, "executed_source_provenance": executed, "counts": counts, "checks": checks.results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/pooling_v6")
    parser.add_argument("--hard", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/skip_v5")
    parser.add_argument("--scale", type=Path, default=ROOT / "experiments/scale_v4")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/POOLING_V6_AUDIT.json")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit record: {output}")
    if any(output.is_relative_to(path.resolve()) for path in (args.record, args.hard, args.source, args.scale)):
        raise ValueError("The audit output must remain outside every immutable scientific record.")
    result = audit_record(args.record, args.hard, args.source, args.scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
