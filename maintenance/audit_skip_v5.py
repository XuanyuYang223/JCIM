"""Independently audit the shallow whole-graph skip intervention.

This auditor imports no experiment modules. It reconstructs identity contracts,
classifier initializations, functional message passing, prediction-derived
statistics, and a fixed sample of saved model predictions on the CPU. Recorded
metadata and final states do not prove every historical optimizer operation.
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
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from maintenance.audit_mechanism_v3 import Checks, digest, encoder_bytes, encoder_digest, verify_executed_source, verify_manifest


REGIMES = ("end_to_end", "frozen_readout")
VARIANTS = ("deep", "deep_control", "shallow_skip")
EXPERIENCES = ("a", "b")
SPLITS = ("iid_a", "iid_b", "ood_common")
METRICS = ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "position_balanced_accuracy",
           "distance_accuracy", "distance_balanced_accuracy", "distance_mae", "joint_accuracy")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, tolerance=1e-12):
    return math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance)


def tensor_state_digest(state):
    checksum = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        checksum.update(name.encode("utf-8"))
        checksum.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def initial_state(random_state, encoder_state, variant):
    """Rebuild exact seed-matched reset heads and the declared encoder source."""
    result = {}
    for name, value in random_state.items():
        if not name.startswith("heads."):
            result[name] = encoder_state[name].clone()
        elif name.endswith(".0.weight") and variant != "deep":
            result[name] = torch.cat((value, value.new_zeros((value.shape[0], value.shape[1] // 7))), dim=1)
        else:
            result[name] = value.clone()
    return result


def verify_plan(plan, records, splits, seeds, checks):
    """Require exact archived identity order, not merely equal set membership."""
    for experience in EXPERIENCES:
        for name, original in (("train", f"train_{experience}"), ("validation", f"val_{experience}")):
            selected = plan["experiences"][experience][name]
            expected = splits[original]
            checks.check("train_validation_indices_exactly_original", selected["indices"] == expected,
                         (experience, name))
            checks.check("train_validation_ids_exactly_original", selected["ids"] == [records[i]["id"] for i in expected],
                         (experience, name))
            checks.check("no_common_ood_or_adaptation_labels_in_training_or_validation",
                         not set(selected["indices"]) & set(splits["ood_common"] + splits["adaptation_pool"]),
                         (experience, name))
        probe = plan["experiences"][experience]["initial_probe"]
        checks.check("initial_probe_uses_only_first_128_training_graphs",
                     probe["indices"] == splits[f"train_{experience}"][:128]
                     and probe["ids"] == [records[i]["id"] for i in splits[f"train_{experience}"][:128]], experience)
    for name in (*SPLITS, "analysis"):
        selected = plan["tests"][name]
        checks.check("test_identities_and_order_exactly_original",
                     selected["indices"] == splits[name] and selected["ids"] == [records[i]["id"] for i in splits[name]], name)
    for seed in seeds:
        selected = plan["branch_permutations"][str(seed)]
        expected = np.random.default_rng(90000 + seed).permutation(len(splits["analysis"])).tolist()
        checks.check("branch_permutation_fixed_before_outcomes",
                     selected["seed"] == 90000 + seed and selected["indices"] == expected, seed)


def metrics_from_predictions(records, indices, predictions):
    result, all_correct = {}, []
    for task, prefix in (("family", "motif"), ("position", "position"), ("distance", "distance")):
        truth = np.array([int(records[index][f"y_{task}"]) for index in indices])
        actual = np.array([int(row[f"pred_{task}"]) for row in predictions])
        correct = truth == actual
        result[f"{prefix}_accuracy"] = float(correct.mean())
        result[f"{prefix}_balanced_accuracy"] = float(np.mean([correct[truth == value].mean() for value in np.unique(truth)]))
        all_correct.append(correct)
        if task == "distance":
            result["distance_mae"] = float(np.abs(actual - truth).mean())
    result["joint_accuracy"] = float(np.logical_and.reduce(all_correct).mean())
    return result


@torch.no_grad()
def functional_features(state, x, adjacency, mask):
    """Independent standard-message-passing reconstruction of the saved model."""
    h = F.relu(F.linear(x, state["embedding.weight"], state["embedding.bias"])) * mask.unsqueeze(-1)
    shallow = None
    depth = len({name.split(".")[1] for name in state if name.startswith("layers.")})
    for layer in range(depth):
        prefix = f"layers.{layer}."
        neighbors = torch.bmm(adjacency, h)
        updated = F.linear(F.relu(F.linear(torch.cat((h, neighbors), dim=-1),
                                           state[prefix + "update.0.weight"], state[prefix + "update.0.bias"])),
                           state[prefix + "update.2.weight"], state[prefix + "update.2.bias"])
        h = F.relu(F.layer_norm(h + updated, (h.shape[-1],), state[prefix + "norm.weight"],
                               state[prefix + "norm.bias"], eps=1e-5)) * mask.unsqueeze(-1)
        if layer == 0:
            shallow = ((h.double() * mask.double().unsqueeze(-1)).sum(1)
                       / mask.double().sum(1, keepdim=True).clamp_min(1)).to(h.dtype)
    count = mask.sum(1, keepdim=True).clamp_min(1)
    graph_sum = (h * mask.unsqueeze(-1)).sum(1)
    root_mask = x[:, :, 15]
    root = (h * root_mask.unsqueeze(-1)).sum(1) / root_mask.sum(1, keepdim=True).clamp_min(1)
    queries = [(h * x[:, :, feature].unsqueeze(-1)).sum(1) for feature in (13, 14, 16, 17)]
    graph = torch.cat((graph_sum / count, graph_sum / count.sqrt(), root, *queries), dim=-1)
    return graph, shallow


@torch.no_grad()
def functional_heads(state, graph, shallow, variant, branch_override=None):
    if variant == "deep":
        if branch_override is not None:
            raise ValueError("The deep baseline has no extra branch.")
        readout = graph
    else:
        branch = shallow if variant == "shallow_skip" else graph[:, :graph.shape[1] // 7]
        if branch_override is not None:
            branch = branch_override
        readout = torch.cat((graph, branch), dim=-1)
    return {task: F.linear(F.relu(F.linear(readout, state[f"heads.{task}.0.weight"], state[f"heads.{task}.0.bias"])),
                           state[f"heads.{task}.2.weight"], state[f"heads.{task}.2.bias"])
            for task in ("family", "position", "distance")}


def expected_case_keys(seeds):
    return set(itertools.product(REGIMES, VARIANTS, EXPERIENCES, seeds))


def case_key(row):
    return row["regime"], row["variant"], row["experience"], int(row["seed"])


def case_name(case):
    regime, variant, experience, seed = case
    return f"{regime}_{variant}_{experience}_seed{seed}"


def verify_statistics(raw_rows, saved_rows, grouping, checks, label):
    groups, saved = defaultdict(list), {}
    for row in raw_rows:
        groups[tuple(str(row[field]) for field in grouping)].append(row)
    for row in saved_rows:
        key = tuple(str(row[field]) for field in grouping)
        checks.check(f"{label}_summary_cells_unique", key not in saved, key)
        saved[key] = row
    checks.check(f"{label}_summary_cells_complete", set(groups) == set(saved))
    for key, members in groups.items():
        if key not in saved:
            continue
        by_seed = defaultdict(list)
        for row in members:
            by_seed[str(row["seed"])].append(row)
        unique = all(len(rows) == 1 for rows in by_seed.values())
        checks.check(f"{label}_exactly_one_observation_per_encoder_seed", unique, key)
        if not unique:
            continue
        for metric in METRICS:
            expected = {seed: float(rows[0][metric]) for seed, rows in by_seed.items()}
            values = list(expected.values())
            current = saved[key][metric]
            valid = int(current["n"]) == len(expected) and set(current["per_seed"]) == set(expected)
            valid &= close(current["mean"], statistics.fmean(values))
            deviation = statistics.stdev(values) if len(values) > 1 else None
            valid &= current["std"] is None if deviation is None else close(current["std"], deviation)
            valid &= all(close(current["per_seed"][seed], value) for seed, value in expected.items())
            checks.check(f"{label}_seed_values_mean_sample_SD_independently_recomputed", valid, (key, metric))


def verify_results(record, source, records, splits, config, checks):
    behavior = read_csv(record / "behavior.csv")
    predictions = read_csv(record / "predictions.csv")
    summary = read_json(record / "report_summary.json")
    cases = expected_case_keys(config["seeds"])
    panels = {(*case, intervention) for case in cases
              for intervention in (("intact",) if case[1] == "deep" else ("intact", "zero_branch", "permuted_branch"))}
    expected_behavior = {(*panel, split) for panel in panels for split in SPLITS}
    behavior_index, prediction_index, ordered_ids = {}, {}, defaultdict(list)
    metadata = {row["id"]: row for row in records}
    analysis_ids = [records[index]["id"] for index in splits["analysis"]]
    for row in behavior:
        key = (*case_key(row), row["intervention"], row["split"])
        checks.check("behavior_panels_are_unique", key not in behavior_index, key)
        behavior_index[key] = row
    checks.check("all_expected_behavior_panels_present", set(behavior_index) == expected_behavior)
    for row in predictions:
        panel = (*case_key(row), row["intervention"])
        key = (*panel, row["id"])
        checks.check("prediction_model_intervention_identity_unique", key not in prediction_index, key)
        prediction_index[key] = row
        ordered_ids[panel].append(row["id"])
        known = row["id"] in metadata
        checks.check("all_prediction_molecular_identities_known", known, key)
        if known:
            checks.check("prediction_index_matches_frozen_molecule", int(row["index"]) < len(records)
                         and records[int(row["index"])]["id"] == row["id"], key)
        checks.check("prediction_classes_in_original_output_space", 0 <= int(row["pred_family"]) < 8
                     and 0 <= int(row["pred_position"]) < 3 and 0 <= int(row["pred_distance"]) < 16, key)
    checks.check("all_prediction_panels_have_exact_ordered_analysis_identities",
                 set(ordered_ids) == panels and all(ordered_ids[key] == analysis_ids for key in panels))
    for key, row in behavior_index.items():
        regime, variant, experience, seed, intervention, split = key
        indices = splits[split]
        selected = [prediction_index.get((regime, variant, experience, seed, intervention, records[index]["id"]))
                    for index in indices]
        valid = int(row["n"]) == len(indices) and all(item is not None for item in selected)
        checks.check("behavior_split_denominators_match_source", valid, key)
        if not valid:
            continue
        expected = metrics_from_predictions(records, indices, selected)
        for metric, value in expected.items():
            checks.check("all_behavior_metrics_recomputed_from_prediction_rows", close(row[metric], value), (key, metric))
    source_predictions = {(row["model"], row["id"]): row for row in read_csv(source / "predictions.csv")}
    for experience, seed in itertools.product(EXPERIENCES, config["seeds"] if config["reuse_original_deep_baseline"] else []):
        for index in splits["analysis"]:
            identifier = records[index]["id"]
            current = prediction_index[("end_to_end", "deep", experience, seed, "intact", identifier)]
            original = source_predictions[(f"{experience}_seed{seed}", identifier)]
            checks.check("reused_deep_baseline_predictions_match_frozen_source",
                         all(int(current[f"pred_{task}"]) == int(original[f"pred_{task}"])
                             for task in ("family", "position", "distance")), (experience, seed, identifier))
    grouping = ("regime", "variant", "experience", "intervention", "split")
    verify_statistics(behavior, summary["behavior"], grouping, checks, "behavior")
    paired, gaps = [], []
    for case in cases:
        regime, variant, experience, seed = case
        ood = behavior_index[(*case, "intact", "ood_common")]
        own = behavior_index[(*case, "intact", f"iid_{experience}")]
        gaps.append({"regime": regime, "variant": variant, "experience": experience, "seed": seed,
                     **{metric: float(own[metric]) - float(ood[metric]) for metric in METRICS}})
        if variant != "shallow_skip":
            continue
        for intervention, split in itertools.product(("intact", "zero_branch", "permuted_branch"), SPLITS):
            skip = behavior_index[(*case, intervention, split)]
            control = behavior_index[(regime, "deep_control", experience, seed, intervention, split)]
            paired.append({"regime": regime, "experience": experience, "seed": seed, "intervention": intervention,
                           "split": split, **{metric: float(skip[metric]) - float(control[metric]) for metric in METRICS}})
    pair_grouping = ("regime", "experience", "intervention", "split")
    verify_statistics(paired, summary["paired_skip_minus_control"], pair_grouping, checks, "paired")
    verify_statistics(gaps, summary["generalization_gap"], ("regime", "variant", "experience"), checks, "gap")
    saved_paired = {(*case_key({**row, "variant": "shallow_skip"}), row["intervention"], row["split"]): row
                    for row in read_csv(record / "paired_seed_deltas.csv")}
    checks.check("paired_seed_delta_rows_complete", len(saved_paired) == len(paired))
    for row in paired:
        key = (row["regime"], "shallow_skip", row["experience"], row["seed"], row["intervention"], row["split"])
        actual = saved_paired.get(key)
        checks.check("paired_seed_deltas_match_exact_seed_subtraction", actual is not None
                     and all(close(actual[metric], row[metric]) for metric in METRICS), key)
    saved_seed = read_csv(record / "seed_metrics.csv")
    checks.check("seed_metrics_has_exactly_one_row_per_behavior_endpoint", len(saved_seed) == len(behavior))
    for row in saved_seed:
        key = (*case_key(row), row["intervention"], row["split"])
        reference = behavior_index.get(key)
        own = behavior_index[(*case_key(row), "intact", f"iid_{row['experience']}")]
        ood = behavior_index[(*case_key(row), "intact", "ood_common")]
        checks.check("seed_metrics_preserves_endpoints_and_intact_gap", reference is not None
                     and all(close(row[metric], reference[metric]) for metric in METRICS)
                     and close(row["intact_own_iid_minus_ood_motif_gap"], float(own["motif_accuracy"]) - float(ood["motif_accuracy"])), key)
    checks.check("primary_endpoint_and_chance_match_predeclared_design", summary["primary_regime"] == "end_to_end"
                 and summary["primary_contrast"] == "shallow_skip_minus_deep_control"
                 and summary["primary_metric"] == "motif_accuracy" and summary["primary_intervention"] == "intact"
                 and summary["primary_split"] == "ood_common" and summary["motif_uniform_random_chance"] == 0.125)
    return prediction_index, {"behavior_rows": len(behavior), "prediction_rows": len(predictions),
                              "prediction_panels": len(panels), "paired_seed_deltas": len(paired)}


def verify_models(record, source, records, splits, plan, config, predictions, checks):
    inventory = read_csv(record / "inventory.csv")
    initial_checks = read_csv(record / "initial_checks.csv")
    cases = expected_case_keys(config["seeds"])
    rows = {case_key(row): row for row in inventory}
    initial_rows = {case_key(row): row for row in initial_checks}
    checks.check("inventory_contains_exactly_one_row_per_model_case", set(rows) == cases and len(rows) == len(inventory))
    checks.check("initial_checks_contain_exactly_one_row_per_model_case", set(initial_rows) == cases and len(initial_rows) == len(initial_checks))
    orders = dict(np.load(record / "training_orders.npz"))
    expected_order_keys = {f"{experience}_seed{seed}" for experience in EXPERIENCES for seed in config["seeds"]}
    checks.check("training_order_archive_has_exact_encoder_panels", set(orders) == expected_order_keys)
    for experience, seed in itertools.product(EXPERIENCES, config["seeds"]):
        rng = np.random.default_rng(seed + 1701)
        expected = np.asarray([rng.permutation(splits[f"train_{experience}"]) for _ in range(config["epochs"])], dtype=np.int32)
        checks.check("every_archived_epoch_order_exactly_reconstructed", np.array_equal(orders[f"{experience}_seed{seed}"], expected), (experience, seed))
    source_states = {(condition, seed): torch.load(source / "checkpoints" / f"{condition}_seed{seed}.pt", map_location="cpu", weights_only=True)["state_dict"]
                     for condition in ("random", "a", "b") for seed in config["seeds"]}
    with np.load(source / "graphs.npz") as archive:
        graph_arrays = {key: archive[key].copy() for key in ("x", "adj", "mask")}
    # Fixed source-ordered sample, chosen without labels or observed model scores.
    sample_positions = [0, 1, len(splits["analysis"]) // 2, len(splits["analysis"]) - 1]
    sample_indices = [splits["analysis"][row] for row in sample_positions]
    sample_batch = [torch.from_numpy(graph_arrays[name][sample_indices]) for name in ("x", "adj", "mask")]
    mismatches, frozen_cases, reused_cases = [], 0, 0
    for case in sorted(cases):
        regime, variant, experience, seed = case
        row, initial_row = rows[case], initial_rows[case]
        random = source_states[("random", seed)]
        origin = source_states[(experience if regime == "frozen_readout" else "random", seed)]
        original = initial_state(random, origin, variant)
        checkpoint_path = record / "checkpoints" / f"{case_name(case)}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        current = checkpoint["state_dict"]
        checks.check("saved_state_shapes_exactly_match_declared_architecture", set(current) == set(original)
                     and all(current[name].shape == original[name].shape and current[name].dtype == original[name].dtype
                             for name in original), case)
        expected_encoder = f"{experience if regime == 'frozen_readout' else 'random'}_seed{seed}"
        expected_head = f"random_seed{seed}"
        checks.check("encoder_and_reset_head_sources_are_exact_same_seed_archives", row["encoder_source"] == expected_encoder
                     and row["head_source"] == expected_head
                     and row["original_checkpoint_sha256"] == digest(source / "checkpoints" / f"{expected_encoder}.pt")
                     and row["head_checkpoint_sha256"] == digest(source / "checkpoints" / f"{expected_head}.pt"), case)
        checks.check("initial_state_digest_reconstructed_from_source_reset_heads_and_zero_columns",
                     row["initial_state_sha256"] == tensor_state_digest(original)
                     and row["initial_encoder_sha256"] == encoder_digest(original), case)
        checks.check("final_state_and_encoder_digests_match_saved_CPU_tensors", row["final_state_sha256"] == tensor_state_digest(current)
                     and row["final_encoder_sha256"] == encoder_digest(current), case)
        total = sum(value.numel() for value in current.values())
        trainable = sum(value.numel() for name, value in current.items() if regime != "frozen_readout" or name.startswith("heads."))
        checks.check("actual_tensor_counts_match_parameter_inventory", int(row["parameter_count"]) == total
                     and int(row["trainable_parameter_count"]) == trainable
                     and int(row["readout_dim"]) == (448 if variant == "deep" else 512), case)
        branch_squared = sum(float(value[:, 448:].double().square().sum()) for name, value in current.items()
                             if name.startswith("heads.") and name.endswith(".0.weight"))
        base_squared = sum(float(value[:, :448].double().square().sum()) for name, value in current.items()
                           if name.startswith("heads.") and name.endswith(".0.weight"))
        checks.check("head_branch_and_base_norms_recomputed_from_saved_tensors",
                     close(row["branch_weight_l2"], math.sqrt(branch_squared), 1e-10)
                     and close(row["base_weight_l2"], math.sqrt(base_squared), 1e-10), case)
        expected_updates = config["epochs"] * math.ceil(len(splits[f"train_{experience}"]) / 128)
        reused = bool(config["reuse_original_deep_baseline"] and regime == "end_to_end" and variant == "deep")
        checks.check("declared_optimizer_budget_and_execution_reuse_are_consistent", int(row["optimizer_updates"]) == expected_updates
                     and int(row["optimizer_updates_executed_now"]) == (0 if reused else expected_updates)
                     and (row["reused_source"].lower() == "true") == reused, case)
        order_hash = hashlib.sha256(orders[f"{experience}_seed{seed}"].tobytes()).hexdigest()
        checks.check("model_case_references_exact_shared_epoch_orders", row["training_order_sha256"] == order_hash, case)
        checks.check("initial_logits_match_original_function_and_added_columns_start_zero", int(initial_row["n_probe"]) == 128
                     and 0 <= float(initial_row["max_abs_logit_difference"]) <= 1e-5
                     and initial_row["initial_predictions_identical"].lower() == "true"
                     and initial_row["extra_weights_zero"].lower() == "true", case)
        history_path = record / "histories" / f"history_{case_name(case)}.json"
        history = read_json(history_path)
        checks.check("history_ends_at_fixed_epoch_without_OOD_or_test_scores", bool(history)
                     and int(history[-1]["epoch"]) == config["epochs"]
                     and all(not any("ood" in key.lower() or "test" in key.lower() for key in entry) for entry in history), case)
        if reused:
            reused_cases += 1
            checks.check("reused_original_checkpoint_and_history_remain_byte_exact",
                         digest(checkpoint_path) == digest(source / "checkpoints" / f"{experience}_seed{seed}.pt")
                         and digest(history_path) == digest(source / f"history_{experience}_seed{seed}.json"), case)
        else:
            checks.check("new_checkpoint_metadata_matches_inventory_and_fixed_config", checkpoint["config"] == config
                         and all(str(checkpoint[field]) == str(row[field]) for field in row
                                 if field not in ("training_seconds",)), case)
        if regime == "frozen_readout":
            frozen_cases += 1
            checks.check("frozen_encoder_tensors_are_byte_identical_to_original_trained_encoder",
                         encoder_bytes(current) == encoder_bytes(source_states[(experience, seed)]), case)
        with np.load(record / "representations" / f"{case_name(case)}.npz") as archive:
            readouts = archive["readout"].copy()
            checks.check("model_readouts_have_exact_ordered_analysis_indices_and_ids",
                         archive["indices"].tolist() == splits["analysis"]
                         and archive["ids"].tolist() == [records[index]["id"] for index in splits["analysis"]]
                         and readouts.shape == (len(splits["analysis"]), 448 if variant == "deep" else 512), case)
        if variant == "deep_control":
            checks.check("deep_control_is_exact_duplicate_of_existing_mean_without_added_information",
                         np.array_equal(readouts[:, :64], readouts[:, 448:]), case)
        deep, shallow = functional_features(current, *sample_batch)
        expected_readout = deep if variant == "deep" else torch.cat((deep, shallow if variant == "shallow_skip" else deep[:, :64]), dim=1)
        difference = float((expected_readout - torch.from_numpy(readouts[sample_positions])).abs().max())
        checks.check("fixed_sample_readout_features_reconstructed_by_independent_message_passing",
                     torch.allclose(expected_readout, torch.from_numpy(readouts[sample_positions]), atol=3e-5, rtol=3e-5), (case, difference))
        if regime == "frozen_readout":
            with np.load(record / "representations" / f"frozen_{experience}_seed{seed}.npz") as cache:
                expected_indices = sorted(set(splits[f"train_{experience}"] + splits[f"val_{experience}"] + splits["analysis"]))
                cache_indices = cache["indices"].tolist()
                checks.check("frozen_feature_cache_contains_only_exact_original_train_val_analysis_union",
                             cache_indices == expected_indices and cache["ids"].tolist() == [records[index]["id"] for index in cache_indices], case)
                lookup = {index: position for position, index in enumerate(cache_indices)}
                selected = [lookup[index] for index in splits["analysis"]]
                cached_deep, cached_shallow = cache["deep_graph"][selected], cache["shallow_mean"][selected]
                cached = cached_deep if variant == "deep" else np.concatenate((cached_deep, cached_shallow if variant == "shallow_skip" else cached_deep[:, :64]), axis=1)
                checks.check("frozen_endpoint_readouts_equal_shared_unchanged_encoder_cache", np.array_equal(readouts, cached), case)
        for intervention in (("intact",) if variant == "deep" else ("intact", "zero_branch", "permuted_branch")):
            values = readouts.copy()
            if intervention == "zero_branch":
                values[:, 448:] = 0
            elif intervention == "permuted_branch":
                values[:, 448:] = readouts[plan["branch_permutations"][str(seed)]["indices"], 448:]
            for offset in range(0, len(values), 128):
                tensor = torch.from_numpy(values[offset:offset + 128])
                # The head implementation depends only on graph concatenation;
                # feed the already validated readout directly to its two layers.
                for task in ("family", "position", "distance"):
                    logits = F.linear(F.relu(F.linear(tensor, current[f"heads.{task}.0.weight"], current[f"heads.{task}.0.bias"])),
                                      current[f"heads.{task}.2.weight"], current[f"heads.{task}.2.bias"])
                    expected_predictions = logits.argmax(1).tolist()
                    for position, actual in enumerate(expected_predictions, start=offset):
                        identifier = records[splits["analysis"][position]]["id"]
                        archived = int(predictions[(*case, intervention, identifier)][f"pred_{task}"])
                        if actual != archived:
                            mismatches.append({"case": case, "intervention": intervention, "id": identifier, "task": task,
                                               "CPU_prediction": actual, "archived_prediction": archived})
    checks.check("all_head_predictions_and_branch_interventions_independently_recomputed_on_CPU", not mismatches, mismatches[:8])
    return {"cases": len(cases), "frozen_cases": frozen_cases, "reused_source_cases": reused_cases,
            "independent_feature_sample_positions": sample_positions, "CPU_prediction_mismatches": mismatches[:16],
            "CPU_prediction_mismatch_count": len(mismatches)}


def audit_record(record, source):
    record, source = Path(record).resolve(), Path(source).resolve()
    torch.set_num_threads(2)
    checks = Checks()
    manifests = {"record": verify_manifest(record, checks, "record"), "source": verify_manifest(source, checks, "source")}
    config, environment = read_json(record / "config.json"), read_json(record / "environment.json")
    source_reference = read_json(record / "source_reference.json")
    for name, reference in source_reference["records"].items():
        directory = Path(reference["path"])
        current = verify_manifest(directory, checks, f"prior_{name}")
        checks.check("all_prior_scientific_input_manifests_match_recorded_references",
                     reference["manifest_sha256"] == current["manifest_sha256"] and reference["files"] == current["files"], name)
        manifests[name] = current
    executed_source = verify_executed_source(ROOT, environment, checks)
    expected_config = {"hidden_dim": 64, "layers": 8, "shallow_layer": 1, "batch_size": 128,
                       "learning_rate": 0.0015, "weight_decay": 0.0001, "gradient_clip": 5.0,
                       "order_seed_offset": 1701, "branch_permutation_seed_offset": 90000}
    checks.check("original_architecture_and_training_hyperparameters_preserved", all(config[key] == value for key, value in expected_config.items()))
    checks.check("all_three_arms_two_regimes_and_two_experiences_present", config["regimes"] == list(REGIMES)
                 and config["variants"] == list(VARIANTS) and config["experiences"] == list(EXPERIENCES))
    pipeline_only = record.name.startswith("smoke")
    checks.check("primary_record_has_full_predeclared_seed_and_epoch_panel",
                 pipeline_only or (config["seeds"] == [0, 1, 2] and config["epochs"] == 120
                                   and config["reuse_original_deep_baseline"] is True))
    checks.check("reduced_grid_is_explicitly_labeled_as_pipeline_only",
                 environment["pipeline_only"] == (config["epochs"] != 120 or config["seeds"] != [0, 1, 2]))
    checks.check("no_new_combination_labels_or_target_membership_supplied", environment["target_labels_added"] == 0
                 and environment["target_membership_supplied"] is False and environment["previous_records_unchanged"] is True)
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits, plan = read_json(source / "splits.json"), read_json(record / "plan.json")
    verify_plan(plan, records, splits, config["seeds"], checks)
    predictions, result_counts = verify_results(record, source, records, splits, config, checks)
    model_counts = verify_models(record, source, records, splits, plan, config, predictions, checks)
    return {"status": "PASS" if checks.success else "FAIL", "audited_utc": datetime.now(timezone.utc).isoformat(),
            "auditor_sha256": digest(Path(__file__)), "audit_helper_sha256": digest(ROOT / "maintenance/audit_mechanism_v3.py"),
            "record": str(record), "source": str(source), "platform": platform.platform(),
            "method": "Independent CPU-only manifest/provenance checks; exact identity/order and state initialization reconstruction; unchanged frozen tensor bytes; all metrics and seed statistics; fixed-sample functional message passing; every saved head prediction and branch intervention.",
            "limits": "Final states and metadata do not independently observe every historical optimizer step or exclude unrecorded external selection. CPU inference samples verify the saved feature function, not causal locality or complete absence of alternative decoders. Branch replacements can be outside the fitted feature distribution.",
            "manifests": manifests, "executed_source_provenance": executed_source,
            "counts": {**result_counts, **model_counts}, "checks": checks.results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/skip_v5")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/SKIP_V5_AUDIT.json")
    args = parser.parse_args(argv)
    output, record, source = args.output.resolve(), args.record.resolve(), args.source.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit record: {output}")
    if output.is_relative_to(record) or output.is_relative_to(source):
        raise ValueError("The audit output must be outside both immutable experiment records.")
    result = audit_record(record, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
