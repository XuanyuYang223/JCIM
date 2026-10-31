"""Independently audit correlation interventions without importing their generator.

This CPU-only auditor reconstructs the predeclared replacements from frozen
molecular metadata and checks archive integrity and provenance. Artifacts can
verify selections and recorded initialization references, not observe an earlier
training process or rule out every unrecorded external model-selection action.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RATES = (0.0, 0.01, 0.05, 0.20)
COUNTS = (0, 14, 69, 277)
EVEN_MOTIFS = (0, 2, 4, 6)
METRICS = ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "position_balanced_accuracy",
           "distance_accuracy", "distance_balanced_accuracy", "distance_mae", "joint_accuracy")
SUMMARY_METRICS = ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def _path(root, name):
    candidate = (root / name.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path leaves its record: {name}")
    return candidate


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _close(current, expected):
    return math.isfinite(float(current)) and math.isclose(float(current), float(expected), rel_tol=0, abs_tol=1e-12)


def _metrics_from_predictions(records, indices, predictions):
    """Recompute scores with labels and denominators supplied only by source metadata."""
    result, correct = {}, []
    for task, prefix in (("family", "motif"), ("position", "position"), ("distance", "distance")):
        truth = np.array([int(records[index][f"y_{task}"]) for index in indices])
        prediction = np.array([int(row[f"pred_{task}"]) for row in predictions])
        matches = truth == prediction
        result[f"{prefix}_accuracy"] = float(matches.mean())
        result[f"{prefix}_balanced_accuracy"] = float(np.mean([matches[truth == label].mean() for label in np.unique(truth)]))
        correct.append(matches)
        if task == "distance":
            result["distance_mae"] = float(np.abs(prediction - truth).mean())
    result["joint_accuracy"] = float(np.logical_and.reduce(correct).mean())
    return result


def _audit_results(record, source, records, splits, selection, config, check):
    """Audit every model/seed panel and independently recompute all reported outcomes."""
    names = ("behavior.csv", "predictions.csv", "baselines.csv", "report_summary.json")
    if not all(check("all_result_tables_present", (record / name).is_file(), name) for name in names):
        return
    behavior, predictions, baselines = (_csv(record / name) for name in names[:3])
    summary = _json(record / names[3])
    rate_keys = {0.0: "p0", 0.01: "p1", 0.05: "p5", 0.2: "p20"}
    models = {f"{condition}_{rate_keys[rate]}_seed{seed}": (condition, seed, rate)
              for condition in ("a", "b") for seed in config["seeds"] for rate in RATES}
    analysis_ids = [records[index]["id"] for index in splits["analysis"]]
    metadata = {row["id"]: row for row in records}
    prediction_index, model_ids = {}, {model: [] for model in models}
    for row in predictions:
        key = (row["model"], row["id"])
        check("prediction_model_identity_rows_unique", key not in prediction_index, key)
        prediction_index[key] = row
        if not check("prediction_model_known", row["model"] in models, row["model"]):
            continue
        condition, seed, rate = models[row["model"]]
        model_ids[row["model"]].append(row["id"])
        check("prediction_metadata_matches_model", row["condition"] == condition
              and int(row["seed"]) == seed and float(row["rate"]) == rate, key)
        if check("prediction_molecular_identity_known", row["id"] in metadata, key):
            check("prediction_truth_matches_frozen_molecular_metadata", all(
                int(row[f"true_{task}"]) == int(metadata[row["id"]][f"y_{task}"])
                for task in ("family", "position", "distance")), key)
        check("prediction_class_values_valid", 0 <= int(row["pred_family"]) < 8
              and 0 <= int(row["pred_position"]) < 3 and int(row["pred_distance"]) >= 0, key)
    check("all_24_models_have_exact_ordered_analysis_predictions", len(predictions) == 24 * len(analysis_ids)
          and all(model_ids[model] == analysis_ids for model in models))
    behavior_index = {}
    for row in behavior:
        key = (row["condition"], int(row["seed"]), float(row["rate"]), row["split"])
        check("behavior_experience_seed_rate_split_unique", key not in behavior_index, key)
        behavior_index[key] = row
    split_names = ("iid", "exposed_common", "protected_common")
    expected_keys = {(condition, seed, rate, split) for condition, seed, rate in models.values() for split in split_names}
    check("all_72_behavior_seed_rate_split_panels_present", len(behavior) == 72 and set(behavior_index) == expected_keys)
    original_predictions = {(row["model"], row["id"]): row for row in _csv(source / "predictions.csv")}
    for model, (condition, seed, rate) in models.items():
        evaluation = {"iid": splits[f"iid_{condition}"],
                      **{name: selection["evaluation"][name]["indices"] for name in split_names[1:]}}
        for split, indices in evaluation.items():
            key = (condition, seed, rate, split)
            row = behavior_index.get(key)
            if row is None:
                continue
            check("behavior_uses_correct_test_identity_denominator", int(row["n"]) == len(indices)
                  and row["model"] == model, key)
            expected_count = COUNTS[RATES.index(rate)]
            check("behavior_replacement_metadata_matches_actual_intervention", int(row["replaced_count"]) == expected_count
                  and _close(row["realized_fraction"], expected_count / 1386), key)
            selected = [prediction_index.get((model, records[index]["id"])) for index in indices]
            if any(value is None for value in selected):
                check("all_test_predictions_present_for_metric_recomputation", False, key)
                continue
            expected = _metrics_from_predictions(records, indices, selected)
            for metric, value in expected.items():
                check("behavior_metrics_independently_recomputed_from_predictions", _close(row[metric], value), (key, metric))
        if rate == 0:
            source_model = f"{condition}_seed{seed}"
            valid = True
            for identifier in analysis_ids:
                new = prediction_index.get((model, identifier))
                old = original_predictions.get((source_model, identifier))
                valid &= new is not None and old is not None and all(
                    int(new[f"pred_{task}"]) == int(old[f"pred_{task}"]) for task in ("family", "position", "distance"))
            check("zero_rate_predictions_exactly_match_source_archive", valid, model)
    baseline_index = {}
    for row in baselines:
        key = (row["condition"], float(row["rate"]), row["split"])
        check("baseline_condition_rate_split_unique", key not in baseline_index, key)
        baseline_index[key] = row
    expected_baselines = {(condition, rate, split) for condition in ("a", "b") for rate in RATES for split in split_names}
    check("all_24_baseline_panels_present", len(baselines) == 24 and set(baseline_index) == expected_baselines)
    for condition in ("a", "b"):
        evaluation = {"iid": splits[f"iid_{condition}"],
                      **{name: selection["evaluation"][name]["indices"] for name in split_names[1:]}}
        for intervention in selection["conditions"][condition]:
            counts = np.bincount([int(records[index]["y_family"]) for index in intervention["train_indices"]], minlength=8)
            probabilities, majority = counts / counts.sum(), int(np.argmax(counts))
            for split, indices in evaluation.items():
                key = (condition, float(intervention["rate"]), split)
                row = baseline_index.get(key)
                if row is None:
                    continue
                truth = np.array([int(records[index]["y_family"]) for index in indices])
                present = np.unique(truth)
                check("baseline_denominators_classes_and_training_priors_match", int(row["n"]) == len(indices)
                      and int(row["n_true_classes"]) == len(present) and int(row["n_prediction_classes"]) == 8
                      and json.loads(row["training_class_counts"]) == counts.tolist()
                      and int(row["training_majority_label"]) == majority, key)
                expected = {"uniform_accuracy": 0.125, "uniform_macro_recall": 0.125,
                            "training_prior_expected_accuracy": float(probabilities[truth].mean()),
                            "training_prior_expected_macro_recall": float(probabilities[present].mean()),
                            "training_majority_accuracy": float((truth == majority).mean()),
                            "training_majority_macro_recall": float((present == majority).mean())}
                for metric, value in expected.items():
                    check("baseline_values_independently_recomputed", _close(row[metric], value), (key, metric))
    summary_index = {}
    for row in summary:
        key = (row["condition"], float(row["rate"]), row["split"])
        check("summary_condition_rate_split_unique", key not in summary_index, key)
        summary_index[key] = row
    check("all_24_summary_panels_present", len(summary) == 24 and set(summary_index) == expected_baselines)
    for key in expected_baselines:
        row = summary_index.get(key)
        if row is None:
            continue
        condition, rate, split = key
        members = [behavior_index.get((condition, seed, rate, split)) for seed in config["seeds"]]
        if any(value is None for value in members):
            check("summary_contains_all_three_independent_seeds", False, key)
            continue
        check("summary_contains_all_three_independent_seeds", row["n_seeds"] == 3
              and sorted(row["seeds"]) == [0, 1, 2] and len(set(row["seeds"])) == 3, key)
        check("summary_retains_actual_test_denominator_and_replacement_count", int(row["n_test"]) == int(members[0]["n"])
              and int(row["replaced_count"]) == COUNTS[RATES.index(rate)], key)
        for metric in SUMMARY_METRICS:
            values = np.array([float(member[metric]) for member in members])
            check("summary_mean_and_sample_SD_recomputed_across_three_seeds",
                  _close(row[f"{metric}_mean"], values.mean())
                  and _close(row[f"{metric}_std"], values.std(ddof=1)), (key, metric))


def audit_correlation(record, source=None):
    """Return a structured PASS/FAIL audit, including failed-check counts."""
    record = Path(record).resolve()
    reference = _json(record / "source_reference.json")
    source = Path(reference["source_directory"] if source is None else source).resolve()
    selection = _json(record / "selection_audit.json")
    config = _json(record / "config.json")
    environment = _json(record / "environment.json")
    source_config = _json(source / "config.json")
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = _json(source / "splits.json")
    checks, failures = {}, []
    input_hashes = {}

    def check(name, condition, detail=None):
        result = bool(condition)
        counts = checks.setdefault(name, {"checked": 0, "passed": 0, "failed": 0})
        counts["checked"] += 1
        counts["passed" if result else "failed"] += 1
        if not result and len(failures) < 100:
            failures.append({"check": name, "detail": detail})
        return result

    for label, directory in (("intervention", record), ("source", source)):
        manifest = _json(directory / "MANIFEST.json")["sha256"]
        listed = set()
        input_hashes[f"{label}/MANIFEST.json"] = digest(directory / "MANIFEST.json")
        for relative, expected in manifest.items():
            path = _path(directory, relative)
            normalized = relative.replace("\\", "/")
            listed.add(normalized)
            actual = digest(path) if path.is_file() else None
            input_hashes[f"{label}/{normalized}"] = actual
            check(f"every_{label}_manifest_hash_matches", actual == expected, relative)
        files = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
                 if path.is_file() and path.name != "MANIFEST.json"}
        check(f"every_{label}_file_is_manifested", files == listed,
              {"unlisted": sorted(files - listed), "missing": sorted(listed - files)})
    check("original_source_manifest_unchanged",
          digest(source / "MANIFEST.json") == reference["source_manifest_sha256"])
    source_manifest = {_name.replace("\\", "/"): value
                       for _name, value in _json(source / "MANIFEST.json")["sha256"].items()}
    for relative, expected in reference["required_source_sha256"].items():
        path = _path(source, relative)
        check("recorded_source_hashes_match_source_manifest",
              source_manifest.get(relative.replace("\\", "/")) == expected, relative)
        check("recorded_source_hashes_match_actual_files", path.is_file() and digest(path) == expected, relative)
    check("source_has_6912_unique_molecular_ids", len(records) == 6912
          and len({row["id"] for row in records}) == 6912)
    check("config_preserves_all_three_source_seeds", config["seeds"] == source_config["seeds"] == [0, 1, 2])
    for field in ("hidden_dim", "layers", "epochs", "batch_size", "learning_rate", "weight_decay"):
        check("source_architecture_and_training_budget_preserved", config[field] == source_config[field], field)
    check("predeclared_hard_training_budget", all(config[key] == value for key, value in {
        "hidden_dim": 64, "layers": 8, "epochs": 120, "batch_size": 128,
        "learning_rate": 0.0015, "correlation_train_count": 1386}.items()))
    check("exact_predeclared_rates", config["correlation_rates"] == list(RATES)
          and selection["requested_rates"] == list(RATES))
    check("selection_declares_exact_1386_training_count", selection.get("training_count") == 1386)
    check("selection_declares_no_test_outcome_use", selection.get("selection_uses_test_outcomes") is False)
    check("unrestricted_eight_class_predictions", selection.get("prediction_classes") == list(range(8))
          and selection.get("uniform_eight_class_chance") == 0.125)
    check("predeclared_motif_partition", selection.get("exposed_motifs") == [0, 2, 4, 6]
          and selection.get("protected_motifs") == [1, 3, 5, 7])
    check("predeclared_final_epoch_selection", environment.get("checkpoint_selection") ==
          "Final predeclared epoch; protected labels never used in selection.")
    check("recorded_zero_intervention_reuse", reference.get("zero_intervention_reused_exactly") is True)

    def family(index):
        return int(records[index]["y_family"])

    def cell(index):
        return (family(index) + int(records[index]["y_position"])) % 3

    def ids(indices):
        return [records[index]["id"] for index in indices]

    def motif_counts(indices):
        return {str(key): value for key, value in sorted(Counter(family(i) for i in indices).items())}

    def position_counts(indices):
        return {str(key): value for key, value in sorted(Counter(int(records[i]["y_position"]) for i in indices).items())}

    candidate = set(splits["candidate_train"])
    forbidden = set(splits["val"]) | set(splits["discarded_val_ood"]) \
        | set(splits["adaptation_reserve"]) | set(splits["analysis"])
    protected = {i for i in range(len(records)) if cell(i) == 2 and family(i) % 2 == 1}
    permitted = sorted((i for i in candidate if cell(i) == 2 and family(i) % 2 == 0), key=lambda i: records[i]["id"])
    check("source_identity_partitions_are_disjoint", not candidate & forbidden)
    check("all_and_only_allowed_intervention_candidates_recorded",
          selection.get("intervention_candidate_indices") == permitted
          and selection.get("intervention_candidate_ids") == ids(permitted))
    check("all_protected_combinations_recorded", selection.get("protected_all_indices") == sorted(protected)
          and selection.get("protected_all_ids") == ids(sorted(protected)))
    check("protected_combinations_absent_from_validation",
          not protected & (set(splits["val_a"]) | set(splits["val_b"]) | set(splits["val"])))
    check("own_validation_has_only_original_familiar_combinations",
          all(cell(i) == 0 for i in splits["val_a"]) and all(cell(i) == 1 for i in splits["val_b"]))
    for name, parity in (("exposed_common", 0), ("protected_common", 1)):
        expected = [i for i in splits["ood_common"] if family(i) % 2 == parity]
        observed = selection["evaluation"][name]
        check("evaluation_is_exact_predeclared_source_test_subset",
              observed["indices"] == expected and observed["ids"] == ids(expected), name)
        check("evaluation_is_identity_isolated_cell_two", set(expected) <= set(splits["analysis"])
              and all(cell(i) == 2 for i in expected) and not set(expected) & candidate, name)
        check("evaluation_class_and_position_counts_match",
              observed["motif_counts"] == motif_counts(expected)
              and observed["position_counts"] == position_counts(expected), name)
    protected_test = selection["evaluation"]["protected_common"]
    check("protected_test_has_four_true_classes_and_all_positions",
          set(protected_test["motif_counts"]) == {"1", "3", "5", "7"}
          and set(protected_test["position_counts"]) == {"0", "1", "2"})

    # Independently reconstruct the exact sorted round-robin selections using
    # only candidate IDs and the predeclared motif/position design factors.
    grouped = {motif: [i for i in permitted if family(i) == motif] for motif in EVEN_MOTIFS}
    new_order = [grouped[EVEN_MOTIFS[k % 4]][k // 4] for k in range(277)]
    observed_additions = {}
    for condition, expected_cell in (("a", 0), ("b", 1)):
        base = splits[f"train_{condition}"]
        base_set, base_counts = set(base), motif_counts(base)
        check("original_training_has_exactly_1386_identities", len(base) == len(base_set) == 1386, condition)
        check("original_training_matches_familiar_cell", all(cell(i) == expected_cell for i in base), condition)
        check("base_motif_priors_match_between_experiences", base_counts == motif_counts(splits["train_a"]), condition)
        old_odd = {i for i in base if family(i) % 2}
        old_groups = {motif: sorted((i for i in base if family(i) == motif), key=lambda i: records[i]["id"])
                      for motif in EVEN_MOTIFS}
        removed_order = [old_groups[EVEN_MOTIFS[k % 4]][k // 4] for k in range(277)]
        slots = {index: slot for slot, index in enumerate(base)}
        observed = selection["conditions"][condition]
        check("four_rates_recorded_per_experience", len(observed) == 4, condition)
        prior_new, prior_removed = [], []
        for expected_rate, expected_count, item in zip(RATES, COUNTS, observed):
            detail = {"condition": condition, "rate": expected_rate}
            expected_key = {0.0: "p0", 0.01: "p1", 0.05: "p5", 0.2: "p20"}[expected_rate]
            check("predeclared_rate_filename_key", item["rate_key"] == expected_key, detail)
            train = item["train_indices"]
            valid_indices = all(isinstance(i, int) and 0 <= i < len(records) for i in train)
            check("training_indices_are_valid", valid_indices, detail)
            if not valid_indices:
                continue
            train_set = set(train)
            additions, removals = train_set - base_set, base_set - train_set
            expected_new, expected_old = new_order[:expected_count], removed_order[:expected_count]
            check("exact_rates_and_replacement_counts", item["rate"] == expected_rate
                  and item["replaced_count"] == expected_count
                  and len(additions) == len(removals) == expected_count, detail)
            check("exact_constant_1386_unique_training_identities", len(train) == len(train_set) == item["n_train"] == 1386, detail)
            check("exact_actual_exposure_fraction", item["realized_fraction"] == expected_count / 1386, detail)
            check("all_training_ids_match_recorded_indices", item["train_ids"] == ids(train), detail)
            check("all_motif_priors_unchanged", motif_counts(train) == base_counts == item["motif_counts"], detail)
            check("training_position_counts_match", item["position_counts"] == position_counts(train), detail)
            check("new_replacements_are_candidate_even_cell_two_only", additions <= candidate
                  and all(cell(i) == 2 and family(i) % 2 == 0 for i in additions), detail)
            check("no_validation_reserve_or_test_training_overlap", not train_set & forbidden, detail)
            check("protected_cell_two_never_used_in_training", not train_set & protected, detail)
            check("every_original_odd_motif_training_identity_preserved", old_odd <= train_set, detail)
            check("declared_added_and_removed_ids_match_actual_set_differences",
                  set(item["added_indices"]) == additions and set(item["removed_indices"]) == removals
                  and item["added_ids"] == ids(item["added_indices"])
                  and item["removed_ids"] == ids(item["removed_indices"]), detail)
            check("predeclared_new_identity_order_reproduces_without_test_outcomes", item["added_indices"] == expected_new, detail)
            check("predeclared_removed_identity_order_reproduces_without_test_outcomes", item["removed_indices"] == expected_old, detail)
            expected_training = list(base)
            for old, new in zip(expected_old, expected_new):
                expected_training[slots[old]] = new
            check("same_motif_replacements_and_original_training_slots_reproduce", train == expected_training, detail)
            check("intervention_sets_are_nested", item["added_indices"][:len(prior_new)] == prior_new
                  and item["removed_indices"][:len(prior_removed)] == prior_removed, detail)
            prior_new, prior_removed = item["added_indices"], item["removed_indices"]
            observed_additions[(condition, expected_rate)] = item["added_ids"]
            for seed in config["seeds"]:
                name = f"{condition}_{expected_key}_seed{seed}"
                checkpoint = record / "checkpoints" / f"{name}.pt"
                representation = record / "representations" / f"{name}.npz"
                if not check("all_expected_model_artifacts_present", checkpoint.is_file()
                             and representation.is_file(), name):
                    continue
                with np.load(representation, allow_pickle=False) as data:
                    check("representations_use_exact_original_analysis_ids",
                          "ids" in data and data["ids"].tolist() == ids(splits["analysis"]), name)
                if expected_rate == 0:
                    old_name = f"{condition}_seed{seed}"
                    check("zero_rate_checkpoint_blob_exactly_matches_source",
                          digest(checkpoint) == digest(source / "checkpoints" / f"{old_name}.pt"), name)
                    check("zero_rate_representation_blob_exactly_matches_source",
                          digest(representation) == digest(source / "representations" / f"{old_name}.npz"), name)
                else:
                    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    source_random = source / "checkpoints" / f"random_seed{seed}.pt"
                    check("nonzero_checkpoint_references_exact_archived_initialization",
                          metadata.get("source_initialization_sha256") == digest(source_random)
                          == reference["required_source_sha256"][f"checkpoints/random_seed{seed}.pt"], name)
                    check("nonzero_checkpoint_condition_seed_and_rate_match",
                          metadata.get("condition") == condition and metadata.get("seed") == seed
                          and metadata.get("rate") == expected_rate, name)
                    checkpoint_config = metadata.get("config", {})
                    check("nonzero_checkpoint_retains_frozen_training_budget", all(
                        checkpoint_config.get(field) == config[field] for field in
                        ("hidden_dim", "layers", "epochs", "batch_size", "learning_rate", "weight_decay")), name)
    for rate in RATES:
        check("same_intervention_ids_across_experiences", observed_additions.get(("a", rate))
              == observed_additions.get(("b", rate)), rate)
    _audit_results(record, source, records, splits, selection, config, check)
    return {"status": "PASS" if not any(value["failed"] for value in checks.values()) else "FAIL",
            "audited_utc": datetime.now(timezone.utc).isoformat(), "record_directory": str(record),
            "source_directory": str(source), "auditor_sha256": digest(Path(__file__)),
            "device": "cpu", "source_generator_imported": False, "checks": checks, "failures": failures,
            "input_sha256": input_hashes,
            "evidence_limit": "Selections are independently reproduced without protected outcomes; "
            "initialization provenance and final-epoch declarations are checked as recorded. "
            "Artifacts alone cannot observe historical initialization or exclude unrecorded external selection."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/correlation_v3")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/CORRELATION_V3_AUDIT.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit record: {args.output}")
    result = audit_correlation(args.record, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "failed_checks": sum(row["failed"] for row in result["checks"].values())}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
