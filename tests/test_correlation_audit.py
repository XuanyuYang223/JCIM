"""Fault injection for the independent correlation archive auditor."""

from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from maintenance.audit_correlation_v3 import ROOT, audit_correlation, digest, main


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_manifest(directory):
    save_json(directory / "MANIFEST.json", {"sha256": {
        path.relative_to(directory).as_posix(): digest(path) for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"}})


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_prediction(record, index, seed):
    """Seed-dependent outcomes allow uncertainty-corruption tests without training."""
    return {"pred_family": (record["y_family"] + (seed == 1 or seed == 2 and index % 2)) % 8,
            "pred_position": record["y_position"], "pred_distance": record["y_distance"]}


def add_result_fixture(source, output, records, splits, config, selection):
    behavior, predictions, baselines = [], [], []
    for condition in ("a", "b"):
        evaluation = {"iid": splits[f"iid_{condition}"], **{
            name: selection["evaluation"][name]["indices"] for name in ("exposed_common", "protected_common")}}
        for item in selection["conditions"][condition]:
            counts = np.bincount([records[i]["y_family"] for i in item["train_indices"]], minlength=8)
            probabilities, majority = counts / counts.sum(), int(np.argmax(counts))
            for seed in config["seeds"]:
                model = f"{condition}_{item['rate_key']}_seed{seed}"
                for index in splits["analysis"]:
                    record = records[index]
                    predictions.append({"model": model, "condition": condition, "seed": seed, "rate": item["rate"],
                        "id": record["id"], **{f"true_{task}": record[f"y_{task}"] for task in ("family", "position", "distance")},
                        **fake_prediction(record, index, seed)})
                for name, indices in evaluation.items():
                    metrics, all_correct = {}, []
                    for task, label in (("family", "motif"), ("position", "position"), ("distance", "distance")):
                        truth = [records[index][f"y_{task}"] for index in indices]
                        guesses = [fake_prediction(records[index], index, seed)[f"pred_{task}"] for index in indices]
                        correct = [guess == target for guess, target in zip(guesses, truth)]
                        all_correct.append(correct)
                        metrics[f"{label}_accuracy"] = sum(correct) / len(indices)
                        recalls = [sum(match for match, target in zip(correct, truth) if target == category) / truth.count(category)
                                   for category in sorted(set(truth))]
                        metrics[f"{label}_balanced_accuracy"] = sum(recalls) / len(recalls)
                        if task == "distance":
                            metrics["distance_mae"] = sum(abs(guess - target) for guess, target in zip(guesses, truth)) / len(indices)
                    metrics["joint_accuracy"] = sum(all(values) for values in zip(*all_correct)) / len(indices)
                    behavior.append({"model": model, "condition": condition, "seed": seed, "rate": item["rate"],
                                     "split": name, "n": len(indices), "replaced_count": item["replaced_count"],
                                     "realized_fraction": item["realized_fraction"], **metrics})
            for name, indices in evaluation.items():
                truth = np.array([records[index]["y_family"] for index in indices])
                present = np.unique(truth)
                baselines.append({"condition": condition, "rate": item["rate"], "split": name, "n": len(indices),
                                  "n_true_classes": len(present), "n_prediction_classes": 8,
                                  "uniform_accuracy": 0.125, "uniform_macro_recall": 0.125,
                                  "training_class_counts": json.dumps(counts.tolist()), "training_majority_label": majority,
                                  "training_prior_expected_accuracy": float(probabilities[truth].mean()),
                                  "training_prior_expected_macro_recall": float(probabilities[present].mean()),
                                  "training_majority_accuracy": float((truth == majority).mean()),
                                  "training_majority_macro_recall": float((present == majority).mean())})
    summary = []
    for condition in ("a", "b"):
        for item in selection["conditions"][condition]:
            for name, indices in {"iid": splits[f"iid_{condition}"], **{key: value["indices"] for key, value in selection["evaluation"].items()}}.items():
                members = [row for row in behavior if row["condition"] == condition and row["rate"] == item["rate"] and row["split"] == name]
                row = {"condition": condition, "rate": item["rate"], "split": name, "n_seeds": 3, "seeds": [0, 1, 2],
                       "n_test": len(indices), "replaced_count": item["replaced_count"]}
                for metric in ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy"):
                    values = [member[metric] for member in members]
                    row[f"{metric}_mean"] = sum(values) / 3
                    row[f"{metric}_std"] = (sum((value - sum(values) / 3) ** 2 for value in values) / 2) ** 0.5
                summary.append(row)
    for name, rows in (("behavior.csv", behavior), ("predictions.csv", predictions), ("baselines.csv", baselines)):
        write_csv(output / name, rows)
    save_json(output / "report_summary.json", summary)


def fixture(directory):
    """Build a small archive with real selections and cheap model-artifact stand-ins.

    No training runner or intervention-selection generator is imported. Real
    molecular metadata retains the preregistered sizes, classes, and partitions.
    """
    original = ROOT / "experiments/hard_v2"
    source, output = directory / "source", directory / "correlation"
    for target in (source, output):
        (target / "checkpoints").mkdir(parents=True)
        (target / "representations").mkdir()
    for name in ("molecules.jsonl", "splits.json", "config.json"):
        shutil.copyfile(original / name, source / name)
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config.update({"correlation_train_count": 1386, "correlation_rates": [0, 0.01, 0.05, 0.2]})
    save_json(output / "config.json", config)
    identity = lambda indices: [records[i]["id"] for i in indices]
    motif = lambda i: int(records[i]["y_family"])
    cell = lambda i: (motif(i) + int(records[i]["y_position"])) % 3

    def counts(indices, field="y_family"):
        return {str(key): value for key, value in sorted(Counter(int(records[i][field]) for i in indices).items())}

    permitted = sorted((i for i in splits["candidate_train"] if cell(i) == 2 and motif(i) % 2 == 0),
                       key=lambda i: records[i]["id"])
    protected = sorted(i for i in range(len(records)) if cell(i) == 2 and motif(i) % 2 == 1)
    groups = {label: [i for i in permitted if motif(i) == label] for label in (0, 2, 4, 6)}
    additions = [groups[(0, 2, 4, 6)[k % 4]][k // 4] for k in range(277)]
    selection = {"training_count": 1386, "requested_rates": [0, 0.01, 0.05, 0.2],
                 "selection_uses_test_outcomes": False, "prediction_classes": list(range(8)),
                 "uniform_eight_class_chance": 0.125, "exposed_motifs": [0, 2, 4, 6],
                 "protected_motifs": [1, 3, 5, 7], "intervention_candidate_indices": permitted,
                 "intervention_candidate_ids": identity(permitted), "protected_all_indices": protected,
                 "protected_all_ids": identity(protected), "conditions": {}, "evaluation": {}}
    for name, parity in (("exposed_common", 0), ("protected_common", 1)):
        indices = [i for i in splits["ood_common"] if motif(i) % 2 == parity]
        selection["evaluation"][name] = {"indices": indices, "ids": identity(indices),
                                          "motif_counts": counts(indices),
                                          "position_counts": counts(indices, "y_position")}
    for seed in (0, 1, 2):
        (source / "checkpoints" / f"random_seed{seed}.pt").write_bytes(f"archived initial {seed}".encode())
        for condition in ("a", "b"):
            (source / "checkpoints" / f"{condition}_seed{seed}.pt").write_bytes(f"archived trained {condition} {seed}".encode())
            np.savez_compressed(source / "representations" / f"{condition}_seed{seed}.npz",
                                ids=np.array(identity(splits["analysis"])), local_0=np.zeros((len(splits["analysis"]), 2)))
    source_predictions = [{"model": f"{condition}_seed{seed}", "id": records[index]["id"],
                           **fake_prediction(records[index], index, seed)}
                          for condition in ("a", "b") for seed in (0, 1, 2) for index in splits["analysis"]]
    write_csv(source / "predictions.csv", source_predictions)
    write_manifest(source)
    source_manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    save_json(output / "source_reference.json", {"source_directory": str(source),
        "source_manifest_sha256": digest(source / "MANIFEST.json"), "required_source_sha256": source_manifest,
        "zero_intervention_reused_exactly": True})
    save_json(output / "environment.json", {"checkpoint_selection":
        "Final predeclared epoch; protected labels never used in selection."})
    for condition in ("a", "b"):
        base = splits[f"train_{condition}"]
        original_slots = {index: slot for slot, index in enumerate(base)}
        old_groups = {label: sorted((i for i in base if motif(i) == label), key=lambda i: records[i]["id"])
                      for label in (0, 2, 4, 6)}
        removals = [old_groups[(0, 2, 4, 6)[k % 4]][k // 4] for k in range(277)]
        items = []
        for rate, count, key in ((0, 0, "p0"), (0.01, 14, "p1"), (0.05, 69, "p5"), (0.2, 277, "p20")):
            train = list(base)
            for old, new in zip(removals[:count], additions[:count]):
                train[original_slots[old]] = new
            items.append({"rate": rate, "rate_key": key, "replaced_count": count,
                          "realized_fraction": count / 1386, "n_train": 1386,
                          "train_indices": train, "train_ids": identity(train),
                          "added_indices": additions[:count], "added_ids": identity(additions[:count]),
                          "removed_indices": removals[:count], "removed_ids": identity(removals[:count]),
                          "motif_counts": counts(train), "position_counts": counts(train, "y_position")})
            for seed in (0, 1, 2):
                name = f"{condition}_{key}_seed{seed}"
                checkpoint, representation = output / "checkpoints" / f"{name}.pt", output / "representations" / f"{name}.npz"
                if rate == 0:
                    shutil.copyfile(source / "checkpoints" / f"{condition}_seed{seed}.pt", checkpoint)
                    shutil.copyfile(source / "representations" / f"{condition}_seed{seed}.npz", representation)
                else:
                    torch.save({"condition": condition, "seed": seed, "rate": rate, "config": config,
                                "source_initialization_sha256": digest(source / "checkpoints" / f"random_seed{seed}.pt")}, checkpoint)
                    np.savez_compressed(representation, ids=np.array(identity(splits["analysis"])))
        selection["conditions"][condition] = items
    save_json(output / "selection_audit.json", selection)
    add_result_fixture(source, output, records, splits, config, selection)
    write_manifest(output)
    return source, output, records, splits


class CorrelationAuditTests(unittest.TestCase):
    def test_independent_clean_archive_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            result = audit_correlation(output, source)
            self.assertEqual(result["status"], "PASS", result["failures"])
            self.assertFalse(result["source_generator_imported"])
            self.assertEqual(result["device"], "cpu")

    def test_protected_training_leak_and_fabricated_ids_are_detected_even_after_rehashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, records, _ = fixture(Path(temporary))
            selection = json.loads((output / "selection_audit.json").read_text(encoding="utf-8"))
            item = selection["conditions"]["a"][3]
            protected = selection["evaluation"]["protected_common"]["indices"][0]
            item["train_indices"][0] = protected
            item["train_ids"][0] = records[protected]["id"]
            item["added_ids"][0] = "invented identity"
            save_json(output / "selection_audit.json", selection)
            write_manifest(output)
            result = audit_correlation(output, source)
            self.assertEqual(result["status"], "FAIL")
            for name in ("protected_cell_two_never_used_in_training", "no_validation_reserve_or_test_training_overlap",
                         "declared_added_and_removed_ids_match_actual_set_differences",
                         "same_motif_replacements_and_original_training_slots_reproduce"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_non_nested_sampling_and_cross_experience_mismatch_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            selection = json.loads((output / "selection_audit.json").read_text(encoding="utf-8"))
            item = selection["conditions"]["b"][2]
            item["added_indices"] = list(reversed(item["added_indices"]))
            item["added_ids"] = list(reversed(item["added_ids"]))
            save_json(output / "selection_audit.json", selection)
            write_manifest(output)
            result = audit_correlation(output, source)
            for name in ("intervention_sets_are_nested", "same_intervention_ids_across_experiences",
                         "predeclared_new_identity_order_reproduces_without_test_outcomes"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_zero_checkpoint_reuse_and_random_initialization_reference_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            (output / "checkpoints/a_p0_seed0.pt").write_bytes(b"changed zero control")
            (output / "representations/b_p0_seed1.npz").write_bytes((output / "representations/a_p1_seed0.npz").read_bytes())
            checkpoint = output / "checkpoints/a_p1_seed0.pt"
            metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata["source_initialization_sha256"] = "fabricated initialization"
            torch.save(metadata, checkpoint)
            write_manifest(output)
            result = audit_correlation(output, source)
            for name in ("zero_rate_checkpoint_blob_exactly_matches_source",
                         "zero_rate_representation_blob_exactly_matches_source",
                         "nonzero_checkpoint_references_exact_archived_initialization"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_source_corruption_and_unmanifested_output_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            (source / "checkpoints/random_seed0.pt").write_bytes(b"source changed")
            (output / "unrecorded.txt").write_text("extra", encoding="utf-8")
            result = audit_correlation(output, source)
            for name in ("every_source_manifest_hash_matches", "every_intervention_file_is_manifested",
                         "recorded_source_hashes_match_actual_files"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_bad_exposure_count_budget_and_selection_declaration_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            selection = json.loads((output / "selection_audit.json").read_text(encoding="utf-8"))
            selection["conditions"]["a"][1]["replaced_count"] = 15
            selection["selection_uses_test_outcomes"] = True
            save_json(output / "selection_audit.json", selection)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            config["epochs"] = 1
            save_json(output / "config.json", config)
            write_manifest(output)
            result = audit_correlation(output, source)
            for name in ("exact_rates_and_replacement_counts", "selection_declares_no_test_outcome_use",
                         "source_architecture_and_training_budget_preserved", "predeclared_hard_training_budget"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_manifest_cannot_reference_files_outside_the_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
            manifest["sha256"]["../outside.txt"] = "unsafe"
            save_json(output / "MANIFEST.json", manifest)
            with self.assertRaisesRegex(ValueError, "leaves its record"):
                audit_correlation(output, source)

    def test_wrong_denominators_behavior_scalar_and_summary_SD_detected_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            with (output / "behavior.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["n"] = str(int(rows[0]["n"]) + 1)
            rows[1]["motif_accuracy"] = "0.75"
            write_csv(output / "behavior.csv", rows)
            summary = json.loads((output / "report_summary.json").read_text(encoding="utf-8"))
            summary[0]["motif_accuracy_std"] = 0.0
            save_json(output / "report_summary.json", summary)
            write_manifest(output)
            result = audit_correlation(output, source)
            for name in ("behavior_uses_correct_test_identity_denominator", "behavior_metrics_independently_recomputed_from_predictions",
                         "summary_mean_and_sample_SD_recomputed_across_three_seeds"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_missing_seed_and_prediction_truth_corruption_detected_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            with (output / "behavior.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            write_csv(output / "behavior.csv", [row for row in rows if row["seed"] != "2"])
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            predictions[0]["true_family"] = str((int(predictions[0]["true_family"]) + 1) % 8)
            predictions.pop()
            write_csv(output / "predictions.csv", predictions)
            write_manifest(output)
            result = audit_correlation(output, source)
            for name in ("all_72_behavior_seed_rate_split_panels_present", "summary_contains_all_three_independent_seeds",
                         "prediction_truth_matches_frozen_molecular_metadata", "all_24_models_have_exact_ordered_analysis_predictions"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_wrong_four_class_chance_and_prior_baseline_detected_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, _, _ = fixture(Path(temporary))
            with (output / "baselines.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["n_prediction_classes"] = "4"
            rows[1]["uniform_accuracy"] = "0.25"
            rows[1]["training_prior_expected_accuracy"] = "0.25"
            write_csv(output / "baselines.csv", rows)
            write_manifest(output)
            result = audit_correlation(output, source)
            for name in ("baseline_denominators_classes_and_training_priors_match", "baseline_values_independently_recomputed"):
                self.assertGreater(result["checks"][name]["failed"], 0, name)

    def test_cli_refuses_audit_overwrite_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--record", str(Path(temporary) / "missing"), "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
