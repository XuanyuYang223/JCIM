"""Measure branch effects on saved v5 outputs without fitting any parameters.

All logits are reconstructed from archived readout vectors and final heads. The
stored v5 permutation is reused, and every reconstructed class prediction must
match its frozen CSV record before numerical effects are reported. Margins are
correct-class logits minus the largest other-class logit; positive margin changes
favor the correct class. Logit differences and margins use float64 arithmetic
after the original float32 head evaluation.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .run import sha256, write_csv, write_json


TASKS = ("family", "position", "distance")
INTERVENTIONS = ("intact", "zero_branch", "permuted_branch")
SPLITS = ("iid_a", "iid_b", "ood_common")
CASE_FIELDS = ("regime", "variant", "experience", "seed")
SUMMARY_FIELDS = (*CASE_FIELDS, "split", "task", "intervention")
QUANTILES = (("q05", .05), ("q25", .25), ("q50", .5), ("q75", .75), ("q95", .95))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _case(row):
    return tuple(int(row[name]) if name == "seed" else row[name] for name in CASE_FIELDS)


def _case_name(case):
    return f"{case[0]}_{case[1]}_{case[2]}_seed{case[3]}"


@torch.no_grad()
def logits_from_readouts(state, readouts, batch_size=128):
    """Evaluate original heads at a fixed batch shape, including the final tail."""
    if batch_size < 1 or readouts.ndim != 2 or not len(readouts):
        raise ValueError("A positive batch size and a nonempty readout matrix are required.")
    if readouts.dtype != torch.float32 or readouts.device.type != "cpu":
        raise ValueError("Archived head reconstruction requires CPU float32 readouts.")
    result = {task: [] for task in TASKS}
    for offset in range(0, len(readouts), batch_size):
        batch = readouts[offset:offset + batch_size]
        genuine = len(batch)
        if genuine != batch_size:
            batch = batch[torch.arange(batch_size) % genuine]
        for task in TASKS:
            hidden = F.relu(F.linear(batch, state[f"heads.{task}.0.weight"], state[f"heads.{task}.0.bias"]))
            logits = F.linear(hidden, state[f"heads.{task}.2.weight"], state[f"heads.{task}.2.bias"])
            result[task].append(logits[:genuine].numpy())
    return {task: np.concatenate(values) for task, values in result.items()}


def paired_effects(intact, altered, truth):
    """Return per-example output changes, including accuracy-neutral label flips."""
    intact, altered = np.asarray(intact, dtype=np.float64), np.asarray(altered, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if (intact.ndim != 2 or intact.shape != altered.shape or intact.shape[1] < 2
            or truth.shape != (len(intact),) or np.any(truth < 0) or np.any(truth >= intact.shape[1])
            or not np.isfinite(intact).all() or not np.isfinite(altered).all()):
        raise ValueError("Finite paired class-logit matrices and valid truth labels are required.")
    rows = np.arange(len(truth))
    other_intact, other_altered = intact.copy(), altered.copy()
    other_intact[rows, truth] = other_altered[rows, truth] = -np.inf
    margin_intact = intact[rows, truth] - other_intact.max(axis=1)
    margin_altered = altered[rows, truth] - other_altered.max(axis=1)
    delta = altered - intact
    centered_delta = delta - delta.mean(axis=1, keepdims=True)
    centered_intact = intact - intact.mean(axis=1, keepdims=True)
    centered_l2 = np.linalg.norm(centered_delta, axis=1)
    intact_prediction, altered_prediction = intact.argmax(axis=1), altered.argmax(axis=1)
    intact_correct, altered_correct = intact_prediction == truth, altered_prediction == truth
    changed = intact_prediction != altered_prediction
    return {
        "truth": truth, "intact_prediction": intact_prediction, "altered_prediction": altered_prediction,
        "intact_correct": intact_correct, "altered_correct": altered_correct, "prediction_changed": changed,
        "wrong_to_correct": ~intact_correct & altered_correct,
        "correct_to_wrong": intact_correct & ~altered_correct,
        "both_wrong_changed": ~intact_correct & ~altered_correct & changed,
        "margin_intact": margin_intact, "margin_altered": margin_altered,
        "margin_delta": margin_altered - margin_intact,
        "absolute_margin_delta": np.abs(margin_altered - margin_intact),
        "raw_logit_l2": np.linalg.norm(delta, axis=1), "centered_logit_l2": centered_l2,
        "relative_centered_logit_l2": centered_l2 / np.maximum(np.linalg.norm(centered_intact, axis=1), 1e-12),
        "max_abs_logit_delta": np.abs(delta).max(axis=1),
        "logits_exactly_unchanged": np.all(delta == 0, axis=1),
        "centered_logits_exactly_unchanged": np.all(centered_delta == 0, axis=1),
        "margin_improved": margin_altered > margin_intact,
        "margin_worsened": margin_altered < margin_intact,
        "margin_exactly_unchanged": margin_altered == margin_intact,
        "margin_negative_to_positive": (margin_intact < 0) & (margin_altered > 0),
        "margin_positive_to_negative": (margin_intact > 0) & (margin_altered < 0),
        "intact_margin_tie": margin_intact == 0, "altered_margin_tie": margin_altered == 0,
    }


def summarize_effects(effects, selected):
    """Summarize a fixed split; rates use every molecule in that split."""
    selected = np.asarray(selected, dtype=np.int64)
    if not len(selected):
        raise ValueError("An empty split cannot be summarized.")
    result = {"n": len(selected)}
    ignored = {"truth", "intact_prediction", "altered_prediction"}
    for name, values in effects.items():
        if name in ignored:
            continue
        values = np.asarray(values)[selected]
        if values.dtype == bool:
            result[f"{name}_count"] = int(values.sum())
            result[f"{name}_rate"] = float(values.mean())
        else:
            result[f"{name}_mean"] = float(values.mean())
            result[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            for label, probability in QUANTILES:
                result[f"{name}_{label}"] = float(np.quantile(values, probability))
    return result


def aggregate_seeds(rows):
    """Average per-seed summaries without treating molecules as independent seeds."""
    grouping = tuple(name for name in SUMMARY_FIELDS if name != "seed")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in grouping)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        seed_values = [row["seed"] for row in values]
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("Duplicate seeds in a branch-summary panel.")
        summary = {**dict(zip(grouping, key)), "n_seeds": len(values), "n_molecules_per_seed": values[0]["n"]}
        if len({row["n"] for row in values}) != 1:
            raise ValueError("Every seed must evaluate the same molecular split.")
        for field in values[0]:
            if field in SUMMARY_FIELDS or field == "n" or field.endswith("_count"):
                continue
            numeric = np.asarray([row[field] for row in values], dtype=np.float64)
            summary[f"{field}_seed_mean"] = float(numeric.mean())
            summary[f"{field}_seed_sd"] = float(numeric.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def run_branch_diagnostics(skip_source, hard_source, output, config=None):
    """Write a new output record and return its per-seed and seed-aggregate rows.

    ``config`` may set ``cpu_threads`` (default 4), ``seeds``, ``regimes``, and
    ``experiences``. Both augmented arms are mandatory for every selected case.
    The inference batch size is inherited from v5 and cannot be changed. A subset
    run remains a smoke-only analysis.
    No optimizer, new label, regenerated representation, or molecular input is
    used. Labels are read only to evaluate predictions and correct-class margins.
    """
    config = {} if config is None else dict(config)
    skip_source, hard_source, output = map(lambda value: Path(value).resolve(), (skip_source, hard_source, output))
    if any(output.is_relative_to(source) for source in (skip_source, hard_source)):
        raise ValueError("Diagnostics must be outside every frozen source record.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    references = {}
    manifests = {source: json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
                 for source in (skip_source, hard_source)}

    def checked(source, relative):
        path = source / relative
        expected = manifests[source].get(relative)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"An archived input failed its manifest: {path}")
        references[str(path)] = expected
        return path

    skip_config = json.loads(checked(skip_source, "config.json").read_text(encoding="utf-8"))
    batch_size = int(skip_config["batch_size"])
    if "batch_size" in config and int(config["batch_size"]) != batch_size:
        raise ValueError("The archived fixed inference batch size cannot change.")
    cpu_threads = int(config.get("cpu_threads", 4))
    if cpu_threads < 1:
        raise ValueError("A positive CPU thread count is required.")
    plan = json.loads(checked(skip_source, "plan.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in checked(hard_source, "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads(checked(hard_source, "splits.json").read_text(encoding="utf-8"))
    analysis = splits["analysis"]
    ids = np.asarray([records[index]["id"] for index in analysis])
    if plan["tests"]["analysis"] != {"indices": analysis, "ids": ids.tolist()}:
        raise ValueError("The original and v5 analysis identities or order differ.")
    split_by_index = {index: split for split in SPLITS for index in splits[split]}
    if len(split_by_index) != sum(len(splits[split]) for split in SPLITS) or set(split_by_index) != set(analysis):
        raise ValueError("The three test splits must partition the archived analysis set.")
    selected = {}
    for field, available in (("seeds", skip_config["seeds"]),
                             ("regimes", ("end_to_end", "frozen_readout")), ("experiences", ("a", "b"))):
        chosen = config.get(field, available)
        if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(available):
            raise ValueError(f"Choose distinct archived {field}.")
        selected[field] = list(chosen)
    inventory = [row for row in _csv(checked(skip_source, "inventory.csv")) if row["variant"] != "deep"
                 and int(row["seed"]) in selected["seeds"] and row["regime"] in selected["regimes"]
                 and row["experience"] in selected["experiences"]]
    cases = [_case(row) for row in inventory]
    expected = set(itertools.product(selected["regimes"], ("deep_control", "shallow_skip"),
                                     selected["experiences"], selected["seeds"]))
    if set(cases) != expected or len(cases) != len(expected):
        raise ValueError("Every augmented archived model is required exactly once.")
    saved = {}
    for row in _csv(checked(skip_source, "predictions.csv")):
        if row["variant"] == "deep" or _case(row) not in expected:
            continue
        key = (*_case(row), row["intervention"], int(row["index"]))
        if key in saved:
            raise ValueError("Duplicate frozen prediction rows.")
        saved[key] = row
    expected_prediction_rows = len(cases) * len(INTERVENTIONS) * len(analysis)
    if len(saved) != expected_prediction_rows:
        raise ValueError("The frozen prediction panel is incomplete.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "representations").mkdir()
    dense = {"indices": np.asarray(analysis), "ids": ids}
    summaries, comparison_count, prediction_rows_checked = [], 0, 0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        with (output / "branch_rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = None
            for case in cases:
                name = _case_name(case)
                metadata = dict(zip(CASE_FIELDS, case))
                with np.load(checked(skip_source, f"representations/{name}.npz")) as archive:
                    if not np.array_equal(archive["indices"], analysis) or not np.array_equal(archive["ids"], ids):
                        raise ValueError(f"A saved readout has different molecular identities: {name}")
                    array = archive["readout"].copy()
                if array.shape != (len(analysis), 512) or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid archived augmented readouts: {name}")
                readout = torch.from_numpy(array)
                state = torch.load(checked(skip_source, f"checkpoints/{name}.pt"), map_location="cpu", weights_only=True)["state_dict"]
                permutation = np.asarray(plan["branch_permutations"][str(case[3])]["indices"], dtype=np.int64)
                if sorted(permutation.tolist()) != list(range(len(analysis))):
                    raise ValueError("The stored branch intervention is not a permutation.")
                predicted = {}
                for intervention in INTERVENTIONS:
                    altered = readout
                    if intervention != "intact":
                        branch = torch.zeros_like(readout[:, 448:]) if intervention == "zero_branch" else readout[permutation, 448:]
                        altered = torch.cat((readout[:, :448], branch), dim=1)
                    logits = logits_from_readouts(state, altered, batch_size)
                    predicted[intervention] = logits
                    for task in TASKS:
                        values = logits[task]
                        if not np.isfinite(values).all():
                            raise ValueError(f"Non-finite reconstructed logits: {name}, {task}")
                        classes = values.argmax(axis=1)
                        dense[f"{name}__{intervention}__{task}"] = values
                        for offset, index in enumerate(analysis):
                            frozen = saved.get((*case, intervention, index))
                            if frozen is None or frozen["id"] != ids[offset] or int(frozen[f"pred_{task}"]) != int(classes[offset]):
                                raise AssertionError(f"Reconstructed class disagrees with frozen v5: {name}, {intervention}, {task}, {index}")
                            prediction_rows_checked += 1
                for task in TASKS:
                    truth = np.asarray([records[index][f"y_{task}"] for index in analysis])
                    for intervention in INTERVENTIONS[1:]:
                        effects = paired_effects(predicted["intact"][task], predicted[intervention][task], truth)
                        for offset, index in enumerate(analysis):
                            row = {**metadata, "split": split_by_index[index], "task": task,
                                   "intervention": intervention, "index": index, "id": str(ids[offset]),
                                   **{key: value[offset].item() for key, value in effects.items()}}
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            comparison_count += 1
                        for split in (*SPLITS, "analysis"):
                            split_positions = [offset for offset, index in enumerate(analysis)
                                               if split == "analysis" or split_by_index[index] == split]
                            summaries.append({**metadata, "split": split, "task": task, "intervention": intervention,
                                              **summarize_effects(effects, split_positions)})
    finally:
        torch.set_num_threads(previous_threads)
    aggregate = aggregate_seeds(summaries)
    write_csv(output / "branch_summary.csv", summaries)
    write_csv(output / "branch_seed_aggregate.csv", aggregate)
    np.savez_compressed(output / "representations/branch_logits.npz", **dense)
    if any(sha256(Path(path)) != digest for path, digest in references.items()):
        raise AssertionError("A frozen input changed during branch diagnostics.")
    validation = {"status": "PASS", "case_count": len(cases), "compared_rows": comparison_count,
        "prediction_rows_checked": prediction_rows_checked, "argmax_mismatches": 0,
        "summary_rows": len(summaries), "seed_aggregate_rows": len(aggregate),
        "cpu_threads": cpu_threads, "inference_batch_size": batch_size,
        "head_dtype": "float32", "diagnostic_arithmetic_dtype": "float64",
        "margin_definition": "correct-class logit minus maximum other-class logit",
        "delta_definition": "altered minus intact; positive margin delta favors the correct class",
        "rate_denominator": "All molecules in each split; no conditioning on initially wrong predictions",
        "sample_sd": "ddof=1; singleton summaries use 0",
        "quantiles": "NumPy default linear interpolation at 0.05, 0.25, 0.50, 0.75, 0.95",
        "relative_centered_l2_denominator": "Maximum of intact centered-logit L2 and 1e-12",
        "ties": "Argmax selects the first maximal class; zero correct-class margin is reported separately",
        "target_labels_added_to_training": 0, "previous_inputs_unchanged": True,
        "selected_cases": selected,
        "pipeline_only": (selected["seeds"] != [0, 1, 2] or len(selected["regimes"]) != 2
                          or len(selected["experiences"]) != 2 or skip_config.get("epochs") != 120)}
    write_json(output / "validation.json", validation)
    write_json(output / "source_reference.json", {"skip_source": str(skip_source), "hard_source": str(hard_source),
        "input_sha256": references, "skip_manifest_sha256": sha256(skip_source / "MANIFEST.json"),
        "hard_manifest_sha256": sha256(hard_source / "MANIFEST.json")})
    return {"output": str(output), "summaries": summaries, "seed_aggregate": aggregate, "validation": validation,
            "files": {"rows": "branch_rows.csv", "summary": "branch_summary.csv",
                      "seed_aggregate": "branch_seed_aggregate.csv", "validation": "validation.json",
                      "source_reference": "source_reference.json", "logits": "representations/branch_logits.npz"}}
