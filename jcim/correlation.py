"""Fixed-size counterexample interventions with protected molecular combinations.

Even-numbered motifs supply nested counterexamples in their original cell-2
combination. Odd-numbered motifs remain protected in cell 2 at every rate.
The eight-way prediction head is never restricted to the four evaluated motifs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

from .hard_data import heldout_cell
from .hard_model import HardMolecularGNN
from .hard_run import metric_row
from .hard_verify import verify_hard
from .run import ROOT, infer, sha256, train_model, write_csv, write_json


DEFAULT_RATES = (0.0, 0.01, 0.05, 0.20)
EXPOSED_MOTIFS = (0, 2, 4, 6)
PROTECTED_MOTIFS = (1, 3, 5, 7)


def rate_key(rate: float) -> str:
    """Return an exact, filename-safe percentage identifier."""
    return f"p{100 * rate:g}".replace(".", "d")


def _counts(records, indices, label="y_family"):
    return {str(key): value for key, value in sorted(Counter(
        int(records[index][label]) for index in indices).items())}


def _ids(records, indices):
    return [records[index]["id"] for index in indices]


def build_intervention_plan(records, splits, rates=DEFAULT_RATES, expected_count=None):
    """Declare nested same-motif replacements without consulting test outcomes.

    Replacements preserve every original odd-motif molecule, each motif's count,
    and the original training list order outside replaced slots. Both training
    experiences receive exactly the same intervention identities. Candidate
    selection is sorted by molecular ID and balanced round robin over even
    motifs, independent of seeds and labels beyond the declared design factors.
    """
    rates = [float(rate) for rate in rates]
    if not rates or rates[0] != 0 or rates != sorted(set(rates)):
        raise ValueError("Intervention rates must be unique, increasing, and start at zero.")
    if any(not np.isfinite(rate) or rate < 0 or rate > 1 for rate in rates):
        raise ValueError("Intervention rates must lie in [0, 1].")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Molecular identities must be unique.")
    required = ("candidate_train", "train_a", "train_b", "val", "val_a", "val_b",
                "discarded_val_ood", "adaptation_reserve", "analysis", "ood_common",
                "iid_a", "iid_b")
    for name in required:
        indices = splits[name]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= len(records) for i in indices):
            raise ValueError(f"Invalid indices in {name}.")
    candidate = set(splits["candidate_train"])
    forbidden = set(splits["val"]) | set(splits["discarded_val_ood"]) \
        | set(splits["adaptation_reserve"]) | set(splits["analysis"])
    if candidate & forbidden:
        raise ValueError("Candidate training identities overlap validation, reserve, or test.")
    exposed = sorted((i for i in candidate if heldout_cell(records[i]) == 2
                      and int(records[i]["y_family"]) in EXPOSED_MOTIFS),
                     key=lambda i: records[i]["id"])
    protected = {i for i, record in enumerate(records)
                 if heldout_cell(record) == 2 and int(record["y_family"]) in PROTECTED_MOTIFS}
    tests = {
        "exposed_common": [i for i in splits["ood_common"]
                           if int(records[i]["y_family"]) in EXPOSED_MOTIFS],
        "protected_common": [i for i in splits["ood_common"]
                             if int(records[i]["y_family"]) in PROTECTED_MOTIFS],
    }
    if any(not indices for indices in tests.values()):
        raise ValueError("Both exposed and protected test subsets must be nonempty.")
    if set(tests["exposed_common"]) | set(tests["protected_common"]) != set(splits["ood_common"]):
        raise ValueError("The common test must contain only the declared eight motifs.")
    if protected & (set(splits["val_a"]) | set(splits["val_b"])):
        raise ValueError("Protected combinations must not enter validation.")
    bases = {condition: list(splits[f"train_{condition}"]) for condition in ("a", "b")}
    count = len(bases["a"])
    if not count or len(bases["b"]) != count or (expected_count is not None and count != expected_count):
        raise ValueError("Both experiences must have the declared equal training size.")
    if _counts(records, bases["a"]) != _counts(records, bases["b"]):
        raise ValueError("Base experiences must have identical per-motif counts.")
    grouped_new = {motif: [i for i in exposed if int(records[i]["y_family"]) == motif]
                   for motif in EXPOSED_MOTIFS}
    replaced_counts = [int(round(rate * count)) for rate in rates]
    max_replacements = max(replaced_counts)
    new_order = []
    for slot in range(max_replacements):
        motif = EXPOSED_MOTIFS[slot % len(EXPOSED_MOTIFS)]
        offset = slot // len(EXPOSED_MOTIFS)
        if offset >= len(grouped_new[motif]):
            raise ValueError(f"Insufficient intervention candidates for motif {motif}.")
        new_order.append(grouped_new[motif][offset])
    plan = {
        "protocol": "Nested same-motif replacement; even motifs exposed, odd motifs protected.",
        "requested_rates": rates, "training_count": count,
        "replacement_rounding": "Python round(rate * training_count), nearest integer; ties to even.",
        "exposed_motifs": list(EXPOSED_MOTIFS), "protected_motifs": list(PROTECTED_MOTIFS),
        "prediction_classes": list(range(8)), "uniform_eight_class_chance": 1 / 8,
        "selection_uses_test_outcomes": False, "conditions": {},
        "intervention_candidate_indices": exposed, "intervention_candidate_ids": _ids(records, exposed),
        "protected_all_indices": sorted(protected), "protected_all_ids": _ids(records, sorted(protected)),
        "evaluation": {name: {"indices": indices, "ids": _ids(records, indices),
                              "motif_counts": _counts(records, indices),
                              "position_counts": _counts(records, indices, "y_position")}
                       for name, indices in tests.items()},
    }
    for condition, base in bases.items():
        if set(base) - candidate or set(base) & forbidden or set(base) & protected:
            raise ValueError("Base pretraining identities violate the declared partitions.")
        expected_cell = 0 if condition == "a" else 1
        if any(heldout_cell(records[i]) != expected_cell for i in base):
            raise ValueError("Base experience has an incorrect checkerboard cell.")
        grouped_old = {motif: sorted((i for i in base if int(records[i]["y_family"]) == motif),
                                     key=lambda i: records[i]["id"])
                       for motif in EXPOSED_MOTIFS}
        removed_order = []
        for slot in range(max_replacements):
            motif, offset = EXPOSED_MOTIFS[slot % 4], slot // 4
            if offset >= len(grouped_old[motif]):
                raise ValueError(f"Insufficient base identities of motif {motif}.")
            removed_order.append(grouped_old[motif][offset])
        base_slots = {index: slot for slot, index in enumerate(base)}
        plans = []
        for rate, replaced in zip(rates, replaced_counts):
            train = list(base)
            for old, new in zip(removed_order[:replaced], new_order[:replaced]):
                train[base_slots[old]] = new
            if len(set(train)) != count or set(train) & (forbidden | protected):
                raise ValueError("An intervention violates size or protected identity isolation.")
            if _counts(records, train) != _counts(records, base):
                raise ValueError("Same-motif replacements changed the motif priors.")
            old_odd = {i for i in base if int(records[i]["y_family"]) in PROTECTED_MOTIFS}
            if not old_odd <= set(train):
                raise ValueError("An intervention removed a protected-motif base identity.")
            plans.append({"rate": rate, "rate_key": rate_key(rate), "replaced_count": replaced,
                          "realized_fraction": replaced / count, "train_indices": train,
                          "train_ids": _ids(records, train),
                          "added_indices": new_order[:replaced],
                          "added_ids": _ids(records, new_order[:replaced]),
                          "removed_indices": removed_order[:replaced],
                          "removed_ids": _ids(records, removed_order[:replaced]),
                          "motif_counts": _counts(records, train),
                          "position_counts": _counts(records, train, "y_position"),
                          "n_train": count, "protected_overlap": 0,
                          "validation_reserve_test_overlap": 0,
                          "all_original_odd_motif_molecules_preserved": True})
        plan["conditions"][condition] = plans
    return plan


def baseline_rows(records, training_indices, evaluation, condition, rate):
    """Evaluate chance using all eight output labels and training-only priors."""
    counts = np.bincount([records[i]["y_family"] for i in training_indices], minlength=8)
    probabilities = counts / counts.sum()
    majority = int(np.argmax(counts))
    rows = []
    for name, indices in evaluation.items():
        truth = np.array([records[i]["y_family"] for i in indices])
        present = np.unique(truth)
        rows.append({"condition": condition, "rate": rate, "split": name, "n": len(indices),
                     "n_true_classes": len(present), "n_prediction_classes": 8,
                     "uniform_accuracy": 1 / 8, "uniform_macro_recall": 1 / 8,
                     "training_prior_expected_accuracy": float(probabilities[truth].mean()),
                     "training_prior_expected_macro_recall": float(probabilities[present].mean()),
                     "training_majority_label": majority,
                     "training_majority_accuracy": float(np.mean(truth == majority)),
                     "training_majority_macro_recall": float(np.mean(present == majority)),
                     "training_class_counts": json.dumps(counts.tolist())})
    return rows


def summarize_behavior(rows):
    """Report seed variability without treating molecule rows as replicates."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], float(row["rate"]), row["split"])].append(row)
    output = []
    for (condition, rate, split), group in sorted(groups.items()):
        summary = {"condition": condition, "rate": rate, "split": split,
                   "n_seeds": len(group), "seeds": [int(row["seed"]) for row in group],
                   "replaced_count": int(group[0]["replaced_count"]), "n_test": int(group[0]["n"])}
        for metric in ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy",
                       "distance_accuracy", "joint_accuracy"):
            values = np.array([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def generate_correlation_report(output):
    """Regenerate an English report and plots from retained per-seed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    with (output / "behavior.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_behavior(rows)
    write_json(output / "report_summary.json", summary)
    plan = json.loads((output / "selection_audit.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    names = (("iid", "Original familiar combinations"),
             ("exposed_common", "Directly exposed combinations"),
             ("protected_common", "Protected unexposed combinations"))
    for axis, (split, title) in zip(axes, names):
        for condition, color in (("a", "#0072B2"), ("b", "#D55E00")):
            group = [row for row in summary if row["condition"] == condition and row["split"] == split]
            x = [100 * row["rate"] for row in group]
            y = [100 * row["motif_accuracy_mean"] for row in group]
            errors = [100 * row["motif_accuracy_std"] for row in group]
            axis.errorbar(x, y, yerr=errors, label=f"Experience {condition.upper()}",
                          color=color, marker="o", capsize=3)
            for seed in config["seeds"]:
                seed_rows = sorted((row for row in rows if row["condition"] == condition
                                    and row["split"] == split and int(row["seed"]) == seed),
                                   key=lambda row: float(row["rate"]))
                axis.plot([100 * float(row["rate"]) for row in seed_rows],
                          [100 * float(row["motif_accuracy"]) for row in seed_rows],
                          color=color, alpha=0.18, linewidth=0.8)
        axis.axhline(12.5, color="#666666", linestyle="--", linewidth=1,
                     label="Uniform 8-class chance")
        axis.set(title=title, xlabel="Training identities replaced (%)", ylim=(-3, 103))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Motif accuracy (%)")
    axes[2].legend(fontsize=8, loc="best")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    figure.savefig(figures / "correlation_summary.png", dpi=180)
    figure.savefig(figures / "correlation_summary.svg")
    plt.close(figure)
    lines = ["# Correlation-strength intervention", "",
             "This experiment asks whether a few counterexamples to motif-position binding "
             "improve generalization to combinations that receive no counterexamples.", "",
             f"Every model uses {plan['training_count']:,} training identities and the same "
             f"{config['epochs']}-epoch budget, architecture, losses, and optimizer as Hard v2. "
             "The zero-intervention checkpoints, predictions, and representations are reused "
             "byte for byte or derived from the archived source predictions; they are not retrained.", "",
             "Counterexamples replace original training identities of the same motif. "
             "This preserves every motif prior and every original odd-motif identity. "
             "Even motifs 0, 2, 4, and 6 supply nested cell-2 counterexamples, balanced in a "
             "predeclared round-robin order and identical across experiences and seeds. "
             "Odd motifs 1, 3, 5, and 7 are never observed in cell 2 during training or validation.", "",
             "The exposed test measures learning of combinations directly represented by "
             "counterexamples. The protected test measures transfer of that intervention to "
             "different motif types whose cell-2 combinations remain unseen. Both tests contain "
             "four true motif classes, while predictions still range over all eight classes. "
             "Uniform chance is therefore 12.5%, not 25%. Training-prior and training-majority "
             "baselines are retained in `baselines.csv`.", "",
             "## Training-size and exposure audit", "",
             "| Requested replacement | Identities replaced | Realized replacement | Training identities |",
             "|---:|---:|---:|---:|"]
    for item in plan["conditions"]["a"]:
        lines.append(f"| {100 * item['rate']:g}% | {item['replaced_count']} | "
                     f"{100 * item['realized_fraction']:.3f}% | {item['n_train']} |")
    lines += ["", "## Motif recognition", "",
              "Values are means ± sample standard deviations across independent initialization "
              "seeds. Macro recall averages over classes present in each test; predictions are "
              "never restricted to those classes. No protected outcome selects a checkpoint, "
              "regularizer, intervention identity, or training duration.", "",
              "| Experience | Replacement | Test | Accuracy (%) | Macro recall (%) |",
              "|---|---:|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['condition'].upper()} | {100 * row['rate']:g}% | {row['split']} | "
                     f"{100 * row['motif_accuracy_mean']:.2f} ± {100 * row['motif_accuracy_std']:.2f} | "
                     f"{100 * row['motif_balanced_accuracy_mean']:.2f} ± "
                     f"{100 * row['motif_balanced_accuracy_std']:.2f} |")
    lines += ["", "## Every seed", "",
              "| Experience | Seed | Replacement | Test | Accuracy (%) | Macro recall (%) |",
              "|---|---:|---:|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition'].upper()} | {row['seed']} | {100 * float(row['rate']):g}% | "
                     f"{row['split']} | {100 * float(row['motif_accuracy']):.2f} | "
                     f"{100 * float(row['motif_balanced_accuracy']):.2f} |")
    lines += ["", "## Interpretation limits", "",
              "An exposed-set improvement alone establishes direct combination learning. "
              "Improvement on protected combinations would support broader transfer, while "
              "poor protected performance would show that the intervention did not induce "
              "such transfer under this fixed budget. The parity partition evaluates one "
              "predeclared partition of motif types; it does not establish independence from "
              "motif chemistry or robustness to every possible partition. These behavioral "
              "curves alone do not locate information in particular layers or establish a "
              "causal representation mechanism.", "",
              "The complete molecular selections are in `selection_audit.json`; per-seed "
              "three-task scores are in `behavior.csv`, individual predictions in "
              "`predictions.csv`, checkpoints and local/global representations in their "
              "respective directories, and source hashes in `source_reference.json`. "
              "The final epoch is fixed in advance; validation uses only original familiar "
              "combinations and does not select checkpoints.", "",
              "![Counterexample intervention curves](figures/correlation_summary.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _source_predictions(source):
    with (source / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        return {(row["model"], row["id"]): row for row in csv.DictReader(handle)}


def _verify_source_files(source, config):
    """Require every reused checkpoint and representation to be in the source manifest."""
    audit = verify_hard(source)
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    required = ["molecules.jsonl", "graphs.npz", "splits.json", "config.json", "predictions.csv"]
    for seed in config["seeds"]:
        required.append(f"checkpoints/random_seed{seed}.pt")
        for condition in ("a", "b"):
            required.extend((f"checkpoints/{condition}_seed{seed}.pt",
                             f"representations/{condition}_seed{seed}.npz"))
    normalized = {str(Path(name)).replace("\\", "/"): value for name, value in manifest.items()}
    for relative in required:
        if relative not in normalized or sha256(source / relative) != normalized[relative]:
            raise ValueError(f"A required source file is not checksum-protected: {relative}")
    return audit, {relative: normalized[relative] for relative in required}


def run_correlation(source, output, configuration=None):
    """Run the declared intervention study; refuse an existing nonempty output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if configuration is not None:
        config.update(configuration)
    # A new intervention cannot silently change the Hard v2 comparison budget.
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    for key in ("hidden_dim", "layers", "epochs", "batch_size", "learning_rate", "weight_decay"):
        if config[key] != source_config[key]:
            raise ValueError(f"Correlation interventions must preserve source {key}.")
    config["experiment"] = "correlation_v3"
    config["correlation_rates"] = list(config.get("correlation_rates", DEFAULT_RATES))
    config["correlation_train_count"] = config.get("correlation_train_count", 1386)
    source_audit, source_hashes = _verify_source_files(source, config)
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    plan = build_intervention_plan(records, splits, config["correlation_rates"], config["correlation_train_count"])
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    requested_device = config["device"]
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available()
                          else "cpu" if requested_device == "auto" else requested_device)
    with np.load(source / "graphs.npz", allow_pickle=False) as stored:
        tensors = {key: torch.from_numpy(stored[key]).to(device) for key in stored.files}
    if len(tensors["x"]) != len(records):
        raise ValueError("Archived tensors and molecular metadata have different lengths.")
    archived_predictions = _source_predictions(source)
    analysis = splits["analysis"]
    lookup = {index: row for row, index in enumerate(analysis)}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "representations"):
        (output / name).mkdir()
    write_json(output / "config.json", config)
    write_json(output / "selection_audit.json", plan)
    write_json(output / "source_reference.json", {
        "source_directory": str(source), "source_manifest_sha256": sha256(source / "MANIFEST.json"),
        "required_source_sha256": source_hashes, "source_verification": source_audit,
        "zero_intervention_reused_exactly": True, "initialization": "Archived random_seedN.pt state_dict.",
    })
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
                   "python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                     for path in sorted((ROOT / "jcim").glob("*.py"))},
                   "checkpoint_selection": "Final predeclared epoch; protected labels never used in selection.",
                   "minibatch_order": "Same seed permutation and original list slots; same-motif replacement in place.",
                   "determinism": "Seeded operations; bitwise equivalence across devices is not guaranteed."}
    write_json(output / "environment.json", environment)
    behavior, baselines, individual_predictions = [], [], []
    for seed in config["seeds"]:
        initial = torch.load(source / "checkpoints" / f"random_seed{seed}.pt", map_location="cpu", weights_only=True)["state_dict"]
        for condition in ("a", "b"):
            evaluation = {"iid": splits[f"iid_{condition}"],
                          **{name: item["indices"] for name, item in plan["evaluation"].items()}}
            original_name = f"{condition}_seed{seed}"
            for selection in plan["conditions"][condition]:
                name = f"{condition}_{selection['rate_key']}_seed{seed}"
                rate = selection["rate"]
                if rate == 0:
                    shutil.copyfile(source / "checkpoints" / f"{original_name}.pt", output / "checkpoints" / f"{name}.pt")
                    shutil.copyfile(source / "representations" / f"{original_name}.npz", output / "representations" / f"{name}.npz")
                    predicted = {task: np.array([int(archived_predictions[(original_name, records[i]["id"])][f"pred_{task}"])
                                                for i in analysis]) for task in ("family", "position", "distance")}
                else:
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
                    model.load_state_dict(initial)
                    elapsed = train_model(model, tensors, selection["train_indices"], splits[f"val_{condition}"],
                                          config, seed, output, name)
                    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                "condition": condition, "seed": seed, "rate": rate, "config": config,
                                "source_initialization_sha256": source_hashes[f"checkpoints/random_seed{seed}.pt"],
                                "training_seconds": elapsed}, output / "checkpoints" / f"{name}.pt")
                    predicted, reps = infer(model, tensors, analysis, config["batch_size"], representations=True)
                    np.savez_compressed(output / "representations" / f"{name}.npz",
                                        ids=np.array(_ids(records, analysis)),
                                        **{f"{pool}_{layer}": values for pool, layers in reps.items()
                                           for layer, values in enumerate(layers)})
                    del model, reps
                for split_name, indices in evaluation.items():
                    subset = [lookup[index] for index in indices]
                    row = metric_row(name, condition, seed, split_name,
                                     {task: values[subset] for task, values in predicted.items()}, indices, records)
                    row.update({"rate": rate, "replaced_count": selection["replaced_count"],
                                "realized_fraction": selection["realized_fraction"],
                                "source_model": original_name if rate == 0 else "new_training"})
                    behavior.append(row)
                    print(json.dumps(row), flush=True)
                if seed == config["seeds"][0]:
                    baselines.extend(baseline_rows(records, selection["train_indices"], evaluation, condition, rate))
                for row_index, index in enumerate(analysis):
                    individual_predictions.append({"model": name, "condition": condition, "seed": seed,
                        "rate": rate, "id": records[index]["id"],
                        **{f"true_{task}": records[index][f"y_{task}"] for task in predicted},
                        **{f"pred_{task}": int(values[row_index]) for task, values in predicted.items()}})
                write_csv(output / "behavior.csv", behavior)
                write_csv(output / "baselines.csv", baselines)
                write_csv(output / "predictions.csv", individual_predictions)
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", environment)
    generate_correlation_report(output)
    write_json(output / "MANIFEST.json", {"sha256": {
        str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"}})
    print(f"Completed scientific record: {output}", flush=True)
    return behavior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/correlation_v3")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    configuration = None if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
    run_correlation(args.source, args.output, configuration)


if __name__ == "__main__":
    main()
