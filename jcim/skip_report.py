"""Report a boundary-free shallow-readout intervention and matched controls."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = ("deep", "deep_control", "shallow_skip")
REGIMES = ("end_to_end", "frozen_readout")
INTERVENTIONS = ("intact", "zero_branch", "permuted_branch")
METRICS = ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy",
           "position_balanced_accuracy", "distance_accuracy", "distance_balanced_accuracy",
           "distance_mae", "joint_accuracy")
GROUP_KEYS = ("regime", "variant", "experience", "intervention", "split")
PAIR_KEYS = ("regime", "experience", "intervention", "split", "seed")
LABELS = {"deep": "Original deep 448D", "deep_control": "Deep + deep mean 512D",
          "shallow_skip": "Deep + shallow mean 512D"}
REGIME_LABELS = {"end_to_end": "End-to-end source training", "frozen_readout": "Frozen pretrained encoder"}
COLORS = {"deep": "#777777", "deep_control": "#cb7c35", "shallow_skip": "#247aab"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty derived table.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select(rows: list[dict], **filters) -> list[dict]:
    return [row for row in rows if all(str(row.get(key)) == str(value) for key, value in filters.items())]


def _lookup(rows: list[dict], **filters) -> dict:
    found = _select(rows, **filters)
    if len(found) != 1:
        raise ValueError(f"Expected one summary cell for {filters!r}, found {len(found)}.")
    return found[0]


def _stats(rows: list[dict], metric: str) -> dict:
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[str(row["seed"])].append(float(row[metric]))
    if any(len(values) != 1 for values in by_seed.values()):
        raise ValueError("Each endpoint summary must have one measurement per encoder seed.")
    per_seed = {seed: values[0] for seed, values in sorted(by_seed.items())}
    values = list(per_seed.values())
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values), "per_seed": per_seed}


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return [{**dict(zip(keys, key)), **{metric: _stats(members, metric) for metric in METRICS}}
            for key, members in sorted(grouped.items())]


def _paired_rows(rows: list[dict]) -> list[dict]:
    """Subtract the parameter-matched deep control within each exact seed."""
    panels = {variant: {} for variant in ("deep_control", "shallow_skip")}
    for row in rows:
        if row["variant"] not in panels:
            continue
        key = tuple(str(row[field]) for field in PAIR_KEYS)
        if key in panels[row["variant"]]:
            raise ValueError(f"Duplicate paired endpoint {key!r}.")
        panels[row["variant"]][key] = row
    control, skip = panels["deep_control"], panels["shallow_skip"]
    if not control or set(control) != set(skip):
        raise ValueError("The shallow skip and deep control lack exactly matched endpoints.")
    output = []
    for key, current in sorted(skip.items()):
        reference = control[key]
        if int(current["n"]) != int(reference["n"]):
            raise ValueError("Paired variants have different test counts.")
        output.append({**dict(zip(PAIR_KEYS, key)), "contrast": "shallow_skip_minus_deep_control",
                       "n_test": int(current["n"]),
                       **{metric: float(current[metric]) - float(reference[metric]) for metric in METRICS}})
    return output


def _validate(rows: list[dict], config: dict) -> None:
    if not rows:
        raise ValueError("Endpoint measurements must be nonempty.")
    seen = set()
    counts = {}
    for row in rows:
        required = (*GROUP_KEYS, "seed", "n", *METRICS)
        if any(field not in row or row[field] == "" for field in required):
            raise ValueError("An endpoint row has missing fields.")
        if row["regime"] not in REGIMES or row["variant"] not in VARIANTS or row["experience"] not in ("a", "b"):
            raise ValueError("Unknown regime, variant, or experience.")
        if row["intervention"] not in INTERVENTIONS or row["split"] not in ("iid_a", "iid_b", "ood_common"):
            raise ValueError("Unknown branch intervention or evaluation split.")
        if row["variant"] == "deep" and row["intervention"] != "intact":
            raise ValueError("The original deep model has no appended branch to intervene on.")
        if int(row["n"]) <= 0:
            raise ValueError("Evaluation sets must be nonempty.")
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value) or value < 0 or (metric != "distance_mae" and value > 1):
                raise ValueError(f"Invalid {metric}: {value!r}.")
        key = tuple(str(row[field]) for field in (*GROUP_KEYS, "seed"))
        if key in seen:
            raise ValueError(f"Duplicate endpoint {key!r}.")
        seen.add(key)
        old_count = counts.setdefault(row["split"], int(row["n"]))
        if old_count != int(row["n"]):
            raise ValueError("All conditions must evaluate identical split counts.")
    expected = {(regime, variant, experience, intervention, split, str(seed))
                for regime in config.get("regimes", REGIMES) for variant in config.get("variants", VARIANTS)
                for experience in config.get("experiences", ("a", "b"))
                for intervention in (("intact",) if variant == "deep" else INTERVENTIONS)
                for split in ("iid_a", "iid_b", "ood_common") for seed in config.get("seeds", (0, 1, 2))}
    if seen != expected:
        raise ValueError(f"Incomplete endpoint panel: missing={sorted(expected - seen)[:2]!r}; extra={sorted(seen - expected)[:2]!r}.")


def _inventory(rows: list[dict], config: dict) -> list[dict]:
    keys = ("regime", "variant", "experience", "seed")
    expected = {(regime, variant, experience, str(seed)) for regime in config.get("regimes", REGIMES)
                for variant in config.get("variants", VARIANTS) for experience in config.get("experiences", ("a", "b"))
                for seed in config.get("seeds", (0, 1, 2))}
    seen = set()
    output = []
    for row in rows:
        key = tuple(str(row[field]) for field in keys)
        if key in seen:
            raise ValueError("Duplicate parameter inventory case.")
        seen.add(key)
        dim = int(row["readout_dim"])
        if dim != (448 if row["variant"] == "deep" else 512):
            raise ValueError("Readout dimensions must be 448D or the matched 512D.")
        total, trainable = int(row["parameter_count"]), int(row["trainable_parameter_count"])
        if not 0 < trainable <= total:
            raise ValueError("Invalid parameter inventory counts.")
        if row["regime"] == "frozen_readout" and row["initial_encoder_sha256"] != row["final_encoder_sha256"]:
            raise ValueError("A frozen encoder changed during readout training.")
        output.append({**row, "readout_dim": dim, "parameter_count": total, "trainable_parameter_count": trainable})
    if seen != expected:
        raise ValueError("Incomplete parameter inventory panel.")
    for row in output:
        if row["variant"] != "shallow_skip":
            continue
        control = _lookup(output, regime=row["regime"], variant="deep_control", experience=row["experience"], seed=row["seed"])
        if any(row[key] != control[key] for key in ("readout_dim", "parameter_count", "trainable_parameter_count")):
            raise ValueError("The shallow skip and deep control must have identical parameter counts.")
    return output


def _histories(output: Path, inventory: list[dict], config: dict) -> list[dict]:
    rows = []
    for case in inventory:
        name = f"{case['regime']}_{case['variant']}_{case['experience']}_seed{case['seed']}"
        history = json.loads((output / "histories" / f"history_{name}.json").read_text(encoding="utf-8"))
        epochs = []
        for entry in history:
            if any("ood" in key.lower() or "test" in key.lower() for key in entry):
                raise ValueError("Training histories must exclude OOD and test evaluation.")
            epoch = int(entry["epoch"])
            if epoch <= 0 or epoch > int(config["epochs"]) or epoch in epochs:
                raise ValueError("Training history contains an invalid or duplicate epoch.")
            epochs.append(epoch)
            for metric in ("train_loss", "val_family_accuracy", "val_position_accuracy", "val_distance_accuracy"):
                value = float(entry[metric])
                if not np.isfinite(value) or value < 0 or (metric != "train_loss" and value > 1):
                    raise ValueError(f"Invalid training-history {metric}.")
            rows.append({**{key: case[key] for key in ("regime", "variant", "experience", "seed")}, **entry})
        if not epochs or epochs != sorted(epochs) or epochs[-1] != int(config["epochs"]):
            raise ValueError("Every training history must end at the predeclared final epoch.")
    return rows


def _format(stat: dict, *, delta: bool = False, mae: bool = False) -> str:
    scale = 1 if mae else 100
    suffix = " bonds" if mae else " pp" if delta else "%"
    deviation = "SD undefined" if stat["std"] is None else f"{scale * stat['std']:.1f}"
    return f"{scale * stat['mean']:.1f} +/- {deviation}{suffix}"


def _table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)]) + "\n"


def _make_figures(output: Path, summary: dict, histories: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    regimes, experiences = summary["regimes"], summary["experiences"]
    figure, axes = plt.subplots(len(regimes), len(experiences) + 1, figsize=(15, 4.0 * len(regimes)),
                                squeeze=False, constrained_layout=True)
    for row_index, regime in enumerate(regimes):
        for column, experience in enumerate(experiences):
            axis = axes[row_index, column]
            for offset, variant in enumerate(VARIANTS):
                cells = [_lookup(summary["behavior"], regime=regime, variant=variant, experience=experience,
                                 intervention="intact", split=split)["motif_accuracy"] for split in (f"iid_{experience}", "ood_common")]
                x = np.arange(2) + (offset - 1) * 0.24
                axis.bar(x, [cell["mean"] for cell in cells], width=0.22, color=COLORS[variant], label=LABELS[variant], alpha=0.85)
                axis.errorbar(x, [cell["mean"] for cell in cells], yerr=[cell["std"] or 0 for cell in cells],
                              fmt="none", color="#333333", capsize=2.5, linewidth=0.9)
                for point, cell in zip(x, cells):
                    values = list(cell["per_seed"].values())
                    axis.scatter(point + np.linspace(-0.045, 0.045, len(values)), values, color="#222222", s=11, zorder=3)
            axis.axhline(0.125, color="#aaaaaa", linestyle=":", linewidth=0.8)
            axis.set(title=f"{REGIME_LABELS[regime]}: {experience.upper()}", ylabel="Motif accuracy", ylim=(-0.03, 1.04))
            axis.set_xticks([0, 1], ["Own IID", "Common unseen combinations"])
            axis.tick_params(axis="x", labelsize=8)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(axis="y", alpha=0.15)
        axis = axes[row_index, -1]
        for offset, variant in enumerate(VARIANTS):
            cells = [_lookup(summary["generalization_gap"], regime=regime, variant=variant, experience=experience)["motif_accuracy"]
                     for experience in experiences]
            x = np.arange(len(experiences)) + (offset - 1) * 0.22
            axis.errorbar(x, [cell["mean"] for cell in cells], yerr=[cell["std"] or 0 for cell in cells],
                          fmt="o", color=COLORS[variant], capsize=3, label=LABELS[variant])
        axis.axhline(0, color="#aaaaaa", linewidth=0.8)
        axis.set(title="Own-IID minus unseen-combination gap", ylabel="Accuracy difference", ylim=(-0.08, 1.08))
        axis.set_xticks(np.arange(len(experiences)), [f"Experience {value.upper()}" for value in experiences])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.15)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    warning = "PIPELINE CHECK ONLY\n" if summary["pipeline_only"] else ""
    figure.suptitle(warning + "Boundary-free shallow readout: predeclared final-epoch endpoints\n"
                   "Dots: individual seeds; error bars: sample SD across seeds; no target-combination training labels", fontsize=12)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(figures / f"skip_primary.{extension}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(regimes), len(experiences), figsize=(12, 3.9 * len(regimes)),
                                squeeze=False, constrained_layout=True)
    for row_index, regime in enumerate(regimes):
        for column, experience in enumerate(experiences):
            axis = axes[row_index, column]
            accuracy_axis = axis.twinx()
            for variant in VARIANTS:
                selected = _select(histories, regime=regime, experience=experience, variant=variant)
                epochs = sorted({int(row["epoch"]) for row in selected})
                for metric, current_axis, style in (("train_loss", axis, "-"), ("val_family_accuracy", accuracy_axis, "--")):
                    cells = [_stats(_select(selected, epoch=epoch), metric) for epoch in epochs]
                    means = np.array([cell["mean"] for cell in cells])
                    deviations = np.array([cell["std"] or 0 for cell in cells])
                    current_axis.plot(epochs, means, style, color=COLORS[variant], linewidth=1.4,
                                      label=LABELS[variant] if metric == "train_loss" else None)
                    low = np.maximum(means - deviations, 1e-8) if metric == "train_loss" else means - deviations
                    current_axis.fill_between(epochs, low, means + deviations, color=COLORS[variant], alpha=0.08)
            axis.set(title=f"{REGIME_LABELS[regime]}: {experience.upper()}", xlabel="Source-training epoch", ylabel="Training loss (solid)")
            axis.set_yscale("log")
            accuracy_axis.set(ylabel="Familiar validation motif accuracy (dashed)", ylim=(-0.03, 1.03))
            accuracy_axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(alpha=0.15)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    figure.suptitle(warning + "Source-training diagnostics only\n"
                   "No common-OOD curves, early stopping, or checkpoint selection; shading: sample SD across seeds", fontsize=12)
    for extension in ("png", "svg"):
        figure.savefig(figures / f"skip_training.{extension}", dpi=180)
    plt.close(figure)


def generate_skip_report(output: Path) -> None:
    """Derive English tables and plots without changing measured inputs."""
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    rows = _read_csv(output / "behavior.csv")
    _validate(rows, config)
    inventory = _inventory(_read_csv(output / "inventory.csv"), config)
    histories = _histories(output, inventory, config)
    paired = _paired_rows(rows)
    intact = _select(rows, intervention="intact")
    gaps = []
    seed_metrics = []
    for row in intact:
        if row["split"] != "ood_common":
            continue
        iid = _lookup(intact, regime=row["regime"], variant=row["variant"], experience=row["experience"],
                      seed=row["seed"], split=f"iid_{row['experience']}")
        gap = {key: row[key] for key in ("regime", "variant", "experience", "seed")}
        gap.update({metric: float(iid[metric]) - float(row[metric]) for metric in METRICS})
        gaps.append(gap)
    for row in rows:
        gap = _lookup(gaps, regime=row["regime"], variant=row["variant"], experience=row["experience"], seed=row["seed"])
        seed_metrics.append({**row, "intact_own_iid_minus_ood_motif_gap": gap["motif_accuracy"]})
    regimes = list(config.get("regimes", REGIMES))
    experiences = list(config.get("experiences", ("a", "b")))
    seed_count = len(config.get("seeds", (0, 1, 2)))
    pipeline_only = seed_count < 3 or int(config["epochs"]) != 120 or set(regimes) != set(REGIMES) or set(experiences) != {"a", "b"}
    summary = {"primary_regime": "end_to_end", "primary_metric": "motif_accuracy",
               "primary_contrast": "shallow_skip_minus_deep_control", "primary_intervention": "intact",
               "primary_split": "ood_common", "motif_uniform_random_chance": 0.125,
               "regimes": regimes, "experiences": experiences, "encoder_seed_count": seed_count,
               "epochs": int(config["epochs"]), "pipeline_only": pipeline_only,
               "statistical_unit": "Encoder initialization seed; A and B reported separately, paired variants subtracted within exact seed.",
               "behavior": _aggregate(rows, GROUP_KEYS),
               "paired_skip_minus_control": _aggregate(paired, tuple(key for key in PAIR_KEYS if key != "seed")),
               "generalization_gap": _aggregate(gaps, ("regime", "variant", "experience")),
               "inventory": inventory}
    (output / "report_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "seed_metrics.csv", seed_metrics)
    _write_csv(output / "paired_seed_deltas.csv", paired)
    _make_figures(output, summary, histories)

    deep_baselines = _select(inventory, regime="end_to_end", variant="deep")
    reused_count = sum(str(row.get("reused_source", "False")).lower() in ("true", "1") for row in deep_baselines)
    if reused_count == len(deep_baselines) and reused_count:
        baseline_provenance = (
            "The original deep end-to-end baselines reuse the exact completed hard_v2 checkpoints and source-training histories; "
            "their final predictions are recomputed. "
        )
    elif reused_count == 0:
        baseline_provenance = (
            "All arms are freshly trained for this run using the configured epoch budget; no original deep end-to-end checkpoint or history is reused. "
            "Frozen-readout arms still begin with archived pretrained encoders. "
        )
    else:
        baseline_provenance = (
            f"Of {len(deep_baselines)} original deep end-to-end baselines, {reused_count} reuse exact completed hard_v2 checkpoints and source-training histories; "
            "the remaining baseline models are freshly trained. All final predictions are recomputed. "
        )

    text = ["# Boundary-free shallow-feature readout intervention", "",
            *(["**Pipeline check only:** this reduced panel validates execution and reporting; it is not the complete predeclared experiment.", ""] if pipeline_only else []),
            "This experiment tests whether exposing shallow whole-graph features to the original task heads improves predictions on unseen motif-position combinations. "
            "All training labels come from the original hard_v2 source training identities. No target-combination labels, adaptation supports, or target-fragment boundaries are added.", "",
            "## Design and primary contrast", "",
            "The primary regime is end-to-end source training. Its primary endpoint is ordinary motif accuracy on the common unseen-combination test set, "
            "and the primary contrast is shallow skip minus the parameter-matched deep control, paired by exact initialization seed. "
            "The original 448D deep readout is a separate baseline. The matched variants both have 512 input features and identical trainable parameter counts: "
            "one appends the whole-graph mean after message layer one; the other appends the final-layer whole-graph mean. "
            "The latter is a deterministic duplicate of a block already present in the 448D vector, so it controls added head width and reparameterization without adding new information.", "",
            f"All runs use {config['epochs']} predeclared epochs, the same source minibatch order within seed, AdamW, "
            "the original three-task loss, and the original training hyperparameters. Added first-layer head columns start at zero; the inherited 448D columns "
            "and other head parameters share the exact archived random initialization. The common OOD set is evaluated only after the fixed final epoch; "
            "there is no OOD checkpoint, hyperparameter, or best-seed selection.", "",
            "End-to-end augmented models start from the archived random encoder and learn encoder plus heads. " + baseline_provenance +
            "Frozen-readout runs keep the corresponding pretrained A/B encoder fixed, reset the heads to that seed's archived random head values, "
            "and train only the heads on cached full-molecule features. They do not continue the inherited trained classifier. "
            "These regimes have different encoder histories and are interpreted separately. Identical update budgets and matched parameters do not imply identical compute or optimization.", "",
            f"Values below are mean +/- sample SD across {seed_count} initialization seeds, with A and B reported separately. "
            "A and B are not pooled into six independent replicates. Paired differences subtract variants within each seed before computing their SD; "
            "they are descriptive contrasts, not significance tests. Eight-output uniform random guessing is 12.5% for the motif task.", "",
            "## Final-epoch intact predictions", ""]
    for regime in regimes:
        text.extend([f"### {REGIME_LABELS[regime]}", ""])
        cells = []
        for experience in experiences:
            for variant in VARIANTS:
                iid = _lookup(summary["behavior"], regime=regime, variant=variant, experience=experience, intervention="intact", split=f"iid_{experience}")
                ood = _lookup(summary["behavior"], regime=regime, variant=variant, experience=experience, intervention="intact", split="ood_common")
                cells.append([experience.upper(), LABELS[variant], _format(iid["motif_accuracy"]), _format(ood["motif_accuracy"]),
                              _format(ood["position_accuracy"]), _format(ood["distance_accuracy"]), _format(ood["distance_mae"], mae=True), _format(ood["joint_accuracy"])])
        text.extend([_table(["Experience", "Readout", "Own-IID motif", "Unseen motif", "Unseen position", "Unseen distance", "Unseen distance MAE", "Unseen joint"], cells), ""])
    text.extend(["![Intact final endpoints and generalization gaps](figures/skip_primary.png)", "", "## Paired primary and secondary task contrasts", "",
                 "All cells below compare the intact shallow skip with the intact 512D deep control. Differences are percentage points; "
                 "positive accuracy differences favor the shallow skip. Balanced accuracy averages recall over test labels, and is reported alongside the predeclared ordinary-accuracy endpoint.", ""])
    cells = []
    for regime in regimes:
        for experience in experiences:
            iid = _lookup(summary["paired_skip_minus_control"], regime=regime, experience=experience, intervention="intact", split=f"iid_{experience}")
            ood = _lookup(summary["paired_skip_minus_control"], regime=regime, experience=experience, intervention="intact", split="ood_common")
            cells.append([REGIME_LABELS[regime], experience.upper(), _format(iid["motif_accuracy"], delta=True),
                          *(_format(ood[metric], delta=True) for metric in ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy"))])
    text.extend([_table(["Regime", "Experience", "Own-IID motif", "Unseen motif (primary)", "Unseen motif balanced", "Unseen position", "Unseen distance", "Unseen joint"], cells), "",
                 "## Secondary appended-branch interventions", "",
                 "After training, the appended branch is either set to zero or permuted across test molecules while the encoder, inherited deep vector, and heads stay fixed. "
                 "These interventions test functional compatibility and reliance of the fitted model on this branch. "
                 "They do not measure information destruction or provide an additive causal decomposition: replacement features can be incompatible with the classifier and training distribution.", ""])
    cells = []
    for regime in regimes:
        for experience in experiences:
            for variant in ("deep_control", "shallow_skip"):
                intact_cell = _lookup(summary["behavior"], regime=regime, experience=experience, variant=variant, intervention="intact", split="ood_common")
                altered = [_lookup(summary["behavior"], regime=regime, experience=experience, variant=variant, intervention=intervention, split="ood_common")
                           for intervention in ("zero_branch", "permuted_branch")]
                cells.append([REGIME_LABELS[regime], experience.upper(), LABELS[variant], _format(intact_cell["motif_accuracy"]),
                              *(_format(cell["motif_accuracy"]) for cell in altered)])
    text.extend([_table(["Regime", "Experience", "Readout", "Intact unseen motif", "Zero branch", "Permuted branch"], cells), "",
                 "## Parameters and source-training diagnostics", ""])
    parameter_rows = []
    for regime in regimes:
        for variant in VARIANTS:
            selected = _select(inventory, regime=regime, variant=variant)
            counts = {(row["readout_dim"], row["parameter_count"], row["trainable_parameter_count"]) for row in selected}
            if len(counts) != 1:
                raise ValueError("Parameter counts must be constant across experiences and seeds for a variant/regime.")
            parameter_rows.append([REGIME_LABELS[regime], LABELS[variant], *next(iter(counts))])
    text.extend([_table(["Regime", "Readout", "Input features", "Total parameters", "Trainable parameters"], parameter_rows), "",
                 "![Source loss and familiar validation curves](figures/skip_training.png)", "",
                 "Training curves display source training loss and familiar validation motif accuracy only. Familiar validation position and distance scores "
                 "remain in the recorded histories. No test trajectory was used to select the endpoint.", "",
                 "## Interpretation limits and preserved records", "",
                 "The added shallow pool averages all real atoms and never receives target_fragment_indices or an oracle membership mask. "
                 "It retains the original atom/query features supplied to every variant, so it is boundary-free rather than query-free. "
                 "Whole-graph shallow pooling can mix the target, ring, anchor, and distractor; it does not isolate a context-independent substituent code. "
                 "A benefit in the frozen regime would implicate access through the readout on fixed encoder states; an end-to-end benefit can additionally involve "
                 "representation and optimization changes. A null result would show that this particular shallow mean and training rule are insufficient, "
                 "not that all useful shallow information is absent. Linear probe readability, labeled adaptation, and zero-target-label task prediction remain distinct measurements.", "",
                 "behavior.csv preserves every split, task metric, seed, and intervention. seed_metrics.csv retains individual endpoints and each intact model's own-IID-minus-common-OOD motif gap; "
                 "paired_seed_deltas.csv retains exact-seed contrasts. report_summary.json records all task summaries, gaps, paired contrasts, and inventory provenance. "
                 "Both IID sets, balanced task accuracies, and distance MAE are available in those complete machine-readable records even when the compact tables focus on own IID and common OOD. "
                 "The report generator preserves measured inputs and prior archived experiments.", ""])
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
