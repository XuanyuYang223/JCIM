"""Report matched frozen readouts without treating support draws as model seeds."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCALES = ("root", "target", "target_plus1", "target_plus2", "target_plus3", "whole_graph", "legacy_graph")
SCALE_LABELS = {"root": "Root", "target": "Target", "target_plus1": "Target + 1", "target_plus2": "Target + 2",
                "target_plus3": "Target + 3", "whole_graph": "Whole mean", "legacy_graph": "Legacy 448D"}
METRICS = ("accuracy", "balanced_accuracy")
COMMON_KEYS = ("experience", "condition", "layer", "scale", "feature_dim")
ADAPT_KEYS = (*COMMON_KEYS, "support", "split")
DIRECT_KEYS = (*COMMON_KEYS, "source_position", "target_position", "split")
DIRECT_AVERAGE_KEYS = (*COMMON_KEYS, "split")
PAIR_ADAPT_KEYS = tuple(key for key in ADAPT_KEYS if key != "condition")
PAIR_DIRECT_KEYS = tuple(key for key in DIRECT_KEYS if key != "condition")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _select(rows: list[dict], **filters: Any) -> list[dict]:
    return [row for row in rows if all(str(row.get(key)) == str(value) for key, value in filters.items())]


def _stats(members: list[dict], metric: str) -> dict:
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in members:
        by_seed[str(row["seed"])].append(float(row[metric]))
    per_seed = {seed: float(np.mean(values)) for seed, values in sorted(by_seed.items())}
    values = list(per_seed.values())
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values), "per_seed": per_seed}


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    return [{**dict(zip(keys, key)), **{metric: _stats(members, metric) for metric in METRICS}}
            for key, members in sorted(groups.items())]


def _paired_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Subtract exactly matched supports before averaging within encoder seeds."""
    full_keys = (*keys, "seed", "repeat")
    by_condition: dict[str, dict[tuple, dict]] = {"trained": {}, "random": {}}
    for row in rows:
        key = tuple(str(row[field]) for field in full_keys)
        if key in by_condition[row["condition"]]:
            raise ValueError(f"Duplicate paired measurement: {key!r}")
        by_condition[row["condition"]][key] = row
    trained, random = by_condition["trained"], by_condition["random"]
    if set(trained) != set(random):
        raise ValueError("Trained and random encoders lack exactly matched seed/support measurements.")
    differences = []
    for key, current in sorted(trained.items()):
        reference = random[key]
        if any(str(current[field]) != str(reference[field]) for field in ("n_fit", "n_test")):
            raise ValueError("A paired trained/random comparison has different fit or test counts.")
        differences.append({**dict(zip(full_keys, key)), **{metric: float(current[metric]) - float(reference[metric])
                                                         for metric in METRICS}})
    return differences


def _validate(rows: list[dict], config: dict, *, direct: bool) -> None:
    if not rows:
        raise ValueError("Readout measurements must be nonempty.")
    keys = DIRECT_KEYS if direct else ADAPT_KEYS
    fields = (*keys, "model", "seed", "repeat", "n_fit", "n_test", *METRICS, "prediction_key")
    observed = set()
    for row in rows:
        if any(field not in row or row[field] == "" for field in fields):
            raise ValueError(f"Missing readout fields: {row!r}")
        if row["experience"] not in ("a", "b") or row["condition"] not in ("trained", "random") or row["scale"] not in SCALES:
            raise ValueError(f"Unknown experience, condition, or pooling scale: {row!r}")
        expected_model = f"{row['experience'] if row['condition'] == 'trained' else 'random'}_seed{row['seed']}"
        if row["model"] != expected_model:
            raise ValueError("Model names do not preserve matched initialization identities.")
        if int(row["feature_dim"]) != (448 if row["scale"] == "legacy_graph" else 64):
            raise ValueError("The single-pool scopes must be 64D; the legacy reference must be 448D.")
        if int(row["n_fit"]) <= 0 or int(row["n_test"]) <= 0:
            raise ValueError("Fit and test sets must be nonempty.")
        if direct:
            if row["source_position"] not in ("0", "1", "2") or row["target_position"] not in ("0", "1", "2") or row["source_position"] == row["target_position"]:
                raise ValueError("Direct transfer requires distinct valid source and target positions.")
            if row["split"] not in ("target_all", "target_common_ood", "target_own_iid"):
                raise ValueError("Unknown direct-transfer target stratum.")
        elif row["support"] not in ("new", "replay") or row["split"] not in ("iid_a", "iid_b", "ood_common"):
            raise ValueError("Unknown adaptation support or test split.")
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {metric}: {value!r}")
        key = tuple(str(row[field]) for field in (*keys, "seed", "repeat"))
        if key in observed:
            raise ValueError(f"Duplicate readout measurement: {key!r}")
        observed.add(key)
    seeds = tuple(map(str, config.get("seeds", (0, 1, 2))))
    repeats = tuple(map(str, config.get("support_repeats", (0, 1, 2))))
    layers = tuple(map(str, config.get("layers", range(9))))
    tails = tuple((str(source), str(target), split) for source in range(3) for target in range(3) if source != target
                  for split in ("target_all", "target_common_ood", "target_own_iid")) if direct else tuple((support, split) for support in ("new", "replay") for split in ("iid_a", "iid_b", "ood_common"))
    expected = {(experience, condition, layer, scale, str(448 if scale == "legacy_graph" else 64), *tail, seed, repeat)
                for experience in ("a", "b") for condition in ("trained", "random") for layer in layers
                for scale in SCALES for tail in tails for seed in seeds for repeat in repeats}
    if observed != expected:
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        raise ValueError(f"Incomplete or unexpected readout panel: missing={missing[:2]!r}, extra={extra[:2]!r}")


def _format(stat: dict, *, delta: bool = False) -> str:
    suffix = " pp" if delta else "%"
    deviation = "SD undefined" if stat["std"] is None else f"{100 * stat['std']:.1f}"
    return f"{100 * stat['mean']:.1f} +/- {deviation}{suffix} (n={stat['n']})"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)]) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty derived table.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_rows(rows: list[dict], keys: tuple[str, ...], *, experiment: str, repeat_scope: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in (*keys, "seed"))].append(row)
    output = []
    for key, members in sorted(groups.items()):
        values = dict(zip((*keys, "seed"), key))
        output.append({"experiment": experiment, "repeat_scope": repeat_scope, "experience": values["experience"],
                       "condition": values.get("condition", "trained_minus_random"), "seed": values["seed"],
                       "layer": values["layer"], "scale": values["scale"], "feature_dim": values["feature_dim"],
                       "support": values.get("support", ""), "split": values.get("split", ""),
                       "source_position": values.get("source_position", ""), "target_position": values.get("target_position", ""),
                       "support_repeats": len({str(row["repeat"]) for row in members}),
                       "measurements_averaged": len(members),
                       **{metric: float(np.mean([float(row[metric]) for row in members])) for metric in METRICS}})
    return output


def _lookup(rows: list[dict], **filters: Any) -> dict:
    selected = _select(rows, **filters)
    if len(selected) != 1:
        raise ValueError(f"Expected one summary cell for {filters!r}; found {len(selected)}.")
    return selected[0]


def _locality_summary(path: Path) -> list[dict]:
    """Summarize structural receptive reach without loading the large table twice."""
    if not path.exists():
        return []
    count_fields = ("pool_atom_count", "target_atom_count", "molecule_atom_count", "pool_outside_target_count",
                    "input_receptive_atom_count", "receptive_outside_target_count", "receptive_outside_pool_count")
    groups: dict[tuple, dict] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (str(row["region"]), str(row["layer"]))
            group = groups.setdefault(key, {"n": 0, "sums": defaultdict(float), "mins": {}, "maxs": {},
                                            "full": 0, "pool_fraction": 0.0, "receptive_fraction": 0.0})
            values = {field: int(row[field]) for field in count_fields}
            if values["molecule_atom_count"] <= 0 or any(value < 0 for value in values.values()):
                raise ValueError("Locality counts must be nonnegative with nonempty molecules.")
            group["n"] += 1
            for field, value in values.items():
                group["sums"][field] += value
                group["mins"][field] = min(group["mins"].get(field, value), value)
                group["maxs"][field] = max(group["maxs"].get(field, value), value)
            group["full"] += str(row["receptive_covers_full_graph"]).lower() in ("true", "1")
            group["pool_fraction"] += values["pool_atom_count"] / values["molecule_atom_count"]
            group["receptive_fraction"] += values["input_receptive_atom_count"] / values["molecule_atom_count"]
    return [{"scale": key[0], "layer": key[1], "n": group["n"],
             **{field: {"mean": group["sums"][field] / group["n"], "min": group["mins"][field], "max": group["maxs"][field]}
                for field in count_fields},
             "mean_pool_fraction": group["pool_fraction"] / group["n"],
             "mean_receptive_fraction": group["receptive_fraction"] / group["n"],
             "fraction_covering_full_graph": group["full"] / group["n"]}
            for key, group in sorted(groups.items())]


def _make_figures(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    layer, metric = str(summary["primary_layer"]), "balanced_accuracy"
    positions, scales = np.arange(len(SCALES)), list(SCALES)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.8), constrained_layout=True)
    for row_index, experience in enumerate(("a", "b")):
        specifications = (("primary_adaptation", {"support": "replay", "split": "ood_common"}, "New combinations, labeled refit"),
                          ("primary_adaptation", {"support": "replay", "split": f"iid_{experience}"}, "Old own-IID, labeled refit"),
                          ("primary_direct_transfer", {"source_position": 0, "target_position": 1, "split": "target_all"}, "Ortho to meta, source labels only"))
        for axis, (name, filters, title) in zip(axes[row_index], specifications):
            for condition, color, label in (("trained", "#2673a6", "Pretrained frozen"), ("random", "#777777", "Random frozen")):
                cells = [_lookup(summary[name], experience=experience, condition=condition, layer=layer, scale=scale, **filters)[metric]
                         for scale in scales]
                means, deviations = np.array([cell["mean"] for cell in cells]), np.array([cell["std"] or 0 for cell in cells])
                axis.errorbar(positions, means, yerr=deviations, color=color, marker="o", capsize=2.5, linewidth=1.4, label=label)
            axis.axvline(5.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
            axis.axhline(0.125, color="#aaaaaa", linewidth=0.8, linestyle=":")
            axis.set(title=f"Experience {experience.upper()}: {title}", ylabel="Motif balanced accuracy", ylim=(-0.03, 1.03))
            axis.set_xticks(positions, [SCALE_LABELS[scale] for scale in scales], rotation=30, ha="right", fontsize=8)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(axis="y", alpha=0.2)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    run_label = "PIPELINE CHECK ONLY\n" if summary["pipeline_only"] else ""
    figure.suptitle(run_label + f"Matched frozen readouts at layer {layer}: primary fixed support draw {summary['primary_repeat']}\n"
                   "Error bars: sample SD across encoder seeds; all single-pool scopes 64D, legacy reference 448D", fontsize=12)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(figures / f"scale_primary.{extension}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for row_index, experience in enumerate(("a", "b")):
        for axis, name, filters, title in ((axes[row_index, 0], "paired_adaptation", {"support": "replay", "split": "ood_common"}, "New labeled refit"),
                                          (axes[row_index, 1], "paired_direct_direction_average", {"split": "target_all"}, "All six source-only directions")):
            layers = sorted({int(row["layer"]) for row in summary[name]})
            values = np.array([[_lookup(summary[name], experience=experience, layer=layer_value, scale=scale, **filters)[metric]["mean"]
                                for scale in SCALES] for layer_value in layers])
            limit = max(0.05, float(np.max(np.abs(values))))
            display = axis.imshow(values, origin="lower", aspect="auto", vmin=-limit, vmax=limit, cmap="RdBu")
            axis.set(title=f"Experience {experience.upper()}: {title}", ylabel="Encoder layer")
            axis.set_xticks(positions, [SCALE_LABELS[scale] for scale in SCALES], rotation=30, ha="right", fontsize=8)
            axis.set_yticks(range(len(layers)), layers)
            axis.axvline(5.5, color="#333333", linestyle="--", linewidth=0.7)
            colorbar = figure.colorbar(display, ax=axis, shrink=0.8)
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    figure.suptitle(run_label + "Pretrained minus matched random: balanced-accuracy differences\n"
                   "Support draws and transfer directions averaged within each seed; no best-layer selection", fontsize=12)
    for extension in ("png", "svg"):
        figure.savefig(figures / f"scale_paired_layers.{extension}", dpi=180)
    plt.close(figure)


def generate_scale_report(output: Path) -> None:
    """Derive summaries and English figures while preserving measured inputs."""
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    adaptation, direct = _read_csv(output / "adaptation.csv"), _read_csv(output / "direct_transfer.csv")
    _validate(adaptation, config, direct=False)
    _validate(direct, config, direct=True)
    primary_repeat, primary_layer = str(config.get("primary_repeat", 0)), int(config.get("primary_layer", 8))
    if primary_repeat not in set(map(str, config.get("support_repeats", (0, 1, 2)))) or primary_layer not in list(map(int, config.get("layers", range(9)))):
        raise ValueError("The predeclared primary support repetition and layer must be recorded.")
    paired_adapt, paired_direct = _paired_rows(adaptation, PAIR_ADAPT_KEYS), _paired_rows(direct, PAIR_DIRECT_KEYS)
    seed_count, repeat_count = len(config.get("seeds", (0, 1, 2))), len(config.get("support_repeats", (0, 1, 2)))
    pipeline_only = seed_count < 3 or repeat_count < 3 or set(map(int, config.get("layers", range(9)))) != set(range(9))
    summary = {"primary_repeat": primary_repeat, "primary_layer": primary_layer, "motif_uniform_random_chance": 0.125,
               "encoder_seed_count": seed_count, "support_repeat_count": repeat_count, "pipeline_only": pipeline_only,
               "statistical_unit": "Encoder seed; average support repetitions and, for direction averages, all six directions within seed first.",
               "adaptation": _aggregate(adaptation, ADAPT_KEYS),
               "primary_adaptation": _aggregate(_select(adaptation, repeat=primary_repeat), ADAPT_KEYS),
               "direct_transfer": _aggregate(direct, DIRECT_KEYS),
               "primary_direct_transfer": _aggregate(_select(direct, repeat=primary_repeat), DIRECT_KEYS),
               "direct_direction_average": _aggregate(direct, DIRECT_AVERAGE_KEYS),
               "primary_direct_direction_average": _aggregate(_select(direct, repeat=primary_repeat), DIRECT_AVERAGE_KEYS),
               "paired_adaptation": _aggregate(paired_adapt, PAIR_ADAPT_KEYS),
               "primary_paired_adaptation": _aggregate(_select(paired_adapt, repeat=primary_repeat), PAIR_ADAPT_KEYS),
               "paired_direct_transfer": _aggregate(paired_direct, PAIR_DIRECT_KEYS),
               "primary_paired_direct_transfer": _aggregate(_select(paired_direct, repeat=primary_repeat), PAIR_DIRECT_KEYS),
               "paired_direct_direction_average": _aggregate(paired_direct, tuple(key for key in DIRECT_AVERAGE_KEYS if key != "condition")),
               "primary_paired_direct_direction_average": _aggregate(_select(paired_direct, repeat=primary_repeat), tuple(key for key in DIRECT_AVERAGE_KEYS if key != "condition")),
               "locality": _locality_summary(output / "locality.csv")}
    (output / "report_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    seed_rows, paired_seed_rows = [], []
    for scope, repeat_value in (("all_repeats_within_seed", None), ("primary_fixed_repeat", primary_repeat)):
        for rows, keys, experiment, destination in ((adaptation, ADAPT_KEYS, "adaptation", seed_rows),
                                                    (direct, DIRECT_KEYS, "direct_transfer", seed_rows),
                                                    (direct, DIRECT_AVERAGE_KEYS, "direct_direction_average", seed_rows),
                                                    (paired_adapt, PAIR_ADAPT_KEYS, "adaptation", paired_seed_rows),
                                                    (paired_direct, PAIR_DIRECT_KEYS, "direct_transfer", paired_seed_rows),
                                                    (paired_direct, tuple(key for key in DIRECT_AVERAGE_KEYS if key != "condition"), "direct_direction_average", paired_seed_rows)):
            selected = rows if repeat_value is None else _select(rows, repeat=repeat_value)
            destination.extend(_seed_rows(selected, keys, experiment=experiment, repeat_scope=scope))
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "paired_seed_deltas.csv", paired_seed_rows)
    _make_figures(output, summary)

    text = ["# Frozen readouts across molecular pooling scales", "",
            *( ["**Pipeline check only:** this reduced run does not supply the full predeclared seed, support-repeat, and layer panel. "
                "Its results validate execution and reporting; they are not primary scientific evidence.", ""] if pipeline_only else []),
            "This analysis asks whether pretraining improves information that a matched frozen readout can use, and where that information is accessible. "
            "Labeled adaptation and source-position-only transfer answer different questions. All results use fresh standardized multiclass ridge readouts; "
            "neither encoder is updated.", "", "## Matched design and statistical unit", "",
            f"The predeclared primary layer is {primary_layer}, primary support repetition is {primary_repeat}, and ridge alpha is {config.get('alpha', config.get('probe_alpha', 1.0))}. "
            "Pretrained A and B are each compared with the exact corresponding random initialization seed. The two conditions receive identical support identities, "
            "feature standardization estimated from fit data only, and the same fixed regularization and fitting procedure. There is no target-test hyperparameter or layer selection.", "",
            f"{repeat_count} support repetition{'s' if repeat_count != 1 else ''} {'are' if repeat_count != 1 else 'is'} nested within each encoder seed. "
            f"Supplemental summaries average repetitions within each seed before computing the mean and sample SD across {seed_count} encoder seed{'s' if seed_count != 1 else ''}; "
            "they do not count nine repetitions as nine independent models. Paired advantages subtract trained and random scores for each exact seed, repetition, scale, and test condition before averaging. "
            "The six-direction summary weights each directed position pair equally within seed. Balanced accuracy averages recalls over the labels present in each test set. "
            "The eight-output uniform reference is 12.5%; an empirical random-encoder score is not a theoretical chance level.", "", "## Primary fixed-support comparison", "",
            f"Only support repetition {primary_repeat} and layer {primary_layer} appear in the following tables. Values are mean +/- sample SD across encoder seeds. "
            "The adaptation table uses 32 new plus 32 old labels; the direct table fits 32 examples per motif (256 labels) at the source position and uses zero target-position labels. "
            "Their label budgets and fit distributions differ, so the gap between these protocols is descriptive rather than a controlled effect of target labels alone.", ""]
    for experience in ("a", "b"):
        text.extend([f"### Experience {experience.upper()}: labeled refit with replay", ""])
        table_rows = []
        for scale in SCALES:
            cells = [_lookup(summary["primary_adaptation"], experience=experience, condition=condition, layer=primary_layer,
                             scale=scale, support="replay", split=split)["balanced_accuracy"]
                     for condition, split in (("trained", "ood_common"), ("random", "ood_common"), ("trained", f"iid_{experience}"), ("random", f"iid_{experience}"))]
            delta = _lookup(summary["primary_paired_adaptation"], experience=experience, layer=primary_layer, scale=scale,
                            support="replay", split="ood_common")["balanced_accuracy"]
            table_rows.append([SCALE_LABELS[scale], *map(_format, cells), _format(delta, delta=True)])
        text.extend([_table(["Pooling scope", "Pretrained new", "Random new", "Pretrained old", "Random old", "Paired new advantage"], table_rows), ""])
        text.extend([f"### Experience {experience.upper()}: ortho-to-meta source-only transfer", ""])
        table_rows = []
        for scale in SCALES:
            cells = [_lookup(summary["primary_direct_transfer"], experience=experience, condition=condition, layer=primary_layer,
                             scale=scale, source_position=0, target_position=1, split="target_all")["balanced_accuracy"] for condition in ("trained", "random")]
            delta = _lookup(summary["primary_paired_direct_transfer"], experience=experience, layer=primary_layer, scale=scale,
                            source_position=0, target_position=1, split="target_all")["balanced_accuracy"]
            protected = [_lookup(summary["primary_direct_transfer"], experience=experience, condition=condition, layer=primary_layer,
                                 scale=scale, source_position=0, target_position=1, split="target_common_ood")["balanced_accuracy"]
                         for condition in ("trained", "random")]
            table_rows.append([SCALE_LABELS[scale], *map(_format, cells), _format(delta, delta=True), *map(_format, protected)])
        text.extend([_table(["Pooling scope", "Pretrained all", "Random all", "Paired all advantage", "Pretrained common OOD", "Random common OOD"], table_rows), ""])
    text.extend(["![Primary matched frozen readouts](figures/scale_primary.png)", "", "## Robustness across support draws and directions", "",
                 "The following layer-fixed summary averages all support repetitions within seed. Direct transfer additionally averages all six ordered position pairs within seed. "
                 "Every individual pair, support draw, encoder seed, layer, ordinary accuracy, and balanced accuracy remains available in the source CSVs and derived seed tables.", ""])
    table_rows = []
    for experience in ("a", "b"):
        for scale in SCALES:
            adapt_delta = _lookup(summary["paired_adaptation"], experience=experience, layer=primary_layer, scale=scale,
                                  support="replay", split="ood_common")["balanced_accuracy"]
            direct_cells = [_lookup(summary["direct_direction_average"], experience=experience, condition=condition, layer=primary_layer,
                                    scale=scale, split="target_all")["balanced_accuracy"] for condition in ("trained", "random")]
            direct_delta = _lookup(summary["paired_direct_direction_average"], experience=experience, layer=primary_layer, scale=scale, split="target_all")["balanced_accuracy"]
            table_rows.append([experience.upper(), SCALE_LABELS[scale], _format(adapt_delta, delta=True), *map(_format, direct_cells), _format(direct_delta, delta=True)])
    text.extend([_table(["Experience", "Pooling scope", "Refit paired new advantage", "Direct pretrained", "Direct random", "Direct paired advantage"], table_rows), "",
                 "![Paired advantages across all recorded layers](figures/scale_paired_layers.png)", "", "## What spatial scope does and does not establish", "",
                 "Root, target-substituent mean, target plus one/two/three graph-neighborhood shells, and whole-graph mean each have 64 features. "
                 "The legacy graph readout has 448 features: whole-graph mean, sum divided by square-root atom count, target root, and four marked-query node sums. "
                 "Its feature width and query-specific channels differ, so it is a separate reference rather than another equal-width pooling scope.", "",
                 "Target atom memberships are oracle metadata applied only after encoding. They are not supplied as encoder inputs and do not test whether a model can discover a substituent boundary. "
                 "All node states come from the original full-molecule encoder. In particular, layer-eight selected nodes already aggregate messages from up to eight graph hops. "
                 "A small pooling set is therefore not a strictly local input: remote context can remain accessible. The saved locality.csv quantifies pooling sizes and receptive-field reach; "
                 "small-pool success establishes accessible information at those nodes, not independence from distant molecular context. "
                 "Even layer-zero degree and hydrogen features retain boundary chemistry. Receptive reach is an architectural upper bound, not evidence that every reachable atom affects the prediction.", ""])
    if summary["locality"]:
        table_rows = []
        for scale in SCALES:
            cell = _lookup(summary["locality"], layer=primary_layer, scale=scale)
            pool, receptive = cell["pool_atom_count"], cell["input_receptive_atom_count"]
            table_rows.append([SCALE_LABELS[scale], f"{pool['mean']:.1f} [{pool['min']}, {pool['max']}]",
                               f"{receptive['mean']:.1f} [{receptive['min']}, {receptive['max']}]",
                               f"{100 * cell['mean_receptive_fraction']:.1f}%", f"{100 * cell['fraction_covering_full_graph']:.1f}%"])
        text.extend([f"Architectural receptive support at layer {primary_layer}, over the saved molecular identities (mean atom count [minimum, maximum]):", "",
                     _table(["Pooling scope", "Pooled atoms", "Reachable input atoms", "Mean molecule fraction reached", "Full graph reached"], table_rows), ""])
    text.extend([
                 "Source-only transfer fits labels at one position and evaluates different molecule identities at another. It is not the inherited chemical task head's zero-shot prediction, "
                 "and success would not alone prove generalization to a novel motif type. Labeled refit uses new-combination labels, so high accuracy there establishes adaptation/readability under that support distribution. "
                 "A pretrained advantage is conditional on its prior labels and training compute. Negative linear readout results cannot rule out information accessible through other pooling operations or nonlinear decoders.", "",
                 "## Records", "", "Reporting preserves recorded measurements and source archives. The machine-readable report_summary.json contains primary and repeat-averaged results separately, "
                 "seed_metrics.csv retains encoder-seed means, and paired_seed_deltas.csv retains exact matched contrasts. adaptation.csv and direct_transfer.csv contain every support-level result, "
                 "including both accuracy definitions and prediction references. No best seed, support draw, pooling scope, or layer was selected for the primary comparison.", ""])
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
