"""Generate an English pilot report directly from recorded experiment results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITION_NAMES = {
    "a": "Experience A",
    "b": "Experience B",
    "random": "Untrained encoder",
    "raw_input": "Raw input features",
    "raw": "Raw input features",
}
PAIR_NAMES = {
    "random_random": "Untrained / untrained",
    "trained_same_data": "Trained / same data",
    "trained_different_data": "Trained / different data",
    "random_trained": "Untrained / trained",
}
COLORS = {"a": "#2673a6", "b": "#df812f", "random": "#777777", "raw_input": "#32815e", "raw": "#32815e"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _number(row: dict, key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite result for {key}: {row!r}")
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def _seed_stats(rows: list[dict], metric: str) -> dict[str, Any] | None:
    """Average repetitions within each seed before calculating seed spread."""
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric, "") != "":
            by_seed[str(row.get("seed", row.get("model", "single")))].append(_number(row, metric))
    if not by_seed:
        return None
    per_seed = {seed: float(np.mean(values)) for seed, values in sorted(by_seed.items())}
    return {**_stats(list(per_seed.values())), "per_seed": per_seed}


def _format_stat(stat: dict | None, percent: bool = True) -> str:
    if stat is None:
        return "Unavailable"
    scale = 100.0 if percent else 1.0
    digits = 1 if percent else 3
    suffix = "%" if percent else ""
    value = f"{stat['mean'] * scale:.{digits}f}"
    if stat["std"] is None:
        return f"{value}{suffix} (n=1; SD undefined)"
    return f"{value} +/- {stat['std'] * scale:.{digits}f}{suffix} (n={stat['n']})"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "No rows were recorded for this comparison.\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _direct_only(row: dict) -> bool:
    value = str(row.get("direct_f", "")).strip().lower()
    return value in {"true", "1", "yes"} or "direct" in row.get("split", "").lower()


def _probe_condition(row: dict) -> str:
    return "raw_input" if int(row["layer"]) < 0 else row["condition"]


def _base_behavior(rows: list[dict], condition: str, split: str) -> list[dict]:
    return [row for row in rows if row["condition"] == condition and row["split"] == split and not _direct_only(row)]


def _aggregate_by(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    result = []
    for key_values, members in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result.append({**dict(zip(keys, key_values)), **{metric: _seed_stats(members, metric) for metric in metrics}})
    return result


def _make_figure(
    result_dir: Path,
    behavior: list[dict],
    probes: list[dict],
    similarities: list[dict],
    adaptation: list[dict],
    final_layer: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.4), constrained_layout=True)
        conditions = [condition for condition in ("a", "b", "random") if any(row["condition"] == condition for row in behavior)]
        panel = axes[0, 0]
        width = 0.24
        for offset, condition in enumerate(conditions):
            stats = [_seed_stats(_base_behavior(behavior, condition, split), "position_accuracy") for split in ("iid_test", "ood_f_meta")]
            positions = np.arange(2) + (offset - (len(conditions) - 1) / 2) * width
            means = [stat["mean"] if stat is not None else np.nan for stat in stats]
            spreads = [stat["std"] or 0.0 if stat is not None else 0.0 for stat in stats]
            panel.bar(positions, means, yerr=spreads, width=width, label=CONDITION_NAMES[condition], color=COLORS[condition], capsize=3)
        panel.set_xticks([0, 1], ["IID molecules", "Fluoro-meta"])
        panel.set_ylim(0, 1.08)
        panel.set_ylabel("Position accuracy")
        panel.set_title("A. Pretrained heads: structure generalization", loc="left", weight="bold")
        panel.legend(fontsize=8, loc="best")
        panel.text(0.02, 0.02, "Fluoro-meta has a constant meta label.", transform=panel.transAxes, fontsize=8)

        panel = axes[0, 1]
        for condition in ("a", "b", "random"):
            members = [row for row in probes if row["probe"] == "cross_position_family" and _probe_condition(row) == condition]
            layers = sorted({int(row["layer"]) for row in members})
            stats = [_seed_stats([row for row in members if int(row["layer"]) == layer], "balanced_accuracy") for layer in layers]
            if layers:
                means = np.array([stat["mean"] for stat in stats])
                spreads = np.array([stat["std"] or 0.0 for stat in stats])
                panel.plot(layers, means, "o-", label=CONDITION_NAMES[condition], color=COLORS[condition], markersize=4)
                panel.fill_between(layers, means - spreads, means + spreads, color=COLORS[condition], alpha=0.12)
        raw = _seed_stats([row for row in probes if row["probe"] == "cross_position_family" and _probe_condition(row) == "raw_input"], "balanced_accuracy")
        if raw:
            panel.axhline(raw["mean"], color=COLORS["raw_input"], linestyle="--", label="Raw input readout")
        panel.axhline(1 / 8, color="#aaaaaa", linestyle=":", label="Uniform 8-class reference")
        panel.set_ylim(0, 1.08)
        panel.set_xlabel("Representation layer")
        panel.set_ylabel("Family balanced accuracy")
        panel.set_title("B. Local family readout: ortho to meta", loc="left", weight="bold")
        panel.legend(fontsize=8, loc="best")

        panel = axes[1, 0]
        pair_kinds = [kind for kind in PAIR_NAMES if any(row["pair_kind"] == kind and int(row["layer"]) == final_layer for row in similarities)]
        stats = [_stats([_number(row, "cka") for row in similarities if row["pair_kind"] == kind and int(row["layer"]) == final_layer]) for kind in pair_kinds]
        panel.bar(np.arange(len(pair_kinds)), [stat["mean"] for stat in stats], yerr=[stat["std"] or 0.0 for stat in stats], color=["#777777", "#2673a6", "#df812f", "#946aa2"][: len(pair_kinds)], capsize=3)
        panel.set_xticks(np.arange(len(pair_kinds)), [PAIR_NAMES[kind].replace(" / ", " /\n") for kind in pair_kinds], fontsize=8)
        panel.set_ylim(0, 1.08)
        panel.set_ylabel("Centered linear CKA")
        panel.set_title(f"C. Shared molecules: final layer {final_layer}", loc="left", weight="bold")

        panel = axes[1, 1]
        for condition in ("a", "b", "random", "raw"):
            members = [row for row in adaptation if row["condition"] == condition]
            shots = sorted({int(row["shots"]) for row in members})
            stats = [_seed_stats([row for row in members if int(row["shots"]) == shot], "balanced_accuracy") for shot in shots]
            if shots:
                means = np.array([stat["mean"] for stat in stats])
                spreads = np.array([stat["std"] or 0.0 for stat in stats])
                panel.plot(shots, means, "o-", color=COLORS[condition], label=CONDITION_NAMES[condition], markersize=4)
                panel.fill_between(shots, means - spreads, means + spreads, color=COLORS[condition], alpha=0.12)
        if adaptation:
            panel.set_xscale("log", base=2)
            shot_ticks = sorted({int(row["shots"]) for row in adaptation})
            panel.set_xticks(shot_ticks, [str(shot) for shot in shot_ticks])
        panel.set_ylim(0, 1.08)
        panel.set_xlabel("Labeled adaptation molecules (total)")
        panel.set_ylabel("Distance balanced accuracy")
        panel.set_title("D. New ridge head; frozen encoder", loc="left", weight="bold")
        panel.legend(fontsize=8, loc="best")
        for panel in axes.flat:
            panel.grid(axis="y", alpha=0.18)
            panel.set_axisbelow(True)
        figure.suptitle("Controlled molecular composition pilot", fontsize=15, weight="bold")
        figure_dir = result_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure.savefig(figure_dir / "pilot_summary.png", dpi=180)
        figure.savefig(figure_dir / "pilot_summary.svg")
        plt.close(figure)


def generate_report(result_dir: Path) -> None:
    """Create REPORT.md, report_summary.json, and figures from saved raw results.

    Training, checkpoints, and recorded CSV/JSON inputs are never modified.
    Repeated adaptation samples are averaged within encoder seed; they do not
    increase the reported number of independently initialized encoders.
    """
    result_dir = Path(result_dir)
    config = _read_json(result_dir / "config.json")
    audit = _read_json(result_dir / "audit.json")
    environment = _read_json(result_dir / "environment.json")
    support_exposure = "Experience B trains on the fluoro-meta combination and is a seen-combination reference rather than an equally unseen transfer condition. Exact support-pool identity overlap is available from the saved split indices."
    if (result_dir / "splits.json").exists():
        saved_splits = _read_json(result_dir / "splits.json")
        support = set(saved_splits["adaptation_pool"])
        exposed = len(support.intersection(saved_splits["train_b"]))
        support_exposure = f"Experience B pretrained on {exposed} of the {len(support)} adaptation-pool molecular identities. It is a seen-combination reference rather than an equally unseen transfer condition."
    behavior = _read_csv(result_dir / "behavior.csv")
    probes = _read_csv(result_dir / "probes.csv")
    similarities = _read_csv(result_dir / "similarities.csv")
    adaptation = _read_csv(result_dir / "adaptation.csv")
    layers = [int(row["layer"]) for row in probes + similarities if int(row["layer"]) >= 0]
    if not layers:
        raise ValueError("No encoder layers were recorded.")
    final_layer = max(layers)
    _make_figure(result_dir, behavior, probes, similarities, adaptation, final_layer)

    normalized_probes = [{**row, "probe_condition": _probe_condition(row)} for row in probes]
    summary = {
        "final_layer": final_layer,
        "seed_aggregation": "Average repeats within each encoder seed, then mean and sample SD across seeds.",
        "behavior": _aggregate_by(behavior, ("condition", "split", "direct_f"), ("family_accuracy", "position_accuracy", "distance_accuracy", "distance_mae")),
        "probes": _aggregate_by(normalized_probes, ("probe_condition", "probe", "layer"), ("accuracy", "balanced_accuracy", "shuffled_accuracy")),
        "adaptation": _aggregate_by(adaptation, ("condition", "shots"), ("accuracy", "balanced_accuracy")),
        "similarities": [],
    }
    for kind in PAIR_NAMES:
        for layer in sorted({int(row["layer"]) for row in similarities if row["pair_kind"] == kind}):
            members = [row for row in similarities if row["pair_kind"] == kind and int(row["layer"]) == layer]
            summary["similarities"].append({"pair_kind": kind, "layer": layer, **{metric: _stats([_number(row, metric) for row in members]) for metric in ("cka", "procrustes_similarity")}})
    (result_dir / "report_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# Controlled molecular composition pilot",
        "",
        "This local pilot tests whether small graph models trained on different molecular combinations expose reusable local structural information. It measures held-out behavior, linear readout, representation geometry, and frozen-encoder adaptation. These measurements are preliminary evidence; they do not establish a causal mechanism or a general chemistry result.",
        "",
        "![Pilot summary](figures/pilot_summary.png)",
        "",
        "Panels A, B, and D show means and sample standard deviations across encoder seeds when at least two seeds are available. Adaptation repeats are averaged within each seed first. Panel C summarizes model pairs; pairs share models and are not independent statistical replicates. Fluoro-meta position labels are all meta, so a constant-meta predictor attains 100% there.",
        "",
        "## Dataset and task scope",
        "",
        f"The enumerated dataset contains {audit.get('n_molecules', 'unrecorded')} molecules and {audit.get('unique_canonical_identities', 'unrecorded')} unique canonical identities. The saved audit records identity leakage as `{audit.get('split_identity_leakage', 'unrecorded')}`. Distinct SMILES serializations of one canonical molecule belong to the same identity partition. The models consume atom graphs, so SMILES traversal is not a learned input convention in this pilot.",
        "",
        "Eight terminal-group families vary across ortho, meta, and para attachment relationships, multiple reference-group backgrounds, and methylene spacers. The name 'fluoro family' includes terminal fluorine on a sidechain when `target_spacer > 0`; only `target_spacer = 0` means directly attached aryl-F. Direct aryl-F results are reported separately when recorded.",
        "",
        "Experience A excludes all fluoro-meta candidate-training molecules. Experience B excludes chloro-meta instead. Both use the same architecture and the recorded matched training budget. IID testing excludes both held-out cells. The shared molecular partition separates candidate training, validation, and testing before experience exclusions and budget matching.",
        "",
        "Atom features include ordinary atomic information and explicit query masks for the two ring attachment atoms, the two terminal-group root atoms, and the target fragment. The query and target fragment are supplied, so this pilot tests classification and structural relationships given localization; it does not establish discovery or localization of an unknown functional group. Exact distance labels are shortest-path bond counts between the marked terminal-group roots.",
        "",
    ]
    split_counts = audit.get("split_counts", {})
    lines.extend([_table(["Split", "Molecules"], [[name, count] for name, count in split_counts.items()]), ""])
    if audit.get("effective_train_counts"):
        lines.extend(["Effective matched training counts: " + ", ".join(f"{name} = {count}" for name, count in audit["effective_train_counts"].items()) + ".", ""])
    if audit.get("distance_range"):
        lines.extend([f"Observed distance range: {audit['distance_range'][0]} to {audit['distance_range'][1]} bonds.", ""])

    lines.extend(["## Pretrained-head behavior", "", "Values are mean +/- sample SD across encoder seeds; n is the number of seeds, not the number of molecules. Distance accuracy is exact-match accuracy, and distance MAE is measured in bonds. An untrained encoder with its untrained task heads is an initialization control, not a fitted supervised baseline.", ""])
    behavior_table = []
    for condition in ("a", "b", "random"):
        for split in ("iid_test", "ood_f_meta", "ood_cl_meta"):
            members = _base_behavior(behavior, condition, split)
            if members:
                n_values = sorted({int(row["n"]) for row in members})
                behavior_table.append([CONDITION_NAMES[condition], split, ", ".join(map(str, n_values)), *[_format_stat(_seed_stats(members, metric), metric != "distance_mae") for metric in ("family_accuracy", "position_accuracy", "distance_accuracy", "distance_mae")]])
    lines.extend([_table(["Model", "Split", "Test molecules", "Family accuracy", "Position accuracy", "Distance accuracy", "Distance MAE"], behavior_table), ""])
    a_iid = _seed_stats(_base_behavior(behavior, "a", "iid_test"), "position_accuracy")
    a_ood = _seed_stats(_base_behavior(behavior, "a", "ood_f_meta"), "position_accuracy")
    if a_iid and a_ood:
        gap = a_iid["mean"] - a_ood["mean"]
        lines.extend([f"Experience A's mean IID-minus-fluoro-meta position gap is {gap * 100:.1f} percentage points. The fluoro-meta cell has one position label; its position accuracy must be interpreted alongside variable shortest-path distances and the other held-out cell. It cannot, by itself, establish compositional reasoning.", ""])

    direct_table = []
    for condition in ("a", "b", "random"):
        members = [row for row in behavior if row["condition"] == condition and _direct_only(row)]
        for split in sorted({row["split"] for row in members}):
            subgroup = [row for row in members if row["split"] == split]
            direct_table.append([CONDITION_NAMES[condition], split, ", ".join(sorted({row["n"] for row in subgroup})), *[_format_stat(_seed_stats(subgroup, metric), metric != "distance_mae") for metric in ("family_accuracy", "position_accuracy", "distance_accuracy", "distance_mae")]])
    lines.extend(["### Direct aryl-F subset", "", _table(["Model", "Split", "Test molecules", "Family accuracy", "Position accuracy", "Distance accuracy", "Distance MAE"], direct_table), ""])

    lines.extend(["## Cross-condition linear readouts", "", "The readout is a standardized multiclass ridge classifier with an unpenalized bias. Its normalization is fitted on probe-training features only, and its regularization is fixed before evaluating the probe target. Cross-position family readout fits ortho molecules and tests meta molecules with disjoint canonical identities, using target-fragment mean embeddings. Cross-family position readouts fit the six other target families and test the fluoro and chloro families. The graph version uses graph/query pooled features; the local version uses only target-fragment mean embeddings, testing whether local representations expose contextual positional information across target families. Readout fitting uses source labels, and is separate from encoder training. Representation layer 0 is the input embedding, and layers 1 through the final layer are message-passing outputs. The raw-input control has no encoder seed; its n=1 does not estimate initialization variability.", "", "Raw input and untrained-encoder controls measure information accessible before learning. In particular, target-fragment query masks and atomic element features already expose strong family cues. High family readout accuracy alone therefore does not show that training created a reusable representation. Shuffled-label readouts are additional probe controls; they are simple sample-label permutations, not the word-type control task introduced in the cited probing paper.", ""])
    probe_table = []
    for probe in ("cross_position_family", "cross_family_position", "cross_family_local_position"):
        for condition in ("raw_input", "a", "b", "random"):
            members = [row for row in normalized_probes if row["probe"] == probe and row["probe_condition"] == condition and (int(row["layer"]) < 0 if condition == "raw_input" else int(row["layer"]) == final_layer)]
            if members:
                probe_table.append([probe, CONDITION_NAMES[condition], "Raw" if condition == "raw_input" else str(final_layer), *[_format_stat(_seed_stats(members, metric)) for metric in ("accuracy", "balanced_accuracy", "shuffled_accuracy")]])
    lines.extend([_table(["Probe", "Representation", "Layer", "Accuracy", "Balanced accuracy", "Shuffled-label accuracy"], probe_table), ""])
    a_probe = _seed_stats([row for row in normalized_probes if row["probe"] == "cross_position_family" and row["probe_condition"] == "a" and int(row["layer"]) == final_layer], "balanced_accuracy")
    random_probe = _seed_stats([row for row in normalized_probes if row["probe"] == "cross_position_family" and row["probe_condition"] == "random" and int(row["layer"]) == final_layer], "balanced_accuracy")
    raw_probe = _seed_stats([row for row in normalized_probes if row["probe"] == "cross_position_family" and row["probe_condition"] == "raw_input"], "balanced_accuracy")
    if a_probe and random_probe:
        lines.extend([f"At the final layer, Experience A minus the untrained encoder has a mean cross-position family balanced-accuracy difference of {(a_probe['mean'] - random_probe['mean']) * 100:.1f} percentage points. This descriptive difference compares accessibility under the specified readout; it is not a causal attribution of the model's predictions.", ""])
    if raw_probe:
        lines.extend([f"The raw-input family readout reaches {_format_stat(raw_probe)} balanced accuracy. This baseline must be considered before attributing family decoding to representation learning.", ""])

    lines.extend(["## Representation geometry", "", "Centered linear CKA and centered, Frobenius-normalized orthogonal Procrustes similarity compare mean target-fragment embeddings on the same ordered molecular examples. Procrustes is the nuclear norm of the normalized cross-covariance; higher values mean an easier orthogonal alignment. The statistics describe geometry rather than functional transfer, and neither is required to increase after training. Wide representations and common input structure can make untrained comparisons high.", "", "The following spread is sample SD across model pairs. Because pairs can reuse a model, these counts do not provide independent confidence intervals or significance tests.", ""])
    similarity_table = []
    for kind in PAIR_NAMES:
        members = [row for row in similarities if row["pair_kind"] == kind and int(row["layer"]) == final_layer]
        if members:
            similarity_table.append([PAIR_NAMES[kind], final_layer, _format_stat(_stats([_number(row, "cka") for row in members]), False), _format_stat(_stats([_number(row, "procrustes_similarity") for row in members]), False)])
    lines.extend([_table(["Pair category", "Layer", "CKA (n=model pairs)", "Procrustes (n=model pairs)"], similarity_table), ""])

    lines.extend(["## Frozen-encoder adaptation", "", "A new ridge distance-classification head is fitted from scratch on each labeled fluoro-meta adaptation sample while the encoder remains frozen. Shot counts denote total labeled molecules, not labels per class. Each held-out adaptation test molecule has a different canonical identity from every adaptation training molecule. The pretrained task head is not reused, and this experiment is neither full-model fine-tuning nor a measurement of pretrained-head sample efficiency. No zero-shot point is inserted into the ridge-head learning curve; zero-shot pretrained-head results appear in the behavior table above.", "", "Experience A did not pretrain on the fluoro-meta adaptation pool. " + support_exposure + " Adaptation repetitions use the same support identities across conditions and seeds, and increasing shot counts uses nested subsets within each repetition. Repetitions vary labeled samples within an encoder seed; only seed-level averages contribute to the reported n and SD. Raw-feature adaptation uses one deterministic feature set, so its n=1 reports no initialization SD even though its labeled subsets are repeated.", ""])
    adaptation_table = []
    for shots in sorted({int(row["shots"]) for row in adaptation}):
        for condition in ("a", "b", "random", "raw"):
            members = [row for row in adaptation if int(row["shots"]) == shots and row["condition"] == condition]
            if members:
                repeats = len({row.get("repeat", "0") for row in members})
                adaptation_table.append([shots, CONDITION_NAMES[condition], repeats, _format_stat(_seed_stats(members, "accuracy")), _format_stat(_seed_stats(members, "balanced_accuracy"))])
    lines.extend([_table(["Total labels", "Encoder", "Repeats per seed", "Distance accuracy", "Distance balanced accuracy"], adaptation_table), ""])
    if adaptation:
        smallest = min(int(row["shots"]) for row in adaptation)
        a_small = _seed_stats([row for row in adaptation if row["condition"] == "a" and int(row["shots"]) == smallest], "balanced_accuracy")
        random_small = _seed_stats([row for row in adaptation if row["condition"] == "random" and int(row["shots"]) == smallest], "balanced_accuracy")
        if a_small and random_small:
            lines.extend([f"With {smallest} labeled molecules, Experience A minus the untrained encoder has a mean distance balanced-accuracy difference of {(a_small['mean'] - random_small['mean']) * 100:.1f} percentage points. The complete curve and seed variation above determine how broadly this observed difference holds; no significance test or causal claim is implied.", ""])

    lines.extend([
        "## Interpretation and limits",
        "",
        "This run demonstrates whether the controlled pipeline is executable and quantifies each specified comparison. Evidence for learned functional reuse requires a trained advantage over raw and untrained baselines on genuinely unseen molecular identities and combinations, together with a corresponding adaptation benefit. High CKA, high family decoding, or a single held-out-cell position score is insufficient. The present observational comparisons do not isolate which internal features cause predictions or adaptation gains.",
        "",
        "The chemical space is a synthetic disubstituted-benzene grid with supplied query locations, not a benchmark of realistic molecular properties or unlabeled motif discovery. There are no experimental property labels, no scaffold-diverse evaluation, and no three-dimensional or stereochemical task. Multiple seeds reduce initialization ambiguity, but a small pilot cannot support a stable relationship between representation geometry and transfer. A larger study should predefine more held-out family-position-background combinations, add simple graph-derived task baselines, use more seeds and scaffolds, and test interventions before claiming that a particular shared representation explains transfer.",
        "",
        "## Recorded configuration and artifacts",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "The raw inputs to this report are [config.json](config.json), [audit.json](audit.json), [environment.json](environment.json), [behavior.csv](behavior.csv), [probes.csv](probes.csv), [similarities.csv](similarities.csv), and [adaptation.csv](adaptation.csv). [report_summary.json](report_summary.json) stores machine-readable aggregates. The figure is available as [PNG](figures/pilot_summary.png) and [SVG](figures/pilot_summary.svg). Checkpoints and full training records, when saved by the runner, remain the scientific record; this report generator changes only derived documentation and figures.",
        "",
        f"Recorded device: `{environment.get('device', 'unrecorded')}`. Encoder and task-head parameter count: {environment.get('parameter_count', 'unrecorded')}. Environment metadata fields recorded: {', '.join(sorted(environment))}.",
        "",
        "## Primary methodological references",
        "",
        "- Canonical molecular identity and randomized SMILES options: [RDKit MolToSmiles API](https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles). Identity grouping here uses canonicalization rather than treating SMILES strings as independent molecules.",
        "- Representation similarity: Kornblith et al. (2019), [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). This motivates CKA as a geometric comparison; functional reuse is assessed separately in this pilot.",
        "- Probe controls: Hewitt and Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/). The paper motivates interpreting probe performance relative to controls; this pilot's sample-label shuffle is a simpler control and does not reproduce their word-type construction.",
        "",
    ])
    (result_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
