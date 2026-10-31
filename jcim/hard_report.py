"""Regenerate the harder pilot report from saved measurements without retraining."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITION_NAMES = {"a": "Experience A", "b": "Experience B", "random": "Random initialization", "raw": "Raw root attributes"}
COLORS = {"a": "#2673a6", "b": "#df812f", "random": "#777777", "raw": "#32815e"}
PAIR_NAMES = {
    "random_random": "Random / random",
    "trained_same_data": "Trained / same data",
    "trained_different_data_same_seed": "Different data / same seed",
    "trained_different_data_different_seed": "Different data / different seed",
    "random_trained": "Random / trained",
}
PAIR_COLORS = dict(zip(PAIR_NAMES, ("#777777", "#2673a6", "#df812f", "#b84e4e", "#946aa2")))
METRICS = ("motif_accuracy", "position_accuracy", "distance_accuracy", "distance_mae")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _number(row: dict, metric: str) -> float:
    value = float(row[metric])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {metric} in result row: {row!r}")
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n": len(values)}


def _seed_stats(rows: list[dict], metric: str) -> dict[str, Any] | None:
    """Repeated support draws are averaged within each encoder seed first."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric, "") != "":
            grouped[str(row["seed"])].append(_number(row, metric))
    if not grouped:
        return None
    per_seed = {seed: float(np.mean(values)) for seed, values in sorted(grouped.items())}
    return {**_stats(list(per_seed.values())), "per_seed": per_seed}


def _aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return [
        {**dict(zip(keys, key)), **{metric: _seed_stats(members, metric) for metric in metrics}}
        for key, members in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]


def _format(stat: dict | None, percent: bool = True) -> str:
    if stat is None:
        return "Unavailable"
    scale, digits, suffix = (100, 1, "%") if percent else (1, 3, "")
    mean = f"{stat['mean'] * scale:.{digits}f}"
    if stat["std"] is None:
        return f"{mean}{suffix} (n=1; SD undefined)"
    return f"{mean} +/- {stat['std'] * scale:.{digits}f}{suffix} (n={stat['n']})"


def _format_difference(stat: dict) -> str:
    return _format(stat).replace("%", " percentage points")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "No measurements were recorded.\n"
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows),
    ]) + "\n"


def _select(rows: list[dict], **filters: Any) -> list[dict]:
    return [row for row in rows if all(str(row.get(key)) == str(value) for key, value in filters.items())]


def _probe_condition(row: dict) -> str:
    return "raw" if int(row["layer"]) < 0 else row["condition"]


def _own_iid(rows: list[dict], condition: str) -> list[dict]:
    """Use each trained experience's IID support; random uses both IID splits."""
    splits = {"a": {"iid_a"}, "b": {"iid_b"}, "random": {"iid_a", "iid_b"}}[condition]
    return [row for row in rows if row["condition"] == condition and row["split"] in splits]


def _paired_difference(first: list[dict], second: list[dict], metric: str) -> dict | None:
    left, right = _seed_stats(first, metric), _seed_stats(second, metric)
    if left is None or right is None:
        return None
    seeds = sorted(set(left["per_seed"]) & set(right["per_seed"]))
    if not seeds:
        return None
    per_seed = {seed: left["per_seed"][seed] - right["per_seed"][seed] for seed in seeds}
    return {**_stats(list(per_seed.values())), "per_seed": per_seed}


def _curve(panel: Any, rows: list[dict], metric: str, x_key: str, condition: str) -> None:
    values = sorted({int(row[x_key]) for row in rows})
    if not values:
        return
    stats = [_seed_stats(_select(rows, **{x_key: value}), metric) for value in values]
    means = np.asarray([stat["mean"] for stat in stats])
    spread = np.asarray([stat["std"] or 0.0 for stat in stats])
    panel.plot(values, means, "o-", markersize=3.5, color=COLORS[condition], label=CONDITION_NAMES[condition])
    panel.fill_between(values, means - spread, means + spread, color=COLORS[condition], alpha=0.12)


def _make_figure(result_dir: Path, behavior: list[dict], probes: list[dict], adaptation: list[dict], similarities: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
        for panel, metric, title in (
            (axes[0, 0], "motif_accuracy", "A. Pretrained heads: motif connectivity"),
            (axes[0, 1], "distance_accuracy", "B. Pretrained heads: shortest path"),
        ):
            conditions = [condition for condition in ("a", "b", "random") if _select(behavior, condition=condition)]
            width = 0.24
            for offset, condition in enumerate(conditions):
                stats = [_seed_stats(_own_iid(behavior, condition), metric), _seed_stats(_select(behavior, condition=condition, split="ood_common"), metric)]
                positions = np.arange(2) + (offset - (len(conditions) - 1) / 2) * width
                panel.bar(positions, [stat["mean"] if stat else np.nan for stat in stats], width=width,
                          yerr=[(stat["std"] or 0.0) if stat else 0.0 for stat in stats], capsize=3,
                          label=CONDITION_NAMES[condition], color=COLORS[condition])
            panel.set_xticks([0, 1], ["Own-experience IID", "Common OOD combinations"])
            panel.set_ylabel("Exact-match accuracy")
            panel.set_title(title, loc="left", weight="bold")
            panel.legend(fontsize=8)

        panel = axes[0, 2]
        for condition in ("a", "b", "random"):
            _curve(panel, _select(probes, probe="cross_position_motif", probe_condition=condition), "balanced_accuracy", "layer", condition)
        raw = _seed_stats(_select(probes, probe="cross_position_motif", probe_condition="raw"), "balanced_accuracy")
        if raw:
            panel.axhline(raw["mean"], color=COLORS["raw"], linestyle="--", label="Raw root attributes")
        panel.axhline(1 / 8, color="#aaaaaa", linestyle=":", label="Uniform 8-class reference")
        panel.set_title("C. Local motif readout: ortho to meta", loc="left", weight="bold")
        panel.set_xlabel("Representation layer")
        panel.set_ylabel("Balanced accuracy")
        panel.legend(fontsize=8)

        for panel, split, metric, title in (
            (axes[1, 0], "ood_common", "motif_accuracy", "D. Common OOD adaptation: motif"),
            (axes[1, 1], "ood_common", "distance_accuracy", "E. Common OOD adaptation: shortest path"),
            (axes[1, 2], "own_iid", "motif_accuracy", "F. Own-experience IID guard: motif"),
        ):
            for condition in ("a", "b", "random"):
                members = _own_iid(adaptation, condition) if split == "own_iid" else _select(adaptation, condition=condition, split=split)
                _curve(panel, members, metric, "shots", condition)
            shots = sorted({int(row["shots"]) for row in adaptation})
            if shots:
                panel.set_xscale("symlog", base=2, linthresh=max(1, min((shot for shot in shots if shot > 0), default=1)))
                panel.set_xticks(shots, [str(shot) for shot in shots])
            panel.set_title(title, loc="left", weight="bold")
            panel.set_xlabel("New labeled support molecules (0 = pretrained head)")
            panel.set_ylabel("Exact-match accuracy")
            panel.legend(fontsize=8)

        panel = axes[2, 0]
        for condition in ("a", "b", "random"):
            _curve(panel, _own_iid(adaptation, condition), "distance_accuracy", "shots", condition)
        shots = sorted({int(row["shots"]) for row in adaptation})
        if shots:
            panel.set_xscale("symlog", base=2, linthresh=max(1, min((shot for shot in shots if shot > 0), default=1)))
            panel.set_xticks(shots, [str(shot) for shot in shots])
        panel.set_title("G. Own-experience IID guard: shortest path", loc="left", weight="bold")
        panel.set_xlabel("New labeled support molecules")
        panel.set_ylabel("Exact-match accuracy")
        panel.legend(fontsize=8)

        for panel, pooling, title in (
            (axes[2, 1], "local", "H. Local representation geometry"),
            (axes[2, 2], "graph", "I. Graph representation geometry"),
        ):
            for kind, label in PAIR_NAMES.items():
                members = _select(similarities, pair_kind=kind, pooling=pooling)
                layers = sorted({int(row["layer"]) for row in members})
                if layers:
                    stats = [_stats([_number(row, "cka") for row in _select(members, layer=layer)]) for layer in layers]
                    panel.plot(layers, [stat["mean"] for stat in stats], "o-", markersize=3, label=label, color=PAIR_COLORS[kind])
            panel.set_title(title, loc="left", weight="bold")
            panel.set_xlabel("Representation layer")
            panel.set_ylabel("Mean centered linear CKA across model pairs")
            panel.legend(fontsize=7)
        for panel in axes.flat:
            panel.set_ylim(-0.02, 1.08)
            panel.grid(axis="y", alpha=0.18)
            panel.set_axisbelow(True)
        figure.suptitle("Harder molecular composition pilot: behavior, local readout, full-model adaptation, and geometry", fontsize=15, weight="bold")
        directory = result_dir / "figures"
        directory.mkdir(exist_ok=True)
        figure.savefig(directory / "hard_summary.png", dpi=160)
        figure.savefig(directory / "hard_summary.svg")
        plt.close(figure)


def _support_audit(saved_splits: dict) -> dict:
    support = set(saved_splits.get("adaptation_pool", []))
    overlap = {condition: len(support & set(saved_splits.get(f"train_{condition}", []))) for condition in ("a", "b")}
    return {"support_molecules": len(support), "pretraining_identity_overlap": overlap} if support else {}


def _motif_figure(result_dir: Path) -> bool:
    """Depict recorded fragments with the single local-readout root highlighted."""
    source = result_dir / "molecules.jsonl"
    if not source.exists():
        return False
    from rdkit import Chem
    from rdkit.Chem import Draw

    motifs = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if "motif_smiles" in record:
                key = int(record.get("y_motif", record.get("y_family", len(motifs))))
                motifs[key] = (record.get("target_family", f"motif_{key}"), record["motif_smiles"])
            if len(motifs) == 8:
                break
    if not motifs:
        return False
    molecules = [Chem.MolFromSmiles(smiles) for _, (_, smiles) in sorted(motifs.items())]
    if any(molecule is None for molecule in molecules):
        raise ValueError("A recorded target motif could not be parsed for its structure figure.")
    grid = Draw.MolsToGridImage(molecules, molsPerRow=4, subImgSize=(310, 220),
        legends=[f"{name}: root highlighted" for _, (name, _) in sorted(motifs.items())],
        highlightAtomLists=[[0] for _ in molecules])
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    grid.save(str(directory / "motif_grid.png"))
    return True


def generate_hard_report(result_dir: Path) -> None:
    """Write English derived report artifacts; preserve every recorded input.

    Support repetitions are nested inside encoder seeds. Model pairs in geometry
    summaries reuse encoders, so their spread is descriptive rather than an
    independent-sample uncertainty estimate.
    """
    result_dir = Path(result_dir)
    config = _read_json(result_dir / "config.json", {})
    audit = _read_json(result_dir / "audit.json", {})
    environment = _read_json(result_dir / "environment.json", {})
    splits = _read_json(result_dir / "splits.json", {})
    behavior = _read_csv(result_dir / "behavior.csv")
    probes = [{**row, "probe_condition": _probe_condition(row)} for row in _read_csv(result_dir / "probes.csv")]
    adaptation = _read_csv(result_dir / "adaptation.csv")
    similarities = _read_csv(result_dir / "similarities.csv")
    layers = [int(row["layer"]) for row in probes + similarities if int(row["layer"]) >= 0]
    if not layers:
        raise ValueError("No encoder layers were recorded.")
    final_layer = max(layers)
    support = _support_audit(splits)
    summary = {
        "final_layer": final_layer,
        "seed_aggregation": "Average support repeats within encoder seed, then mean and sample SD across seeds.",
        "geometry_aggregation": "Descriptive model-pair mean and sample SD; pairs share models and are not independent replicates.",
        "support_audit": support,
        "behavior": _aggregate(behavior, ("condition", "split"), METRICS),
        "probes": _aggregate(probes, ("probe_condition", "probe", "layer"), ("accuracy", "balanced_accuracy", "shuffled_accuracy")),
        "adaptation": _aggregate(adaptation, ("condition", "split", "shots"), METRICS),
        "similarities": [],
        "paired_contrasts": [],
        "adaptation_changes": [],
    }
    for pooling in sorted({row["pooling"] for row in similarities}):
        for kind in PAIR_NAMES:
            for layer in sorted({int(row["layer"]) for row in _select(similarities, pair_kind=kind, pooling=pooling)}):
                members = _select(similarities, pair_kind=kind, pooling=pooling, layer=layer)
                summary["similarities"].append({"pooling": pooling, "pair_kind": kind, "layer": layer,
                    **{metric: _stats([_number(row, metric) for row in members]) for metric in ("cka", "procrustes_similarity")}})
    for split in sorted({row["split"] for row in behavior}):
        for metric in METRICS:
            difference = _paired_difference(_select(behavior, condition="a", split=split), _select(behavior, condition="b", split=split), metric)
            if difference:
                summary["paired_contrasts"].append({"comparison": "a_minus_b", "split": split, "metric": metric, **difference})
    for condition in ("a", "b", "random"):
        for shots in sorted({int(row["shots"]) for row in adaptation if int(row["shots"]) > 0}):
            current = _select(adaptation, condition=condition, shots=shots)
            baseline = _select(adaptation, condition=condition, shots=0)
            changes = {"condition": condition, "shots": shots}
            for label, current_rows, baseline_rows in (
                ("ood", _select(current, split="ood_common"), _select(baseline, split="ood_common")),
                ("own_iid", _own_iid(current, condition), _own_iid(baseline, condition)),
            ):
                for metric in ("motif_accuracy", "distance_accuracy"):
                    changes[f"{label}_{metric}_change"] = _paired_difference(current_rows, baseline_rows, metric)
            if current:
                summary["adaptation_changes"].append(changes)
    (result_dir / "report_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _make_figure(result_dir, behavior, probes, adaptation, similarities)
    has_motif_figure = _motif_figure(result_dir)

    lines = [
        "# Harder molecular composition pilot", "",
        "This controlled local experiment tests whether graph models trained on different motif-position combinations expose local connectivity information that can be read across positions and used when adapting to held-out combinations. The measurements are descriptive and do not establish a causal representation mechanism or a molecular-property result.", "",
        "![Harder pilot summary](figures/hard_summary.png)", "",
        "Behavior, readout, and adaptation panels show means and sample standard deviations across encoder seeds. Support repetitions are averaged within each seed first; they do not increase the number of independently initialized encoders. Geometry curves show descriptive means across model pairs, which can share models and are not independent statistical replicates.", "",
        "## Design and identity safeguards", "",
        f"The recorded dataset contains {audit.get('n_molecules', 'unrecorded')} molecules and {audit.get('unique_canonical_identities', 'unrecorded')} unique canonical identities. Molecules are partitioned by canonical identity before training; graph inputs do not expose alternative SMILES traversals as independent samples. The recorded identity-leakage check is `{audit.get('split_identity_leakage', 'unrecorded')}`.", "",
        "Eight nine-heavy-atom target sidechain motifs contain seven carbons and two ether oxygens and have equal multisets of ordinary input atom attributes, while their connectivity differs. Their marked attachment roots also have identical raw attributes. Element counts and other order-free ordinary atom-attribute bags therefore cannot identify the motif. Ortho, meta, and para query relationships, benzene and two pyridine scaffold variants, and a third distractor vary the molecular context. Query markers identify attachment and endpoint atoms. The model receives no oracle target-fragment mask. The local representation is the embedding at the marked target attachment atom; classification therefore requires information beyond that atom's ordinary attributes.", "",
        "The combination partition uses `(motif + position) % 3`: Experience A pretrains only on residue 0, Experience B only on residue 1, and their common OOD evaluation uses residue 2. Each experience sees all eight motifs and all three positions but only one training position for each motif. This deliberately strong motif-position correlation makes shortcut use a serious risk. Both experiences use the same architecture, equal training counts, and matched training budgets. The common OOD evaluation includes all three positions and all eight motif classes, and its combination types are unseen by both experiences. A constant position predictor cannot solve it.", "",
        "The supervised tasks classify target connectivity, classify the relative attachment position, and predict the exact shortest-path bond count from the marked anchor functional root to a marked target-sidechain carbon leaf. The leaf query varies by molecular identity between the nearest and farthest target carbon leaves relative to the target root. These are graph-structure tasks with computed answers; no experimental chemical-property labels, three-dimensional conformations, or stereochemical predictions are used.", "",
    ]
    if has_motif_figure:
        lines.extend(["![Recorded target motifs](figures/motif_grid.png)", "", "Target fragments are drawn from recorded motif SMILES. The highlighted atom is the attachment root used for the single-atom local readout; the full molecule also supplies the ring and marked path endpoint.", ""])
    lines.extend([_table(["Split", "Molecules"], [[key, value] for key, value in audit.get("split_counts", {}).items()]), ""])
    if audit.get("effective_train_counts"):
        lines.extend(["Effective training counts: " + ", ".join(f"{key} = {value}" for key, value in audit["effective_train_counts"].items()) + ".", ""])
    if audit.get("train_position_counts"):
        lines.extend(["Training is matched by motif count across experiences, while global position marginals can differ because each motif occurs at one position per experience. Matching total counts does not remove the designed motif-position correlation.", "",
            _table(["Experience", "Recorded training position counts"], [[key, json.dumps(value, sort_keys=True)] for key, value in audit["train_position_counts"].items()]), ""])
    if support:
        overlap = support["pretraining_identity_overlap"]
        lines.extend([f"The adaptation pool contains {support['support_molecules']} canonical molecular identities. Exact support-pool identity overlap with pretraining is A = {overlap['a']}, B = {overlap['b']}. Support availability does not mean that these labels were used in pretraining.", ""])

    behavior_table = []
    for condition in ("a", "b", "random"):
        for split in sorted({row["split"] for row in behavior}):
            members = _select(behavior, condition=condition, split=split)
            if members:
                behavior_table.append([CONDITION_NAMES[condition], split, ", ".join(sorted({row["n"] for row in members})),
                    *[_format(_seed_stats(members, metric), metric != "distance_mae") for metric in METRICS]])
    lines.extend(["## Pretrained-head behavior", "",
        "Accuracy is exact-match accuracy; distance MAE is measured in bonds. The random-initialization rows include untrained encoders and untrained task heads, so they are initialization controls rather than fitted supervised baselines. All comparisons report seed variation, not a molecule-level confidence interval. In the figure, own-experience IID means `iid_a` for A and `iid_b` for B; the random reference averages its two IID split scores within seed. Those IID distributions differ, so the table reports both splits for every condition and the shared OOD set is the directly comparable challenge.", "",
        _table(["Model", "Split", "Test molecules", "Motif accuracy", "Position accuracy", "Distance accuracy", "Distance MAE"], behavior_table), ""])
    for metric in ("motif_accuracy", "distance_accuracy"):
        difference = _paired_difference(_select(behavior, condition="a", split="ood_common"), _select(behavior, condition="b", split="ood_common"), metric)
        if difference:
            lines.extend([f"The matched-seed A-minus-B common OOD {metric.replace('_', ' ')} difference is {_format_difference(difference)}. This paired descriptive contrast compares training experiences; it does not isolate the causal effect of any representation feature.", ""])

    probe_table = []
    for condition in ("raw", "a", "b", "random"):
        members = [row for row in _select(probes, probe="cross_position_motif", probe_condition=condition) if int(row["layer"]) == (-1 if condition == "raw" else final_layer)]
        if members:
            probe_table.append([CONDITION_NAMES[condition], "Raw" if condition == "raw" else final_layer,
                *[_format(_seed_stats(members, metric)) for metric in ("accuracy", "balanced_accuracy", "shuffled_accuracy")]])
    lines.extend(["## Local cross-position connectivity readout", "",
        "A standardized multiclass ridge readout fits ortho local representations and evaluates meta local representations on disjoint molecular identities. Normalization uses probe-training features only, and regularization is fixed before evaluating the target split. Balanced accuracy averages recall across target classes. Layer 0 is the input embedding; higher layers are message-passing outputs. The raw control uses the marked attachment atom's input attributes rather than an oracle fragment average. Readout fitting uses labeled source examples and is separate from encoder training.", "",
        "The raw-root and random-encoder controls measure information accessible before learning. Random message passing can expose connectivity even when raw single-atom attributes cannot. A trained readout advantage indicates improved accessibility under this probe, and does not by itself demonstrate that the model uses that information for its task. Shuffled training labels provide an additional sample-label permutation control.", "",
        _table(["Representation", "Layer", "Accuracy", "Balanced accuracy", "Shuffled-label accuracy"], probe_table), ""])
    raw = _seed_stats(_select(probes, probe="cross_position_motif", probe_condition="raw"), "balanced_accuracy")
    if raw:
        lines.extend([f"The raw-root control has {_format(raw)} balanced accuracy. Its single deterministic feature set has no encoder-initialization standard deviation.", ""])
    for condition in ("a", "b"):
        difference = _paired_difference(_select(probes, probe="cross_position_motif", probe_condition=condition, layer=final_layer), _select(probes, probe="cross_position_motif", probe_condition="random", layer=final_layer), "balanced_accuracy")
        if difference:
            lines.extend([f"At the final layer, {CONDITION_NAMES[condition]} minus the matched-seed random encoder has a cross-position balanced-accuracy difference of {_format_difference(difference)}. Positive and negative differences are both reported without treating high decoding as proof of functional reuse.", ""])

    adaptation_table = []
    for shots in sorted({int(row["shots"]) for row in adaptation}):
        for condition in ("a", "b", "random"):
            for split in ("ood_common", "iid_a", "iid_b"):
                members = _select(adaptation, shots=shots, condition=condition, split=split)
                if members:
                    adaptation_table.append([shots, CONDITION_NAMES[condition], split, len({row.get("repeat", "0") for row in members}),
                        *[_format(_seed_stats(members, metric), metric != "distance_mae") for metric in METRICS]])
    lines.extend(["## Full-model adaptation with the original heads", "",
        "Zero shots reuses each model's original task heads without updating weights. Positive shot counts fine-tune the encoder and those same task heads, rather than fitting a new frozen-encoder ridge head. The random condition fine-tunes from initialization. Shot counts denote total newly labeled adaptation molecules, not examples per class. All conditions and seeds receive matched support identities; increasing counts use nested support subsets within each repetition. Support identities are excluded from both pretraining experiences and from the adaptation evaluation sets.", "",
        "The common OOD curve must be assessed alongside the IID guard curve. Higher OOD accuracy accompanied by a large IID decrease can reflect specialization or forgetting rather than broadly useful transfer. This run does not replay pretraining data during adaptation. The exact adaptation steps, learning rate, support repetitions, and sampling seeds are recorded in the configuration and training logs. Both IID splits appear below; the figure's IID guard uses each experience's own IID split, as in the behavior panels.", "",
        _table(["New labels", "Initialization", "Evaluation", "Support repeats", "Motif accuracy", "Position accuracy", "Distance accuracy", "Distance MAE"], adaptation_table), ""])
    change_table = []
    for row in summary["adaptation_changes"]:
        change_table.append([row["shots"], CONDITION_NAMES[row["condition"]], *[
            _format_difference(row[key]) if row[key] else "Unavailable" for key in (
                "ood_motif_accuracy_change", "ood_distance_accuracy_change",
                "own_iid_motif_accuracy_change", "own_iid_distance_accuracy_change")]])
    lines.extend(["Changes below compare each adapted model against its own zero-shot baseline within the same seed. Support repetitions are averaged before taking the paired seed difference. Gains and retention are separate measurements; aggregate motif and distance accuracies do not identify the fraction of molecules correct on both tasks.", "",
        _table(["New labels", "Initialization", "OOD motif change", "OOD distance change", "Own-IID motif change", "Own-IID distance change"], change_table), ""])

    similarity_table = []
    for item in summary["similarities"]:
        if item["layer"] == final_layer:
            similarity_table.append([item["pooling"], PAIR_NAMES[item["pair_kind"]], final_layer, _format(item["cka"], False), _format(item["procrustes_similarity"], False)])
    lines.extend(["## Representation geometry and initialization controls", "",
        "Centered linear CKA and centered, Frobenius-normalized orthogonal Procrustes similarity compare identical ordered molecules at each layer. Local comparisons use the marked target attachment atom; graph comparisons use the recorded graph representation. The factorial controls separate same-data/different-seed, different-data/same-seed, and different-data/different-seed comparisons, alongside random/random and random/trained pairs. Neither similarity measure is required to increase with learning, and random encoders can already have high similarity.", "",
        "The counts below are model-pair counts. Pairs reuse encoders and therefore do not provide independent replicates, confidence intervals, or evidence of statistical significance. Cross-experience same-seed comparisons can also reflect shared initialization. High CKA or Procrustes similarity alone does not establish functional transfer.", "",
        _table(["Representation", "Pair category", "Layer", "CKA (n=pairs)", "Procrustes (n=pairs)"], similarity_table), "",
        "## Interpretation and limits", "",
        "Evidence for learned reusable local information requires an advantage over both raw-root and random-encoder readouts, behavior on genuinely held-out combination types, and useful adaptation with acceptable IID retention. These comparisons can disagree. Saturated readouts can occur before learning; similar geometry can coexist with poor transfer; and adaptation improvements can arise from supervision and optimization without identifying a local representation mechanism. No automatic positive conclusion is inferred from any one metric.", "",
        "The chemical space remains a controlled synthetic aromatic scaffold grid with supplied query locations and a limited collection of backgrounds. Scaffold variants and distractors increase difficulty, but this is not a broad scaffold-held-out benchmark, experimental molecular-property prediction, or unknown-motif localization. Three seeds, when used, provide a small descriptive sample; no population-level correlation between geometry and transfer or causal explanation is established. The previous pilot archive remains separate from this experiment.", "",
        "## Recorded configuration and artifacts", "", "```json", json.dumps(config, indent=2, sort_keys=True), "```", "",
        "Recorded inputs: [config.json](config.json), [audit.json](audit.json), [environment.json](environment.json), [splits.json](splits.json), [behavior.csv](behavior.csv), [probes.csv](probes.csv), [adaptation.csv](adaptation.csv), and [similarities.csv](similarities.csv). Machine-readable aggregates: [report_summary.json](report_summary.json). Figures: [PNG](figures/hard_summary.png) and [SVG](figures/hard_summary.svg). Report regeneration changes only derived documentation and figures, preserving recorded measurements, model weights, and other archived scientific inputs.", "",
        f"Recorded device: `{environment.get('device', 'unrecorded')}`. Encoder and task-head parameters: {environment.get('parameter_count', 'unrecorded')}. Environment metadata fields: {', '.join(sorted(environment))}.", "",
    ])
    (result_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
