"""Report branch sensitivity and two explicitly separate frozen-readout protocols."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .run import write_csv, write_json


REGIME_LABELS = {"end_to_end": "V5 end-to-end encoder",
                 "frozen_readout": "Original trained encoder", "random": "Frozen random encoder"}
POOL_LABELS = {"whole_graph": "Whole graph", "target": "Oracle target"}
SCORE_KEYS = ("accuracy", "balanced_accuracy")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def seed_statistics(values):
    """One value per encoder seed, with sample SD and retained paired values."""
    if not values:
        raise ValueError("Cannot summarize an empty seed panel.")
    values = {str(key): float(value) for key, value in sorted(values.items(), key=lambda item: str(item[0]))}
    numeric = list(values.values())
    return {"mean": float(np.mean(numeric)),
            "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else None,
            "n_seeds": len(numeric), "per_seed": values}


def summarize_source(rows):
    groups = defaultdict(dict)
    fields = ("regime", "experience", "classifier", "pool", "split")
    for row in rows:
        key = tuple(row[field] for field in fields)
        seed = str(row["seed"])
        if seed in groups[key]:
            raise ValueError("Duplicate source-readout seed endpoint.")
        groups[key][seed] = row
    summaries = []
    for key, seeds in sorted(groups.items()):
        summaries.append({**dict(zip(fields, key)), **{
            metric: seed_statistics({seed: row[metric] for seed, row in seeds.items()})
            for metric in SCORE_KEYS}})
    return summaries


def pair_source(rows):
    groups = defaultdict(dict)
    fields = ("regime", "experience", "classifier", "split", "seed")
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if row["pool"] in groups[key]:
            raise ValueError("Duplicate pooling scope in paired source endpoints.")
        groups[key][row["pool"]] = row
    result = []
    for key, pools in sorted(groups.items()):
        if set(pools) != {"whole_graph", "target"}:
            raise ValueError("A paired source endpoint is missing a pooling scope.")
        if pools["target"]["n_test"] != pools["whole_graph"]["n_test"]:
            raise ValueError("Paired source endpoints have different denominators.")
        result.append({**dict(zip(fields, key)), "contrast": "oracle_target_minus_whole_graph",
            **{metric: float(pools["target"][metric]) - float(pools["whole_graph"][metric])
               for metric in SCORE_KEYS}})
    return result


def summarize_pairs(rows):
    groups = defaultdict(list)
    fields = ("regime", "experience", "classifier", "split")
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        if len({row["seed"] for row in members}) != len(members):
            raise ValueError("Duplicate paired seed.")
        result.append({**dict(zip(fields, key)), **{
            metric: seed_statistics({row["seed"]: row[metric] for row in members})
            for metric in SCORE_KEYS}})
    return result


def summarize_direct(rows):
    """Average all support draws and directions within seed before uncertainty."""
    groups = defaultdict(list)
    fields = ("regime", "experience", "pool", "split", "seed")
    seen = set()
    random_duplicates = {}
    for row in rows:
        if row["regime"] == "random" and row["split"] != "target_own_iid":
            key = tuple(str(row[field]) for field in
                        ("seed", "pool", "split", "repeat", "source_position", "target_position"))
            score = tuple(str(row[field]) for field in ("n_test", *SCORE_KEYS))
            if key in random_duplicates and random_duplicates[key] != score:
                raise ValueError("Shared random-encoder reference differs between A and B.")
            random_duplicates[key] = score
    for original in rows:
        row = dict(original)
        # These full-analysis and common-OOD predictions are identical for the
        # shared random encoder scored under A/B. Count that encoder only once.
        if row["regime"] == "random" and row["split"] != "target_own_iid":
            if row["experience"] != "a":
                continue
            row["experience"] = "shared"
        identity = tuple(str(row[field]) for field in (*fields, "repeat", "source_position", "target_position"))
        if identity in seen:
            raise ValueError("Duplicate directed probe endpoint.")
        seen.add(identity)
        groups[tuple(str(row[field]) for field in fields)].append(row)
    seed_rows = []
    for key, members in sorted(groups.items()):
        directions = {(row["source_position"], row["target_position"]) for row in members}
        repeats = {row["repeat"] for row in members}
        if len(directions) != 6 or len(members) != len(repeats) * 6:
            raise ValueError("Directed probes must include all six comparisons for every support draw.")
        seed_rows.append({**dict(zip(fields, key)), "n_directions": 6,
            "n_support_draws": len(repeats),
            **{metric: float(np.mean([float(row[metric]) for row in members])) for metric in SCORE_KEYS}})
    panel_groups = defaultdict(list)
    for row in seed_rows:
        panel_groups[tuple(row[field] for field in fields[:-1])].append(row)
    panels = [{**dict(zip(fields[:-1], key)), **{
        metric: seed_statistics({row["seed"]: row[metric] for row in members})
        for metric in SCORE_KEYS}} for key, members in sorted(panel_groups.items())]
    return seed_rows, panels


def select_one(rows, **filters):
    selected = [row for row in rows if all(str(row.get(key)) == str(value) for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one panel for {filters}, found {len(selected)}.")
    return selected[0]


def score_text(stats, multiplier=100, precision=1):
    mean = stats["mean"] * multiplier
    if stats["std"] is None:
        return f"{mean:.{precision}f} (one seed)"
    return f"{mean:.{precision}f} +/- {stats['std'] * multiplier:.{precision}f}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def _branch_stat(row, key):
    return {"mean": float(row[f"{key}_seed_mean"]),
            "std": float(row[f"{key}_seed_sd"]) if int(row["n_seeds"]) > 1 else None}


def _figures(output, source, pairs, direct, branch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    regimes = ("end_to_end", "frozen_readout", "random")
    colors = {"whole_graph": "#317ba3", "target": "#bb7236"}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharey=True)
    for row_index, classifier in enumerate(("ridge", "mlp")):
        for col_index, regime in enumerate(regimes):
            axis = axes[row_index, col_index]
            for pool_index, pool in enumerate(("whole_graph", "target")):
                for experience_index, experience in enumerate(("a", "b")):
                    stats = select_one(source, regime=regime, experience=experience,
                                       classifier=classifier, pool=pool, split="ood_common")["accuracy"]
                    x = experience_index + (pool_index - .5) * .30
                    axis.bar(x, 100 * stats["mean"], .27, color=colors[pool], alpha=.8,
                             label=POOL_LABELS[pool] if experience_index == 0 else None)
                    if stats["std"] is not None:
                        axis.errorbar(x, 100 * stats["mean"], yerr=100 * stats["std"],
                                      color="#333333", capsize=3, fmt="none", linewidth=1)
                    values = list(stats["per_seed"].values())
                    jitter = np.linspace(-.045, .045, len(values))
                    axis.scatter(x + jitter, np.asarray(values) * 100, s=18, color="#222222", zorder=4)
            axis.axhline(12.5, linestyle=":", color="#777777", linewidth=1)
            axis.set(xticks=[0, 1], xticklabels=["A", "B"], ylim=(-2, 111),
                     title=f"{REGIME_LABELS[regime]}\n{classifier.upper()} readout")
            axis.grid(axis="y", alpha=.17)
            if col_index == 0:
                axis.set_ylabel("Common-OOD motif accuracy (%)")
            if row_index == 0 and col_index == 2:
                axis.legend(loc="upper right", fontsize=9)
    fig.suptitle("Original source labels: matched frozen readouts with and without oracle localization", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for extension in ("png", "svg"):
        fig.savefig(figures / f"pooling_source.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    panels = [(regime, experience) for regime in ("end_to_end", "frozen_readout")
              for experience in ("a", "b")]
    for metric_index, (metric, ylabel, multiplier) in enumerate((
            ("prediction_changed_rate", "OOD motif predictions changed (%)", 100),
            ("absolute_margin_delta_mean", "Mean absolute correct-class margin change", 1))):
        axis = axes[metric_index]
        for panel_index, (regime, experience) in enumerate(panels):
            for intervention_index, intervention in enumerate(("zero_branch", "permuted_branch")):
                row = select_one(branch, regime=regime, experience=experience, variant="shallow_skip",
                                 split="ood_common", task="family", intervention=intervention)
                stats = _branch_stat(row, metric)
                x = panel_index + (intervention_index - .5) * .31
                axis.bar(x, multiplier * stats["mean"], .28,
                         color=("#bb7236", "#317ba3")[intervention_index],
                         label=("Zero branch", "Permute branch")[intervention_index] if panel_index == 0 else None)
                if stats["std"] is not None:
                    axis.errorbar(x, multiplier * stats["mean"], yerr=multiplier * stats["std"],
                                  fmt="none", color="#333333", capsize=3, linewidth=1)
        axis.set(xticks=range(4), xticklabels=["Joint A", "Joint B", "Frozen A", "Frozen B"], ylabel=ylabel)
        axis.grid(axis="y", alpha=.18)
        axis.legend(fontsize=9)
    axes[1].set_title("Raw logit units are specific to each trained head", fontsize=10)
    fig.suptitle("Existing shallow-skip models: branch output sensitivity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for extension in ("png", "svg"):
        fig.savefig(figures / f"branch_outputs.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_pooling_report(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    environment = json.loads((output / "environment.json").read_text())
    source_rows = read_csv(output / "pooling/pooling_source_metrics.csv")
    direct_rows = read_csv(output / "pooling/pooling_direct_metrics.csv")
    branch = read_csv(output / "branch/branch_seed_aggregate.csv")
    source = summarize_source(source_rows)
    paired_rows = pair_source(source_rows)
    pairs = summarize_pairs(paired_rows)
    direct_seed, direct = summarize_direct(direct_rows)
    expected_seeds = {str(seed) for seed in config["seeds"]}
    for panel in (*source, *pairs, *direct):
        for metric in SCORE_KEYS:
            if set(panel[metric]["per_seed"]) != expected_seeds:
                raise ValueError("A derived endpoint is missing a configured encoder seed.")
    if any(int(row["n_seeds"]) != len(expected_seeds) for row in branch):
        raise ValueError("A branch summary is missing a configured encoder seed.")
    write_csv(output / "pooling_source_paired.csv", paired_rows)
    write_csv(output / "pooling_direct_seed.csv", direct_seed)
    write_json(output / "report_summary.json", {"source": source, "source_pairs": pairs,
                                               "direct": direct, "branch": branch})
    source_table = []
    for regime in ("end_to_end", "frozen_readout", "random"):
        for experience in ("a", "b"):
            for classifier in ("ridge", "mlp"):
                scores = [select_one(source, regime=regime, experience=experience, classifier=classifier,
                                     pool=pool, split="ood_common")["accuracy"]
                          for pool in ("whole_graph", "target")]
                contrast = select_one(pairs, regime=regime, experience=experience, classifier=classifier,
                                      split="ood_common")["accuracy"]
                source_table.append([REGIME_LABELS[regime], experience.upper(), classifier.upper(),
                                     *(score_text(score) for score in scores), score_text(contrast)])
    direct_table = []
    for regime, experience in [(regime, experience) for regime in ("end_to_end", "frozen_readout")
                              for experience in ("a", "b")] + [("random", "shared")]:
        scores = [select_one(direct, regime=regime, experience=experience, pool=pool,
                             split="target_all")["balanced_accuracy"] for pool in ("whole_graph", "target")]
        direct_table.append([REGIME_LABELS[regime], experience.upper(), *(score_text(score) for score in scores)])
    branch_table = []
    for regime in ("end_to_end", "frozen_readout"):
        for experience in ("a", "b"):
            for intervention in ("zero_branch", "permuted_branch"):
                row = select_one(branch, regime=regime, experience=experience, variant="shallow_skip",
                                 split="ood_common", task="family", intervention=intervention)
                values = [_branch_stat(row, metric) for metric in ("prediction_changed_rate", "margin_delta_mean",
                          "absolute_margin_delta_mean", "centered_logit_l2_mean")]
                branch_table.append([REGIME_LABELS[regime], experience.upper(), intervention,
                    score_text(values[0], 100, 3), *(score_text(value, 1, 5) for value in values[1:])])
    _figures(output, source, pairs, direct, branch)
    n_seeds = len(config["seeds"])
    smoke = "\n**Pipeline smoke run only; excluded from primary evidence.**\n" if environment["pipeline_only"] else ""
    prose = f"""# Frozen-model branch sensitivity and pooling diagnostics
{smoke}
This record separates output sensitivity of the completed v5 classifiers from two new frozen-feature fitting protocols. No encoder or architecture is trained. Known target membership is supplied only to the oracle diagnostic pool. Source-only fits and separately labeled cross-position probes are different experiments.

## Original-source-label readouts

Each fit uses exactly the experience's original 1,386 training identities and motif labels. The whole-graph vector is the actual 64D layer-1 feature appended to the v5 classifier; it is checked against the archived v5 vector. The oracle-target vector averages the nine annotated substituent atoms at the same layer, with the same full molecular input and encoder. It supplies additional localization information. Both pools use support-only standardization and a matched readout: fixed-alpha-1 multiclass ridge, or a fresh 64-to-64-to-8 ReLU classifier. MLP initialization and all original training orders are matched across pools; every fit receives {config['epochs']} epochs. All settings and the final epoch are fixed before observing OOD outcomes.

The new MLP uses motif-only loss, whereas v5 trained three task heads jointly. Comparisons between the two pools hold this diagnostic fitting protocol fixed. Comparisons with the original v5 classifier additionally change its fitting conditions. Oracle performance is conditional on supplied boundaries and is a diagnostic upper bound, not a boundary-free prediction result.

The co-primary diagnostic endpoints below are ordinary common-OOD motif accuracy for each fixed classifier. Entries are percentages, or percentage points for the paired oracle-minus-whole contrast, as mean +/- sample SD across {n_seeds} seeds. Seed contrasts are computed before aggregation. A/B and encoder regimes remain separate.

{_table(['Encoder', 'Experience', 'Readout', 'Whole graph (%)', 'Oracle target (%)', 'Paired gain (pp)'], source_table)}

![Original-source-label frozen readouts](figures/pooling_source.png)

Fit accuracy, both IID guards, ordinary and balanced accuracy, all individual seeds, and readout states remain in the machine-readable record. The random encoder is shared across A/B fits; those are not six independent initializations. Frozen-v5 encoders are the original trained encoders. Uniform eight-output guessing has expected accuracy 12.5%, distinct from measured random-encoder performance.

## Separately labeled cross-position probes

These standardized alpha-1 ridge fits reuse the exact archived v4 supports: 256 balanced labels from one source position, three draws, tested on the other positions. A source support can contain combinations excluded from original encoder training. No target-position label occurs in its directed fit, but this is additional diagnostic supervision and is not the original-source-only task protocol.

The table reports balanced accuracy on all target-position analysis examples. First average all six directions and all support draws within each seed, then report mean +/- sample SD across encoder seeds. Directions and draws are not independent seeds. The shared random all-analysis reference is counted once.

{_table(['Encoder', 'Experience', 'Whole graph (%)', 'Oracle target (%)'], direct_table)}

## Existing branch outputs

Correct-class margin is `z[y] - max(z[k] for k != y)`. Margin delta is altered minus intact; positive values favor the truth. Every saved intact, zeroed, and permuted class prediction must reproduce v5 before analysis. Deep coordinates and final heads remain fixed. The following table uses the shallow branch and common-OOD motif head. Flip rates use all test examples. Each mean is first computed within model; uncertainty is sample SD across seeds. Signed averages can cancel large individual changes, so absolute changes and centered-logit distances are retained too.

{_table(['Encoder', 'Experience', 'Intervention', 'Predictions changed (%)', 'Mean margin delta', 'Mean absolute delta', 'Mean centered-logit L2'], branch_table)}

![Existing shallow branch output sensitivity](figures/branch_outputs.png)

Raw margins and distances are specific to each trained head's logit scale, rather than calibrated effect sizes across models. Complete per-example values, quantiles, correct-to-wrong and wrong-to-correct transitions, accuracy-neutral wrong-label changes, relative centered-logit distances, all three tasks, both IID sets, and the full analysis set remain in `branch/`. Unchanged accuracy alone cannot establish branch nonuse. Replacement features can be outside the fitted joint distribution and do not provide an additive causal decomposition.

## Interpretation and provenance

An oracle-versus-whole advantage under the matched source-only readout supports localization/pooling as a limitation for that setup. It does not prove that whole-graph averaging erased all useful information. A successful whole-graph cross-position probe alongside a failing original-source-label fit would additionally implicate differences in supervision or learned rule selection. If both fail, the concrete decoder and fitting protocol must be considered before inferring unavailable information. Strong motif-position binding allows different rules to fit original labels equally well.

All encoders, original records, and identity partitions remain frozen. `pooling/pooling_plan.json` preserves exact supports and test identities, `pooling_source_paired.csv` preserves individual paired contrasts, and `pooling_direct_seed.csv` preserves within-seed directed-probe aggregates. Source references, execution environment, fit inventory, features, logits, predictions, and this report are covered by the new manifest. Large representations and readout checkpoints remain local. This diagnostic concerns eight known synthetic motifs and three initialization seeds; it does not test unseen motif topology.

Execution source commit: `{environment['git_head_at_start']}`.
"""
    (output / "REPORT.md").write_text(prose, encoding="utf-8")
    return {"source_panels": len(source), "source_pairs": len(paired_rows), "direct_panels": len(direct)}
