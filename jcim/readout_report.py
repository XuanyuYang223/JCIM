"""Paired conditional effects for frozen readout protocols."""

from __future__ import annotations

from collections import defaultdict
import csv
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from .run import write_csv, write_json


FACTORS = {"scope": ("full512", "shallow_only"),
           "normalization": ("raw", "train_z"), "objective": ("joint", "family")}
CELL_KEYS = ("regime", "experience", "classifier", "scope", "normalization", "objective", "split")
BASE_KEYS = ("regime", "experience", "classifier", "split", "seed")
METRICS = ("accuracy", "balanced_accuracy")
REGIMES = ("end_to_end", "frozen_readout", "random")
LABELS = {"end_to_end": "V5 end-to-end", "frozen_readout": "Original trained", "random": "Random"}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def stats(seed_values):
    if not seed_values:
        raise ValueError("A summary requires at least one encoder seed.")
    values = {str(seed): float(value) for seed, value in sorted(seed_values.items(), key=lambda pair: str(pair[0]))}
    return {"mean": float(np.mean(list(values.values()))),
            "std": float(np.std(list(values.values()), ddof=1)) if len(values) > 1 else None,
            "n_seeds": len(values), "per_seed": values}


def aggregate(rows, keys):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(str(row[field]) for field in keys)
        seed = str(row["seed"])
        if seed in groups[key]:
            raise ValueError(f"Duplicate seed endpoint: {key}, {seed}.")
        groups[key][seed] = row
    return [{**dict(zip(keys, key)), **{
        metric: stats({seed: row[metric] for seed, row in seeds.items()}) for metric in METRICS}}
        for key, seeds in sorted(groups.items())]


def factorial_contrasts(rows):
    """Derive conditional edges and unscaled finite-difference interactions.

    Main effects average over all other factor settings. Higher-order effects
    are difference-in-differences averaged over factors outside the contrast;
    they are not coefficients from a +/-1 regression coding.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in BASE_KEYS)].append(row)
    edges, effects = [], []
    for key, members in sorted(grouped.items()):
        base = dict(zip(BASE_KEYS, key))
        names = tuple(FACTORS) if base["classifier"] == "mlp" else ("scope", "normalization")
        cells = {}
        for row in members:
            bits = tuple(FACTORS[name].index(row[name]) for name in names)
            if bits in cells:
                raise ValueError("A factorial cell is duplicated.")
            cells[bits] = row
        if set(cells) != set(product((0, 1), repeat=len(names))):
            raise ValueError("A predeclared factorial cell is missing.")
        for factor_index, factor in enumerate(names):
            other = [index for index in range(len(names)) if index != factor_index]
            for fixed in product((0, 1), repeat=len(other)):
                low, high = [0] * len(names), [0] * len(names)
                for index, bit in zip(other, fixed):
                    low[index] = high[index] = bit
                high[factor_index] = 1
                tags = {name: ("*" if index == factor_index else FACTORS[name][low[index]])
                        for index, name in enumerate(names)}
                if "objective" not in tags:
                    tags["objective"] = "ridge"
                edges.append({**base, **tags, "factor": factor,
                    "contrast": f"{FACTORS[factor][1]}_minus_{FACTORS[factor][0]}",
                    **{metric: float(cells[tuple(high)][metric]) - float(cells[tuple(low)][metric])
                       for metric in METRICS}})
        for order in range(1, len(names) + 1):
            for selected in combinations(range(len(names)), order):
                other = [index for index in range(len(names)) if index not in selected]
                output = {**base, "factors": "+".join(names[index] for index in selected), "order": order}
                for metric in METRICS:
                    totals = []
                    for fixed in product((0, 1), repeat=len(other)):
                        total = 0.
                        for active in product((0, 1), repeat=order):
                            bits = [0] * len(names)
                            for index, bit in zip(other, fixed):
                                bits[index] = bit
                            for index, bit in zip(selected, active):
                                bits[index] = bit
                            total += (-1) ** (order - sum(active)) * float(cells[tuple(bits)][metric])
                        totals.append(total)
                    output[metric] = float(np.mean(totals))
                effects.append(output)
    return edges, effects


def matched_random(rows):
    keys = ("experience", "classifier", "scope", "normalization", "objective", "split", "seed")
    random = {}
    for row in rows:
        if row["regime"] == "random":
            key = tuple(str(row[field]) for field in keys)
            if key in random:
                raise ValueError("Duplicate matched random condition.")
            random[key] = row
    paired = []
    for row in rows:
        if row["regime"] == "random":
            continue
        key = tuple(str(row[field]) for field in keys)
        if key not in random:
            raise ValueError("A trained endpoint lacks its exact-protocol random control.")
        reference = random[key]
        if str(reference["n_test"]) != str(row["n_test"]):
            raise ValueError("A matched random endpoint uses different test identities.")
        paired.append({**{field: row[field] for field in (*keys, "regime")},
            "contrast": "trained_minus_matched_random",
            **{metric: float(row[metric]) - float(reference[metric]) for metric in METRICS}})
    return paired


def one(rows, **filters):
    values = [row for row in rows if all(str(row.get(field)) == str(value) for field, value in filters.items())]
    if len(values) != 1:
        raise ValueError(f"Expected exactly one endpoint for {filters}: {len(values)}.")
    return values[0]


def text_score(value, multiplier=100):
    if value["std"] is None:
        return f"{multiplier * value['mean']:.1f} (one seed)"
    return f"{multiplier * value['mean']:.1f} +/- {multiplier * value['std']:.1f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def figures(output, cells, effect_panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    row_keys = [(norm, objective) for norm in ("raw", "train_z") for objective in ("joint", "family")]
    for column, regime in enumerate(REGIMES):
        for row_number, experience in enumerate(("a", "b")):
            axis = axes[row_number, column]
            values = np.asarray([[one(cells, regime=regime, experience=experience, classifier="mlp", split="ood_common",
                scope=scope, normalization=norm, objective=objective)["accuracy"]["mean"] * 100
                for scope in FACTORS["scope"]] for norm, objective in row_keys])
            image = axis.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
            for (y, x), value in np.ndenumerate(values):
                axis.text(x, y, f"{value:.1f}", ha="center", va="center", color="white" if value > 55 else "black")
            axis.set(xticks=[0, 1], xticklabels=["Full 512D", "Shallow only"], yticks=range(4),
                     yticklabels=[f"{norm} / {objective}" for norm, objective in row_keys],
                     title=f"{LABELS[regime]} / {experience.upper()}")
    fig.subplots_adjust(left=.15, right=.90, hspace=.45, wspace=.8, top=.86)
    fig.colorbar(image, cax=fig.add_axes([.92, .16, .015, .65]), label="Common-OOD motif accuracy (%)")
    fig.suptitle("Frozen readout factorial: identical initialization and original source labels", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"readout_factorial.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for factor_index, factor in enumerate(FACTORS):
        axis = axes[factor_index]
        for group_index, (regime, experience) in enumerate(product(REGIMES, ("a", "b"))):
            cell = one(effect_panels, regime=regime, experience=experience, classifier="mlp", split="ood_common", factors=factor, order=1)
            value = cell["accuracy"]
            axis.scatter(group_index, value["mean"] * 100, color=("#327ca4", "#b86d32", "#777777")[group_index // 2], s=30)
            if value["std"] is not None:
                axis.errorbar(group_index, value["mean"] * 100, yerr=value["std"] * 100, color="#333333", fmt="none", capsize=3)
            numeric = list(value["per_seed"].values())
            axis.scatter(group_index + np.linspace(-.07, .07, len(numeric)), np.asarray(numeric) * 100,
                         s=12, color="#444444", alpha=.6)
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set(xticks=range(6), xticklabels=["V5 A", "V5 B", "Orig A", "Orig B", "Rand A", "Rand B"],
                 title=f"{FACTORS[factor][1]} minus {FACTORS[factor][0]}")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Averaged paired effect (percentage points)")
    fig.suptitle("Main effects average the other factors; conditional effects and interactions remain archived", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for extension in ("png", "svg"):
        fig.savefig(directory / f"readout_effects.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_readout_report(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    environment = json.loads((output / "environment.json").read_text())
    rows = [row for row in read_csv(output / "panel/source_metrics.csv") if row["task"] == "family"]
    cells = aggregate(rows, CELL_KEYS)
    edges, effects = factorial_contrasts(rows)
    random_rows = matched_random(rows)
    effect_panels = aggregate(effects, (*BASE_KEYS[:-1], "factors", "order"))
    random_panels = aggregate(random_rows, CELL_KEYS)
    edge_panels = aggregate(edges, (*BASE_KEYS[:-1], "factor", "scope", "normalization", "objective", "contrast"))
    expected = {str(seed) for seed in config["seeds"]}
    for panel in (*cells, *effect_panels, *random_panels, *edge_panels):
        if any(set(panel[metric]["per_seed"]) != expected for metric in METRICS):
            raise ValueError("A factorial summary is missing an expected seed.")
    write_csv(output / "conditional_factor_edges.csv", edges)
    write_csv(output / "factorial_effects.csv", effects)
    write_csv(output / "matched_random_deltas.csv", random_rows)
    summary = {"cells": cells, "conditional_edges": edge_panels, "effects": effect_panels, "random_pairs": random_panels}
    write_json(output / "report_summary.json", summary)
    inventory = read_csv(output / "panel/fit_inventory.csv")
    budget = [{"fit_key": row["fit_key"], **{key: row[key] for key in ("regime", "experience", "seed", "scope", "normalization", "objective")},
               "optimizer_updates": row["optimizer_updates"], "clipped_steps": row["clip_count"],
               "minimum_clip_coefficient": row["minimum_clip_coefficient"]}
              for row in inventory if row["classifier"] == "mlp"]
    write_csv(output / "clipping_by_fit.csv", budget)
    figures(output, cells, effect_panels)
    columns = [(regime, experience) for regime in REGIMES for experience in ("a", "b")]
    headers = ["Scope", "Normalization", "Objective", *(f"{LABELS[regime]} {experience.upper()}" for regime, experience in columns)]
    mlp_table, ridge_table = [], []
    for scope, normalization, objective in product(FACTORS["scope"], FACTORS["normalization"], FACTORS["objective"]):
        mlp_table.append([scope, normalization, objective, *(text_score(one(cells, regime=regime, experience=experience,
            classifier="mlp", scope=scope, normalization=normalization, objective=objective, split="ood_common")["accuracy"])
            for regime, experience in columns)])
    for scope, normalization in product(FACTORS["scope"], FACTORS["normalization"]):
        ridge_table.append([scope, normalization, "ridge", *(text_score(one(cells, regime=regime, experience=experience,
            classifier="ridge", scope=scope, normalization=normalization, objective="ridge", split="ood_common")["accuracy"])
            for regime, experience in columns)])
    effects_table = []
    for regime, experience, factor in product(REGIMES, ("a", "b"), FACTORS):
        row = one(effect_panels, regime=regime, experience=experience, classifier="mlp", split="ood_common", factors=factor, order=1)
        effects_table.append([LABELS[regime], experience.upper(), f"{FACTORS[factor][1]} minus {FACTORS[factor][0]}", text_score(row["accuracy"])])
    smoke = "\n**Pipeline smoke only; excluded from primary evidence.**\n" if environment["pipeline_only"] else ""
    prose = f"""# Frozen readout factors and matched random controls
{smoke}
This diagnostic holds molecular identity splits, original source labels, encoder states, head initialization, and fitting budgets fixed. It compares readout input scope, feature standardization, and the frozen-head task objective. It adds no target-combination labels or target-fragment membership. The scope of attribution is this fixed factorial protocol, not a unique decomposition of the earlier v5-to-v6 change.

## Fixed comparison and its limits

The MLP grid crosses full512 versus shallow_only, raw versus train_z, and joint versus family. Full512 is the actual v5 readout, comprising 448 deep coordinates and 64 shallow coordinates. Shallow_only zeros its first 448 coordinates while preserving a 512-dimensional head, so input width and head parameters remain identical. Every head starts from the same archived seed-matched random first 448 weights and zero appended 64 columns. This differs from the standalone 64D MLP initializer used in v6. Train_z fits each coordinate's mean and population SD on original training identities only; raw preserves the original feature scale.

All MLP source fits use the same 1,386 identities and stored epoch orders per experience, with {config['epochs']} epochs, batches of 128, AdamW learning rate 0.0015, weight decay 0.0001, and global gradient clipping at 5. No encoder parameter is updated. Joint sums the original family, position, and distance cross entropies; family uses only family cross entropy with the same coefficient. The available source annotations are identical; the objective determines which of them are used. Auxiliary predictions from untrained family-only heads are marked as untrained and are not meaningful task comparisons.

The task heads have no shared trainable parameters when the encoder is frozen. The other task losses therefore contribute no raw family-head gradient. Their only update coupling here is the original global gradient clip, with floating-point execution as a numerical consideration. Objective differences cannot establish multitask semantic interference or shared-feature learning. Per-update total and family gradient norms, clipping coefficients, and clip counts remain archived.

Every table entry is ordinary common-OOD motif accuracy in percent, mean +/- sample SD across {len(config['seeds'])} encoder seeds. A/B and encoder regimes remain separate. The same random encoder is reused across A/B fits and does not create six independent initializations. Complete training and both IID guards, balanced scores, original task metrics, individual seeds, and all predictions remain in the record.

## Complete fixed MLP grid

{table(headers, mlp_table)}

![All frozen MLP cells](figures/readout_factorial.png)

## Separate ridge panel

Ridge uses the identical support identities and alpha 1, an unpenalized intercept, and one-hot targets. Its shallow_only solve uses the active 64 coordinates; zero padding adds no ridge information. Raw ridge still centers the design as part of intercept fitting and omits SD scaling. Standardization also changes the effective penalty in original feature units. Ridge is closed form with zero optimizer updates; its loss, parameterization, and regularization differ from the MLP, so it is a complementary decoder panel rather than a level of the MLP objective factor.

{table(headers, ridge_table)}

The standardized shallow ridge is checked against every corresponding v6 source-only whole-graph fit. The raw/full512/joint MLP with original-trained encoders must reproduce the completed v5 frozen-readout shallow-skip heads and analysis predictions exactly at 120 epochs. These are continuity anchors, not new independent replications. Exact re-extracted full and shallow feature continuity is also checked.

## Paired factors and interactions

The table below averages a factor's high-minus-low effect over the other factors within each encoder seed, before summarizing across seeds. Scope is shallow_only minus full512, standardization is train_z minus raw, and objective is family minus joint. These averages can conceal condition-dependent changes. Every conditional edge, all two-factor difference-in-differences, and the three-factor interaction are preserved separately; they must not be interpreted as additive percentages of a total historical performance change.

{table(['Encoder', 'Experience', 'Averaged contrast', 'Effect (pp)'], effects_table)}

![Averaged effects with individual seeds](figures/readout_effects.png)

`conditional_factor_edges.csv` retains changes with the other levels held fixed. `factorial_effects.csv` retains averaged main effects and finite-difference interactions, not +/-1 regression coefficients. `matched_random_deltas.csv` compares each trained encoder with its same-seed random encoder under exactly the same scope, normalization, objective, classifier, labels, and budget. These paired contrasts, including negative values, are retained without selecting a favorable cell. Three seeds provide descriptive variability, not a general significance claim.

## Preservation and interpretation scope

The earlier 100% oracle-local result remains conditional on supplied target membership; no oracle feature enters this factorial. Frozen readout comparisons do not identify effects on learned encoder representations, and this synthetic eight-motif task does not validate real chemical-property prediction. Strong source motif-position binding permits multiple rules with the same source accuracy. Readable information and a fitted generalizing decision rule remain distinct measurements.

All eight preceding records remain unchanged. The fixed protocol, identity plans, epoch orders, encoder/initializer inventories, feature continuity, fit states, clipping trajectories, final predictions, summaries, and figures are archived separately with a new manifest. Large feature arrays and model states remain local. Source commit: `{environment['git_head_at_start']}`.
"""
    (output / "REPORT.md").write_text(prose, encoding="utf-8")
    return {"cell_panels": len(cells), "conditional_edges": len(edges), "factor_effects": len(effects), "random_pairs": len(random_rows)}
