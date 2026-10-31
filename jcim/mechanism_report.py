"""Regenerate the frozen-encoder and replay report from saved measurements."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ARM_NAMES = {
    "frozen_new": "Frozen encoder / 32 new",
    "full_new": "Full update / 32 new",
    "frozen_replay": "Frozen encoder / 32 new + 32 old",
    "full_replay": "Full update / 32 new + 32 old",
    "scratch_replay": "Random initialization / 32 new + 32 old",
}
ARM_COLORS = dict(zip(ARM_NAMES, ("#277da8", "#df812f", "#369f73", "#a657a0", "#777777")))
METRICS = ("motif_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy", "distance_mae")
CONTRASTS = (
    ("frozen_replay", "frozen_new"),
    ("full_replay", "full_new"),
    ("full_replay", "frozen_replay"),
    ("full_replay", "scratch_replay"),
)


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _select(rows: list[dict], **filters: Any) -> list[dict]:
    return [row for row in rows if all(str(row.get(key)) == str(value) for key, value in filters.items())]


def _number(row: dict, metric: str) -> float:
    value = float(row[metric])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {metric} in measurement: {row!r}")
    return value


def _stats(values: dict[str, float]) -> dict:
    sequence = list(values.values())
    return {
        "mean": float(np.mean(sequence)),
        "std": float(np.std(sequence, ddof=1)) if len(sequence) > 1 else None,
        "n": len(sequence),
        "per_seed": values,
    }


def _seed_stats(rows: list[dict], metric: str) -> dict | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric, "") != "":
            grouped[str(row["seed"])].append(_number(row, metric))
    return _stats({seed: float(np.mean(values)) for seed, values in sorted(grouped.items())}) if grouped else None


def _aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...] = METRICS) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return [{**dict(zip(keys, key)), **{metric: _seed_stats(members, metric) for metric in metrics}}
            for key, members in sorted(grouped.items())]


def _format(stat: dict | None, *, points: bool = False, raw: bool = False) -> str:
    if stat is None:
        return "Unavailable"
    scale, digits = (1, 4) if raw else (100, 1)
    suffix = "" if raw else " percentage points" if points else "%"
    result = f"{stat['mean'] * scale:.{digits}f}"
    if stat["std"] is None:
        return f"{result}{suffix} (n=1; SD undefined)"
    return f"{result} +/- {stat['std'] * scale:.{digits}f}{suffix} (n={stat['n']})"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "No measurements were recorded.\n"
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows),
    ]) + "\n"


def _validate_trajectory(rows: list[dict], config: dict) -> list[int]:
    """Reject incomplete or duplicated panels before producing scientific summaries."""
    if not rows:
        raise ValueError("The trajectory contains no measurements.")
    required = ("experience", "seed", "repeat", "arm", "step", "split", "n",
                "motif_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy")
    observed: set[tuple] = set()
    for row in rows:
        if any(key not in row or row[key] == "" for key in required):
            raise ValueError(f"Missing trajectory fields: {row!r}")
        if row["experience"] not in ("a", "b") or row["arm"] not in ARM_NAMES:
            raise ValueError(f"Unknown experience or arm: {row!r}")
        key = tuple(str(row[field]) for field in ("experience", "seed", "repeat", "arm", "step", "split"))
        if key in observed:
            raise ValueError(f"Duplicate trajectory measurement: {key!r}")
        observed.add(key)
        if int(row["n"]) <= 0:
            raise ValueError("Evaluation sets must be nonempty.")
        for metric in METRICS:
            if row.get(metric, "") != "":
                value = _number(row, metric)
                if (metric.endswith("accuracy") and not 0 <= value <= 1) or value < 0:
                    raise ValueError(f"Invalid {metric}: {value}")
    steps = sorted({int(row["step"]) for row in rows})
    if not steps or steps[0] != 0:
        raise ValueError("A step-zero baseline is required for every adaptation arm.")
    configured_steps = config.get("evaluation_steps", config.get("eval_steps", config.get("checkpoints")))
    if isinstance(configured_steps, list) and steps != sorted(set(map(int, configured_steps))):
        raise ValueError("Recorded checkpoints do not match the configured evaluation steps.")
    seeds = list(map(str, config.get("seeds", sorted({row["seed"] for row in rows}))))
    repeats = config.get("support_repeats", config.get("repeats", config.get("adaptation_repeats")))
    repeat_values = (list(map(str, repeats)) if isinstance(repeats, list)
                     else list(map(str, range(int(repeats)))) if repeats is not None
                     else sorted({row["repeat"] for row in rows}))
    if {str(row["seed"]) for row in rows} != set(seeds) or {str(row["repeat"]) for row in rows} != set(repeat_values):
        raise ValueError("Recorded initialization seeds or support repetitions differ from the configuration.")
    for experience in ("a", "b"):
        for seed in seeds:
            for repeat in repeat_values:
                for arm in ARM_NAMES:
                    for step in steps:
                        for split in (f"iid_{experience}", "ood_common"):
                            key = (experience, seed, repeat, arm, str(step), split)
                            if key not in observed:
                                raise ValueError(f"Incomplete trajectory panel: missing {key!r}")
    return steps


def _paired_stat(current: list[dict], reference: list[dict], metric: str) -> dict | None:
    """Pair the support draw as well as the initialization before averaging."""
    reference_values = {(str(row["seed"]), str(row["repeat"])): _number(row, metric) for row in reference}
    differences = []
    for row in current:
        key = (str(row["seed"]), str(row["repeat"]))
        if key in reference_values:
            differences.append({"seed": row["seed"], metric: _number(row, metric) - reference_values[key]})
    if len(differences) != len(current) or len(current) != len(reference):
        raise ValueError("A paired contrast is missing a matched seed/support repeat.")
    return _seed_stats(differences, metric)


def _write_seed_metrics(path: Path, rows: list[dict]) -> None:
    keys = ("experience", "arm", "seed", "step", "split")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*keys, "support_repeats", "n_test", *METRICS])
        writer.writeheader()
        for key, members in sorted(grouped.items()):
            result = {**dict(zip(keys, key)), "support_repeats": len(members),
                      "n_test": ",".join(sorted({str(row["n"]) for row in members}))}
            for metric in METRICS:
                values = [_number(row, metric) for row in members if row.get(metric, "") != ""]
                result[metric] = float(np.mean(values)) if values else ""
            writer.writerow(result)


def _make_figure(result_dir: Path, rows: list[dict], steps: list[int], *, stem: str = "mechanism_summary", primary: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row_index, experience in enumerate(("a", "b")):
        new_axis, old_axis, tradeoff_axis = axes[row_index]
        for arm, label in ARM_NAMES.items():
            selected = _select(rows, experience=experience, arm=arm)
            new_stats = [_seed_stats(_select(selected, step=step, split="ood_common"), "motif_accuracy") for step in steps]
            old_stats = [_seed_stats(_select(selected, step=step, split=f"iid_{experience}"), "motif_accuracy") for step in steps]
            color = ARM_COLORS[arm]
            for axis, statistics in ((new_axis, new_stats), (old_axis, old_stats)):
                means = np.array([stat["mean"] for stat in statistics])
                stds = np.array([stat["std"] or 0 for stat in statistics])
                axis.plot(steps, means, color=color, label=label, marker="o", markersize=3)
                axis.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=color, alpha=0.12)
            for seed in new_stats[0]["per_seed"]:
                tradeoff_axis.plot([stat["per_seed"][seed] for stat in old_stats],
                                   [stat["per_seed"][seed] for stat in new_stats], color=color, alpha=0.16, linewidth=0.8)
            old_means, new_means = [stat["mean"] for stat in old_stats], [stat["mean"] for stat in new_stats]
            tradeoff_axis.plot(old_means, new_means, color=color, marker="o", markersize=4, label=label)
            tradeoff_axis.scatter(old_means[0], new_means[0], color=color, marker="s", s=40, zorder=4)
            tradeoff_axis.scatter(old_means[-1], new_means[-1], color=color, marker="*", s=110, zorder=5)
        for axis, label in ((new_axis, "New combinations"), (old_axis, "Old own-experience IID")):
            axis.set(title=f"Experience {experience.upper()}: {label}", xlabel="Optimizer steps", ylabel="Motif accuracy", ylim=(-0.02, 1.02))
            axis.set_xticks(steps)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.axhline(1 / 8, color="#aaaaaa", linestyle=":", linewidth=1)
        tradeoff_axis.set(title=f"Experience {experience.upper()}: adaptation trajectory", xlabel="Old own-IID motif accuracy", ylabel="New OOD motif accuracy", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        tradeoff_axis.xaxis.set_major_formatter(PercentFormatter(1))
        tradeoff_axis.yaxis.set_major_formatter(PercentFormatter(1))
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    support_note = "Primary fixed support draw" if primary else "Support draws averaged within seed"
    figure.suptitle(f"Frozen encoder and replay: means across initialization seeds\n{support_note}; bands = sample SD; square = step 0, star = final step", fontsize=13)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{stem}.{extension}", dpi=180)
    plt.close(figure)


def generate_mechanism_report(result_dir: Path) -> None:
    """Write English derived outputs without altering recorded scientific inputs."""
    result_dir = Path(result_dir)
    config = _read_json(result_dir / "config.json", {})
    environment = _read_json(result_dir / "environment.json", {})
    support = _read_json(result_dir / "support_sets.json", {})
    audit = _read_json(result_dir / "audit.json", {})
    trajectory = _read_csv(result_dir / "trajectory.csv")
    drift = _read_csv(result_dir / "drift.csv") if (result_dir / "drift.csv").exists() else []
    components = _read_csv(result_dir / "component_eval.csv") if (result_dir / "component_eval.csv").exists() else []
    ridge = _read_csv(result_dir / "ridge_diagnostic.csv") if (result_dir / "ridge_diagnostic.csv").exists() else []
    steps = _validate_trajectory(trajectory, config)
    final_step = steps[-1]
    primary_repeat = str(config.get("primary_repeat", 0))
    primary_trajectory = _select(trajectory, repeat=primary_repeat)
    if not primary_trajectory:
        raise ValueError("The configured primary support repetition was not recorded.")
    final = _select(trajectory, step=final_step)
    primary_final = _select(final, repeat=primary_repeat)
    initial = _select(trajectory, step=0)
    summary = {
        "final_step": final_step,
        "evaluation_steps": steps,
        "primary_repeat": primary_repeat,
        "seed_aggregation": "Average support repeats within each initialization seed, then mean and sample SD across seeds; support draws are not independent initialization replicates.",
        "motif_uniform_random_chance": 1 / 8,
        "trajectory": _aggregate(trajectory, ("experience", "arm", "step", "split")),
        "final": _aggregate(final, ("experience", "arm", "split")),
        "primary_final": _aggregate(primary_final, ("experience", "arm", "split")),
        "changes_from_initial": [],
        "paired_arm_contrasts": [],
        "primary_paired_arm_contrasts": [],
        "support_metadata": support,
        "audit": audit,
    }
    if components:
        summary["component_eval"] = _aggregate(components, ("experience", "arm", "composition", "split"))
        summary["primary_component_eval"] = _aggregate(_select(components, repeat=primary_repeat), ("experience", "arm", "composition", "split"))
    if ridge:
        summary["ridge_diagnostic"] = _aggregate(ridge, ("experience", "readout", "support", "split"))
        summary["primary_ridge_diagnostic"] = _aggregate(_select(ridge, repeat=primary_repeat), ("experience", "readout", "support", "split"))
    support_rows = []
    recorded_new_ids: dict[str, list[str]] = {}
    recorded_support_keys: set[tuple[str, str]] = set()
    for recorded_set in support.get("sets", []):
        new_ids, old_ids = recorded_set["new_ids"], recorded_set["old_ids"]
        support_key = (str(recorded_set["experience"]), str(recorded_set["repeat"]))
        if support_key in recorded_support_keys:
            raise ValueError("Duplicate experience/repetition in support metadata.")
        recorded_support_keys.add(support_key)
        if len(new_ids) != len(set(new_ids)) or len(old_ids) != len(set(old_ids)) or set(new_ids) & set(old_ids):
            raise ValueError("Support identities must be unique within each set and disjoint between new and old.")
        repetition = str(recorded_set["repeat"])
        if repetition in recorded_new_ids and recorded_new_ids[repetition] != new_ids:
            raise ValueError("Experiences must use the same ordered new-support identities within each repetition.")
        recorded_new_ids[repetition] = new_ids
        for key, ids in (("new_count", new_ids), ("old_count", old_ids)):
            if key in config and len(ids) != int(config[key]):
                raise ValueError(f"Support metadata does not match configured {key}.")
        support_rows.append([str(recorded_set["experience"]).upper(), recorded_set["repeat"], len(new_ids), len(old_ids), len(set(new_ids) & set(old_ids))])
    summary["support_set_counts"] = support_rows
    for experience in ("a", "b"):
        for arm in ARM_NAMES:
            for split in ("ood_common", f"iid_{experience}"):
                selected = _select(final, experience=experience, arm=arm, split=split)
                baseline = _select(initial, experience=experience, arm=arm, split=split)
                summary["changes_from_initial"].append({"experience": experience, "arm": arm, "split": split,
                    **{metric: _paired_stat(selected, baseline, metric) for metric in METRICS if all(row.get(metric, "") != "" for row in selected + baseline)}})
        for current_arm, reference_arm in CONTRASTS:
            for split in ("ood_common", f"iid_{experience}"):
                selected = _select(final, experience=experience, arm=current_arm, split=split)
                reference = _select(final, experience=experience, arm=reference_arm, split=split)
                summary["paired_arm_contrasts"].append({"experience": experience, "current_arm": current_arm,
                    "reference_arm": reference_arm, "split": split,
                    **{metric: _paired_stat(selected, reference, metric) for metric in METRICS if all(row.get(metric, "") != "" for row in selected + reference)}})
                primary_selected = _select(selected, repeat=primary_repeat)
                primary_reference = _select(reference, repeat=primary_repeat)
                summary["primary_paired_arm_contrasts"].append({"experience": experience, "current_arm": current_arm,
                    "reference_arm": reference_arm, "split": split,
                    **{metric: _paired_stat(primary_selected, primary_reference, metric) for metric in METRICS if all(row.get(metric, "") != "" for row in primary_selected + primary_reference)}})
    if drift:
        drift_metrics = tuple(metric for metric in ("encoder_max_abs_delta", "head_max_abs_delta", "graph_cka", "local_cka") if any(row.get(metric, "") != "" for row in drift))
        summary["drift"] = _aggregate(drift, ("experience", "arm", "step"), drift_metrics)
        frozen = [row for row in drift if row["arm"].startswith("frozen_")]
        summary["frozen_encoder_checks"] = {
            "recorded_rows": len(frozen),
            "changed_rows": sum(str(row.get("encoder_changed", "")).lower() in ("true", "1") for row in frozen),
            "max_recorded_abs_delta": max((_number(row, "encoder_max_abs_delta") for row in frozen if row.get("encoder_max_abs_delta", "") != ""), default=None),
            "hash_checked_rows": sum(bool(row.get("encoder_original_sha256") and row.get("encoder_current_sha256")) for row in frozen),
            "hash_matched_rows": sum(bool(row.get("encoder_original_sha256") and row.get("encoder_current_sha256"))
                                     and row["encoder_original_sha256"] == row["encoder_current_sha256"] for row in frozen),
        }
    _write_seed_metrics(result_dir / "trajectory_seed_metrics.csv", trajectory)
    _write_seed_metrics(result_dir / "final_seed_metrics.csv", final)
    (result_dir / "report_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _make_figure(result_dir, primary_trajectory, steps, primary=True)
    _make_figure(result_dir, trajectory, steps, stem="mechanism_support_robustness")

    main_rows, primary_rows, task_rows, seed_rows = [], [], [], []
    for experience in ("a", "b"):
        for arm, label in ARM_NAMES.items():
            new = _select(final, experience=experience, arm=arm, split="ood_common")
            old = _select(final, experience=experience, arm=arm, split=f"iid_{experience}")
            change = next(row for row in summary["changes_from_initial"] if row["experience"] == experience and row["arm"] == arm and row["split"] == f"iid_{experience}")
            main_rows.append([experience.upper(), label, _format(_seed_stats(new, "motif_accuracy")),
                              _format(_seed_stats(old, "motif_accuracy")), _format(change["motif_accuracy"], points=True)])
            primary_new = _select(primary_final, experience=experience, arm=arm, split="ood_common")
            primary_old = _select(primary_final, experience=experience, arm=arm, split=f"iid_{experience}")
            primary_baseline = _select(initial, experience=experience, arm=arm, split=f"iid_{experience}", repeat=primary_repeat)
            primary_rows.append([experience.upper(), label, _format(_seed_stats(primary_new, "motif_accuracy")),
                _format(_seed_stats(primary_old, "motif_accuracy")), _format(_paired_stat(primary_old, primary_baseline, "motif_accuracy"), points=True)])
            for label_split, members in (("New common OOD", new), ("Old own IID", old)):
                task_rows.append([experience.upper(), label, label_split,
                    *[_format(_seed_stats(members, metric), raw=metric == "distance_mae") for metric in METRICS]])
            new_stats, old_stats = _seed_stats(new, "motif_accuracy"), _seed_stats(old, "motif_accuracy")
            for seed in new_stats["per_seed"]:
                seed_rows.append([experience.upper(), label, seed,
                                  f"{100 * new_stats['per_seed'][seed]:.1f}%", f"{100 * old_stats['per_seed'][seed]:.1f}%"])
    contrast_rows = [[row["experience"].upper(), ARM_NAMES[row["current_arm"]] + " minus " + ARM_NAMES[row["reference_arm"]], row["split"],
                      _format(row["motif_accuracy"], points=True)] for row in summary["paired_arm_contrasts"]]
    input_names = ("config.json", "environment.json", "audit.json", "support_sets.json", "trajectory.csv", "metrics.csv", "drift.csv", "component_eval.csv", "ridge_diagnostic.csv")
    input_links = ", ".join(f"[{name}]({name})" for name in input_names if (result_dir / name).exists())
    lines = [
        "# Frozen encoder and old-example replay experiment", "",
        "This experiment asks whether updating inherited task heads on fixed learned features can resolve held-out motif-position combinations, and whether replay changes the tradeoff between adaptation and old-combination performance. It compares interventions under a fixed optimizer-step budget and reports new and old accuracy separately. It does not infer feature absence from a failed readout or establish a causal mechanism from representation similarity.", "",
        "![Frozen encoder and replay summary](figures/mechanism_summary.png)", "",
        f"The primary comparison and figure use the one fixed support set with repeat = {primary_repeat}, shared across arms and initialization seeds. Other support repetitions are supplemental robustness checks. Tables and shaded bands show means and sample standard deviations across seeds; three support draws on each of three seeds still give n=3 initialization replicates. The faint tradeoff curves show individual seeds. The square marks step 0 and the star the final checkpoint; intervening circles follow increasing optimizer steps. These curves are adaptation trajectories, not an optimized Pareto frontier.", "",
        "## Matched design and label accounting", "",
        "Every arm receives the same 32 new molecular identities for a given support repetition, reused from the archived hard-v2 32-shot support draw. Replay arms additionally receive the same fixed 32 old training identities within an experience and repetition. Old identities differ between experiences because their pretraining sets differ. Support identities are excluded from the independent new and old evaluation sets. Replaying old labels requires access to 32 previously labeled molecules: the new-only arms use 32 distinct labels per adaptation run, while replay arms use 64 distinct labels, of which 32 are reused pretraining labels. They have equal numbers of optimizer updates, rather than equal total example presentations or equal data access.", "",
        "The encoder comprises the atom-input embedding and all message-passing layers. Frozen arms hold every encoder parameter fixed and train all three inherited MLP task heads, including their hidden layers, for motif, position, and path-distance classification. Full-update arms train that encoder and those same inherited heads. The scratch arm trains the complete randomly initialized model and its random task heads on the same replay support; its archived initialization is matched to the corresponding hard-v2 random seed. No task head is reset in the pretrained arms. Thus the frozen intervention is inherited-MLP adaptation, rather than a reset final linear layer.", "",
        f"The final predeclared checkpoint is step {final_step}; recorded evaluation checkpoints are {', '.join(map(str, steps))}. Each update uses the entire support set: 32 examples in new-only arms and 64 examples in replay arms. With 32 new and 32 old examples, the ordinary full-batch loss gives equal new/old weight. All three task losses contribute to training. Fixed-step matching does not match floating-point work, because frozen encoders and different batch sizes have different costs. No test-based checkpoint selection is performed.", "",
        "The new test is `ood_common`; the old test is `iid_a` for Experience A and `iid_b` for Experience B. Both IID splits, when recorded, remain available in the machine-readable files. The old tests have different distributions across experiences; the shared OOD test is the common challenge. Final accuracies below therefore retain the experience label rather than pooling A and B.", "",
        _table(["Experience", "Support repetition", "Unique new identities", "Unique old identities", "New/old identity overlap"], support_rows), "",
        "## Final motif accuracy and retention", "",
        f"Primary comparison, using only the predeclared fixed support repetition {primary_repeat}:", "",
        _table(["Experience", "Adaptation arm", "New OOD motif accuracy", "Old own-IID motif accuracy", "Old change from own step 0"], primary_rows), "",
        "Supplemental summary, averaging all support repetitions within each seed:", "",
        _table(["Experience", "Adaptation arm", "New OOD motif accuracy", "Old own-IID motif accuracy", "Old change from own step 0"], main_rows), "",
        "![Support-draw robustness summary](figures/mechanism_support_robustness.png)", "",
        "The supplemental figure averages the support draws within each seed before showing the initialization mean and sample SD. Faint new-versus-old traces are individual seed averages over the draws.", "",
        "For pretrained arms, old-IID decrease relative to their own step-zero model measures forgetting on that distribution. The scratch arm starts without pretrained performance; its change from step zero is learning from random initialization and is not a measure of forgetting. Separate new and old accuracies do not give the fraction of individual molecules correct on both distributions.", "",
        "Matched-seed and matched-support contrasts below report current-arm minus reference-arm final motif accuracy. Support-draw differences are averaged within seed before computing the seed mean and sample SD. They are descriptive contrasts without significance claims.", "",
        _table(["Experience", "Paired contrast", "Evaluation split", "Motif accuracy difference"], contrast_rows), "",
        "## All tasks and each initialization seed", "",
        "Joint accuracy is the fraction of evaluated molecules whose motif, position, and distance predictions are all correct. It is computed per molecule before aggregation. Distance MAE is in bonds. All accuracies are exact-match fractions, and the reported SD describes initialization variation rather than a molecule-level confidence interval.", "",
        _table(["Experience", "Arm", "Evaluation", "Motif", "Position", "Distance", "All-three joint", "Distance MAE"], task_rows), "",
        "The following seed values average support repetitions; individual support draws remain in `trajectory.csv`. [Final seed-level measurements](final_seed_metrics.csv) and [all checkpoint seed-level measurements](trajectory_seed_metrics.csv) retain full numerical precision and all recorded test splits.", "",
        _table(["Experience", "Arm", "Seed", "New OOD motif", "Old own-IID motif"], seed_rows), "",
        "## Where forgetting can occur", "",
        "If old accuracy falls while the encoder is bitwise fixed, the changed task heads are sufficient to produce that behavioral loss on the fixed representation. This does not mean the representation contains no old information. If the full model changes, encoder drift and head drift can both accompany forgetting; their coexistence does not determine which change caused the loss. The crossed-state tests below assess functional compatibility conditional on the other component. Even failed swaps can reflect a changed feature coordinate system and coadaptation rather than information loss.", "",
    ]
    if drift:
        checks = summary["frozen_encoder_checks"]
        drift_rows = []
        for experience in ("a", "b"):
            for arm, label in ARM_NAMES.items():
                members = _select(drift, experience=experience, arm=arm, step=final_step)
                if members:
                    drift_rows.append([experience.upper(), label, *[_format(_seed_stats(members, metric), raw=True)
                        for metric in ("encoder_max_abs_delta", "head_max_abs_delta", "graph_cka", "local_cka")]])
        lines.extend([
            f"Recorded frozen-encoder checks: {checks['recorded_rows']} rows, {checks['changed_rows']} reporting changed encoder parameters, maximum recorded absolute encoder-parameter delta = {checks['max_recorded_abs_delta']}. Of {checks['hash_checked_rows']} rows containing original and current encoder SHA-256 hashes, {checks['hash_matched_rows']} have matching hashes. A zero parameter delta is distinct from this hash-based bitwise check; the saved hashes provide the stronger identity evidence where recorded.", "",
            "Parameter deltas compare each adaptation state with that run's own step-zero state. A maximum absolute delta is over parameter entries within a run; values below then average support repetitions within seed. CKA, when recorded, compares the same ordered held-out molecules before and after adaptation and describes representational geometry. High CKA can coexist with changed parameters or changed behavior and does not establish useful reuse.", "",
            _table(["Experience", "Arm", "Encoder maximum delta", "Task-head maximum delta", "Graph CKA", "Local CKA"], drift_rows), "",
        ])
    else:
        lines.extend(["No drift file was recorded. The behavioral comparisons alone cannot verify bitwise encoder freezing or partition full-model forgetting between encoder and task heads.", ""])
    if components:
        component_rows = []
        for experience in ("a", "b"):
            for arm in ("full_new", "full_replay"):
                for composition in ("E0_H0", "E0_H1", "E1_H0", "E1_H1"):
                    members = _select(components, experience=experience, arm=arm, composition=composition, repeat=primary_repeat)
                    if members:
                        component_rows.append([experience.upper(), ARM_NAMES[arm], composition,
                            _format(_seed_stats(_select(members, split="ood_common"), "motif_accuracy")),
                            _format(_seed_stats(_select(members, split=f"iid_{experience}"), "motif_accuracy"))])
        lines.extend([
            "### Crossed encoder and task-head states", "",
            f"The following primary-support table recombines step-zero and final components without additional fitting. `E0` and `H0` are the pretrained encoder and all three pretrained task heads; `E1` and `H1` are their adapted counterparts. It uses only support repetition {primary_repeat}. The full saved file includes the other pretrained arms, all support repetitions, all tasks, and all recorded splits; random-scratch runs are excluded from this intervention.", "",
            _table(["Experience", "Adaptation arm", "Recombined state", "New OOD motif", "Old own-IID motif"], component_rows), "",
            "Comparing `E0_H1` with `E0_H0` measures the effect of the changed task heads conditional on the original encoder. Comparing `E1_H0` with `E0_H0` measures the changed encoder conditional on the original task heads. The fully adapted `E1_H1` may behave differently from either swap because representations and task heads can coadapt. A failed crossed state does not prove that the encoder lost information: an altered feature basis can make an otherwise informative representation incompatible with the other task heads. These conditional interventions localize functional compatibility and do not give an additive causal decomposition of forgetting.", "",
        ])
    if ridge:
        ridge_rows = []
        for experience in ("a", "b"):
            for readout in sorted({row["readout"] for row in ridge}):
                for support_kind in sorted({row["support"] for row in ridge}):
                    members = _select(ridge, experience=experience, readout=readout, support=support_kind, repeat=primary_repeat)
                    if members:
                        ridge_rows.append([experience.upper(), readout, support_kind,
                            _format(_seed_stats(_select(members, split="ood_common"), "motif_accuracy")),
                            _format(_seed_stats(_select(members, split=f"iid_{experience}"), "motif_accuracy"))])
        lines.extend([
            "### Fresh frozen-feature ridge readout", "",
            f"As an optimization diagnostic, a fresh standardized linear ridge readout with predeclared regularization alpha = 1 is fit to the original frozen encoder features. This table uses the primary support repetition {primary_repeat}. Normalization and readout fitting use support data only. Local features are the marked attachment atom's final-layer embedding; graph features are the original pooled task-head input. New-only and replay fits use the same support identities as the neural interventions. This is a separate readout family, rather than additional steps of the inherited MLP or a replication of DFR.", "",
            _table(["Experience", "Frozen features", "Labeled support", "New OOD motif", "Old own-IID motif"], ridge_rows), "",
            "If this fresh readout succeeds where inherited-head adaptation fails, that difference supports an optimization or readout explanation under the tested features and regularization. A graph readout advantage does not by itself establish local information at the attachment atom. Failure of both tested readouts still cannot prove feature absence at all layers or under all readout families.", "",
        ])
    lines.extend([
        "## Diagnostic controls and limits", "",
        "There are eight motif classes, so uniform random class prediction has expected accuracy 1/8 = 12.5%. A fitted probe on random-encoder features is a different baseline: message passing can expose graph information before training, and its previously reported approximately 37.1% balanced accuracy is not a theoretical chance level. A label-shuffled probe measures a further empirical control. Probe results must be read by layer and seed; a weak probe at one layer or under one optimization setting cannot show that every layer lacks local information.", "",
        "Successful frozen-encoder adaptation with retention would support the existence of information usable by these task heads under this support and optimization protocol. Failure at the fixed budget would only show that this inherited-head procedure did not recover it. Additional steps, a reset readout, regularization, or a different probe may change the result. If full updating outperforms freezing, the observed improvement requires the tested encoder updates relative to this specific frozen-head procedure; it does not prove that no successful frozen readout exists. Scratch performance is essential when assessing whether rapid learning actually benefits from pretraining.", "",
        "Position-only prediction and a local-input positive control are complementary diagnostics. Each position can support two or three training motif classes in this partition, so the relation is not a one-to-one position-to-class mapping. With eight classes across three positions, deterministic position-only prediction has a 3/8 = 37.5% balanced-accuracy ceiling when all supported class recalls are considered. Agreement with one empirical majority-class predictor and compatibility with the full training-supported class set measure different things. A position-based classifier can reproduce a shortcut without identifying the actual computation inside the neural model. A local-input model can demonstrate an available local solution while using a different input protocol. An oracle-local subgraph with repeated motif graph types tests local solvability, rather than generalization to unseen motif topologies. Neither diagnostic alone resolves whether the original model uses local information. The deliberately strong motif-position association, supplied query locations, eight synthetic target motifs, fixed optimizer budget, and small number of initialization seeds limit generalization of these results.", "",
        "The complementary diagnostics have their own scientific record: [position shortcuts, local-input positive control, and all-layer probe controls](../mechanism_diagnostics_v3/REPORT.md). Its oracle-local input protocol and per-seed, per-layer probe tables should be read alongside this intervention rather than pooled into its neural adaptation arms.", "",
        "Work on spurious correlations motivates testing whether useful information survives shortcut-dependent prediction: [Kirichenko, Izmailov, and Wilson, ICLR 2023](https://arxiv.org/abs/2204.02937) found that retraining the last layer can improve robustness on their studied benchmarks. The present inherited-MLP molecular intervention differs from that method and is not a replication. [Qiu, Kuang, and Goel, ICML 2024](https://proceedings.mlr.press/v235/qiu24e.html) varied spurious-feature complexity and correlation strength in a controlled feature-learning study, motivating a subsequent molecular correlation-strength intervention. Neither paper establishes the outcome of this experiment.", "",
        "No correlation-strength sweep is inferred from this five-arm comparison. A subsequent fixed-size training intervention should distinguish directly exposed target combinations from a second set of combinations that remains entirely unexposed, so improvements in few-shot learning and broader combination generalization can be assessed separately.", "",
        "## Recorded configuration and reproducibility", "",
        "```json", json.dumps(config, indent=2, sort_keys=True), "```", "",
        f"Recorded device: `{environment.get('device', 'unrecorded')}`. Parameter count: {environment.get('parameter_count', 'unrecorded')}. Environment fields: {', '.join(sorted(environment)) or 'unrecorded'}.", "",
        f"Recorded inputs: {input_links}. Derived artifacts: [machine-readable summary](report_summary.json), [final seed metrics](final_seed_metrics.csv), [trajectory seed metrics](trajectory_seed_metrics.csv), primary [PNG](figures/mechanism_summary.png) and [SVG](figures/mechanism_summary.svg), supplemental [PNG](figures/mechanism_support_robustness.png) and [SVG](figures/mechanism_support_robustness.svg). Report regeneration changes only derived documentation, aggregates, and figures. It preserves recorded measurements, support identities, model weights, and the original pilot archives.", "",
    ])
    (result_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
