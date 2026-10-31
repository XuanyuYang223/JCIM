"""Archived position baselines and an oracle local-input solvability control.

These diagnostics do not change the frozen hard-v2 experiment. The local
control removes molecular context using supervised target-fragment membership;
it tests task solvability, not automatic fragment discovery or novel motifs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .analysis import balanced_accuracy
from .hard_model import HardMolecularGNN, tensorize_hard
from .hard_verify import verify_hard
from .run import ROOT, infer, sha256, take, write_csv, write_json


N_CLASSES = 8
CHANCE = 1 / N_CLASSES
EVALUATION_SPLITS = ("iid_a", "iid_b", "ood_common")


def position_baseline(records, train_indices, evaluation_indices, saved_predictions=None):
    """Fit the empirical position-only MAP rule, breaking ties by class index.

    ``evaluation_indices`` maps split names to record indices. Optional saved
    predictions use the hard-v2 CSV schema. Compatibility with the *set* of
    labels observed at a position is different from exact MAP agreement: each
    position supports two or three motifs in the hard-v2 training design.
    """
    counts = np.zeros((3, N_CLASSES), dtype=np.int64)
    for index in train_indices:
        record = records[index]
        counts[int(record["y_position"]), int(record["y_family"])] += 1
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError("Every position needs training observations for a position-only rule.")
    labels = counts.argmax(axis=1)
    grouped = {}
    for row in saved_predictions or []:
        model_rows = grouped.setdefault(row["model"], {})
        if row["id"] in model_rows:
            raise ValueError("Duplicate model/identity prediction in the saved archive.")
        model_rows[row["id"]] = int(row["pred_family"])
    baseline_rows, agreement_rows = [], []
    for split_name, indices in evaluation_indices.items():
        positions = np.array([int(records[i]["y_position"]) for i in indices])
        truth = np.array([int(records[i]["y_family"]) for i in indices])
        prediction = labels[positions]
        baseline_rows.append({"split": split_name, "n": len(indices),
            "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": balanced_accuracy(truth, prediction),
            "true_label_training_supported_fraction": float(np.mean(counts[positions, truth] > 0)),
            "n_classes": N_CLASSES, "uniform_chance": CHANCE})
        for model, saved in sorted(grouped.items()):
            missing = [records[i]["id"] for i in indices if records[i]["id"] not in saved]
            if missing:
                raise ValueError(f"Saved predictions missing {len(missing)} identities for {model}.")
            model_prediction = np.array([saved[records[i]["id"]] for i in indices])
            if np.any((model_prediction < 0) | (model_prediction >= N_CLASSES)):
                raise ValueError("Saved motif predictions exceed the class vocabulary.")
            wrong = model_prediction != truth
            compatible = counts[positions, model_prediction] > 0
            agreement_rows.append({"model": model, "split": split_name, "n": len(indices),
                "model_accuracy": float(np.mean(~wrong)),
                "position_map_agreement": float(np.mean(model_prediction == prediction)),
                "prediction_training_supported_fraction": float(np.mean(compatible)),
                "n_errors": int(wrong.sum()),
                "error_training_supported_fraction": float(np.mean(compatible[wrong])) if wrong.any() else None})
    return {"class_counts_by_position": counts.tolist(), "map_labels_by_position": labels.tolist(),
        "supported_labels_by_position": [np.flatnonzero(row).tolist() for row in counts],
        "tie_break": "Smallest motif label among maximum empirical counts.",
        "balanced_accuracy_upper_bound": min(1.0, counts.shape[0] / N_CLASSES),
        "rows": baseline_rows, "agreements": agreement_rows}


def local_records(records, graphs=None):
    """Extract each known nine-atom target; keep only its root query marker.

    Ordinary atom features are copied exactly from the molecule, including the
    boundary atom's degree and hydrogen count. Removing the remote ring does
    not create a new capped chemical species. Channels 13, 14, 16, and 17 are
    zero; channel 15 marks one root. ``graphs`` may contain padded archive arrays.
    """
    result = []
    for row, record in enumerate(records):
        selected = np.asarray(record["target_fragment_indices"], dtype=np.int64)
        root = int(record["target_attachment_idx"])
        if len(selected) != 9 or len(set(selected.tolist())) != 9 or root not in selected:
            raise ValueError("The local positive control requires the full nine-atom target with its root.")
        original_x = np.asarray(record["x"] if graphs is None else graphs["x"][row])
        original_adj = np.asarray(record["adj"] if graphs is None else graphs["adj"][row])
        x = original_x[selected].astype(np.float32, copy=True)
        adjacency = original_adj[np.ix_(selected, selected)].astype(np.float32, copy=True)
        if x.shape != (9, 18):
            raise ValueError("Expected the unchanged 18-channel molecular feature vocabulary.")
        x[:, 13:18] = 0
        local_root = int(np.flatnonzero(selected == root)[0])
        x[local_root, 15] = 1
        # Check that the full root-connected component was supplied.
        reached, frontier, distance = {local_root}, [local_root], 0
        while frontier:
            following = {int(j) for i in frontier for j in np.flatnonzero(adjacency[i])} - reached
            if not following:
                break
            reached.update(following)
            frontier = list(following)
            distance += 1
        if len(reached) != 9 or distance > 6:
            raise ValueError("The target is not a connected nine-atom motif within root radius six.")
        result.append({key: record[key] for key in
            ("id", "y_family", "y_position", "y_distance", "position", "target_family")} | {
            "x": x, "adj": adjacency, "local_attachment_idx": local_root,
            "local_source_atom_indices": selected.tolist(), "root_radius": distance})
    return result


def _local_signature(record):
    return hashlib.sha256(record["x"].tobytes() + record["adj"].tobytes()).hexdigest()


def local_input_audit(records, splits):
    """Verify the intended replicated-local-type control and molecular isolation."""
    counts = {name: Counter(_local_signature(records[i]) for i in splits[name])
              for name in ("train_a", "train_b", *EVALUATION_SPLITS)}
    if counts["train_a"] != counts["train_b"]:
        raise ValueError("A and B do not have identical local-input empirical distributions.")
    train_ids = {records[i]["id"] for i in splits["train_a"]}
    for split_name in EVALUATION_SPLITS:
        if train_ids & {records[i]["id"] for i in splits[split_name]}:
            raise ValueError("Molecular training and test identities overlap.")
    return {"local_input_train_a_b_distributions_identical": True,
        "unique_local_input_types": {key: len(value) for key, value in counts.items()},
        "local_type_overlap_train_a_test": {name: len(set(counts["train_a"]) & set(counts[name]))
                                            for name in EVALUATION_SPLITS},
        "n_target_atoms": 9, "maximum_root_radius": max(r["root_radius"] for r in records),
        "molecular_train_test_identity_overlap": False,
        "ordinary_features_copied_from_full_molecule": True,
        "oracle_target_membership_used": True,
        "claim": "Local task solvability across removed contexts; not novel local-type generalization."}


def train_local_control(model, tensors, train_indices, validation_indices, config, seed):
    """Train only the motif task with a fixed final epoch and no OOD selection."""
    for name, head in model.heads.items():
        if name != "family":
            for parameter in head.parameters():
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(1701 + seed)
    history = []
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        model.train()
        ordering = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(ordering), config["batch_size"]):
            batch = take(tensors, ordering[start:start + config["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch)["family"], batch["y_family"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config["epochs"]:
            predicted, _ = infer(model, tensors, validation_indices, config["batch_size"])
            truth = tensors["y_family"][validation_indices].cpu().numpy()
            row = {"seed": seed, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   "val_a_motif_accuracy": float(np.mean(predicted["family"] == truth))}
            history.append(row)
            print(json.dumps({"diagnostic": "local_input_control", **row}), flush=True)
    return history, time.monotonic() - started


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _probe_rows(source):
    rows = []
    for original in _read_csv(source / "probes.csv"):
        row = dict(original)
        for key in ("seed", "layer", "n_fit", "n_test"):
            row[key] = int(row[key])
        for key in ("accuracy", "balanced_accuracy", "shuffled_accuracy"):
            row[key] = float(row[key])
        row.update(n_classes=N_CLASSES, uniform_chance=CHANCE,
            shuffled_label_seed=8107, origin="Frozen hard-v2 probes.csv; unchanged values.")
        rows.append(row)
    return rows


def _evaluate_local(model, tensors, records, splits, seed, batch_size):
    rows, saved = [], []
    for name in EVALUATION_SPLITS:
        indices = splits[name]
        prediction, _ = infer(model, tensors, indices, batch_size)
        truth = np.array([records[i]["y_family"] for i in indices])
        positions = np.array([records[i]["y_position"] for i in indices])
        for position in (-1, 0, 1, 2):
            selected = np.ones(len(indices), dtype=bool) if position == -1 else positions == position
            rows.append({"model": f"local_a_seed{seed}", "seed": seed, "train_curriculum": "a",
                "split": name, "position": position, "n": int(selected.sum()),
                "motif_accuracy": float(np.mean(prediction["family"][selected] == truth[selected])),
                "motif_balanced_accuracy": balanced_accuracy(truth[selected], prediction["family"][selected]),
                "n_classes": N_CLASSES, "uniform_chance": CHANCE})
        saved.extend({"model": f"local_a_seed{seed}", "seed": seed, "split": name,
            "id": records[index]["id"], "true_family": int(truth[j]),
            "pred_family": int(prediction["family"][j])} for j, index in enumerate(indices))
    return rows, saved


def _write_report(output, config, baseline_rows, agreements, local_rows, probes, audit):
    pct = lambda value: f"{100 * value:.1f}%"
    lines = ["# Mechanism diagnostics v3", "",
        "This record adds diagnostics to the frozen hard-v2 experiment. It does not retrain its full-input models.", "",
        "## Position-only baseline", "",
        "The baseline fits empirical motif counts conditional on the true queried position in each actual training curriculum. "
        "It predicts the largest count and breaks ties by smallest label. A position supports multiple motifs, so "
        "position alone cannot identify every IID motif. With three positions and eight classes, its balanced accuracy "
        "cannot exceed 37.5%. This is distinct from the eight-class uniform chance of 12.5%.", "",
        "| Curriculum | Split | n | MAP accuracy | MAP balanced accuracy | True label in training-supported set |",
        "|---|---|---:|---:|---:|---:|"]
    lines.extend(f"| {r['curriculum']} | {r['split']} | {r['n']} | {pct(r['accuracy'])} | "
                 f"{pct(r['balanced_accuracy'])} | {pct(r['true_label_training_supported_fraction'])} |" for r in baseline_rows)
    lines.extend(["", "### Saved model predictions against each position baseline", "",
        "The supported set contains every motif seen at the given position in that curriculum. "
        "High compatibility, especially among errors, is consistent with dependence on the learned binding. "
        "It does not prove that a pure position-only predictor explains the model: exact MAP agreement is shown separately.", "",
        "| Curriculum | Model | Split | Model accuracy | MAP agreement | Prediction in supported set | Errors in supported set |",
        "|---|---|---|---:|---:|---:|---:|"])
    for r in agreements:
        error = "No errors" if r["error_training_supported_fraction"] is None else pct(r["error_training_supported_fraction"])
        lines.append(f"| {r['curriculum']} | {r['model']} | {r['split']} | {pct(r['model_accuracy'])} | "
                     f"{pct(r['position_map_agreement'])} | {pct(r['prediction_training_supported_fraction'])} | {error} |")
    lines.extend(["", "## Oracle local-input positive control", "",
        "Each input is the complete known nine-atom target induced subgraph, at root radius at most six. "
        "The remote ring, anchor, position, and path-endpoint query markers are removed. Only the target root marker "
        "(channel 15) remains. Ordinary atom features are copied from the original molecule, including boundary "
        "degree and hydrogen count; no context-dependent capping operation is introduced.", "",
        f"The control uses HardMolecularGNN(64, 8), its existing graph-pooling motif head, and only motif cross entropy. "
        f"The learning rate is {config['learning_rate']}, batch size {config['batch_size']}, and duration "
        f"{config['epochs']} epochs. The final epoch is fixed before evaluation; OOD results select no checkpoints. "
        "Position and distance heads receive no training. All actual curriculum-A molecular training IDs are used. "
        "Curriculum A and B have identical empirical local-input distributions, so training A only per seed is sufficient "
        "for this particular control.", "",
        "Molecular train/test identities remain disjoint, but all eight local input graph types intentionally occur on "
        "both sides after context removal. This measures whether a usable local solution exists when an oracle supplies "
        "the target. It does not measure generalization to novel motif types, automatic localization, or reuse by the "
        "original full-input encoder.", "",
        "| Seed | Test split | n | Motif accuracy | Balanced accuracy |",
        "|---:|---|---:|---:|---:|"])
    lines.extend(f"| {r['seed']} | {r['split']} | {r['n']} | {pct(r['motif_accuracy'])} | {pct(r['motif_balanced_accuracy'])} |"
                 for r in local_rows if r["position"] == -1)
    lines.extend(["", "Per-position results and molecule-level predictions are retained in local_behavior.csv and "
        "local_predictions.csv. local_input_audit.json records the repeated local types and distribution match.", "",
        "## Frozen cross-position probe controls", "",
        "The following are copied without numerical changes from hard-v2, including every model seed and every layer. "
        "The uniform eight-class chance is 12.5%; a random encoder's observed accuracy is an empirical control, "
        "not theoretical chance. Shuffled labels use the archived deterministic permutation seed 8107. "
        "The archived shuffled metric is ordinary accuracy; it is not relabeled as balanced accuracy. A low linear "
        "probe score constrains this readout and sample protocol, not every possible extraction of local information.", "",
        "| Model | Seed | Layer | Fit n | Test n | Probe accuracy | Balanced accuracy | Shuffled-label accuracy | Uniform chance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    lines.extend(f"| {r['model']} | {r['seed']} | {r['layer']} | {r['n_fit']} | {r['n_test']} | "
                 f"{pct(r['accuracy'])} | {pct(r['balanced_accuracy'])} | {pct(r['shuffled_accuracy'])} | {pct(CHANCE)} |" for r in probes)
    lines.extend(["", "## Interpretation and archival scope", "",
        "Successful local controls would establish task solvability with supplied local membership. The position baseline "
        "and supported-set error analysis can diagnose dependence on training associations. Neither alone distinguishes "
        "information absent from the original encoder from information obscured by its readout; the separately controlled "
        "frozen-encoder and replay experiment addresses that question.", "",
        "The source hard-v2 manifest and all its listed file hashes are verified before and after this run. "
        "The new record contains its own fixed configuration, source hashes, checkpoints, training histories, "
        "local input arrays, per-seed results, copied probe controls, and an output manifest.", "",
        f"Local graph types in curriculum A: {audit['unique_local_input_types']['train_a']}. "
        f"Maximum local root radius: {audit['maximum_root_radius']}.", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _figure(output, baseline_rows, local_rows, probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for curriculum, color in (("a", "#365d9d"), ("b", "#d68a35")):
        values = [next(r["accuracy"] for r in baseline_rows if r["curriculum"] == curriculum and r["split"] == name)
                  for name in EVALUATION_SPLITS]
        axes[0].plot(EVALUATION_SPLITS, values, "o-", label=f"Train {curriculum.upper()}", color=color)
    axes[0].set_title("Position-only MAP baseline")
    for seed in sorted({r["seed"] for r in local_rows}):
        rows = [r for r in local_rows if r["seed"] == seed and r["position"] == -1]
        axes[1].plot([r["split"] for r in rows], [r["motif_accuracy"] for r in rows], "o-", label=f"Seed {seed}")
    axes[1].set_title("Oracle local-input control")
    for condition, color in (("random", "#777777"), ("a", "#365d9d"), ("b", "#d68a35")):
        for seed in sorted({r["seed"] for r in probes if r["condition"] == condition}):
            rows = [r for r in probes if r["condition"] == condition and r["seed"] == seed]
            axes[2].plot([r["layer"] for r in rows], [r["balanced_accuracy"] for r in rows],
                         color=color, alpha=0.65, label=condition if seed == 0 else None)
    axes[2].set_title("Frozen cross-position probes")
    axes[2].set_xlabel("Message layer")
    for ax in axes:
        ax.axhline(CHANCE, color="black", linestyle=":", linewidth=1, label="12.5% chance")
        ax.set_ylim(-0.02, 1.04)
        ax.set_ylabel("Accuracy" if ax is not axes[2] else "Balanced accuracy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    (output / "figures").mkdir()
    fig.savefig(output / "figures" / "diagnostic_summary.png", dpi=170)
    plt.close(fig)


def run_diagnostics(source: Path, output: Path, device="auto", seeds=(0, 1, 2), epochs=120):
    """Create a new immutable diagnostic record; never overwrite a prior output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Diagnostic outputs must be outside the frozen source record.")
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use positive epochs and a nonempty set of distinct seeds.")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    source_check_before = verify_hard(source)
    source_manifest_hash = sha256(source / "MANIFEST.json")
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    archived_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    with np.load(source / "graphs.npz") as graphs:
        local = local_records(records, graphs)
    audit = local_input_audit(local, splits)
    config = {"experiment": "mechanism_diagnostics_v3", "source_record": str(source),
        "seeds": list(seeds), "epochs": int(epochs), "hidden_dim": 64, "layers": 8,
        "learning_rate": 0.0015, "weight_decay": 0.0001, "batch_size": 128,
        "cpu_threads": archived_config.get("cpu_threads", 4), "device": device,
        "local_training_curriculum": "a", "trained_tasks": ["family"],
        "checkpoint_selection": "Final predeclared epoch; no OOD selection.",
        "uniform_chance": CHANCE, "n_classes": N_CLASSES}
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    write_json(output / "config.json", config)
    write_json(output / "local_input_audit.json", audit)
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "source_manifest_sha256": source_manifest_hash,
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "jcim").glob("*.py"))}}
    write_json(output / "environment.json", environment)
    saved_predictions = _read_csv(source / "predictions.csv")
    baselines, baseline_rows, agreements = {}, [], []
    for curriculum in ("a", "b"):
        result = position_baseline(records, splits[f"train_{curriculum}"],
            {name: splits[name] for name in EVALUATION_SPLITS}, saved_predictions)
        baselines[curriculum] = result
        baseline_rows.extend(dict(curriculum=curriculum, **row) for row in result["rows"])
        agreements.extend(dict(curriculum=curriculum, **row) for row in result["agreements"])
    write_json(output / "position_baselines.json", baselines)
    write_csv(output / "position_baselines.csv", baseline_rows)
    write_csv(output / "position_model_agreement.csv", agreements)
    probes = _probe_rows(source)
    write_csv(output / "frozen_probe_controls.csv", probes)
    shutil.copyfile(source / "probes.csv", output / "source_probes_unchanged.csv")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    actual_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                else "cpu" if device == "auto" else device)
    environment["actual_device"] = str(actual_device)
    tensors = tensorize_hard(local, actual_device)
    np.savez_compressed(output / "local_graphs.npz", ids=np.array([r["id"] for r in local]),
                        **{key: value.cpu().numpy() for key, value in tensors.items()})
    write_json(output / "local_input_indices.json", [{"id": r["id"],
        "source_atom_indices": r["local_source_atom_indices"], "root_index": r["local_attachment_idx"]} for r in local])
    local_rows, prediction_rows = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        if actual_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(actual_device)
        history, elapsed = train_local_control(model, tensors, splits["train_a"], splits["val_a"], config, seed)
        write_json(output / f"history_local_a_seed{seed}.json", history)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": config, "seed": seed, "training_seconds": elapsed,
            "train_ids": [records[i]["id"] for i in splits["train_a"]]},
            output / "checkpoints" / f"local_a_seed{seed}.pt")
        rows, predictions = _evaluate_local(model, tensors, local, splits, seed, config["batch_size"])
        local_rows.extend(rows)
        prediction_rows.extend(predictions)
        write_csv(output / "local_behavior.csv", local_rows)
        write_csv(output / "local_predictions.csv", prediction_rows)
        del model
    _write_report(output, config, baseline_rows, agreements, local_rows, probes, audit)
    _figure(output, baseline_rows, local_rows, probes)
    source_check_after = verify_hard(source)
    if sha256(source / "MANIFEST.json") != source_manifest_hash:
        raise ValueError("The frozen source manifest changed during diagnostics.")
    write_json(output / "source_integrity.json", {"before": source_check_before, "after": source_check_after,
        "manifest_sha256_before": source_manifest_hash, "manifest_sha256_after": sha256(source / "MANIFEST.json"),
        "source_record_unchanged": True})
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", environment)
    summary = {"position_baselines": baseline_rows,
        "local_final": [r for r in local_rows if r["position"] == -1],
        "source_record_unchanged": True, "probe_rows_preserved": len(probes), "local_input_audit": audit}
    write_json(output / "summary.json", summary)
    write_json(output / "MANIFEST.json", {"sha256": {str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}})
    print(json.dumps({"completed_diagnostics": str(output), **summary}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/mechanism_diagnostics_v3")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    run_diagnostics(args.source, args.output, args.device, tuple(args.seeds), args.epochs)


if __name__ == "__main__":
    main()
