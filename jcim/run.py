"""Run and archive the controlled molecular pilot, including random controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .analysis import balanced_accuracy, fit_readout, layerwise_similarity
from .data import audit_splits, generate_dataset, split_dataset
from .model import MolecularGNN, pool_nodes, tensorize


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def take(tensors: dict, indices) -> dict:
    return {key: value[indices] for key, value in tensors.items()}


def match_training_experiences(records: list[dict], splits: dict) -> dict:
    """Keep shared identities identical and match the two differing cell counts."""
    a, b = set(splits["train_a"]), set(splits["train_b"])
    shared = a & b
    exclusive_a = sorted(a - b, key=lambda i: records[i]["id"])
    exclusive_b = sorted(b - a, key=lambda i: records[i]["id"])
    count = min(len(exclusive_a), len(exclusive_b))
    result = {"shared_training_molecules": len(shared), "exclusive_per_experience": count,
              "dropped_train_a_ids": [records[i]["id"] for i in exclusive_a[count:]],
              "dropped_train_b_ids": [records[i]["id"] for i in exclusive_b[count:]]}
    splits["train_a"] = sorted(shared | set(exclusive_a[:count]))
    splits["train_b"] = sorted(shared | set(exclusive_b[:count]))
    result["effective_train_counts"] = {name: len(splits[name]) for name in ("train_a", "train_b")}
    return result


def train_model(model, tensors, train_indices, val_indices, config, seed, output, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed + 1701)
    history = []
    start = time.monotonic()
    for epoch in range(config["epochs"]):
        model.train()
        indices = rng.permutation(train_indices)
        losses = []
        for offset in range(0, len(indices), config["batch_size"]):
            batch = take(tensors, indices[offset:offset + config["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = sum(F.cross_entropy(logits[task], batch[f"y_{task}"])
                       for task in ("family", "position", "distance"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config["epochs"]:
            validation, _ = infer(model, tensors, val_indices, config["batch_size"])
            row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   **{f"val_{task}_accuracy": float(np.mean(validation[task] ==
                       tensors[f"y_{task}"][val_indices].cpu().numpy()))
                       for task in ("family", "position", "distance")}}
            history.append(row)
            print(json.dumps({"model": name, **row}), flush=True)
    # The final epoch is fixed in advance; no OOD labels select checkpoints.
    write_json(output / f"history_{name}.json", history)
    return time.monotonic() - start


@torch.no_grad()
def infer(model, tensors, indices, batch_size=128, representations=False):
    model.eval()
    predictions = {task: [] for task in ("family", "position", "distance")}
    local, graph = None, None
    for offset in range(0, len(indices), batch_size):
        result = model(take(tensors, indices[offset:offset + batch_size]), representations)
        for task in predictions:
            predictions[task].append(result[task].argmax(-1).cpu().numpy())
        if representations:
            if local is None:
                local = [[] for _ in result["local"]]
                graph = [[] for _ in result["graph"]]
            for layer in range(len(local)):
                local[layer].append(result["local"][layer].cpu().numpy())
                graph[layer].append(result["graph"][layer].cpu().numpy())
    predictions = {key: np.concatenate(value) for key, value in predictions.items()}
    reps = None if not representations else {
        "local": [np.concatenate(layer) for layer in local],
        "graph": [np.concatenate(layer) for layer in graph]}
    return predictions, reps


def behavioral_rows(name, condition, seed, predictions, indices, split_name, records):
    metrics = {}
    for task in ("family", "position", "distance"):
        truth = np.array([records[i][f"y_{task}"] for i in indices])
        metrics[f"{task}_accuracy"] = float(np.mean(predictions[task] == truth))
        metrics[f"{task}_balanced_accuracy"] = balanced_accuracy(truth, predictions[task])
        if task == "distance":
            metrics["distance_mae"] = float(np.mean(np.abs(predictions[task] - truth)))
    return {"model": name, "condition": condition, "seed": seed, "split": split_name,
            "n": len(indices), **metrics}


def make_probes(name, condition, seed, representations, analysis_indices, records, alpha):
    positions = np.array([records[i]["y_position"] for i in analysis_indices])
    families = np.array([records[i]["y_family"] for i in analysis_indices])
    rows = []
    specifications = (
        ("cross_position_family", positions == 0, positions == 1, families, 8, "local"),
        ("cross_family_position", families >= 2, families < 2, positions, 3, "graph"),
        ("cross_family_local_position", families >= 2, families < 2, positions, 3, "local"),
    )
    for probe, source, target, labels, classes, pooling in specifications:
        for layer, features in enumerate(representations[pooling]):
            fitted = fit_readout(features[source], labels[source], features[target], labels[target],
                                 num_classes=classes, alpha=alpha)
            shuffled_labels = np.random.default_rng(8107).permutation(labels[source])
            shuffled = fit_readout(features[source], shuffled_labels, features[target], labels[target],
                                   num_classes=classes, alpha=alpha)
            rows.append({"model": name, "condition": condition, "seed": seed, "probe": probe,
                         "layer": -1 if condition == "raw" else layer,
                         "n_fit": int(source.sum()), "n_test": int(target.sum()),
                         "accuracy": fitted["accuracy"], "balanced_accuracy": fitted["balanced_accuracy"],
                         "shuffled_accuracy": shuffled["accuracy"]})
    return rows


def adaptation_rows(name, condition, seed, reps, representation_indices, splits, records, config):
    """Fit fresh distance readouts on nested few-shot sets; never fine-tune encoders."""
    lookup = {index: row for row, index in enumerate(representation_indices)}
    pool, test = splits["adaptation_pool"], splits["adaptation_test"]
    labels = np.array([records[i]["y_distance"] for i in pool])
    test_labels = np.array([records[i]["y_distance"] for i in test])
    features = reps["graph"][-1]
    test_features = features[[lookup[i] for i in test]]
    rows = []
    for repeat in range(config["adaptation_repeats"]):
        # The same support identities serve all conditions and pretraining seeds.
        permutation = np.random.default_rng(6000 + repeat).permutation(len(pool))
        for shots in config["adaptation_shots"]:
            if shots > len(pool):
                raise ValueError("Adaptation budget exceeds the untouched OOD support pool.")
            chosen = permutation[:shots]
            fit = fit_readout(features[[lookup[pool[i]] for i in chosen]], labels[chosen],
                              test_features, test_labels, num_classes=12, alpha=config["probe_alpha"])
            rows.append({"model": name, "condition": condition, "seed": seed, "shots": shots,
                         "repeat": repeat, "n_test": len(test), "accuracy": fit["accuracy"],
                         "balanced_accuracy": fit["balanced_accuracy"]})
    return rows


def archive_dataset(output, records, splits, tensors):
    with (output / "molecules.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({key: value for key, value in record.items()
                                     if not isinstance(value, np.ndarray)}) + "\n")
    write_json(output / "splits.json", splits)
    np.savez_compressed(output / "graphs.npz", **{key: value.cpu().numpy() for key, value in tensors.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/pilot_v1")
    parser.add_argument("--epochs", type=int, help="Explicit override, recorded in the run config.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed override.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "seeds", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing scientific record: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    (output / "representations").mkdir()
    torch.set_num_threads(config["cpu_threads"])
    # Disable TF32 so the recorded pilot does not depend on implicit low precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_name = config["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    write_json(output / "config.json", config)
    records = generate_dataset()
    splits = split_dataset(records, config["split_seed"])
    matching = match_training_experiences(records, splits)
    audit = {**audit_splits(records, splits), **matching}
    write_json(output / "audit.json", audit)
    tensors = tensorize(records, device)
    archive_dataset(output, records, splits, tensors)
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(),
                   "python": sys.version, "platform": platform.platform(), "device": str(device),
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("torch", "numpy", "rdkit", "matplotlib")},
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "cuda_runtime": torch.version.cuda,
                   "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                               cwd=ROOT, text=True).strip(),
                   "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                     for path in sorted((ROOT / "jcim").glob("*.py"))},
                   "determinism": "Seeded CPU/CUDA operations; bitwise cross-device reproducibility not guaranteed."}
    write_json(output / "environment.json", environment)
    print(json.dumps({"device": str(device), **audit}), flush=True)
    eval_splits = {name: list(splits[name]) for name in ("iid_test", "ood_f_meta", "ood_cl_meta")}
    eval_splits["ood_f_meta_direct_f"] = [i for i in splits["ood_f_meta"] if records[i]["target_spacer"] == 0]
    analysis_indices = splits["analysis"]
    representation_indices = analysis_indices + splits["adaptation_pool"]
    lookup = {index: row for row, index in enumerate(representation_indices)}
    common_rows = [lookup[i] for i in analysis_indices]
    behavior, probes, adaptation, predictions_rows, similarities = [], [], [], [], []
    cached = {}

    raw_local, raw_graph = pool_nodes(tensors["x"][representation_indices], tensors["x"][representation_indices],
                                      tensors["mask"][representation_indices])
    raw_reps = {"local": [raw_local.cpu().numpy()], "graph": [raw_graph.cpu().numpy()]}
    probes.extend(make_probes("raw", "raw", -1,
                  {key: [layer[common_rows] for layer in layers] for key, layers in raw_reps.items()},
                  analysis_indices, records, config["probe_alpha"]))
    adaptation.extend(adaptation_rows("raw", "raw", -1, raw_reps, representation_indices,
                                      splits, records, config))

    for seed in config["seeds"]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        initial_model = MolecularGNN(config["hidden_dim"], config["layers"]).to(device)
        initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
        environment["parameter_count"] = sum(p.numel() for p in initial_model.parameters())
        for condition in ("random", "a", "b"):
            name = f"{condition}_seed{seed}"
            model = MolecularGNN(config["hidden_dim"], config["layers"]).to(device)
            model.load_state_dict(initial_state)
            elapsed = 0.0
            if condition != "random":
                elapsed = train_model(model, tensors, splits[f"train_{condition}"], splits["val"],
                                      config, seed, output, name)
            torch.save({"state_dict": model.state_dict(), "condition": condition, "seed": seed,
                        "config": config, "training_seconds": elapsed}, output / "checkpoints" / f"{name}.pt")
            predicted, representations = infer(model, tensors, representation_indices,
                                                config["batch_size"], True)
            for split_name, indices in eval_splits.items():
                if not indices:
                    continue
                selected = [lookup[i] for i in indices]
                subset_predictions = {task: values[selected] for task, values in predicted.items()}
                row = behavioral_rows(name, condition, seed, subset_predictions, indices, split_name, records)
                behavior.append(row)
                print(json.dumps(row), flush=True)
            for row_index, index in enumerate(analysis_indices):
                predictions_rows.append({"model": name, "id": records[index]["id"],
                    **{f"true_{task}": records[index][f"y_{task}"] for task in predicted},
                    **{f"pred_{task}": int(values[row_index]) for task, values in predicted.items()}})
            common = {key: [layer[common_rows] for layer in layers]
                      for key, layers in representations.items()}
            probes.extend(make_probes(name, condition, seed, common, analysis_indices,
                                       records, config["probe_alpha"]))
            adaptation.extend(adaptation_rows(name, condition, seed, representations,
                                              representation_indices, splits, records, config))
            cached[name] = common["local"]
            np.savez_compressed(output / "representations" / f"{name}.npz",
                                ids=np.array([records[i]["id"] for i in representation_indices]),
                                **{f"{pool}_{layer}": value for pool, values in representations.items()
                                   for layer, value in enumerate(values)})
            write_csv(output / "behavior.csv", behavior)
            write_csv(output / "probes.csv", probes)
            write_csv(output / "adaptation.csv", adaptation)
            del model

    pair_specs = []
    for a, b in itertools.combinations(config["seeds"], 2):
        pair_specs.append(("random_random", f"random_seed{a}", f"random_seed{b}"))
        for condition in ("a", "b"):
            pair_specs.append(("trained_same_data", f"{condition}_seed{a}", f"{condition}_seed{b}"))
    for seed in config["seeds"]:
        pair_specs.append(("trained_different_data", f"a_seed{seed}", f"b_seed{seed}"))
        for condition in ("a", "b"):
            pair_specs.append(("random_trained", f"random_seed{seed}", f"{condition}_seed{seed}"))
    for kind, a, b in pair_specs:
        similarities.extend({"pair_kind": kind, "model_a": a, "model_b": b, **row}
                            for row in layerwise_similarity(cached[a], cached[b]))
    write_csv(output / "similarities.csv", similarities)
    write_csv(output / "predictions.csv", predictions_rows)
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", environment)
    from .report import generate_report
    generate_report(output)
    manifest = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "MANIFEST.json"}
    write_json(output / "MANIFEST.json", {"sha256": manifest})
    print(f"Completed scientific record: {output}", flush=True)


if __name__ == "__main__":
    main()
