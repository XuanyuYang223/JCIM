"""Run the preregistered common-unseen-combination molecular stress test."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .analysis import balanced_accuracy, fit_readout, layerwise_similarity
from .hard_data import audit_hard_splits, generate_hard_dataset, split_hard_dataset
from .hard_model import HardMolecularGNN, tensorize_hard
from .model import pool_nodes
from .run import ROOT, archive_dataset, infer, sha256, take, train_model, write_csv, write_json


def metric_row(name, condition, seed, split, predictions, indices, records):
    row = {"model": name, "condition": condition, "seed": seed, "split": split, "n": len(indices)}
    for task in ("family", "position", "distance"):
        truth = np.array([records[i][f"y_{task}"] for i in indices])
        label = "motif" if task == "family" else task
        row[f"{label}_accuracy"] = float(np.mean(predictions[task] == truth))
        row[f"{label}_balanced_accuracy"] = balanced_accuracy(truth, predictions[task])
        if task == "distance":
            row["distance_mae"] = float(np.mean(np.abs(predictions[task] - truth)))
    row["joint_accuracy"] = float(np.mean(np.logical_and.reduce([
        predictions[task] == np.array([records[i][f"y_{task}"] for i in indices])
        for task in predictions])))
    return row


def make_probes(name, condition, seed, local_reps, indices, records, alpha):
    positions = np.array([records[i]["y_position"] for i in indices])
    labels = np.array([records[i]["y_family"] for i in indices])
    source, target = positions == 0, positions == 1
    rows = []
    for layer, features in enumerate(local_reps):
        fit = fit_readout(features[source], labels[source], features[target], labels[target], alpha=alpha)
        shuffled = fit_readout(features[source], np.random.default_rng(8107).permutation(labels[source]),
                               features[target], labels[target], alpha=alpha)
        rows.append({"model": name, "condition": condition, "seed": seed,
                     "probe": "cross_position_motif", "layer": -1 if condition == "raw" else layer,
                     "n_fit": int(source.sum()), "n_test": int(target.sum()),
                     "accuracy": fit["accuracy"], "balanced_accuracy": fit["balanced_accuracy"],
                     "shuffled_accuracy": shuffled["accuracy"]})
    return rows


def finetune_model(model, tensors, support, config, repeat):
    """Update the entire encoder and existing heads using only labeled support.

    Updates and minibatch orders are fixed across conditions. No IID replay,
    test labels, test-time normalization, or checkpoint selection is used.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["adaptation_learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(91000 + repeat)
    model.train()
    for _ in range(config["adaptation_steps"]):
        chosen = rng.choice(support, size=min(len(support), config["adaptation_batch_size"]), replace=False)
        batch = take(tensors, chosen)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = sum(F.cross_entropy(logits[task], batch[f"y_{task}"])
                   for task in ("family", "position", "distance"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()


def run_adaptation(name, condition, seed, state, tensors, records, splits, config, output):
    rows = []
    device = tensors["x"].device
    for repeat in range(config["adaptation_repeats"]):
        ordering = np.random.default_rng(6000 + repeat).permutation(splits["adaptation_pool"])
        for shots in config["adaptation_shots"]:
            if shots > len(ordering):
                raise ValueError("Adaptation budget exceeds the reserved support pool.")
            support = ordering[:shots].tolist()
            model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
            model.load_state_dict(state)
            finetune_model(model, tensors, support, config, repeat)
            support_path = output / "adaptation_support" / f"repeat{repeat}_shots{shots}.json"
            support_record = {"repeat": repeat, "shots": shots,
                              "molecule_ids": [records[i]["id"] for i in support]}
            if support_path.exists():
                assert json.loads(support_path.read_text(encoding="utf-8")) == support_record
            else:
                write_json(support_path, support_record)
            for split_name in ("iid_a", "iid_b", "ood_common"):
                indices = splits[split_name]
                predicted, _ = infer(model, tensors, indices, config["batch_size"])
                row = metric_row(name, condition, seed, split_name, predicted, indices, records)
                rows.append({**row, "shots": shots, "repeat": repeat})
            # Preserve adapted weights as part of the locally retained record.
            torch.save({"state_dict": model.state_dict(), "model": name, "shots": shots,
                        "repeat": repeat, "support_ids": support_record["molecule_ids"]},
                       output / "checkpoints" / f"{name}_adapt_r{repeat}_n{shots}.pt")
            del model
    return rows


def similarity_pairs(seeds):
    result = []
    for a, b in itertools.combinations(seeds, 2):
        result.append(("random_random", f"random_seed{a}", f"random_seed{b}"))
        for condition in ("a", "b"):
            result.append(("trained_same_data", f"{condition}_seed{a}", f"{condition}_seed{b}"))
    for a in seeds:
        for b in seeds:
            kind = "trained_different_data_same_seed" if a == b else "trained_different_data_different_seed"
            result.append((kind, f"a_seed{a}", f"b_seed{b}"))
        for condition in ("a", "b"):
            result.append(("random_trained", f"random_seed{a}", f"{condition}_seed{a}"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hard_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--adaptation-steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "seeds", "device", "adaptation_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "representations", "adaptation_support"):
        (output / name).mkdir()
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_name = config["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    write_json(output / "config.json", config)
    records = generate_hard_dataset()
    splits = split_hard_dataset(records, config["split_seed"])
    audit = audit_hard_splits(records, splits)
    write_json(output / "audit.json", audit)
    tensors = tensorize_hard(records, device)
    archive_dataset(output, records, splits, tensors)
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "device": str(device),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rdkit", "matplotlib")},
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "jcim").glob("*.py"))},
        "geometry_row_protocol": "Same first sorted test identities in both pooling definitions; cap config.geometry_examples.",
        "adaptation_protocol": "Full encoder and inherited heads; no replay; fixed steps; same nested support across all conditions.",
        "checkpoint_selection": "Final predeclared epoch, no OOD selection.",
        "determinism": "Seeded operations; cross-device bitwise reproducibility not guaranteed."}
    write_json(output / "environment.json", environment)
    print(json.dumps({"device": str(device), **audit}), flush=True)
    analysis_indices = splits["analysis"]
    lookup = {index: row for row, index in enumerate(analysis_indices)}
    geometry_rows = np.argsort([records[i]["id"] for i in analysis_indices])[:config["geometry_examples"]]
    write_json(output / "geometry_ids.json", [records[analysis_indices[i]]["id"] for i in geometry_rows])
    raw_local, _ = pool_nodes(tensors["x"][analysis_indices], tensors["x"][analysis_indices], tensors["mask"][analysis_indices])
    behavior, adaptation, predictions_rows, similarities = [], [], [], []
    probes = make_probes("raw", "raw", -1, [raw_local.cpu().numpy()], analysis_indices, records, config["probe_alpha"])
    cached = {}

    for seed in config["seeds"]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        initial = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
        initial_state = {key: value.detach().cpu().clone() for key, value in initial.state_dict().items()}
        environment["parameter_count"] = sum(p.numel() for p in initial.parameters())
        del initial
        for condition in ("random", "a", "b"):
            name = f"{condition}_seed{seed}"
            model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
            model.load_state_dict(initial_state)
            elapsed = 0.0
            if condition != "random":
                elapsed = train_model(model, tensors, splits[f"train_{condition}"], splits[f"val_{condition}"],
                                      config, seed, output, name)
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": state, "condition": condition, "seed": seed, "config": config,
                        "training_seconds": elapsed}, output / "checkpoints" / f"{name}.pt")
            predictions, reps = infer(model, tensors, analysis_indices, config["batch_size"], True)
            for split_name in ("iid_a", "iid_b", "ood_common"):
                indices = splits[split_name]
                selected = [lookup[i] for i in indices]
                row = metric_row(name, condition, seed, split_name,
                                {key: values[selected] for key, values in predictions.items()}, indices, records)
                behavior.append(row)
                adaptation.append({**row, "shots": 0, "repeat": -1})
                print(json.dumps(row), flush=True)
            for j, index in enumerate(analysis_indices):
                predictions_rows.append({"model": name, "id": records[index]["id"],
                    **{f"true_{task}": records[index][f"y_{task}"] for task in predictions},
                    **{f"pred_{task}": int(values[j]) for task, values in predictions.items()}})
            probes.extend(make_probes(name, condition, seed, reps["local"], analysis_indices, records, config["probe_alpha"]))
            cached[name] = {pool: [values[geometry_rows] for values in layers] for pool, layers in reps.items()}
            np.savez_compressed(output / "representations" / f"{name}.npz",
                                ids=np.array([records[i]["id"] for i in analysis_indices]),
                                **{f"{pool}_{j}": values for pool, layers in reps.items() for j, values in enumerate(layers)})
            del model
            adaptation.extend(run_adaptation(name, condition, seed, state, tensors, records, splits, config, output))
            for filename, rows in (("behavior.csv", behavior), ("adaptation.csv", adaptation), ("probes.csv", probes)):
                write_csv(output / filename, rows)
            print(json.dumps({"model": name, "adaptation_complete": True}), flush=True)

    for kind, a, b in similarity_pairs(config["seeds"]):
        for pooling in ("local", "graph"):
            layers = list(range(config["layers"] + 1)) if pooling == "local" else sorted({0, config["layers"] // 2, config["layers"]})
            compared = layerwise_similarity([cached[a][pooling][j] for j in layers], [cached[b][pooling][j] for j in layers])
            for layer, row in zip(layers, compared):
                similarities.append({"pair_kind": kind, "model_a": a, "model_b": b,
                                     "pooling": pooling, **row, "layer": layer})
    write_csv(output / "similarities.csv", similarities)
    write_csv(output / "predictions.csv", predictions_rows)
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", environment)
    from .hard_report import generate_hard_report
    generate_hard_report(output)
    write_json(output / "MANIFEST.json", {"sha256": {str(path.relative_to(output)): sha256(path)
               for path in sorted(output.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}})
    print(f"Completed scientific record: {output}", flush=True)


if __name__ == "__main__":
    main()
