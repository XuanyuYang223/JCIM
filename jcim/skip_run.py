"""Train boundary-free shallow paths using only original molecular labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.nn import functional as F

from .hard_model import HardMolecularGNN
from .hard_run import metric_row
from .mechanism import checkpoint_state, encoder_state_sha256
from .run import ROOT, sha256, take, write_csv, write_json
from .scale_run import state_sha256, verify_record
from .skip_model import SkipMolecularGNN, VARIANTS


def validate_config(config):
    if config["hidden_dim"] != 64 or config["layers"] != 8 or config["shallow_layer"] != 1:
        raise ValueError("The archived encoder and declared shallow depth are fixed.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or not set(config["seeds"]) <= {0, 1, 2}:
        raise ValueError("Choose distinct archived seeds.")
    if config["experiences"] != ["a", "b"] or config["variants"] != list(VARIANTS):
        raise ValueError("All original experiences and three readout arms are required.")
    if config["regimes"] != ["end_to_end", "frozen_readout"] or config["epochs"] < 1:
        raise ValueError("Both regimes and a positive fixed training budget are required.")
    if config["reuse_original_deep_baseline"] and config["epochs"] != 120:
        raise ValueError("Reused source models require their original 120-epoch budget.")
    expected = {"batch_size": 128, "learning_rate": 0.0015, "weight_decay": 0.0001,
                "gradient_clip": 5.0, "order_seed_offset": 1701}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("The original training hyperparameters and order must remain fixed.")


def identity_entry(records, indices):
    return {"indices": list(indices), "ids": [records[index]["id"] for index in indices]}


def build_plan(records, splits, config):
    plan = {"experiences": {}, "tests": {}, "branch_permutations": {},
            "protocol": "Original source labels only; original query markers; no target membership inputs.",
            "training_order": "default_rng(seed + order_seed_offset).permutation(original_train_indices), once per epoch"}
    common = set(splits["ood_common"])
    for experience in config["experiences"]:
        train, validation = splits[f"train_{experience}"], splits[f"val_{experience}"]
        if len(train) != 1386 or set(train) & set(splits["analysis"]) or set(validation) & set(splits["analysis"]):
            raise ValueError("Original train, validation, and test identities are not isolated.")
        if any((records[index]["y_family"] + records[index]["y_position"]) % 3 == 2
               for index in train + validation):
            raise ValueError("A common-OOD combination entered training or validation.")
        if set(train + validation) & set(splits["adaptation_pool"]) or common & set(train + validation):
            raise ValueError("Reserved new labels or test identities entered training.")
        plan["experiences"][experience] = {
            "train": identity_entry(records, train), "validation": identity_entry(records, validation),
            "initial_probe": identity_entry(records, train[:config["batch_size"]])}
    for split in ("iid_a", "iid_b", "ood_common", "analysis"):
        plan["tests"][split] = identity_entry(records, splits[split])
    for seed in config["seeds"]:
        sampling_seed = config["branch_permutation_seed_offset"] + seed
        plan["branch_permutations"][str(seed)] = {"seed": sampling_seed,
            "indices": np.random.default_rng(sampling_seed).permutation(len(splits["analysis"])).tolist()}
    return plan


def training_orders(splits, config):
    result = {}
    for seed in config["seeds"]:
        for experience in config["experiences"]:
            rng = np.random.default_rng(seed + config["order_seed_offset"])
            result[f"{experience}_seed{seed}"] = np.asarray(
                [rng.permutation(splits[f"train_{experience}"]) for _ in range(config["epochs"])], dtype=np.int32)
    return result


def initial_base(source, experience, seed, regime, device):
    encoder_source = f"{experience}_seed{seed}" if regime == "frozen_readout" else f"random_seed{seed}"
    random = checkpoint_state(source, "random", seed)
    encoder = checkpoint_state(source, experience if regime == "frozen_readout" else "random", seed)
    state = {name: (random[name] if name.startswith("heads.") else value).clone()
             for name, value in encoder.items()}
    model = HardMolecularGNN(64, 8).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, encoder_source, f"random_seed{seed}"


@torch.no_grad()
def extract_readouts(model, tensors, indices, batch_size):
    """Use fixed inference shapes; padded slots repeat valid independent graphs."""
    model.eval()
    values = []
    for offset in range(0, len(indices), batch_size):
        chosen = np.asarray(indices[offset:offset + batch_size], dtype=np.int64)
        genuine = len(chosen)
        chosen = np.resize(chosen, batch_size)
        result = model(take(tensors, chosen), representations=True)
        values.append(result["readout"][:genuine].detach())
    return torch.cat(values, dim=0)


def readout_from_cache(cache, variant):
    deep = cache[:, :448]
    if variant == "deep":
        return deep
    branch = deep[:, :64] if variant == "deep_control" else cache[:, 448:]
    return torch.cat((deep, branch), dim=1)


@torch.no_grad()
def predict_heads(model, readouts, batch_size):
    model.eval()
    predicted = {task: [] for task in ("family", "position", "distance")}
    for offset in range(0, len(readouts), batch_size):
        genuine = min(batch_size, len(readouts) - offset)
        batch = readouts[offset:offset + batch_size]
        if genuine != batch_size:
            positions = torch.arange(batch_size, device=batch.device) % genuine
            batch = batch[positions]
        for task, head in model.heads.items():
            predicted[task].append(head(batch)[:genuine].argmax(-1).cpu().numpy())
    return {task: np.concatenate(rows) for task, rows in predicted.items()}


def cached_training(model, tensors, feature_cache, cache_indices, orders, validation, config, history_path, case):
    lookup = {index: row for row, index in enumerate(cache_indices)}
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, updates = [], 0
    start = time.monotonic()
    for epoch, order in enumerate(orders):
        model.train()
        losses = []
        for offset in range(0, len(order), config["batch_size"]):
            chosen = order[offset:offset + config["batch_size"]].tolist()
            readouts = feature_cache[[lookup[index] for index in chosen]]
            optimizer.zero_grad(set_to_none=True)
            logits = {task: head(readouts) for task, head in model.heads.items()}
            loss = sum(F.cross_entropy(logits[task], tensors[f"y_{task}"][chosen])
                       for task in ("family", "position", "distance"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"])
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == len(orders):
            prediction = predict_heads(model, feature_cache[[lookup[index] for index in validation]], config["batch_size"])
            row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   **{f"val_{task}_accuracy": float(np.mean(prediction[task] ==
                       tensors[f"y_{task}"][validation].cpu().numpy())) for task in prediction}}
            history.append(row)
            print(json.dumps({"model": case, **row}), flush=True)
    write_json(history_path, history)
    return updates, time.monotonic() - start


def joint_training(model, tensors, orders, validation, config, history_path, case):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, updates = [], 0
    start = time.monotonic()
    for epoch, order in enumerate(orders):
        model.train()
        losses = []
        for offset in range(0, len(order), config["batch_size"]):
            chosen = order[offset:offset + config["batch_size"]]
            batch = take(tensors, chosen)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = sum(F.cross_entropy(logits[task], batch[f"y_{task}"]) for task in ("family", "position", "distance"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == len(orders):
            readouts = extract_readouts(model, tensors, validation, config["batch_size"])
            prediction = predict_heads(model, readouts, config["batch_size"])
            row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   **{f"val_{task}_accuracy": float(np.mean(prediction[task] ==
                       tensors[f"y_{task}"][validation].cpu().numpy())) for task in prediction}}
            history.append(row)
            print(json.dumps({"model": case, **row}), flush=True)
    write_json(history_path, history)
    return updates, time.monotonic() - start


def initial_check(model, reference, tensors, probe, metadata):
    model.eval()
    reference.eval()
    with torch.no_grad():
        actual, expected = model(take(tensors, probe)), reference(take(tensors, probe))
    maximum = max(float((actual[task] - expected[task]).abs().max()) for task in expected)
    identical_classes = all(torch.equal(actual[task].argmax(-1), expected[task].argmax(-1)) for task in expected)
    extra_zero = all(not torch.any(head[0].weight[:, 448:]).item() for head in model.heads.values())
    if maximum > 1e-5 or not identical_classes or not extra_zero:
        raise AssertionError("The initial prediction or zero-extension control failed.")
    return {**metadata, "n_probe": len(probe), "max_abs_logit_difference": maximum,
            "initial_predictions_identical": identical_classes, "extra_weights_zero": extra_zero}


def endpoint_rows(model, readouts, records, splits, seed, config, metadata, permutation):
    analysis_lookup = {index: row for row, index in enumerate(splits["analysis"])}
    behavior, prediction_rows = [], []
    interventions = ("intact",) if metadata["variant"] == "deep" else ("intact", "zero_branch", "permuted_branch")
    for intervention in interventions:
        values = readouts
        if intervention != "intact":
            branch = torch.zeros_like(readouts[:, 448:]) if intervention == "zero_branch" else readouts[permutation, 448:]
            values = torch.cat((readouts[:, :448], branch), dim=1)
        predicted = predict_heads(model, values, config["batch_size"])
        for split in ("iid_a", "iid_b", "ood_common"):
            indices = splits[split]
            selected = [analysis_lookup[index] for index in indices]
            row = metric_row("skip_v5", metadata["experience"], seed, split,
                             {task: array[selected] for task, array in predicted.items()}, indices, records)
            row = {key: value for key, value in row.items() if key not in ("model", "condition", "seed")}
            behavior.append({**metadata, "intervention": intervention, **row})
        for position, index in enumerate(splits["analysis"]):
            prediction_rows.append({**metadata, "intervention": intervention, "index": index,
                                    "id": records[index]["id"],
                                    **{f"pred_{task}": int(array[position]) for task, array in predicted.items()}})
    return behavior, prediction_rows


def run_skip(source, output, config):
    validate_config(config)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    prior_names = ("pilot_v1", "hard_v2", "readout_replay_v3", "mechanism_diagnostics_v3", "correlation_v3", "scale_v4")
    previous = {name: source.parent / name for name in prior_names if (source.parent / name).is_dir()}
    references = {name: verify_record(directory) for name, directory in previous.items()}
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    plan = build_plan(records, splits, config)
    orders = training_orders(splits, config)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("checkpoints", "representations", "histories"):
        (output / directory).mkdir()
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    write_json(output / "source_reference.json", {"records": references, "source_directory": str(source)})
    np.savez_compressed(output / "training_orders.npz", **orders)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    with np.load(source / "graphs.npz") as arrays:
        tensors = {name: torch.as_tensor(arrays[name].copy(), device=device) for name in arrays.files}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None, "python": sys.version,
        "platform": platform.platform(), "git_head_at_start": head,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rdkit", "matplotlib")},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((ROOT / "jcim").glob("*.py"))},
        "source_checkpoint_sha256": {f"{condition}_seed{seed}": sha256(source / "checkpoints" / f"{condition}_seed{seed}.pt")
                                    for seed in config["seeds"] for condition in ("a", "b", "random")},
        "target_labels_added": 0, "target_membership_supplied": False,
        "head_initialization": "Exact archived random first448 columns, biases, and output layers; appended64 columns zero.",
        "training": "Original train identities and epoch orders only; final fixed epoch; source-familiar validation.",
        "inference_padding": "Final inference batch repeats valid independent graphs to configured size; repeated outputs discarded.",
        "pipeline_only": config["epochs"] != 120 or config["seeds"] != [0, 1, 2]}
    write_json(output / "environment.json", environment)
    inventory, initial_checks, behavior, predictions = [], [], [], []
    cache_by_encoder = {}
    for seed in config["seeds"]:
        for experience in config["experiences"]:
            train, validation = splits[f"train_{experience}"], splits[f"val_{experience}"]
            for regime in config["regimes"]:
                base, encoder_source, head_source = initial_base(source, experience, seed, regime, device)
                reference = SkipMolecularGNN.from_base(base, "deep")
                if regime == "frozen_readout":
                    key = (experience, seed)
                    if key not in cache_by_encoder:
                        cache_indices = sorted(set(train + validation + splits["analysis"]))
                        extractor = SkipMolecularGNN.from_base(base, "shallow_skip")
                        extractor.requires_grad_(False)
                        encoded = extract_readouts(extractor, tensors, cache_indices, config["batch_size"]).detach()
                        np.savez_compressed(output / "representations" / f"frozen_{experience}_seed{seed}.npz",
                            indices=np.asarray(cache_indices), ids=np.asarray([records[index]["id"] for index in cache_indices]),
                            deep_graph=encoded[:, :448].cpu().numpy(), shallow_mean=encoded[:, 448:].cpu().numpy())
                        cache_by_encoder[key] = (cache_indices, encoded)
                        del extractor
                    cache_indices, cache = cache_by_encoder[key]
                for variant in config["variants"]:
                    metadata = {"regime": regime, "variant": variant, "experience": experience, "seed": seed}
                    case = f"{regime}_{variant}_{experience}_seed{seed}"
                    model = SkipMolecularGNN.from_base(base, variant)
                    for name, parameter in model.named_parameters():
                        parameter.requires_grad_(regime != "frozen_readout" or name.startswith("heads."))
                    initial = state_sha256(model.state_dict())
                    encoder_initial = encoder_state_sha256(model.state_dict())
                    initial_checks.append(initial_check(model, reference, tensors, train[:config["batch_size"]], metadata))
                    reused = config["reuse_original_deep_baseline"] and regime == "end_to_end" and variant == "deep"
                    history_path = output / "histories" / f"history_{case}.json"
                    expected_updates = config["epochs"] * math.ceil(len(train) / config["batch_size"])
                    if reused:
                        model.load_state_dict(checkpoint_state(source, experience, seed))
                        shutil.copyfile(source / f"history_{experience}_seed{seed}.json", history_path)
                        updates, elapsed = 0, 0.0
                    elif regime == "frozen_readout":
                        features = readout_from_cache(cache, variant)
                        updates, elapsed = cached_training(model, tensors, features, cache_indices,
                            orders[f"{experience}_seed{seed}"], validation, config, history_path, case)
                    else:
                        updates, elapsed = joint_training(model, tensors, orders[f"{experience}_seed{seed}"],
                            validation, config, history_path, case)
                    if not reused and updates != expected_updates:
                        raise AssertionError("An arm received a different optimizer-step budget.")
                    final = state_sha256(model.state_dict())
                    encoder_final = encoder_state_sha256(model.state_dict())
                    if regime == "frozen_readout" and encoder_initial != encoder_final:
                        raise AssertionError("A frozen original encoder changed.")
                    if regime == "frozen_readout":
                        lookup = {index: row for row, index in enumerate(cache_indices)}
                        readouts = readout_from_cache(cache, variant)[[lookup[index] for index in splits["analysis"]]]
                    else:
                        readouts = extract_readouts(model, tensors, splits["analysis"], config["batch_size"])
                    np.savez_compressed(output / "representations" / f"{case}.npz",
                        indices=np.asarray(splits["analysis"]), ids=np.asarray([records[index]["id"] for index in splits["analysis"]]),
                        readout=readouts.cpu().numpy())
                    row = {**metadata, "readout_dim": model.readout_dim,
                        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                        "encoder_source": encoder_source, "head_source": head_source,
                        "initial_encoder_sha256": encoder_initial, "final_encoder_sha256": encoder_final,
                        "initial_state_sha256": initial, "final_state_sha256": final,
                        "original_checkpoint_sha256": environment["source_checkpoint_sha256"][encoder_source],
                        "head_checkpoint_sha256": environment["source_checkpoint_sha256"][head_source],
                        "optimizer_updates": expected_updates, "optimizer_updates_executed_now": updates,
                        "reused_source": bool(reused), "training_seconds": elapsed,
                        "training_order_sha256": hashlib.sha256(orders[f"{experience}_seed{seed}"].tobytes()).hexdigest(),
                        "branch_weight_l2": float(sum(head[0].weight[:, 448:].detach().double().square().sum()
                                                      for head in model.heads.values()).sqrt()),
                        "base_weight_l2": float(sum(head[0].weight[:, :448].detach().double().square().sum()
                                                    for head in model.heads.values()).sqrt())}
                    inventory.append(row)
                    checkpoint_path = output / "checkpoints" / f"{case}.pt"
                    if reused:
                        shutil.copyfile(source / "checkpoints" / f"{experience}_seed{seed}.pt", checkpoint_path)
                    else:
                        torch.save({"state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                                    **row, "config": config}, checkpoint_path)
                    evaluated, per_molecule = endpoint_rows(model, readouts, records, splits, seed, config, metadata,
                        plan["branch_permutations"][str(seed)]["indices"])
                    behavior.extend(evaluated)
                    predictions.extend(per_molecule)
                    write_csv(output / "inventory.csv", inventory)
                    write_csv(output / "initial_checks.csv", initial_checks)
                    write_csv(output / "behavior.csv", behavior)
                    write_csv(output / "predictions.csv", predictions)
                    print(json.dumps({"completed": case, "new_updates": updates,
                                      "training_seconds": elapsed, "cases": len(inventory)}), flush=True)
                    del model, readouts
                del base, reference
            cache_by_encoder.pop((experience, seed), None)
    if {name: verify_record(directory) for name, directory in previous.items()} != references:
        raise AssertionError("A prior scientific record changed.")
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    environment["previous_records_unchanged"] = True
    environment["case_count"] = len(inventory)
    write_json(output / "environment.json", environment)
    from .skip_report import generate_skip_report
    generate_skip_report(output)
    manifest = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "MANIFEST.json"}
    write_json(output / "MANIFEST.json", {"sha256": manifest})
    print(json.dumps({"completed": str(output), "cases": len(inventory), "manifest_files": len(manifest)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/skip_v5.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/skip_v5")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "seeds", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["epochs"] != 120:
        config["reuse_original_deep_baseline"] = False
    run_skip(args.source or ROOT / config["source"], args.output, config)


if __name__ == "__main__":
    main()
