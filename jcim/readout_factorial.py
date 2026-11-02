"""Frozen readout factorial with matched random encoders and source-only labels.

The three independent task heads keep their original 512-input architecture and
archived random initialization. A shallow-only arm zeros the inherited 448
coordinates rather than changing the MLP. The objective factor can influence
the motif head through global gradient clipping; it does not create shared
trainable features or shared task-head parameters.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .analysis import balanced_accuracy
from .hard_model import HardMolecularGNN
from .mechanism import checkpoint_state, encoder_state_sha256
from .pooling_diagnostics import standardizer
from .run import sha256, write_csv, write_json
from .scale_run import id_digest, state_sha256, verify_record
from .skip_model import SkipMolecularGNN
from .skip_run import extract_readouts


SCOPES = ("full512", "shallow_only")
NORMALIZATIONS = ("raw", "train_z")
OBJECTIVES = ("family", "joint")
TASKS = ("family", "position", "distance")


def validate_config(config):
    """Keep every proposed factor and the historical fitting budget explicit."""
    fixed = {"hidden_dim": 64, "layers": 8, "shallow_layer": 1,
             "batch_size": 128, "learning_rate": 0.0015, "weight_decay": 0.0001,
             "gradient_clip": 5.0, "order_seed_offset": 1701, "alpha": 1.0}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("The factorial requires the fixed archived architecture and optimizer.")
    for name, values in (("experiences", ["a", "b"]),
                         ("regimes", ["end_to_end", "frozen_readout"]),
                         ("scopes", list(SCOPES)),
                         ("normalizations", list(NORMALIZATIONS)),
                         ("objectives", list(OBJECTIVES))):
        if config.get(name) != values:
            raise ValueError(f"The complete declared {name} factor is required.")
    seeds = config.get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= {0, 1, 2}:
        raise ValueError("Choose distinct archived seeds.")
    if (isinstance(config["epochs"], bool) or not isinstance(config["epochs"], int)
            or not 1 <= config["epochs"] <= 120):
        raise ValueError("Epochs must be an integer between one and 120.")
    if config["device"] not in ("auto", "cpu", "cuda") or config["cpu_threads"] < 1:
        raise ValueError("A valid device and positive CPU thread count are required.")


def validate_output(output, sources):
    output = Path(output).resolve()
    for path in sources:
        source = Path(path).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("New diagnostics must be separate from archived scientific records.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    return output


def identity_entry(records, indices):
    return {"indices": [int(index) for index in indices],
            "ids": [records[int(index)]["id"] for index in indices]}


def build_plan(records, splits, archived_orders, config):
    """Reject held-out identities, forbidden combinations, and changed orders."""
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Molecular identities must be unique.")
    plan = {"source_only": {}, "tests": {}, "fits": [],
        "labels": "Original 1,386 source molecules only; no target-combination or adaptation labels.",
        "selection": "Fixed final epoch; familiar validation is descriptive and never selects a fit.",
        "scope": "MLP inputs always have 512 coordinates; shallow_only zeros the first 448.",
        "normalization": "train_z uses source-only float64 population mean/std; std <= epsilon becomes one. Raw MLP has zero mean and unit scale.",
        "ridge": "Unpenalized intercept, alpha one. Raw fits center internally to solve the intercept without scaling. Shallow ridge solves the active 64 coordinates.",
        "objective": "Independent frozen-encoder heads share only the global gradient-clipping coefficient; family has coefficient one in both objectives.",
        "head_initialization": "Original same-seed random task heads with zero appended columns; cloned across every factorial cell and encoder."}
    required = ("train_a", "train_b", "val_a", "val_b", "iid_a", "iid_b",
                "ood_common", "analysis", "adaptation_pool")
    for key in required:
        values = splits[key]
        if (not values or len(set(values)) != len(values)
                or any(not isinstance(index, int) or not 0 <= index < len(records) for index in values)):
            raise ValueError(f"Invalid molecular indices: {key}.")
    excluded = set(splits["analysis"] + splits["adaptation_pool"] + splits["ood_common"])
    for experience in config["experiences"]:
        train, val = splits[f"train_{experience}"], splits[f"val_{experience}"]
        if len(train) != 1386 or set(train) & set(val) or set(train + val) & excluded:
            raise ValueError("Source training, familiar validation, and held-out identities overlap.")
        if any((records[index]["y_family"] + records[index]["y_position"]) % 3 == 2 for index in train + val):
            raise ValueError("A forbidden target combination entered source fitting.")
        for seed in config["seeds"]:
            order = archived_orders[f"{experience}_seed{seed}"]
            if order.shape != (120, 1386) or not np.issubdtype(order.dtype, np.integer):
                raise ValueError("All original 120 epoch orders are required.")
            rng = np.random.default_rng(seed + config["order_seed_offset"])
            if any(not np.array_equal(order[epoch], rng.permutation(train)) for epoch in range(120)):
                raise ValueError("A training order differs from the original source curriculum.")
        plan["source_only"][experience] = {"train": identity_entry(records, train),
                                           "validation": identity_entry(records, val)}
    for name in ("iid_a", "iid_b", "ood_common", "analysis"):
        plan["tests"][name] = identity_entry(records, splits[name])
    return plan


def scoped_features(values, scope):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 512 or not np.isfinite(values).all():
        raise ValueError("The actual 512-dimensional v5 readout is required.")
    if scope not in SCOPES:
        raise ValueError("Unknown feature scope.")
    result = values.copy()
    if scope == "shallow_only":
        result[:, :448] = 0
    return result


def fit_scaler(train_values, normalization):
    if normalization == "train_z":
        return standardizer(train_values)
    if normalization != "raw":
        raise ValueError("Unknown normalization.")
    values = np.asarray(train_values)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Fit features must be nonempty and finite.")
    return np.zeros(values.shape[1], dtype=np.float64), np.ones(values.shape[1], dtype=np.float64)


def transform(values, mean, scale):
    return ((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32)


def original_heads(hard_source, seed):
    """Clone the archived 448-column initializer with zero appended columns."""
    base = HardMolecularGNN(64, 8)
    base.load_state_dict(checkpoint_state(Path(hard_source), "random", seed))
    extended = SkipMolecularGNN.from_base(base, "shallow_skip")
    return copy.deepcopy(extended.heads)


def gradient_diagnostics(heads, max_norm):
    """Apply the historical global clip and retain its preclip coupling."""
    family = [parameter.grad for parameter in heads["family"].parameters() if parameter.grad is not None]
    if not family:
        raise ValueError("The motif head must have a gradient.")
    family_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value, 2) for value in family]), 2)
    total_norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm)
    coefficient = torch.clamp(torch.as_tensor(max_norm, device=total_norm.device) / (total_norm + 1e-6), max=1.0)
    return {"total_gradient_norm": float(total_norm), "family_gradient_norm": float(family_norm),
            "clip_coefficient": float(coefficient), "clipped": bool(coefficient < 1)}


@torch.no_grad()
def head_scores(heads, values, batch_size=128):
    """Keep the original fixed inference shape, including the final partial batch."""
    heads.eval()
    result = {task: [] for task in TASKS}
    for offset in range(0, len(values), batch_size):
        count = min(batch_size, len(values) - offset)
        batch = values[offset:offset + count]
        rows = torch.arange(batch_size, device=batch.device) % count
        batch = batch[rows]
        for task, head in heads.items():
            result[task].append(head(batch)[:count].cpu().numpy())
    return {task: np.concatenate(parts) for task, parts in result.items()}


def fit_mlp(heads, values, labels, train_indices, orders, validation_rows, config, objective, device, fit_key):
    """Fit only archived task heads; source labels and exact order determine updates."""
    if objective not in OBJECTIVES:
        raise ValueError("Unknown task objective.")
    heads = copy.deepcopy(heads).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in heads.state_dict().items()}
    parameters = list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lookup = {int(index): row for row, index in enumerate(train_indices)}
    if len(lookup) != len(train_indices):
        raise ValueError("Training identities must be unique.")
    selected_tasks = TASKS if objective == "joint" else ("family",)
    history, diagnostics, updates = [], [], 0
    started = time.monotonic()
    for epoch, order in enumerate(orders):
        if len(order) != len(lookup) or set(map(int, order)) != set(lookup):
            raise ValueError("Every epoch must use exactly the original source identities.")
        heads.train()
        losses = []
        for offset in range(0, len(order), config["batch_size"]):
            chosen = [lookup[int(index)] for index in order[offset:offset + config["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            logits = {task: head(values[chosen]) for task, head in heads.items()}
            loss = sum(F.cross_entropy(logits[task], labels[task][chosen]) for task in selected_tasks)
            loss.backward()
            diagnostics.append({"fit_key": fit_key, "epoch": epoch + 1, "step": updates + 1,
                                **gradient_diagnostics(heads, config["gradient_clip"])})
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == len(orders):
            predictions = head_scores(heads, values[validation_rows], config["batch_size"])
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                **{f"val_{task}_accuracy": float(np.mean(predictions[task].argmax(1) == labels[task][validation_rows].cpu().numpy())) for task in TASKS}})
    expected = len(orders) * math.ceil(len(train_indices) / config["batch_size"])
    if updates != expected:
        raise AssertionError("A factorial cell received a different update budget.")
    state = {key: value.detach().cpu().clone() for key, value in heads.state_dict().items()}
    return {"initial_state_dict": initial, "state_dict": state, "history": history,
            "optimizer_updates": updates, "training_seconds": time.monotonic() - started,
            "clipping": diagnostics}, heads


def fit_ridge(values, labels, normalization, scope, alpha=1.0):
    """Fit a fixed-regularization readout, keeping the intercept unpenalized."""
    if alpha != 1.0 or scope not in SCOPES or normalization not in NORMALIZATIONS:
        raise ValueError("The declared ridge protocol is fixed.")
    active = np.arange(512) if scope == "full512" else np.arange(448, 512)
    selected = np.ascontiguousarray(np.asarray(values)[:, active], dtype=np.float64)
    mean, scale = fit_scaler(selected, normalization)
    z = (selected - mean) / scale
    targets = np.eye(8)[np.asarray(labels)]
    target_mean = targets.mean(0)
    if normalization == "raw":
        center = z.mean(0)
        design = z - center
    else:
        # Preserve the exact historical standardized ridge implementation.
        center = np.zeros(z.shape[1], dtype=np.float64)
        design = z
    covariance = design.T @ design
    covariance.flat[::len(covariance) + 1] += alpha
    weights = np.linalg.solve(covariance, design.T @ (targets - target_mean))
    return {"active_columns": active, "mean": mean, "scale": scale,
            "weights": weights, "bias": target_mean - center @ weights}


def ridge_scores(fitted, values):
    selected = np.ascontiguousarray(np.asarray(values)[:, fitted["active_columns"]], dtype=np.float64)
    return ((selected - fitted["mean"]) / fitted["scale"]) @ fitted["weights"] + fitted["bias"]


def encoder_cases(hard_source, skip_source, config):
    cases = []
    for seed in config["seeds"]:
        for regime in config["regimes"]:
            for experience in config["experiences"]:
                name = f"{regime}_shallow_skip_{experience}_seed{seed}"
                cases.append({"model": name, "condition": "trained", "regime": regime,
                    "experience": experience, "seed": seed,
                    "checkpoint": Path(skip_source) / "checkpoints" / f"{name}.pt"})
        cases.append({"model": f"random_seed{seed}", "condition": "random", "regime": "random",
            "experience": "shared", "seed": seed,
            "checkpoint": Path(hard_source) / "checkpoints" / f"random_seed{seed}.pt"})
    return cases


def exact_array_check(model, reference, actual, expected, kind):
    actual, expected = np.asarray(actual), np.asarray(expected)
    equal = actual.shape == expected.shape and np.array_equal(actual, expected)
    delta = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else -1.0
    row = {"model": model, "reference": str(reference), "kind": kind,
           "shape": str(actual.shape), "exact_match": equal, "max_abs_delta": delta}
    if not equal:
        raise AssertionError(f"Archived feature or readout continuity failed: {row}")
    return row


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_readout_factorial(hard_source, skip_source, pooling_source, output, config):
    """Run one complete frozen panel; the coordinator creates report and manifest."""
    validate_config(config)
    hard_source, skip_source, pooling_source = map(lambda path: Path(path).resolve(),
                                                  (hard_source, skip_source, pooling_source))
    output = validate_output(output, (hard_source, skip_source, pooling_source))
    references = {path.name: verify_record(path) for path in (hard_source, skip_source, pooling_source)}
    records = [json.loads(line) for line in (hard_source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((hard_source / "splits.json").read_text(encoding="utf-8"))
    with np.load(skip_source / "training_orders.npz") as archive:
        archived_orders = {key: archive[key].copy() for key in archive.files}
    plan = build_plan(records, splits, archived_orders, config)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("representations", "predictions", "checkpoints", "histories"):
        (output / name).mkdir()
    write_json(output / "panel_config.json", config)
    write_json(output / "source_reference.json", {"records": references})
    write_json(output / "plan.json", plan)
    orders = {key: value[:config["epochs"]] for key, value in archived_orders.items()
              if int(key.rsplit("seed", 1)[1]) in config["seeds"]}
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
    with np.load(hard_source / "graphs.npz") as archive:
        tensors = {key: torch.as_tensor(archive[key].copy(), device=device) for key in ("x", "adj", "mask")}
    indices = sorted(set(splits["train_a"] + splits["train_b"] + splits["val_a"] + splits["val_b"] + splits["analysis"]))
    lookup = {index: row for row, index in enumerate(indices)}
    ids = np.asarray([records[index]["id"] for index in indices])
    analysis_rows = [lookup[index] for index in splits["analysis"]]
    truth = {task: np.asarray([record[f"y_{task}"] for record in records], dtype=np.int64) for task in TASKS}
    pooling_root = pooling_source / "pooling"
    previous_ridge = {(row["model"], row["experience"]): row for row in _read_csv(pooling_root / "pooling_fit_inventory.csv")
                      if row["classifier"] == "ridge" and row["protocol"] == "source_only" and row["pool"] == "whole_graph"}
    v5_predictions = {}
    for row in _read_csv(skip_source / "predictions.csv"):
        if row["regime"] == "frozen_readout" and row["variant"] == "shallow_skip" and row["intervention"] == "intact":
            key = (row["experience"], int(row["seed"]), int(row["index"]))
            v5_predictions[key] = tuple(int(row[f"pred_{task}"]) for task in TASKS)
    encoders, feature_checks, fits, metrics, continuity, all_clipping = [], [], [], [], [], []

    def save_metrics(fit_key, metadata, scores):
        predicted = {task: values.argmax(1) for task, values in scores.items()}
        for name in ("train", "iid_a", "iid_b", "ood_common"):
            selected = splits[f"train_{metadata['experience']}"] if name == "train" else splits[name]
            rows = [lookup[index] for index in selected]
            for task, prediction in predicted.items():
                expected = truth[task][selected]
                metrics.append({"fit_key": fit_key, **metadata, "split": name, "task": task,
                    "task_trained": task == "family" or metadata["objective"] == "joint",
                    "n_test": len(selected), "accuracy": float(np.mean(prediction[rows] == expected)),
                    "balanced_accuracy": balanced_accuracy(expected, prediction[rows])})
        return predicted

    for case in encoder_cases(hard_source, skip_source, config):
        source_state = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
        base = HardMolecularGNN(64, 8)
        base_state = base.state_dict()
        base_state.update({key: value for key, value in source_state.items() if not key.startswith("heads.")})
        base.load_state_dict(base_state)
        model = SkipMolecularGNN.from_base(base, "shallow_skip").to(device)
        model.requires_grad_(False)
        before = encoder_state_sha256(model.state_dict())
        if before != encoder_state_sha256(source_state):
            raise AssertionError("An encoder tensor was omitted when loading a frozen source.")
        values = extract_readouts(model, tensors, indices, config["batch_size"]).cpu().numpy()
        feature_path = output / "representations" / f"{case['model']}.npz"
        np.savez_compressed(feature_path, indices=np.asarray(indices, dtype=np.int32), ids=ids, readout=values)
        v6_features = pooling_root / "representations" / f"pooling_{case['model']}.npz"
        with np.load(v6_features) as archived:
            if archived["indices"].tolist() != indices or not np.array_equal(archived["ids"], ids):
                raise AssertionError("v6 feature identities changed.")
            feature_checks.append(exact_array_check(case["model"], v6_features, values[:, 448:], archived["whole_graph"], "v6_shallow64"))
        if case["condition"] == "trained":
            v5_features = skip_source / "representations" / f"{case['model']}.npz"
            with np.load(v5_features) as archived:
                if archived["indices"].tolist() != splits["analysis"]:
                    raise AssertionError("v5 analysis identities changed.")
                feature_checks.append(exact_array_check(case["model"], v5_features, values[analysis_rows], archived["readout"], "v5_analysis512"))
            if case["regime"] == "frozen_readout":
                v5_cache = skip_source / "representations" / f"frozen_{case['experience']}_seed{case['seed']}.npz"
                with np.load(v5_cache) as archived:
                    rows = [lookup[int(index)] for index in archived["indices"]]
                    expected = np.concatenate((archived["deep_graph"], archived["shallow_mean"]), axis=1)
                    feature_checks.append(exact_array_check(case["model"], v5_cache, values[rows], expected, "v5_frozen_cache512"))
        initializer = original_heads(hard_source, case["seed"])
        initial_hash = state_sha256(initializer.state_dict())
        experiences = config["experiences"] if case["condition"] == "random" else [case["experience"]]
        for experience in experiences:
            train = splits[f"train_{experience}"]
            train_rows = [lookup[index] for index in train]
            # Training rows lead the cache, reproducing v5's original gather order.
            train_row_set = set(train_rows)
            ordered_rows = train_rows + [row for row in range(len(indices)) if row not in train_row_set]
            local_lookup = {indices[row]: position for position, row in enumerate(ordered_rows)}
            validation_rows = [local_lookup[index] for index in splits[f"val_{experience}"]]
            labels = {task: torch.as_tensor(truth[task][np.asarray(indices)[ordered_rows]], device=device) for task in TASKS}
            base_metadata = {key: case[key] for key in ("model", "condition", "regime", "seed")}
            base_metadata["experience"] = experience
            for scope in SCOPES:
                scoped = scoped_features(values, scope)
                for normalization in NORMALIZATIONS:
                    mean, scale = fit_scaler(scoped[train_rows], normalization)
                    normalized = transform(scoped, mean, scale)
                    transformed = torch.as_tensor(normalized[ordered_rows], device=device)
                    for objective in OBJECTIVES:
                        fit_key = f"fit_{len(fits):04d}"
                        metadata = {**base_metadata, "classifier": "mlp", "scope": scope,
                                    "normalization": normalization, "objective": objective}
                        fitted, heads = fit_mlp(initializer, transformed, labels, train,
                            orders[f"{experience}_seed{case['seed']}"], validation_rows,
                            config, objective, device, fit_key)
                        scores = head_scores(heads, torch.as_tensor(normalized, device=device), config["batch_size"])
                        predicted = save_metrics(fit_key, metadata, scores)
                        checkpoint = output / "checkpoints" / f"{fit_key}.pt"
                        torch.save({"mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                                    "initial_state_dict": fitted["initial_state_dict"],
                                    "state_dict": fitted["state_dict"], "metadata": metadata}, checkpoint)
                        prediction_path = output / "predictions" / f"{fit_key}.npz"
                        np.savez_compressed(prediction_path, indices=np.asarray(indices, dtype=np.int32), ids=ids,
                            **{f"logits_{task}": array for task, array in scores.items()},
                            **{f"pred_{task}": array.astype(np.int8) for task, array in predicted.items()})
                        history_path = output / "histories" / f"{fit_key}.json"
                        write_json(history_path, fitted["history"])
                        clipping = fitted["clipping"]
                        all_clipping.extend(clipping)
                        fits.append({"fit_key": fit_key, **metadata, "n_fit": len(train),
                            "encoder_sha256": before, "support_ids_sha256": id_digest([records[index]["id"] for index in train]),
                            "support_labels_sha256": hashlib.sha256(truth["family"][train].tobytes()).hexdigest(),
                            "training_order_sha256": hashlib.sha256(orders[f"{experience}_seed{case['seed']}"].tobytes()).hexdigest(),
                            "initial_head_sha256": state_sha256(fitted["initial_state_dict"]),
                            "final_head_sha256": state_sha256(fitted["state_dict"]),
                            "optimizer_updates": fitted["optimizer_updates"], "training_seconds": fitted["training_seconds"],
                            "clip_count": sum(row["clipped"] for row in clipping),
                            "minimum_clip_coefficient": min(row["clip_coefficient"] for row in clipping),
                            "feature_dim": 512, "active_feature_dim": 512 if scope == "full512" else 64,
                            "classifier_parameter_count": sum(parameter.numel() for parameter in heads.parameters()),
                            "family_parameter_count": sum(parameter.numel() for parameter in heads["family"].parameters()),
                            "checkpoint": checkpoint.relative_to(output).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                            "predictions": prediction_path.relative_to(output).as_posix(), "predictions_sha256": sha256(prediction_path),
                            "history": history_path.relative_to(output).as_posix()})
                        if fits[-1]["initial_head_sha256"] != initial_hash:
                            raise AssertionError("A factorial arm received a different initial classifier.")
                        plan["fits"].append({"fit_key": fit_key, **metadata, "support": identity_entry(records, train)})
                        if (config["epochs"] == 120 and case["regime"] == "frozen_readout"
                                and scope == "full512" and normalization == "raw" and objective == "joint"):
                            expected = {key.removeprefix("heads."): value for key, value in source_state.items() if key.startswith("heads.")}
                            identical = all(torch.equal(fitted["state_dict"][key], expected[key]) for key in expected)
                            prediction_equal = all(tuple(int(predicted[task][lookup[index]]) for task in TASKS)
                                == v5_predictions[(experience, case["seed"], index)] for index in splits["analysis"])
                            continuity.append({"fit_key": fit_key, "model": case["model"], "experience": experience,
                                "reference": str(case["checkpoint"]), "kind": "v5_raw_full_joint",
                                "state_exact_match": identical, "predictions_exact_match": prediction_equal,
                                "max_abs_delta": max(float((fitted["state_dict"][key] - expected[key]).abs().max()) for key in expected)})
                            if not identical or not prediction_equal:
                                raise AssertionError(f"The exact v5 frozen-head continuity failed: {continuity[-1]}")
                        print(json.dumps({"fit_completed": fit_key, **metadata, "optimizer_updates": fitted["optimizer_updates"]}), flush=True)
                        del heads
                    fit_key = f"fit_{len(fits):04d}"
                    metadata = {**base_metadata, "classifier": "ridge", "scope": scope,
                                "normalization": normalization, "objective": "ridge"}
                    fitted = fit_ridge(scoped[train_rows], truth["family"][train], normalization, scope, config["alpha"])
                    scores = {"family": ridge_scores(fitted, scoped)}
                    predicted = save_metrics(fit_key, metadata, scores)
                    checkpoint = output / "checkpoints" / f"{fit_key}.npz"
                    np.savez_compressed(checkpoint, **fitted)
                    prediction_path = output / "predictions" / f"{fit_key}.npz"
                    np.savez_compressed(prediction_path, indices=np.asarray(indices, dtype=np.int32), ids=ids,
                                        logits_family=scores["family"], pred_family=predicted["family"].astype(np.int8))
                    fits.append({"fit_key": fit_key, **metadata, "n_fit": len(train),
                        "encoder_sha256": before, "support_ids_sha256": id_digest([records[index]["id"] for index in train]),
                        "support_labels_sha256": hashlib.sha256(truth["family"][train].tobytes()).hexdigest(),
                        "training_order_sha256": "", "initial_head_sha256": "", "final_head_sha256": "",
                        "optimizer_updates": 0, "training_seconds": 0.0, "clip_count": 0, "minimum_clip_coefficient": 1.0,
                        "feature_dim": 512 if scope == "full512" else 64,
                        "active_feature_dim": 512 if scope == "full512" else 64,
                        "classifier_parameter_count": fitted["weights"].size + fitted["bias"].size,
                        "family_parameter_count": fitted["weights"].size + fitted["bias"].size,
                        "checkpoint": checkpoint.relative_to(output).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                        "predictions": prediction_path.relative_to(output).as_posix(), "predictions_sha256": sha256(prediction_path), "history": ""})
                    plan["fits"].append({"fit_key": fit_key, **metadata, "support": identity_entry(records, train)})
                    if scope == "shallow_only" and normalization == "train_z":
                        reference = previous_ridge[(case["model"], experience)]
                        with np.load(pooling_root / reference["predictions"]) as archived:
                            if not np.array_equal(archived["indices"], indices):
                                raise AssertionError("v6 prediction identities changed.")
                            continuity.append({"fit_key": fit_key, "model": case["model"], "experience": experience,
                                "reference": str(pooling_root / reference["predictions"]), "kind": "v6_shallow_standardized_ridge",
                                "state_exact_match": True,
                                "predictions_exact_match": bool(np.array_equal(predicted["family"], archived["prediction"])),
                                "max_abs_delta": float(np.max(np.abs(scores["family"] - archived["logits"])))})
                            if not continuity[-1]["predictions_exact_match"] or continuity[-1]["max_abs_delta"] != 0:
                                raise AssertionError(f"The exact v6 ridge continuity failed: {continuity[-1]}")
                        with np.load(pooling_root / reference["checkpoint"]) as archived:
                            if any(not np.array_equal(fitted[key], archived[key]) for key in ("mean", "scale", "weights", "bias")):
                                raise AssertionError("v6 standardized ridge coefficients changed.")
        after = encoder_state_sha256(model.state_dict())
        if before != after:
            raise AssertionError("A diagnostic fit updated a frozen encoder.")
        encoders.append({**case, "checkpoint": str(case["checkpoint"]), "checkpoint_sha256": sha256(case["checkpoint"]),
            "encoder_before_sha256": before, "encoder_after_sha256": after, "encoder_bitwise_unchanged": True,
            "features": feature_path.relative_to(output).as_posix(), "features_sha256": sha256(feature_path),
            "n_features": len(indices), "feature_dim": 512, "initial_head_sha256": initial_hash})
        for name, rows in (("encoder_inventory.csv", encoders), ("feature_checks.csv", feature_checks),
                           ("fit_inventory.csv", fits), ("source_metrics.csv", metrics),
                           ("continuity_checks.csv", continuity), ("gradient_clipping.csv", all_clipping)):
            write_csv(output / name, rows)
        write_json(output / "plan.json", plan)
        print(json.dumps({"encoder_completed": case["model"], "encoders": len(encoders), "fits": len(fits)}), flush=True)
        del model, values
    if {path.name: verify_record(path) for path in (hard_source, skip_source, pooling_source)} != references:
        raise AssertionError("A previous scientific record changed during fitting.")
    return {"encoder_count": len(encoders), "fit_count": len(fits),
            "mlp_fit_count": sum(row["classifier"] == "mlp" for row in fits),
            "ridge_fit_count": sum(row["classifier"] == "ridge" for row in fits),
            "metric_rows": len(metrics), "gradient_clipping_rows": len(all_clipping),
            "feature_checks": len(feature_checks), "continuity_checks": len(continuity),
            "encoders_bitwise_unchanged": True, "previous_records_unchanged": True}
