"""Separate supervised pooling diagnostics from source-only task readouts.

Every encoder receives the unchanged complete molecular input and stays frozen.
The target pool uses oracle fragment membership only after message layer one.
Source-only fits use the original curriculum labels. Directed probes instead use
the archived balanced source-position diagnostic labels; these protocols must
never be interpreted as the same experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .analysis import balanced_accuracy
from .hard_model import HardMolecularGNN
from .mechanism import encoder_state_sha256
from .run import sha256, write_csv, write_json
from .scale_run import id_digest, state_sha256
from .skip_model import stable_graph_mean


POOLS = ("whole_graph", "target")


def validate_config(config):
    """Keep the declared architecture, label budget, and optimizer fixed."""
    expected = {"hidden_dim": 64, "layers": 8, "shallow_layer": 1,
                "batch_size": 128, "learning_rate": 0.0015, "weight_decay": 0.0001,
                "gradient_clip": 5.0, "alpha": 1.0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Pooling diagnostics require the fixed matched protocol.")
    for key, allowed in (("seeds", {0, 1, 2}), ("support_repeats", {0, 1, 2})):
        values = config[key]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"{key} must select distinct archived values.")
    if config["regimes"] != ["end_to_end", "frozen_readout"] or config["experiences"] != ["a", "b"]:
        raise ValueError("Both archived regimes and curricula are required.")
    if isinstance(config["epochs"], bool) or not 1 <= config["epochs"] <= 120:
        raise ValueError("Choose a positive budget no larger than the archived epoch order.")
    if not isinstance(config["mlp_seed_offset"], int):
        raise ValueError("The MLP initialization seed offset must be declared.")


def checked_indices(values, count, name):
    result = np.asarray(values)
    if (result.ndim != 1 or not result.size or not np.issubdtype(result.dtype, np.integer)
            or len(set(result.tolist())) != len(result) or np.any(result < 0) or np.any(result >= count)):
        raise ValueError(f"{name} must contain unique valid molecular indices.")
    return result.astype(np.int64)


def identity_entry(records, indices):
    return {"indices": [int(index) for index in indices],
            "ids": [records[int(index)]["id"] for index in indices]}


def build_plan(records, splits, direct, archived_orders, config):
    """Validate original source isolation and the separate diagnostic labels."""
    count = len(records)
    if len({row["id"] for row in records}) != count:
        raise ValueError("Molecular identities must be unique.")
    needed = ("train_a", "train_b", "val_a", "val_b", "iid_a", "iid_b", "ood_common",
              "analysis", "adaptation_pool")
    selected = {key: checked_indices(splits[key], count, key).tolist() for key in needed}
    plan = {"source_only": {}, "tests": {}, "direct_support": [], "fits": [],
            "source_protocol": "Original curriculum motif labels only; fixed final epoch; no validation selection.",
            "direct_protocol": "Archived balanced source-position diagnostic labels from analysis; zero target-position labels per direction.",
            "oracle_target_pooling": "Known target_fragment_indices used only after the frozen full-input first message layer.",
            "standardization": "Mean and population standard deviation estimated on each fit support only, float64; scale <= float64 epsilon replaced by one.",
            "mlp_loss": "Motif cross-entropy only, unlike the original v5 three-task loss."}
    common = set(selected["ood_common"])
    analysis = set(selected["analysis"])
    reserved = set(selected["adaptation_pool"])
    for experience in config["experiences"]:
        train, validation = selected[f"train_{experience}"], selected[f"val_{experience}"]
        if len(train) != 1386 or set(train) & set(validation):
            raise ValueError("The original 1,386-molecule training curriculum is required.")
        if (set(train + validation) & (analysis | reserved | common)
                or any((records[index]["y_family"] + records[index]["y_position"]) % 3 == 2
                       for index in train + validation)):
            raise ValueError("Source fitting must exclude every reserved identity and common-OOD combination.")
        for seed in config["seeds"]:
            orders = archived_orders[f"{experience}_seed{seed}"]
            if orders.shape != (120, 1386) or not np.issubdtype(orders.dtype, np.integer):
                raise ValueError("The original complete 120-epoch order is required.")
            expected_rng = np.random.default_rng(1701 + seed)
            for epoch in range(120):
                expected = expected_rng.permutation(train)
                if not np.array_equal(orders[epoch], expected):
                    raise ValueError("An archived training order differs from its original curriculum.")
        plan["source_only"][experience] = {"train": identity_entry(records, train),
                                           "validation_not_used_for_selection": identity_entry(records, validation)}
    for split in ("iid_a", "iid_b", "ood_common", "analysis"):
        plan["tests"][split] = identity_entry(records, selected[split])
    keys = set()
    for support in direct["sets"]:
        repeat, position = support["repeat"], support["source_position"]
        if repeat not in config["support_repeats"]:
            continue
        key = (repeat, position)
        indices = checked_indices(support["indices"], count, "direct support").tolist()
        if key in keys or position not in range(3) or len(indices) != 256:
            raise ValueError("Each directed diagnostic needs one unique balanced 256-label support.")
        if (not set(indices) <= analysis or support["ids"] != [records[index]["id"] for index in indices]
                or any(records[index]["y_position"] != position for index in indices)
                or any(sum(records[index]["y_family"] == label for index in indices) != 32 for label in range(8))):
            raise ValueError("The archived directed support violates label, identity, or position isolation.")
        keys.add(key)
        plan["direct_support"].append(support)
    if keys != {(repeat, position) for repeat in config["support_repeats"] for position in range(3)}:
        raise ValueError("A directed diagnostic support is missing.")
    return plan


def target_masks(records, atom_mask, indices):
    """Build oracle membership for pooling, without changing encoder inputs."""
    indices = checked_indices(indices, len(records), "feature selection")
    valid = atom_mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(atom_mask) else np.asarray(atom_mask, dtype=bool)
    if valid.ndim != 2 or len(valid) != len(records):
        raise ValueError("Padded atom masks and molecular records disagree.")
    result = np.zeros((len(indices), valid.shape[1]), dtype=bool)
    for row, index in enumerate(indices):
        fragment = checked_indices(records[index]["target_fragment_indices"], valid.shape[1], "target membership")
        root = records[index]["target_attachment_idx"]
        if root not in fragment or not valid[index, fragment].all():
            raise ValueError("Oracle membership must contain the marked root and exclude padding.")
        result[row, fragment] = True
    return result


@torch.no_grad()
def extract_first_layer(model, tensors, indices, membership, batch_size=128):
    """Extract matched 64-vectors at fixed shape from the unchanged full input."""
    selected = checked_indices(indices, len(tensors["x"]), "feature selection")
    membership = np.asarray(membership, dtype=bool)
    if membership.shape != (len(selected), tensors["x"].shape[1]) or np.any(membership.sum(1) == 0):
        raise ValueError("Oracle membership must be nonempty and follow selected graph rows.")
    if batch_size != 128:
        raise ValueError("The archived CUDA inference shape is fixed at 128.")
    was_training = model.training
    result = {name: [] for name in POOLS}
    model.eval()
    try:
        for offset in range(0, len(selected), batch_size):
            local = np.arange(offset, min(offset + batch_size, len(selected)))
            genuine = len(local)
            local = np.resize(local, batch_size)
            chosen = selected[local]
            x, adjacency, mask = (tensors[name][chosen] for name in ("x", "adj", "mask"))
            oracle = torch.as_tensor(membership[local], device=x.device)
            if torch.any(oracle & ~mask.bool()).item():
                raise ValueError("Oracle membership includes a padded atom.")
            h = torch.relu(model.embedding(x)) * mask.unsqueeze(-1)
            h = model.layers[0](h, adjacency, mask)
            for name, region in (("whole_graph", mask), ("target", oracle)):
                result[name].append(stable_graph_mean(h, region)[:genuine].cpu().numpy())
    finally:
        model.train(was_training)
    return {name: np.concatenate(parts) for name, parts in result.items()}


def standardizer(train_x):
    values = np.asarray(train_x, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Fit features must be a nonempty finite matrix.")
    mean, scale = values.mean(0), values.std(0)
    return mean, np.where(scale > np.finfo(np.float64).eps, scale, 1.0)


def ridge_fit(train_x, train_y, alpha=1.0):
    """Store the exact support-only standardized multiclass ridge solution."""
    mean, scale = standardizer(train_x)
    labels = np.asarray(train_y)
    if labels.shape != (len(train_x),) or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 7)):
        raise ValueError("Eight-class integer fit labels are required.")
    if alpha != 1.0:
        raise ValueError("The ridge regularization was fixed at one.")
    z = (np.asarray(train_x, dtype=np.float64) - mean) / scale
    targets = np.eye(8)[labels]
    bias = targets.mean(0)
    covariance = z.T @ z
    covariance.flat[::len(covariance) + 1] += alpha
    weights = np.linalg.solve(covariance, z.T @ (targets - bias))
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def new_head(seed, offset=61000):
    """Create one CPU initializer; callers clone it for all paired conditions."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(offset + seed)
        return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))


def mlp_fit(train_x, train_y, fit_indices, orders, seed, config, device):
    """Fit a fresh standardized motif head without access to encoder weights."""
    mean, scale = standardizer(train_x)
    head = new_head(seed, config["mlp_seed_offset"]).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    z = torch.as_tensor(((np.asarray(train_x, dtype=np.float64) - mean) / scale).astype(np.float32), device=device)
    labels = torch.as_tensor(train_y, dtype=torch.long, device=device)
    lookup = {int(index): row for row, index in enumerate(fit_indices)}
    if len(lookup) != len(fit_indices) or labels.shape != (len(fit_indices),):
        raise ValueError("Training identities and labels must correspond exactly.")
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    updates, history = 0, []
    for epoch, order in enumerate(orders):
        if set(map(int, order)) != set(lookup) or len(order) != len(lookup):
            raise ValueError("Every MLP epoch must use precisely its original source identities.")
        losses = []
        for offset in range(0, len(order), config["batch_size"]):
            rows = [lookup[int(index)] for index in order[offset:offset + config["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(z[rows]), labels[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == len(orders):
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
    expected = len(orders) * math.ceil(len(fit_indices) / config["batch_size"])
    if updates != expected:
        raise AssertionError("A pooling arm received a different optimizer budget.")
    head.eval()
    state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    return {"mean": mean, "scale": scale, "initial_state_dict": initial, "state_dict": state,
            "optimizer_updates": updates, "history": history}, head


@torch.no_grad()
def mlp_scores(head, values, mean, scale, batch_size=128):
    rows = []
    device = next(head.parameters()).device
    standardized = ((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32)
    for offset in range(0, len(values), batch_size):
        genuine = min(batch_size, len(values) - offset)
        positions = np.resize(np.arange(genuine), batch_size)
        batch = torch.as_tensor(standardized[offset:offset + genuine][positions], device=device)
        rows.append(head(batch)[:genuine].cpu().numpy())
    return np.concatenate(rows)


def metric(prediction, truth):
    return {"n_test": len(truth), "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": balanced_accuracy(truth, prediction)}


def exact_feature_check(model_name, reference, actual, expected, indices, records):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("Archived branch dimensions differ from re-extracted pooling.")
    equal = np.array_equal(actual, expected)
    result = {"model": model_name, "reference": str(reference), "n_rows": len(indices),
              "ids_sha256": id_digest([records[index]["id"] for index in indices]),
              "exact_match": equal, "max_abs_delta": float(np.max(np.abs(actual - expected)))}
    if not equal:
        raise AssertionError(f"Actual v5 appended branch changed: {result}")
    return result


def _encoder_cases(skip_source, hard_source, config):
    cases = []
    for seed in config["seeds"]:
        for regime in config["regimes"]:
            for experience in config["experiences"]:
                name = f"{regime}_shallow_skip_{experience}_seed{seed}"
                cases.append({"model": name, "condition": "trained", "regime": regime,
                              "experience": experience, "seed": seed,
                              "checkpoint": skip_source / "checkpoints" / f"{name}.pt"})
        cases.append({"model": f"random_seed{seed}", "condition": "random", "regime": "random",
                      "experience": "shared", "seed": seed,
                      "checkpoint": hard_source / "checkpoints" / f"random_seed{seed}.pt"})
    return cases


def run_pooling_diagnostics(skip_source, hard_source, scale_source, output, config):
    """Run frozen diagnostics, leaving manifest/report creation to the coordinator."""
    validate_config(config)
    skip_source, hard_source, scale_source, output = map(lambda value: Path(value).resolve(),
                                                       (skip_source, hard_source, scale_source, output))
    if any(output.glob("pooling_*")) or (output / "representations").exists():
        raise FileExistsError("Refusing to overwrite pooling diagnostic records.")
    records = [json.loads(line) for line in (hard_source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((hard_source / "splits.json").read_text(encoding="utf-8"))
    direct = json.loads((scale_source / "direct_support.json").read_text(encoding="utf-8"))
    with np.load(skip_source / "training_orders.npz") as archive:
        orders = {key: archive[key].copy() for key in archive.files}
    plan = build_plan(records, splits, direct, orders, config)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("representations", "predictions", "checkpoints"):
        (output / name).mkdir(exist_ok=True)
    write_json(output / "pooling_plan.json", plan)
    np.savez_compressed(output / "pooling_training_orders.npz", **{
        key: value[:config["epochs"]] for key, value in orders.items()
        if int(key.rsplit("seed", 1)[1]) in config["seeds"]})
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
    feature_indices = sorted(set(splits["train_a"] + splits["train_b"] + splits["val_a"] +
                                 splits["val_b"] + splits["analysis"]))
    feature_lookup = {index: row for row, index in enumerate(feature_indices)}
    analysis_indices = splits["analysis"]
    analysis_lookup = {index: row for row, index in enumerate(analysis_indices)}
    analysis_rows = [feature_lookup[index] for index in analysis_indices]
    membership = target_masks(records, tensors["mask"], feature_indices)
    if not np.all(membership.sum(1) == 9):
        raise ValueError("The archived target motifs must each contain exactly nine atoms.")
    np.savez_compressed(output / "pooling_target_membership.npz",
                        indices=np.asarray(feature_indices, dtype=np.int32),
                        ids=np.asarray([records[index]["id"] for index in feature_indices]),
                        target_membership=membership)
    encoders, checks, fits, source_metrics, direct_metrics = [], [], [], [], []

    def save_fit(values, train_indices, metadata, orders_for_fit=None):
        fit_key = f"pooling_fit_{len(fits):04d}"
        support_rows = [feature_lookup[index] for index in train_indices]
        labels = np.asarray([records[index]["y_family"] for index in train_indices], dtype=np.int64)
        selected = feature_indices if metadata["protocol"] == "source_only" else analysis_indices
        selected_rows = [feature_lookup[index] for index in selected]
        if metadata["classifier"] == "ridge":
            fitted = ridge_fit(values[support_rows], labels, config["alpha"])
            scores = ((values[selected_rows].astype(np.float64) - fitted["mean"]) / fitted["scale"]) @ fitted["weights"] + fitted["bias"]
            checkpoint = output / "checkpoints" / f"{fit_key}.npz"
            np.savez_compressed(checkpoint, **fitted)
            init_hash, final_hash, updates, order_hash = "", "", 0, ""
        else:
            fitted, head = mlp_fit(values[support_rows], labels, train_indices, orders_for_fit,
                                   metadata["seed"], config, device)
            scores = mlp_scores(head, values[selected_rows], fitted["mean"], fitted["scale"])
            checkpoint = output / "checkpoints" / f"{fit_key}.pt"
            torch.save({**fitted, "mean": torch.from_numpy(fitted["mean"]),
                        "scale": torch.from_numpy(fitted["scale"])}, checkpoint)
            init_hash, final_hash = state_sha256(fitted["initial_state_dict"]), state_sha256(fitted["state_dict"])
            updates = fitted["optimizer_updates"]
            order_hash = hashlib.sha256(orders_for_fit.tobytes()).hexdigest()
            del head
        prediction = scores.argmax(1).astype(np.int8)
        prediction_path = output / "predictions" / f"{fit_key}.npz"
        np.savez_compressed(prediction_path, indices=np.asarray(selected, dtype=np.int32),
                            ids=np.asarray([records[index]["id"] for index in selected]),
                            logits=scores, prediction=prediction)
        row = {"fit_key": fit_key, **metadata, "n_fit": len(train_indices),
               "support_ids_sha256": id_digest([records[index]["id"] for index in train_indices]),
               "support_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
               "training_order_sha256": order_hash, "initial_head_sha256": init_hash,
               "final_head_sha256": final_hash, "optimizer_updates": updates,
               "checkpoint": checkpoint.relative_to(output).as_posix(), "checkpoint_sha256": sha256(checkpoint),
               "predictions": prediction_path.relative_to(output).as_posix(),
               "predictions_sha256": sha256(prediction_path), "feature_dim": 64,
               "classifier_parameter_count": 4680 if metadata["classifier"] == "mlp" else 520}
        fits.append(row)
        plan["fits"].append({"fit_key": fit_key, "protocol": metadata["protocol"],
                             "support": identity_entry(records, train_indices)})
        return fit_key, prediction

    for case in _encoder_cases(skip_source, hard_source, config):
        checkpoint = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)
        source_state = checkpoint["state_dict"]
        model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
        state = model.state_dict()
        state.update({key: value for key, value in source_state.items() if not key.startswith("heads.")})
        model.load_state_dict(state)
        model.requires_grad_(False)
        before = encoder_state_sha256(model.state_dict())
        if before != encoder_state_sha256(source_state):
            raise ValueError("An archived encoder tensor was omitted during checkpoint loading.")
        features = extract_first_layer(model, tensors, feature_indices, membership, config["batch_size"])
        feature_path = output / "representations" / f"pooling_{case['model']}.npz"
        np.savez_compressed(feature_path, indices=np.asarray(feature_indices, dtype=np.int32),
                            ids=np.asarray([records[index]["id"] for index in feature_indices]), **features)
        if case["condition"] == "trained":
            reference = skip_source / "representations" / f"{case['model']}.npz"
            with np.load(reference) as archived:
                reference_indices = archived["indices"].tolist()
                if reference_indices != analysis_indices or archived["ids"].tolist() != [records[index]["id"] for index in analysis_indices]:
                    raise ValueError("The actual v5 branch archive uses different analysis identities.")
                checks.append(exact_feature_check(case["model"], reference,
                              features["whole_graph"][analysis_rows], archived["readout"][:, 448:],
                              analysis_indices, records))
            if case["regime"] == "frozen_readout":
                reference = skip_source / "representations" / f"frozen_{case['experience']}_seed{case['seed']}.npz"
                with np.load(reference) as archived:
                    indices = archived["indices"].tolist()
                    if archived["ids"].tolist() != [records[index]["id"] for index in indices]:
                        raise ValueError("The frozen v5 cache contains different identities.")
                    checks.append(exact_feature_check(case["model"], reference,
                                  features["whole_graph"][[feature_lookup[index] for index in indices]],
                                  archived["shallow_mean"], indices, records))
        experiences = config["experiences"] if case["condition"] == "random" else [case["experience"]]
        base = {key: case[key] for key in ("model", "condition", "regime", "seed")}
        for experience in experiences:
            for pool in POOLS:
                for classifier in ("ridge", "mlp"):
                    metadata = {**base, "experience": experience, "pool": pool, "classifier": classifier,
                                "protocol": "source_only", "repeat": -1, "source_position": -1}
                    fit_key, predicted = save_fit(features[pool], splits[f"train_{experience}"], metadata,
                                                   orders[f"{experience}_seed{case['seed']}"][:config["epochs"]])
                    for split in ("train", "iid_a", "iid_b", "ood_common"):
                        indices = splits[f"train_{experience}"] if split == "train" else splits[split]
                        truth = np.asarray([records[index]["y_family"] for index in indices], dtype=np.int64)
                        source_metrics.append({"fit_key": fit_key, **metadata, "split": split,
                            **metric(predicted[[feature_lookup[index] for index in indices]], truth)})
        for pool in POOLS:
            for support in plan["direct_support"]:
                metadata = {**base, "experience": "shared", "pool": pool, "classifier": "ridge",
                            "protocol": "direct_diagnostic", "repeat": support["repeat"],
                            "source_position": support["source_position"]}
                fit_key, predicted = save_fit(features[pool], support["indices"], metadata)
                for experience in experiences:
                    own = set(splits[f"iid_{experience}"])
                    common = set(splits["ood_common"])
                    for target_position in range(3):
                        if target_position == support["source_position"]:
                            continue
                        target = [index for index in analysis_indices if records[index]["y_position"] == target_position]
                        if set(target) & set(support["indices"]):
                            raise AssertionError("A direct diagnostic used a target-position training label.")
                        for split, indices in (("target_all", target), ("target_common_ood", [i for i in target if i in common]),
                                               ("target_own_iid", [i for i in target if i in own])):
                            truth = np.asarray([records[index]["y_family"] for index in indices], dtype=np.int64)
                            direct_metrics.append({"fit_key": fit_key, **metadata, "experience": experience,
                                "target_position": target_position, "split": split,
                                **metric(predicted[[analysis_lookup[index] for index in indices]], truth)})
        after = encoder_state_sha256(model.state_dict())
        if before != after:
            raise AssertionError("A diagnostic classifier changed its frozen encoder.")
        encoders.append({**case, "checkpoint": str(case["checkpoint"]), "checkpoint_sha256": sha256(case["checkpoint"]),
                         "encoder_before_sha256": before, "encoder_after_sha256": after,
                         "encoder_bitwise_unchanged": True, "n_features": len(feature_indices),
                         "features": feature_path.relative_to(output).as_posix(), "features_sha256": sha256(feature_path)})
        for filename, rows in (("pooling_encoders.csv", encoders), ("pooling_feature_checks.csv", checks),
                               ("pooling_fit_inventory.csv", fits), ("pooling_source_metrics.csv", source_metrics),
                               ("pooling_direct_metrics.csv", direct_metrics)):
            write_csv(output / filename, rows)
        write_json(output / "pooling_plan.json", plan)
        print(json.dumps({"pooling_encoder_completed": case["model"], "encoders": len(encoders),
                          "classifier_fits": len(fits), "source_metric_rows": len(source_metrics),
                          "direct_metric_rows": len(direct_metrics)}), flush=True)
        del model, features
    return {"encoder_count": len(encoders), "fit_count": len(fits), "source_metric_rows": len(source_metrics),
            "direct_metric_rows": len(direct_metrics), "exact_branch_checks": len(checks),
            "frozen_encoders_unchanged": True}
