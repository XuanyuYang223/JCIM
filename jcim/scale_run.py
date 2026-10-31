"""Matched frozen molecular readouts across post-hoc pooling scales."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Small deterministic ridge solves do not benefit from many BLAS workers.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from .analysis import balanced_accuracy, fit_readout
from .hard_model import HardMolecularGNN
from .mechanism import checkpoint_state, encoder_state_sha256
from .run import ROOT, sha256, write_csv, write_json
from .scale_features import REGION_NAMES, build_scale_regions, extract_scale_features


def relative_file(directory, relative):
    return Path(directory).joinpath(*relative.replace("\\", "/").split("/"))


def verify_record(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
    for relative, expected in manifest.items():
        path = relative_file(directory, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Archived input checksum mismatch: {path}")
    return {"path": str(directory), "manifest_sha256": sha256(manifest_path), "files": len(manifest)}


def validate_config(config):
    for name in ("seeds", "support_repeats", "layers"):
        if not config[name] or len(set(config[name])) != len(config[name]):
            raise ValueError(f"{name} must contain unique values.")
    if not set(config["seeds"]) <= {0, 1, 2} or not set(config["support_repeats"]) <= {0, 1, 2}:
        raise ValueError("Only the archived encoder seeds and support samples are available.")
    if not set(config["layers"]) <= set(range(9)) or config["primary_layer"] not in config["layers"]:
        raise ValueError("Depths must belong to 0 through 8 and include the primary depth.")
    if config["primary_repeat"] not in config["support_repeats"]:
        raise ValueError("The primary support sample must be retained.")
    if config["hidden_dim"] != 64 or config["encoder_layers"] != 8:
        raise ValueError("The frozen hard-v2 architecture is fixed at width 64 and eight layers.")
    if config["alpha"] <= 0 or not np.isfinite(config["alpha"]):
        raise ValueError("The fixed regularization must be positive and finite.")
    if config["direct_per_class"] < 1:
        raise ValueError("Source-position fits require a positive per-class support count.")


def build_direct_support(records, splits, config):
    result = []
    for repeat in config["support_repeats"]:
        for source_position in range(3):
            sampling_seed = config["direct_sampling_base"] + 100 * repeat + source_position
            rng = np.random.default_rng(sampling_seed)
            chosen = []
            for motif in range(8):
                eligible = sorted(
                    (index for index in splits["analysis"]
                     if records[index]["y_position"] == source_position
                     and records[index]["y_family"] == motif),
                    key=lambda index: records[index]["id"])
                if len(eligible) < config["direct_per_class"]:
                    raise ValueError("The analysis source position cannot supply the balanced label budget.")
                chosen.extend(rng.choice(eligible, size=config["direct_per_class"], replace=False).tolist())
            chosen.sort(key=lambda index: records[index]["id"])
            result.append({"repeat": repeat, "source_position": source_position,
                           "sampling_seed": sampling_seed, "indices": chosen,
                           "ids": [records[index]["id"] for index in chosen]})
    return {"protocol": "Balanced source-position labels; zero target-position labels in each directed comparison.",
            "sets": result}


def validate_support_sets(records, splits, supports, config):
    lookup = {row["id"]: index for index, row in enumerate(records)}
    selected = []
    keys = set()
    for sample in supports["sets"]:
        if sample["repeat"] not in config["support_repeats"]:
            continue
        experience = sample["experience"]
        key = (experience, sample["repeat"])
        if key in keys or experience not in ("a", "b"):
            raise ValueError("An adaptation support key is duplicated or invalid.")
        keys.add(key)
        if len(sample["new_ids"]) != 32 or len(sample["old_ids"]) != 32:
            raise ValueError("The exact v3 label budgets are required.")
        new = [lookup[identifier] for identifier in sample["new_ids"]]
        old = [lookup[identifier] for identifier in sample["old_ids"]]
        if len(set(new + old)) != 64 or not set(new) <= set(splits["adaptation_pool"]):
            raise ValueError("Adaptation support violates reserved identity isolation.")
        if not set(old) <= set(splits[f"train_{experience}"]):
            raise ValueError("Replay support must belong to the original experience.")
        if set(new + old) & set(splits["analysis"]):
            raise ValueError("Adaptation support and analysis tests overlap.")
        selected.append(sample)
    if keys != {(experience, repeat) for experience in ("a", "b") for repeat in config["support_repeats"]}:
        raise ValueError("An experience or adaptation support draw is missing.")
    return selected


def state_sha256(state):
    checksum = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        checksum.update(name.encode("utf-8"))
        checksum.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def id_digest(identifiers):
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def score_row(predictions, indices, records, analysis_lookup, metadata):
    if not indices:
        raise ValueError("An evaluation stratum is empty.")
    predicted = predictions[[analysis_lookup[index] for index in indices]]
    truth = np.asarray([records[index]["y_family"] for index in indices], dtype=np.int64)
    return {**metadata, "n_test": len(indices), "accuracy": float(np.mean(predicted == truth)),
            "balanced_accuracy": balanced_accuracy(truth, predicted),
            "test_class_counts": json.dumps({label: int(np.sum(truth == label)) for label in range(8)}, sort_keys=True)}


def evaluate_model(model_name, features, records, splits, supports, direct, feature_indices, config,
                   prediction_arrays, fit_rows):
    """Fit once per unique support and predict the fixed ordered analysis pool."""
    seed = int(model_name.rsplit("seed", 1)[1])
    source_condition = model_name.split("_", 1)[0]
    experiences = ("a", "b") if source_condition == "random" else (source_condition,)
    kind = "random" if source_condition == "random" else "trained"
    feature_lookup = {index: row for row, index in enumerate(feature_indices)}
    analysis_lookup = {index: row for row, index in enumerate(splits["analysis"])}
    identity_lookup = {record["id"]: index for index, record in enumerate(records)}
    ood_set = set(splits["ood_common"])
    own_sets = {experience: set(splits[f"iid_{experience}"]) for experience in ("a", "b")}
    analysis_rows = [feature_lookup[index] for index in splits["analysis"]]
    truth = np.asarray([records[index]["y_family"] for index in splits["analysis"]], dtype=np.int64)
    cache = {}
    adaptation_rows, direct_rows = [], []

    def fitted(scope, layer, fit_indices, descriptor):
        key = (scope, layer, tuple(fit_indices))
        if key not in cache:
            layer_row = config["layers"].index(layer)
            values = features[scope][layer_row]
            fit = fit_readout(values[[feature_lookup[index] for index in fit_indices]],
                              np.asarray([records[index]["y_family"] for index in fit_indices], dtype=np.int64),
                              values[analysis_rows], truth, alpha=config["alpha"])
            predicted = np.asarray(fit["predictions"], dtype=np.int8)
            prediction_key = f"readout_{len(prediction_arrays):06d}"
            prediction_arrays[prediction_key] = predicted
            fit_rows.append({"prediction_key": prediction_key, "model": model_name, "seed": seed,
                             "scale": scope, "layer": layer, "feature_dim": values.shape[-1],
                             "n_fit": len(fit_indices), **descriptor,
                             "support_ids_sha256": id_digest([records[index]["id"] for index in fit_indices]),
                             "predictions_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
                             "fit_class_counts": json.dumps({label: sum(records[index]["y_family"] == label
                                 for index in fit_indices) for label in range(8)}, sort_keys=True),
                             "classes_seen": json.dumps(fit["classes_seen"])})
            cache[key] = (prediction_key, predicted)
        return cache[key]

    for layer in config["layers"]:
        for scale in REGION_NAMES:
            base = {"condition": kind, "model": model_name, "seed": seed,
                    "layer": layer, "scale": scale, "feature_dim": features[scale].shape[-1]}
            for experience in experiences:
                for sample in supports:
                    if sample["experience"] != experience:
                        continue
                    for support_kind in ("new", "replay"):
                        fit_ids = sample["new_ids"] + (sample["old_ids"] if support_kind == "replay" else [])
                        fit_indices = [identity_lookup[identifier] for identifier in fit_ids]
                        key, predicted = fitted(scale, layer, fit_indices,
                            {"kind": "adaptation", "experience": experience, "repeat": sample["repeat"],
                             "support": support_kind, "source_position": -1})
                        metadata = {**base, "experience": experience, "repeat": sample["repeat"],
                                    "support": support_kind, "n_fit": len(fit_indices), "prediction_key": key}
                        for split in ("iid_a", "iid_b", "ood_common"):
                            adaptation_rows.append(score_row(predicted, splits[split], records, analysis_lookup,
                                                             {**metadata, "split": split}))
                for sample in direct["sets"]:
                    fit_indices = sample["indices"]
                    key, predicted = fitted(scale, layer, fit_indices,
                        {"kind": "direct", "experience": "shared", "repeat": sample["repeat"],
                         "support": "source_position", "source_position": sample["source_position"]})
                    metadata = {**base, "experience": experience, "repeat": sample["repeat"],
                                "source_position": sample["source_position"], "n_fit": len(fit_indices),
                                "prediction_key": key}
                    for target_position in range(3):
                        if target_position == sample["source_position"]:
                            continue
                        target = [index for index in splits["analysis"]
                                  if records[index]["y_position"] == target_position]
                        if set(target) & set(fit_indices):
                            raise AssertionError("A direct readout saw a target-position identity.")
                        strata = {"target_all": target,
                                  "target_common_ood": [index for index in target if index in ood_set],
                                  "target_own_iid": [index for index in target if index in own_sets[experience]]}
                        for split, test in strata.items():
                            direct_rows.append(score_row(predicted, test, records, analysis_lookup,
                                {**metadata, "target_position": target_position, "split": split}))
    return adaptation_rows, direct_rows


def run_scale(source, mechanism_source, output, config):
    validate_config(config)
    source, mechanism_source, output = Path(source).resolve(), Path(mechanism_source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    inputs = {"hard_v2": source, "readout_replay_v3": mechanism_source}
    for name in ("pilot_v1", "mechanism_diagnostics_v3", "correlation_v3"):
        directory = source.parent / name
        if directory.is_dir():
            inputs[name] = directory
    references = {name: verify_record(directory) for name, directory in inputs.items()}
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (source_config["hidden_dim"], source_config["layers"]) != (config["hidden_dim"], config["encoder_layers"]):
        raise ValueError("Architecture differs from the archived checkpoints.")
    supports = json.loads((mechanism_source / "support_sets.json").read_text(encoding="utf-8"))
    samples = validate_support_sets(records, splits, supports, config)
    direct = build_direct_support(records, splits, config)
    identity_lookup = {record["id"]: index for index, record in enumerate(records)}
    feature_indices = sorted(set(splits["analysis"]) | {
        identity_lookup[identifier] for sample in samples for identifier in sample["new_ids"] + sample["old_ids"]})
    with np.load(source / "graphs.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    masks, locality = build_scale_regions(records, arrays["adj"], arrays["mask"], feature_indices, config["layers"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "representations").mkdir()
    write_json(output / "config.json", config)
    write_json(output / "source_reference.json", {"records": references,
        "hard_source": str(source), "mechanism_source": str(mechanism_source),
        "support_sets_sha256": sha256(mechanism_source / "support_sets.json")})
    # Preserve the exact existing support record, including draws omitted by a smoke run.
    (output / "support_sets.json").write_bytes((mechanism_source / "support_sets.json").read_bytes())
    write_json(output / "direct_support.json", direct)
    write_json(output / "feature_indices.json", {"indices": feature_indices,
        "ids": [records[index]["id"] for index in feature_indices],
        "analysis_indices": splits["analysis"], "analysis_ids": [records[index]["id"] for index in splits["analysis"]]})
    np.savez_compressed(output / "region_masks.npz", **masks)
    write_csv(output / "locality.csv", locality)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    tensors = {name: torch.as_tensor(value, device=device) for name, value in arrays.items()}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "python": sys.version, "platform": platform.platform(), "git_head_at_start": head.stdout.strip(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "rdkit", "matplotlib")},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((ROOT / "jcim").glob("*.py"))},
        "input_checkpoint_sha256": {}, "models": {}, "encoder_updates": 0,
        "normalization": "Existing support-only float64 standardization; std <= float64 epsilon uses scale 1.",
        "readout": "Existing deterministic eight-output standardized ridge, alpha fixed before evaluation.",
        "primary_layer": config["primary_layer"], "primary_repeat": config["primary_repeat"],
        "inference_batch_padding": "Repeat valid graphs within the final short batch to the fixed configured graph count; discard repeated outputs. No cross-graph model operations.",
        "pooling_precision": "Non-root 64-D means accumulate float64 and return float32; original root and legacy operations retained."}
    write_json(output / "environment.json", environment)
    all_adaptation, all_direct, fits, predictions, constant_controls = [], [], [], {}, []
    for seed in config["seeds"]:
        for condition in ("a", "b", "random"):
            name = f"{condition}_seed{seed}"
            checkpoint = source / "checkpoints" / f"{name}.pt"
            environment["input_checkpoint_sha256"][name] = sha256(checkpoint)
            state = checkpoint_state(source, condition, seed)
            model = HardMolecularGNN(config["hidden_dim"], config["encoder_layers"]).to(device)
            model.load_state_dict(state)
            model.requires_grad_(False)
            model.eval()
            before, encoder_before = state_sha256(model.state_dict()), encoder_state_sha256(model.state_dict())
            features = extract_scale_features(model, tensors, feature_indices, masks, config["layers"], config["batch_size"])
            if 0 in config["layers"]:
                depth_zero = config["layers"].index(0)
                for scope in ("root", "target"):
                    values = features[scope][depth_zero]
                    unique_count = len(np.unique(values, axis=0))
                    span = float(np.ptp(values, axis=0).max())
                    constant_controls.append({"model": name, "scale": scope, "layer": 0,
                                              "n": len(values), "unique_vectors": unique_count,
                                              "maximum_coordinate_range": span})
                    if unique_count != 1 or span != 0:
                        raise AssertionError("A constant-input layer-zero control contains a numerical fingerprint.")
            after, encoder_after = state_sha256(model.state_dict()), encoder_state_sha256(model.state_dict())
            if before != after or encoder_before != encoder_after:
                raise AssertionError("A frozen encoder changed during feature extraction.")
            environment["models"][name] = {"state_before_sha256": before, "state_after_sha256": after,
                "encoder_before_sha256": encoder_before, "encoder_after_sha256": encoder_after,
                "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
                "feature_shapes": {scope: list(value.shape) for scope, value in features.items()}}
            np.savez_compressed(output / "representations" / f"{name}.npz",
                                indices=np.asarray(feature_indices), ids=np.asarray([records[index]["id"] for index in feature_indices]),
                                layers=np.asarray(config["layers"]), **features)
            adaptation_rows, direct_rows = evaluate_model(name, features, records, splits, samples, direct,
                                                         feature_indices, config, predictions, fits)
            all_adaptation.extend(adaptation_rows)
            all_direct.extend(direct_rows)
            write_csv(output / "adaptation.csv", all_adaptation)
            write_csv(output / "direct_transfer.csv", all_direct)
            write_csv(output / "fits.csv", fits)
            if constant_controls:
                write_csv(output / "constant_controls.csv", constant_controls)
            write_json(output / "environment.json", environment)
            print(json.dumps({"completed": name, "adaptation_rows": len(all_adaptation),
                              "direct_rows": len(all_direct), "unique_readout_fits": len(fits)}), flush=True)
            del model, features
    np.savez_compressed(output / "predictions.npz", **predictions)
    if {name: verify_record(directory) for name, directory in inputs.items()} != references:
        raise AssertionError("A previous scientific record changed.")
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    environment["previous_records_unchanged"] = True
    write_json(output / "environment.json", environment)
    from .scale_report import generate_scale_report
    generate_scale_report(output)
    manifest = {path.relative_to(output).as_posix(): sha256(path)
                for path in sorted(output.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}
    write_json(output / "MANIFEST.json", {"sha256": manifest})
    print(json.dumps({"completed": str(output), "manifest_files": len(manifest),
                      "frozen_models": len(environment["models"]), "unique_readout_fits": len(fits)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scale_v4.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mechanism-source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/scale_v4")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for argument, key in (("seeds", "seeds"), ("repeats", "support_repeats"), ("layers", "layers"), ("device", "device")):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    source = args.source or ROOT / config["source"]
    mechanism_source = args.mechanism_source or ROOT / config["mechanism_source"]
    run_scale(source, mechanism_source, args.output, config)


if __name__ == "__main__":
    main()
