"""Frozen-encoder by replay interventions on the inherited molecular models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .analysis import fit_readout, linear_cka
from .hard_model import HardMolecularGNN
from .hard_run import metric_row
from .hard_verify import verify_hard
from .run import ROOT, infer, sha256, take, write_csv, write_json


ARMS = ("frozen_new", "full_new", "frozen_replay", "full_replay", "scratch_replay")


def load_source(source: Path, device: torch.device):
    """Load verified archived arrays directly; never regenerate a scientific input."""
    verify_hard(source)
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    with np.load(source / "graphs.npz") as arrays:
        tensors = {name: torch.as_tensor(arrays[name].copy(), device=device) for name in arrays.files}
    return records, splits, tensors


def checkpoint_state(source: Path, condition: str, seed: int):
    checkpoint = torch.load(source / "checkpoints" / f"{condition}_seed{seed}.pt", map_location="cpu", weights_only=True)
    return checkpoint["state_dict"]


def support_sets(source: Path, records, splits, config):
    lookup = {record["id"]: i for i, record in enumerate(records)}
    result = []
    for repeat in config["support_repeats"]:
        archived = json.loads((source / "adaptation_support" / f"repeat{repeat}_shots32.json").read_text(encoding="utf-8"))
        new_ids = archived["molecule_ids"]
        if len(new_ids) != config["new_count"] or len(set(new_ids)) != len(new_ids) or not {lookup[x] for x in new_ids} <= set(splits["adaptation_pool"]):
            raise ValueError("New support must exactly match the reserved archived 32-molecule sample.")
        for experience in ("a", "b"):
            old_indices = np.random.default_rng(7000 + repeat).permutation(splits[f"train_{experience}"])[:config["old_count"]]
            old_ids = [records[i]["id"] for i in old_indices]
            if len(old_ids) != config["old_count"] or len(set(old_ids)) != len(old_ids):
                raise ValueError("The old training pool cannot supply the declared unique replay budget.")
            if set(new_ids) & set(old_ids):
                raise ValueError("New and old support overlap.")
            result.append({"experience": experience, "repeat": repeat, "new_ids": new_ids, "old_ids": old_ids})
    return {"sets": result, "primary_repeat": config["primary_repeat"],
            "protocol": "Fixed primary support; the other archived new samples are supplemental repetitions."}


def set_trainable(model, frozen: bool):
    """Freeze embedding and every message layer; retain all three MLP task heads."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not frozen or name.startswith("heads."))


def encoder_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith("heads."):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_drift(model, original):
    current = model.state_dict()
    encoder = [name for name in original if not name.startswith("heads.")]
    heads = [name for name in original if name.startswith("heads.")]
    encoder_delta = max(float((current[name].detach().cpu() - original[name]).abs().max()) for name in encoder)
    head_delta = max(float((current[name].detach().cpu() - original[name]).abs().max()) for name in heads)
    before_hash, after_hash = encoder_state_sha256(original), encoder_state_sha256(current)
    return {"encoder_changed": before_hash != after_hash, "encoder_max_abs_delta": encoder_delta,
            "head_max_abs_delta": head_delta, "encoder_original_sha256": before_hash,
            "encoder_current_sha256": after_hash, "encoder_bitwise_unchanged": before_hash == after_hash}


def mix_components(original, adapted, composition):
    """Swap weights as a functional intervention, without changing either input state."""
    encoder_version, head_version = composition.split("_")
    return {name: (adapted if (head_version == "H1" if name.startswith("heads.") else encoder_version == "E1")
                   else original)[name].clone() for name in original}


def evaluate(model, tensors, records, splits, lookup, config, metadata):
    predicted, _ = infer(model, tensors, splits["analysis"], config["batch_size"])
    rows = []
    for split in ("iid_a", "iid_b", "ood_common"):
        indices = splits[split]
        selected = [lookup[i] for i in indices]
        row = metric_row("mechanism", metadata["experience"], metadata["seed"], split,
                         {task: values[selected] for task, values in predicted.items()}, indices, records)
        rows.append({**metadata, **row})
    return rows


def ridge_diagnostics(model, tensors, records, splits, supports, config, experience, seed):
    """Cheap fresh linear-readout controls for limits of inherited-head optimization."""
    indices = sorted(set(splits["analysis"] + splits["adaptation_pool"] + splits[f"train_{experience}"]))
    lookup = {index: row for row, index in enumerate(indices)}
    id_lookup = {record["id"]: i for i, record in enumerate(records)}
    _, reps = infer(model, tensors, indices, config["batch_size"], representations=True)
    rows = []
    for support in supports:
        if support["experience"] != experience:
            continue
        for support_kind in ("new", "replay"):
            ids = support["new_ids"] + (support["old_ids"] if support_kind == "replay" else [])
            fit_indices = [id_lookup[identifier] for identifier in ids]
            labels = np.array([records[i]["y_family"] for i in fit_indices])
            for readout in ("local", "graph"):
                features = reps[readout][-1]
                for split in ("iid_a", "iid_b", "ood_common"):
                    test = splits[split]
                    fitted = fit_readout(features[[lookup[i] for i in fit_indices]], labels,
                                         features[[lookup[i] for i in test]],
                                         np.array([records[i]["y_family"] for i in test]), alpha=config["probe_alpha"])
                    rows.append({"experience": experience, "seed": seed, "repeat": support["repeat"],
                                 "readout": readout, "support": support_kind, "split": split,
                                 "n_fit": len(fit_indices), "n_test": len(test),
                                 "motif_accuracy": fitted["accuracy"],
                                 "motif_balanced_accuracy": fitted["balanced_accuracy"]})
    return rows


def run_case(model, original, tensors, records, splits, id_lookup, eval_lookup, diagnostic_indices,
             support, arm, experience, seed, config, output):
    frozen = arm.startswith("frozen_")
    set_trainable(model, frozen)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=config["adaptation_learning_rate"], weight_decay=config["weight_decay"])
    train_ids = support["new_ids"] + (support["old_ids"] if arm.endswith("replay") else [])
    train_indices = [id_lookup[identifier] for identifier in train_ids]
    rng = np.random.default_rng(91000 + support["repeat"])
    metadata = {"experience": experience, "seed": seed, "repeat": support["repeat"], "arm": arm}
    trajectory, drift = [], []
    _, reference = infer(model, tensors, diagnostic_indices, config["batch_size"], representations=True)
    for step in range(config["mechanism_steps"] + 1):
        if step in config["evaluation_steps"]:
            trajectory.extend(evaluate(model, tensors, records, splits, eval_lookup, config, {**metadata, "step": step}))
            change = parameter_drift(model, original)
            if frozen and change["encoder_changed"]:
                raise AssertionError("A frozen encoder changed during adaptation.")
            _, current = infer(model, tensors, diagnostic_indices, config["batch_size"], representations=True)
            drift.append({**metadata, "step": step, **change,
                          "local_cka": linear_cka(reference["local"][-1], current["local"][-1]),
                          "graph_cka": linear_cka(reference["graph"][-1], current["graph"][-1])})
        if step == config["mechanism_steps"]:
            break
        model.train()
        # Both arms use all available examples per update: 32 new or 32+32.
        # Replaying groups have equal weight; optimizer-step count is held fixed.
        chosen = rng.choice(train_indices, size=len(train_indices), replace=False)
        batch = take(tensors, chosen)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = sum(F.cross_entropy(logits[task], batch[f"y_{task}"])
                   for task in ("family", "position", "distance"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
        optimizer.step()
    adapted = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"state_dict": adapted, **metadata, "config": config, "support": support},
               output / "checkpoints" / f"{experience}_s{seed}_r{support['repeat']}_{arm}.pt")
    components = []
    if arm != "scratch_replay":
        for composition in ("E0_H0", "E0_H1", "E1_H0", "E1_H1"):
            hybrid = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(tensors["x"].device)
            hybrid.load_state_dict(mix_components(original, adapted, composition))
            components.extend(evaluate(hybrid, tensors, records, splits, eval_lookup, config,
                                       {**metadata, "composition": composition}))
            del hybrid
    return trajectory, drift, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mechanism_v3.json")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/readout_replay_v3")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.steps is not None:
        config["mechanism_steps"] = args.steps
        config["evaluation_steps"] = sorted(set([0, args.steps] + [x for x in config["evaluation_steps"] if x <= args.steps]))
    for name, key in (("seeds", "seeds"), ("repeats", "support_repeats"), ("device", "device")):
        if getattr(args, name) is not None:
            config[key] = getattr(args, name)
    if config["primary_repeat"] not in config["support_repeats"]:
        raise ValueError("The fixed primary support repetition must be included.")
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_name = config["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    records, splits, tensors = load_source(source, device)
    supports = support_sets(source, records, splits, config)
    write_json(output / "config.json", config)
    write_json(output / "support_sets.json", supports)
    id_lookup = {record["id"]: i for i, record in enumerate(records)}
    eval_lookup = {index: row for row, index in enumerate(splits["analysis"])}
    diagnostic_indices = sorted(splits["analysis"], key=lambda i: records[i]["id"])[:config["diagnostic_examples"]]
    write_json(output / "diagnostic_ids.json", [records[i]["id"] for i in diagnostic_indices])
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rdkit", "matplotlib")},
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_directory": str(source), "source_manifest_sha256": sha256(source / "MANIFEST.json"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "jcim").glob("*.py"))},
        "pretrained_checkpoint_sha256": {f"{condition}_seed{seed}": sha256(source / "checkpoints" / f"{condition}_seed{seed}.pt")
            for condition in ("a", "b", "random") for seed in config["seeds"]},
        "training_protocol": "Full batches, identical update count; replay uses 32+32 labels; no checkpoint selection.",
        "frozen_scope": "Input embedding and every message layer; all three inherited MLP heads remain trainable.",
        "primary_repeat": config["primary_repeat"]}
    write_json(output / "environment.json", environment)
    trajectory, drift, components, ridge = [], [], [], []
    for seed in config["seeds"]:
        for experience in ("a", "b"):
            pretrained = checkpoint_state(source, experience, seed)
            random_state = checkpoint_state(source, "random", seed)
            original_model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
            original_model.load_state_dict(pretrained)
            ridge.extend(ridge_diagnostics(original_model, tensors, records, splits, supports["sets"], config, experience, seed))
            del original_model
            for support in supports["sets"]:
                if support["experience"] != experience:
                    continue
                for arm in ARMS:
                    state = random_state if arm == "scratch_replay" else pretrained
                    model = HardMolecularGNN(config["hidden_dim"], config["layers"]).to(device)
                    model.load_state_dict(state)
                    case_rows, case_drift, case_components = run_case(model, state, tensors, records, splits,
                        id_lookup, eval_lookup, diagnostic_indices, support, arm, experience, seed, config, output)
                    trajectory.extend(case_rows)
                    drift.extend(case_drift)
                    components.extend(case_components)
                    print(json.dumps({"experience": experience, "seed": seed, "repeat": support["repeat"],
                                      "arm": arm, "final": [r for r in case_rows if r["step"] == config["mechanism_steps"]]}), flush=True)
                    del model
                write_csv(output / "trajectory.csv", trajectory)
                write_csv(output / "drift.csv", drift)
                write_csv(output / "component_eval.csv", components)
                write_csv(output / "ridge_diagnostic.csv", ridge)
    if sha256(source / "MANIFEST.json") != environment["source_manifest_sha256"]:
        raise AssertionError("The original experiment manifest changed.")
    verify_hard(source)
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", environment)
    from .mechanism_report import generate_mechanism_report
    generate_mechanism_report(output)
    write_json(output / "MANIFEST.json", {"sha256": {str(path.relative_to(output)): sha256(path)
               for path in sorted(output.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}})
    print(f"Completed scientific record: {output}", flush=True)


if __name__ == "__main__":
    main()
