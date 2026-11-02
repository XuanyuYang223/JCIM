"""Archive a frozen-encoder readout factorial and matched random controls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from .pooling_run import PRIOR_RECORDS, validate_output
from .run import ROOT, sha256, write_json
from .scale_run import verify_record


def validate_config(config):
    expected = {"experiences": ["a", "b"], "regimes": ["end_to_end", "frozen_readout"],
                "scopes": ["full512", "shallow_only"], "normalizations": ["raw", "train_z"],
                "objectives": ["family", "joint"], "hidden_dim": 64, "layers": 8,
                "shallow_layer": 1, "batch_size": 128, "learning_rate": .0015,
                "weight_decay": .0001, "gradient_clip": 5., "order_seed_offset": 1701, "alpha": 1.}
    for field, value in expected.items():
        if config[field] != value:
            raise ValueError(f"The predeclared readout protocol requires {field}={value}.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or not set(config["seeds"]) <= {0, 1, 2}:
        raise ValueError("Select distinct archived initialization seeds.")
    if isinstance(config["epochs"], bool) or not isinstance(config["epochs"], int) or not 1 <= config["epochs"] <= 120:
        raise ValueError("Epochs must be an integer from 1 through 120.")
    if config["device"] not in ("auto", "cuda", "cpu") or config["cpu_threads"] < 1:
        raise ValueError("A supported device and positive CPU thread count are required.")


def run_readout(hard_source, skip_source, pooling_source, output, config):
    validate_config(config)
    hard_source, skip_source, pooling_source = (Path(path).resolve() for path in
                                              (hard_source, skip_source, pooling_source))
    previous = {name: hard_source.parent / name for name in (*PRIOR_RECORDS, "pooling_v6")}
    output = validate_output(output, [*previous.values(), hard_source, skip_source, pooling_source])
    references = {name: verify_record(path) for name, path in previous.items()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "source_reference.json", {"records": references,
        "hard_source": str(hard_source), "skip_source": str(skip_source), "pooling_source": str(pooling_source)})
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rdkit", "matplotlib")},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((ROOT / "jcim").glob("*.py"))},
        "protocol_sha256": sha256(ROOT / "configs/READOUT_V7_PROTOCOL.md"),
        "blas_threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "encoder_updates": 0, "target_membership_supplied": False, "new_combination_labels": 0,
        "objective_control": "Independent frozen task heads; other losses affect family updates only through global gradient clipping and numerical execution.",
        "factorial_scope": "Conditional readout effects under fixed archived initialization; not a unique decomposition of the v5-to-v6 difference.",
        "pipeline_only": config["epochs"] != 120 or config["seeds"] != [0, 1, 2]}
    write_json(output / "environment.json", environment)
    from .readout_factorial import run_readout_factorial
    from .readout_report import generate_readout_report
    environment["panel"] = run_readout_factorial(hard_source, skip_source, pooling_source, output / "panel", config)
    if {name: verify_record(path) for name, path in previous.items()} != references:
        raise AssertionError("A completed scientific record changed.")
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    environment["previous_records_unchanged"] = True
    write_json(output / "environment.json", environment)
    generate_readout_report(output)
    files = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "MANIFEST.json"}
    write_json(output / "MANIFEST.json", {"sha256": files})
    print(json.dumps({"completed": str(output), "manifest_files": len(files)}), flush=True)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/readout_v7.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/readout_v7")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "seeds", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run_readout(*(ROOT / config[key] for key in ("hard_source", "skip_source", "pooling_source")),
                args.output, config)


if __name__ == "__main__":
    main()
