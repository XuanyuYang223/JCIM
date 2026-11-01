"""Frozen-model diagnostics of branch outputs, spatial pooling, and readout."""

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

# Match the deterministic, fixed regularization solves in the scale experiment.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from .run import ROOT, sha256, write_json
from .scale_run import verify_record


PRIOR_RECORDS = ("pilot_v1", "hard_v2", "mechanism_diagnostics_v3",
                 "readout_replay_v3", "correlation_v3", "scale_v4", "skip_v5")


def validate_config(config):
    """Reject a changed label, spatial, or optimization protocol."""
    for name, allowed in (("seeds", (0, 1, 2)), ("experiences", ("a", "b")),
                          ("regimes", ("end_to_end", "frozen_readout")),
                          ("support_repeats", (0, 1, 2))):
        values = config[name]
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f"Invalid or duplicate {name}.")
    fixed = {"hidden_dim": 64, "layers": 8, "shallow_layer": 1, "batch_size": 128,
             "learning_rate": 0.0015, "weight_decay": 0.0001, "gradient_clip": 5.0,
             "alpha": 1.0, "mlp_seed_offset": 61000}
    for name, expected in fixed.items():
        if config[name] != expected:
            raise ValueError(f"The fixed diagnostic protocol requires {name}={expected}.")
    if isinstance(config["epochs"], bool) or not isinstance(config["epochs"], int) or not 1 <= config["epochs"] <= 120:
        raise ValueError("Epochs must be a positive integer at most 120.")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Unknown device.")
    if not isinstance(config["cpu_threads"], int) or config["cpu_threads"] < 1:
        raise ValueError("A positive CPU thread count is required.")


def validate_output(output, source_paths):
    output = Path(output).resolve()
    for source in source_paths:
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("A diagnostic output must be separate from every archived source.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a scientific record: {output}")
    return output


def run_pooling(hard_source, skip_source, scale_source, output, config):
    validate_config(config)
    hard_source, skip_source, scale_source = (Path(path).resolve() for path in
                                            (hard_source, skip_source, scale_source))
    previous = {name: hard_source.parent / name for name in PRIOR_RECORDS}
    output = validate_output(output, [*previous.values(), skip_source, scale_source])
    references = {name: verify_record(path) for name, path in previous.items()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "source_reference.json", {"records": references,
        "hard_source": str(hard_source), "skip_source": str(skip_source),
        "scale_source": str(scale_source)})
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pooling_device = config["device"]
    if pooling_device == "auto":
        pooling_device = "cuda" if torch.cuda.is_available() else "cpu"
    environment = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "branch_device": "cpu", "pooling_device": pooling_device,
        "gpu": torch.cuda.get_device_name() if pooling_device == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rdkit", "matplotlib")},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                          for path in sorted((ROOT / "jcim").glob("*.py"))},
        "protocol_sha256": sha256(ROOT / "configs/POOLING_V6_PROTOCOL.md"),
        "blas_threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "encoder_updates": 0,
        "oracle_target_membership": "Supplied only to diagnostic target pooling, never encoder input.",
        "source_only_labels": "Original train identities and motif labels; no target-combination or adaptation labels.",
        "direct_probe_labels": "Archived v4 balanced source-position labels; a distinct supervised diagnostic protocol.",
        "pipeline_only": not (config["epochs"] == 120 and config["seeds"] == [0, 1, 2]
                              and config["experiences"] == ["a", "b"]
                              and config["regimes"] == ["end_to_end", "frozen_readout"]
                              and config["support_repeats"] == [0, 1, 2])}
    write_json(output / "environment.json", environment)
    from .branch_diagnostics import run_branch_diagnostics
    from .pooling_diagnostics import run_pooling_diagnostics
    from .pooling_report import generate_pooling_report
    branch_result = run_branch_diagnostics(skip_source, hard_source, output / "branch", config)
    environment["branch"] = {key: value for key, value in branch_result.items()
                             if key not in ("summaries", "seed_aggregate")}
    environment["pooling"] = run_pooling_diagnostics(skip_source, hard_source, scale_source,
                                                    output / "pooling", config)
    if {name: verify_record(path) for name, path in previous.items()} != references:
        raise AssertionError("A previous scientific record changed.")
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    environment["previous_records_unchanged"] = True
    write_json(output / "environment.json", environment)
    generate_pooling_report(output)
    manifest = {path.relative_to(output).as_posix(): sha256(path)
                for path in sorted(output.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}
    write_json(output / "MANIFEST.json", {"sha256": manifest})
    print(json.dumps({"completed": str(output), "manifest_files": len(manifest)}), flush=True)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pooling_v6.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/pooling_v6")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "seeds", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run_pooling(*(ROOT / config[key] for key in ("hard_source", "skip_source", "scale_source")),
                args.output, config)


if __name__ == "__main__":
    main()
