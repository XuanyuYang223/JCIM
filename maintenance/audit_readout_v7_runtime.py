"""Replay the frozen v7 audit with only head evaluation moved to recorded CUDA.

The original CPU audit remains untouched. All identity, ridge, scaler, statistics,
CPU feature, CPU first-gradient, and tolerance checks are the frozen checks.
Only the device used by its functional saved-head evaluation changes. Class
predictions remain exact comparisons; no threshold is relaxed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def recorded_device_head_logits(state, values, mean, scale, task):
    # The frozen auditor's transform, matrix operations, and fixed batch size
    # are retained; parameters and evaluation tensors move to recorded CUDA.
    z = ((np.asarray(values, dtype=np.float64) - np.asarray(mean))
         / np.asarray(scale)).astype(np.float32)
    parameters = {name: value.to("cuda") for name, value in state.items()}
    output = []
    for offset in range(0, len(z), 128):
        current = z[offset:offset + 128]
        n = len(current)
        tensor = torch.from_numpy(current[np.arange(128) % n]).to("cuda")
        hidden = F.relu(F.linear(tensor, parameters[f"{task}.0.weight"], parameters[f"{task}.0.bias"]))
        logits = F.linear(hidden, parameters[f"{task}.2.weight"], parameters[f"{task}.2.bias"])
        output.append(logits[:n].cpu().numpy())
    return np.concatenate(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/readout_v7")
    parser.add_argument("--hard", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/skip_v5")
    parser.add_argument("--pooling", type=Path, default=ROOT / "experiments/pooling_v6")
    parser.add_argument("--original-audit", type=Path, default=ROOT / "maintenance/READOUT_V7_AUDIT.json")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/READOUT_V7_RUNTIME_AUDIT.json")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit evidence: {output}")
    if output.is_relative_to(ROOT / "experiments") or output.is_relative_to(args.record.resolve()):
        raise ValueError("Runtime audit evidence must remain outside every scientific record.")
    environment = json.loads((args.record / "environment.json").read_text(encoding="utf-8"))
    if environment["device"] != "cuda" or not torch.cuda.is_available():
        raise ValueError("The recorded CUDA execution device is required.")
    base = ROOT / "maintenance/audit_readout_v7.py"
    base_hash, original_hash = sha256(base), sha256(args.original_audit)
    original = json.loads(args.original_audit.read_text(encoding="utf-8"))
    revision = environment["git_head_at_start"]
    original_blob = subprocess.check_output(["git", "cat-file", "blob",
                                             f"{revision}:maintenance/audit_readout_v7.py"], cwd=ROOT)
    if hashlib.sha256(original_blob).hexdigest() != base_hash or original["auditor_sha256"] != base_hash:
        raise ValueError("The base auditor must be bit-identical to the pre-run committed and first-executed auditor.")
    spec = importlib.util.spec_from_file_location("frozen_readout_v7_auditor", base)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    auditor.independent_head_logits = recorded_device_head_logits
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    result = auditor.audit_record(args.record, args.hard, args.source, args.pooling)
    old_check = "every_saved_MLP_head_prediction_independently_reconstructed_on_CPU"
    result["checks"]["every_saved_MLP_head_prediction_independently_reconstructed_on_recorded_CUDA"] = result["checks"].pop(old_check)
    result["counts"]["MLP_CUDA_prediction_mismatches"] = result["counts"].pop("MLP_CPU_prediction_mismatches")
    result["method"] = ("Frozen auditor replay with one documented runtime device override: saved MLP head evaluation uses independent functional code on the original CUDA device; every other frozen check is unchanged. "
                        + result["method"].replace("every MLP head prediction recomputed", "every MLP head prediction recomputed on recorded CUDA"))
    result["limits"] = result["limits"].replace("CPU features and logits", "CPU encoder features and recorded-CUDA head logits")
    unchanged = sha256(base) == base_hash and sha256(args.original_audit) == original_hash
    result["runtime_override"] = {"wrapper_sha256": sha256(Path(__file__)),
        "base_auditor_sha256": base_hash, "base_auditor_source_commit": revision,
        "overridden_symbol": "independent_head_logits", "original_device": "cpu", "replay_device": "cuda",
        "cuda_device": torch.cuda.get_device_name(), "fixed_batch_size": 128,
        "operation_changes": "Only transfer saved head parameters and transformed evaluation tensors to recorded CUDA before the same two F.linear operations and ReLU.",
        "tolerances_changed": False, "class_predictions_require_exact_match": True,
        "all_other_checks_unchanged": True,
        "original_audit": str(args.original_audit.resolve()), "original_audit_sha256": original_hash,
        "original_audit_status_preserved": original["status"], "base_and_original_audit_unchanged": unchanged}
    if not unchanged:
        result["status"] = "FAIL"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
