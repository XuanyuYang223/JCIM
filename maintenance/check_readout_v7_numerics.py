"""Explain the preserved CPU auxiliary-head tie without modifying v7 evidence.

This is a post-run numerical check, not a replacement for the frozen auditor.
It evaluates every saved MLP head using independent functional code on both CPU
and the recorded CUDA device, retaining every class discrepancy and logit delta.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@torch.no_grad()
def independent_logits(state, values, task, device):
    parameters = {name: value.to(device) for name, value in state.items() if name.startswith(task + ".")}
    output = []
    for offset in range(0, len(values), 128):
        current = values[offset:offset + 128]
        count = len(current)
        tensor = torch.as_tensor(current[np.arange(128) % count], device=device)
        first = F.linear(tensor, parameters[f"{task}.0.weight"], parameters[f"{task}.0.bias"])
        logits = F.linear(F.relu(first), parameters[f"{task}.2.weight"], parameters[f"{task}.2.bias"])
        output.append(logits[:count].cpu().numpy())
    return np.concatenate(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/readout_v7")
    parser.add_argument("--audit", type=Path, default=ROOT / "maintenance/READOUT_V7_AUDIT.json")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/READOUT_V7_NUMERICAL_CHECK.json")
    args = parser.parse_args(argv)
    record, output = args.record.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite numerical evidence: {output}")
    if output.is_relative_to(record) or output.is_relative_to(ROOT / "experiments"):
        raise ValueError("Numerical evidence must remain outside completed experiment records.")
    audit_hash, manifest_hash = sha256(args.audit), sha256(record / "MANIFEST.json")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    environment = json.loads((record / "environment.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available() or environment["device"] != "cuda":
        raise ValueError("The recorded CUDA execution device is required.")
    torch.set_num_threads(environment["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    panel = record / "panel"
    with (panel / "fit_inventory.csv").open(encoding="utf-8", newline="") as stream:
        fits = [row for row in csv.DictReader(stream) if row["classifier"] == "mlp"]
    split_path = ROOT / "experiments/hard_v2/splits.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    counts = {device: {"class_comparisons": 0, "class_mismatches": 0,
                       "trained_task_class_mismatches": 0, "family_class_mismatches": 0,
                       "head_arrays_exact": 0, "max_abs_logit_delta": 0.0}
              for device in ("cpu", "cuda")}
    mismatches, features, files = [], {}, {}
    for row in fits:
        model = row["model"]
        feature_path = panel / "representations" / f"{model}.npz"
        if model not in features:
            with np.load(feature_path) as archive:
                features[model] = archive["readout"].copy()
            files[feature_path.relative_to(ROOT).as_posix()] = sha256(feature_path)
        values = features[model].copy()
        if row["scope"] == "shallow_only":
            values[:, :448] = 0
        checkpoint_path, prediction_path = panel / row["checkpoint"], panel / row["predictions"]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        values = ((values.astype(np.float64) - state["mean"].numpy()) / state["scale"].numpy()).astype(np.float32)
        files[checkpoint_path.relative_to(ROOT).as_posix()] = sha256(checkpoint_path)
        files[prediction_path.relative_to(ROOT).as_posix()] = sha256(prediction_path)
        with np.load(prediction_path) as archive:
            for task in ("family", "position", "distance"):
                saved, saved_pred = archive[f"logits_{task}"], archive[f"pred_{task}"]
                trained = task == "family" or row["objective"] == "joint"
                for device in ("cpu", "cuda"):
                    actual = independent_logits(state["state_dict"], values, task, device)
                    prediction = actual.argmax(1)
                    changed = np.flatnonzero(prediction != saved_pred)
                    summary = counts[device]
                    summary["class_comparisons"] += len(values)
                    summary["class_mismatches"] += len(changed)
                    summary["trained_task_class_mismatches"] += len(changed) if trained else 0
                    summary["family_class_mismatches"] += len(changed) if task == "family" else 0
                    summary["head_arrays_exact"] += int(np.array_equal(actual, saved))
                    summary["max_abs_logit_delta"] = max(summary["max_abs_logit_delta"], float(np.max(np.abs(actual - saved))))
                    for position in changed:
                        index = int(archive["indices"][position])
                        candidates = sorted(set([int(prediction[position]), int(saved_pred[position])]))
                        mismatches.append({"device": device, "fit_key": row["fit_key"], "model": model,
                            "scope": row["scope"], "normalization": row["normalization"], "objective": row["objective"],
                            "task": task, "task_trained": trained, "index": index, "id": str(archive["ids"][position]),
                            "source_training_identity": index in splits[f"train_{row['experience']}"],
                            "test_identity": index in splits["analysis"], "actual_prediction": int(prediction[position]),
                            "saved_prediction": int(saved_pred[position]), "candidate_classes": candidates,
                            "actual_candidate_logits": [float(actual[position, label]) for label in candidates],
                            "saved_candidate_logits": [float(saved[position, label]) for label in candidates],
                            "max_abs_logit_delta": float(np.max(np.abs(actual[position] - saved[position])))})
    previous_failures = {name: value for name, value in audit["checks"].items() if value["failed"]}
    unchanged = (sha256(args.audit) == audit_hash and sha256(record / "MANIFEST.json") == manifest_hash
                 and all(sha256(ROOT / relative) == expected for relative, expected in files.items()))
    success = (counts["cuda"]["class_mismatches"] == 0
               and counts["cpu"]["trained_task_class_mismatches"] == 0
               and counts["cpu"]["family_class_mismatches"] == 0
               and counts["cpu"]["class_mismatches"] == audit["counts"]["MLP_CPU_prediction_mismatches"] == 1
               and all(not row["task_trained"] and not row["test_identity"] for row in mismatches)
               and unchanged)
    result = {"status": "PASS" if success else "FAIL", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "Post-run numerical reconciliation; the frozen CPU auditor remains FAIL on one untrained auxiliary-head tie.",
        "script_sha256": sha256(Path(__file__)), "record": str(record), "record_manifest_sha256": manifest_hash,
        "preserved_audit_sha256": audit_hash, "preserved_audit_status": audit["status"],
        "preserved_audit_failed_checks": previous_failures, "cuda_device": torch.cuda.get_device_name(),
        "MLP_fits": len(fits), "task_heads_per_fit": 3, "counts": counts, "mismatches": mismatches,
        "all_input_hashes_unchanged": unchanged, "input_sha256": files,
        "interpretation": "All primary motif predictions and all trained-task predictions reproduce on CPU. Every saved head prediction reproduces on the original CUDA device. One untrained distance head on one source-training molecule changes argmax because CPU logits tie while CUDA logits differ by approximately 1.8e-7. No held-out or primary prediction changes. The original FAIL and frozen implementation are preserved; no tolerance or record is amended."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": counts, "mismatches": len(mismatches)}), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
