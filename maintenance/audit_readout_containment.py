"""Post-run check that shallow ridge rules remain representable in full inputs.

No model is fitted. This check evaluates saved ridge states and the original
source objective. It is a separate post-run analysis, not a predeclared endpoint
or an independent replication of the primary experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/readout_v7")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/READOUT_V7_CONTAINMENT.json")
    args = parser.parse_args()
    record, output = args.record.resolve(), args.output.resolve()
    if output.exists() or record == output or record in output.parents:
        raise FileExistsError("Containment analysis must use a fresh path outside the frozen record.")
    manifest_hash = digest(record / "MANIFEST.json")
    manifest = json.loads((record / "MANIFEST.json").read_text())["sha256"]
    config = json.loads((record / "config.json").read_text())
    assert config["alpha"] == 1.
    reference = json.loads((record / "source_reference.json").read_text())
    hard = Path(reference["hard_source"])
    molecules = [json.loads(line) for line in (hard / "molecules.jsonl").read_text().splitlines()]
    splits = json.loads((hard / "splits.json").read_text())
    inventory = [row for row in csv_rows(record / "panel/fit_inventory.csv") if row["classifier"] == "ridge"]
    encoders = {row["model"]: row for row in csv_rows(record / "panel/encoder_inventory.csv")}
    groups = {}
    for row in inventory:
        key = (row["model"], row["experience"], row["normalization"])
        assert row["scope"] not in groups.setdefault(key, {})
        groups[key][row["scope"]] = row
    rows = []
    for (model, experience, normalization), pair in sorted(groups.items()):
        assert set(pair) == {"full512", "shallow_only"}
        feature_path = record / "panel" / encoders[model]["features"]
        assert digest(feature_path) == manifest[feature_path.relative_to(record).as_posix()]
        with np.load(feature_path) as data:
            features, indices = data["readout"].astype(np.float64), data["indices"].tolist()
        lookup = {index: row for row, index in enumerate(indices)}
        full_path = record / "panel" / pair["full512"]["checkpoint"]
        shallow_path = record / "panel" / pair["shallow_only"]["checkpoint"]
        for path in (full_path, shallow_path):
            assert digest(path) == manifest[path.relative_to(record).as_posix()]
        with np.load(full_path) as data:
            full = {name: data[name] for name in data.files}
        with np.load(shallow_path) as data:
            shallow = {name: data[name] for name in data.files}
        assert np.array_equal(full["active_columns"], np.arange(512))
        assert np.array_equal(shallow["active_columns"], np.arange(448, 512))
        assert np.array_equal(full["mean"][448:], shallow["mean"])
        assert np.array_equal(full["scale"][448:], shallow["scale"])
        z = (features - full["mean"]) / full["scale"]
        weights = np.zeros_like(full["weights"])
        weights[448:] = shallow["weights"]
        embedded_scores = z @ weights + shallow["bias"]
        shallow_scores = ((features[:, 448:] - shallow["mean"]) / shallow["scale"]) @ shallow["weights"] + shallow["bias"]
        maximum_error = float(np.max(np.abs(embedded_scores - shallow_scores)))
        assert maximum_error <= 1e-10
        assert np.array_equal(embedded_scores.argmax(1), shallow_scores.argmax(1))
        assert np.sum(weights ** 2) == np.sum(shallow["weights"] ** 2)
        full_scores = z @ full["weights"] + full["bias"]
        train = splits[f"train_{experience}"]
        train_rows = [lookup[index] for index in train]
        targets = np.eye(8)[[molecules[index]["y_family"] for index in train]]
        full_objective = float(np.sum((full_scores[train_rows] - targets) ** 2) + np.sum(full["weights"] ** 2))
        candidate_objective = float(np.sum((embedded_scores[train_rows] - targets) ** 2) + np.sum(weights ** 2))
        assert full_objective <= candidate_objective + 1e-8
        row = {"model": model, "regime": pair["full512"]["regime"], "experience": experience,
               "seed": int(pair["full512"]["seed"]), "normalization": normalization,
               "feature_sha256": digest(feature_path), "full_checkpoint_sha256": digest(full_path),
               "shallow_checkpoint_sha256": digest(shallow_path), "common_scaler_bit_exact": True,
               "embedded_prediction_mismatches": 0, "embedding_maximum_logit_error": maximum_error,
               "coefficient_l2_penalty_identical": True, "full_source_objective": full_objective,
               "embedded_shallow_source_objective": candidate_objective,
               "source_objective_improvement": candidate_objective - full_objective}
        for split in ("ood_common", f"iid_{experience}"):
            selected = splits[split]
            selected_rows = [lookup[index] for index in selected]
            labels = np.asarray([molecules[index]["y_family"] for index in selected])
            for name, scores in (("full", full_scores), ("shallow", embedded_scores)):
                row[f"{name}_{split}_accuracy"] = float(np.mean(scores[selected_rows].argmax(1) == labels))
        rows.append(row)
    assert digest(record / "MANIFEST.json") == manifest_hash
    result = {"status": "PASS", "analysis": "Post-run containment and saved source-objective evaluation; no refitting.",
        "record": str(record), "manifest_sha256": manifest_hash, "auditor_sha256": digest(__file__),
        "pipeline_only": json.loads((record / "environment.json").read_text())["pipeline_only"],
        "ridge_pair_count": len(rows), "all_common_scalers_bit_exact": True,
        "all_embedded_predictions_preserved": True, "all_l2_penalties_identical": True,
        "all_full_source_objectives_no_larger_than_feasible_shallow_objective": True,
        "limits": "Representability does not imply the fitted ridge rule is invariant. This analysis cannot identify a unique semantic shortcut or explain every optimizer/readout condition. Pairs reuse encoders and molecules and are not independent experimental replicates.",
        "pairs": rows}
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "ridge_pairs": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
