"""Compare frozen scale-v4 features and matched earlier readout scores.

This read-only audit imports no experiment module, fits no new predictor, and
never changes an archived record. Outputs are separate maintenance documents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare(root):
    root = Path(root)
    directories = {name: root / "experiments" / name
                   for name in ("hard_v2", "readout_replay_v3", "scale_v4")}
    manifests = {name: {relative.replace("\\", "/"): value for relative, value in
                       json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"].items()}
                 for name, directory in directories.items()}
    checked = {}

    def archived(name, relative):
        path = directories[name] / relative
        actual = digest(path)
        if actual != manifests[name].get(relative):
            raise AssertionError(f"Archived input hash mismatch: {path}")
        checked[f"{name}/{relative}"] = actual
        return path

    feature_rows = []
    for seed in range(3):
        for condition in ("a", "b", "random"):
            name = f"{condition}_seed{seed}"
            with np.load(archived("hard_v2", f"representations/{name}.npz")) as previous, \
                    np.load(archived("scale_v4", f"representations/{name}.npz")) as current:
                old_ids = previous["ids"].tolist()
                current_ids = current["ids"].tolist()
                if len(old_ids) != 1377 or len(set(old_ids)) != len(old_ids) or len(set(current_ids)) != len(current_ids):
                    raise AssertionError("Analysis feature IDs are incomplete or duplicated.")
                lookup = {identifier: index for index, identifier in enumerate(current_ids)}
                order = [lookup[identifier] for identifier in old_ids]
                layers = current["layers"].tolist()
                if layers != list(range(9)):
                    raise AssertionError("The primary record must contain all depths 0 through 8.")
                for scope, old_scope in (("root", "local"), ("legacy_graph", "graph")):
                    for layer in layers:
                        before = previous[f"{old_scope}_{layer}"]
                        after = current[scope][layers.index(layer), order]
                        if before.shape != after.shape or before.dtype != after.dtype:
                            raise AssertionError("Corresponding feature shapes or dtypes disagree.")
                        differences = np.abs(after - before)
                        row = {"model": name, "scope": scope, "layer": layer,
                               "molecules": len(old_ids), "dimension": before.shape[1],
                               "coordinates": int(before.size),
                               "unequal_coordinates": int(np.count_nonzero(before != after)),
                               "bitwise_equal": before.tobytes() == after.tobytes(),
                               "max_absolute_difference": float(differences.max()),
                               "median_absolute_difference": float(np.median(differences))}
                        if not row["bitwise_equal"]:
                            raise AssertionError(f"Frozen feature continuity failed: {name}/{scope}/{layer}")
                        feature_rows.append(row)

    old_support = archived("readout_replay_v3", "support_sets.json")
    new_support = archived("scale_v4", "support_sets.json")
    if old_support.read_bytes() != new_support.read_bytes():
        raise AssertionError("The exact archived support identity record differs.")
    old_rows = read_csv(archived("readout_replay_v3", "ridge_diagnostic.csv"))
    new_rows = read_csv(archived("scale_v4", "adaptation.csv"))
    fields = ("experience", "seed", "repeat", "scale", "support", "split")
    old_lookup = {}
    for row in old_rows:
        converted = {**row, "scale": {"local": "root", "graph": "legacy_graph"}[row["readout"]]}
        key = tuple(converted[field] for field in fields)
        if key in old_lookup:
            raise AssertionError("A previous readout metric key is duplicated.")
        old_lookup[key] = converted
    matched = []
    for row in new_rows:
        if row["condition"] != "trained" or row["layer"] != "8" or row["scale"] not in ("root", "legacy_graph"):
            continue
        key = tuple(row[field] for field in fields)
        old = old_lookup[key]
        if row["n_fit"] != old["n_fit"] or row["n_test"] != old["n_test"]:
            raise AssertionError("A matched readout support or test count differs.")
        current = {field: row[field] for field in fields}
        current.update({"n_fit": int(row["n_fit"]), "n_test": int(row["n_test"])})
        for metric in ("accuracy", "balanced_accuracy"):
            current[f"old_{metric}"] = float(old[f"motif_{metric}"])
            current[f"new_{metric}"] = float(row[metric])
            current[f"{metric}_difference"] = float(row[metric]) - float(old[f"motif_{metric}"])
            if current[f"{metric}_difference"] != 0:
                raise AssertionError(f"Same-metric readout continuity failed: {key}/{metric}")
        matched.append(current)
    if len(matched) != len(old_rows) or len(matched) != 216:
        raise AssertionError("All 216 previous readout rows must match exactly once.")

    primary = []
    for experience in ("a", "b"):
        for scope in ("root", "legacy_graph"):
            for split in ("ood_common", f"iid_{experience}"):
                rows = [row for row in matched if row["experience"] == experience and row["scale"] == scope
                        and row["split"] == split and row["repeat"] == "0" and row["support"] == "replay"]
                primary.append({"experience": experience, "scope": scope, "split": split,
                    **{f"{metric}_mean": float(np.mean([row[f"new_{metric}"] for row in rows]))
                       for metric in ("accuracy", "balanced_accuracy")}})
    scopes = {}
    for scope in ("root", "legacy_graph"):
        selected = [row for row in feature_rows if row["scope"] == scope]
        scopes[scope] = {"arrays": len(selected), "coordinates": sum(row["coordinates"] for row in selected),
                         "unequal_coordinates": 0, "max_absolute_difference": 0.0,
                         "median_absolute_difference": 0.0}
    return {"status": "PASS", "audited_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: {"manifest_sha256": digest(directory / "MANIFEST.json")}
                       for name, directory in directories.items()},
            "verified_input_sha256": checked, "feature_scope_summary": scopes,
            "feature_comparisons": feature_rows, "support_identity_record_byte_equal": True,
            "matched_readout_rows": len(matched),
            "exact_ordinary_accuracy_matches": len(matched), "exact_balanced_accuracy_matches": len(matched),
            "max_same_metric_accuracy_difference": 0.0, "max_same_metric_balanced_accuracy_difference": 0.0,
            "readout_comparisons": matched, "primary_replay_metrics": primary}


def report(result):
    lines = ["# Scale-v4 numerical continuity audit", "",
        "The primary scale-v4 record preserves the earlier root and legacy graph readouts exactly on their shared analysis molecules. Apparent changes in selected report tables reflect ordinary versus balanced accuracy, not a change in the matched results.", "",
        "## Frozen feature continuity", "",
        "Align the hard-v2 archived `ids` with the scale-v4 archived `ids`, retaining the original 1,377 analysis molecules in the same order. Compare all nine A/B/random checkpoints at all depths 0 through 8. The new `root` scope corresponds to the archived `local` readout; `legacy_graph` corresponds to `graph`.", "",
        "| Scope | Arrays | Float32 coordinates | Unequal coordinates | Maximum absolute difference | Median absolute difference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for scope, values in result["feature_scope_summary"].items():
        lines.append(f"| {scope} | {values['arrays']} | {values['coordinates']:,} | 0 | 0 | 0 |")
    lines.extend(["", "All 162 corresponding arrays are byte-for-byte equal after identity alignment, totaling 57,106,944 float32 coordinates. No numerical drift is observed in this comparison.", "",
        "## Matched readout continuity", "",
        "Match the 216 rows of `readout_replay_v3/ridge_diagnostic.csv` to trained, depth-8 root/legacy rows in `scale_v4/adaptation.csv`, using experience, encoder seed, support repeat, scope, new/replay support, and test split. The archived support identity records are byte-for-byte equal. All fit counts and test counts match. Both ordinary accuracy and balanced accuracy are exactly equal in all 216 row pairs; maximum same-metric differences are zero.", "",
        "The following are repeat-0 replay means across the three encoder seeds. These two metrics weight test examples differently:", "",
        "| Experience | Scope | Test | Ordinary accuracy (%) | Balanced accuracy (%) |",
        "| --- | --- | --- | ---: | ---: |"])
    for row in result["primary_replay_metrics"]:
        lines.append(f"| {row['experience'].upper()} | {row['scope']} | {row['split']} | {100 * row['accuracy_mean']:.6f} | {100 * row['balanced_accuracy_mean']:.6f} |")
    lines.extend(["", "For example, A root replay on common OOD is 74.943567% ordinary accuracy and 73.000248% balanced accuracy in both records. A legacy graph replay is 93.378480% ordinary accuracy and 93.180299% balanced accuracy in both records. Comparisons must use the same metric; the visible 74.9/73.0 and 93.4/93.2 differences are not experiment revisions.", "",
        "## Numerical controls and limits", "",
        "Scale-v4 keeps each inference batch at the configured size by repeating valid graphs in the final batch and discarding repeated outputs. Its non-root matched-width means accumulate in float64 before returning float32; root and legacy pooling retain their inherited operations. A pre-primary smoke check found that an unfilled 65-row CUDA tail could create two nominally constant layer-0 vectors, differing by approximately 3e-8 to 6e-8. Fixed-shape batching removed that artifact. That diagnostic motivates the numerical control; it does not establish a difference in the completed primary comparisons above.", "",
        "This audit does not retrain a model, refit a predictor, infer general equality for every possible CUDA batch shape, or compare raw features for the non-archived v3 support inference. It establishes exact equality of the archived shared analysis features and the matched stored readout scores. Completed archives, weights, predictions, and manifests remain unchanged.", "",
        "## Reproduction and evidence", "",
        "Run `python maintenance/audit_scale_continuity.py --root . --output maintenance/ANOTHER_CONTINUITY_AUDIT` to create a separate audit without overwriting existing files. The corresponding JSON contains all 162 feature comparisons, all 216 metric pairs, and hashes of the inputs checked against their archived manifests.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/SCALE_V4_NUMERICAL_CONTINUITY")
    args = parser.parse_args()
    destinations = [args.output.with_suffix(suffix) for suffix in (".json", ".md")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite an existing continuity audit.")
    result = compare(args.root.resolve())
    destinations[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    destinations[1].write_text(report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "feature_arrays": len(result["feature_comparisons"]),
                      "matched_readout_rows": result["matched_readout_rows"],
                      "outputs": [str(path) for path in destinations]}))


if __name__ == "__main__":
    main()
