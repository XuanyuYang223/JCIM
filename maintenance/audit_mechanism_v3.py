"""Independently audit the frozen readout/replay record without training models.

This module imports no experiment implementation. It reads tensor checkpoints on
the CPU, checks archive contracts and component identities, and independently
recomputes the reported nested initialization/support-draw statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("frozen_new", "full_new", "frozen_replay", "full_replay", "scratch_replay")
STEPS = (0, 1, 5, 10, 20, 40, 80)
SPLITS = ("iid_a", "iid_b", "ood_common")
METRICS = ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "position_balanced_accuracy",
           "distance_accuracy", "distance_balanced_accuracy", "distance_mae", "joint_accuracy")
REPORT_METRICS = ("motif_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy", "distance_mae")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class Checks:
    def __init__(self):
        self.results: dict[str, dict] = {}

    def check(self, name: str, condition: bool, detail="") -> None:
        result = self.results.setdefault(name, {"passed": 0, "failed": 0, "examples": []})
        result["passed" if condition else "failed"] += 1
        if not condition and len(result["examples"]) < 8:
            result["examples"].append(str(detail))

    @property
    def success(self) -> bool:
        return bool(self.results) and all(result["failed"] == 0 for result in self.results.values())


def _relative_path(root: Path, relative: str) -> Path:
    candidate = (root / relative.replace("\\", "/")).resolve()
    if candidate == root.resolve() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes the archive: {relative}")
    return candidate


def verify_manifest(root: Path, checks: Checks, label: str) -> dict:
    manifest = read_json(root / "MANIFEST.json")
    paths = manifest["sha256"]
    checks.check(f"{label}_manifest_nonempty", bool(paths))
    expected_paths = set()
    for relative, expected in paths.items():
        path = _relative_path(root, relative)
        expected_paths.add(path)
        checks.check(f"{label}_every_manifest_file_matches", path.is_file() and digest(path) == expected, relative)
    actual_paths = {path.resolve() for path in root.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    checks.check(f"{label}_manifest_covers_all_files", actual_paths == expected_paths,
                 f"unlisted={actual_paths - expected_paths}; absent={expected_paths - actual_paths}")
    return {"manifest_sha256": digest(root / "MANIFEST.json"), "files": len(paths)}


def indexed(rows: list[dict], fields: tuple[str, ...], checks: Checks, label: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        checks.check(f"{label}_unique_rows", key not in result, key)
        result[key] = row
    return result


def same_metrics(current: dict, reference: dict) -> bool:
    return (int(current["n"]) == int(reference["n"])
            and all(float(current[key]) == float(reference[key]) for key in METRICS))


def encoder_bytes(state: dict) -> dict:
    """Compare raw dtype, shape, and storage bytes, including signed zero bits."""
    return {name: (str(value.dtype), tuple(value.shape), value.detach().cpu().contiguous().numpy().tobytes())
            for name, value in state.items() if not name.startswith("heads.")}


def encoder_digest(state: dict) -> str:
    """Independently implement the serialized digest recorded in drift rows."""
    checksum = hashlib.sha256()
    for name, (dtype, shape, raw_bytes) in sorted(encoder_bytes(state).items()):
        checksum.update(name.encode("utf-8"))
        checksum.update(str((dtype, shape)).encode("utf-8"))
        checksum.update(raw_bytes)
    return checksum.hexdigest()


def verify_supports(source: Path, records: list[dict], splits: dict, support: dict, checks: Checks) -> dict:
    ids = [row["id"] for row in records]
    checks.check("source_molecular_IDs_unique", len(ids) == len(set(ids)))
    lookup = {identifier: index for index, identifier in enumerate(ids)}
    support_index = indexed(support["sets"], ("experience", "repeat"), checks, "support_sets")
    checks.check("six_fixed_support_sets", set(support_index) == set(itertools.product(("a", "b"), ("0", "1", "2"))))
    evaluation = set().union(*(set(splits[split]) for split in SPLITS))
    new_ordering = {}
    for key, row in support_index.items():
        experience, repeat = key
        new, old = row["new_ids"], row["old_ids"]
        checks.check("support_unique_32_new_32_old", len(new) == len(set(new)) == len(old) == len(set(old)) == 32, key)
        checks.check("new_and_old_support_IDs_disjoint", not set(new) & set(old), key)
        known = set(new + old) <= set(lookup)
        checks.check("all_support_IDs_known", known, key)
        if not known:
            continue
        archived = read_json(source / "adaptation_support" / f"repeat{repeat}_shots32.json")["molecule_ids"]
        checks.check("new_IDs_exactly_archived_32_shot_support", new == archived, key)
        checks.check("same_new_IDs_across_experiences", repeat not in new_ordering or new_ordering[repeat] == new, key)
        new_ordering[repeat] = new
        new_indices, old_indices = {lookup[identifier] for identifier in new}, {lookup[identifier] for identifier in old}
        checks.check("new_support_in_reserved_adaptation_pool", new_indices <= set(splits["adaptation_pool"]), key)
        checks.check("old_support_in_own_pretraining_set", old_indices <= set(splits[f"train_{experience}"]), key)
        expected_old = np.random.default_rng(7000 + int(repeat)).permutation(splits[f"train_{experience}"])[:32]
        checks.check("old_support_exact_fixed_sampling_protocol", old == [ids[index] for index in expected_old], key)
        checks.check("support_excluded_from_all_evaluation_IDs", not (new_indices | old_indices) & evaluation, key)
        support_smiles = {records[index]["smiles"] for index in new_indices | old_indices}
        evaluation_smiles = {records[index]["smiles"] for index in evaluation}
        checks.check("support_excluded_from_all_evaluation_canonical_SMILES", not support_smiles & evaluation_smiles, key)
    return support_index


def verify_seed_summary(raw: list[dict], aggregates: list[dict], grouping: tuple[str, ...], checks: Checks, label: str) -> None:
    groups = defaultdict(list)
    for row in raw:
        groups[tuple(str(row[field]) for field in grouping)].append(row)
    saved = indexed(aggregates, grouping, checks, label)
    checks.check(f"{label}_complete_groups", set(groups) == set(saved))
    for key, members in groups.items():
        if key not in saved:
            continue
        for metric in REPORT_METRICS:
            per_seed = defaultdict(list)
            for row in members:
                per_seed[str(row["seed"])].append(float(row[metric]))
            independent_means = {seed: statistics.fmean(values) for seed, values in per_seed.items()}
            recorded = saved[key][metric]
            valid = (recorded["n"] == len(per_seed) == 3
                     and set(recorded["per_seed"]) == set(independent_means)
                     and all(math.isclose(recorded["per_seed"][seed], value, abs_tol=1e-12, rel_tol=0)
                             for seed, value in independent_means.items())
                     and math.isclose(recorded["mean"], statistics.fmean(independent_means.values()), abs_tol=1e-12, rel_tol=0)
                     and math.isclose(recorded["std"], statistics.stdev(independent_means.values()), abs_tol=1e-12, rel_tol=0))
            checks.check(f"{label}_repeat_nested_inside_three_seeds", valid, (key, metric))


def audit_record(record: Path, source: Path) -> dict:
    record, source = Path(record).resolve(), Path(source).resolve()
    if not (record / "MANIFEST.json").is_file():
        raise FileNotFoundError("The completed record MANIFEST.json is required; do not audit a running experiment.")
    checks = Checks()
    manifests = {"record": verify_manifest(record, checks, "record"), "source": verify_manifest(source, checks, "source")}
    config, environment = read_json(record / "config.json"), read_json(record / "environment.json")
    checks.check("declared_full_80_step_matched_budget", config["mechanism_steps"] == 80
                 and config["evaluation_steps"] == list(STEPS) and config["seeds"] == [0, 1, 2]
                 and config["support_repeats"] == [0, 1, 2] and config["primary_repeat"] == 0
                 and config["new_count"] == config["old_count"] == 32
                 and config["adaptation_learning_rate"] == 0.0005 and config["weight_decay"] == 0.0001)
    checks.check("original_hard_manifest_unchanged_since_run_start", environment["source_manifest_sha256"] == manifests["source"]["manifest_sha256"])
    checks.check("recording_finished", bool(environment.get("finished_utc")))
    for relative, expected in environment["source_sha256"].items():
        path = _relative_path(ROOT, relative)
        checks.check("executed_source_files_still_match_run_hashes", path.is_file() and digest(path) == expected, relative)
    for name, expected in environment["pretrained_checkpoint_sha256"].items():
        checks.check("input_checkpoint_hashes_match_provenance", digest(source / "checkpoints" / f"{name}.pt") == expected, name)
    records = [json.loads(line) for line in (source / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = read_json(source / "splits.json")
    supports = verify_supports(source, records, splits, read_json(record / "support_sets.json"), checks)
    cases = list(itertools.product(("a", "b"), ("0", "1", "2"), ("0", "1", "2"), ARMS))
    trajectory = read_csv(record / "trajectory.csv")
    trajectory_index = indexed(trajectory, ("experience", "seed", "repeat", "arm", "step", "split"), checks, "trajectory")
    expected_trajectory = {(*case, str(step), split) for case in cases for step in STEPS for split in SPLITS}
    checks.check("all_1890_trajectory_rows_present", len(trajectory) == 1890 and set(trajectory_index) == expected_trajectory)
    for key, row in trajectory_index.items():
        valid = int(row["n"]) == len(splits[row["split"]])
        valid &= all(math.isfinite(float(row[metric])) and (float(row[metric]) >= 0 if metric == "distance_mae"
                     else 0 <= float(row[metric]) <= 1) for metric in METRICS)
        checks.check("trajectory_finite_metrics_and_correct_test_counts", valid, key)
    drift = read_csv(record / "drift.csv")
    drift_index = indexed(drift, ("experience", "seed", "repeat", "arm", "step"), checks, "drift")
    checks.check("all_630_drift_rows_present", len(drift) == 630 and set(drift_index) == {(*case, str(step)) for case in cases for step in STEPS})
    components = read_csv(record / "component_eval.csv")
    component_index = indexed(components, ("experience", "seed", "repeat", "arm", "composition", "split"), checks, "components")
    expected_components = {(*case, composition, split) for case in cases if case[3] != "scratch_replay"
                           for composition in ("E0_H0", "E0_H1", "E1_H0", "E1_H1") for split in SPLITS}
    checks.check("all_864_component_rows_present", len(components) == 864 and set(component_index) == expected_components)
    reference_behavior = indexed(read_csv(source / "behavior.csv"), ("condition", "seed", "split"), checks, "source_behavior")
    reference_adaptation = indexed(read_csv(source / "adaptation.csv"), ("condition", "seed", "shots", "repeat", "split"), checks, "source_adaptation")
    original_states = {}
    for experience, seed in itertools.product(("a", "b", "random"), ("0", "1", "2")):
        original_states[(experience, seed)] = torch.load(source / "checkpoints" / f"{experience}_seed{seed}.pt", map_location="cpu", weights_only=True)["state_dict"]
    frozen_cases = []
    for experience, seed, repeat, arm in cases:
        case = (experience, seed, repeat, arm)
        original = original_states[("random" if arm == "scratch_replay" else experience, seed)]
        original_hash = encoder_digest(original)
        checkpoint_path = record / "checkpoints" / f"{experience}_s{seed}_r{repeat}_{arm}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checks.check("all_90_checkpoint_support_and_config_records_match", checkpoint["support"] == supports[(experience, repeat)]
                     and checkpoint["config"] == config
                     and all(str(checkpoint[field]) == value for field, value in zip(("experience", "seed", "repeat", "arm"), case)), case)
        current = checkpoint["state_dict"]
        if arm.startswith("frozen_"):
            frozen_cases.append(case)
            checks.check("all_36_frozen_final_encoders_bitwise_equal_original", encoder_bytes(current) == encoder_bytes(original), case)
        for step in STEPS:
            row = drift_index.get((*case, str(step)))
            if row is None:
                continue
            hash_equal = row["encoder_original_sha256"] == row["encoder_current_sha256"]
            valid = (row["encoder_original_sha256"] == original_hash
                     and (row["encoder_bitwise_unchanged"].lower() == "true") == hash_equal
                     and (row["encoder_changed"].lower() == "true") != hash_equal)
            if step == 0 or arm.startswith("frozen_"):
                valid &= hash_equal and float(row["encoder_max_abs_delta"]) == 0
            if step == 80:
                valid &= row["encoder_current_sha256"] == encoder_digest(current)
                encoder_names = [name for name in original if not name.startswith("heads.")]
                head_names = [name for name in original if name.startswith("heads.")]
                actual_encoder_delta = max(float((current[name] - original[name]).abs().max()) for name in encoder_names)
                actual_head_delta = max(float((current[name] - original[name]).abs().max()) for name in head_names)
                checks.check("final_encoder_and_head_deltas_recomputed_from_CPU_checkpoints",
                             float(row["encoder_max_abs_delta"]) == actual_encoder_delta
                             and float(row["head_max_abs_delta"]) == actual_head_delta, case)
            checks.check("all_drift_hash_flags_consistent_with_source_and_final_checkpoint", valid, (*case, step))
        for split in SPLITS:
            baseline = trajectory_index.get((*case, "0", split))
            final = trajectory_index.get((*case, "80", split))
            reference = reference_behavior[("random" if arm == "scratch_replay" else experience, seed, split)]
            if baseline:
                checks.check("all_step_zero_scalars_match_original_archive", same_metrics(baseline, reference), (*case, split))
            if arm == "full_new" and final:
                previous = reference_adaptation[(experience, seed, "32", repeat, split)]
                checks.check("full_new_exactly_reproduces_18_prior_32_shot_cases", same_metrics(final, previous), (*case, split))
            if arm != "scratch_replay" and baseline and final:
                before = component_index.get((*case, "E0_H0", split))
                after = component_index.get((*case, "E1_H1", split))
                checks.check("component_E0_H0_matches_baseline", before is not None and same_metrics(before, baseline), (*case, split))
                checks.check("component_E1_H1_matches_final_trajectory", after is not None and same_metrics(after, final), (*case, split))
                if arm.startswith("frozen_"):
                    encoder_swap = component_index.get((*case, "E1_H0", split))
                    head_swap = component_index.get((*case, "E0_H1", split))
                    checks.check("frozen_E1_H0_matches_original_baseline", encoder_swap is not None and same_metrics(encoder_swap, baseline), (*case, split))
                    checks.check("frozen_E0_H1_matches_fully_adapted_state", head_swap is not None and same_metrics(head_swap, final), (*case, split))
    checks.check("exactly_36_frozen_encoder_cases_checked", len(frozen_cases) == 36)
    summary = read_json(record / "report_summary.json")
    final_rows = [row for row in trajectory if row["step"] == "80"]
    verify_seed_summary(trajectory, summary["trajectory"], ("experience", "arm", "step", "split"), checks, "trajectory_summary")
    verify_seed_summary(final_rows, summary["final"], ("experience", "arm", "split"), checks, "final_summary")
    verify_seed_summary([row for row in final_rows if row["repeat"] == "0"], summary["primary_final"], ("experience", "arm", "split"), checks, "primary_summary")
    checks.check("primary_support_and_chance_explicit", str(summary["primary_repeat"]) == "0"
                 and summary["motif_uniform_random_chance"] == 0.125 and summary["final_step"] == 80)
    for filename, selected in (("trajectory_seed_metrics.csv", trajectory), ("final_seed_metrics.csv", final_rows)):
        saved = indexed(read_csv(record / filename), ("experience", "arm", "seed", "step", "split"), checks, filename)
        groups = defaultdict(list)
        for row in selected:
            groups[tuple(row[key] for key in ("experience", "arm", "seed", "step", "split"))].append(row)
        checks.check(f"{filename}_complete_seed_rows", set(saved) == set(groups))
        for key, members in groups.items():
            row = saved.get(key)
            valid = row is not None and int(row["support_repeats"]) == 3
            if valid:
                valid &= all(math.isclose(float(row[metric]), statistics.fmean(float(item[metric]) for item in members), abs_tol=1e-12, rel_tol=0)
                             for metric in REPORT_METRICS)
            checks.check(f"{filename}_correct_repeat_means", valid, key)
    return {"status": "PASS" if checks.success else "FAIL", "audited_utc": datetime.now(timezone.utc).isoformat(),
            "record": str(record), "source": str(source), "platform": platform.platform(),
            "method": "Independent CPU-only manifest, identity, metadata, tensor-byte, scalar-identity, component, and nested-statistics checks; no training and no jcim imports.",
            "limits": "Metadata and final-state checks do not independently reconstruct every optimizer update or prove feature absence. Component identities validate the intervention records; they are not an additive causal decomposition of forgetting.",
            "manifests": manifests, "counts": {"cases": len(cases), "frozen_cases": len(frozen_cases), "trajectory_rows": len(trajectory),
            "drift_rows": len(drift), "component_rows": len(components)}, "checks": checks.results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/readout_replay_v3")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/hard_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/READOUT_REPLAY_V3_AUDIT.json")
    args = parser.parse_args(argv)
    output, record, source = args.output.resolve(), args.record.resolve(), args.source.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit record: {output}")
    if output.is_relative_to(record) or output.is_relative_to(source):
        raise ValueError("The audit output must be outside both immutable experiment records.")
    result = audit_record(record, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
