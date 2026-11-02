"""Rebuild the manuscript's numerical ledger from frozen v2-v7 records.

This script only reads experiment records and writes manuscript/CLAIMS.csv and
manuscript/CLAIMS_CHECK.json. It does not rerun scientific experiments. A later
manuscript revision with additional experiments must extend this ledger builder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript"
CLAIMS: list[dict] = []
CACHE: dict[str, list[dict]] = {}
SOURCE_ONLY = "Exactly 1386 original source identities and motif labels; no new-combination labels"
SOURCE_JOINT = "Exactly 1386 original source identities with motif, position, and distance labels; no new-combination labels"
DIRECT = "256 balanced labels from one source position per directed fit; additional to encoder training"
ADAPT = "32 new-combination labels plus 32 replayed old source labels per fit"


def digest(name: str) -> str:
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def add(claim_id, location, text, source, selector, metric, result, *, rows=None,
        aggregation="Read the specified frozen fields", labels="Not a fitted readout",
        localization="No additional target boundary", caveat="", n_seeds=""):
    CLAIMS.append({
        "claim_id": claim_id,
        "manuscript_location": location,
        "claim_text": text,
        "source_file": source,
        "source_sha256": digest(source),
        "selector_json": json.dumps(selector, sort_keys=True),
        "source_rows_1based": json.dumps(rows or []),
        "metric": metric,
        "result_json": json.dumps(result, sort_keys=True),
        "aggregation": aggregation,
        "n_encoder_seeds": n_seeds,
        "label_budget": labels,
        "localization_budget": localization,
        "scope_caveat": caveat,
    })


def select(source: str, query: dict) -> list[tuple[int, dict]]:
    if source not in CACHE:
        with (ROOT / source).open(newline="", encoding="utf-8") as handle:
            CACHE[source] = list(csv.DictReader(handle))
    selected = []
    for number, row in enumerate(CACHE[source], start=2):
        if all(row[key] in wanted if isinstance(wanted, list) else row[key] == wanted
               for key, wanted in query.items()):
            selected.append((number, row))
    if not selected:
        raise ValueError(f"Empty selection: {source}, {query}")
    return selected


def summarize_seed_values(selected, metric, *, group_keys=(), factor=100.0):
    panels = defaultdict(lambda: defaultdict(list))
    for _, row in selected:
        key = tuple(row[k] for k in group_keys)
        panels[key][row["seed"]].append(float(row[metric]))
    result = {}
    for key, seed_rows in panels.items():
        if len(seed_rows) != 3:
            raise ValueError(f"Expected three encoder seeds; got {seed_rows}")
        seed_values = {seed: statistics.mean(values) * factor
                       for seed, values in sorted(seed_rows.items())}
        result["|".join(key) or "all"] = {
            "mean": statistics.mean(seed_values.values()),
            "sample_sd": statistics.stdev(seed_values.values()),
            "seed_values": seed_values,
            "within_seed_rows": {seed: len(values) for seed, values in sorted(seed_rows.items())},
            "units": "percent or percentage points" if factor == 100 else "raw metric units",
        }
    return result


def csv_claim(claim_id, location, text, source, query, metric="accuracy", *,
              group_keys=(), labels=SOURCE_ONLY, localization="No additional target boundary",
              caveat="Descriptive mean and sample SD across three encoder seeds", factor=100):
    selected = select(source, query)
    result = summarize_seed_values(selected, metric, group_keys=group_keys, factor=factor)
    add(claim_id, location, text, source, query, metric, result,
        rows=[number for number, _ in selected],
        aggregation="Average selected measurements within each seed and stated panel, then mean and sample SD across seeds",
        labels=labels, localization=localization, caveat=caveat, n_seeds=3)
    return result


def main():
    audit_file = "experiments/hard_v2/audit.json"
    audit = json.loads((ROOT / audit_file).read_text())
    add("C00", "Abstract; Methods; Supplement S1", "6912 molecules and unique canonical identities",
        audit_file, ["n_molecules", "unique_canonical_identities", "split_identity_leakage"],
        "counts and identity isolation", {k: audit[k] for k in (
            "n_molecules", "unique_canonical_identities", "split_identity_leakage")})
    add("C01", "Methods; Supplement S2", "Identity partition and effective split counts",
        audit_file, ["split_counts", "effective_train_counts"], "molecule counts", {
            "split_counts": audit["split_counts"],
            "effective_train_counts": audit["effective_train_counts"],
            "all_validation_candidates": audit["split_counts"]["val"] + audit["split_counts"]["discarded_val_ood"],
        }, caveat="The analysis set is the union of the three test residues, not an independent fourth test")

    molecule_file = "experiments/hard_v2/molecules.jsonl"
    with (ROOT / molecule_file).open(encoding="utf-8") as handle:
        molecules = [json.loads(line) for line in handle]
    motif_examples = {}
    for line_number, record in enumerate(molecules, start=1):
        motif_examples.setdefault(record["y_family"], (line_number, record))
    assert len(motif_examples) == 8
    motifs = {}
    for index, (_, record) in sorted(motif_examples.items()):
        smiles = record["motif_smiles"]
        assert smiles.count("C") == 7 and smiles.count("O") == 2
        assert len(record["target_fragment_indices"]) == 9
        motifs[index] = {"smiles": smiles, "carbon_count": 7, "oxygen_count": 2, "heavy_atom_count": 9}
    grid = {key: sorted({r[key] for r in molecules}) for key in (
        "anchor_family", "anchor_spacer", "scaffold", "distractor_family", "position")}
    add("C02", "Methods; Supplement S1", "Eight nine-heavy-atom motifs and generation grid",
        molecule_file, {"motif_examples": "First record for each y_family", "grid": "all records"},
        "motif and background counts", {"motifs": motifs, "grid": grid},
        rows=[x[0] for _, x in sorted(motif_examples.items())],
        caveat="JSONL row numbers; equal ordinary-feature multisets also checked in frozen data audits")
    config_file = "experiments/hard_v2/config.json"
    config = json.loads((ROOT / config_file).read_text())
    fields = ["seeds", "hidden_dim", "layers", "epochs", "batch_size", "learning_rate", "weight_decay", "probe_alpha"]
    add("C03", "Methods; Supplement S3", "Original model dimensions and training settings",
        config_file, fields, "fixed hyperparameters", {k: config[k] for k in fields},
        labels="1386 source identities per experience; three computed task labels", n_seeds=3)

    inventory = "experiments/skip_v5/inventory.csv"
    selected = select(inventory, {"regime": "end_to_end", "experience": "a", "seed": "0"})
    add("C04", "Methods; Supplement S3 and S5", "189531/201819 parameters, 448/512 features, 1320 updates",
        inventory, {"regime": "end_to_end", "experience": "a", "seed": "0"},
        "parameter_count; readout_dim; optimizer_updates", {
            r["variant"]: {k: int(r[k]) for k in (
                "parameter_count", "readout_dim", "optimizer_updates", "optimizer_updates_executed_now")}
            for _, r in selected}, rows=[n for n, _ in selected], labels=SOURCE_JOINT,
        caveat="Original deep end-to-end baseline reuses its exact historical 1320-update source model")
    add("C05", "Methods; Supplement S2", "Motif-matched source experiences and unequal position marginals",
        audit_file, ["train_motif_counts", "train_position_counts"], "source class counts",
        {k: audit[k] for k in ("train_motif_counts", "train_position_counts")}, labels=SOURCE_JOINT)
    pos_file = "experiments/mechanism_diagnostics_v3/position_baselines.csv"
    selected = select(pos_file, {"curriculum": "a", "split": "ood_common"})
    add("C06", "Methods; Supplement S7", "Eight-output uniform expected accuracy is 12.5%",
        pos_file, {"curriculum": "a", "split": "ood_common"}, "uniform_chance; n_classes",
        {k: selected[0][1][k] for k in ("uniform_chance", "n_classes")},
        rows=[n for n, _ in selected], caveat="Theoretical expectation; distinct from empirical fitted random-feature performance")

    hard = "experiments/hard_v2/behavior.csv"
    for cid, experience in [("C07", "a"), ("C08", "b")]:
        csv_claim(cid, "Results: original failure", f"Original deep {experience.upper()} unseen motif accuracy",
                  hard, {"condition": experience, "split": "ood_common"}, "motif_accuracy", labels=SOURCE_JOINT)
    diag = [(n, r) for n, r in select(hard, {"condition": ["a", "b"]})
            if r["split"] == "iid_" + r["condition"]]
    add("C09", "Results: original failure", "Original own-familiar three-task accuracy is 100%",
        hard, {"condition": ["a", "b"], "split": "iid_{condition}"},
        "motif_accuracy; position_accuracy; distance_accuracy",
        {metric: summarize_seed_values(diag, metric, group_keys=("condition",))
         for metric in ("motif_accuracy", "position_accuracy", "distance_accuracy")},
        rows=[n for n, _ in diag], labels=SOURCE_JOINT, n_seeds=3,
        aggregation="Own-experience diagonal; mean and sample SD across seeds separately by task and experience")
    assert all(float(r[k]) == 1 for _, r in diag for k in ("motif_accuracy", "position_accuracy", "distance_accuracy"))
    pos_rows = select(pos_file, {})
    add("C10", "Results: original failure", "Position-only MAP familiar A/B 35.9/37.9%; common unseen 0%",
        pos_file, {}, "accuracy", {f"{r['curriculum']}|{r['split']}": float(r["accuracy"]) * 100 for _, r in pos_rows},
        rows=[n for n, _ in pos_rows], labels=SOURCE_ONLY,
        caveat="No seed SD for a deterministic empirical MAP baseline")

    scale = "experiments/scale_v4/seed_metrics.csv"
    direct_query = {"experiment": "direct_direction_average", "repeat_scope": "all_repeats_within_seed",
                    "scale": "target", "split": "target_all"}
    csv_claim("C11", "Results: local readability", "Layer-one oracle target cross-position decoding is 100%",
              scale, direct_query | {"layer": "1"}, "balanced_accuracy",
              group_keys=("condition", "experience"), labels=DIRECT,
              localization="Known target membership for post-encoding pooling",
              caveat="Source-position labels are additional supervision; random A/B references duplicate an encoder")
    for cid, experience, condition in [("C12", "a", "trained"), ("C13", "b", "trained"), ("C14", "a", "random")]:
        csv_claim(cid, "Results: local readability", f"Layer-eight target balanced decoding {condition} {experience.upper()}",
                  scale, direct_query | {"layer": "8", "experience": experience, "condition": condition},
                  "balanced_accuracy", labels=DIRECT,
                  localization="Known target membership for post-encoding pooling",
                  caveat="Three support draws and all six directions averaged within encoder seed; random counted once")
    for cid, experience in [("C15", "a"), ("C16", "b")]:
        csv_claim(cid, "Results: labeled adaptation", f"Graph/query trained-minus-random adaptation advantage {experience.upper()}",
                  "experiments/scale_v4/paired_seed_deltas.csv", {
                      "experiment": "adaptation", "repeat_scope": "all_repeats_within_seed",
                      "condition": "trained_minus_random", "experience": experience,
                      "layer": "8", "scale": "legacy_graph", "support": "replay", "split": "ood_common"},
                  "balanced_accuracy", labels=ADAPT,
                  caveat="Support-averaged summary, distinct from the v4 primary fixed support draw zero")
    supports_file = "experiments/scale_v4/support_sets.json"
    support_sets = json.loads((ROOT / supports_file).read_text())["sets"]
    support_counts = []
    for support in support_sets:
        assert len(set(support["new_ids"])) == 32
        assert len(set(support["old_ids"])) == 32
        assert not set(support["new_ids"]) & set(support["old_ids"])
        support_counts.append({"experience": support["experience"], "repeat": support["repeat"],
                               "new": len(support["new_ids"]), "old": len(support["old_ids"])})
    assert sorted({r["repeat"] for r in support_sets}) == [0, 1, 2]
    add("C17", "Methods; Supplement S4", "Adaptation uses 32 new plus 32 old labels per support draw",
        supports_file, "all supports", "support identities", support_counts,
        labels=ADAPT, caveat="Supports are inherited from the frozen v3 archive; distinct labels, not examples per class")
    scale_config = "experiments/scale_v4/config.json"
    sc = json.loads((ROOT / scale_config).read_text())
    add("C18", "Methods; Supplement S4", "Directed probes use 32 examples per motif, 256 labels, three draws",
        scale_config, ["direct_per_class", "support_repeats", "layers", "alpha"], "probe budget",
        {"per_class": sc["direct_per_class"], "total": 8 * sc["direct_per_class"],
         "support_repeats": sc["support_repeats"], "alpha": sc["alpha"]}, labels=DIRECT,
        localization="Scope-dependent oracle mask", caveat="Six ordered pairs from three positions; pairs are not independent seeds")
    add("C19", "Methods: endpoints", "Three encoder seeds; sample SD; support draws nested within seed",
        scale_config, ["seeds", "support_repeats"], "statistical units",
        {k: sc[k] for k in ("seeds", "support_repeats")}, n_seeds=3,
        caveat="The source report records the aggregation rule; A/B and shared random models are not six independent initializations")

    skip = "experiments/skip_v5/seed_metrics.csv"
    for cid, experience, variant in [
        ("C20", "a", "deep"), ("C21", "a", "deep_control"), ("C22", "a", "shallow_skip"),
        ("C23", "b", "deep"), ("C24", "b", "deep_control"), ("C25", "b", "shallow_skip")]:
        csv_claim(cid, "Table 1", f"End-to-end {experience.upper()} {variant} unseen accuracy", skip,
                  {"regime": "end_to_end", "experience": experience, "variant": variant,
                   "intervention": "intact", "split": "ood_common"}, "motif_accuracy", labels=SOURCE_JOINT)
    for cid, experience in [("C26", "a"), ("C27", "b")]:
        csv_claim(cid, "Table 1", f"Frozen-encoder {experience.upper()} source-head unseen accuracy", skip,
                  {"regime": "frozen_readout", "experience": experience,
                   "intervention": "intact", "split": "ood_common"}, "motif_accuracy", group_keys=("variant",), labels=SOURCE_JOINT)
    for cid, regime in [("C28", "end_to_end"), ("C29", "frozen_readout")]:
        csv_claim(cid, "Results: shallow path", f"{regime} paired shallow-minus-control accuracy gain",
                  "experiments/skip_v5/paired_seed_deltas.csv", {
                      "regime": regime, "intervention": "intact", "split": "ood_common",
                      "contrast": "shallow_skip_minus_deep_control"},
                  "motif_accuracy", group_keys=("experience",), labels=SOURCE_JOINT)

    pool = "experiments/pooling_v6/pooling/pooling_source_metrics.csv"
    oracle = csv_claim("C30", "Abstract; Results; Table 2", "Every original-source-label oracle-target fit reaches 100% unseen accuracy",
                       pool, {"pool": "target", "split": "ood_common"}, group_keys=("regime", "experience", "classifier"),
                       localization="Supplied nine-atom target membership after full-molecule encoding",
                       caveat="Diagnostic upper bound; all labels are original-source-only; random A/B fits share encoders")
    assert all(r["mean"] == 100 and r["sample_sd"] == 0 for r in oracle.values())
    for cid, regime, experience, classifier in [
        ("C31", "end_to_end", "a", "ridge"), ("C32", "end_to_end", "a", "mlp"),
        ("C33", "end_to_end", "b", "ridge"), ("C34", "end_to_end", "b", "mlp"),
        ("C35", "frozen_readout", "a", "ridge"), ("C36", "frozen_readout", "a", "mlp"),
        ("C37", "frozen_readout", "b", "ridge"), ("C38", "frozen_readout", "b", "mlp"),
        ("C39", "random", "a", "ridge"), ("C40", "random", "a", "mlp"),
        ("C41", "random", "b", "ridge"), ("C42", "random", "b", "mlp")]:
        csv_claim(cid, "Table 2", f"{regime} {experience.upper()} whole-graph {classifier} unseen accuracy", pool,
                  {"regime": regime, "experience": experience, "classifier": classifier,
                   "pool": "whole_graph", "split": "ood_common"},
                  caveat="Standardized fixed diagnostic; differs from original head fitting in multiple conditions")
    own = [(n, r) for n, r in select(pool, {"regime": "end_to_end", "classifier": "ridge", "pool": "whole_graph"})
           if r["split"] == "iid_" + r["experience"]]
    assert all(float(r["accuracy"]) == 1 for _, r in own)
    add("C43", "Results: matched pooling", "Recovered shallow whole-graph ridge retains 100% own-familiar accuracy",
        pool, {"regime": "end_to_end", "classifier": "ridge", "pool": "whole_graph", "split": "iid_{experience}"},
        "accuracy", summarize_seed_values(own, "accuracy", group_keys=("experience",)),
        rows=[n for n, _ in own], labels=SOURCE_ONLY, n_seeds=3)

    branch = "experiments/pooling_v6/branch/branch_summary.csv"
    bq = {"variant": "shallow_skip", "split": "ood_common", "task": "family", "intervention": "permuted_branch"}
    bs = select(branch, bq)
    counts = {k: sum(int(r[k]) for _, r in bs) for k in (
        "n", "prediction_changed_count", "wrong_to_correct_count", "correct_to_wrong_count", "both_wrong_changed_count")}
    counts["n_models"] = len(bs)
    assert counts["n"] == 5316 and counts["n_models"] == 12
    assert counts["prediction_changed_count"] == counts["both_wrong_changed_count"] == 1
    add("C44", "Results: branch outputs", "One wrong-to-wrong class change across 12 models and 443 molecules each",
        branch, bq, "prediction transition counts", counts, rows=[n for n, _ in bs],
        aggregation="Sum event counts for the stated twelve fixed model panels; not a claim of 5316 independent molecules",
        labels="No fitting or new labels; fixed-model output sensitivity", n_seeds=3,
        caveat="The same 443 molecular identities occur in each model panel")
    for cid, intervention in [("C45", "permuted_branch"), ("C46", "zero_branch")]:
        q = bq | {"intervention": intervention}
        rows = select(branch, q)
        panels = summarize_seed_values(rows, "absolute_margin_delta_mean", group_keys=("regime", "experience"), factor=1)
        means = [r["mean"] for r in panels.values()]
        add(cid, "Results: branch outputs", f"{intervention} panel-mean absolute correct-class margin change range",
            branch, q, "absolute_margin_delta_mean", {"range": [min(means), max(means)], "panels": panels},
            rows=[n for n, _ in rows], aggregation="Within-model example mean, then seed mean; report range over four regime/experience panels",
            labels="No fitting or new labels", caveat="Raw logit units are head-specific, not calibrated cross-model effects", n_seeds=3)
    assert all(int(r["logits_exactly_unchanged_count"]) == 0 for _, r in bs)
    add("C47", "Results: branch outputs", "Permutation changes continuous logits on every tested unseen molecule",
        branch, bq, "logits_exactly_unchanged_count", {"sum": 0, "model_panels": len(bs), "total_model_molecule_pairs": counts["n"]},
        rows=[n for n, _ in bs], aggregation="Check zero exact-logit-unchanged counts in every panel", n_seeds=3,
        labels="No fitting or new labels", caveat="This does not mean every change alters the margin, class, or correctness")

    locality = "experiments/scale_v4/locality.csv"
    lr = select(locality, {"layer": "8", "region": ["root", "target"]})
    grouped = defaultdict(list)
    for _, r in lr:
        grouped[r["region"]].append(r)
    add("C48", "Supplement S4", "Layer-eight root and target receptive reach covers whole graph for 98.7% of archived feature identities",
        locality, {"layer": "8", "region": ["root", "target"]}, "receptive_covers_full_graph",
        {key: {"fraction": statistics.mean(r["receptive_covers_full_graph"] == "True" for r in rows), "n": len(rows)}
         for key, rows in grouped.items()}, rows=[n for n, _ in lr],
        aggregation="Fraction over all 1649 archived feature identities, separately by pooling scope",
        caveat="Architectural upper bound on reach; not measured functional influence")

    agreed_file = "experiments/mechanism_diagnostics_v3/position_model_agreement.csv"
    agreed = [(n, r) for n, r in select(agreed_file, {"split": "ood_common"})
              if r["model"].split("_", 1)[0] == r["curriculum"]]
    assert len(agreed) == 6
    assert all(float(r["error_training_supported_fraction"]) > 0.99 for _, r in agreed)
    add("C49", "Results: original failure", "Most unseen errors predict a label supported at that position in source training",
        agreed_file, {"split": "ood_common", "curriculum": "model.split('_')[0]; trained A/B models only"},
        "error_training_supported_fraction",
        {f"{r['curriculum']}|{r['model']}": float(r["error_training_supported_fraction"]) for _, r in agreed},
        rows=[n for n, _ in agreed], labels="No refit; model error diagnostic from source binding",
        caveat="Agreement with the supported label set does not identify the exact fitted rule or pure MAP prediction", n_seeds=3)

    data_audit_file = "maintenance/HARD_V2_DATA_AUDIT.json"
    independent_data_audit = json.loads((ROOT / data_audit_file).read_text())
    audit_keys = ["single_raw_local_root_feature_vector", "identical_target_ordinary_feature_multisets"]
    assert independent_data_audit["status"] == "PASS"
    assert all(independent_data_audit["checks"][k]["failed"] == 0 for k in audit_keys)
    add("C50", "Methods; Supplement S1", "Matched ordinary target-feature multisets and identical raw-root attributes",
        data_audit_file, ["checks." + k for k in audit_keys], "independent data audit checks",
        {k: independent_data_audit["checks"][k] for k in audit_keys},
        caveat="Independent reconstruction from mapped frozen molecules, not evidence of general chemical-property equivalence")
    csv_claim("C51", "Results: original failure", "Original common-unseen position accuracy remains 100%",
              hard, {"condition": ["a", "b"], "split": "ood_common"},
              "position_accuracy", group_keys=("condition",), labels=SOURCE_JOINT)
    own_skip = [(n, r) for n, r in select(skip, {"intervention": "intact"})
                if r["split"] == "iid_" + r["experience"]]
    assert all(float(r["motif_accuracy"]) == 1 for _, r in own_skip)
    add("C52", "Table 1 caption; Results: shallow path", "Every intact v5 variant retains 100% own-familiar motif accuracy",
        skip, {"intervention": "intact", "split": "iid_{experience}"}, "motif_accuracy",
        summarize_seed_values(own_skip, "motif_accuracy", group_keys=("regime", "experience", "variant")),
        rows=[n for n, _ in own_skip], labels=SOURCE_JOINT, n_seeds=3,
        aggregation="Mean and sample SD across seeds separately by regime, experience, and variant")
    add("C53", "Supplement S1", "18 input channels, five query markers, no target membership input",
        audit_file, ["feature_dimension", "feature_names", "target_fragment_mask_used_as_input"],
        "input feature inventory", {k: audit[k] for k in (
            "feature_dimension", "feature_names", "target_fragment_mask_used_as_input")},
        caveat="Known boundary metadata is introduced only in explicitly labeled oracle analyses")

    # C100 and above extend the frozen initial-draft claim IDs without renumbering.
    v7 = "experiments/readout_v7/"
    v7_labels = ("Exactly 1386 original source identities; family cells use motif labels; "
                 "joint cells also use source position and distance labels; no new-combination labels")
    fit_inventory = v7 + "panel/fit_inventory.csv"
    fits = select(fit_inventory, {})
    fit_counts = {classifier: sum(r["classifier"] == classifier for _, r in fits)
                  for classifier in ("mlp", "ridge")}
    assert fit_counts == {"mlp": 144, "ridge": 72}
    assert all(int(r["n_fit"]) == 1386 for _, r in fits)
    add("C100", "Methods; Supplement S9", "144 MLP fits and 72 ridge fits on original source identities",
        fit_inventory, {}, "fit counts; source fitting identities", fit_counts,
        rows=[n for n, _ in fits], labels=v7_labels, n_seeds=3,
        caveat="Fits, scopes, and shared A/B random encoders are not independent initialization replicates")
    enc_file = v7 + "panel/encoder_inventory.csv"
    encoders = select(enc_file, {})
    assert len(encoders) == 15 and all(r["encoder_bitwise_unchanged"] == "True" for _, r in encoders)
    add("C101", "Methods; Supplement S9", "All fifteen encoders remain frozen; three random encoders shared across A/B fits",
        enc_file, {}, "encoder inventory and state continuity",
        {"n_encoders": len(encoders), "n_random": sum(r["condition"] == "random" for _, r in encoders),
         "all_encoder_hashes_unchanged": True}, rows=[n for n, _ in encoders], labels=v7_labels,
        caveat="There are eighteen source-label cases because the three random encoders are reused for A/B")
    metrics = v7 + "panel/source_metrics.csv"
    for cid, classifier, split in [("C102", "mlp", "ood_common"), ("C103", "ridge", "ood_common"),
                                   ("C104", "mlp", "own_iid"), ("C105", "mlp", "train"),
                                   ("C106", "ridge", "own_iid"), ("C107", "ridge", "train")]:
        query = {"classifier": classifier, "task": "family"}
        selected = select(metrics, query)
        selected = [(n, r) for n, r in selected if r["split"] == (
            "iid_" + r["experience"] if split == "own_iid" else split)]
        query["split"] = "iid_{experience}" if split == "own_iid" else split
        result = summarize_seed_values(selected, "accuracy", group_keys=(
            "regime", "experience", "scope", "normalization", "objective"))
        add(cid, "Results: readout factors; Supplement S9", f"Complete {classifier} motif accuracy panel: {split}",
            metrics, query, "accuracy", result, rows=[n for n, _ in selected], labels=v7_labels,
            aggregation="Mean and sample SD across three seeds for each exact protocol cell", n_seeds=3,
            caveat="Raw shallow MLP cells underfit source/IID; report guards alongside OOD rather than inferring feature absence")
    csv_claim("C108", "Results: readout factors; Supplement S9", "All fixed-companion conditional factorial contrasts",
              v7 + "conditional_factor_edges.csv", {"split": "ood_common"},
              group_keys=("regime", "experience", "classifier", "factor", "scope", "normalization", "objective"),
              labels=v7_labels, caveat="Conditional scope/normalization/objective differences are not additive shares of a historical gap")
    csv_claim("C109", "Results: readout factors; Supplement S9", "Averaged factor effects and finite-difference interactions",
              v7 + "factorial_effects.csv", {"split": "ood_common"},
              group_keys=("regime", "experience", "classifier", "factors"), labels=v7_labels,
              caveat="Companion settings averaged within each seed; higher-order contrasts are finite differences, not regression coefficients")
    csv_claim("C110", "Results: readout factors; Supplement S9", "All same-protocol trained-minus-random common-unseen contrasts",
              v7 + "matched_random_deltas.csv", {"split": "ood_common"},
              group_keys=("regime", "experience", "classifier", "scope", "normalization", "objective"),
              labels=v7_labels, caveat="Random reference is shared across trained regimes; no universal source-only training benefit is inferred")
    clip_file = v7 + "clipping_by_fit.csv"
    clipped = select(clip_file, {})
    clip_counts = {"fits": len(clipped), "fits_with_clipping": sum(int(r["clipped_steps"]) > 0 for _, r in clipped),
                   "clipped_updates": sum(int(r["clipped_steps"]) for _, r in clipped),
                   "total_updates": sum(int(r["optimizer_updates"]) for _, r in clipped)}
    add("C111", "Methods; Supplement S9", "Frozen independent heads are coupled only by the global clip in this objective comparison",
        clip_file, {}, "clipping diagnostics", clip_counts, rows=[n for n, _ in clipped], labels=v7_labels,
        caveat="Raw motif gradients are unchanged by other disjoint heads at identical states; this does not test multitask encoder learning")
    continuity = v7 + "panel/continuity_checks.csv"
    anchors = select(continuity, {})
    assert all(r["predictions_exact_match"] == "True" and r["state_exact_match"] == "True" for _, r in anchors)
    add("C112", "Supplement S9", "Exact v5 frozen raw/full/joint and v6 standardized shallow-ridge continuity anchors",
        continuity, {}, "state and prediction continuity",
        {"n_checks": len(anchors), "all_states_exact": True, "all_predictions_exact": True,
         "by_kind": {kind: sum(r["kind"] == kind for _, r in anchors)
                     for kind in sorted({r["kind"] for _, r in anchors})}},
        rows=[n for n, _ in anchors], labels=v7_labels,
        caveat="Anchors are continuations of existing numerical evidence, not independent replications")
    containment_file = "maintenance/READOUT_V7_CONTAINMENT.json"
    certificate = json.loads((ROOT / containment_file).read_text())
    certificate_keys = ["ridge_pair_count", "all_common_scalers_bit_exact", "all_embedded_predictions_preserved",
                        "all_l2_penalties_identical", "all_full_source_objectives_no_larger_than_feasible_shallow_objective"]
    assert certificate["status"] == "PASS" and certificate["ridge_pair_count"] == 36
    assert all(certificate[k] for k in certificate_keys[1:])
    add("C113", "Results: readout factors; Supplement S10", "Post-run nested-ridge saved-state containment and source-objective certificate",
        containment_file, certificate_keys, "no-refit containment checks",
        {k: certificate[k] for k in certificate_keys},
        aggregation="Check every paired ridge state; these pairs reuse encoders and are not independent replications",
        labels="No new fitting or labels; original source objective evaluated on saved states",
        caveat="Post-run analysis separate from predeclared endpoints; no unique shortcut mechanism is established")
    numerical_file = "maintenance/READOUT_V7_NUMERICAL_CHECK.json"
    numerical = json.loads((ROOT / numerical_file).read_text())
    assert numerical["status"] == "PASS" and numerical["preserved_audit_status"] == "FAIL"
    assert numerical["counts"]["cpu"]["family_class_mismatches"] == 0
    assert numerical["counts"]["cpu"]["trained_task_class_mismatches"] == 0
    assert numerical["counts"]["cuda"]["class_mismatches"] == 0
    add("C114", "Supplement S11", "Preserved CPU auxiliary-head tie and exact matched-CUDA numerical replay",
        numerical_file, ["classification", "counts", "mismatches", "all_input_hashes_unchanged"],
        "numerical reproducibility checks", {k: numerical[k] for k in (
            "classification", "counts", "mismatches", "all_input_hashes_unchanged")},
        aggregation="All saved head arrays and class decisions; preserve CPU failure rather than changing its criterion",
        labels="No new fitting or labels; frozen saved-state reconstruction",
        caveat="One CPU mismatch is an untrained auxiliary distance head on a source identity; motif outcomes and trained tasks agree")
    runtime_file = "maintenance/READOUT_V7_RUNTIME_SINGLE_THREAD_AUDIT.json"
    runtime = json.loads((ROOT / runtime_file).read_text())
    assert runtime["status"] == "PASS" and all(v.get("failed", 0) == 0 for v in runtime["checks"].values())
    assert not runtime["runtime_override"]["tolerances_changed"]
    add("C115", "Data/code status; Supplement S11", "Full matched-runtime audit passes with original CPU failure preserved",
        runtime_file, ["status", "counts", "runtime_override"], "full independent replay status and counts",
        {k: runtime[k] for k in ("status", "counts", "runtime_override")},
        aggregation="All frozen checks rerun with the documented saved-head device override and startup BLAS configuration",
        labels="No new training or labels; saved states and independent numerical refits",
        caveat="Original CPU audit and first wrapper failure remain unchanged; this is a matched-runtime audit, not universal cross-device bit identity")
    runtime_env_file = "maintenance/READOUT_V7_RUNTIME_ENVIRONMENT.json"
    runtime_env = json.loads((ROOT / runtime_env_file).read_text())
    env_keys = ["startup_environment", "unchanged_wrapper_between_initial_and_single_thread_audits",
                "initial_wrapper_status", "single_thread_wrapper_status",
                "launch_environment_must_be_set_before_python_and_numpy_initialization"]
    assert runtime_env["startup_environment"] == {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    assert runtime_env["initial_wrapper_status"] == "FAIL"
    assert runtime_env["single_thread_wrapper_status"] == "PASS"
    add("C116", "Supplement S11", "Startup BLAS configuration and unchanged-wrapper failure/pass provenance",
        runtime_env_file, env_keys, "launch environment evidence", {k: runtime_env[k] for k in env_keys},
        labels="No new fitting or labels", caveat="Configured values are recorded; active BLAS worker count was not independently observed")

    if len({r["claim_id"] for r in CLAIMS}) != len(CLAIMS):
        raise ValueError("Duplicate claim IDs")
    out = OUT / "CLAIMS.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLAIMS[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(CLAIMS)
    # This checks the ledger's source identities, selectors, expected seed counts,
    # and stated universal/count claims, rather than auditing the whole experiment.
    check = {
        "status": "PASS",
        "scope": "Manuscript numerical ledger through v7, not a new scientific experiment or independent audit",
        "n_claims": len(CLAIMS),
        "claim_ids": [r["claim_id"] for r in CLAIMS],
        "n_source_files": len({r["source_file"] for r in CLAIMS}),
        "claims_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {r["source_file"]: r["source_sha256"] for r in CLAIMS},
        "numerical_values_recomputed_from_frozen_rows": True,
        "future_factor_controls_included": True,
    }
    (OUT / "CLAIMS_CHECK.json").write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: {len(CLAIMS)} claims from {check['n_source_files']} record and audit source files")


if __name__ == "__main__":
    main()
