"""Negative controls for the independent readout/replay archival auditor."""

import copy
import json
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from maintenance.audit_mechanism_v3 import (
    Checks, digest, encoder_bytes, encoder_digest, indexed, main,
    verify_executed_source, verify_manifest, verify_seed_summary, verify_supports,
)


class IndependentMechanismAuditTests(unittest.TestCase):
    def _source_repository(self, directory):
        subprocess.run(["git", "init", "--quiet"], cwd=directory, check=True, capture_output=True)
        source = directory / "jcim" / "encoder.py"
        source.parent.mkdir()
        source.write_bytes(b"frozen source bytes\n")
        subprocess.run(["git", "add", "jcim/encoder.py"], cwd=directory, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Audit Test", "-c", "user.email=audit@example.invalid",
                        "commit", "--quiet", "-m", "Archive source bytes"], cwd=directory, check=True, capture_output=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        return source, {"git_head_at_start": commit, "source_sha256": {"jcim/encoder.py": digest(source)}}

    def test_exact_recorded_commit_recovers_snapshot_after_worktree_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, environment = self._source_repository(directory)
            checks = Checks()
            current = verify_executed_source(directory, environment, checks)
            self.assertTrue(checks.success)
            self.assertEqual(current["files"]["jcim/encoder.py"]["verified_from"], "current_worktree")
            source.write_bytes(b"later performance fix\n")
            checks = Checks()
            historical = verify_executed_source(directory, environment, checks)
            self.assertTrue(checks.success)
            self.assertEqual(historical["recorded_full_commit"], environment["git_head_at_start"])
            self.assertEqual(historical["files"]["jcim/encoder.py"]["verified_from"], "recorded_immutable_git_commit")
            self.assertEqual(historical["files"]["jcim/encoder.py"]["historical_git_blob_sha256"], environment["source_sha256"]["jcim/encoder.py"])

    def test_wrong_recorded_historical_hash_and_abbreviated_commit_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, environment = self._source_repository(directory)
            source.write_bytes(b"later performance fix\n")
            wrong_hash = copy.deepcopy(environment)
            wrong_hash["source_sha256"]["jcim/encoder.py"] = hashlib.sha256(b"unrecorded historical bytes\n").hexdigest()
            checks = Checks()
            verify_executed_source(directory, wrong_hash, checks)
            self.assertFalse(checks.success)
            self.assertEqual(checks.results["executed_source_files_match_recorded_snapshot"]["failed"], 1)
            for invalid in (environment["git_head_at_start"][:12], "HEAD", "0" * 40):
                with self.subTest(revision=invalid):
                    checks = Checks()
                    verify_executed_source(directory, {**environment, "git_head_at_start": invalid}, checks)
                    self.assertFalse(checks.success)
                    self.assertEqual(checks.results["executed_source_revision_is_full_immutable_commit"]["failed"], 1)

    def test_another_full_commit_cannot_supply_different_snapshot_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, environment = self._source_repository(directory)
            source.write_bytes(b"different committed bytes\n")
            subprocess.run(["git", "add", "jcim/encoder.py"], cwd=directory, check=True, capture_output=True)
            subprocess.run(["git", "-c", "user.name=Audit Test", "-c", "user.email=audit@example.invalid",
                            "commit", "--quiet", "-m", "Different snapshot"], cwd=directory, check=True, capture_output=True)
            wrong_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
            checks = Checks()
            verify_executed_source(directory, {**environment, "git_head_at_start": wrong_commit}, checks)
            self.assertFalse(checks.success)
            self.assertEqual(checks.results["executed_source_files_match_recorded_snapshot"]["failed"], 1)

    def test_manifest_checks_hashes_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "input.txt").write_text("frozen", encoding="utf-8")
            (directory / "MANIFEST.json").write_text(json.dumps({"sha256": {"input.txt": digest(directory / "input.txt")}}), encoding="utf-8")
            checks = Checks()
            verify_manifest(directory, checks, "record")
            self.assertTrue(checks.success)
            (directory / "input.txt").write_text("changed", encoding="utf-8")
            (directory / "unlisted.txt").write_text("extra", encoding="utf-8")
            checks = Checks()
            verify_manifest(directory, checks, "record")
            self.assertEqual(checks.results["record_every_manifest_file_matches"]["failed"], 1)
            self.assertEqual(checks.results["record_manifest_covers_all_files"]["failed"], 1)

    def test_manifest_rejects_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "MANIFEST.json").write_text(json.dumps({"sha256": {"../outside.txt": "0" * 64}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_manifest(directory, Checks(), "record")

    def test_raw_encoder_identity_catches_signed_zero_and_ignores_heads(self):
        original = {"embedding.weight": torch.tensor([0.0, 1.0]), "heads.family.weight": torch.tensor([1.0])}
        changed_heads = {name: value.clone() for name, value in original.items()}
        changed_heads["heads.family.weight"].add_(1)
        self.assertEqual(encoder_bytes(original), encoder_bytes(changed_heads))
        self.assertEqual(encoder_digest(original), encoder_digest(changed_heads))
        changed_encoder = {name: value.clone() for name, value in original.items()}
        changed_encoder["embedding.weight"][0] = -0.0
        self.assertTrue(torch.equal(original["embedding.weight"], changed_encoder["embedding.weight"]))
        self.assertNotEqual(encoder_bytes(original), encoder_bytes(changed_encoder))
        self.assertNotEqual(encoder_digest(original), encoder_digest(changed_encoder))

    def test_support_audit_detects_wrong_replay_draw_and_new_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "adaptation_support").mkdir()
            records = [{"id": f"mol_{index}", "smiles": f"canonical_{index}"} for index in range(230)]
            splits = {"train_a": list(range(64)), "train_b": list(range(64, 128)),
                      "adaptation_pool": list(range(128, 224)), "iid_a": [224, 225], "iid_b": [226, 227], "ood_common": [228, 229]}
            support = {"sets": []}
            for repeat in range(3):
                new = [records[index]["id"] for index in range(128 + repeat * 32, 160 + repeat * 32)]
                (source / "adaptation_support" / f"repeat{repeat}_shots32.json").write_text(json.dumps({"molecule_ids": new}), encoding="utf-8")
                for experience in ("a", "b"):
                    indices = np.random.default_rng(7000 + repeat).permutation(splits[f"train_{experience}"])[:32]
                    support["sets"].append({"experience": experience, "repeat": repeat, "new_ids": new,
                                            "old_ids": [records[index]["id"] for index in indices]})
            checks = Checks()
            verify_supports(source, records, splits, support, checks)
            self.assertTrue(checks.success)
            altered = copy.deepcopy(support)
            altered["sets"][0]["old_ids"] = list(reversed(altered["sets"][0]["old_ids"]))
            altered["sets"][1]["new_ids"][0] = "mol_224"
            checks = Checks()
            verify_supports(source, records, splits, altered, checks)
            self.assertGreater(checks.results["old_support_exact_fixed_sampling_protocol"]["failed"], 0)
            self.assertGreater(checks.results["new_IDs_exactly_archived_32_shot_support"]["failed"], 0)
            self.assertGreater(checks.results["support_excluded_from_all_evaluation_IDs"]["failed"], 0)

    def test_nested_statistics_reject_nine_pseudoreplicates(self):
        metrics = ("motif_accuracy", "position_accuracy", "distance_accuracy", "joint_accuracy", "distance_mae")
        raw = [{"experience": "a", "seed": seed, "repeat": repeat,
                **{metric: 0.2 + 0.1 * seed + 0.03 * repeat for metric in metrics}}
               for seed in range(3) for repeat in range(3)]
        statistic = {"mean": 0.33, "std": 0.1, "n": 3, "per_seed": {"0": 0.23, "1": 0.33, "2": 0.43}}
        saved = [{"experience": "a", **{metric: copy.deepcopy(statistic) for metric in metrics}}]
        checks = Checks()
        verify_seed_summary(raw, saved, ("experience",), checks, "summary")
        self.assertTrue(checks.success)
        saved[0]["motif_accuracy"]["n"] = 9
        checks = Checks()
        verify_seed_summary(raw, saved, ("experience",), checks, "summary")
        self.assertEqual(checks.results["summary_repeat_nested_inside_three_seeds"]["failed"], 1)

    def test_duplicate_scientific_rows_are_detected(self):
        checks = Checks()
        indexed([{"seed": 0}, {"seed": 0}], ("seed",), checks, "trajectory")
        self.assertEqual(checks.results["trajectory_unique_rows"]["failed"], 1)

    def test_cli_refuses_overwrite_before_reading_and_outputs_outside_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--record", str(root / "absent"), "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
            with self.assertRaisesRegex(ValueError, "outside"):
                main(["--record", str(root / "record"), "--output", str(root / "record" / "audit.json")])


if __name__ == "__main__":
    unittest.main()
