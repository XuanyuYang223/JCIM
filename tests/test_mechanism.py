"""Intervention integrity tests for frozen encoders, replay, and component swaps."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcim import mechanism
from jcim.hard_model import HardMolecularGNN


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class MechanismIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        # Small synthetic graphs exercise actual optimizer and inference code.
        # Their labels are software-test fixtures, not chemical measurements.
        generator = torch.Generator().manual_seed(271)
        n, atoms = 102, 4
        x = torch.rand((n, atoms, 18), generator=generator) * 0.2
        x[:, :, 13:18] = 0
        for channel, node in ((13, 1), (14, 2), (15, 0), (16, 2), (17, 3)):
            x[:, node, channel] = 1
        adjacency = torch.zeros(n, atoms, atoms)
        for i in range(atoms - 1):
            adjacency[:, i, i + 1] = adjacency[:, i + 1, i] = 1
        self.tensors = {"x": x, "adj": adjacency, "mask": torch.ones(n, atoms),
            "y_family": torch.arange(n) % 8, "y_position": torch.arange(n) % 3,
            "y_distance": torch.arange(n) % 16, "record_index": torch.arange(n)}
        self.records = [{"id": f"id_{i:03}", "y_family": i % 8,
                         "y_position": i % 3, "y_distance": i % 16} for i in range(n)]
        self.splits = {"train_a": list(range(32, 64)), "train_b": list(range(64, 96)),
            "adaptation_pool": list(range(32)), "analysis": list(range(96, 102)),
            "iid_a": [96, 97], "iid_b": [98, 99], "ood_common": [100, 101]}
        self.id_lookup = {record["id"]: i for i, record in enumerate(self.records)}
        self.eval_lookup = {i: j for j, i in enumerate(self.splits["analysis"])}
        self.support = {"experience": "a", "repeat": 0,
                        "new_ids": [self.records[i]["id"] for i in range(32)],
                        "old_ids": [self.records[i]["id"] for i in self.splits["train_a"]]}
        self.config = {"hidden_dim": 16, "layers": 2, "batch_size": 128,
            "adaptation_learning_rate": 0.002, "weight_decay": 0.0001,
            "mechanism_steps": 2, "evaluation_steps": [0, 1, 2],
            "support_repeats": [0, 1, 2], "primary_repeat": 0,
            "new_count": 32, "old_count": 32}
        torch.manual_seed(123)
        self.model = HardMolecularGNN(16, 2)
        self.original = clone_state(self.model)

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0,
                                       msg=lambda message, name=name: f"{name}: {message}")

    def run_actual_case(self, arm, config=None):
        observed_training_batches = []

        def observe(module, args):
            if module.training:
                observed_training_batches.append(args[0]["record_index"].detach().cpu().tolist())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoints").mkdir()
            hook = self.model.register_forward_pre_hook(observe)
            try:
                result = mechanism.run_case(self.model, self.original, self.tensors, self.records,
                    self.splits, self.id_lookup, self.eval_lookup, self.splits["analysis"], self.support,
                    arm, "a", 0, config or self.config, output)
            finally:
                hook.remove()
            checkpoint = torch.load(output / "checkpoints" / f"a_s0_r0_{arm}.pt", weights_only=True)
        return result, observed_training_batches, checkpoint

    def test_freezing_covers_embedding_and_all_message_layers_but_keeps_all_heads_trainable(self):
        mechanism.set_trainable(self.model, True)
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("heads."), name)
        for head in ("family", "position", "distance"):
            self.assertTrue(all(p.requires_grad for p in self.model.heads[head].parameters()))
        mechanism.set_trainable(self.model, False)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_actual_frozen_case_preserves_encoder_bitwise_and_trains_all_three_heads(self):
        original_snapshot = {key: value.clone() for key, value in self.original.items()}
        (trajectory, drift, components), batches, checkpoint = self.run_actual_case("frozen_replay")
        after = self.model.state_dict()
        for name in self.original:
            if not name.startswith("heads."):
                torch.testing.assert_close(after[name], self.original[name], rtol=0, atol=0)
        for head in ("family", "position", "distance"):
            keys = [k for k in after if k.startswith(f"heads.{head}.")]
            self.assertTrue(any(not torch.equal(after[k], self.original[k]) for k in keys), head)
        self.assertTrue(all(not row["encoder_changed"] for row in drift))
        self.assertTrue(all(row["encoder_max_abs_delta"] == 0 for row in drift))
        self.assertEqual(len(trajectory), 9)
        self.assertEqual(len(components), 12)
        self.assertEqual(len(batches), 2)
        self.assert_state_equal(original_snapshot, self.original)
        self.assert_state_equal(clone_state(self.model), checkpoint["state_dict"])

    def test_actual_full_case_changes_encoder_and_every_head_without_mutating_initial_state(self):
        snapshot = {key: value.clone() for key, value in self.original.items()}
        (_, drift, _), batches, _ = self.run_actual_case("full_new")
        after = self.model.state_dict()
        for prefix in ("embedding.", "layers.", "heads.family.", "heads.position.", "heads.distance."):
            self.assertTrue(any(not torch.equal(after[k], self.original[k])
                                for k in after if k.startswith(prefix)), prefix)
        self.assertFalse(drift[0]["encoder_changed"])
        self.assertTrue(drift[-1]["encoder_changed"])
        self.assertEqual(len(batches), 2)
        self.assert_state_equal(snapshot, self.original)

    def test_new_only_and_replay_use_full_fixed_support_with_equal_update_counts(self):
        seen = {}
        for arm in ("frozen_new", "full_new", "frozen_replay", "full_replay", "scratch_replay"):
            self.model.load_state_dict(self.original)
            (_, _, components), batches, _ = self.run_actual_case(arm)
            self.assertEqual(len(batches), self.config["mechanism_steps"])
            expected = set(range(64)) if arm.endswith("replay") else set(range(32))
            for batch in batches:
                self.assertEqual(len(batch), len(expected))
                self.assertEqual(set(batch), expected)
                self.assertFalse(set(batch) & set(self.splits["analysis"]))
            if arm == "scratch_replay":
                self.assertEqual(components, [])
            seen[arm] = batches
        # The same support ordering is used whenever the support composition matches.
        self.assertEqual(seen["frozen_new"], seen["full_new"])
        self.assertEqual(seen["frozen_replay"], seen["full_replay"])
        self.assertEqual(seen["full_replay"], seen["scratch_replay"])

    def test_component_swaps_copy_correct_sides_and_never_alias_either_source(self):
        original = {key: value.clone() for key, value in self.original.items()}
        adapted = {key: value + 0.75 for key, value in self.original.items()}
        snapshot_original = {key: value.clone() for key, value in original.items()}
        snapshot_adapted = {key: value.clone() for key, value in adapted.items()}
        for composition in ("E0_H0", "E0_H1", "E1_H0", "E1_H1"):
            mixed = mechanism.mix_components(original, adapted, composition)
            encoder, head = composition.split("_")
            for key, value in mixed.items():
                chosen = head == "H1" if key.startswith("heads.") else encoder == "E1"
                torch.testing.assert_close(value, (adapted if chosen else original)[key], rtol=0, atol=0)
                self.assertNotEqual(value.data_ptr(), original[key].data_ptr())
                self.assertNotEqual(value.data_ptr(), adapted[key].data_ptr())
                value.add_(100)
        self.assert_state_equal(snapshot_original, original)
        self.assert_state_equal(snapshot_adapted, adapted)

    def test_trajectory_evaluation_and_zero_update_case_do_not_change_weights(self):
        snapshot = clone_state(self.model)
        for _ in range(3):
            rows = mechanism.evaluate(self.model, self.tensors, self.records, self.splits,
                self.eval_lookup, self.config, {"experience": "a", "seed": 0, "step": 0})
            self.assertEqual(len(rows), 3)
            self.assert_state_equal(snapshot, clone_state(self.model))
        config = dict(self.config, mechanism_steps=0, evaluation_steps=[0])
        (trajectory, drift, _), batches, _ = self.run_actual_case("full_replay", config)
        self.assertEqual(batches, [])
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(drift[0]["encoder_max_abs_delta"], 0)
        self.assertEqual(drift[0]["head_max_abs_delta"], 0)
        self.assert_state_equal(snapshot, clone_state(self.model))

    def archive_support_fixture(self, source):
        (source / "adaptation_support").mkdir()
        for repeat in self.config["support_repeats"]:
            ordering = np.random.default_rng(6000 + repeat).permutation(32)
            ids = [self.records[int(i)]["id"] for i in ordering]
            (source / "adaptation_support" / f"repeat{repeat}_shots32.json").write_text(
                json.dumps({"repeat": repeat, "shots": 32, "molecule_ids": ids}), encoding="utf-8")

    def test_support_uses_exact_archived_32_ids_for_both_experiences_and_excludes_test(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.archive_support_fixture(source)
            result = mechanism.support_sets(source, self.records, self.splits, self.config)
            self.assertEqual(len(result["sets"]), 6)
            self.assertEqual(result["primary_repeat"], 0)
            for support in result["sets"]:
                archived = json.loads((source / "adaptation_support" /
                    f"repeat{support['repeat']}_shots32.json").read_text(encoding="utf-8"))
                self.assertEqual(support["new_ids"], archived["molecule_ids"])
                self.assertEqual(len(set(support["new_ids"])), 32)
                self.assertEqual(len(set(support["old_ids"])), 32)
                self.assertFalse(set(support["new_ids"]) & set(support["old_ids"]))
                new = {self.id_lookup[x] for x in support["new_ids"]}
                old = {self.id_lookup[x] for x in support["old_ids"]}
                self.assertLessEqual(new, set(self.splits["adaptation_pool"]))
                self.assertLessEqual(old, set(self.splits[f"train_{support['experience']}"]))
                self.assertFalse((new | old) & set(self.splits["analysis"]))

    def test_support_outside_archived_reserve_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.archive_support_fixture(source)
            path = source / "adaptation_support/repeat0_shots32.json"
            archived = json.loads(path.read_text(encoding="utf-8"))
            archived["molecule_ids"][0] = self.records[self.splits["ood_common"][0]]["id"]
            path.write_text(json.dumps(archived), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved archived 32"):
                mechanism.support_sets(source, self.records, self.splits, self.config)

    def test_main_loads_fresh_pretrained_or_random_weights_for_every_case(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "source", directory / "output"
            source.mkdir()
            self.archive_support_fixture(source)
            (source / "MANIFEST.json").write_text("{}", encoding="utf-8")
            (source / "checkpoints").mkdir()
            states = {}
            for condition, shift in (("a", 0), ("b", 0.1), ("random", 0.2)):
                states[condition] = {key: value + shift for key, value in self.original.items()}
                (source / "checkpoints" / f"{condition}_seed0.pt").write_bytes(b"fixture")
            snapshots = {condition: {key: value.clone() for key, value in state.items()}
                         for condition, state in states.items()}
            config = dict(self.config, support_repeats=[0], seeds=[0], device="cpu", cpu_threads=2,
                          diagnostic_examples=6)
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            starts = []

            def fake_case(model, original, tensors, records, splits, id_lookup, eval_lookup,
                          diagnostic_indices, support, arm, experience, seed, config, output):
                condition = "random" if arm == "scratch_replay" else experience
                self.assert_state_equal(states[condition], clone_state(model))
                self.assert_state_equal(states[condition], original)
                starts.append((experience, arm))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(10)
                return [], [], []

            argv = ["mechanism", "--source", str(source), "--output", str(output), "--config", str(config_path)]
            with patch.object(sys, "argv", argv), \
                 patch.object(mechanism, "load_source", return_value=(self.records, self.splits, self.tensors)), \
                 patch.object(mechanism, "checkpoint_state", side_effect=lambda source, condition, seed: states[condition]), \
                 patch.object(mechanism, "ridge_diagnostics", return_value=[]), \
                 patch.object(mechanism, "run_case", side_effect=fake_case), \
                 patch.object(mechanism, "verify_hard", return_value={}), \
                 patch("jcim.mechanism_report.generate_mechanism_report"), \
                 contextlib.redirect_stdout(io.StringIO()):
                mechanism.main()
            self.assertEqual(starts, [(experience, arm) for experience in ("a", "b") for arm in mechanism.ARMS])
            for condition in states:
                self.assert_state_equal(snapshots[condition], states[condition])
            self.assertTrue((output / "MANIFEST.json").is_file())


if __name__ == "__main__":
    unittest.main()
