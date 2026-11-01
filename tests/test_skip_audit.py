"""Negative controls for identity, initialization, and functional skip auditing."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from jcim.hard_model import HardMolecularGNN
from jcim.skip_model import SkipMolecularGNN
from maintenance.audit_skip_v5 import (
    Checks, encoder_bytes, expected_case_keys, functional_features, functional_heads,
    initial_state, metrics_from_predictions, tensor_state_digest, verify_plan,
    verify_statistics, main,
)


class SkipAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_plan_requires_ordered_source_ids_and_excludes_new_labels(self):
        records = [{"id": f"m{index}"} for index in range(12)]
        splits = {"train_a": [0, 1], "val_a": [2], "train_b": [3, 4], "val_b": [5],
                  "iid_a": [6], "iid_b": [7], "ood_common": [8, 9], "analysis": [6, 7, 8, 9],
                  "adaptation_pool": [10, 11]}
        selected = lambda indices: {"indices": indices, "ids": [records[i]["id"] for i in indices]}
        plan = {"experiences": {experience: {"train": selected(splits[f"train_{experience}"]),
                                             "validation": selected(splits[f"val_{experience}"]),
                                             "initial_probe": selected(splits[f"train_{experience}"][:128])}
                                for experience in ("a", "b")},
                "tests": {name: selected(splits[name]) for name in ("iid_a", "iid_b", "ood_common", "analysis")},
                "branch_permutations": {"0": {"seed": 90000, "indices": np.random.default_rng(90000).permutation(4).tolist()}}}
        checks = Checks()
        verify_plan(plan, records, splits, [0], checks)
        self.assertTrue(checks.success)
        changed = copy.deepcopy(plan)
        changed["experiences"]["a"]["train"] = selected([1, 0])
        changed["experiences"]["b"]["validation"] = selected([10])
        checks = Checks()
        verify_plan(changed, records, splits, [0], checks)
        self.assertFalse(checks.success)
        self.assertEqual(checks.results["train_validation_indices_exactly_original"]["failed"], 2)
        self.assertEqual(checks.results["no_common_ood_or_adaptation_labels_in_training_or_validation"]["failed"], 1)

    def test_branch_permutation_is_seed_fixed_and_cannot_follow_outcomes(self):
        records = [{"id": f"m{index}"} for index in range(7)]
        splits = {"train_a": [0], "val_a": [1], "train_b": [0], "val_b": [1],
                  "iid_a": [2], "iid_b": [3], "ood_common": [4, 5], "analysis": [2, 3, 4, 5], "adaptation_pool": [6]}
        selected = lambda indices: {"indices": indices, "ids": [records[i]["id"] for i in indices]}
        plan = {"experiences": {experience: {"train": selected([0]), "validation": selected([1]),
                                              "initial_probe": selected([0])} for experience in ("a", "b")},
                "tests": {name: selected(splits[name]) for name in ("iid_a", "iid_b", "ood_common", "analysis")},
                "branch_permutations": {"0": {"seed": 90000, "indices": [0, 1, 2, 3]}}}
        checks = Checks()
        verify_plan(plan, records, splits, [0], checks)
        self.assertEqual(checks.results["branch_permutation_fixed_before_outcomes"]["failed"], 1)

    def test_initial_state_resets_heads_and_uses_declared_encoder(self):
        torch.manual_seed(8)
        random_model = HardMolecularGNN(hidden_dim=4, layers=2)
        random_state = random_model.state_dict()
        trained_state = {name: value + 0.25 for name, value in random_state.items()}
        for variant in ("deep", "deep_control", "shallow_skip"):
            with self.subTest(variant=variant):
                state = initial_state(random_state, trained_state, variant)
                self.assertEqual(encoder_bytes(state), encoder_bytes(trained_state))
                for name, value in state.items():
                    if name.startswith("heads."):
                        if name.endswith(".0.weight") and variant != "deep":
                            torch.testing.assert_close(value[:, :28], random_state[name], rtol=0, atol=0)
                            self.assertEqual(torch.count_nonzero(value[:, 28:]).item(), 0)
                        else:
                            torch.testing.assert_close(value, random_state[name], rtol=0, atol=0)
        self.assertEqual(tensor_state_digest(initial_state(random_state, trained_state, "deep_control")),
                         tensor_state_digest(initial_state(random_state, trained_state, "shallow_skip")))

    def test_full_tensor_digest_detects_signed_zero_and_head_changes(self):
        state = {"embedding.weight": torch.tensor([0.0, 1.0]), "heads.family.0.weight": torch.tensor([1.0])}
        changed = copy.deepcopy(state)
        changed["heads.family.0.weight"][0] = 2
        self.assertNotEqual(tensor_state_digest(state), tensor_state_digest(changed))
        self.assertEqual(encoder_bytes(state), encoder_bytes(changed))
        changed = copy.deepcopy(state)
        changed["embedding.weight"][0] = -0.0
        self.assertNotEqual(tensor_state_digest(state), tensor_state_digest(changed))

    def test_independent_functional_reconstruction_matches_all_skip_variants(self):
        torch.manual_seed(123)
        base = HardMolecularGNN(hidden_dim=4, layers=2).eval()
        x = torch.randn(2, 6, 18)
        x[:, :, 13:] = 0
        for marker, atom in ((13, 0), (14, 1), (15, 2), (16, 3), (17, 4)):
            x[:, atom, marker] = 1
        mask = torch.tensor([[1., 1., 1., 1., 1., 0.], [1., 1., 1., 1., 1., 1.]])
        adjacency = torch.zeros(2, 6, 6)
        for graph, size in ((0, 5), (1, 6)):
            for atom in range(size - 1):
                adjacency[graph, atom, atom + 1] = adjacency[graph, atom + 1, atom] = 1
        for variant in ("deep", "deep_control", "shallow_skip"):
            model = SkipMolecularGNN.from_base(base, variant).eval()
            if variant != "deep":
                with torch.no_grad():
                    for head in model.heads.values():
                        head[0].weight[:, 28:].fill_(0.15)
            state = model.state_dict()
            graph, shallow = functional_features(state, x, adjacency, mask)
            with torch.no_grad():
                reference = model({"x": x, "adj": adjacency, "mask": mask}, representations=True)
            torch.testing.assert_close(graph, reference["graph"][-1], rtol=0, atol=0)
            logits = functional_heads(state, graph, shallow, variant)
            for task in logits:
                torch.testing.assert_close(logits[task], reference[task], rtol=0, atol=0)
            if variant != "deep":
                for mode, branch in (("zero", torch.zeros_like(shallow)), ("swap", shallow.flip(0))):
                    if variant == "deep_control" and mode == "swap":
                        branch = graph[:, :4].flip(0)
                    expected = model({"x": x, "adj": adjacency, "mask": mask}, branch_override=branch)
                    current = functional_heads(state, graph, shallow, variant, branch)
                    for task in current:
                        torch.testing.assert_close(current[task], expected[task], rtol=0, atol=0)

    def test_scores_keep_balanced_accuracy_and_joint_denominators(self):
        records = [{"y_family": family, "y_position": 0, "y_distance": 8} for family in (0, 0, 0, 1)]
        predictions = [{"pred_family": 0, "pred_position": 0, "pred_distance": 8} for _ in records]
        predictions[-1]["pred_distance"] = 9
        scores = metrics_from_predictions(records, list(range(4)), predictions)
        self.assertEqual(scores["motif_accuracy"], 0.75)
        self.assertEqual(scores["motif_balanced_accuracy"], 0.5)
        self.assertEqual(scores["joint_accuracy"], 0.75)
        self.assertEqual(scores["distance_mae"], 0.25)

    def test_expected_cases_count_seeds_as_encoders_without_augmentation(self):
        keys = expected_case_keys([0, 1, 2])
        self.assertEqual(len(keys), 36)
        self.assertEqual(len([key for key in keys if key[0] == "frozen_readout"]), 18)

    def test_statistics_reject_duplicate_seed_rows_and_wrong_SD(self):
        metrics = ("motif_accuracy", "motif_balanced_accuracy", "position_accuracy", "position_balanced_accuracy",
                   "distance_accuracy", "distance_balanced_accuracy", "distance_mae", "joint_accuracy")
        rows = [{"experience": "a", "seed": seed, **{metric: value for metric in metrics}}
                for seed, value in enumerate((0.1, 0.2, 0.3))]
        saved = [{"experience": "a", **{metric: {"mean": 0.2, "std": 0.1, "n": 3,
                                                 "per_seed": {"0": 0.1, "1": 0.2, "2": 0.3}}
                                        for metric in metrics}}]
        checks = Checks()
        verify_statistics(rows, saved, ("experience",), checks, "test")
        self.assertTrue(checks.success)
        changed = copy.deepcopy(saved)
        changed[0]["motif_accuracy"]["std"] = float(np.std([0.1, 0.2, 0.3], ddof=0))
        checks = Checks()
        verify_statistics(rows, changed, ("experience",), checks, "test")
        self.assertEqual(checks.results["test_seed_values_mean_sample_SD_independently_recomputed"]["failed"], 1)
        checks = Checks()
        verify_statistics(rows + [rows[0]], saved, ("experience",), checks, "test")
        self.assertEqual(checks.results["test_exactly_one_observation_per_encoder_seed"]["failed"], 1)

    def test_audit_output_cannot_overwrite_or_enter_a_frozen_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing.json"
            output.write_text("preserve audit bytes", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--record", str(root / "missing"), "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve audit bytes")
            with self.assertRaisesRegex(ValueError, "outside"):
                main(["--record", str(root / "record"), "--output", str(root / "record" / "audit.json")])


if __name__ == "__main__":
    unittest.main()
