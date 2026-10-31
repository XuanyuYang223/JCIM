"""Scientific controls for the position baseline and isolated local inputs."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from jcim.hard_data import generate_hard_dataset, split_hard_dataset
from jcim.hard_model import HardMolecularGNN, tensorize_hard
from jcim import mechanism_diagnostics as diagnostics


class MechanismDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_hard_dataset((0,))
        cls.splits = split_hard_dataset(cls.records)
        cls.local = diagnostics.local_records(cls.records)

    def test_local_input_is_complete_target_without_remote_query_markers(self):
        for original, local in zip(self.records, self.local):
            selected = original["target_fragment_indices"]
            self.assertEqual(local["x"].shape, (9, 18))
            np.testing.assert_array_equal(local["x"][:, :13], original["x"][selected, :13])
            np.testing.assert_array_equal(local["adj"], original["adj"][np.ix_(selected, selected)])
            np.testing.assert_array_equal(local["x"][:, [13, 14, 16, 17]], 0)
            self.assertEqual(local["x"][:, 15].sum(), 1)
            self.assertEqual(local["x"][local["local_attachment_idx"], 15], 1)
            self.assertLessEqual(local["root_radius"], 6)
            # The cut root retains full-molecule boundary chemistry.
            root = local["local_attachment_idx"]
            self.assertEqual(local["x"][root, 10] * 4, local["adj"][root].sum() + 1)
            self.assertEqual(local["x"][root, 11] * 4, 2)

    def test_local_input_types_match_a_b_and_repeat_across_contexts(self):
        audit = diagnostics.local_input_audit(self.local, self.splits)
        self.assertTrue(audit["local_input_train_a_b_distributions_identical"])
        self.assertFalse(audit["molecular_train_test_identity_overlap"])
        self.assertEqual(set(audit["unique_local_input_types"].values()), {8})
        self.assertEqual(set(audit["local_type_overlap_train_a_test"].values()), {8})
        self.assertEqual(audit["maximum_root_radius"], 6)

    def test_local_record_extraction_supports_archived_padded_arrays(self):
        chosen = self.records[:4]
        tensors = tensorize_hard(chosen, torch.device("cpu"))
        graphs = {k: v.numpy() for k, v in tensors.items()}
        metadata = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in chosen]
        extracted = diagnostics.local_records(metadata, graphs)
        for expected, actual in zip(self.local[:4], extracted):
            np.testing.assert_array_equal(expected["x"], actual["x"])
            np.testing.assert_array_equal(expected["adj"], actual["adj"])

    def test_incomplete_target_and_distribution_mismatch_are_rejected(self):
        broken = dict(self.records[0], target_fragment_indices=self.records[0]["target_fragment_indices"][:-1])
        with self.assertRaisesRegex(ValueError, "full nine-atom"):
            diagnostics.local_records([broken])
        bad_splits = dict(self.splits, train_b=self.splits["train_b"][:-1])
        with self.assertRaisesRegex(ValueError, "identical local-input"):
            diagnostics.local_input_audit(self.local, bad_splits)

    def test_position_map_is_not_equivalent_to_training_supported_label_set(self):
        result = diagnostics.position_baseline(self.records, self.splits["train_a"],
            {name: self.splits[name] for name in diagnostics.EVALUATION_SPLITS})
        self.assertEqual(result["balanced_accuracy_upper_bound"], 3 / 8)
        self.assertEqual([len(x) for x in result["supported_labels_by_position"]], [3, 2, 3])
        iid, other, ood = result["rows"]
        self.assertEqual(iid["true_label_training_supported_fraction"], 1)
        self.assertLess(iid["accuracy"], 0.5)
        self.assertEqual(iid["balanced_accuracy"], 3 / 8)
        self.assertEqual(other["accuracy"], 0)
        self.assertEqual(ood["accuracy"], 0)

    def test_map_ties_are_deterministic_and_set_compatibility_does_not_imply_map_agreement(self):
        records = [{"id": str(i), "y_position": position, "y_family": label}
                   for i, (position, label) in enumerate(((0, 0), (0, 3), (1, 1), (1, 4), (2, 2), (2, 5)))]
        predictions = [{"model": "bound", "id": r["id"], "pred_family": [3, 0, 4, 1, 5, 2][i]}
                       for i, r in enumerate(records)]
        result = diagnostics.position_baseline(records, list(range(6)), {"test": list(range(6))}, predictions)
        self.assertEqual(result["map_labels_by_position"], [0, 1, 2])
        agreement = result["agreements"][0]
        self.assertEqual(agreement["position_map_agreement"], 0.5)
        self.assertEqual(agreement["prediction_training_supported_fraction"], 1)
        self.assertEqual(agreement["error_training_supported_fraction"], 1)
        self.assertEqual(agreement["model_accuracy"], 0)

    def test_missing_predictions_fail_instead_of_silently_changing_denominator(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            diagnostics.position_baseline(self.records, self.splits["train_a"],
                {"test": self.splits["iid_a"]}, [{"model": "broken", "id": "absent", "pred_family": 0}])

    def test_existing_output_is_rejected_before_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prior"
            output.mkdir()
            (output / "keep.txt").write_text("frozen", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                diagnostics.run_diagnostics(Path(directory) / "missing", output)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "frozen")

    def test_copied_probes_preserve_every_seed_layer_and_actual_shuffled_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = [{"model": f"random_seed{s}", "condition": "random", "seed": s,
                         "probe": "cross_position_motif", "layer": layer, "n_fit": 100, "n_test": 90,
                         "accuracy": .2, "balanced_accuracy": .19, "shuffled_accuracy": .1}
                        for s in (0, 1, 2) for layer in range(9)]
            diagnostics.write_csv(path / "probes.csv", original)
            copied = diagnostics._probe_rows(path)
            self.assertEqual(len(copied), 27)
            self.assertEqual({(r["seed"], r["layer"]) for r in copied},
                             {(s, layer) for s in (0, 1, 2) for layer in range(9)})
            for expected, actual in zip(original, copied):
                for key in expected:
                    self.assertEqual(expected[key], actual[key])
                self.assertEqual(actual["uniform_chance"], .125)
                self.assertNotIn("shuffled_balanced_accuracy", actual)

    def test_motif_only_control_updates_encoder_but_not_auxiliary_heads(self):
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            torch.manual_seed(31)
            model = HardMolecularGNN(12, 2)
            before = {key: value.detach().clone() for key, value in model.state_dict().items()}
            tensors = tensorize_hard(self.local, torch.device("cpu"))
            config = {"learning_rate": .0015, "weight_decay": .0001, "batch_size": 128, "epochs": 1}
            train = self.splits["train_a"][:128]
            validation = self.splits["val_a"][:32]
            diagnostics.train_local_control(model, tensors, train, validation, config, 0)
            after = model.state_dict()
            self.assertFalse(torch.equal(before["embedding.weight"], after["embedding.weight"]))
            self.assertFalse(torch.equal(before["heads.family.0.weight"], after["heads.family.0.weight"]))
            for key in before:
                if key.startswith(("heads.position.", "heads.distance.")):
                    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
