"""Identity and fitted-control contracts for the frozen scale experiment."""

import copy
from collections import Counter
import unittest

import numpy as np

from jcim.scale_features import REGION_NAMES
from jcim.scale_run import build_direct_support, evaluate_model, validate_config, validate_support_sets


def fixture():
    records = [{"id": f"m{motif}_p{position}_r{copy_index}", "y_family": motif, "y_position": position}
               for motif in range(8) for position in range(3) for copy_index in range(8)]
    analysis = [index for index, row in enumerate(records) if int(row["id"].rsplit("r", 1)[1]) < 3]
    cells = lambda cell, pool: [index for index in pool
                               if (records[index]["y_family"] + records[index]["y_position"]) % 3 == cell]
    training = [index for index, row in enumerate(records) if 3 <= int(row["id"].rsplit("r", 1)[1]) < 7]
    splits = {"analysis": analysis, "iid_a": cells(0, analysis), "iid_b": cells(1, analysis),
              "ood_common": cells(2, analysis), "train_a": cells(0, training), "train_b": cells(1, training),
              "adaptation_pool": cells(2, training)}
    supports = {"sets": [{"experience": experience, "repeat": 0,
                          "new_ids": [records[index]["id"] for index in splits["adaptation_pool"]],
                          "old_ids": [records[index]["id"] for index in splits[f"train_{experience}"]]}
                         for experience in ("a", "b")]}
    config = {"seeds": [0], "support_repeats": [0], "primary_repeat": 0, "layers": [8], "primary_layer": 8,
              "hidden_dim": 64, "encoder_layers": 8, "alpha": 1.0, "direct_per_class": 2,
              "direct_sampling_base": 50000}
    return records, splits, supports, config


class ScaleRunTests(unittest.TestCase):
    def test_source_support_is_balanced_and_position_disjoint(self):
        records, splits, _, config = fixture()
        direct = build_direct_support(records, splits, config)
        self.assertEqual(direct, build_direct_support(records, splits, config))
        for sample in direct["sets"]:
            self.assertEqual(Counter(records[index]["y_family"] for index in sample["indices"]), {i: 2 for i in range(8)})
            self.assertEqual({records[index]["y_position"] for index in sample["indices"]}, {sample["source_position"]})
            target = {index for index in splits["analysis"] if records[index]["y_position"] != sample["source_position"]}
            self.assertFalse(target.intersection(sample["indices"]))
            self.assertEqual(sample["ids"], sorted(sample["ids"]))
        broken = copy.deepcopy(config)
        broken["direct_per_class"] = 4
        with self.assertRaises(ValueError):
            build_direct_support(records, splits, broken)

    def test_adaptation_support_rejects_test_overlap_and_missing_draws(self):
        records, splits, supports, config = fixture()
        self.assertEqual(len(validate_support_sets(records, splits, supports, config)), 2)
        broken = copy.deepcopy(supports)
        broken["sets"][0]["new_ids"][0] = records[splits["analysis"][0]]["id"]
        with self.assertRaises(ValueError):
            validate_support_sets(records, splits, broken, config)
        broken = copy.deepcopy(supports)
        broken["sets"].pop()
        with self.assertRaises(ValueError):
            validate_support_sets(records, splits, broken, config)

    def test_random_readout_uses_exactly_matched_support_and_all_output_labels(self):
        records, splits, supports, config = fixture()
        direct = build_direct_support(records, splits, config)
        indices = list(range(len(records)))
        labels = np.asarray([row["y_family"] for row in records])
        basic = np.pad(np.eye(8, dtype=np.float32)[labels], ((0, 0), (0, 56)))
        features = {scope: (np.tile(basic, (1, 7)) if scope == "legacy_graph" else basic)[None]
                    for scope in REGION_NAMES}
        predictions, fits = {}, []
        adaptation, transferred = evaluate_model("random_seed0", features, records, splits,
            supports["sets"], direct, indices, config, predictions, fits)
        self.assertEqual(len(adaptation), 84)
        self.assertEqual(len(transferred), 252)
        self.assertTrue(all(row["accuracy"] == 1.0 for row in adaptation + transferred))
        self.assertTrue(all(row["balanced_accuracy"] == 1.0 for row in adaptation + transferred))
        new = [row for row in adaptation if row["support"] == "new" and row["scale"] == "root"]
        self.assertEqual(len({row["prediction_key"] for row in new}), 1)
        direct_root = [row for row in transferred if row["scale"] == "root" and row["source_position"] == 0]
        self.assertEqual(len({row["prediction_key"] for row in direct_root}), 1)
        self.assertEqual({row["split"] for row in transferred},
                         {"target_all", "target_common_ood", "target_own_iid"})
        self.assertTrue(all(set(prediction.tolist()) == set(range(8)) for prediction in predictions.values()))

    def test_invalid_primary_settings_are_rejected(self):
        _, _, _, config = fixture()
        validate_config(config)
        for changes in ({"layers": [0]}, {"support_repeats": [1]}, {"alpha": 0}, {"seeds": [0, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(config | changes)


if __name__ == "__main__":
    unittest.main()
