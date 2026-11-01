"""Scientific identity and cache controls for the boundary-free intervention."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from jcim.hard_model import HardMolecularGNN
from jcim.skip_model import SkipMolecularGNN
from jcim.skip_run import build_plan, cached_training, training_orders


def identity_fixture():
    records, splits = [], {}
    def add(name, count, cell):
        indices = []
        for number in range(count):
            label = number % 8
            indices.append(len(records))
            records.append({"id": f"m{len(records):05d}", "y_family": label, "y_position": (cell - label) % 3})
        splits[name] = indices
    for name, count, cell in (("train_a", 1386, 0), ("train_b", 1386, 1),
                              ("val_a", 16, 0), ("val_b", 16, 1),
                              ("iid_a", 16, 0), ("iid_b", 16, 1),
                              ("ood_common", 24, 2), ("adaptation_pool", 8, 2)):
        add(name, count, cell)
    splits["analysis"] = splits["iid_a"] + splits["iid_b"] + splits["ood_common"]
    config = {"experiences": ["a", "b"], "seeds": [0], "batch_size": 128,
              "branch_permutation_seed_offset": 90000, "epochs": 2, "order_seed_offset": 1701}
    return records, splits, config


class SkipRunTests(unittest.TestCase):
    def test_source_plan_preserves_original_ids_and_rejects_holdout_labels(self):
        records, splits, config = identity_fixture()
        plan = build_plan(records, splits, config)
        self.assertEqual(plan["experiences"]["a"]["train"]["indices"], splits["train_a"])
        self.assertEqual(sorted(plan["branch_permutations"]["0"]["indices"]), list(range(len(splits["analysis"]))))
        self.assertEqual(plan, build_plan(records, splits, config))
        broken = copy.deepcopy(splits)
        broken["train_a"][0] = broken["ood_common"][0]
        with self.assertRaises(ValueError):
            build_plan(records, broken, config)

    def test_orders_match_original_rng_and_preserve_every_epoch(self):
        _, splits, config = identity_fixture()
        orders = training_orders(splits, config)
        for experience in config["experiences"]:
            rng = np.random.default_rng(1701)
            expected = np.asarray([rng.permutation(splits[f"train_{experience}"]) for _ in range(2)], dtype=np.int32)
            self.assertTrue(np.array_equal(orders[f"{experience}_seed0"], expected))
            self.assertTrue(all(set(order.tolist()) == set(splits[f"train_{experience}"]) for order in expected))

    def test_cached_frozen_head_update_matches_full_graph_update(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(86)
            base = HardMolecularGNN(64, 8)
            model = SkipMolecularGNN.from_base(base, "shallow_skip")
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("heads."))
            reference = copy.deepcopy(model)
            count, atoms = 4, 6
            features = torch.randn(count, atoms, 18)
            features[:, :, 13:] = 0
            for column, atom in ((13, 0), (14, 1), (15, 2), (16, 3), (17, 4)):
                features[:, atom, column] = 1
            adjacency = torch.zeros(count, atoms, atoms)
            for atom in range(atoms - 1):
                adjacency[:, atom, atom + 1] = adjacency[:, atom + 1, atom] = 1
            tensors = {"x": features, "adj": adjacency, "mask": torch.ones(count, atoms),
                       "y_family": torch.tensor([0, 1, 2, 3]), "y_position": torch.tensor([0, 1, 2, 0]),
                       "y_distance": torch.tensor([4, 5, 6, 7])}
            with torch.no_grad():
                cache = model(tensors, representations=True)["readout"].detach()
            encoder_before = {name: value.clone() for name, value in model.state_dict().items() if not name.startswith("heads.")}
            config = {"batch_size": 128, "learning_rate": 0.0015, "weight_decay": 0.0001, "gradient_clip": 5.0}
            chosen = [2, 0, 3, 1]
            params = [parameter for parameter in reference.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=0.0015, weight_decay=0.0001)
            logits = reference({name: value[chosen] for name, value in tensors.items()})
            loss = sum(F.cross_entropy(logits[task], tensors[f"y_{task}"][chosen])
                       for task in ("family", "position", "distance"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            with tempfile.TemporaryDirectory() as directory:
                updates, _ = cached_training(model, tensors, cache, list(range(count)), np.asarray([chosen]),
                    [0, 1], config, Path(directory) / "history.json", "cache_control")
            self.assertEqual(updates, 1)
            for name, value in model.heads.state_dict().items():
                torch.testing.assert_close(value, reference.heads.state_dict()[name], rtol=2e-6, atol=2e-7)
            for name, value in encoder_before.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
