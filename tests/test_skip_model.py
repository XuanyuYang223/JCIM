"""Controls for encoder preservation and whole-graph skip interventions."""

import unittest

import numpy as np
import torch

from jcim.hard_model import HardMolecularGNN, tensorize_hard
from jcim.skip_model import VARIANTS, SkipMolecularGNN, stable_graph_mean


def path_record(size):
    rng = np.random.default_rng(size + 43)
    features = rng.normal(size=(size, 18)).astype(np.float32)
    features[:, 13:] = 0
    for column, atom in ((13, 0), (14, 1), (15, 2), (16, 3), (17, 4)):
        features[atom, column] = 1
    adjacency = np.zeros((size, size), dtype=np.float32)
    for atom in range(size - 1):
        adjacency[atom, atom + 1] = adjacency[atom + 1, atom] = 1
    return {"x": features, "adj": adjacency,
            "y_family": 0, "y_position": 1, "y_distance": 3}


class SkipModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.records = [path_record(6), path_record(9)]
        cls.batch = tensorize_hard(cls.records, torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(101)
        self.base = HardMolecularGNN(hidden_dim=8, layers=3).eval()

    def test_all_variants_preserve_copied_encoder_without_aliasing(self):
        original = {name: value.clone() for name, value in self.base.state_dict().items()}
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = SkipMolecularGNN.from_base(self.base, variant)
                for name, value in model.state_dict().items():
                    if not name.startswith("heads."):
                        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
                        self.assertNotEqual(value.data_ptr(), self.base.state_dict()[name].data_ptr())
                self.assertFalse(model.training)
                with torch.no_grad():
                    model.embedding.weight.add_(1)
                for name, value in self.base.state_dict().items():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_initial_outputs_match_original_and_extended_heads_are_paired(self):
        with torch.no_grad():
            expected = self.base(self.batch, representations=True)
        models = {variant: SkipMolecularGNN.from_base(self.base, variant) for variant in VARIANTS}
        for variant, model in models.items():
            with self.subTest(variant=variant), torch.no_grad():
                result = model(self.batch, representations=True)
                for task in ("family", "position", "distance"):
                    torch.testing.assert_close(result[task], expected[task], rtol=2e-6, atol=2e-6)
                    torch.testing.assert_close(result[task].argmax(1), expected[task].argmax(1), rtol=0, atol=0)
                for actual, inherited in zip(result["graph"], expected["graph"]):
                    torch.testing.assert_close(actual, inherited, rtol=0, atol=0)
        shallow, control = models["shallow_skip"], models["deep_control"]
        self.assertEqual(sum(p.numel() for p in shallow.parameters()),
                         sum(p.numel() for p in control.parameters()))
        self.assertGreater(sum(p.numel() for p in shallow.parameters()),
                           sum(p.numel() for p in models["deep"].parameters()))
        for name, value in shallow.heads.state_dict().items():
            torch.testing.assert_close(value, control.heads.state_dict()[name], rtol=0, atol=0)
        for head in shallow.heads.values():
            self.assertEqual(torch.count_nonzero(head[0].weight[:, 56:]).item(), 0)

    def test_deep_control_adds_exact_duplicate_of_existing_mean(self):
        model = SkipMolecularGNN.from_base(self.base, "deep_control")
        with torch.no_grad():
            result = model(self.batch, representations=True)
        self.assertEqual(model.readout_dim, 64)
        torch.testing.assert_close(result["appended_branch"], result["graph"][-1][:, :8], rtol=0, atol=0)
        torch.testing.assert_close(result["readout"][:, 56:], result["readout"][:, :8], rtol=0, atol=0)

    def test_shallow_branch_uses_every_real_atom_at_first_message_layer(self):
        model = SkipMolecularGNN.from_base(self.base, "shallow_skip")
        with torch.no_grad():
            x, mask = self.batch["x"], self.batch["mask"]
            h = torch.relu(model.embedding(x)) * mask.unsqueeze(-1)
            first = model.layers[0](h, self.batch["adj"], mask)
            expected = torch.stack([first[i, :len(record["x"])].double().mean(0).float()
                                    for i, record in enumerate(self.records)])
            result = model(self.batch, representations=True)
        torch.testing.assert_close(result["appended_branch"], expected, rtol=0, atol=0)
        self.assertEqual(model.readout_dim, 64)

    def test_atom_permutation_and_padding_do_not_change_prediction(self):
        record = self.records[0]
        order = np.random.default_rng(12).permutation(len(record["x"]))
        reordered = {**record, "x": record["x"][order],
                     "adj": record["adj"][np.ix_(order, order)]}
        for variant in VARIANTS:
            with self.subTest(variant=variant), torch.no_grad():
                model = SkipMolecularGNN.from_base(self.base, variant)
                # Nonzero branch weights ensure invariance also covers the added path.
                if variant != "deep":
                    for head in model.heads.values():
                        head[0].weight[:, 56:].fill_(0.15)
                original = model(tensorize_hard([record], torch.device("cpu")), representations=True)
                permuted = model(tensorize_hard([reordered], torch.device("cpu")), representations=True)
                padded = model(self.batch, representations=True)
                for key in ("family", "position", "distance", "readout", "appended_branch"):
                    torch.testing.assert_close(original[key], permuted[key], rtol=3e-6, atol=3e-6)
                    torch.testing.assert_close(original[key][0], padded[key][0], rtol=3e-6, atol=3e-6)

    def test_zero_and_swap_interventions_leave_deep_features_unchanged(self):
        model = SkipMolecularGNN.from_base(self.base, "shallow_skip")
        with torch.no_grad():
            for head in model.heads.values():
                head[0].weight[:, 56:].fill_(0.2)
            ordinary = model(self.batch, representations=True)
            zeroed = model(self.batch, representations=True, branch_override="zero")
            swapped_branch = ordinary["appended_branch"].flip(0)
            swapped = model(self.batch, representations=True, branch_override=swapped_branch)
        for result in (zeroed, swapped):
            torch.testing.assert_close(result["readout"][:, :56], ordinary["readout"][:, :56], rtol=0, atol=0)
            for before, after in zip(result["local"], ordinary["local"]):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(zeroed["appended_branch"]).item(), 0)
        torch.testing.assert_close(swapped["appended_branch"], swapped_branch, rtol=0, atol=0)
        self.assertFalse(torch.equal(ordinary["family"], zeroed["family"]))
        for task in ("family", "position", "distance"):
            with torch.no_grad():
                expected = model.heads[task](zeroed["readout"])
            torch.testing.assert_close(zeroed[task], expected, rtol=0, atol=0)

    def test_shallow_only_loss_backpropagates_through_first_layer(self):
        model = SkipMolecularGNN.from_base(self.base, "shallow_skip")
        head = model.heads["family"]
        with torch.no_grad():
            head[0].weight.zero_()
            head[0].weight[:, 56:].fill_(0.1)
            head[0].bias.fill_(1)
            head[2].weight.fill_(0.1)
        model(self.batch)["family"].sum().backward()
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.layers[0].update[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.layers[-1].update[0].weight.grad.abs().sum().item(), 0)

    def test_head_only_update_preserves_encoder_with_active_skip(self):
        model = SkipMolecularGNN.from_base(self.base, "shallow_skip")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("heads."))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.heads.parameters(), lr=0.01)
        result = model(self.batch)
        loss = sum(torch.nn.functional.cross_entropy(result[task], self.batch[label])
                   for task, label in (("family", "y_family"), ("position", "y_position"),
                                       ("distance", "y_distance")))
        loss.backward()
        optimizer.step()
        for name, value in model.state_dict().items():
            if not name.startswith("heads."):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.heads["family"][0].weight, before["heads.family.0.weight"]))

    def test_stable_mean_ignores_padding_and_preserves_autograd(self):
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1e12, -1e12]]], requires_grad=True)
        mean = stable_graph_mean(values, torch.tensor([[1.0, 1.0, 0.0]]))
        torch.testing.assert_close(mean, torch.tensor([[2.0, 3.0]]), rtol=0, atol=0)
        self.assertEqual(mean.dtype, values.dtype)
        mean.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]]),
                                   rtol=0, atol=0)

    def test_rejects_undefined_or_malformed_interventions(self):
        with self.assertRaises(ValueError):
            SkipMolecularGNN("unknown")
        with self.assertRaises(ValueError):
            SkipMolecularGNN("shallow_skip", layers=0)
        original = SkipMolecularGNN.from_base(self.base, "deep")
        with self.assertRaises(ValueError):
            original(self.batch, branch_override="zero")
        model = SkipMolecularGNN.from_base(self.base, "shallow_skip")
        for override in ("unknown", torch.zeros(2, 7), torch.zeros(2, 8, dtype=torch.float64), 0):
            with self.subTest(override_type=type(override).__name__), self.assertRaises(ValueError):
                model(self.batch, branch_override=override)
        with self.assertRaises(ValueError):
            SkipMolecularGNN.from_base(model, "deep_control")

    def test_copy_preserves_source_dtype_and_mode(self):
        source = self.base.double().train()
        copied = SkipMolecularGNN.from_base(source, "shallow_skip")
        self.assertTrue(copied.training)
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in copied.parameters()))
        batch = {name: value.double() if value.is_floating_point() else value
                 for name, value in self.batch.items()}
        self.assertEqual(copied(batch)["family"].dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
