"""Independent checks for frozen spatial pooling and receptive-field support."""

import unittest

import numpy as np
import torch

from jcim.hard_model import HardMolecularGNN, tensorize_hard
from jcim.scale_features import (
    POOLED_REGIONS, REGION_NAMES, build_scale_regions, expand_regions, extract_scale_features,
)


def path_record(identifier, size):
    rng = np.random.default_rng(size)
    x = rng.uniform(size=(size, 18)).astype(np.float32)
    x[:, 13:18] = 0
    for column, atom in ((13, 0), (14, 1), (15, 2), (16, 0), (17, 4)):
        x[atom, column] = 1
    adjacency = np.zeros((size, size), dtype=np.float32)
    for atom in range(size - 1):
        adjacency[atom, atom + 1] = adjacency[atom + 1, atom] = 1
    return {"id": identifier, "x": x, "adj": adjacency, "target_attachment_idx": 2,
            "target_fragment_indices": [2, 3, 4], "y_family": 0, "y_position": 0, "y_distance": 4}


class ScaleFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.records = [path_record("short", 7), path_record("long", 9)]
        cls.tensors = tensorize_hard(cls.records, torch.device("cpu"))
        torch.manual_seed(102)
        cls.model = HardMolecularGNN(hidden_dim=8, layers=8).eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def regions(self, indices=(0, 1), layers=tuple(range(9))):
        return build_scale_regions(self.records, self.tensors["adj"], self.tensors["mask"], indices, layers)

    def test_bond_expansions_are_nested_and_ignore_padding(self):
        masks, _ = self.regions()
        np.testing.assert_array_equal(np.flatnonzero(masks["root"][0]), [2])
        np.testing.assert_array_equal(np.flatnonzero(masks["target"][0]), [2, 3, 4])
        np.testing.assert_array_equal(np.flatnonzero(masks["target_plus1"][0]), [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(np.flatnonzero(masks["target_plus2"][0]), range(7))
        previous = masks["root"]
        for name in POOLED_REGIONS[1:]:
            self.assertTrue(np.all(~previous | masks[name]))
            self.assertFalse(masks[name][0, 7:].any())
            previous = masks[name]
        # Even malformed padded edges cannot create a path through absent atoms.
        adjacency = self.tensors["adj"].numpy().copy()
        adjacency[0, 6, 7] = adjacency[0, 7, 6] = 1
        expanded = expand_regions(adjacency, masks["target"], self.tensors["mask"].numpy(), 20)
        np.testing.assert_array_equal(expanded, masks["whole_graph"])

    def test_pooling_support_is_distinguished_from_full_input_receptive_support(self):
        _, rows = self.regions()
        lookup = {(row["index"], row["region"], row["layer"]): row for row in rows}
        shallow = lookup[0, "target", 0]
        self.assertEqual((shallow["pool_atom_count"], shallow["input_receptive_atom_count"],
                          shallow["receptive_outside_target_count"]), (3, 3, 0))
        expanded = lookup[0, "root", 2]
        self.assertEqual((expanded["pool_atom_count"], expanded["input_receptive_atom_count"],
                          expanded["receptive_outside_target_count"]), (1, 5, 2))
        deep = lookup[0, "root", 8]
        self.assertEqual(deep["pool_atom_count"], 1)
        self.assertTrue(deep["receptive_covers_full_graph"])
        self.assertEqual(deep["receptive_outside_pool_count"], 6)
        for layer in (0, 2, 4, 8):
            self.assertEqual(lookup[0, "legacy_graph", layer]["input_receptive_atom_count"], 7)

    def test_root_and_legacy_features_exactly_reproduce_inherited_model(self):
        indices = [1, 0]
        layers = (8, 0, 4, 2)
        masks, _ = self.regions(indices, layers)
        extracted = extract_scale_features(self.model, self.tensors, indices, masks, layers, batch_size=2)
        with torch.no_grad():
            result = self.model({name: value[indices] for name, value in self.tensors.items()}, representations=True)
        self.assertEqual(set(extracted), set(REGION_NAMES))
        self.assertEqual(extracted["root"].shape, (4, 2, 8))
        self.assertEqual(extracted["legacy_graph"].shape, (4, 2, 56))
        for row, layer in enumerate(layers):
            np.testing.assert_array_equal(extracted["root"][row], result["local"][layer].numpy())
            np.testing.assert_array_equal(extracted["legacy_graph"][row], result["graph"][layer].numpy())

    def test_tail_batch_repeats_valid_graphs_to_preserve_encoder_shape(self):
        indices = [1, 0]
        masks, _ = self.regions(indices)
        seen_shapes = []
        hook = self.model.embedding.register_forward_pre_hook(
            lambda module, inputs: seen_shapes.append(tuple(inputs[0].shape)))
        try:
            extracted = extract_scale_features(self.model, self.tensors, indices, masks, batch_size=3)
        finally:
            hook.remove()
        self.assertEqual(seen_shapes, [(3, self.tensors["x"].shape[1], 18)])
        repeated = [1, 0, 1]
        with torch.no_grad():
            expected = self.model({name: value[repeated] for name, value in self.tensors.items()}, representations=True)
        for layer in range(9):
            np.testing.assert_array_equal(extracted["root"][layer], expected["local"][layer][:2].numpy())
            np.testing.assert_array_equal(extracted["legacy_graph"][layer], expected["graph"][layer][:2].numpy())

    def test_means_match_independent_node_state_calculation(self):
        masks, _ = self.regions()
        extracted = extract_scale_features(self.model, self.tensors, [0, 1], masks)
        with torch.no_grad():
            h = torch.relu(self.model.embedding(self.tensors["x"])) * self.tensors["mask"].unsqueeze(-1)
            for depth in range(9):
                for region in POOLED_REGIONS:
                    for sample in range(2):
                        selected = h[sample, masks[region][sample]]
                        expected = (selected if region == "root" else selected.double()).mean(0).float().numpy()
                        np.testing.assert_allclose(extracted[region][depth, sample], expected, rtol=1e-6, atol=1e-7)
                if depth < 8:
                    h = self.model.layers[depth](h, self.tensors["adj"], self.tensors["mask"])

    def test_extraction_preserves_parameters_inputs_mode_and_batch_order(self):
        masks, _ = self.regions()
        old_state = {name: value.clone() for name, value in self.model.state_dict().items()}
        old_inputs = {name: value.clone() for name, value in self.tensors.items()}
        self.model.train()
        try:
            singleton = extract_scale_features(self.model, self.tensors, [0, 1], masks, batch_size=1)
            self.assertTrue(self.model.training)
            batched = extract_scale_features(self.model, self.tensors, [0, 1], masks, batch_size=2)
            for region in REGION_NAMES:
                np.testing.assert_allclose(singleton[region], batched[region], atol=2e-6, rtol=2e-6)
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(value, old_state[name], rtol=0, atol=0)
            for name, value in self.tensors.items():
                torch.testing.assert_close(value, old_inputs[name], rtol=0, atol=0)
        finally:
            self.model.eval()

    def test_extraction_is_equivariant_to_joint_atom_and_metadata_permutation(self):
        masks, _ = self.regions()
        original = extract_scale_features(self.model, self.tensors, [0, 1], masks)
        permutations = [np.random.default_rng(40 + i).permutation(len(record["x"]))
                        for i, record in enumerate(self.records)]
        permuted = []
        for record, permutation in zip(self.records, permutations):
            inverse = np.argsort(permutation)
            permuted.append({**record, "x": record["x"][permutation],
                "adj": record["adj"][np.ix_(permutation, permutation)],
                "target_attachment_idx": int(inverse[record["target_attachment_idx"]]),
                "target_fragment_indices": inverse[record["target_fragment_indices"]].tolist()})
        tensors = tensorize_hard(permuted, torch.device("cpu"))
        masks, _ = build_scale_regions(permuted, tensors["adj"], tensors["mask"], [0, 1])
        reordered = extract_scale_features(self.model, tensors, [0, 1], masks)
        for region in REGION_NAMES:
            np.testing.assert_allclose(original[region], reordered[region], rtol=3e-6, atol=3e-6)

    def test_equal_noninteger_node_state_multisets_have_bit_identical_means(self):
        record = path_record("original", 9)
        permutation = np.array([7, 0, 1, 4, 2, 5, 8, 6, 3])
        inverse = np.argsort(permutation)
        permuted = {**record, "id": "permuted", "x": record["x"][permutation],
            "adj": record["adj"][np.ix_(permutation, permutation)],
            "target_attachment_idx": int(inverse[record["target_attachment_idx"]]),
            "target_fragment_indices": inverse[record["target_fragment_indices"]].tolist()}
        records = [record, permuted]
        tensors = tensorize_hard(records, torch.device("cpu"))
        masks, _ = build_scale_regions(records, tensors["adj"], tensors["mask"], [0, 1], layers=(0,))
        with torch.no_grad():
            h = torch.relu(self.model.embedding(tensors["x"]))
        # Floating embedding values make integer-only invariance checks inadequate.
        self.assertTrue(torch.any(h != h.round()).item())
        torch.testing.assert_close(h[0][permutation], h[1], atol=0, rtol=0)
        extracted = extract_scale_features(self.model, tensors, [0, 1], masks, layers=(0,))
        for region in POOLED_REGIONS:
            with self.subTest(region=region):
                np.testing.assert_array_equal(extracted[region][0, 0], extracted[region][0, 1])

    def test_malformed_indices_membership_queries_and_padding_are_rejected(self):
        for indices in ([], [0, 0], [-1], [2]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.regions(indices)
        for fragment in ([3, 4], [2, 3, 3], [2, 3, 8]):
            records = [{**self.records[0], "target_fragment_indices": fragment}, self.records[1]]
            with self.subTest(fragment=fragment), self.assertRaisesRegex(ValueError, "Target membership"):
                build_scale_regions(records, self.tensors["adj"], self.tensors["mask"], [0, 1])
        masks, _ = self.regions()
        bad = {name: value.copy() for name, value in masks.items()}
        bad["target"][0, 7] = True
        with self.assertRaisesRegex(ValueError, "padded atom"):
            extract_scale_features(self.model, self.tensors, [0, 1], bad)
        bad = {name: value.copy() for name, value in masks.items()}
        bad["root"][0] = False
        bad["root"][0, 1] = True
        with self.assertRaisesRegex(ValueError, "unique attachment"):
            extract_scale_features(self.model, self.tensors, [0, 1], bad)
        with self.assertRaisesRegex(ValueError, "message depth"):
            extract_scale_features(self.model, self.tensors, [0, 1], masks, layers=(9,))


if __name__ == "__main__":
    unittest.main()
