"""Frozen full-input node states pooled over supervised molecular regions.

Pooling membership is analysis metadata, never an encoder input. A small pooled
region does not imply a local input: after ``layer`` message steps its states
can depend on the union of the region's ``layer``-hop neighborhoods. Even at
layer zero, the original degree and hydrogen features contain boundary chemistry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .model import pool_nodes


POOLED_REGIONS = ("root", "target", "target_plus1", "target_plus2", "target_plus3", "whole_graph")
REGION_NAMES = (*POOLED_REGIONS, "legacy_graph")
DEFAULT_LAYERS = tuple(range(9))


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _indices(indices, count):
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("Record indices must be a nonempty sequence without duplicates.")
    if np.any(selected < 0) or np.any(selected >= count):
        raise ValueError("A selected record index is outside the molecular archive.")
    return selected


def _layers(layers):
    selected = tuple(int(layer) for layer in layers)
    if not selected or len(set(selected)) != len(selected) or min(selected) < 0:
        raise ValueError("Layers must be distinct nonnegative message depths.")
    return selected


def expand_regions(adjacency, regions, atom_mask, steps=1):
    """Expand each Boolean region by graph distance, excluding padded atoms.

    All arrays have a batch axis. ``regions`` and ``atom_mask`` have shape
    ``[n, atoms]``; ``adjacency`` has shape ``[n, atoms, atoms]``. Nonzero
    adjacency values indicate graph edges; no paths can pass through padding.
    """
    adjacency = np.asarray(adjacency) != 0
    atom_mask = np.asarray(atom_mask, dtype=bool)
    regions = np.asarray(regions, dtype=bool)
    if regions.ndim != 2 or regions.shape != atom_mask.shape or adjacency.shape != (*regions.shape, regions.shape[1]):
        raise ValueError("Region, adjacency, and atom-mask dimensions disagree.")
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("Neighborhood expansion requires a nonnegative integer depth.")
    adjacency = adjacency & atom_mask[:, :, None] & atom_mask[:, None, :]
    reached = regions.copy() & atom_mask
    for _ in range(int(steps)):
        reached |= np.any(adjacency & reached[:, None, :], axis=2)
    return reached


def build_scale_regions(records: Sequence[dict], adjacency, atom_mask, indices,
                        layers=DEFAULT_LAYERS):
    """Build oracle pooling masks and per-molecule receptive-support metadata.

    ``adjacency`` and ``atom_mask`` index the complete record archive; the
    returned masks follow ``indices`` exactly. ``root`` is the original target
    attachment atom. ``target`` uses the known target-fragment membership;
    ``target_plus1/2/3`` add graph neighbors to that complete fragment.

    The returned pair is ``(region_masks, locality_rows)``. Masks include only
    the six 64-dimensional mean pools; the separate legacy readout concatenates
    seven vectors and has whole-graph receptive support at every depth.
    """
    selected = _indices(indices, len(records))
    layers = _layers(layers)
    adjacency = _numpy(adjacency)
    atom_mask = _numpy(atom_mask).astype(bool, copy=False)
    if atom_mask.ndim != 2 or len(atom_mask) != len(records) or adjacency.shape != (*atom_mask.shape, atom_mask.shape[1]):
        raise ValueError("Archive adjacency and atom-mask dimensions disagree with records.")
    selected_adj = adjacency[selected] != 0
    valid = atom_mask[selected]
    root = np.zeros_like(valid)
    target = np.zeros_like(valid)
    for row, index in enumerate(selected):
        record = records[int(index)]
        root_index = int(record["target_attachment_idx"])
        fragment = np.asarray(record["target_fragment_indices"], dtype=np.int64)
        if (fragment.ndim != 1 or len(fragment) == 0 or len(set(fragment.tolist())) != len(fragment)
                or np.any(fragment < 0) or np.any(fragment >= valid.shape[1])
                or not 0 <= root_index < valid.shape[1]
                or root_index not in fragment or not valid[row, fragment].all()):
            raise ValueError("Target membership must contain unique valid atoms and its attachment root.")
        root[row, root_index] = True
        target[row, fragment] = True
    masks = {"root": root, "target": target}
    expanded = target
    for radius in range(1, 4):
        expanded = expand_regions(selected_adj, expanded, valid)
        masks[f"target_plus{radius}"] = expanded
    masks["whole_graph"] = valid.copy()

    rows = []
    for region in REGION_NAMES:
        pooled = valid if region == "legacy_graph" else masks[region]
        reached = pooled.copy()
        previous = 0
        for layer in sorted(layers):
            reached = expand_regions(selected_adj, reached, valid, layer - previous)
            previous = layer
            pooled_count = pooled.sum(axis=1)
            receptive_count = reached.sum(axis=1)
            outside_target = (reached & ~target).sum(axis=1)
            outside_pool = (reached & ~pooled).sum(axis=1)
            pooled_outside_target = (pooled & ~target).sum(axis=1)
            for row, index in enumerate(selected):
                rows.append({"index": int(index), "id": records[int(index)]["id"], "region": region,
                    "layer": layer, "pool_atom_count": int(pooled_count[row]),
                    "target_atom_count": int(target[row].sum()), "molecule_atom_count": int(valid[row].sum()),
                    "pool_outside_target_count": int(pooled_outside_target[row]),
                    "input_receptive_atom_count": int(receptive_count[row]),
                    "receptive_outside_target_count": int(outside_target[row]),
                    "receptive_outside_pool_count": int(outside_pool[row]),
                    "receptive_covers_full_graph": bool(receptive_count[row] == valid[row].sum())})
    return masks, rows


@torch.no_grad()
def extract_scale_features(model, tensors: Mapping[str, torch.Tensor], indices,
                           region_masks: Mapping[str, np.ndarray], layers=DEFAULT_LAYERS,
                           batch_size=128):
    """Return frozen full-input features as ``region -> [layer, sample, dim]``.

    ``region_masks`` must follow the same ``indices`` order. Mean-pooled regions
    all have the encoder's hidden dimension. ``legacy_graph`` uses the exact
    inherited seven-vector readout (448 dimensions for hidden dimension 64).
    The complete original node features, query markers, adjacency, and atom
    masks enter every encoder pass. Oracle masks are applied only afterward.
    Non-root means sum in float64 before returning float32. This suppresses
    atom-order roundoff fingerprints in equal-feature-multiset controls. The
    root and legacy readouts retain the inherited operations exactly. Model
    parameters, input tensors, and the model's train/eval mode are kept.

    The final batch repeats valid selected graphs to reach ``batch_size`` and
    discards the repeated outputs. Keeping the dense encoder batch shape fixed
    prevents CUDA matrix-multiplication rounding from marking tail-batch rows.
    This model has independent graph messages and per-node LayerNorm, so the
    repeated graphs cannot alter another graph's mathematical computation.
    """
    selected = _indices(indices, len(tensors["x"]))
    layers = _layers(layers)
    if max(layers) > len(model.layers):
        raise ValueError("A requested layer exceeds the encoder's message depth.")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("Feature extraction requires a positive integer batch size.")
    if set(region_masks) != set(POOLED_REGIONS):
        raise ValueError("Feature extraction needs exactly the declared mean-pooling regions.")
    atom_count = tensors["x"].shape[1]
    region_arrays = {name: np.asarray(region_masks[name], dtype=bool) for name in POOLED_REGIONS}
    if any(value.shape != (len(selected), atom_count) or np.any(value.sum(axis=1) == 0)
           for value in region_arrays.values()):
        raise ValueError("Each pooled region must be nonempty and match the selected padded graphs.")
    hidden_dim = model.embedding.out_features
    result = {name: np.empty((len(layers), len(selected), hidden_dim * (7 if name == "legacy_graph" else 1)),
                             dtype=np.float32) for name in REGION_NAMES}
    layer_positions = {layer: position for position, layer in enumerate(layers)}
    was_training = model.training
    model.eval()
    try:
        for offset in range(0, len(selected), int(batch_size)):
            stop = min(offset + int(batch_size), len(selected))
            genuine_count = stop - offset
            source_rows = np.arange(offset, stop)
            if genuine_count < batch_size:
                source_rows = np.resize(source_rows, int(batch_size))
            rows = selected[source_rows]
            x, adj, mask = (tensors[name][rows] for name in ("x", "adj", "mask"))
            valid = mask.bool()
            pooled_masks = {name: torch.as_tensor(value[source_rows], device=x.device)
                            for name, value in region_arrays.items()}
            if any(torch.any(region & ~valid).item() for region in pooled_masks.values()):
                raise ValueError("A pooling mask includes a padded atom.")
            if not torch.equal(pooled_masks["root"], x[:, :, 15].bool()) or not torch.all(x[:, :, 15].sum(1) == 1):
                raise ValueError("The root pooling mask must match the original unique attachment query.")
            h = torch.relu(model.embedding(x)) * mask.unsqueeze(-1)
            for depth in range(max(layers) + 1):
                if depth in layer_positions:
                    layer_row = layer_positions[depth]
                    for name, region in pooled_masks.items():
                        pooling_dtype = h.dtype if name == "root" else torch.float64
                        weights = region.to(pooling_dtype)
                        pooled = ((h.to(pooling_dtype) * weights.unsqueeze(-1)).sum(1)
                                  / weights.sum(1, keepdim=True)).to(h.dtype)
                        result[name][layer_row, offset:stop] = pooled[:genuine_count].cpu().numpy()
                    _, legacy = pool_nodes(h, x, mask)
                    result["legacy_graph"][layer_row, offset:stop] = legacy[:genuine_count].cpu().numpy()
                if depth < max(layers):
                    h = model.layers[depth](h, adj, mask)
    finally:
        model.train(was_training)
    return result
