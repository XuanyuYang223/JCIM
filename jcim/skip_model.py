"""Whole-graph shallow skips and a parameter-matched deep-only control.

All variants retain the inherited encoder and its original query features. The
additional branch pools every real atom; target-fragment membership is never
required. Extending an existing model copies its encoder and classifier exactly,
with zero initial weights for the appended branch.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .hard_model import HardMolecularGNN
from .model import pool_nodes


VARIANTS = ("deep", "deep_control", "shallow_skip")


def stable_graph_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid atoms, accumulating in float64 before restoring dtype."""
    weights = mask.to(torch.float64)
    numerator = (h.to(torch.float64) * weights.unsqueeze(-1)).sum(dim=1)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1)
    return (numerator / denominator).to(h.dtype)


class SkipMolecularGNN(HardMolecularGNN):
    """Compare inherited readout, shallow skip, and duplicate deep mean.

    ``deep`` uses the original seven-vector graph readout. ``shallow_skip``
    appends the whole-graph mean after message layer one; ``deep_control``
    appends an exact duplicate of the mean block in the final inherited graph
    readout. The latter two have identical classifier dimensions and parameter
    counts. Only the shallow mean uses stable accumulation; the inherited
    seven-vector readout keeps its original operations unchanged.
    """

    def __init__(self, variant: str = "deep", hidden_dim: int = 64, layers: int = 8):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown readout variant: {variant!r}.")
        if layers < 1:
            raise ValueError("The shallow skip requires at least one message layer.")
        super().__init__(hidden_dim=hidden_dim, layers=layers)
        self.variant = variant
        self.readout_dim = hidden_dim * (7 if variant == "deep" else 8)
        if variant != "deep":
            for head in self.heads.values():
                inherited = head[0]
                extended = nn.Linear(self.readout_dim, hidden_dim,
                                     device=inherited.weight.device, dtype=inherited.weight.dtype)
                with torch.no_grad():
                    extended.weight.zero_()
                    extended.weight[:, :hidden_dim * 7].copy_(inherited.weight)
                    extended.bias.copy_(inherited.bias)
                head[0] = extended

    @classmethod
    def from_base(cls, base: HardMolecularGNN, variant: str) -> "SkipMolecularGNN":
        """Copy an original encoder and all heads without sharing parameters.

        For both extended variants the extra first-layer columns start at zero.
        Thus copied classifiers initially compute the inherited prediction, and
        the two extended variants receive identical head parameter values.
        Copying consumes constructor random draws but does not change ``base``.
        """
        if not isinstance(base, HardMolecularGNN):
            raise TypeError("The source must be an original HardMolecularGNN.")
        if isinstance(base, SkipMolecularGNN) and base.variant != "deep":
            raise ValueError("Copy from an original-width classifier, not an extended head.")
        model = cls(variant=variant, hidden_dim=base.embedding.out_features, layers=len(base.layers))
        model.to(device=base.embedding.weight.device, dtype=base.embedding.weight.dtype)
        model.embedding.load_state_dict(copy.deepcopy(base.embedding.state_dict()))
        model.layers.load_state_dict(copy.deepcopy(base.layers.state_dict()))
        with torch.no_grad():
            for name, source in base.heads.items():
                target = model.heads[name]
                target[0].weight.zero_()
                target[0].weight[:, :source[0].in_features].copy_(source[0].weight)
                target[0].bias.copy_(source[0].bias)
                target[2].load_state_dict(copy.deepcopy(source[2].state_dict()))
        model.train(base.training)
        return model

    def forward(self, batch: dict[str, torch.Tensor], representations: bool = False,
                branch_override: torch.Tensor | str | None = None) -> dict:
        """Predict with an optional appended-branch intervention.

        ``branch_override='zero'`` removes only the appended readout branch.
        A tensor of identical shape, device, and dtype can replace that branch,
        for example by swapping its rows across molecules. These interventions
        leave all encoder states and the inherited deep readout unchanged.
        """
        if self.variant == "deep" and branch_override is not None:
            raise ValueError("The inherited deep readout has no appended branch.")
        x, adj, mask = batch["x"], batch["adj"], batch["mask"]
        h = torch.relu(self.embedding(x)) * mask.unsqueeze(-1)
        local, graph = pool_nodes(h, x, mask)
        local_layers, graph_layers = [local], [graph]
        shallow = None
        for depth, layer in enumerate(self.layers, start=1):
            h = layer(h, adj, mask)
            if depth == 1 and self.variant == "shallow_skip":
                shallow = stable_graph_mean(h, mask)
            local, graph = pool_nodes(h, x, mask)
            local_layers.append(local)
            graph_layers.append(graph)

        if self.variant == "deep":
            branch = h.new_empty((h.shape[0], 0))
        else:
            branch = shallow if self.variant == "shallow_skip" else graph[:, :self.embedding.out_features]
            if isinstance(branch_override, str):
                if branch_override != "zero":
                    raise ValueError("The only named branch intervention is 'zero'.")
                branch = torch.zeros_like(branch)
            elif branch_override is not None:
                if (not isinstance(branch_override, torch.Tensor)
                        or branch_override.shape != branch.shape
                        or branch_override.device != branch.device
                        or branch_override.dtype != branch.dtype):
                    raise ValueError("A replacement branch must have matching shape, device, and dtype.")
                branch = branch_override
        readout = graph if self.variant == "deep" else torch.cat((graph, branch), dim=-1)
        result = {name: head(readout) for name, head in self.heads.items()}
        if representations:
            result.update(local=local_layers, graph=graph_layers, readout=readout,
                          appended_branch=branch)
        return result
