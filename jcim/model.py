"""A small permutation-invariant message-passing model for queried atom graphs."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import N_DISTANCE_CLASSES, N_FEATURES


def tensorize(records: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Pad graphs once; padded atoms never contribute to messages or pooling."""
    max_atoms = max(len(record["x"]) for record in records)
    n = len(records)
    x = np.zeros((n, max_atoms, N_FEATURES), dtype=np.float32)
    adj = np.zeros((n, max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((n, max_atoms), dtype=np.float32)
    for i, record in enumerate(records):
        size = len(record["x"])
        x[i, :size] = record["x"]
        adj[i, :size, :size] = record["adj"]
        mask[i, :size] = 1
    arrays = {"x": x, "adj": adj, "mask": mask}
    arrays.update({key: np.array([r[key] for r in records], dtype=np.int64)
                   for key in ("y_family", "y_position", "y_distance")})
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def pool_nodes(h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return queried-fragment means and graph/query readouts, without atom order."""
    count = mask.sum(1, keepdim=True).clamp_min(1)
    graph_sum = (h * mask.unsqueeze(-1)).sum(1)
    local_mask = x[:, :, 15]
    local = (h * local_mask.unsqueeze(-1)).sum(1) / local_mask.sum(1, keepdim=True).clamp_min(1)
    queries = [(h * x[:, :, feature].unsqueeze(-1)).sum(1) for feature in (13, 14, 16, 17)]
    graph = torch.cat([graph_sum / count, graph_sum / count.sqrt(), local, *queries], dim=-1)
    return local, graph


class MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neighbors = torch.bmm(adj, h)
        updated = self.norm(h + self.update(torch.cat([h, neighbors], dim=-1)))
        return torch.relu(updated) * mask.unsqueeze(-1)


class MolecularGNN(nn.Module):
    """Six local message steps by default; global readouts expose marked queries.

    The longest functional-root path can exceed the message-passing radius.
    Pooled context helps prediction but does not remove this architectural limit.
    """

    def __init__(self, hidden_dim: int = 64, layers: int = 6):
        super().__init__()
        self.embedding = nn.Linear(N_FEATURES, hidden_dim)
        self.layers = nn.ModuleList([MessageLayer(hidden_dim) for _ in range(layers)])
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(hidden_dim * 7, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, classes))
                                   for name, classes in (("family", 8), ("position", 3),
                                                         ("distance", N_DISTANCE_CLASSES))})

    def forward(self, batch: dict[str, torch.Tensor], representations: bool = False) -> dict:
        x, adj, mask = batch["x"], batch["adj"], batch["mask"]
        h = torch.relu(self.embedding(x)) * mask.unsqueeze(-1)
        local, graph = pool_nodes(h, x, mask)
        local_layers, graph_layers = [local], [graph]
        for layer in self.layers:
            h = layer(h, adj, mask)
            local, graph = pool_nodes(h, x, mask)
            local_layers.append(local)
            graph_layers.append(graph)
        result = {name: head(graph) for name, head in self.heads.items()}
        if representations:
            result.update(local=local_layers, graph=graph_layers)
        return result
