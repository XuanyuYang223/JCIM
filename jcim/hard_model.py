"""Same small GNN family, with single-root local readouts and a wider path head."""

from torch import nn

from .hard_data import N_DISTANCE_CLASSES, N_FEATURES
from .model import MolecularGNN, tensorize


class HardMolecularGNN(MolecularGNN):
    """Query channel 15 marks one attachment atom, never a target-fragment mask.

    Eight message layers cover every nine-atom target motif (root radius six).
    Global readouts use the same pooling form as v1 to limit architecture changes.
    """

    def __init__(self, hidden_dim=64, layers=8):
        assert N_FEATURES == 18
        super().__init__(hidden_dim, layers)
        self.heads["distance"] = nn.Sequential(nn.Linear(hidden_dim * 7, hidden_dim),
                                               nn.ReLU(), nn.Linear(hidden_dim, N_DISTANCE_CLASSES))


tensorize_hard = tensorize
