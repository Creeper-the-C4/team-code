"""Residual bidirectional GRU for offline, trial-bounded orientation reconstruction."""

import torch
from torch import nn

from scripts.preprocessing.independent_data_access import SENSORS
from .rotation_6d import geodesic_loss, rot6d_to_matrix


class SensorDropoutBiGRU(nn.Module):
    def __init__(self, hidden=192, layers=2, dropout=0.12):
        super().__init__()
        width = 2 * hidden
        self.input = nn.Sequential(nn.Linear(13 * len(SENSORS), width), nn.LayerNorm(width))
        self.layers = nn.ModuleList(nn.GRU(width, hidden, batch_first=True, bidirectional=True)
                                    for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 6 * len(SENSORS))
        with torch.no_grad():
            identity = torch.tensor([1., 0., 0., 0., 1., 0.])
            self.head.bias.copy_(identity.repeat(len(SENSORS)))

    def forward(self, features):
        sequence = self.input(features)
        for layer, norm in zip(self.layers, self.norms):
            update, _ = layer(sequence)
            sequence = norm(sequence + self.dropout(update))
        columns = self.head(sequence).reshape(*features.shape[:2], len(SENSORS), 6)
        return rot6d_to_matrix(columns)


def reconstruction_loss(predicted, target, observed, observed_weight=0.05):
    """Mean per layout, so layouts with many holes do not outweigh single holes."""
    angles = geodesic_loss(predicted.float(), target.float(), reduction="none").mean(dim=1)
    hidden = ~observed
    main = (angles * hidden).sum(dim=1) / hidden.sum(dim=1).clamp_min(1)
    regularizer = (angles * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1)
    return (main + observed_weight * regularizer).mean()
