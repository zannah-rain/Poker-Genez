"""The advantage network: a plain MLP regressing each of
strategy.ACTION_CATEGORIES' counterfactual regret, given the configured
feature subset (cfr_features.py) -- Single Deep CFR's only network (see
cfr_tree.py's module docstring for why there's just one, shared by every
seat, rather than a separate advantage net per player plus a second
strategy net).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import cfr_features
import strategy

DEFAULT_HIDDEN_SIZES = (128, 128)


class AdvantageNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        output_dim: int = strategy.NUM_ACTION_CATEGORIES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # linear output: this is a regression head, not a classifier
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features: shape (input_dim,). Returns raw regret estimates,
        shape (output_dim,) -- the interface cfr_tree.py's traversal
        expects from a `net`."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            out = self.forward(x).squeeze(0)
        return out.numpy()


def clone(net: AdvantageNet) -> AdvantageNet:
    """An independent copy of `net`'s weights -- e.g. a snapshot taken
    right before a training update, so it can still be played against
    (and, if that update didn't help, reverted to) after the original has
    since moved on. `load_state_dict` copies tensor *values* into the new
    module's own parameters rather than aliasing the source's, so later
    training on `net` can never leak into the clone."""
    cloned = AdvantageNet(input_dim=net.input_dim, hidden_sizes=net.hidden_sizes, output_dim=net.output_dim)
    cloned.load_state_dict(net.state_dict())
    cloned.eval()
    return cloned


@dataclass
class AdvantageNetConfig:
    feature_keys: tuple[str, ...]
    hidden_sizes: tuple[int, ...]
    table_size: int


def save(net: AdvantageNet, config: AdvantageNetConfig, path: str) -> None:
    """Writes `<path>.pt` (weights) and `<path>.json` (a self-describing
    sidecar: feature keys, architecture, table size) -- mirrors genome.py's
    save-what-you-need-to-reconstruct-with philosophy."""
    torch.save(net.state_dict(), f"{path}.pt")
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_keys": list(config.feature_keys),
                "hidden_sizes": list(config.hidden_sizes),
                "table_size": config.table_size,
                "action_categories": strategy.ACTION_CATEGORIES,
            },
            f,
        )


def load(path: str) -> tuple[AdvantageNet, AdvantageNetConfig]:
    with open(f"{path}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_keys = tuple(meta["feature_keys"])
    hidden_sizes = tuple(meta["hidden_sizes"])
    input_dim = len(cfr_features.feature_indices(feature_keys))

    net = AdvantageNet(input_dim=input_dim, hidden_sizes=hidden_sizes)
    net.load_state_dict(torch.load(f"{path}.pt", map_location="cpu"))
    net.eval()

    config = AdvantageNetConfig(feature_keys=feature_keys, hidden_sizes=hidden_sizes, table_size=meta["table_size"])
    return net, config
