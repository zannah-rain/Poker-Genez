"""Fixed-capacity uniform reservoir buffer (classic Algorithm R) of
(features, regrets, legal_mask, t) samples collected during cfr_tree
traversals -- Single Deep CFR's only memory (one shared buffer for the one
shared advantage network, see cfr_tree.py's module docstring). Sampling for
insertion is uniform over every sample ever seen; the linear-CFR-style
"later iterations matter more" weighting instead happens at *training*
time, via each sample's stored `t` scaling its loss term (see cfr_train.py).
"""

from __future__ import annotations

import numpy as np
import torch


class ReservoirBuffer:
    def __init__(self, capacity: int, feature_dim: int, num_actions: int, rng: np.random.Generator):
        self.capacity = capacity
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.regrets = np.zeros((capacity, num_actions), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.weights = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.n_seen = 0
        self.rng = rng

    def add(self, features: np.ndarray, regrets: np.ndarray, legal_mask: np.ndarray, t: float) -> None:
        i = self.n_seen
        if i < self.capacity:
            idx = i
            self.size = i + 1
        else:
            j = int(self.rng.integers(0, i + 1))  # Algorithm R: j uniform in [0, i]
            if j >= self.capacity:
                self.n_seen += 1
                return
            idx = j
        self.features[idx] = features
        self.regrets[idx] = regrets
        self.legal_masks[idx] = legal_mask
        self.weights[idx] = t
        self.n_seen += 1

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform random minibatch (with replacement) as torch tensors:
        (features, regret_targets, legal_mask, weights)."""
        idx = self.rng.integers(0, self.size, size=min(batch_size, self.size))
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.regrets[idx]),
            torch.from_numpy(self.legal_masks[idx]),
            torch.from_numpy(self.weights[idx]),
        )
