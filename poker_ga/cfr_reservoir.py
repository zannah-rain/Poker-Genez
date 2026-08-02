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

    def save(self, path: str) -> None:
        """Writes `<path>.npz` -- only the filled `:size` prefix of each
        array, not the full (possibly much larger, zero-padded) capacity,
        plus capacity/n_seen/size so load() can reconstruct an identical
        buffer (including future Algorithm R replacement odds, which depend
        on n_seen, not just size)."""
        np.savez(
            f"{path}.npz",
            capacity=self.capacity,
            feature_dim=self.features.shape[1],
            num_actions=self.regrets.shape[1],
            size=self.size,
            n_seen=self.n_seen,
            features=self.features[: self.size],
            regrets=self.regrets[: self.size],
            legal_masks=self.legal_masks[: self.size],
            weights=self.weights[: self.size],
        )

    @classmethod
    def load(cls, path: str, rng: np.random.Generator) -> "ReservoirBuffer":
        with np.load(f"{path}.npz") as data:
            buf = cls(
                capacity=int(data["capacity"]), feature_dim=int(data["feature_dim"]),
                num_actions=int(data["num_actions"]), rng=rng,
            )
            size = int(data["size"])
            buf.features[:size] = data["features"]
            buf.regrets[:size] = data["regrets"]
            buf.legal_masks[:size] = data["legal_masks"]
            buf.weights[:size] = data["weights"]
            buf.size = size
            buf.n_seen = int(data["n_seen"])
        return buf
