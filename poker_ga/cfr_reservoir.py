"""Fixed-capacity uniform reservoir buffer (classic Algorithm R) of
(features, regrets, legal_mask, t) samples collected during cfr_tree
traversals -- Single Deep CFR's only memory (one shared buffer for the one
shared advantage network, see cfr_tree.py's module docstring). Sampling for
insertion is uniform over every sample ever seen; the linear-CFR-style
"later iterations matter more" weighting instead happens at *training*
time, via each sample's stored `t` scaling its loss term (see cfr_train.py).

Each sample also stores `iteration` -- the raw outer training iteration it
was collected at (see cfr_tree.py's `_TraversalContext.t`), *not* scaled by
that traversal's own path_weight the way `t` is. `t`/`iteration` coincide
whenever path_weight is 1 (the common case), but path_weight can shrink `t`
well below the true iteration for a sample from deep in a heavily-explored
branch -- so `t` alone can't be compared against a raw iteration number to
tell which iteration a still-held row actually came from. `iteration`
exists purely so cfr_main.py's own benchmark-pool weighting (see its module
docstring) can attribute currently-held rows back to whichever past
checkpoint was training when they were collected, without that
path_weight-induced skew.
"""

from __future__ import annotations

import numpy as np
import torch

# Sentinel `iterations` reading for a row loaded from a reservoir saved
# before this field existed -- genuinely unknown, not 0 (a real iteration
# number) or `weights` (a real but path_weight-skewed value that would
# silently misattribute the row -- see this module's own docstring).
# Callers attributing rows to a checkpoint's own iteration span should
# exclude UNKNOWN_ITERATION rows rather than guess.
UNKNOWN_ITERATION = -1.0


class ReservoirBuffer:
    def __init__(self, capacity: int, feature_dim: int, num_actions: int, rng: np.random.Generator):
        self.capacity = capacity
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.regrets = np.zeros((capacity, num_actions), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.weights = np.zeros(capacity, dtype=np.float32)
        self.iterations = np.full(capacity, UNKNOWN_ITERATION, dtype=np.float32)
        self.size = 0
        self.n_seen = 0
        self.rng = rng

    def add(
        self, features: np.ndarray, regrets: np.ndarray, legal_mask: np.ndarray, t: float,
        iteration: float | None = None,
    ) -> None:
        """`iteration` defaults to `t` itself (the common case, path_weight
        1) when not given -- e.g. every caller that doesn't care about the
        raw-iteration/path_weight distinction (most tests here included),
        not just cfr_tree.py's own real traversals with path_weight < 1."""
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
        self.iterations[idx] = t if iteration is None else iteration
        self.n_seen += 1

    def grow(self, new_capacity: int) -> None:
        """Raises `capacity` to `new_capacity` in place, preserving every
        already-collected sample (and `size`/`n_seen`, so future Algorithm R
        replacement odds stay correct) -- e.g. a reloaded reservoir whose
        saved capacity is smaller than what's now requested (see
        cfr_main.py's own --reservoir-capacity handling). A no-op if
        `new_capacity` isn't actually larger than the current one: shrinking
        would mean discarding already-collected samples, which isn't what
        this is for.

        Growing a reservoir mid-flight is algorithmically sound, not just a
        storage resize: Algorithm R only requires that at any point, the
        `min(n_seen, capacity)` filled slots be a uniform random sample of
        the `n_seen` items seen so far -- true here both before and
        immediately after, since nothing about which samples are already
        held changes. The newly added slots are simply empty capacity for
        future add() calls to fill (up to the new capacity) before
        probabilistic replacement resumes, exactly as if that capacity had
        been there from the start."""
        if new_capacity <= self.capacity:
            return
        for attr in ("features", "regrets", "legal_masks", "weights", "iterations"):
            old = getattr(self, attr)
            fill = UNKNOWN_ITERATION if attr == "iterations" else 0
            new = np.full((new_capacity, *old.shape[1:]), fill, dtype=old.dtype)
            new[: self.size] = old[: self.size]
            setattr(self, attr, new)
        self.capacity = new_capacity

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
            iterations=self.iterations[: self.size],
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
            # A reservoir saved before `iterations` existed has no such key
            # -- every one of its rows' true collection iteration is
            # genuinely unknown (see UNKNOWN_ITERATION), not recoverable
            # from `weights` alone (path_weight-skewed -- see this module's
            # own docstring), so __init__'s own UNKNOWN_ITERATION fill is
            # left as-is for them rather than guessed at.
            if "iterations" in data:
                buf.iterations[:size] = data["iterations"]
            buf.size = size
            buf.n_seen = int(data["n_seen"])
        return buf
