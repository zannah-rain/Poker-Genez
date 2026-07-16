"""The evolvable unit: a single linear scoring function over situation
features, deliberately simple enough that a human could hand-compute it.

For every decision, one score = weights . features + bias is computed, and
thresholded into an action:
    score <= 0        -> fold (or check, if there's nothing to call)
    0 < score <= 1     -> check/call
    score > 1          -> bet/raise, sized at (score - 1) x pot
                          (e.g. score 2.0 = a pot-sized raise)

Genes are simply the numbers in `weights` plus `bias` -- that's what the GA
mutates and recombines.
"""

from __future__ import annotations

import numpy as np

from features import NUM_FEATURES, Situation, extract_features

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]


class Genome:
    """weights: (NUM_FEATURES,); bias: scalar
    noise_std: exploration noise added to the score before thresholding
    """

    __slots__ = ("weights", "bias", "noise_std")

    def __init__(self, weights: np.ndarray, bias: float, noise_std: float):
        self.weights = weights
        self.bias = bias
        self.noise_std = noise_std

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 0.5) -> "Genome":
        return cls(
            weights=rng.normal(0, scale, size=NUM_FEATURES),
            bias=float(rng.normal(0, scale)),
            noise_std=float(abs(rng.normal(0.15, 0.1))),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.weights, [self.bias], [self.noise_std]])

    @classmethod
    def unflatten(cls, vec: np.ndarray) -> "Genome":
        weights = vec[:NUM_FEATURES]
        bias = float(vec[NUM_FEATURES])
        noise_std = float(abs(vec[NUM_FEATURES + 1]))
        return cls(weights, bias, noise_std)

    def save(self, path: str) -> None:
        np.save(path, self.flatten())

    @classmethod
    def load(cls, path: str) -> "Genome":
        return cls.unflatten(np.load(path))

    def copy(self) -> "Genome":
        return Genome(self.weights.copy(), self.bias, self.noise_std)

    def score(self, features: np.ndarray, rng: np.random.Generator | None = None) -> float:
        value = float(self.weights @ features + self.bias)
        if rng is not None and self.noise_std > 0:
            value += float(rng.normal(0, self.noise_std))
        return value

    def decide(
        self,
        situation: Situation,
        legal_actions: list[int],
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Returns (action, raw_bet_size_in_chips_if_betting_else_0)."""
        features = extract_features(situation)
        value = self.score(features, rng)

        if value <= 0:
            action = FOLD if FOLD in legal_actions else CHECK_CALL
        elif value <= 1:
            action = CHECK_CALL
        else:
            action = BET_RAISE if BET_RAISE in legal_actions else CHECK_CALL

        bet_size = 0.0
        if action == BET_RAISE:
            bet_size = (value - 1.0) * max(situation.pot, 1.0)
        return action, bet_size


def save_population(genomes: list[Genome], path: str) -> None:
    """Saves a whole generation (best-first, if the caller ranked it) as one
    file so a later run can reload it wholesale as its starting population."""
    np.save(path, np.stack([g.flatten() for g in genomes]))


def load_population(path: str) -> list[Genome]:
    return [Genome.unflatten(row) for row in np.load(path)]
