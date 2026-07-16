"""The evolvable unit: a linear scoring function over situation features.

For every decision, each of the 3 basic actions (FOLD, CHECK/CALL, BET/RAISE)
gets a score = weights[action] . features + bias[action]. The legal action
with the highest score is chosen. A second small linear function decides how
big to bet/raise when that action wins. Genes are simply the numbers in
these weight matrices/vectors -- that's what the GA mutates and recombines.
"""

from __future__ import annotations

import numpy as np

from features import NUM_FEATURES, Situation, extract_features

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
NUM_ACTIONS = 3
ACTION_NAMES = ["fold", "check/call", "bet/raise"]

# Bet sizing is expressed as a fraction of the pot, mapped from a sigmoid
# output onto this [min, max] pot-multiple range, then clamped to legal size.
MIN_POT_FRACTION = 0.25
MAX_POT_FRACTION = 2.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Genome:
    """action_weights: (NUM_ACTIONS, NUM_FEATURES); action_bias: (NUM_ACTIONS,)
    sizing_weights: (NUM_FEATURES,); sizing_bias: scalar
    noise_std: exploration noise added to action scores before argmax
    """

    __slots__ = ("action_weights", "action_bias", "sizing_weights", "sizing_bias", "noise_std")

    def __init__(
        self,
        action_weights: np.ndarray,
        action_bias: np.ndarray,
        sizing_weights: np.ndarray,
        sizing_bias: float,
        noise_std: float,
    ):
        self.action_weights = action_weights
        self.action_bias = action_bias
        self.sizing_weights = sizing_weights
        self.sizing_bias = sizing_bias
        self.noise_std = noise_std

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 0.5) -> "Genome":
        return cls(
            action_weights=rng.normal(0, scale, size=(NUM_ACTIONS, NUM_FEATURES)),
            action_bias=rng.normal(0, scale, size=NUM_ACTIONS),
            sizing_weights=rng.normal(0, scale, size=NUM_FEATURES),
            sizing_bias=float(rng.normal(0, scale)),
            noise_std=float(abs(rng.normal(0.15, 0.1))),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([
            self.action_weights.ravel(),
            self.action_bias.ravel(),
            self.sizing_weights.ravel(),
            [self.sizing_bias],
            [self.noise_std],
        ])

    @classmethod
    def unflatten(cls, vec: np.ndarray) -> "Genome":
        i = 0
        aw_size = NUM_ACTIONS * NUM_FEATURES
        action_weights = vec[i : i + aw_size].reshape(NUM_ACTIONS, NUM_FEATURES)
        i += aw_size
        action_bias = vec[i : i + NUM_ACTIONS]
        i += NUM_ACTIONS
        sizing_weights = vec[i : i + NUM_FEATURES]
        i += NUM_FEATURES
        sizing_bias = float(vec[i])
        i += 1
        noise_std = float(abs(vec[i]))
        return cls(action_weights, action_bias, sizing_weights, sizing_bias, noise_std)

    def save(self, path: str) -> None:
        np.save(path, self.flatten())

    @classmethod
    def load(cls, path: str) -> "Genome":
        return cls.unflatten(np.load(path))

    def copy(self) -> "Genome":
        return Genome(
            self.action_weights.copy(),
            self.action_bias.copy(),
            self.sizing_weights.copy(),
            self.sizing_bias,
            self.noise_std,
        )

    def score_actions(self, features: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        scores = self.action_weights @ features + self.action_bias
        if rng is not None and self.noise_std > 0:
            scores = scores + rng.normal(0, self.noise_std, size=NUM_ACTIONS)
        return scores

    def bet_size_fraction(self, features: np.ndarray) -> float:
        raw = float(self.sizing_weights @ features + self.sizing_bias)
        frac = _sigmoid(raw)  # 0..1
        return MIN_POT_FRACTION + frac * (MAX_POT_FRACTION - MIN_POT_FRACTION)

    def decide(
        self,
        situation: Situation,
        legal_actions: list[int],
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Returns (action, raw_bet_size_in_chips_if_betting_else_0)."""
        features = extract_features(situation)
        scores = self.score_actions(features, rng)
        best_action = max(legal_actions, key=lambda a: scores[a])
        bet_size = 0.0
        if best_action == BET_RAISE:
            frac = self.bet_size_fraction(features)
            bet_size = frac * max(situation.pot, 1.0)
        return best_action, bet_size


def save_population(genomes: list[Genome], path: str) -> None:
    """Saves a whole generation (best-first, if the caller ranked it) as one
    file so a later run can reload it wholesale as its starting population."""
    np.save(path, np.stack([g.flatten() for g in genomes]))


def load_population(path: str) -> list[Genome]:
    return [Genome.unflatten(row) for row in np.load(path)]
