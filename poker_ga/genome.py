"""The evolvable unit: two near-orthogonal axes per feature instead of one.

Every feature gets two weights, one feeding V and one feeding L:

    V -- showdown value. Roughly "my equity against the range that
         continues." A linear combination of the features, offset and
         clipped to land on 0-100 (read it as a percentage).
    L -- leverage. Roughly "how much of villain's range folds to me" --
         fold equity shaped by blockers, initiative, board texture,
         position, and SPR. Also a linear, clipped 0-100 combination.

These two are close to independent (a nut flush blocker is almost pure L
with near-zero V; a set on a dry board is almost pure V with low L), which
is what lets 2 numbers per feature carry more than 1 did.

The decision rule is deliberately non-convex -- a plain aV + bL would just
be a 1D score again -- so it's a max of two linear terms:

    A = max( V - theta_value,  L - theta_bluff - kappa * V )
    A > 0              -> bet/raise, sized off A as a fraction of pot
    elif V > theta_call -> call/check
    else               -> fold/check

Feature weights (weights_v/weights_l) are quantized to WEIGHT_ALPHABET, a
small fixed set like {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}, rather than being
continuous -- so a genome reduces to "which of ~130 features matter, how
much, in which direction," a table small enough a human could plausibly
memorize and apply at the table. The GA's fitness function separately
penalizes nonzero weights (see main.py's --sparsity-penalty), pushing
evolution toward genomes where most weights land on exactly 0.
"""

from __future__ import annotations

import numpy as np

from features import NUM_FEATURES, Situation, extract_features

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]

# Feature weights (weights_v/weights_l) are constrained to this small alphabet
# rather than being continuous -- a genome is then just a lookup table of
# "which features matter, how much, in which direction" that a human could
# plausibly memorize, instead of an arbitrary-precision float vector.
WEIGHT_ALPHABET = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])


def quantize(values: np.ndarray) -> np.ndarray:
    """Snaps each value to its nearest member of WEIGHT_ALPHABET."""
    distances = np.abs(values[..., None] - WEIGHT_ALPHABET)
    return WEIGHT_ALPHABET[np.argmin(distances, axis=-1)]


class Genome:
    """weights_v/weights_l: (NUM_FEATURES,) each; bias_v/bias_l: scalars.
    theta_value/theta_bluff/theta_call: raw thresholds
    kappa: non-negative damping of the bluff term as V rises.
    noise_std: exploration noise added to V and L (independently) before
    thresholding.
    """

    __slots__ = (
        "weights_v", "bias_v", "weights_l", "bias_l",
        "theta_value", "theta_bluff", "theta_call", "kappa", "noise_std",
    )

    def __init__(
        self,
        weights_v: np.ndarray, bias_v: float,
        weights_l: np.ndarray, bias_l: float,
        theta_value: float, theta_bluff: float, theta_call: float,
        kappa: float, noise_std: float,
    ):
        self.weights_v = weights_v
        self.bias_v = bias_v
        self.weights_l = weights_l
        self.bias_l = bias_l
        self.theta_value = theta_value
        self.theta_bluff = theta_bluff
        self.theta_call = theta_call
        self.kappa = kappa
        self.noise_std = noise_std

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 0.5) -> "Genome":
        return cls(
            weights_v=quantize(rng.normal(0, scale, size=NUM_FEATURES)),
            bias_v=float(rng.normal(0, scale)),
            weights_l=quantize(rng.normal(0, scale, size=NUM_FEATURES)),
            bias_l=float(rng.normal(0, scale)),
            theta_value=float(rng.normal(0.0, 1.0)),
            theta_bluff=float(rng.normal(0.0, 1.0)),
            theta_call=float(rng.normal(-0.5, 1.0)),
            kappa=float(abs(rng.normal(0.5, 0.3))),
            noise_std=float(abs(rng.normal(5.0, 3.0))),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([
            self.weights_v, [self.bias_v],
            self.weights_l, [self.bias_l],
            [self.theta_value, self.theta_bluff, self.theta_call, self.kappa, self.noise_std],
        ])

    @classmethod
    def unflatten(cls, vec: np.ndarray) -> "Genome":
        """Reconstructs a Genome from a flat gene vector, re-quantizing the
        weight slices back onto WEIGHT_ALPHABET. This is the single point
        every genome passes through after crossover/mutation (see ga.py,
        which treats genomes as opaque flat vectors), so it's what keeps
        weights on-alphabet without ga.py needing to know which genes are
        "feature weights" versus continuous scalars like the thresholds."""
        i = 0
        weights_v = quantize(vec[i : i + NUM_FEATURES])
        i += NUM_FEATURES
        bias_v = float(vec[i])
        i += 1
        weights_l = quantize(vec[i : i + NUM_FEATURES])
        i += NUM_FEATURES
        bias_l = float(vec[i])
        i += 1
        theta_value = float(vec[i])
        i += 1
        theta_bluff = float(vec[i])
        i += 1
        theta_call = float(vec[i])
        i += 1
        kappa = float(abs(vec[i]))
        i += 1
        noise_std = float(abs(vec[i]))
        return cls(weights_v, bias_v, weights_l, bias_l, theta_value, theta_bluff, theta_call, kappa, noise_std)

    def save(self, path: str) -> None:
        np.save(path, self.flatten())

    @classmethod
    def load(cls, path: str) -> "Genome":
        return cls.unflatten(np.load(path))

    def copy(self) -> "Genome":
        return Genome(
            self.weights_v.copy(), self.bias_v,
            self.weights_l.copy(), self.bias_l,
            self.theta_value, self.theta_bluff, self.theta_call, self.kappa, self.noise_std,
        )

    def nonzero_weight_count(self) -> int:
        """Number of nonzero feature weights across both axes -- a proxy for
        how complex/hard-to-memorize this genome's strategy is."""
        return int(np.count_nonzero(self.weights_v)) + int(np.count_nonzero(self.weights_l))

    def compute_v_l(self, features: np.ndarray) -> tuple[float, float]:
        raw_v = float(self.weights_v @ features + self.bias_v)
        raw_l = float(self.weights_l @ features + self.bias_l)
        return raw_v, raw_l

    def decide(
        self,
        situation: Situation,
        legal_actions: list[int],
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Returns (action, raw_bet_size_in_chips_if_betting_else_0)."""
        features = extract_features(situation)
        v, l = self.compute_v_l(features)
        if rng is not None and self.noise_std > 0:
            v += float(rng.normal(0, self.noise_std))
            l += float(rng.normal(0, self.noise_std))

        theta_value = self.theta_value
        theta_bluff = self.theta_bluff
        theta_call = self.theta_call

        a = max(v - theta_value, l - theta_bluff - self.kappa * v)

        if a > 0:
            action = BET_RAISE if BET_RAISE in legal_actions else CHECK_CALL
        elif v > theta_call:
            action = CHECK_CALL
        else:
            action = FOLD if FOLD in legal_actions else CHECK_CALL

        bet_size = 0.0
        if action == BET_RAISE:
            bet_size = a * max(situation.pot, 1.0)
        return action, bet_size


def save_population(genomes: list[Genome], path: str) -> None:
    """Saves a whole generation (best-first, if the caller ranked it) as one
    file so a later run can reload it wholesale as its starting population."""
    np.save(path, np.stack([g.flatten() for g in genomes]))


def load_population(path: str) -> list[Genome]:
    return [Genome.unflatten(row) for row in np.load(path)]
