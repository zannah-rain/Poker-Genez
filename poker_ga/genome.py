"""The evolvable unit: two near-orthogonal axes per feature instead of one.

Every feature gets two weights, one feeding V and one feeding L:

    V -- showdown value. Roughly "my equity against the range that
         continues." A linear combination of the features, offset by a
         bias and clamped (plain min/max, not a sigmoid) to land on 0-100
         -- read it as a percentile ("V=90" ~ a top-10% hand).
    L -- leverage. Roughly "how much of villain's range folds to me" --
         fold equity shaped by blockers, initiative, board texture,
         position, and SPR. Also a linear, clamped 0-100 combination.

These two are close to independent (a nut flush blocker is almost pure L
with near-zero V; a set on a dry board is almost pure V with low L), which
is what lets 2 numbers per feature carry more than 1 did.

The decision rule is deliberately non-convex -- a plain aV + bL would just
be a 1D score again -- so it's a max of two linear terms:

    A = max( V - theta_value,  L - theta_bluff - kappa * V )
    A > 0              -> bet/raise, sized off A as a fraction of pot
    elif V > theta_call -> call/check
    else               -> fold/check

Weights, biases, and thresholds are all scaled to live on that same 0-100
range (weights big enough that a handful of active features move V/L
meaningfully; biases centered near 50, the "no information" percentile;
thresholds initialized in 0-100 too), so every number in a genome reads
the same way a human would think about it -- no separate unit conversion
needed at the table. The only nonlinear-looking step anywhere is the
min/max clamp on V/L, which is just "cap it at 0 or 100," not a curve.

Feature weights (weights_v/weights_l) are quantized to WEIGHT_ALPHABET
rather than being continuous -- so a genome reduces to "which of ~130
features matter, how much, in which direction," a table small enough a
human could plausibly memorize and apply at the table. The GA's fitness
function separately penalizes nonzero weights (see main.py's
--sparsity-penalty), pushing evolution toward genomes where most weights
land on exactly 0.
"""

from __future__ import annotations

import numpy as np

from features import NUM_FEATURES, Situation, extract_features

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]

V_SCALE = 100.0  # V and L both live on this range, clamped -- read them as percentiles.

# Feature weights (weights_v/weights_l) are constrained to this small alphabet
# rather than being continuous -- a genome is then just a lookup table of
# "which features matter, how much, in which direction" that a human could
# plausibly memorize, instead of an arbitrary-precision float vector. Values
# are sized so that a handful of active features can meaningfully move a
# 0-100 score (see V_SCALE) -- e.g. 3 active weight-20 features already
# swings the score by more than half its range.
WEIGHT_ALPHABET_SCALE = 10.0
WEIGHT_ALPHABET = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]) * WEIGHT_ALPHABET_SCALE


def quantize(values: np.ndarray) -> np.ndarray:
    """Snaps each value to its nearest member of WEIGHT_ALPHABET."""
    distances = np.abs(values[..., None] - WEIGHT_ALPHABET)
    return WEIGHT_ALPHABET[np.argmin(distances, axis=-1)]


def mutate_weights(weights: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Mutates a quantized weight vector by jumping between WEIGHT_ALPHABET
    entries directly, instead of adding continuous noise and re-quantizing.
    A perturbation small enough to be sane for a continuous gene (e.g.
    GAConfig.mutation_scale ~ a few units) is almost always *far* smaller
    than the ~5-10 unit gap between adjacent alphabet values, so
    continuous-style mutation silently does nothing to a quantized gene on
    nearly every mutation event -- weights would never actually change.

    Each selected gene gets one of three moves, picked at random:
      - nudge one step up/down the (ordered) alphabet -- local search
      - reset straight to 0 -- a direct one-mutation path to sparsity,
        complementing the fitness function's --sparsity-penalty (reaching 0
        by nudging alone could take up to 4 successive mutations from the
        alphabet's extremes)
      - jump to a uniformly random alphabet value -- occasional big jumps
        so search isn't limited to exploring one neighbor at a time
    """
    mask = rng.random(weights.shape) < rate
    if not mask.any():
        return weights

    n = len(WEIGHT_ALPHABET)
    zero_index = int(np.argmin(np.abs(WEIGHT_ALPHABET)))
    current_index = np.argmin(np.abs(weights[..., None] - WEIGHT_ALPHABET), axis=-1)

    step = np.where(rng.random(weights.shape) < 0.5, 1, -1)
    nudged_index = np.clip(current_index + step, 0, n - 1)
    random_index = rng.integers(0, n, size=weights.shape)

    move = rng.random(weights.shape)
    new_index = np.where(move < 0.5, nudged_index, np.where(move < 0.75, zero_index, random_index))

    return np.where(mask, WEIGHT_ALPHABET[new_index], weights)


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
        # `scale` is on the same relative footing as before quantization was
        # introduced (WEIGHT_ALPHABET is just the old {-3..3} pattern scaled
        # up), so the fraction of weights that land on each alphabet value
        # is unchanged by that rescaling -- only what each value *means* is
        # bigger now.
        weights_v = quantize(rng.normal(0, scale * WEIGHT_ALPHABET_SCALE, size=NUM_FEATURES))
        weights_l = quantize(rng.normal(0, scale * WEIGHT_ALPHABET_SCALE, size=NUM_FEATURES))

        return cls(
            weights_v=weights_v,
            bias_v=float(rng.normal(50.0, 20.0)),  # centered on 50: the "no information" percentile
            weights_l=weights_l,
            bias_l=float(rng.normal(50.0, 20.0)),
            theta_value=float(rng.normal(65.0, 15.0)),
            theta_bluff=float(rng.normal(65.0, 15.0)),
            theta_call=float(rng.normal(40.0, 15.0)),
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

    def mutate(self, rng: np.random.Generator, rate: float, continuous_scale: float) -> "Genome":
        """Returns a mutated copy. Feature weights get alphabet-jump
        mutation (see mutate_weights) since additive noise re-quantized
        almost never actually moves them; the continuous scalars (biases,
        thresholds, kappa, noise) keep simple additive-gaussian mutation.
        Each gene is independently selected for mutation with probability
        `rate`, whichever kind it is."""
        def mutate_scalar(value: float) -> float:
            if rng.random() < rate:
                return value + float(rng.normal(0, continuous_scale))
            return value

        return Genome(
            mutate_weights(self.weights_v, rate, rng), mutate_scalar(self.bias_v),
            mutate_weights(self.weights_l, rate, rng), mutate_scalar(self.bias_l),
            mutate_scalar(self.theta_value), mutate_scalar(self.theta_bluff), mutate_scalar(self.theta_call),
            abs(mutate_scalar(self.kappa)), abs(mutate_scalar(self.noise_std)),
        )

    def nonzero_weight_count(self) -> int:
        """Number of nonzero feature weights across both axes -- a proxy for
        how complex/hard-to-memorize this genome's strategy is."""
        return int(np.count_nonzero(self.weights_v)) + int(np.count_nonzero(self.weights_l))

    def compute_v_l(self, features: np.ndarray) -> tuple[float, float]:
        raw_v = float(self.weights_v @ features + self.bias_v)
        raw_l = float(self.weights_l @ features + self.bias_l)
        # Clamp (plain min/max, not a curve) rather than let a linear sum
        # wander arbitrarily -- V/L are meant to read as percentiles, which
        # can't go below 0 or above 100.
        v = min(max(raw_v, 0.0), V_SCALE)
        l = min(max(raw_l, 0.0), V_SCALE)
        return v, l

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
            bet_size = (a / V_SCALE) * max(situation.pot, 1.0)
        return action, bet_size


def save_population(genomes: list[Genome], path: str) -> None:
    """Saves a whole generation (best-first, if the caller ranked it) as one
    file so a later run can reload it wholesale as its starting population."""
    np.save(path, np.stack([g.flatten() for g in genomes]))


def load_population(path: str) -> list[Genome]:
    return [Genome.unflatten(row) for row in np.load(path)]
