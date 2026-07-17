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

    A = max( V - THETA_VALUE,  L - THETA_BLUFF - KAPPA * V )
    A > 0              -> bet/raise, sized off A as a fraction of pot
    elif V > THETA_CALL -> call/check
    else               -> fold/check

THETA_VALUE/THETA_BLUFF/THETA_CALL/BIAS_V/BIAS_L/KAPPA are all fixed
constants, not evolvable genes. They started out as per-genome genes, but
that let each one random-walk without any bound -- and since bet/raise
fires on *either* axis clearing its bar (an OR) while folding needs *both*
to fail (an AND), selection can cheapen folding to near-zero by drifting
*any* number that sits on the "makes it easier to clear a bar" side of
either inequality, not just the thresholds themselves. This was measured
directly: fixing just the thresholds didn't stop the collapse (fold rate
still fell to ~7%, hands survived per session still collapsed to ~1) --
evolution simply moved the same exploit onto BIAS_L (drifted ~48 -> ~65-75
across generations, which alone pushes L past THETA_BLUFF before any
feature is even considered) and KAPPA (drifted ~0.52 -> ~0.30, weakening
the term that's supposed to make bluffing harder as V rises). There's no
real reason any of these need to be learned per genome -- they're all just
"how good does my hand/leverage need to be, on the same 0-100 scale as
V/L, to raise / bluff / continue," and "how much bluffing gets suppressed
by having a strong hand anyway" -- a human would pick one sensible set of
numbers and stick with it, same as here. Only the feature weights (which
feed V/L based on the position's actual features) and the exploration
noise are left to evolve.

Weights are scaled to live on the same 0-100 range as V/L (big enough that
a handful of active features move V/L meaningfully), and BIAS_V/BIAS_L sit
at 50 -- the "no information" percentile -- so every number in a genome
reads the same way a human would think about it: no separate unit
conversion needed at the table. The only nonlinear-looking step anywhere
is the min/max clamp on V/L, which is just "cap it at 0 or 100," not a
curve.

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

# Fixed decision thresholds, on the same 0-100 scale as V/L (see module
# docstring for why these are constants rather than evolvable genes).
THETA_VALUE = 70.0  # V needed to raise for value: roughly "top 30%" showdown equity
THETA_BLUFF = 70.0  # L needed to raise as a bluff: roughly "top 30%" leverage
THETA_CALL = 40.0  # V needed to continue at all rather than fold: better than roughly average

# Fixed baseline offsets for V/L, and fixed bluff-suppression strength (see
# module docstring for why these are constants rather than evolvable genes).
BIAS_V = 50.0  # the "no information" percentile: a featureless hand reads as average
BIAS_L = 50.0
KAPPA = 0.5  # how much having a strong hand (V) suppresses the bluff term

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

    Each selected gene gets one of two moves, picked at random:
      - nudge one step up/down the (ordered) alphabet -- local search
      - jump to a uniformly random alphabet value -- occasional big jumps
        so search isn't limited to exploring one neighbor at a time
    """
    mask = rng.random(weights.shape) < rate
    if not mask.any():
        return weights

    n = len(WEIGHT_ALPHABET)
    current_index = np.argmin(np.abs(weights[..., None] - WEIGHT_ALPHABET), axis=-1)

    step = np.where(rng.random(weights.shape) < 0.5, 1, -1)
    nudged_index = np.clip(current_index + step, 0, n - 1)
    random_index = rng.integers(0, n, size=weights.shape)

    move = rng.random(weights.shape)
    new_index = np.where(move < 0.8, nudged_index, random_index)

    return np.where(mask, WEIGHT_ALPHABET[new_index], weights)


def crossover_weights(a_weights: np.ndarray, b_weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform (discrete) crossover for quantized weights: each gene is
    inherited whole from one parent or the other (50/50, independently per
    gene), instead of blended like a continuous gene. Blending two exact
    alphabet values and re-quantizing invents intermediate values neither
    parent had, and systematically dilutes sparsity whenever a sparse
    parent (weight=0) meets a dense one -- only the narrow slice of the
    blend range nearest 0 rounds back to 0, so a 0 gene mated with a large
    nonzero gene mostly produces a nonzero child. Picking a value each
    parent actually had preserves whatever already survived selection --
    including zeros -- the way discrete/categorical genes should cross."""
    from_a = rng.random(a_weights.shape) < 0.5
    return np.where(from_a, a_weights, b_weights)


class Genome:
    """weights_v/weights_l: (NUM_FEATURES,) each.
    noise_std: exploration noise added to V and L (independently) before
    thresholding. (THETA_VALUE/THETA_BLUFF/THETA_CALL/BIAS_V/BIAS_L/KAPPA
    are all fixed constants, not part of the genome -- see module
    docstring.)
    """

    __slots__ = ("weights_v", "weights_l", "noise_std")

    def __init__(self, weights_v: np.ndarray, weights_l: np.ndarray, noise_std: float):
        self.weights_v = weights_v
        self.weights_l = weights_l
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
            weights_l=weights_l,
            noise_std=float(abs(rng.normal(5.0, 3.0))),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.weights_v, self.weights_l, [self.noise_std]])

    @classmethod
    def unflatten(cls, vec: np.ndarray) -> "Genome":
        """Reconstructs a Genome from a flat gene vector, re-quantizing the
        weight slices back onto WEIGHT_ALPHABET. This is the single point
        every genome passes through after crossover/mutation (see ga.py,
        which treats genomes as opaque flat vectors), so it's what keeps
        weights on-alphabet without ga.py needing to know which genes are
        "feature weights" versus continuous scalars."""
        i = 0
        weights_v = quantize(vec[i : i + NUM_FEATURES])
        i += NUM_FEATURES
        weights_l = quantize(vec[i : i + NUM_FEATURES])
        i += NUM_FEATURES
        noise_std = float(abs(vec[i]))
        return cls(weights_v, weights_l, noise_std)

    def save(self, path: str) -> None:
        np.save(path, self.flatten())

    @classmethod
    def load(cls, path: str) -> "Genome":
        return cls.unflatten(np.load(path))

    def copy(self) -> "Genome":
        return Genome(self.weights_v.copy(), self.weights_l.copy(), self.noise_std)

    def mutate(self, rng: np.random.Generator, rate: float, continuous_scale: float) -> "Genome":
        """Returns a mutated copy. Feature weights get alphabet-jump
        mutation (see mutate_weights) since additive noise re-quantized
        almost never actually moves them; noise_std keeps simple
        additive-gaussian mutation. Each gene is independently selected for
        mutation with probability `rate`, whichever kind it is."""
        def mutate_scalar(value: float) -> float:
            if rng.random() < rate:
                return value + float(rng.normal(0, continuous_scale))
            return value

        return Genome(
            mutate_weights(self.weights_v, rate, rng),
            mutate_weights(self.weights_l, rate, rng),
            abs(mutate_scalar(self.noise_std)),
        )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        """Returns a child combining self and other. Feature weights use
        uniform (discrete) crossover (see crossover_weights) since they're
        quantized -- blending two alphabet values and re-quantizing invents
        values neither parent had and dilutes sparsity. noise_std keeps
        blend crossover (a random weighted average), the right operator for
        a genuinely real-valued gene."""
        def blend_scalar(x: float, y: float) -> float:
            alpha = rng.uniform(0.0, 1.0)
            return alpha * x + (1 - alpha) * y

        return Genome(
            crossover_weights(self.weights_v, other.weights_v, rng),
            crossover_weights(self.weights_l, other.weights_l, rng),
            abs(blend_scalar(self.noise_std, other.noise_std)),
        )

    def nonzero_weight_count(self) -> int:
        """Number of nonzero feature weights across both axes -- a proxy for
        how complex/hard-to-memorize this genome's strategy is."""
        return int(np.count_nonzero(self.weights_v)) + int(np.count_nonzero(self.weights_l))

    def compute_v_l(self, features: np.ndarray) -> tuple[float, float]:
        raw_v = float(self.weights_v @ features + BIAS_V)
        raw_l = float(self.weights_l @ features + BIAS_L)
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

        a = max(v - THETA_VALUE, l - THETA_BLUFF - KAPPA * v)

        if a > 0:
            action = BET_RAISE if BET_RAISE in legal_actions else CHECK_CALL
        elif v > THETA_CALL:
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
