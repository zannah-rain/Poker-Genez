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

theta_value/theta_bluff/theta_call/bias_v/bias_l/kappa are evolvable
per-genome genes. They were fixed as constants for a while, because letting
them free-drift under selection turned out to be exploitable: bet/raise
fires on *either* axis clearing its bar (an OR) while folding needs *both*
to fail (an AND), so selection could cheapen folding to near-zero just by
drifting any of these numbers toward the "makes it easier to clear a bar"
side of either inequality -- measured directly as a fold-rate collapse
(~34% to ~7%) and average hands survived per session collapsing to ~1
within a handful of generations, tracked all the way down to specific genes
walking in that direction (bias_l drifting up, kappa drifting toward 0).
What actually turned out to be driving that collapse, though, was a game
mechanic: sessions used to end outright once a couple of players busted,
letting a lucky win get "locked in" before it could be punished -- fixed by
refilling busted seats with fresh players instead of ending the session
(see simulate.py). With that root cause fixed, and with several other
safeguards added since (a pot-scaled minimum raise floor, a sparsity
penalty, island-isolated sub-populations, and benchmark-checkpoint
reverting/early-stopping that catches and undoes a population that's
measurably gotten worse -- see main.py), these six scalars are safe to let
evolve again: the mechanism that made an unbounded drift pay off is gone,
and there's now a safety net that reverts training if a similar drift ever
did start paying off. `random()` still initializes them centered on the
same reasonable values previously used as fixed constants (theta_value=70,
theta_bluff=70, theta_call=40, bias_v=bias_l=50, kappa=0.5), so evolution
starts from a sane baseline and has to earn any drift away from it.

Weights are scaled to live on the same 0-100 range as V/L (big enough that
a handful of active features move V/L meaningfully), and biases are
centered at 50 -- the "no information" percentile -- so every number in a
genome reads the same way a human would think about it: no separate unit
conversion needed at the table. The only nonlinear-looking step anywhere is
the min/max clamp on V/L, which is just "cap it at 0 or 100," not a curve.

Feature weights (weights_v/weights_l) are quantized to WEIGHT_ALPHABET
rather than being continuous -- so a genome reduces to "which of ~130
features matter, how much, in which direction," a table small enough a
human could plausibly memorize and apply at the table. The GA's fitness
function separately penalizes nonzero weights (see main.py's
--sparsity-penalty), pushing evolution toward genomes where most weights
land on exactly 0.

On top of the linear V/L system, a genome can also memorize exact charts
for specific, well-defined spots (see gto.py) -- e.g. "UTG open, 100BB":
situations narrow enough that a human plays them from a memorized range
chart rather than by feel. `gto_flags` is one evolvable boolean gene per
catalog entry in gto.py's GTO_SPOTS ("does this genome trust this spot's
chart or not"); when a situation matches an active spot, `decide()` looks
the hand up in that spot's chart and plays exactly what it says, bypassing
V/L entirely for that decision. This is a structurally different kind of
gene from everything else here -- a hard override, not another linear
term -- because a chart lookup is exactly as memorizable for a human as the
chart itself, whereas trying to fold "always raise AA in this one exact
spot" into the linear weights would require it to also make sense as a
general trend across every other spot, which it doesn't.
"""

from __future__ import annotations

import numpy as np

from features import NUM_FEATURES, Situation, extract_features
from gto import GTO_SPOTS, NUM_GTO_SPOTS, resolve_spot_action

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]

# Fraction of GTO_SPOTS a freshly random genome starts out trusting -- low,
# so the GA has to actively select for a spot's chart being useful (mirrors
# the sparsity-penalty philosophy for weights: earn complexity, don't start
# with it) while still giving mutation/crossover some initial material.
GTO_INIT_PROB = 0.1

V_SCALE = 100.0  # V and L both live on this range, clamped -- read them as percentiles.

# Init centers for the evolvable scalars below, on the same 0-100 scale as
# V/L -- these are where `random()` centers its initial spread, not fixed
# values (see module docstring for why they're evolvable, and why these
# particular numbers are still a sane starting point).
THETA_VALUE_INIT = 70.0  # V needed to raise for value: roughly "top 30%" showdown equity
THETA_BLUFF_INIT = 70.0  # L needed to raise as a bluff: roughly "top 30%" leverage
THETA_CALL_INIT = 40.0  # V needed to continue at all rather than fold: better than roughly average
BIAS_INIT = 50.0  # the "no information" percentile: a featureless hand reads as average
KAPPA_INIT = 0.5  # how much having a strong hand (V) suppresses the bluff term

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


def mutate_bool_flags(flags: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation for boolean genes (0.0/1.0 floats): each selected
    gene flips to its opposite, rather than being nudged like a continuous
    value or alphabet-jumped like a quantized weight -- there's nothing
    "in between" on or off."""
    mask = rng.random(flags.shape) < rate
    return np.where(mask, 1.0 - flags, flags)


class Genome:
    """weights_v/weights_l: (NUM_FEATURES,) each.
    bias_v/bias_l: baseline V/L before any feature is considered.
    theta_value/theta_bluff/theta_call: decision thresholds (see module
    docstring for the decision rule).
    kappa: non-negative damping of the bluff term as V rises.
    noise_std: exploration noise added to V and L (independently) before
    thresholding.
    """

    __slots__ = (
        "weights_v", "weights_l", "bias_v", "bias_l",
        "theta_value", "theta_bluff", "theta_call", "kappa", "noise_std",
        "gto_flags",
    )

    def __init__(
        self,
        weights_v: np.ndarray, weights_l: np.ndarray,
        bias_v: float, bias_l: float,
        theta_value: float, theta_bluff: float, theta_call: float,
        kappa: float, noise_std: float,
        gto_flags: np.ndarray,
    ):
        self.weights_v = weights_v
        self.weights_l = weights_l
        self.bias_v = bias_v
        self.bias_l = bias_l
        self.theta_value = theta_value
        self.theta_bluff = theta_bluff
        self.theta_call = theta_call
        self.kappa = kappa
        self.noise_std = noise_std
        self.gto_flags = gto_flags

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
            bias_v=float(rng.normal(BIAS_INIT, 20.0)),
            bias_l=float(rng.normal(BIAS_INIT, 20.0)),
            theta_value=float(rng.normal(THETA_VALUE_INIT, 15.0)),
            theta_bluff=float(rng.normal(THETA_BLUFF_INIT, 15.0)),
            theta_call=float(rng.normal(THETA_CALL_INIT, 15.0)),
            kappa=float(abs(rng.normal(KAPPA_INIT, 0.3))),
            noise_std=float(abs(rng.normal(5.0, 3.0))),
            gto_flags=(rng.random(NUM_GTO_SPOTS) < GTO_INIT_PROB).astype(np.float64),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([
            self.weights_v, self.weights_l,
            [self.bias_v, self.bias_l, self.theta_value, self.theta_bluff, self.theta_call,
             self.kappa, self.noise_std],
            self.gto_flags,
        ])

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
        bias_v = float(vec[i]); i += 1
        bias_l = float(vec[i]); i += 1
        theta_value = float(vec[i]); i += 1
        theta_bluff = float(vec[i]); i += 1
        theta_call = float(vec[i]); i += 1
        kappa = float(abs(vec[i])); i += 1
        noise_std = float(abs(vec[i])); i += 1
        gto_flags = (vec[i : i + NUM_GTO_SPOTS] > 0.5).astype(np.float64)
        return cls(
            weights_v, weights_l, bias_v, bias_l, theta_value, theta_bluff, theta_call,
            kappa, noise_std, gto_flags,
        )

    def save(self, path: str) -> None:
        np.save(path, self.flatten())

    @classmethod
    def load(cls, path: str) -> "Genome":
        return cls.unflatten(np.load(path))

    def copy(self) -> "Genome":
        return Genome(
            self.weights_v.copy(), self.weights_l.copy(),
            self.bias_v, self.bias_l,
            self.theta_value, self.theta_bluff, self.theta_call,
            self.kappa, self.noise_std,
            self.gto_flags.copy(),
        )

    def mutate(self, rng: np.random.Generator, rate: float, continuous_scale: float) -> "Genome":
        """Returns a mutated copy. Feature weights get alphabet-jump
        mutation (see mutate_weights) since additive noise re-quantized
        almost never actually moves them; gto_flags get bit-flip mutation
        (see mutate_bool_flags); the continuous scalars (biases, thresholds,
        kappa, noise) keep simple additive-gaussian mutation. Each gene is
        independently selected for mutation with probability `rate`,
        whichever kind it is."""
        def mutate_scalar(value: float) -> float:
            if rng.random() < rate:
                return value + float(rng.normal(0, continuous_scale))
            return value

        return Genome(
            mutate_weights(self.weights_v, rate, rng),
            mutate_weights(self.weights_l, rate, rng),
            mutate_scalar(self.bias_v), mutate_scalar(self.bias_l),
            mutate_scalar(self.theta_value), mutate_scalar(self.theta_bluff), mutate_scalar(self.theta_call),
            abs(mutate_scalar(self.kappa)), abs(mutate_scalar(self.noise_std)),
            mutate_bool_flags(self.gto_flags, rate, rng),
        )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        """Returns a child combining self and other. Feature weights and
        gto_flags use uniform (discrete) crossover (see crossover_weights)
        since they're both categorical/boolean genes, not continuous --
        blending two alphabet values (or two booleans) and re-quantizing
        invents values neither parent had and dilutes sparsity. The
        continuous scalars (biases, thresholds, kappa, noise) keep blend
        crossover (a random weighted average per gene), the right operator
        for genuinely real-valued genes."""
        def blend_scalar(x: float, y: float) -> float:
            alpha = rng.uniform(0.0, 1.0)
            return alpha * x + (1 - alpha) * y

        return Genome(
            crossover_weights(self.weights_v, other.weights_v, rng),
            crossover_weights(self.weights_l, other.weights_l, rng),
            blend_scalar(self.bias_v, other.bias_v), blend_scalar(self.bias_l, other.bias_l),
            blend_scalar(self.theta_value, other.theta_value),
            blend_scalar(self.theta_bluff, other.theta_bluff),
            blend_scalar(self.theta_call, other.theta_call),
            abs(blend_scalar(self.kappa, other.kappa)),
            abs(blend_scalar(self.noise_std, other.noise_std)),
            crossover_weights(self.gto_flags, other.gto_flags, rng),
        )

    def nonzero_weight_count(self) -> int:
        """Number of nonzero feature weights across both axes -- a proxy for
        how complex/hard-to-memorize this genome's strategy is. Doesn't
        include gto_flags -- a different kind of complexity (how many
        memorized charts, not how many linear weights)."""
        return int(np.count_nonzero(self.weights_v)) + int(np.count_nonzero(self.weights_l))

    def active_gto_spots(self) -> list:
        """The GTO_SPOTS entries this genome currently trusts (gto_flags is
        on), in catalog order -- the same order decide() checks them in."""
        return [spot for spot, flag in zip(GTO_SPOTS, self.gto_flags) if flag > 0.5]

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
        gto_result = self._decide_from_gto_charts(situation, legal_actions)
        if gto_result is not None:
            return gto_result

        features = extract_features(situation)
        v, l = self.compute_v_l(features)
        if rng is not None and self.noise_std > 0:
            v += float(rng.normal(0, self.noise_std))
            l += float(rng.normal(0, self.noise_std))

        a = max(v - self.theta_value, l - self.theta_bluff - self.kappa * v)

        if a > 0:
            action = BET_RAISE if BET_RAISE in legal_actions else CHECK_CALL
        elif v > self.theta_call:
            action = CHECK_CALL
        else:
            action = FOLD if FOLD in legal_actions else CHECK_CALL

        bet_size = 0.0
        if action == BET_RAISE:
            bet_size = (a / V_SCALE) * max(situation.pot, 1.0)
        return action, bet_size

    def _decide_from_gto_charts(
        self, situation: Situation, legal_actions: list[int],
    ) -> tuple[int, float] | None:
        """Checks this genome's active GTO_SPOTS (gto_flags on), in catalog
        order, for the first one whose spot matches `situation` -- if found,
        plays exactly what that spot's chart says for this hand, bypassing
        V/L entirely. Returns None if no active spot applies, so decide()
        falls through to the normal linear decision."""
        for i, spot in enumerate(GTO_SPOTS):
            if self.gto_flags[i] <= 0.5:
                continue
            resolved = resolve_spot_action(spot, situation)
            if resolved is None:
                continue
            kind, size_spec = resolved
            if kind == "fold":
                action = FOLD if FOLD in legal_actions else CHECK_CALL
                return action, 0.0
            if kind == "call":
                return CHECK_CALL, 0.0
            # kind == "raise"
            if BET_RAISE not in legal_actions:
                return CHECK_CALL, 0.0
            if size_spec is None:
                bet_size = situation.my_stack  # "allin": shove the full stack
            elif size_spec[0] == "pot":
                bet_size = size_spec[1] * max(situation.pot, 1.0)
            else:  # ("bb", n): raise so my total commitment this street reaches n big blinds.
                # call_amount == current_bet for every non-blind seat that hasn't
                # committed anything yet this street (the common case these
                # spots are written for), so this is exact there and a close
                # approximation for the blinds.
                target_total = size_spec[1] * situation.big_blind
                bet_size = max(target_total - situation.call_amount, 0.0)
            return BET_RAISE, bet_size
        return None


def save_population(genomes: list[Genome], path: str) -> None:
    """Saves a whole generation (best-first, if the caller ranked it) as one
    file so a later run can reload it wholesale as its starting population."""
    np.save(path, np.stack([g.flatten() for g in genomes]))


def load_population(path: str) -> list[Genome]:
    return [Genome.unflatten(row) for row in np.load(path)]
