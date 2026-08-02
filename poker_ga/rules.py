"""The object-oriented decision layer: a genome's strategy is a per-street,
priority-ordered list of `Rule` objects (each a `StandardRule` -- an evolved
conjunction of `Condition`s plus an `Action` -- or a `GTOSpotRule` wrapping a
memorized `gto.py` chart), replacing the earlier array-based
condition_features/condition_buckets/rule_actions/... representation.

Everything here is a frozen (immutable) dataclass: `mutate()` always returns
a *new* instance rather than mutating in place, the same copy-on-write style
`Genome` already used for its array genes. This means these objects ARE the
genes now -- not a cached view over arrays -- so mutation/crossover operate
directly on Rule/Action/Condition object graphs (see genome.py). Immutability
buys two things for free: safe sharing of unchanged rules across genomes
(crossover/copy never need to deep-copy a rule it didn't touch) and
structural `__eq__`/`__hash__` (round-trip tests, dedup-by-content).

Rule shape is now fully variable and evolvable, not fixed-size:
  - A rule's number of conditions varies in [MIN_CONDITIONS_PER_RULE,
    MAX_CONDITIONS_PER_RULE] -- there is no wildcard sentinel anymore (see
    Condition below); every condition in a rule's list is a real, active
    condition, and "how many conditions" is itself a mutable, bounded gene.
  - A street's number of rules varies in [MIN_RULES_PER_STREET,
    MAX_RULES_PER_STREET], independently per street per genome (see
    genome.py's Genome.rules_by_street) -- no more fixed RULES_PER_STREET.
  - Each StandardRule carries its own evolvable `priority` (float, higher
    checked first); a street's rules are always stored priority-sorted, so
    decide()/reporting just walk them in stored order.
Both of these feed a single combined complexity metric (Genome.
nonzero_weight_count(): total conditions + total rules) driving main.py's
existing --sparsity-penalty, so evolution is pushed toward simpler rule
lists exactly the way it was already pushed toward fewer active conditions.

Bucketing (num_buckets/thresholds -- "which cut points turn this feature's
0-1 normalized value into a bucket index") also moved here from being
genome-wide: each Condition now evolves its own independently (see
Condition below) rather than every rule sharing one definition per feature.

GTO spot rules (GTOSpotRule) stay a hard-override layer, exactly as before
this refactor: Genome.decide() always checks active GTOSpotRules first,
unconditionally, ahead of the street's StandardRule list -- they do NOT
carry a priority or interleave with StandardRules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import strategy
from features import Situation
from gto import GTO_SPOTS, GTOSpot, resolve_spot_action

# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """What a Rule wants to do, independent of game mechanics (legal_actions,
    chip clamping) -- the common return type both StandardRule and
    GTOSpotRule resolve to. `bet_size` (chips) is only meaningful when
    kind == "raise"."""

    kind: str  # "fold" | "call" | "raise"
    bet_size: float = 0.0


@dataclass(frozen=True)
class MatchContext:
    """Built once per Genome.decide() call, passed to every rule's
    evaluate(). `condition_values`: raw normalized 0-1 feature readings
    (extract_features' output sliced via strategy.CONDITION_FEATURES_FULL_INDEX)
    -- NOT pre-bucketed, since bucketing is now a per-Condition gene rather
    than a genome-wide one; each Condition buckets its own feature value
    independently. `hole_category`: exact preflop hole-hand-category index
    (only meaningful/computed on the preflop street; -1 otherwise)."""

    situation: Situation
    condition_values: np.ndarray
    hole_category: int


# ---------------------------------------------------------------------------
# Action hierarchy
# ---------------------------------------------------------------------------


class Action(ABC):
    """What a matched rule actually plays: one fixed category (SingleAction)
    or a 50/50 coin flip between two (MixAction) -- the only source of
    randomized/mixed-strategy behavior in a genome's decisions."""

    @abstractmethod
    def resolve(self, rng: np.random.Generator | None) -> int:
        """Returns the action-category index (into strategy.ACTION_CATEGORIES)
        to play this time."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable label for reports."""
        ...

    @abstractmethod
    def mutate(self, rng: np.random.Generator, rate: float) -> "Action":
        """Returns a mutated copy."""
        ...

    @abstractmethod
    def to_indices(self) -> tuple[int, int]:
        """(primary_index, mix_index_or_NO_MIX) -- avoids isinstance checks
        in serialization."""
        ...


def _mutate_ordinal_scalar(value: int, upper_exclusive: int, rng: np.random.Generator) -> int:
    """Scalar counterpart to strategy.py's old array-vectorized ordinal
    mutation: 80% of the time nudge +-1 (clipped), 20% of the time jump to a
    uniformly random value in range -- used for gene kinds where the integer
    *is* meaningfully ordered (action category, bucket index)."""
    if rng.random() < 0.8:
        step = 1 if rng.random() < 0.5 else -1
        return int(np.clip(value + step, 0, upper_exclusive - 1))
    return int(rng.integers(0, upper_exclusive))


@dataclass(frozen=True)
class SingleAction(Action):
    action_index: int

    def resolve(self, rng: np.random.Generator | None = None) -> int:
        return self.action_index

    def describe(self) -> str:
        return strategy.ACTION_CATEGORIES[self.action_index]

    def mutate(self, rng: np.random.Generator, rate: float) -> Action:
        new_index = self.action_index
        if rng.random() < rate:
            new_index = _mutate_ordinal_scalar(new_index, strategy.NUM_ACTION_CATEGORIES, rng)
        if rng.random() < rate:
            mix_index = int(rng.integers(-1, strategy.NUM_ACTION_CATEGORIES))
            if mix_index != strategy.NO_MIX:
                return MixAction(new_index, mix_index)
        return SingleAction(new_index)

    def to_indices(self) -> tuple[int, int]:
        return self.action_index, strategy.NO_MIX


@dataclass(frozen=True)
class MixAction(Action):
    primary_index: int
    secondary_index: int

    def resolve(self, rng: np.random.Generator | None = None) -> int:
        if rng is not None and rng.random() < 0.5:
            return self.secondary_index
        return self.primary_index

    def describe(self) -> str:
        return (
            f"50% {strategy.ACTION_CATEGORIES[self.primary_index]} / "
            f"50% {strategy.ACTION_CATEGORIES[self.secondary_index]}"
        )

    def mutate(self, rng: np.random.Generator, rate: float) -> Action:
        new_primary = self.primary_index
        if rng.random() < rate:
            new_primary = _mutate_ordinal_scalar(new_primary, strategy.NUM_ACTION_CATEGORIES, rng)
        if rng.random() < rate:
            mix_index = int(rng.integers(-1, strategy.NUM_ACTION_CATEGORIES))
            if mix_index == strategy.NO_MIX:
                return SingleAction(new_primary)
            return MixAction(new_primary, mix_index)
        return MixAction(new_primary, self.secondary_index)

    def to_indices(self) -> tuple[int, int]:
        return self.primary_index, self.secondary_index


MIX_INIT_PROB = 0.2  # fraction of freshly random rules that start as a Mix


def random_action(rng: np.random.Generator, mix_init_prob: float = MIX_INIT_PROB) -> Action:
    """Factory for a freshly random Action -- not an Action classmethod since
    which concrete type comes back is inherently ambiguous."""
    primary = int(rng.integers(0, strategy.NUM_ACTION_CATEGORIES))
    if rng.random() < mix_init_prob:
        secondary = int(rng.integers(0, strategy.NUM_ACTION_CATEGORIES))
        return MixAction(primary, secondary)
    return SingleAction(primary)


# ---------------------------------------------------------------------------
# Condition -- a single (feature, bucket) check, owning its own bucketing genes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One (feature, required bucket) check. Unlike the old array
    representation, there is no wildcard sentinel -- every Condition in a
    rule's conditions tuple is real and active; "no conditions" is prevented
    structurally by MIN_CONDITIONS_PER_RULE >= 1 (see StandardRule), not by
    a repair pass over dead wildcard slots.

    num_buckets/thresholds are this condition's OWN bucketing scheme --
    genome-wide sharing was removed at the user's request, so two different
    conditions on the same feature (in different rules) can each evolve
    different cut points. For a boolean feature these are always fixed
    (2 buckets, split at 0.5) and never mutate -- a structural fact about
    booleans, not an evolvable choice."""

    feature_index: int
    num_buckets: int
    thresholds: tuple[float, ...]  # length MAX_BUCKETS - 1, sorted; only the first (num_buckets-1) entries are read
    bucket: int

    @property
    def _is_boolean(self) -> bool:
        return strategy.CONDITION_FEATURES[self.feature_index].kind == "boolean"

    @property
    def effective_bucket(self) -> int:
        """This condition's stored `bucket` clipped to one that actually
        exists for its own `num_buckets` -- can only go stale if
        `num_buckets` itself mutates down after `bucket` was set relative to
        a larger count (see Condition.mutate)."""
        return min(self.bucket, self.num_buckets - 1)

    def matches(self, condition_values: np.ndarray) -> bool:
        value = float(condition_values[self.feature_index])
        actual_bucket = strategy.compute_bucket(value, self.num_buckets, self.thresholds)
        return actual_bucket == self.effective_bucket

    def describe(self) -> str:
        spec = strategy.CONDITION_FEATURES[self.feature_index]
        label = strategy.describe_bucket(spec, self.bucket, self.num_buckets, self.thresholds)
        return label if spec.kind == "boolean" else f"{spec.label} = {label}"

    def mutate(
        self, rng: np.random.Generator, rate: float, eligible_indices: np.ndarray, threshold_scale: float,
    ) -> "Condition":
        """Four independent per-gene rolls: feature identity, num_buckets,
        thresholds, and the required bucket -- matching the granularity
        every other gene kind in this codebase already uses. If feature
        identity changes to/from a boolean feature, num_buckets/thresholds
        reset to the appropriate fixed/fresh-random shape for the new
        feature's kind (a boolean's 2-buckets-at-0.5 is structural, not
        inherited from whatever the previous feature had)."""
        feature_index = self.feature_index
        if rng.random() < rate:
            feature_index = int(eligible_indices[rng.integers(0, len(eligible_indices))])

        is_boolean = strategy.CONDITION_FEATURES[feature_index].kind == "boolean"
        if feature_index != self.feature_index:
            # Feature identity changed -- reset bucketing genes for the new
            # feature's kind rather than carrying over values that made
            # sense for a different feature.
            if is_boolean:
                num_buckets = 2
                thresholds = (0.5,) * (strategy.MAX_BUCKETS - 1)
            else:
                num_buckets = int(rng.integers(strategy.MIN_BUCKETS, strategy.MAX_BUCKETS + 1))
                thresholds = tuple(sorted(float(x) for x in rng.random(strategy.MAX_BUCKETS - 1)))
            bucket = int(rng.integers(0, num_buckets))
            return Condition(feature_index, num_buckets, thresholds, bucket)

        num_buckets = self.num_buckets
        thresholds = self.thresholds
        bucket = self.bucket
        if not is_boolean:
            if rng.random() < rate:
                num_buckets = strategy.MAX_BUCKETS if num_buckets == strategy.MIN_BUCKETS else strategy.MIN_BUCKETS
            if rng.random() < rate:
                noise = rng.normal(0.0, threshold_scale, size=len(thresholds))
                mutated = np.clip(np.array(thresholds) + noise, 0.0, 1.0)
                thresholds = tuple(sorted(float(x) for x in mutated))
        if rng.random() < rate:
            bucket = _mutate_ordinal_scalar(bucket, strategy.MAX_BUCKETS, rng)

        return Condition(feature_index, num_buckets, thresholds, bucket)

    @classmethod
    def random(cls, rng: np.random.Generator, eligible_indices: np.ndarray) -> "Condition":
        feature_index = int(eligible_indices[rng.integers(0, len(eligible_indices))])
        is_boolean = strategy.CONDITION_FEATURES[feature_index].kind == "boolean"
        if is_boolean:
            num_buckets = 2
            thresholds = (0.5,) * (strategy.MAX_BUCKETS - 1)
        else:
            num_buckets = int(rng.integers(strategy.MIN_BUCKETS, strategy.MAX_BUCKETS + 1))
            thresholds = tuple(sorted(float(x) for x in rng.random(strategy.MAX_BUCKETS - 1)))
        bucket = int(rng.integers(0, num_buckets))
        return cls(feature_index, num_buckets, thresholds, bucket)


# ---------------------------------------------------------------------------
# Rule hierarchy
# ---------------------------------------------------------------------------


class Rule(ABC):
    """One "if this situation, then this action" entry a decision list
    checks in order. Concrete subclasses: StandardRule (an evolved rule with
    a learnable priority) and GTOSpotRule (a memorized named-spot chart,
    always checked as a hard override ahead of any StandardRule -- see
    genome.py's Genome.decide())."""

    @abstractmethod
    def evaluate(self, context: MatchContext, rng: np.random.Generator | None) -> Decision | None:
        """Returns this rule's Decision if it fires for `context`, else
        None."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable rendering for strategy reports. Uniform signature
        across subclasses even though GTOSpotRule needs no extra state --
        nothing today calls describe() polymorphically across mixed rule
        lists, but the ABC should stay honest about its contract."""
        ...


def _standard_decision(action_index: int, situation: Situation) -> Decision:
    if action_index == strategy.ACTION_FOLD:
        return Decision("fold")
    if action_index == strategy.ACTION_CALL:
        return Decision("call")
    if action_index == strategy.ACTION_ALLIN:
        return Decision("raise", situation.my_stack)
    fraction = strategy.RAISE_POT_FRACTION[action_index]
    return Decision("raise", fraction * max(situation.pot, 1.0))


def _gto_decision(kind: str, size_spec: tuple[str, float] | None, situation: Situation) -> Decision:
    if kind == "fold":
        return Decision("fold")
    if kind == "call":
        return Decision("call")
    # kind == "raise"
    if size_spec is None:
        return Decision("raise", situation.my_stack)  # "allin": shove the full stack
    if size_spec[0] == "pot":
        return Decision("raise", size_spec[1] * max(situation.pot, 1.0))
    # ("bb", n): raise so my total commitment this street reaches n big blinds.
    target_total = size_spec[1] * situation.big_blind
    return Decision("raise", max(target_total - situation.call_amount, 0.0))


# Rule-shape policy constants: how many conditions a rule may carry, how many
# rules a street may carry, and how much less often a structural (grow/
# shrink) mutation fires relative to an ordinary gene nudge at the same
# `rate` -- adding/removing a whole condition or rule is a bigger move than
# nudging a value, so it's damped the same way THRESHOLD_MUTATION_SCALE_FACTOR
# already damps continuous mutation for a different gene kind.
MIN_CONDITIONS_PER_RULE = 1
MAX_CONDITIONS_PER_RULE = 5
MIN_RULES_PER_STREET = 1
MAX_RULES_PER_STREET = 30
STRUCTURAL_MUTATION_RATE_FACTOR = 0.3

# Per-slot probability a freshly random rule starts with one more condition
# beyond its MIN_CONDITIONS_PER_RULE floor -- mirrors the old
# CONDITION_ACTIVE_INIT_PROB "start sparse" philosophy.
CONDITION_GROWTH_INIT_PROB = 0.3

# Init probability per category per freshly random preflop rule -- a neutral
# coin flip (unlike CONDITION_GROWTH_INIT_PROB's "start sparse" philosophy)
# since this field is mandatory, not optional: there's no "inactive" state to
# default to, so a 50/50 starting spread is the least-biased starting point.
HOLE_CATEGORY_INIT_PROB = 0.5

PRIORITY_MUTATION_SCALE_FACTOR = strategy.THRESHOLD_MUTATION_SCALE_FACTOR


@dataclass(frozen=True)
class StandardRule(Rule):
    """conditions: variable length in [MIN_CONDITIONS_PER_RULE,
    MAX_CONDITIONS_PER_RULE] -- a conjunction, all must match. action: this
    rule's Action (single or Mix). hole_category_mask: mandatory for every
    preflop rule (which of the 12 hole-hand-category buckets this rule
    applies to, checked exactly, not through the bucket mechanism -- see
    features.py's hole_hand_category_norm), None for flop/turn/river rules.
    priority: evolvable float: within its street's pool, higher priority is
    checked first -- Genome always stores each street's rules pre-sorted
    descending by priority, so decide()/reporting just walk stored order."""

    conditions: tuple[Condition, ...]
    action: Action
    hole_category_mask: tuple[bool, ...] | None
    priority: float

    def evaluate(self, context: MatchContext, rng: np.random.Generator | None) -> Decision | None:
        if self.hole_category_mask is not None and not self.hole_category_mask[context.hole_category]:
            return None
        for cond in self.conditions:
            if not cond.matches(context.condition_values):
                return None
        action_index = self.action.resolve(rng)
        return _standard_decision(action_index, context.situation)

    def describe(self) -> str:
        phrases = []
        if self.hole_category_mask is not None:
            claimed = [
                label for label, flag in zip(strategy.HOLE_CATEGORY_LABELS, self.hole_category_mask) if flag
            ]
            categories_text = ", ".join(claimed) if claimed else "*none -- never matches*"
            phrases.append(f"Hole Category in {{{categories_text}}}")
        phrases.extend(cond.describe() for cond in self.conditions)
        condition_text = " AND ".join(phrases) if phrases else "*(any situation)*"
        return f"IF {condition_text} THEN **{self.action.describe()}**"

    def mutate(
        self, rng: np.random.Generator, rate: float, eligible_indices: np.ndarray, threshold_scale: float,
    ) -> "StandardRule":
        structural_rate = rate * STRUCTURAL_MUTATION_RATE_FACTOR

        conditions = [cond.mutate(rng, rate, eligible_indices, threshold_scale) for cond in self.conditions]
        if len(conditions) < MAX_CONDITIONS_PER_RULE and rng.random() < structural_rate:
            conditions.append(Condition.random(rng, eligible_indices))
        if len(conditions) > MIN_CONDITIONS_PER_RULE and rng.random() < structural_rate:
            conditions.pop(int(rng.integers(0, len(conditions))))

        action = self.action.mutate(rng, rate)

        hole_category_mask = self.hole_category_mask
        if hole_category_mask is not None:
            hole_category_mask = tuple(
                (not v) if rng.random() < rate else v for v in hole_category_mask
            )

        priority = self.priority
        if rng.random() < rate:
            noise = float(rng.normal(0.0, PRIORITY_MUTATION_SCALE_FACTOR))
            priority = float(np.clip(priority + noise, 0.0, 1.0))

        return StandardRule(tuple(conditions), action, hole_category_mask, priority)

    @classmethod
    def random(
        cls, rng: np.random.Generator, eligible_indices: np.ndarray, hole_category_init_prob: float | None = None,
        condition_growth_prob: float = CONDITION_GROWTH_INIT_PROB,
    ) -> "StandardRule":
        """hole_category_init_prob: pass a value (e.g. HOLE_CATEGORY_INIT_PROB)
        for a preflop rule; None (the default) for flop/turn/river rules.
        condition_growth_prob: lets a caller (Genome.random's `scale` knob)
        scale the starting condition count up/down from the default."""
        num_conditions = MIN_CONDITIONS_PER_RULE
        while num_conditions < MAX_CONDITIONS_PER_RULE and rng.random() < condition_growth_prob:
            num_conditions += 1
        conditions = tuple(Condition.random(rng, eligible_indices) for _ in range(num_conditions))

        action = random_action(rng, MIX_INIT_PROB)
        priority = float(rng.random())

        hole_category_mask = None
        if hole_category_init_prob is not None:
            hole_category_mask = tuple(
                bool(rng.random() < hole_category_init_prob) for _ in range(strategy.NUM_HOLE_CATEGORIES)
            )

        return cls(conditions, action, hole_category_mask, priority)


@dataclass(frozen=True)
class GTOSpotRule(Rule):
    """Wraps one gto.py GTOSpot catalog entry. `active`: whether this
    genome currently trusts this spot's chart (the evolvable gene -- the
    spot itself, `spot`, is a fixed catalog entry, never evolved). Always a
    hard-override layer: Genome.decide() checks every active GTOSpotRule
    before any StandardRule, regardless of any StandardRule's priority."""

    spot: GTOSpot
    active: bool

    def evaluate(self, context: MatchContext, rng: np.random.Generator | None) -> Decision | None:
        if not self.active:
            return None
        resolved = resolve_spot_action(self.spot, context.situation)
        if resolved is None:
            return None
        kind, size_spec = resolved
        return _gto_decision(kind, size_spec, context.situation)

    def describe(self) -> str:
        return f"{self.spot.label} ({self.spot.matcher.describe()})"

    def mutate(self, rng: np.random.Generator, rate: float) -> "GTOSpotRule":
        active = (not self.active) if rng.random() < rate else self.active
        return GTOSpotRule(self.spot, active)

    @classmethod
    def catalog(cls, rng: np.random.Generator, init_prob: float) -> tuple["GTOSpotRule", ...]:
        return tuple(cls(spot, bool(rng.random() < init_prob)) for spot in GTO_SPOTS)


def gto_rules_from_saved(
    saved_gto: dict, rng: np.random.Generator, init_prob: float,
) -> tuple[tuple[GTOSpotRule, ...], set[str], list[str]]:
    """Rebuilds one GTOSpotRule per GTO_SPOTS catalog entry from a saved
    {spot_key: 0.0/1.0} dict (see Genome.to_dict's "gto_flags"). `rng` is
    drawn only for keys missing from `saved_gto`, in catalog order,
    interleaved with reading present keys -- not a draw for every spot
    regardless. Returns (gto_rules, unknown_keys, missing_keys) so the
    caller (Genome.from_dict) can print its own aggregated warnings --
    keeps genome.py from needing to import gto.py directly at all."""
    known_keys = {spot.key for spot in GTO_SPOTS}
    unknown_keys = set(saved_gto) - known_keys
    missing_keys = [spot.key for spot in GTO_SPOTS if spot.key not in saved_gto]
    gto_rules = []
    for spot in GTO_SPOTS:
        if spot.key in saved_gto:
            active = float(saved_gto[spot.key]) > 0.5
        else:
            active = bool(rng.random() < init_prob)
        gto_rules.append(GTOSpotRule(spot, active))
    return tuple(gto_rules), unknown_keys, missing_keys
