"""The rule-list decision engine: turns a genome's evolved genes into a small,
ordered "if this situation, then this action" list a human could actually
memorize and execute at a table -- the replacement for the old linear V/L/
theta system (see genome.py's module docstring for that history).

Rules are organized into 4 independent, fixed-size pools, one per street
(RULES_PER_STREET each) -- every rule belongs to exactly one street, and each
street has its own separate rule-count limit, so a genome can't spend its
whole "budget" of rules on preflop and leave the river bare. Two fields are
*mandatory* on every rule rather than optional conditions competing for one
of its CONDITIONS_PER_RULE slots (and so don't count against
nonzero_weight_count()'s complexity/sparsity accounting, which only counts
active *optional* conditions):

  - Which street a rule belongs to isn't a gene at all -- it's simply which
    of the 4 per-street pools the rule lives in (see Genome.condition_features
    etc., shaped (NUM_STREETS, RULES_PER_STREET, ...)). decide() only ever
    checks the pool matching the current Situation.street.
  - Every preflop rule also carries a mandatory hole_category_mask (see
    Genome) -- which of the 12 hole-hand-category buckets (features.py's
    hole_hand_category_norm -- Premium Pairs, Axs, Suited Connectors, Junk,
    etc.) it applies to. Preflop strategy is built around exactly this axis
    on a real range chart, so it gets full 12-way resolution and a mandatory
    presence, rather than being squeezed through the usual 2-3-bucket
    mechanism and an optional condition slot like everything else.

Two kinds of evolvable gene live here besides the rules themselves:

  Feature bucketing (genome-wide, shared by every rule): every "condition"
  feature (see CONDITION_FEATURES -- the ~49 generalized concepts a rule can
  reference, not the ~150 one-hot indicator children, and not street_norm,
  which is redundant now that street is handled structurally) already lives
  on an evenly-spaced 0-1 scale (features.py normalizes everything that
  way). That means "learn a threshold" and "learn to bucket a categorical
  into groups" are the *same* mechanism: pick `num_buckets` (2 or 3) cut
  points on that 0-1 line. This is genome-global rather than per-rule, so a
  concept like "Hand Strength" has one shared definition every rule reads
  from -- "weak/medium/strong, cut here" -- rather than every rule
  re-deriving its own cutoffs. Boolean features need no genes at all:
  they're already exactly 2 buckets (0/1).

  Rules (a fixed-size ordered decision list, per street): each rule is a
  small conjunction of up to CONDITIONS_PER_RULE (feature, required-bucket)
  checks -- any individual slot can be a wildcard, "don't care," but every
  rule must have at least one active (non-wildcard) condition slot (see
  enforce_min_one_condition) -- plus one action category (and, for preflop
  rules, the mandatory hole_category_mask above). A rule with every slot
  wildcard would match literally anything, which is what the pool's own
  trailing default is for; only that default (not a "rule") is allowed to
  match unconditionally. Checked in fixed array order within their street's
  pool, first full match wins (the same idiom gto.py's GTOSpot.action_ranges/
  SpotMatcher already use); no match at all falls back to Fold (matching
  GTOSpot.default_action's "everything not colored in is a fold"
  convention). A rule's array *position* carries no priority meaning beyond
  that fixed check order -- like a weight vector's index today, a slot's
  content is what matters, not which slot it happens to sit in -- so
  there's no reordering mutation operator.

Multi-street plans ("raise now, fold if raised back") are never represented
explicitly: every decision re-evaluates the current street's rule pool fresh
from the current Situation. "Fold if raised back" simply falls out of a
*later* decision's bigger call_amount/facing_bet routing to a different
(likely Fold) rule -- no state needs to be carried between decisions.
Likewise "call up to X, else fold" is just an ordinary rule condition on the
existing call_amount_norm feature's bucket, not a special parameter.
"""

from __future__ import annotations

import numpy as np

from features import FEATURE_SPECS, FeatureSpec, group_of

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------

# The "generalized" top-level features (parents of a one-hot family, or
# standalone booleans) usable as rule conditions -- the ~150 linked
# indicator children are just precomputed alternate views of these same
# concepts (see features.py's module docstring), so exposing them too would
# only let two conditions restate the same fact under different keys.
# street_norm is excluded too: which street a rule applies to is now a
# mandatory, structural property of which per-street pool it lives in (see
# module docstring), so referencing it again as a regular condition would be
# permanently redundant -- every rule in, say, the flop pool only ever sees
# flop situations, making any street_norm condition on it either trivially
# always-true or permanently dead.
CONDITION_FEATURES: list[FeatureSpec] = [
    s for s in FEATURE_SPECS if s.linked_to is None and s.key != "street_norm"
]
NUM_CONDITION_FEATURES = len(CONDITION_FEATURES)

# Index into the *full* features.py vector (as extract_features returns it)
# for each condition feature, precomputed once so decide() doesn't need to
# rebuild this mapping (or do a linear FEATURE_NAMES.index lookup) every call.
CONDITION_FEATURES_FULL_INDEX = np.array(
    [i for i, s in enumerate(FEATURE_SPECS) if s.linked_to is None and s.key != "street_norm"],
    dtype=np.int64,
)

_FEATURE_INDEX_BY_KEY: dict[str, int] = {s.key: i for i, s in enumerate(CONDITION_FEATURES)}

BOOLEAN_MASK = np.array([s.kind == "boolean" for s in CONDITION_FEATURES], dtype=bool)
# Indices (into CONDITION_FEATURES) of features that need bucketing genes --
# every non-boolean feature. Booleans are already exactly 0/1, no genes needed.
BUCKETABLE_INDICES = np.flatnonzero(~BOOLEAN_MASK)
NUM_BUCKETABLE = len(BUCKETABLE_INDICES)
# Inverse of the above: condition feature index -> row in the (NUM_BUCKETABLE,
# ...) gene arrays, or -1 for a boolean feature (which has no such row).
_BUCKET_GENE_ROW = np.full(NUM_CONDITION_FEATURES, -1, dtype=np.int64)
_BUCKET_GENE_ROW[BUCKETABLE_INDICES] = np.arange(NUM_BUCKETABLE)

MIN_BUCKETS = 2
MAX_BUCKETS = 3  # most features settle on 2-3 buckets; nothing hard-caps below this

WILDCARD = -1  # condition_features sentinel: "don't care" for this condition slot

# ---------------------------------------------------------------------------
# Streets and rule pools
# ---------------------------------------------------------------------------

NUM_STREETS = 4  # matches Situation.street: 0=preflop, 1=flop, 2=turn, 3=river
STREET_LABELS = ("Preflop", "Flop", "Turn", "River")
PREFLOP = 0

RULES_PER_STREET = 30  # each street's own independent rule-count limit
CONDITIONS_PER_RULE = 3
NUM_RULES = NUM_STREETS * RULES_PER_STREET  # derived: total rule slots across all 4 streets

# Fraction of a rule's condition slots that start non-wildcard for a freshly
# random genome -- kept low (mirroring gto.py's GTO_INIT_PROB philosophy:
# "earn complexity, don't start with it") while still giving crossover/
# mutation real material to work with immediately.
CONDITION_ACTIVE_INIT_PROB = 0.3

# Old GAConfig.mutation_scale (~0.3 by default) was calibrated for the V/L
# system's 0-100 range. Thresholds live on features.py's native 0-1 scale,
# so mutating them with the same raw scale would jitter by up to ~30% of
# the entire range per mutation event -- this factor rescales
# `continuous_scale` down to something sane for a 0-1 domain instead.
THRESHOLD_MUTATION_SCALE_FACTOR = 0.1

# Named ACTION_* (not FOLD/CALL/...) to stay visually distinct from
# genome.py's own FOLD/CHECK_CALL/BET_RAISE game-action constants -- these
# index ACTION_CATEGORIES (a rule's chosen strategy category), those index
# legal_actions (what the game engine will actually accept).
#
# Raise is 6 separate fixed-size categories rather than one "Raise" category
# plus a shared genome-wide size gene: a rule now picks its size directly
# (e.g. "Raise 75% Pot"), so different rules -- a value raise on the river vs
# a preflop 3-bet -- can each evolve their own size independently, still off
# a small, human-memorizable menu (the same pot-fraction sizing convention
# gto.py's "raise_NN" action tokens already use). Ordered from least to most
# aggressive alongside Fold/Call/All-In so the whole list is one continuum --
# mutation's nudge-mostly/jump-sometimes move (see _mutate_ordinal) then
# means "shift to a slightly bigger/smaller size" most of the time.
(
    ACTION_FOLD, ACTION_CALL,
    ACTION_RAISE_25, ACTION_RAISE_50, ACTION_RAISE_75,
    ACTION_RAISE_100, ACTION_RAISE_125, ACTION_RAISE_150,
    ACTION_ALLIN,
) = range(9)
ACTION_CATEGORIES = [
    "Check / Fold (Give up)", "Call",
    "Raise 25% Pot", "Raise 50% Pot", "Raise 75% Pot",
    "Raise 100% Pot", "Raise 125% Pot", "Raise 150% Pot",
    "All-In",
]
NUM_ACTION_CATEGORIES = len(ACTION_CATEGORIES)

# action index -> pot-fraction raise size, for every Raise category (added
# on top of whatever's already bet -- the same convention gto.py's pot-
# fraction "raise_NN" tokens use).
RAISE_ACTIONS = (
    ACTION_RAISE_25, ACTION_RAISE_50, ACTION_RAISE_75,
    ACTION_RAISE_100, ACTION_RAISE_125, ACTION_RAISE_150,
)
RAISE_POT_FRACTION = {
    ACTION_RAISE_25: 0.25, ACTION_RAISE_50: 0.5, ACTION_RAISE_75: 0.75,
    ACTION_RAISE_100: 1.0, ACTION_RAISE_125: 1.25, ACTION_RAISE_150: 1.5,
}

# A rule's action is either one fixed category (as above) or a "Mix": a
# 50/50 coin flip between two categories, decided fresh every time the rule
# fires. This is the *only* source of randomized/mixed-strategy behavior in
# a genome's decisions -- it replaces the earlier bucket-threshold jitter
# mechanism entirely (deterministic, explicit, and directly readable in a
# strategy report, unlike a hidden noise-std blurring which bucket a hand
# fell into).
NO_MIX = -1  # rule_mix_actions sentinel: this rule is a single fixed action, not a Mix

# Fraction of freshly random rules that start as a Mix rather than a single
# fixed action -- kept low, same "earn complexity, don't start with it"
# philosophy as CONDITION_ACTIVE_INIT_PROB/GTO_INIT_PROB.
MIX_INIT_PROB = 0.2


def resolve_rule_action(primary_action: int, mix_action: int, rng: np.random.Generator | None) -> int:
    """Returns the action a rule actually plays for one decision: its
    primary action, or -- if it's a Mix (mix_action != NO_MIX) and an rng
    was given -- primary/mix_action with 50/50 probability. Without an rng,
    always plays the primary action (deterministic), the same convention
    the old noise mechanism used when called without one."""
    if mix_action != NO_MIX and rng is not None and rng.random() < 0.5:
        return mix_action
    return primary_action


def describe_rule_action(primary_action: int, mix_action: int) -> str:
    """Human-readable label for a rule's action -- its fixed category, or
    "50% X / 50% Y" if it's a Mix."""
    if mix_action == NO_MIX:
        return ACTION_CATEGORIES[primary_action]
    return f"50% {ACTION_CATEGORIES[primary_action]} / 50% {ACTION_CATEGORIES[mix_action]}"


# ---------------------------------------------------------------------------
# Hole hand category -- the mandatory preflop rule axis
# ---------------------------------------------------------------------------

HOLE_CATEGORY_FEATURE_KEY = "hole_hand_category_norm"
_HOLE_CATEGORY_CONDITION_INDEX = _FEATURE_INDEX_BY_KEY[HOLE_CATEGORY_FEATURE_KEY]
HOLE_CATEGORY_LABELS = [label for _point, label in CONDITION_FEATURES[_HOLE_CATEGORY_CONDITION_INDEX].value_table]
NUM_HOLE_CATEGORIES = len(HOLE_CATEGORY_LABELS)

# Init probability per category per freshly random preflop rule -- a neutral
# coin flip (unlike CONDITION_ACTIVE_INIT_PROB's "start sparse" philosophy)
# since this field is mandatory, not optional: there's no "inactive" state to
# default to, so a 50/50 starting spread is the least-biased starting point.
HOLE_CATEGORY_INIT_PROB = 0.5


def feature_index(key: str) -> int:
    return _FEATURE_INDEX_BY_KEY[key]


def bucket_gene_row(condition_index: int) -> int:
    """Row into the (NUM_BUCKETABLE, ...) num_buckets/thresholds gene arrays
    for this condition feature, or -1 if it's boolean (no bucketing genes)."""
    return int(_BUCKET_GENE_ROW[condition_index])


def hole_category_index(condition_values: np.ndarray) -> int:
    """Exact (not bucketed, not jittered) hole-hand-category index 0..11,
    read directly out of `condition_values` (the same array decide() slices
    out of extract_features via CONDITION_FEATURES_FULL_INDEX). features.py
    normalizes hole_hand_category_norm as category_index / 11, so this
    round-trips exactly. Preflop rules match on this directly (see
    match_preflop_rule) rather than through the usual bucket-threshold
    mechanism -- a real range chart is exact about which starting hands are
    in which line, not fuzzy, and this is preflop's one mandatory axis."""
    value = float(condition_values[_HOLE_CATEGORY_CONDITION_INDEX])
    return int(round(value * (NUM_HOLE_CATEGORIES - 1)))


# ---------------------------------------------------------------------------
# Per-street condition eligibility: which features a street's rules may
# reference at all
# ---------------------------------------------------------------------------

# Preflop has no board yet, so anything that only makes sense relative to
# one is meaningless there -- excluded from the pool preflop rules can draw
# conditions from at all (not just discouraged: flop/turn/river rules have
# no such restriction and may reference anything in CONDITION_FEATURES).
# Two kinds of feature qualify:
#   - The whole "Board / Flop Characteristics" group (board suit/pairing/
#     connectivity/wetness texture, the board's own high card).
#   - A specific handful of "Made Hand Features" that compare the hole cards
#     to the board (top/second/third pair, over/under/low pair, board
#     overcards) -- structurally these always evaluate to their same
#     "nothing to compare against yet" value preflop (extract_features
#     computes them from `board_ranks`, empty preflop), so referencing them
#     would be a permanently dead condition, same reasoning as the board-
#     texture group. hand_category_norm (Pair vs High Card from the hole
#     cards alone) and ace_high_no_pair/king_high_no_pair (from the hole
#     cards' own high card) are *not* excluded -- both are well-defined,
#     genuinely informative preflop.
_BOARD_TEXTURE_GROUP = "Board / Flop Characteristics"
_POST_FLOP_ONLY_MADE_HAND_KEYS = frozenset({
    "num_overcards_norm", "top_pair", "second_pair", "third_pair", "overpair", "underpair", "low_pair",
})
_PREFLOP_EXCLUDED_INDICES = frozenset(
    i for i, s in enumerate(CONDITION_FEATURES)
    if group_of(s) == _BOARD_TEXTURE_GROUP or s.key in _POST_FLOP_ONLY_MADE_HAND_KEYS
)
ELIGIBLE_CONDITION_INDICES_BY_STREET: tuple[np.ndarray, ...] = tuple(
    (
        np.array(sorted(set(range(NUM_CONDITION_FEATURES)) - _PREFLOP_EXCLUDED_INDICES), dtype=np.int64)
        if street == PREFLOP
        else np.arange(NUM_CONDITION_FEATURES, dtype=np.int64)
    )
    for street in range(NUM_STREETS)
)


# ---------------------------------------------------------------------------
# Rule validity: every rule must narrow down to *something*
# ---------------------------------------------------------------------------

def enforce_min_one_condition(
    condition_features: np.ndarray, condition_buckets: np.ndarray, eligible_indices: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """condition_features/condition_buckets: (num_rules, CONDITIONS_PER_RULE)
    for one street's rule pool. A rule with every condition slot wildcard
    matches literally any situation -- strictly weaker than, and
    indistinguishable in effect from, that pool's own trailing default (see
    genome.py's decide()), so it's never a meaningful thing for a rule
    (as opposed to the pool's default) to have evolved into. Any such rule
    gets exactly one randomly chosen slot filled with a random
    eligible-for-this-street feature and bucket -- the same "repair a
    violated structural invariant after random init/mutation/a stale save"
    pattern this representation has used before (see the removed
    made-hand/draw requirement this replaces the spirit of, in git history).
    Returns (new_cf, new_cb, num_repaired); does not mutate its inputs."""
    cf = condition_features.copy()
    cb = condition_buckets.copy()
    all_wildcard_rows = np.flatnonzero(np.all(cf == WILDCARD, axis=-1))
    if len(all_wildcard_rows) == 0:
        return cf, cb, 0
    slot = rng.integers(0, CONDITIONS_PER_RULE, size=len(all_wildcard_rows))
    cf[all_wildcard_rows, slot] = rng.choice(eligible_indices, size=len(all_wildcard_rows))
    cb[all_wildcard_rows, slot] = rng.integers(0, MAX_BUCKETS, size=len(all_wildcard_rows))
    return cf, cb, len(all_wildcard_rows)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def compute_bucket(value: float, num_buckets: int, thresholds: np.ndarray) -> int:
    """value: a feature's normalized 0-1 reading. thresholds: this feature's
    full (MAX_BUCKETS - 1,) cut-point row -- only the first (num_buckets - 1)
    entries are meaningful; the rest are ignored. Returns a bucket index in
    [0, num_buckets)."""
    cuts = np.sort(thresholds[: num_buckets - 1])
    return int(np.searchsorted(cuts, value, side="right"))


def compute_all_buckets(condition_values: np.ndarray, num_buckets: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """condition_values: (NUM_CONDITION_FEATURES,) normalized 0-1 readings,
    already sliced out of extract_features' full vector via
    CONDITION_FEATURES_FULL_INDEX. num_buckets/thresholds: this genome's
    bucketing genes (see Genome). Returns one bucket index per condition
    feature. Deterministic -- mixed-strategy randomness lives entirely in
    the action a matched rule plays (see resolve_rule_action), not here."""
    buckets = np.empty(NUM_CONDITION_FEATURES, dtype=np.int64)
    buckets[BOOLEAN_MASK] = (condition_values[BOOLEAN_MASK] > 0.5).astype(np.int64)
    for row, idx in enumerate(BUCKETABLE_INDICES):
        buckets[idx] = compute_bucket(float(condition_values[idx]), int(num_buckets[row]), thresholds[row])
    return buckets


def effective_condition_bucket(condition_feature_index: int, condition_bucket: int, num_buckets: np.ndarray) -> int:
    """Clips a condition's stored bucket requirement down to a bucket that
    actually exists for its feature. num_buckets (a genome's per-feature
    bucketing gene) can independently evolve to 2 or 3 for any feature,
    while a condition's stored bucket value always ranges over the full
    [0, MAX_BUCKETS) regardless of that -- so a condition can end up
    referencing a bucket its feature doesn't currently have (e.g. bucket 2
    on a feature genome-wide bucketed into just 2 groups). Rather than let
    that silently become a dead, "never matches" condition, it's clipped to
    the highest bucket that does exist. Boolean features are always exactly
    2 buckets, so this is a no-op for them."""
    row = _BUCKET_GENE_ROW[condition_feature_index]
    feature_num_buckets = int(num_buckets[row]) if row >= 0 else 2
    return min(int(condition_bucket), feature_num_buckets - 1)


def describe_bucket(spec: FeatureSpec, bucket_index: int, num_buckets: int, thresholds: np.ndarray) -> str:
    """Human-readable label for one bucket of a feature, e.g. "High Card -
    Two Pair" for hand_category_norm's low bucket at num_buckets=3, reusing
    the value_table labels features.py already defines for every
    categorical/continuous feature (including a representative-point table
    for genuinely continuous ones -- see features.py's _BUCKET_POINTS) --
    so a bucket reads the way a human names a chart region, not as a raw
    "0.00-0.33" cutoff. `bucket_index` is clipped to an existing bucket the
    same way effective_condition_bucket does, so a stored value the current
    num_buckets doesn't have never renders as a nonsensical/impossible label."""
    if spec.kind == "boolean":
        return spec.label if bucket_index == 1 else f"Not {spec.label}"

    bucket_index = min(bucket_index, num_buckets - 1)
    cuts = sorted(float(x) for x in thresholds[: num_buckets - 1])
    lo = cuts[bucket_index - 1] if bucket_index > 0 else 0.0
    hi = cuts[bucket_index] if bucket_index < num_buckets - 1 else 1.0
    is_last = bucket_index == num_buckets - 1

    labels_in_range = [
        label for point, label in spec.value_table
        if (lo <= point < hi) or (is_last and point == hi)
    ]
    if not labels_in_range:
        return f"{spec.label} {lo:.2f}-{hi:.2f}"
    if len(labels_in_range) == 1:
        return labels_in_range[0]
    return f"{labels_in_range[0]} - {labels_in_range[-1]}"


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------

def match_rule(
    buckets: np.ndarray, condition_features: np.ndarray, condition_buckets: np.ndarray, num_buckets: np.ndarray,
) -> bool:
    """condition_features/condition_buckets: one rule's (CONDITIONS_PER_RULE,)
    gene rows. num_buckets: this genome's per-feature bucketing gene (see
    Genome), needed to clip each condition's stored bucket down to one that
    actually exists (see effective_condition_bucket). A rule matches when
    every non-wildcard condition's (clipped) required bucket equals the
    current situation's bucket for that feature."""
    for cf, cb in zip(condition_features, condition_buckets):
        if cf == WILDCARD:
            continue
        if buckets[cf] != effective_condition_bucket(cf, cb, num_buckets):
            return False
    return True


def first_matching_rule_index(
    buckets: np.ndarray, condition_features: np.ndarray, condition_buckets: np.ndarray, num_buckets: np.ndarray,
) -> int | None:
    """Returns the index (within the given street's rule pool, in fixed
    array order) of the first matching rule, or None if no rule matches
    (caller should default to Fold). The caller looks up that rule's action
    itself (see Genome.rule_actions/rule_mix_actions and
    resolve_rule_action) -- this only identifies *which* rule fired."""
    for r in range(len(condition_features)):
        if match_rule(buckets, condition_features[r], condition_buckets[r], num_buckets):
            return r
    return None


def match_preflop_rule(
    buckets: np.ndarray, condition_features: np.ndarray, condition_buckets: np.ndarray, num_buckets: np.ndarray,
    hole_category_mask: np.ndarray, hole_category: int,
) -> bool:
    """Like match_rule, but a preflop rule also always requires the current
    hand's exact hole-hand-category to be one this rule claims
    (hole_category_mask[hole_category]) -- the mandatory, full-resolution
    axis every real preflop range chart is built around, on top of the
    (up to 3) regular optional conditions."""
    if not hole_category_mask[hole_category]:
        return False
    return match_rule(buckets, condition_features, condition_buckets, num_buckets)


def first_matching_preflop_rule_index(
    buckets: np.ndarray, condition_features: np.ndarray, condition_buckets: np.ndarray, num_buckets: np.ndarray,
    hole_category_mask: np.ndarray, hole_category: int,
) -> int | None:
    """Preflop counterpart to first_matching_rule_index -- see match_preflop_rule."""
    for r in range(len(condition_features)):
        if match_preflop_rule(
            buckets, condition_features[r], condition_buckets[r], num_buckets, hole_category_mask[r], hole_category,
        ):
            return r
    return None


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate_num_buckets(num_buckets: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Flips MIN_BUCKETS<->MAX_BUCKETS for selected genes -- there are only
    two valid values, so this is a bit-flip, not a jump/nudge choice."""
    mask = rng.random(num_buckets.shape) < rate
    flipped = np.where(num_buckets == MIN_BUCKETS, MAX_BUCKETS, MIN_BUCKETS)
    return np.where(mask, flipped, num_buckets)


def mutate_thresholds(thresholds: np.ndarray, rate: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Additive-gaussian mutation per cut point, clamped to (0,1) and
    re-sorted per feature row afterward (only the row's *active* first
    (num_buckets-1) entries are ever read, but keeping the whole row sorted
    is cheap and keeps the invariant simple to reason about)."""
    mask = rng.random(thresholds.shape) < rate
    noise = rng.normal(0.0, scale, size=thresholds.shape)
    mutated = np.clip(thresholds + noise, 0.0, 1.0)
    result = np.where(mask, mutated, thresholds)
    return np.sort(result, axis=-1)


def mutate_condition_features(
    condition_features: np.ndarray, rate: float, rng: np.random.Generator, eligible_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Each selected condition slot jumps to a uniformly random feature (or
    the wildcard sentinel) -- restricted to `eligible_indices` if given (see
    ELIGIBLE_CONDITION_INDICES_BY_STREET; defaults to every condition
    feature). Feature identity has no natural ordering, unlike a bucket
    index or an action category, so there's no meaningful "nudge" move here
    -- only a full reassignment."""
    mask = rng.random(condition_features.shape) < rate
    pool = np.concatenate([[WILDCARD], eligible_indices if eligible_indices is not None else np.arange(NUM_CONDITION_FEATURES)])
    choices = pool[rng.integers(0, len(pool), size=condition_features.shape)]
    return np.where(mask, choices, condition_features)


def _mutate_ordinal(values: np.ndarray, upper_exclusive: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Shared mutation move for gene kinds where the integer value *is*
    meaningfully ordered (bucket index: weak->strong; action category:
    Fold->Call->Raise->All-In), so local search along that order is useful:
    80% of the time nudge +-1 (clipped), 20% of the time jump to a uniformly
    random value in range -- the same two-move blend genome.py's old
    mutate_weights used for the same reason (occasional bigger jumps so
    search isn't limited to exploring one neighbor at a time)."""
    mask = rng.random(values.shape) < rate
    if not mask.any():
        return values
    step = np.where(rng.random(values.shape) < 0.5, 1, -1)
    nudged = np.clip(values + step, 0, upper_exclusive - 1)
    jumped = rng.integers(0, upper_exclusive, size=values.shape)
    move = rng.random(values.shape)
    new_values = np.where(move < 0.8, nudged, jumped)
    return np.where(mask, new_values, values)


def mutate_condition_buckets(condition_buckets: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    return _mutate_ordinal(condition_buckets, MAX_BUCKETS, rate, rng)


def mutate_rule_actions(rule_actions: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    return _mutate_ordinal(rule_actions, NUM_ACTION_CATEGORIES, rate, rng)


def mutate_rule_mix_actions(rule_mix_actions: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Each selected slot jumps to a uniformly random action category
    (turning this rule into a Mix, or changing its second action) or
    NO_MIX (making/keeping it a single fixed action) -- the same "no
    natural ordering, so only full reassignment" reasoning as
    mutate_condition_features: whether a rule mixes at all isn't an
    incremental move, and once it does, which second category it mixes in
    has no more of a meaningful neighbor than a condition's feature
    identity does."""
    mask = rng.random(rule_mix_actions.shape) < rate
    choices = rng.integers(-1, NUM_ACTION_CATEGORIES, size=rule_mix_actions.shape)  # -1 (NO_MIX) .. N-1
    return np.where(mask, choices, rule_mix_actions)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def row_crossover_mask(num_rows: int, rng: np.random.Generator) -> np.ndarray:
    """One 50/50 coin flip per row, to be reused across every gene array
    that shares that row's meaning (see apply_row_mask)."""
    return rng.random(num_rows) < 0.5


def apply_row_mask(a: np.ndarray, b: np.ndarray, from_a: np.ndarray) -> np.ndarray:
    """Whole-row crossover over the first axis using a *precomputed* mask
    (see row_crossover_mask): each row is inherited whole from one parent or
    the other, never mixed gene-by-gene within a row. Callers reuse the same
    mask across multiple arrays that must stay internally coherent -- e.g.
    num_buckets[i] and thresholds[i] always from the same parent (never a
    bucket count from one parent paired with cut points sized for a
    different bucket count), or a rule's conditions, action, and (for
    preflop) hole_category_mask always from the same parent (never a feature
    index from one parent matched against an unrelated bucket index from the
    other). Same reasoning genome.py's old crossover_weights docstring gives
    for why quantized/categorical genes use uniform discrete inheritance
    instead of blending."""
    return np.where(from_a.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
