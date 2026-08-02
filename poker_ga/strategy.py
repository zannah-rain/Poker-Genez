"""Feature vocabulary, per-street eligibility, and low-level bucketing math
shared by the object-oriented rule/action decision layer in rules.py -- the
part of the old array-based rule-list engine that's still genuinely
genome-agnostic, reusable machinery rather than gene storage itself.

Everything that used to live here operating on numpy gene *arrays*
(rule matching, mutation/crossover of conditions/actions/rules, the
wildcard-repair pass) has moved into rules.py, where Condition/StandardRule/
GTOSpotRule/Action objects now own that behavior directly as methods --
see rules.py's module docstring for the full picture of what changed and
why (objects are the genes now, not arrays; per-condition bucketing
replaces genome-wide bucketing; rule/condition counts are variable and
evolvable instead of fixed-size).

What's still here:

  Feature vocabulary (CONDITION_FEATURES -- the ~49 generalized concepts a
  rule can reference, not the ~150 one-hot indicator children, and not
  street_norm, which is redundant now that street is handled structurally
  by which per-street pool a rule lives in) and per-street eligibility
  (ELIGIBLE_CONDITION_INDICES_BY_STREET -- preflop excludes board-relative
  features that don't exist yet).

  Bucketing math (compute_bucket/describe_bucket): every non-boolean
  condition feature lives on an evenly-spaced 0-1 scale (features.py
  normalizes everything that way), so "learn a threshold" and "learn to
  bucket a categorical into groups" are the same mechanism -- pick
  num_buckets (2 or 3) cut points on that line. This machinery is reused by
  every Condition (each of which now owns its own num_buckets/thresholds --
  see rules.py) rather than being genome-wide.

  Action category constants/labels (ACTION_*, RAISE_POT_FRACTION, ...) and
  hole-hand-category constants (HOLE_CATEGORY_*, hole_category_index) --
  read-only vocabulary referenced by rules.py's Action/StandardRule classes.
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
# street_norm is excluded too: which street a rule applies to is a
# mandatory, structural property of which per-street pool it lives in, so
# referencing it again as a regular condition would be permanently
# redundant -- every rule in, say, the flop pool only ever sees flop
# situations, making any street_norm condition on it either trivially
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

MIN_BUCKETS = 2
MAX_BUCKETS = 3  # most features settle on 2-3 buckets; nothing hard-caps below this

# ---------------------------------------------------------------------------
# Streets
# ---------------------------------------------------------------------------

NUM_STREETS = 4  # matches Situation.street: 0=preflop, 1=flop, 2=turn, 3=river
STREET_LABELS = ("Preflop", "Flop", "Turn", "River")
PREFLOP = 0

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
# plus a shared genome-wide size gene: a rule picks its size directly (e.g.
# "Raise 75% Pot"), so different rules -- a value raise on the river vs a
# preflop 3-bet -- can each evolve their own size independently, still off a
# small, human-memorizable menu (the same pot-fraction sizing convention
# gto.py's "raise_NN" action tokens already use). Ordered from least to most
# aggressive alongside Fold/Call/All-In so the whole list is one continuum.
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
# fires (see rules.py's SingleAction/MixAction) -- the only source of
# randomized/mixed-strategy behavior in a genome's decisions.
NO_MIX = -1  # sentinel: this action is a single fixed category, not a Mix


# ---------------------------------------------------------------------------
# Hole hand category -- the mandatory preflop rule axis
# ---------------------------------------------------------------------------

HOLE_CATEGORY_FEATURE_KEY = "hole_hand_category_norm"
_HOLE_CATEGORY_CONDITION_INDEX = _FEATURE_INDEX_BY_KEY[HOLE_CATEGORY_FEATURE_KEY]
HOLE_CATEGORY_LABELS = [label for _point, label in CONDITION_FEATURES[_HOLE_CATEGORY_CONDITION_INDEX].value_table]
NUM_HOLE_CATEGORIES = len(HOLE_CATEGORY_LABELS)


def feature_index(key: str) -> int:
    return _FEATURE_INDEX_BY_KEY[key]


def hole_category_index(condition_values: np.ndarray) -> int:
    """Exact (not bucketed, not jittered) hole-hand-category index 0..11,
    read directly out of `condition_values` (the same array decide() slices
    out of extract_features via CONDITION_FEATURES_FULL_INDEX). features.py
    normalizes hole_hand_category_norm as category_index / 11, so this
    round-trips exactly. Preflop rules match on this directly (via
    StandardRule.hole_category_mask) rather than through the usual
    bucket-threshold mechanism -- a real range chart is exact about which
    starting hands are in which line, not fuzzy, and this is preflop's one
    mandatory axis."""
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
#   - A specific handful of "Made Hand Features"/"Draw Features" that compare
#     the hole cards to the board (top/second/third pair, over/under/low
#     pair, board overcards, a flush draw's nut status, a gutshot straight
#     draw) -- structurally these always evaluate to their same "nothing to
#     compare against yet" value preflop (extract_features computes them
#     from `board_ranks`, empty preflop), so referencing them would be a
#     permanently dead condition, same reasoning as the board-texture group.
#     hand_category_norm (Pair vs High Card from the hole cards alone) and
#     ace_high_no_pair/king_high_no_pair (from the hole cards' own high
#     card) are *not* excluded -- both are well-defined, genuinely
#     informative preflop.
_BOARD_TEXTURE_GROUP = "Board / Flop Characteristics"
_POST_FLOP_ONLY_FEATURE_KEYS = frozenset({
    "num_overcards_norm", "top_pair", "second_pair", "third_pair", "overpair", "underpair", "low_pair",
    "nuts_flush_draw", "gutshot",
})
_PREFLOP_EXCLUDED_INDICES = frozenset(
    i for i, s in enumerate(CONDITION_FEATURES)
    if group_of(s) == _BOARD_TEXTURE_GROUP or s.key in _POST_FLOP_ONLY_FEATURE_KEYS
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
# Bucketing
# ---------------------------------------------------------------------------

def compute_bucket(value: float, num_buckets: int, thresholds) -> int:
    """value: a feature's normalized 0-1 reading. thresholds: this
    condition's cut-point sequence -- only the first (num_buckets - 1)
    entries are meaningful; the rest are ignored. Returns a bucket index in
    [0, num_buckets)."""
    cuts = np.sort(np.asarray(thresholds[: num_buckets - 1]))
    return int(np.searchsorted(cuts, value, side="right"))


def describe_bucket(spec: FeatureSpec, bucket_index: int, num_buckets: int, thresholds) -> str:
    """Human-readable label for one bucket of a feature, e.g. "High Card -
    Two Pair" for hand_category_norm's low bucket at num_buckets=3, reusing
    the value_table labels features.py already defines for every
    categorical/continuous feature (including a representative-point table
    for genuinely continuous ones -- see features.py's _BUCKET_POINTS) --
    so a bucket reads the way a human names a chart region, not as a raw
    "0.00-0.33" cutoff. `bucket_index` is clipped to an existing bucket, so
    a stored value the current num_buckets doesn't have never renders as a
    nonsensical/impossible label."""
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
