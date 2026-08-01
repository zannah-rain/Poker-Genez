"""The rule-list decision engine: turns a genome's evolved genes into a small,
ordered "if this situation, then this action" list a human could actually
memorize and execute at a table -- the replacement for the old linear V/L/
theta system (see genome.py's module docstring for that history).

Two kinds of evolvable gene live here:

  Feature bucketing (genome-wide, shared by every rule): every "top-level"
  feature (see TOP_LEVEL_FEATURES -- the ~50 generalized concepts, not the
  ~150 one-hot indicator children) already lives on an evenly-spaced 0-1
  scale (features.py normalizes everything that way). That means "learn a
  threshold" and "learn to bucket a categorical into groups" are the *same*
  mechanism: pick `num_buckets` (2 or 3) cut points on that 0-1 line. This is
  genome-global rather than per-rule, so a concept like "Hand Strength" has
  one shared definition every rule reads from -- "weak/medium/strong, cut
  here" -- rather than every rule re-deriving its own cutoffs. Boolean
  features need no genes at all: they're already exactly 2 buckets (0/1).

  Rules (a fixed-size ordered decision list): each rule is a small
  conjunction of up to CONDITIONS_PER_RULE (feature, required-bucket) checks
  -- or a wildcard, "don't care" -- plus one action category. Checked in
  fixed array order, first full match wins (the same idiom gto.py's
  GTOSpot.action_ranges/SpotMatcher already use); no match at all falls back
  to Fold (matching GTOSpot.default_action's "everything not colored in is a
  fold" convention). A rule's array *position* carries no priority meaning
  beyond that fixed check order -- like a weight vector's index today, a
  slot's content is what matters, not which slot it happens to sit in -- so
  there's no reordering mutation operator.

Multi-street plans ("raise now, fold if raised back") are never represented
explicitly: every decision re-evaluates this same rule list fresh from the
current Situation. "Fold if raised back" simply falls out of a *later*
decision's bigger call_amount/facing_bet routing to a different (likely
Fold) rule -- no state needs to be carried between decisions. Likewise "call
up to X, else fold" is just an ordinary rule condition on the existing
call_amount_norm feature's bucket, not a special parameter.
"""

from __future__ import annotations

import numpy as np

from features import FEATURE_SPECS, FeatureSpec

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------

# Only the "generalized" top-level features (parents of a one-hot family, or
# standalone booleans) are usable as rule conditions -- the ~150 linked
# indicator children are just precomputed alternate views of these same
# concepts (see features.py's module docstring), so exposing them too would
# only let two conditions restate the same fact under different keys.
TOP_LEVEL_FEATURES: list[FeatureSpec] = [s for s in FEATURE_SPECS if s.linked_to is None]
NUM_TOP_LEVEL_FEATURES = len(TOP_LEVEL_FEATURES)

# Index into the *full* features.py vector (as extract_features returns it)
# for each top-level feature, precomputed once so decide() doesn't need to
# rebuild this mapping (or do a linear FEATURE_NAMES.index lookup) every call.
TOP_LEVEL_FULL_INDEX = np.array(
    [i for i, s in enumerate(FEATURE_SPECS) if s.linked_to is None], dtype=np.int64
)

_FEATURE_INDEX_BY_KEY: dict[str, int] = {s.key: i for i, s in enumerate(TOP_LEVEL_FEATURES)}

BOOLEAN_MASK = np.array([s.kind == "boolean" for s in TOP_LEVEL_FEATURES], dtype=bool)
# Indices (into TOP_LEVEL_FEATURES) of features that need bucketing genes --
# every non-boolean feature. Booleans are already exactly 0/1, no genes needed.
BUCKETABLE_INDICES = np.flatnonzero(~BOOLEAN_MASK)
NUM_BUCKETABLE = len(BUCKETABLE_INDICES)
# Inverse of the above: top-level feature index -> row in the (NUM_BUCKETABLE,
# ...) gene arrays, or -1 for a boolean feature (which has no such row).
_BUCKET_GENE_ROW = np.full(NUM_TOP_LEVEL_FEATURES, -1, dtype=np.int64)
_BUCKET_GENE_ROW[BUCKETABLE_INDICES] = np.arange(NUM_BUCKETABLE)

MIN_BUCKETS = 2
MAX_BUCKETS = 3  # most features settle on 2-3 buckets; nothing hard-caps below this

WILDCARD = -1  # condition_features sentinel: "don't care" for this condition slot

NUM_RULES = 24
CONDITIONS_PER_RULE = 3

# Fraction of a rule's condition slots that start non-wildcard for a freshly
# random genome -- kept low (mirroring gto.py's GTO_INIT_PROB philosophy:
# "earn complexity, don't start with it") while still giving crossover/
# mutation real material to work with immediately.
CONDITION_ACTIVE_INIT_PROB = 0.3

# Old GAConfig.mutation_scale (~0.3 by default) was calibrated for the V/L
# system's 0-100 range. Thresholds and bucket_noise_std live on features.py's
# native 0-1 scale, so mutating them with the same raw scale would jitter by
# up to ~30% of the entire range per mutation event -- this factor rescales
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
    "Fold", "Call",
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


def feature_index(key: str) -> int:
    return _FEATURE_INDEX_BY_KEY[key]


def bucket_gene_row(top_level_index: int) -> int:
    """Row into the (NUM_BUCKETABLE, ...) num_buckets/thresholds gene arrays
    for this top-level feature, or -1 if it's boolean (no bucketing genes)."""
    return int(_BUCKET_GENE_ROW[top_level_index])


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


def compute_all_buckets(
    top_level_values: np.ndarray,
    num_buckets: np.ndarray,
    thresholds: np.ndarray,
    noise_std: float,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """top_level_values: (NUM_TOP_LEVEL_FEATURES,) normalized 0-1 readings,
    already sliced out of extract_features' full vector via
    TOP_LEVEL_FULL_INDEX. num_buckets/thresholds: this genome's bucketing
    genes (see Genome). Returns one bucket index per top-level feature.

    `noise_std` jitters non-boolean values before bucketing -- near a
    threshold this occasionally flips which bucket a hand falls in, giving
    cheap, natural mixed-strategy behavior at decision boundaries (booleans
    are hard 0/1 facts, e.g. "is this a pair," so they're never jittered)."""
    values = top_level_values
    if noise_std > 0 and rng is not None:
        values = values.copy()
        jitter = rng.normal(0.0, noise_std, size=NUM_BUCKETABLE)
        values[BUCKETABLE_INDICES] = np.clip(values[BUCKETABLE_INDICES] + jitter, 0.0, 1.0)

    buckets = np.empty(NUM_TOP_LEVEL_FEATURES, dtype=np.int64)
    buckets[BOOLEAN_MASK] = (values[BOOLEAN_MASK] > 0.5).astype(np.int64)
    for row, idx in enumerate(BUCKETABLE_INDICES):
        buckets[idx] = compute_bucket(float(values[idx]), int(num_buckets[row]), thresholds[row])
    return buckets


def describe_bucket(spec: FeatureSpec, bucket_index: int, num_buckets: int, thresholds: np.ndarray) -> str:
    """Human-readable label for one bucket of a feature, e.g. "High Card -
    Two Pair" for hand_category_norm's low bucket at num_buckets=3, reusing
    the value_table labels features.py already defines for every
    categorical/continuous feature (including a representative-point table
    for genuinely continuous ones -- see features.py's _BUCKET_POINTS) --
    so a bucket reads the way a human names a chart region, not as a raw
    "0.00-0.33" cutoff."""
    if spec.kind == "boolean":
        return spec.label if bucket_index == 1 else f"Not {spec.label}"

    if bucket_index >= num_buckets:
        # A condition can reference a bucket index this feature doesn't
        # actually have (condition_buckets' gene range is fixed at
        # MAX_BUCKETS regardless of a given feature's own, possibly smaller,
        # num_buckets) -- match_rule already treats that as "never matches"
        # harmlessly; the report just needs to say so instead of indexing
        # past the end of this feature's actual cut points.
        return f"{spec.label} bucket {bucket_index} (impossible -- only {num_buckets} groups exist, never matches)"

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

def match_rule(buckets: np.ndarray, condition_features: np.ndarray, condition_buckets: np.ndarray) -> bool:
    """condition_features/condition_buckets: one rule's (CONDITIONS_PER_RULE,)
    gene rows. A rule matches when every non-wildcard condition's required
    bucket equals the current situation's bucket for that feature. A
    condition whose required bucket can never actually occur (e.g. it asks
    for bucket 2 on a feature this genome only bucketed into 2 groups) simply
    never matches -- no clamping needed, evolution finds and fixes these
    dead conditions the same way it finds and fixes unhelpful weights today."""
    for cf, cb in zip(condition_features, condition_buckets):
        if cf == WILDCARD:
            continue
        if buckets[cf] != cb:
            return False
    return True


def first_matching_rule(
    buckets: np.ndarray, condition_features: np.ndarray, condition_buckets: np.ndarray, rule_actions: np.ndarray
) -> int | None:
    """Returns the action index of the first (in fixed array order) matching
    rule, or None if no rule matches (caller should default to Fold)."""
    for r in range(NUM_RULES):
        if match_rule(buckets, condition_features[r], condition_buckets[r]):
            return int(rule_actions[r])
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


def mutate_condition_features(condition_features: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Each selected condition slot jumps to a uniformly random feature (or
    the wildcard sentinel). Feature identity has no natural ordering, unlike
    a bucket index or an action category, so there's no meaningful "nudge"
    move here -- only a full reassignment."""
    mask = rng.random(condition_features.shape) < rate
    choices = rng.integers(-1, NUM_TOP_LEVEL_FEATURES, size=condition_features.shape)  # -1 (wildcard) .. N-1
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
    different bucket count), or a rule's conditions and its action always
    from the same parent (never a feature index from one parent matched
    against an unrelated bucket index from the other). Same reasoning
    genome.py's old crossover_weights docstring gives for why quantized/
    categorical genes use uniform discrete inheritance instead of blending."""
    return np.where(from_a.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
