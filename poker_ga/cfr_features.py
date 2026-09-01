"""Maps a configurable subset of features.py's feature vocabulary to the
positions Single Deep CFR reads out of extract_features()'s output vector.

A Deep CFR advantage network can be pointed at *any* of features.py's keys
-- there's no rule-bucketing machinery here, just a plain feature vector fed
straight into a neural net. The default subset (see DEFAULT_FEATURE_KEYS
below) takes advantage of that and reads the full vocabulary.
"""

from __future__ import annotations

import numpy as np

from features import FEATURE_NAMES, FEATURE_SPECS, MASKED, extract_features

_INDEX_BY_KEY: dict[str, int] = {key: i for i, key in enumerate(FEATURE_NAMES)}
_SPEC_BY_KEY = {spec.key: spec for spec in FEATURE_SPECS}

# bucket_label/bucket_categories/bucket_labels' shared label for a
# `maskable` feature's features.MASKED reading -- e.g. a draw-shape feature
# on the river, once no next card is left to complete or miss it.
MASKED_LABEL = "N/A"

# Opponent-tendency features (opp_vpip_norm, ...) read session history
# accumulated across many hands (see opponent_model.py) -- a single-hand CFR
# traversal has no session to draw that from, so these would always read as
# the neutral 0.5 default. Excluded from the default subset (still
# selectable explicitly via --feature-keys, just not useful yet).
_OPPONENT_TENDENCY_PREFIX = "opp_"

# Sane starting default: every feature.py key, minus the opponent-tendency
# ones (see above). Fully overridable to any subset of features.FEATURE_NAMES.
DEFAULT_FEATURE_KEYS: tuple[str, ...] = tuple(
    key for key in FEATURE_NAMES
    if not key.startswith(_OPPONENT_TENDENCY_PREFIX)
)


def feature_indices(keys: list[str] | tuple[str, ...]) -> np.ndarray:
    """Positions into extract_features()'s output vector for `keys`, in the
    order given. Raises KeyError (listing every offending key at once) if
    any key isn't in features.FEATURE_NAMES."""
    unknown = [k for k in keys if k not in _INDEX_BY_KEY]
    if unknown:
        raise KeyError(f"Unknown feature key(s): {', '.join(unknown)}")
    return np.array([_INDEX_BY_KEY[k] for k in keys], dtype=np.int64)


def extract_subset(situation, indices: np.ndarray) -> np.ndarray:
    """The configured feature subset for one decision, as float32 (the
    precision torch tensors want) -- indices as returned by feature_indices."""
    return extract_features(situation)[indices].astype(np.float32)


def feature_label(key: str) -> str:
    """Short human-readable name for `key` (features.FeatureSpec.label),
    for display in place of the raw internal key."""
    return _SPEC_BY_KEY[key].label


def feature_description(key: str) -> str:
    """Precise definition of how `key` is computed
    (features.FeatureSpec.description) -- for a hover tooltip."""
    return _SPEC_BY_KEY[key].description


def is_maskable(key: str) -> bool:
    """Whether `key` ever reads as features.MASKED in some situations (see
    features.FeatureSpec.maskable) -- e.g. a hole-card fact masked past
    preflop, or a draw-shape read masked at the river specifically. Exactly
    which situations vary per feature; see feature_description(key) for the
    precise condition."""
    return _SPEC_BY_KEY[key].maskable


def bucket_label(key: str, value: float) -> str:
    """Human-readable label for one raw normalized feature value -- for a
    `maskable` feature's features.MASKED reading, MASKED_LABEL; otherwise,
    for a boolean feature, `spec.label`/f"Not {spec.label}"; otherwise
    the label of the nearest point in the feature's own
    features.FeatureSpec.value_table (e.g. 0.33 -> "Flop" for street_norm)
    -- except for a `zero_bucket_is_exact` feature (e.g. call_amount_norm),
    where the 0.0 point only ever matches an exact 0.0 reading; every other
    value matches among the *remaining* points instead, so a small nonzero
    reading is never absorbed into a bucket that's meant to mean "none at
    all" purely for being numerically closest to it. Reuses features.py's
    existing human-facing names for each value rather than inventing a
    second labeling scheme -- used by cfr_explorer.py to turn raw
    reservoir columns into filter/split/table values a person can read."""
    spec = _SPEC_BY_KEY[key]
    if spec.maskable and value == MASKED:
        return MASKED_LABEL
    if spec.kind == "boolean":
        return spec.label if value >= 0.5 else f"Not {spec.label}"
    points = np.array([p for p, _ in spec.value_table])
    if spec.zero_bucket_is_exact and value != 0.0:
        candidate_idx = np.flatnonzero(points != 0.0)
        nearest = candidate_idx[np.argmin(np.abs(points[candidate_idx] - value))]
    else:
        nearest = int(np.argmin(np.abs(points - value)))
    return spec.value_table[nearest][1]


def bucket_categories(key: str) -> list[str]:
    """Every label bucket_label/bucket_labels could return for this
    feature, in the feature's own natural value order (e.g. "Preflop" <
    "Flop" < "Turn" < "River" for street_norm, not alphabetical) -- used to
    order a feature's buckets consistently for display/grouping rather than
    alphabetically. MASKED_LABEL is appended last for a `maskable` feature,
    since it isn't a real point on that order -- it's "this reading doesn't
    apply here" instead."""
    spec = _SPEC_BY_KEY[key]
    if spec.kind == "boolean":
        categories = [f"Not {spec.label}", spec.label]
    else:
        categories = [label for _, label in spec.value_table]
    if spec.maskable:
        categories = categories + [MASKED_LABEL]
    return categories


def display_feature_keys(feature_keys: list[str] | tuple[str, ...]) -> list[str]:
    """`feature_keys` with every linked child dropped in favor of its parent
    -- e.g. has_pair/has_two_pair/... collapse down to just
    hand_category_norm -- since a child is just an alternate view of the
    same concept its parent already represents (see features.py's module
    docstring). A child is kept standalone if its parent isn't itself in
    `feature_keys` (nothing to fold into). Used by cfr_explorer.py so the
    sidebar only ever offers one control per underlying concept."""
    present = set(feature_keys)
    return [key for key in feature_keys if not (_SPEC_BY_KEY[key].linked_to in present)]


def fold_child_contributions(contributions: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """`contributions` (feature_key, value) pairs -- as cfr_networks.
    mean_shap_contributions_for_samples returns -- with every linked child's value
    summed into its parent's instead of listed separately (mirroring
    display_feature_keys' choice of which key represents a family), then
    re-sorted most-to-least. A child whose parent isn't present in
    `contributions` keeps its own entry, since there's nothing to fold it
    into."""
    present = {key for key, _ in contributions}
    totals: dict[str, float] = {}
    for key, value in contributions:
        parent = _SPEC_BY_KEY[key].linked_to
        target = parent if parent in present else key
        totals[target] = totals.get(target, 0.0) + value
    return sorted(totals.items(), key=lambda kv: -kv[1])


def bucket_labels(key: str, values: np.ndarray) -> np.ndarray:
    """Vectorized bucket_label over a whole array of one feature's raw
    values -- the same masked/nearest-point/boolean/zero_bucket_is_exact
    logic (see bucket_label's own docstring), without a Python loop per row
    (cfr_explorer.py calls this once per feature for the whole loaded
    reservoir)."""
    spec = _SPEC_BY_KEY[key]
    if spec.kind == "boolean":
        labels = np.where(values >= 0.5, spec.label, f"Not {spec.label}")
    else:
        points = np.array([p for p, _ in spec.value_table])
        table_labels = np.array([label for _, label in spec.value_table])
        distances = np.abs(values[:, None] - points[None, :])
        if spec.zero_bucket_is_exact:
            zero_idx = np.flatnonzero(points == 0.0)
            # Disqualify the exact-zero point as a nearest-match candidate
            # for every row that isn't itself exactly 0.0 (an infinite
            # distance can never win an argmin) -- a 0.0-reading row still
            # matches it exactly (distance 0), so it's untouched there.
            distances[np.ix_(values != 0.0, zero_idx)] = np.inf
        nearest = np.argmin(distances, axis=1)
        labels = table_labels[nearest]
    if spec.maskable:
        labels = np.where(values == MASKED, MASKED_LABEL, labels)
    return labels


def observed_level_count(key: str, values: np.ndarray) -> int:
    """How many distinct real (non-MASKED) levels `key` actually takes on
    across `values` -- the size of the implementable lookup table this
    feature alone would need. Deliberately what's actually OBSERVED, not
    len(bucket_categories(key))'s full theoretical count: an unobserved
    value (or, for a `maskable` feature, never being masked at all in this
    sample) contributes nothing anyone would actually have to memorize --
    the same principle cfr_explorer.py's own Split By cell-count math uses
    (see its _observed_categories/_split_by_cell_count)."""
    return len(set(bucket_labels(key, values)) - {MASKED_LABEL})


def unmasked_validity(feature_keys: list[str] | tuple[str, ...], values: np.ndarray) -> np.ndarray:
    """(n_rows, len(feature_keys)) boolean array -- True wherever `values`
    (raw, un-normalized feature readings, one column per feature_keys entry,
    in that order) holds a real reading for that (row, feature); False only
    for a `maskable` feature's features.MASKED rows (e.g. a draw-shape
    feature on the river). Always True for a non-maskable feature, at every
    row. Used by cfr_networks.mean_shap_contributions_for_samples so a
    maskable feature's masked rows -- which carry no information for it by
    construction -- don't get counted as real observations of its
    importance."""
    valid = np.ones_like(values, dtype=bool)
    for i, key in enumerate(feature_keys):
        if _SPEC_BY_KEY[key].maskable:
            valid[:, i] = values[:, i] != MASKED
    return valid
