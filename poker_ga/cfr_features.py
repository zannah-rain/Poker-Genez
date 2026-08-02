"""Maps a configurable subset of features.py's feature vocabulary to the
positions Single Deep CFR reads out of extract_features()'s output vector.

Unlike strategy.py's CONDITION_FEATURES (the ~49 "generalized" features the
GA's rule conditions may reference), a Deep CFR advantage network can be
pointed at *any* of features.py's ~200 keys, including the one-hot indicator
children -- there's no rule-bucketing machinery here that would make the
children redundant with their parent, just a plain feature vector fed
straight into a neural net.
"""

from __future__ import annotations

import numpy as np

import strategy
from features import FEATURE_NAMES, FEATURE_SPECS, extract_features

_INDEX_BY_KEY: dict[str, int] = {key: i for i, key in enumerate(FEATURE_NAMES)}
_SPEC_BY_KEY = {spec.key: spec for spec in FEATURE_SPECS}

# Opponent-tendency features (opp_vpip_norm, ...) read session history
# accumulated across many hands (see opponent_model.py) -- a single-hand CFR
# traversal has no session to draw that from, so these would always read as
# the neutral 0.5 default. Excluded from the default subset (still
# selectable explicitly via --feature-keys, just not useful yet).
_OPPONENT_TENDENCY_PREFIX = "opp_"

# Sane starting default: the same ~49 generalized features strategy.py
# already curates for the GA's rule conditions, minus the opponent-tendency
# ones (see above). Fully overridable to any subset of features.FEATURE_NAMES.
DEFAULT_FEATURE_KEYS: tuple[str, ...] = tuple(
    spec.key for spec in strategy.CONDITION_FEATURES
    if not spec.key.startswith(_OPPONENT_TENDENCY_PREFIX)
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


def bucket_label(key: str, value: float) -> str:
    """Human-readable label for one raw normalized feature value -- for a
    boolean feature, `spec.label`/f"Not {spec.label}" (mirrors
    strategy.describe_bucket's own boolean-labeling convention); otherwise
    the label of the nearest point in the feature's own
    features.FeatureSpec.value_table (e.g. 0.33 -> "Flop" for street_norm).
    Reuses features.py's existing human-facing names for each value rather
    than inventing a second labeling scheme -- used by cfr_explorer.py to
    turn raw reservoir columns into filter/split/table values a person can
    read."""
    spec = _SPEC_BY_KEY[key]
    if spec.kind == "boolean":
        return spec.label if value >= 0.5 else f"Not {spec.label}"
    points = np.array([p for p, _ in spec.value_table])
    nearest = int(np.argmin(np.abs(points - value)))
    return spec.value_table[nearest][1]


def bucket_categories(key: str) -> list[str]:
    """Every label bucket_label/bucket_labels could return for this
    feature, in the feature's own natural value order (e.g. "Preflop" <
    "Flop" < "Turn" < "River" for street_norm, not alphabetical) -- used to
    order a feature's buckets consistently for display/grouping rather than
    alphabetically."""
    spec = _SPEC_BY_KEY[key]
    if spec.kind == "boolean":
        return [f"Not {spec.label}", spec.label]
    return [label for _, label in spec.value_table]


def bucket_labels(key: str, values: np.ndarray) -> np.ndarray:
    """Vectorized bucket_label over a whole array of one feature's raw
    values -- the same nearest-point/boolean logic, without a Python loop
    per row (cfr_explorer.py calls this once per feature for the whole
    loaded reservoir)."""
    spec = _SPEC_BY_KEY[key]
    if spec.kind == "boolean":
        return np.where(values >= 0.5, spec.label, f"Not {spec.label}")
    points = np.array([p for p, _ in spec.value_table])
    labels = np.array([label for _, label in spec.value_table])
    nearest = np.argmin(np.abs(values[:, None] - points[None, :]), axis=1)
    return labels[nearest]
