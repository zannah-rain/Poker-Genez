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
from features import FEATURE_NAMES, extract_features

_INDEX_BY_KEY: dict[str, int] = {key: i for i, key in enumerate(FEATURE_NAMES)}

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
