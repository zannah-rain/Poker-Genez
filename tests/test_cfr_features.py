import numpy as np
import pytest

import cfr_features
from cards import Card
from features import FEATURE_NAMES, Situation, extract_features


def _make_situation(**overrides):
    defaults = dict(
        hole=[Card.from_str("Ah"), Card.from_str("Kh")],
        board=[Card.from_str(t) for t in "2c 5d 9h".split()],
        street=1,
        pot=10.0,
        call_amount=0.0,
        my_stack=190.0,
        effective_stack=190.0,
        position=0,
        num_seats_this_street=3,
        seat_index=0,
        button_idx=0,
        num_seats_total=3,
        num_active=3,
        num_raises_this_street=0,
        num_preflop_raises=1,
        is_aggressor=False,
        starting_stack=200.0,
    )
    defaults.update(overrides)
    return Situation(**defaults)


class TestFeatureIndices:
    def test_maps_keys_to_correct_positions(self):
        keys = ["hand_category_norm", "street_norm", "facing_bet"]
        indices = cfr_features.feature_indices(keys)
        expected = [FEATURE_NAMES.index(k) for k in keys]
        assert list(indices) == expected

    def test_preserves_requested_order_even_if_out_of_order(self):
        keys = ["facing_bet", "hand_category_norm"]
        indices = cfr_features.feature_indices(keys)
        assert list(indices) == [FEATURE_NAMES.index(k) for k in keys]

    def test_unknown_key_raises_key_error_listing_offenders(self):
        with pytest.raises(KeyError) as exc_info:
            cfr_features.feature_indices(["hand_category_norm", "not_a_real_feature"])
        assert "not_a_real_feature" in str(exc_info.value)

    def test_empty_keys_returns_empty_array(self):
        indices = cfr_features.feature_indices([])
        assert indices.shape == (0,)


class TestDefaultFeatureKeys:
    def test_all_default_keys_are_valid(self):
        # Should not raise.
        cfr_features.feature_indices(cfr_features.DEFAULT_FEATURE_KEYS)

    def test_excludes_opponent_tendency_features(self):
        assert not any(k.startswith("opp_") for k in cfr_features.DEFAULT_FEATURE_KEYS)

    def test_nonempty_and_deduplicated(self):
        assert len(cfr_features.DEFAULT_FEATURE_KEYS) > 0
        assert len(set(cfr_features.DEFAULT_FEATURE_KEYS)) == len(cfr_features.DEFAULT_FEATURE_KEYS)


class TestExtractSubset:
    def test_matches_full_extraction_at_selected_positions(self):
        situation = _make_situation()
        keys = ["hand_category_norm", "street_norm", "facing_bet"]
        indices = cfr_features.feature_indices(keys)
        subset = cfr_features.extract_subset(situation, indices)
        full = extract_features(situation)
        expected = full[indices]
        assert np.allclose(subset, expected)

    def test_returns_float32(self):
        situation = _make_situation()
        indices = cfr_features.feature_indices(["hand_category_norm"])
        subset = cfr_features.extract_subset(situation, indices)
        assert subset.dtype == np.float32
