import numpy as np
import pytest

import cfr_features
import features
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
        num_raises_previous_street=0,
        num_raises_preflop=1,
        num_raises_flop=0,
        num_raises_turn=0,
        is_aggressor_previous_street=False,
        is_aggressor_preflop=False,
        is_aggressor_flop=False,
        is_aggressor_turn=False,
        starting_stack=200.0,
    )
    defaults.update(overrides)
    return Situation(**defaults)


class TestFeatureIndices:
    def test_maps_keys_to_correct_positions(self):
        keys = ["hand_category_norm", "street_norm", "hole_suited"]
        indices = cfr_features.feature_indices(keys)
        expected = [FEATURE_NAMES.index(k) for k in keys]
        assert list(indices) == expected

    def test_preserves_requested_order_even_if_out_of_order(self):
        keys = ["hole_suited", "hand_category_norm"]
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


class TestBucketLabel:
    def test_boolean_feature_uses_spec_label_and_negation(self):
        assert cfr_features.bucket_label("hole_suited", 1.0) == "Suited Hole Cards"
        assert cfr_features.bucket_label("hole_suited", 0.0) == "Not Suited Hole Cards"

    def test_categorical_feature_matches_exact_value_table_points(self):
        assert cfr_features.bucket_label("street_norm", 0.0) == "Preflop"
        assert cfr_features.bucket_label("street_norm", 1 / 3) == "Flop"
        assert cfr_features.bucket_label("street_norm", 2 / 3) == "Turn"
        assert cfr_features.bucket_label("street_norm", 1.0) == "River"

    def test_categorical_feature_snaps_to_nearest_point(self):
        # 0.3 is closer to 1/3 (Flop) than to 0.0 (Preflop).
        assert cfr_features.bucket_label("street_norm", 0.3) == "Flop"

    def test_27_way_hand_category(self):
        assert cfr_features.bucket_label("hand_category_norm", 0.0) == "High Card"
        assert cfr_features.bucket_label("hand_category_norm", 13 / 26) == "Two Pair"
        assert cfr_features.bucket_label("hand_category_norm", 1.0) == "Straight Flush"

    def test_maskable_categorical_feature_masked_reading(self):
        # draw_norm is masked (features.MASKED) on the river -- should read
        # as MASKED_LABEL rather than snapping to whichever real
        # value_table point happens to sit nearest -1.0.
        assert cfr_features.bucket_label("draw_norm", features.MASKED) == cfr_features.MASKED_LABEL

    def test_maskable_boolean_feature_masked_reading(self, monkeypatch):
        # No real feature is currently both kind="boolean" and maskable
        # (nuts_flush_draw used to be, before it was folded into draw_norm
        # -- see features.py's own history) -- inject a throwaway synthetic
        # spec so this still exercises that combination's own code path in
        # bucket_label directly, rather than depending on one happening to
        # exist in the real catalog. MASKED must win out over the normal
        # >=0.5 true/false split, not read as "Not Test Bool".
        spec = features.FeatureSpec("_test_maskable_bool", "Test Bool", "", kind="boolean", maskable=True)
        monkeypatch.setitem(cfr_features._SPEC_BY_KEY, spec.key, spec)
        assert cfr_features.bucket_label(spec.key, features.MASKED) == cfr_features.MASKED_LABEL

    def test_non_maskable_feature_ignores_negative_values(self):
        # street_norm was never declared maskable -- a negative reading
        # (which extract_features never actually produces for it) should
        # still nearest-point-match rather than being treated as masked.
        assert cfr_features.bucket_label("street_norm", -1.0) == "Preflop"

    def test_zero_bucket_is_exact_matches_a_true_zero_reading(self):
        assert cfr_features.bucket_label("call_amount_norm", 0.0) == "No bet to call"

    def test_zero_bucket_is_exact_excludes_a_small_nonzero_reading(self):
        # Regression test: 0.05 is numerically closer to the 0.0 point
        # ("No bet to call") than to 0.2 ("Call is 1/4 pot"), but a call of
        # 1/20 pot is still a real bet -- plain nearest-point matching used
        # to lump it in with "No bet to call" anyway. zero_bucket_is_exact
        # (see features.FeatureSpec) excludes that point for any nonzero
        # reading, so it should land on the nearest *real* bucket instead.
        assert cfr_features.bucket_label("call_amount_norm", 0.05) == "Call is 1/4 pot"

    def test_zero_bucket_is_exact_still_nearest_point_matches_among_the_rest(self):
        assert cfr_features.bucket_label("call_amount_norm", 0.2) == "Call is 1/4 pot"
        assert cfr_features.bucket_label("call_amount_norm", 0.19) == "Call is 1/4 pot"
        assert cfr_features.bucket_label("call_amount_norm", 1.0) == "Overbet (more than a full pot, clipped)"


class TestBucketLabels:
    def test_vectorized_matches_scalar_per_element(self):
        values = np.array([0.0, 1 / 3, 2 / 3, 1.0, 0.3])
        labels = cfr_features.bucket_labels("street_norm", values)
        expected = [cfr_features.bucket_label("street_norm", v) for v in values]
        assert list(labels) == expected

    def test_boolean_vectorized(self):
        values = np.array([0.0, 1.0, 1.0, 0.0])
        labels = cfr_features.bucket_labels("hole_suited", values)
        assert list(labels) == ["Not Suited Hole Cards", "Suited Hole Cards", "Suited Hole Cards", "Not Suited Hole Cards"]

    def test_maskable_feature_vectorized(self):
        values = np.array([0 / 7, 3 / 7, 7 / 7, features.MASKED])
        labels = cfr_features.bucket_labels("draw_norm", values)
        assert list(labels) == [
            "No Draw", "Gutshot (4 Outs)", "Combo Draw (Flush + Straight)", cfr_features.MASKED_LABEL,
        ]

    def test_zero_bucket_is_exact_vectorized_matches_scalar_per_element(self):
        values = np.array([0.0, 0.05, 0.2, 0.19, 1.0])
        labels = cfr_features.bucket_labels("call_amount_norm", values)
        expected = [cfr_features.bucket_label("call_amount_norm", v) for v in values]
        assert list(labels) == expected
        assert list(labels) == [
            "No bet to call", "Call is 1/4 pot", "Call is 1/4 pot", "Call is 1/4 pot",
            "Overbet (more than a full pot, clipped)",
        ]


class TestBucketCategories:
    def test_categorical_order_matches_value_table_not_alphabetical(self):
        assert cfr_features.bucket_categories("street_norm") == ["Preflop", "Flop", "Turn", "River"]

    def test_boolean_order_is_false_then_true(self):
        assert cfr_features.bucket_categories("hole_suited") == ["Not Suited Hole Cards", "Suited Hole Cards"]

    def test_every_bucket_label_output_is_a_valid_category(self):
        values = np.array([0.0, 1 / 3, 2 / 3, 1.0, 0.1, 0.9])
        categories = set(cfr_features.bucket_categories("street_norm"))
        for label in cfr_features.bucket_labels("street_norm", values):
            assert label in categories

    def test_maskable_feature_appends_masked_label_last(self):
        assert cfr_features.bucket_categories("draw_norm") == [
            "No Draw", "Backdoor Straight Draw (BDSD, Flop Only)", "Backdoor Flush Draw (BDFD, Flop Only)",
            "Gutshot (4 Outs)", "Open Ended Straight Draw (OESD, 8 Outs)", "Flush Draw (9 Outs)",
            "Nuts Flush Draw (9 Outs)", "Combo Draw (Flush + Straight)", cfr_features.MASKED_LABEL,
        ]

    def test_maskable_boolean_feature_appends_masked_label_last(self, monkeypatch):
        # See test_maskable_boolean_feature_masked_reading above for why
        # this needs an injected synthetic spec rather than a real feature.
        spec = features.FeatureSpec("_test_maskable_bool", "Test Bool", "", kind="boolean", maskable=True)
        monkeypatch.setitem(cfr_features._SPEC_BY_KEY, spec.key, spec)
        assert cfr_features.bucket_categories(spec.key) == [
            "Not Test Bool", "Test Bool", cfr_features.MASKED_LABEL,
        ]

    def test_every_bucket_label_output_is_a_valid_category_including_masked(self):
        values = np.array([0 / 7, 3 / 7, 7 / 7, features.MASKED])
        categories = set(cfr_features.bucket_categories("draw_norm"))
        for label in cfr_features.bucket_labels("draw_norm", values):
            assert label in categories


class TestUnmaskedValidity:
    def test_non_maskable_feature_is_always_valid(self):
        keys = ("street_norm",)
        values = np.array([[0.0], [1 / 3], [features.MASKED]])  # last row never actually happens, but isn't special-cased
        assert cfr_features.unmasked_validity(keys, values).tolist() == [[True], [True], [True]]

    def test_maskable_feature_invalid_only_at_masked_rows(self):
        keys = ("draw_norm",)
        values = np.array([[0.0], [0.5], [features.MASKED], [1.0]])
        assert cfr_features.unmasked_validity(keys, values).tolist() == [[True], [True], [False], [True]]

    def test_mixed_maskable_and_non_maskable_columns(self):
        keys = ("draw_norm", "street_norm", "hole_hand_grid_x_norm")
        values = np.array([
            [0.5, 1.0, 1.0],
            [features.MASKED, 1.0, features.MASKED],
        ])
        assert cfr_features.unmasked_validity(keys, values).tolist() == [
            [True, True, True],
            [False, True, False],
        ]


class TestDisplayFeatureKeys:
    # hole_hand_grid_y_norm is the one remaining `linked_to` relationship in
    # features.py (the 2D Exact Hole Hand grid's second axis, linked to
    # hole_hand_grid_x_norm) -- the old one-hot indicator children (e.g.
    # has_pair, linked to hand_category_norm) were removed as redundant for
    # a NN, which can read an ordinal/categorical value directly.
    def test_drops_child_when_parent_present(self):
        keys = ["hole_hand_grid_x_norm", "hole_hand_grid_y_norm", "street_norm"]
        assert cfr_features.display_feature_keys(keys) == ["hole_hand_grid_x_norm", "street_norm"]

    def test_keeps_child_when_parent_absent(self):
        keys = ["hole_hand_grid_y_norm", "street_norm"]
        assert cfr_features.display_feature_keys(keys) == ["hole_hand_grid_y_norm", "street_norm"]

    def test_keeps_standalone_features_untouched(self):
        keys = ["street_norm", "hole_suited"]
        assert cfr_features.display_feature_keys(keys) == keys


class TestFoldChildContributions:
    def test_child_contribution_folds_into_present_parent(self):
        contributions = [("hole_hand_grid_x_norm", 0.5), ("hole_hand_grid_y_norm", 0.2), ("street_norm", 0.1)]
        folded = dict(cfr_features.fold_child_contributions(contributions))
        assert folded == {"hole_hand_grid_x_norm": pytest.approx(0.7), "street_norm": pytest.approx(0.1)}

    def test_child_kept_standalone_when_parent_absent(self):
        contributions = [("hole_hand_grid_y_norm", 0.2), ("street_norm", 0.1)]
        folded = dict(cfr_features.fold_child_contributions(contributions))
        assert folded == {"hole_hand_grid_y_norm": pytest.approx(0.2), "street_norm": pytest.approx(0.1)}

    def test_sorted_most_to_least_after_folding(self):
        contributions = [("street_norm", 0.3), ("hole_hand_grid_x_norm", 0.1), ("hole_hand_grid_y_norm", 0.5)]
        folded = cfr_features.fold_child_contributions(contributions)
        assert [key for key, _ in folded] == ["hole_hand_grid_x_norm", "street_norm"]


class TestExtractSubset:
    def test_matches_full_extraction_at_selected_positions(self):
        situation = _make_situation()
        keys = ["hand_category_norm", "street_norm", "hole_suited"]
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
