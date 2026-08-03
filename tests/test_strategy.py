import numpy as np
import pytest

import strategy
from features import FEATURE_SPECS


class TestConditionFeatures:
    def test_matches_specs_with_no_linked_to_excluding_street(self):
        # street_norm is excluded: which street a rule applies to is a
        # mandatory, structural property (which per-street pool it lives
        # in), not a regular condition -- see strategy.py's module docstring.
        expected = [s.key for s in FEATURE_SPECS if s.linked_to is None and s.key != "street_norm"]
        assert [s.key for s in strategy.CONDITION_FEATURES] == expected

    def test_street_norm_is_not_a_condition_feature(self):
        assert "street_norm" not in [s.key for s in strategy.CONDITION_FEATURES]

    def test_boolean_mask_matches_kind(self):
        for spec, is_bool in zip(strategy.CONDITION_FEATURES, strategy.BOOLEAN_MASK):
            assert is_bool == (spec.kind == "boolean")

    def test_feature_index_round_trips_with_key(self):
        for i, spec in enumerate(strategy.CONDITION_FEATURES):
            assert strategy.feature_index(spec.key) == i

    def test_feature_index_raises_for_unknown_key(self):
        with pytest.raises(KeyError):
            strategy.feature_index("not_a_real_feature")


class TestComputeBucket:
    def test_below_first_threshold_is_bucket_zero(self):
        assert strategy.compute_bucket(0.2, 2, np.array([0.5, 0.9])) == 0

    def test_above_first_threshold_is_bucket_one(self):
        assert strategy.compute_bucket(0.6, 2, np.array([0.5, 0.9])) == 1

    def test_three_buckets_uses_both_thresholds(self):
        thresholds = np.array([0.3, 0.8])
        assert strategy.compute_bucket(0.1, 3, thresholds) == 0
        assert strategy.compute_bucket(0.5, 3, thresholds) == 1
        assert strategy.compute_bucket(0.95, 3, thresholds) == 2

    def test_exact_threshold_value_falls_into_the_higher_bucket(self):
        # side="right": a value exactly on the cut point counts as "at or above."
        assert strategy.compute_bucket(0.5, 2, np.array([0.5, 0.9])) == 1

    def test_unsorted_thresholds_are_sorted_before_use(self):
        # num_buckets=3 uses both entries -- given out of order, they must
        # be sorted first or this would (wrongly) read as "below the first
        # cut, above the second," an impossible bucket.
        assert strategy.compute_bucket(0.6, 3, np.array([0.9, 0.5])) == 1


class TestDescribeBucket:
    def test_boolean_true_bucket_is_the_label(self):
        spec = next(s for s in strategy.CONDITION_FEATURES if s.kind == "boolean")
        assert strategy.describe_bucket(spec, 1, 2, None) == spec.label

    def test_boolean_false_bucket_is_negated(self):
        spec = next(s for s in strategy.CONDITION_FEATURES if s.kind == "boolean")
        assert strategy.describe_bucket(spec, 0, 2, None) == f"Not {spec.label}"

    def test_categorical_two_buckets_span_the_value_table(self):
        spec = strategy.feature_index("hand_category_norm")
        spec = strategy.CONDITION_FEATURES[spec]
        label = strategy.describe_bucket(spec, 0, 2, np.array([0.5, 0.5]))
        assert "High Card" in label

    def test_last_bucket_includes_the_top_value_table_point(self):
        spec = strategy.CONDITION_FEATURES[strategy.feature_index("hand_category_norm")]
        label = strategy.describe_bucket(spec, 1, 2, np.array([0.5, 0.5]))
        assert "Straight Flush" in label

    def test_out_of_range_bucket_index_is_clipped_not_crashed_on(self):
        spec = strategy.CONDITION_FEATURES[strategy.feature_index("hand_category_norm")]
        label = strategy.describe_bucket(spec, 2, 2, np.array([0.5, 0.5]))
        assert label == strategy.describe_bucket(spec, 1, 2, np.array([0.5, 0.5]))


class TestRaiseActionCategories:
    def test_every_raise_action_has_a_pot_fraction(self):
        for action in strategy.RAISE_ACTIONS:
            assert action in strategy.RAISE_POT_FRACTION

    def test_fold_call_allin_are_not_raise_actions(self):
        for action in (strategy.ACTION_FOLD, strategy.ACTION_CALL, strategy.ACTION_ALLIN):
            assert action not in strategy.RAISE_ACTIONS
            assert action not in strategy.RAISE_POT_FRACTION

    def test_pot_fractions_ascend_with_action_index(self):
        fractions = [strategy.RAISE_POT_FRACTION[a] for a in strategy.RAISE_ACTIONS]
        assert fractions == sorted(fractions)

    def test_action_categories_length_matches_num_action_categories(self):
        assert len(strategy.ACTION_CATEGORIES) == strategy.NUM_ACTION_CATEGORIES
        assert strategy.NUM_ACTION_CATEGORIES == 2 + len(strategy.RAISE_ACTIONS) + 1


class TestStreets:
    def test_street_labels_match_num_streets(self):
        assert len(strategy.STREET_LABELS) == strategy.NUM_STREETS

    def test_preflop_is_street_zero(self):
        assert strategy.PREFLOP == 0
        assert strategy.STREET_LABELS[strategy.PREFLOP] == "Preflop"


class TestHoleCategory:
    def test_num_hole_categories_matches_labels(self):
        assert len(strategy.HOLE_CATEGORY_LABELS) == strategy.NUM_HOLE_CATEGORIES

    def test_hole_hand_category_norm_is_still_a_condition_feature(self):
        # Unlike street_norm, hole_hand_category_norm stays available as an
        # ordinary (optional) condition too -- e.g. a turn/river rule caring
        # "what did I start with" -- on top of being mandatory for preflop.
        assert strategy.HOLE_CATEGORY_FEATURE_KEY in [s.key for s in strategy.CONDITION_FEATURES]

    def test_index_round_trips_through_the_norm_value(self):
        for idx in range(strategy.NUM_HOLE_CATEGORIES):
            condition_values = np.zeros(strategy.NUM_CONDITION_FEATURES)
            condition_values[strategy._HOLE_CATEGORY_CONDITION_INDEX] = idx / (strategy.NUM_HOLE_CATEGORIES - 1)
            assert strategy.hole_category_index(condition_values) == idx

    def test_does_not_mutate_input(self):
        condition_values = np.zeros(strategy.NUM_CONDITION_FEATURES)
        condition_values[strategy._HOLE_CATEGORY_CONDITION_INDEX] = 0.5
        before = condition_values.copy()
        strategy.hole_category_index(condition_values)
        assert np.array_equal(condition_values, before)


class TestEligibleConditionIndicesByStreet:
    def test_one_array_per_street(self):
        assert len(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET) == strategy.NUM_STREETS

    def test_preflop_excludes_board_texture_and_postflop_only_features(self):
        from features import group_of
        preflop_eligible = set(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP].tolist())
        for i, spec in enumerate(strategy.CONDITION_FEATURES):
            is_excluded = group_of(spec) == "Board / Flop Characteristics" or spec.key in strategy._POST_FLOP_ONLY_FEATURE_KEYS
            if is_excluded:
                assert i not in preflop_eligible
            else:
                assert i in preflop_eligible

    def test_preflop_excludes_the_specific_keys_named_in_the_request(self):
        excluded_keys = {strategy.CONDITION_FEATURES[i].key for i in strategy._PREFLOP_EXCLUDED_INDICES}
        for key in ("num_overcards_norm", "nuts_flush_draw", "gutshot"):
            assert key in excluded_keys

    def test_hand_category_and_high_card_features_stay_eligible_preflop(self):
        # Well-defined from the hole cards alone -- not board-relative.
        preflop_eligible = set(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP].tolist())
        for key in ("hand_category_norm", "ace_high_no_pair", "king_high_no_pair"):
            assert strategy.feature_index(key) in preflop_eligible

    def test_flop_turn_river_have_no_exclusions(self):
        full = set(range(strategy.NUM_CONDITION_FEATURES))
        for street in (1, 2, 3):
            assert set(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street].tolist()) == full


class TestActionLabels:
    def test_fold_label_reads_check_fold_give_up(self):
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_FOLD] == "Check / Fold (Give up)"
