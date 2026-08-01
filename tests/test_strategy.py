import numpy as np
import pytest

import strategy
from features import FEATURE_SPECS


class TestTopLevelFeatures:
    def test_matches_specs_with_no_linked_to(self):
        expected = [s.key for s in FEATURE_SPECS if s.linked_to is None]
        assert [s.key for s in strategy.TOP_LEVEL_FEATURES] == expected

    def test_boolean_mask_matches_kind(self):
        for spec, is_bool in zip(strategy.TOP_LEVEL_FEATURES, strategy.BOOLEAN_MASK):
            assert is_bool == (spec.kind == "boolean")

    def test_bucketable_indices_exclude_booleans(self):
        for idx in strategy.BUCKETABLE_INDICES:
            assert strategy.TOP_LEVEL_FEATURES[idx].kind != "boolean"

    def test_bucket_gene_row_is_valid_for_bucketable_and_negative_for_boolean(self):
        for idx, spec in enumerate(strategy.TOP_LEVEL_FEATURES):
            row = strategy.bucket_gene_row(idx)
            if spec.kind == "boolean":
                assert row == -1
            else:
                assert 0 <= row < strategy.NUM_BUCKETABLE

    def test_feature_index_round_trips_with_key(self):
        for i, spec in enumerate(strategy.TOP_LEVEL_FEATURES):
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


class TestComputeAllBuckets:
    def test_boolean_features_bucket_at_0_5(self):
        values = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES)
        bool_idx = np.flatnonzero(strategy.BOOLEAN_MASK)[0]
        values[bool_idx] = 1.0
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.5)
        buckets = strategy.compute_all_buckets(values, num_buckets, thresholds, 0.0, None)
        assert buckets[bool_idx] == 1
        other_bool_idx = np.flatnonzero(strategy.BOOLEAN_MASK)[1]
        assert buckets[other_bool_idx] == 0

    def test_bucketable_feature_uses_its_own_thresholds(self):
        values = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES)
        target = strategy.BUCKETABLE_INDICES[0]
        values[target] = 0.7
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.5)
        buckets = strategy.compute_all_buckets(values, num_buckets, thresholds, 0.0, None)
        assert buckets[target] == 1

    def test_zero_noise_is_deterministic(self):
        values = np.random.default_rng(0).random(strategy.NUM_TOP_LEVEL_FEATURES)
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 3)
        thresholds = np.sort(np.random.default_rng(1).random((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1)), axis=-1)
        rng = np.random.default_rng(2)
        a = strategy.compute_all_buckets(values, num_buckets, thresholds, 0.0, rng)
        b = strategy.compute_all_buckets(values, num_buckets, thresholds, 0.0, rng)
        assert np.array_equal(a, b)

    def test_does_not_mutate_input_values(self):
        values = np.full(strategy.NUM_TOP_LEVEL_FEATURES, 0.5)
        original = values.copy()
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.3)
        rng = np.random.default_rng(3)
        strategy.compute_all_buckets(values, num_buckets, thresholds, 0.2, rng)
        assert np.array_equal(values, original)


class TestDescribeBucket:
    def test_boolean_true_bucket_is_the_label(self):
        spec = next(s for s in strategy.TOP_LEVEL_FEATURES if s.kind == "boolean")
        assert strategy.describe_bucket(spec, 1, 2, None) == spec.label

    def test_boolean_false_bucket_is_negated(self):
        spec = next(s for s in strategy.TOP_LEVEL_FEATURES if s.kind == "boolean")
        assert strategy.describe_bucket(spec, 0, 2, None) == f"Not {spec.label}"

    def test_categorical_two_buckets_span_the_value_table(self):
        spec = strategy.feature_index("hand_category_norm")
        spec = strategy.TOP_LEVEL_FEATURES[spec]
        label = strategy.describe_bucket(spec, 0, 2, np.array([0.5, 0.5]))
        assert "High Card" in label

    def test_last_bucket_includes_the_top_value_table_point(self):
        spec = strategy.TOP_LEVEL_FEATURES[strategy.feature_index("hand_category_norm")]
        label = strategy.describe_bucket(spec, 1, 2, np.array([0.5, 0.5]))
        assert "Straight Flush" in label

    def test_out_of_range_bucket_index_is_described_not_crashed_on(self):
        # condition_buckets' gene range is fixed at MAX_BUCKETS regardless of
        # a feature's own (possibly smaller) num_buckets -- match_rule
        # already treats this as "never matches" harmlessly; describe_bucket
        # must degrade gracefully too, not index past the real cut points.
        spec = strategy.TOP_LEVEL_FEATURES[strategy.feature_index("hand_category_norm")]
        label = strategy.describe_bucket(spec, 2, 2, np.array([0.5, 0.5]))
        assert "never matches" in label


class TestMatchRule:
    def test_all_wildcard_always_matches(self):
        buckets = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES, dtype=np.int64)
        cf = np.full(strategy.CONDITIONS_PER_RULE, strategy.WILDCARD)
        cb = np.zeros(strategy.CONDITIONS_PER_RULE, dtype=np.int64)
        assert strategy.match_rule(buckets, cf, cb) is True

    def test_matches_only_when_every_active_condition_agrees(self):
        buckets = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES, dtype=np.int64)
        buckets[0] = 1
        buckets[1] = 2
        cf = np.array([0, 1, strategy.WILDCARD])
        cb = np.array([1, 2, 0])
        assert strategy.match_rule(buckets, cf, cb) is True

    def test_one_disagreeing_condition_fails_the_match(self):
        buckets = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES, dtype=np.int64)
        buckets[0] = 1
        cf = np.array([0, strategy.WILDCARD, strategy.WILDCARD])
        cb = np.array([2, 0, 0])
        assert strategy.match_rule(buckets, cf, cb) is False


class TestFirstMatchingRule:
    def test_first_match_in_array_order_wins(self):
        buckets = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES, dtype=np.int64)
        cf = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cb = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        actions = np.zeros(strategy.NUM_RULES, dtype=np.int64)
        actions[0] = strategy.ACTION_RAISE
        actions[1] = strategy.ACTION_ALLIN
        result = strategy.first_matching_rule(buckets, cf, cb, actions)
        assert result == strategy.ACTION_RAISE

    def test_none_when_nothing_matches(self):
        buckets = np.zeros(strategy.NUM_TOP_LEVEL_FEATURES, dtype=np.int64)
        buckets[0] = 1
        cf = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)  # feature 0
        cb = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)  # required bucket 0
        actions = np.zeros(strategy.NUM_RULES, dtype=np.int64)
        assert strategy.first_matching_rule(buckets, cf, cb, actions) is None


class TestMutateNumBuckets:
    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        values = np.full(10, 2)
        result = strategy.mutate_num_buckets(values, 0.0, rng)
        assert np.array_equal(result, values)

    def test_full_rate_flips_every_value(self):
        rng = np.random.default_rng(0)
        values = np.array([2, 3, 2, 3])
        result = strategy.mutate_num_buckets(values, 1.0, rng)
        assert np.array_equal(result, np.array([3, 2, 3, 2]))


class TestMutateThresholds:
    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        thresholds = np.sort(rng.random((5, 2)), axis=-1)
        result = strategy.mutate_thresholds(thresholds, 0.0, 0.1, rng)
        assert np.array_equal(result, thresholds)

    def test_stays_within_0_and_1_and_sorted(self):
        rng = np.random.default_rng(1)
        thresholds = np.sort(rng.random((20, 2)), axis=-1)
        result = strategy.mutate_thresholds(thresholds, 1.0, 5.0, rng)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        assert np.all(result[:, 0] <= result[:, 1])


class TestMutateConditionFeatures:
    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        cf = np.full((4, 3), strategy.WILDCARD)
        result = strategy.mutate_condition_features(cf, 0.0, rng)
        assert np.array_equal(result, cf)

    def test_full_rate_values_stay_in_valid_range(self):
        rng = np.random.default_rng(0)
        cf = np.full((10, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        result = strategy.mutate_condition_features(cf, 1.0, rng)
        assert np.all(result >= strategy.WILDCARD)
        assert np.all(result < strategy.NUM_TOP_LEVEL_FEATURES)


class TestMutateConditionBucketsAndRuleActions:
    def test_condition_buckets_stay_in_range(self):
        rng = np.random.default_rng(0)
        cb = np.zeros((10, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        result = strategy.mutate_condition_buckets(cb, 1.0, rng)
        assert np.all(result >= 0) and np.all(result < strategy.MAX_BUCKETS)

    def test_rule_actions_stay_in_range(self):
        rng = np.random.default_rng(0)
        actions = np.zeros(strategy.NUM_RULES, dtype=np.int64)
        result = strategy.mutate_rule_actions(actions, 1.0, rng)
        assert np.all(result >= 0) and np.all(result < strategy.NUM_ACTION_CATEGORIES)

    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        actions = np.array([0, 1, 2, 3])
        result = strategy.mutate_rule_actions(actions, 0.0, rng)
        assert np.array_equal(result, actions)


class TestMutateAlphabetIndex:
    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        assert strategy.mutate_alphabet_index(2, 5, 0.0, rng) == 2

    def test_stays_within_alphabet_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            result = strategy.mutate_alphabet_index(2, 5, 1.0, rng)
            assert 0 <= result < 5


class TestRowCrossover:
    def test_each_row_comes_from_one_parent_only(self):
        rng = np.random.default_rng(0)
        a = np.full((20, 2), 1.0)
        b = np.full((20, 2), 2.0)
        mask = strategy.row_crossover_mask(20, rng)
        child = strategy.apply_row_mask(a, b, mask)
        for row in child:
            assert (row == 1.0).all() or (row == 2.0).all()

    def test_both_parents_contribute_over_a_large_sample(self):
        rng = np.random.default_rng(0)
        a = np.full((200, 2), 1.0)
        b = np.full((200, 2), 2.0)
        mask = strategy.row_crossover_mask(200, rng)
        child = strategy.apply_row_mask(a, b, mask)
        assert np.any(child == 1.0)
        assert np.any(child == 2.0)

    def test_1d_arrays_supported(self):
        rng = np.random.default_rng(0)
        a = np.full(20, 1.0)
        b = np.full(20, 2.0)
        mask = strategy.row_crossover_mask(20, rng)
        child = strategy.apply_row_mask(a, b, mask)
        assert all(v in (1.0, 2.0) for v in child)
