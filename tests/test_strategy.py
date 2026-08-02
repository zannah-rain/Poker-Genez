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

    def test_bucketable_indices_exclude_booleans(self):
        for idx in strategy.BUCKETABLE_INDICES:
            assert strategy.CONDITION_FEATURES[idx].kind != "boolean"

    def test_bucket_gene_row_is_valid_for_bucketable_and_negative_for_boolean(self):
        for idx, spec in enumerate(strategy.CONDITION_FEATURES):
            row = strategy.bucket_gene_row(idx)
            if spec.kind == "boolean":
                assert row == -1
            else:
                assert 0 <= row < strategy.NUM_BUCKETABLE

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


class TestComputeAllBuckets:
    def test_boolean_features_bucket_at_0_5(self):
        values = np.zeros(strategy.NUM_CONDITION_FEATURES)
        bool_idx = np.flatnonzero(strategy.BOOLEAN_MASK)[0]
        values[bool_idx] = 1.0
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.5)
        buckets = strategy.compute_all_buckets(values, num_buckets, thresholds)
        assert buckets[bool_idx] == 1
        other_bool_idx = np.flatnonzero(strategy.BOOLEAN_MASK)[1]
        assert buckets[other_bool_idx] == 0

    def test_bucketable_feature_uses_its_own_thresholds(self):
        values = np.zeros(strategy.NUM_CONDITION_FEATURES)
        target = strategy.BUCKETABLE_INDICES[0]
        values[target] = 0.7
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.5)
        buckets = strategy.compute_all_buckets(values, num_buckets, thresholds)
        assert buckets[target] == 1

    def test_is_deterministic(self):
        values = np.random.default_rng(0).random(strategy.NUM_CONDITION_FEATURES)
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 3)
        thresholds = np.sort(np.random.default_rng(1).random((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1)), axis=-1)
        a = strategy.compute_all_buckets(values, num_buckets, thresholds)
        b = strategy.compute_all_buckets(values, num_buckets, thresholds)
        assert np.array_equal(a, b)

    def test_does_not_mutate_input_values(self):
        values = np.full(strategy.NUM_CONDITION_FEATURES, 0.5)
        original = values.copy()
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.3)
        strategy.compute_all_buckets(values, num_buckets, thresholds)
        assert np.array_equal(values, original)


class TestEffectiveConditionBucket:
    def test_boolean_feature_is_always_treated_as_2_buckets(self):
        bool_idx = np.flatnonzero(strategy.BOOLEAN_MASK)[0]
        num_buckets = np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)
        assert strategy.effective_condition_bucket(bool_idx, 1, num_buckets) == 1
        assert strategy.effective_condition_bucket(bool_idx, 0, num_buckets) == 0

    def test_clips_to_highest_existing_bucket(self):
        idx = strategy.BUCKETABLE_INDICES[0]
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        assert strategy.effective_condition_bucket(idx, 2, num_buckets) == 1

    def test_no_clip_needed_returns_unchanged(self):
        idx = strategy.BUCKETABLE_INDICES[0]
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 3)
        assert strategy.effective_condition_bucket(idx, 1, num_buckets) == 1


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
        # condition_buckets' gene range is fixed at MAX_BUCKETS regardless of
        # a feature's own (possibly smaller) num_buckets -- match_rule
        # clips via effective_condition_bucket; describe_bucket must degrade
        # the same way, rendering the highest bucket that actually exists
        # rather than indexing past the real cut points.
        spec = strategy.CONDITION_FEATURES[strategy.feature_index("hand_category_norm")]
        label = strategy.describe_bucket(spec, 2, 2, np.array([0.5, 0.5]))
        assert label == strategy.describe_bucket(spec, 1, 2, np.array([0.5, 0.5]))


class TestMatchRule:
    def _full_num_buckets(self):
        return np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)

    def test_all_wildcard_always_matches(self):
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        cf = np.full(strategy.CONDITIONS_PER_RULE, strategy.WILDCARD)
        cb = np.zeros(strategy.CONDITIONS_PER_RULE, dtype=np.int64)
        assert strategy.match_rule(buckets, cf, cb, self._full_num_buckets()) is True

    def test_matches_only_when_every_active_condition_agrees(self):
        f0, f1 = int(strategy.BUCKETABLE_INDICES[0]), int(strategy.BUCKETABLE_INDICES[1])
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        buckets[f0] = 1
        buckets[f1] = 2
        cf = np.array([f0, f1, strategy.WILDCARD])
        cb = np.array([1, 2, 0])
        assert strategy.match_rule(buckets, cf, cb, self._full_num_buckets()) is True

    def test_one_disagreeing_condition_fails_the_match(self):
        f0 = int(strategy.BUCKETABLE_INDICES[0])
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        buckets[f0] = 1
        cf = np.array([f0, strategy.WILDCARD, strategy.WILDCARD])
        cb = np.array([2, 0, 0])
        assert strategy.match_rule(buckets, cf, cb, self._full_num_buckets()) is False

    def test_a_condition_bucket_beyond_the_feature_num_buckets_is_clipped(self):
        f0 = int(strategy.BUCKETABLE_INDICES[0])
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        buckets[f0] = 1  # highest existing bucket when num_buckets == 2
        cf = np.array([f0, strategy.WILDCARD, strategy.WILDCARD])
        cb = np.array([2, 0, 0])  # stored bucket 2, but this feature only has 2 buckets
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
        assert strategy.match_rule(buckets, cf, cb, num_buckets) is True


class TestFirstMatchingRuleIndex:
    def test_first_match_in_array_order_wins(self):
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        cf = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cb = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        num_buckets = np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)
        result = strategy.first_matching_rule_index(buckets, cf, cb, num_buckets)
        assert result == 0

    def test_none_when_nothing_matches(self):
        f0 = int(strategy.BUCKETABLE_INDICES[0])
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        buckets[f0] = 1
        cf = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), f0, dtype=np.int64)
        cb = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)  # required bucket 0
        num_buckets = np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)
        assert strategy.first_matching_rule_index(buckets, cf, cb, num_buckets) is None

    def test_finds_a_later_match_when_earlier_rules_fail(self):
        f0 = int(strategy.BUCKETABLE_INDICES[0])
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        buckets[f0] = 1
        cf = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD, dtype=np.int64)
        cb = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        cf[0, 0] = f0
        cb[0, 0] = 0  # rule 0 requires bucket 0, but buckets[f0] == 1 -- fails
        num_buckets = np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)
        assert strategy.first_matching_rule_index(buckets, cf, cb, num_buckets) == 1


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
        assert np.all(result < strategy.NUM_CONDITION_FEATURES)


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


class TestEnforceMinOneCondition:
    def _eligible(self):
        return strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]  # flop: every feature eligible

    def test_all_wildcard_rule_gets_exactly_one_condition_filled(self):
        rng = np.random.default_rng(0)
        cf = np.full((5, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD, dtype=np.int64)
        cb = np.zeros((5, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        new_cf, new_cb, num_repaired = strategy.enforce_min_one_condition(cf, cb, self._eligible(), rng)
        assert num_repaired == 5
        for row in new_cf:
            assert np.count_nonzero(row != strategy.WILDCARD) == 1

    def test_already_valid_rule_is_untouched(self):
        rng = np.random.default_rng(0)
        cf = np.full((5, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD, dtype=np.int64)
        cf[:, 0] = int(self._eligible()[0])
        cb = np.zeros((5, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        new_cf, new_cb, num_repaired = strategy.enforce_min_one_condition(cf, cb, self._eligible(), rng)
        assert num_repaired == 0
        assert np.array_equal(new_cf, cf)
        assert np.array_equal(new_cb, cb)

    def test_filled_feature_comes_from_the_eligible_pool(self):
        rng = np.random.default_rng(0)
        cf = np.full((50, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD, dtype=np.int64)
        cb = np.zeros((50, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP]
        new_cf, _, _ = strategy.enforce_min_one_condition(cf, cb, eligible, rng)
        eligible_set = set(eligible.tolist())
        for row in new_cf:
            active = [int(x) for x in row if x != strategy.WILDCARD]
            assert len(active) == 1
            assert active[0] in eligible_set

    def test_does_not_mutate_input(self):
        rng = np.random.default_rng(0)
        cf = np.full((5, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD, dtype=np.int64)
        cb = np.zeros((5, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        cf_before, cb_before = cf.copy(), cb.copy()
        strategy.enforce_min_one_condition(cf, cb, self._eligible(), rng)
        assert np.array_equal(cf, cf_before)
        assert np.array_equal(cb, cb_before)


class TestMutateRuleMixActions:
    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        values = np.full(10, strategy.NO_MIX)
        result = strategy.mutate_rule_mix_actions(values, 0.0, rng)
        assert np.array_equal(result, values)

    def test_full_rate_values_stay_in_valid_range(self):
        rng = np.random.default_rng(0)
        values = np.full(50, strategy.NO_MIX)
        result = strategy.mutate_rule_mix_actions(values, 1.0, rng)
        assert np.all(result >= strategy.NO_MIX)
        assert np.all(result < strategy.NUM_ACTION_CATEGORIES)

    def test_full_rate_can_produce_both_no_mix_and_mix_values(self):
        rng = np.random.default_rng(0)
        values = np.full(200, strategy.NO_MIX)
        result = strategy.mutate_rule_mix_actions(values, 1.0, rng)
        assert np.any(result == strategy.NO_MIX)
        assert np.any(result != strategy.NO_MIX)


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


class TestResolveRuleAction:
    def test_no_mix_always_returns_primary(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert strategy.resolve_rule_action(strategy.ACTION_CALL, strategy.NO_MIX, rng) == strategy.ACTION_CALL

    def test_no_rng_always_returns_primary_even_when_mix(self):
        for _ in range(20):
            assert strategy.resolve_rule_action(strategy.ACTION_CALL, strategy.ACTION_ALLIN, None) == strategy.ACTION_CALL

    def test_mix_with_rng_produces_both_actions_over_many_draws(self):
        rng = np.random.default_rng(0)
        outcomes = {strategy.resolve_rule_action(strategy.ACTION_CALL, strategy.ACTION_ALLIN, rng) for _ in range(200)}
        assert outcomes == {strategy.ACTION_CALL, strategy.ACTION_ALLIN}


class TestDescribeRuleAction:
    def test_fixed_action_is_just_its_label(self):
        label = strategy.describe_rule_action(strategy.ACTION_CALL, strategy.NO_MIX)
        assert label == strategy.ACTION_CATEGORIES[strategy.ACTION_CALL]

    def test_mix_shows_both_labels_and_the_50_50_split(self):
        label = strategy.describe_rule_action(strategy.ACTION_CALL, strategy.ACTION_ALLIN)
        assert "50%" in label
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_CALL] in label
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_ALLIN] in label


class TestStreetsAndRulePools:
    def test_street_labels_match_num_streets(self):
        assert len(strategy.STREET_LABELS) == strategy.NUM_STREETS

    def test_preflop_is_street_zero(self):
        assert strategy.PREFLOP == 0
        assert strategy.STREET_LABELS[strategy.PREFLOP] == "Preflop"

    def test_num_rules_is_streets_times_rules_per_street(self):
        assert strategy.NUM_RULES == strategy.NUM_STREETS * strategy.RULES_PER_STREET

    def test_rules_per_street_default_is_30(self):
        assert strategy.RULES_PER_STREET == 30


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


class TestMatchPreflopRule:
    def _buckets(self):
        return np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)

    def _num_buckets(self):
        return np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)

    def test_fails_when_hole_category_not_claimed(self):
        mask = np.zeros(strategy.NUM_HOLE_CATEGORIES)
        cf = np.full(strategy.CONDITIONS_PER_RULE, strategy.WILDCARD)
        cb = np.zeros(strategy.CONDITIONS_PER_RULE, dtype=np.int64)
        assert strategy.match_preflop_rule(self._buckets(), cf, cb, self._num_buckets(), mask, hole_category=3) is False

    def test_succeeds_when_hole_category_claimed_and_conditions_match(self):
        mask = np.zeros(strategy.NUM_HOLE_CATEGORIES)
        mask[3] = 1.0
        cf = np.full(strategy.CONDITIONS_PER_RULE, strategy.WILDCARD)
        cb = np.zeros(strategy.CONDITIONS_PER_RULE, dtype=np.int64)
        assert strategy.match_preflop_rule(self._buckets(), cf, cb, self._num_buckets(), mask, hole_category=3) is True

    def test_claimed_category_but_failing_regular_condition_still_fails(self):
        mask = np.ones(strategy.NUM_HOLE_CATEGORIES)
        f0 = int(strategy.BUCKETABLE_INDICES[0])
        buckets = self._buckets()
        buckets[f0] = 1
        cf = np.array([f0, strategy.WILDCARD, strategy.WILDCARD])
        cb = np.array([2, 0, 0])
        assert strategy.match_preflop_rule(buckets, cf, cb, self._num_buckets(), mask, hole_category=0) is False

    def test_empty_mask_never_matches_any_category(self):
        mask = np.zeros(strategy.NUM_HOLE_CATEGORIES)
        cf = np.full(strategy.CONDITIONS_PER_RULE, strategy.WILDCARD)
        cb = np.zeros(strategy.CONDITIONS_PER_RULE, dtype=np.int64)
        for category in range(strategy.NUM_HOLE_CATEGORIES):
            assert strategy.match_preflop_rule(
                self._buckets(), cf, cb, self._num_buckets(), mask, hole_category=category,
            ) is False


class TestFirstMatchingPreflopRuleIndex:
    def _num_buckets(self):
        return np.full(strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS)

    def test_first_match_wins(self):
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        cf = np.full((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cb = np.zeros((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        mask = np.ones((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        result = strategy.first_matching_preflop_rule_index(buckets, cf, cb, self._num_buckets(), mask, hole_category=0)
        assert result == 0

    def test_none_when_no_rule_claims_the_category(self):
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        cf = np.full((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cb = np.zeros((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        assert strategy.first_matching_preflop_rule_index(
            buckets, cf, cb, self._num_buckets(), mask, hole_category=5,
        ) is None

    def test_skips_rules_that_dont_claim_the_category_even_if_earlier(self):
        buckets = np.zeros(strategy.NUM_CONDITION_FEATURES, dtype=np.int64)
        cf = np.full((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cb = np.zeros((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        mask[1, 2] = 1.0  # only rule 1 claims category 2
        result = strategy.first_matching_preflop_rule_index(buckets, cf, cb, self._num_buckets(), mask, hole_category=2)
        assert result == 1


class TestEligibleConditionIndicesByStreet:
    def test_one_array_per_street(self):
        assert len(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET) == strategy.NUM_STREETS

    def test_preflop_excludes_board_texture_and_postflop_only_made_hand_features(self):
        from features import group_of
        preflop_eligible = set(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP].tolist())
        for i, spec in enumerate(strategy.CONDITION_FEATURES):
            is_excluded = group_of(spec) == "Board / Flop Characteristics" or spec.key in strategy._POST_FLOP_ONLY_MADE_HAND_KEYS
            if is_excluded:
                assert i not in preflop_eligible
            else:
                assert i in preflop_eligible

    def test_preflop_excludes_the_specific_made_hand_keys_named_in_the_request(self):
        excluded_keys = {strategy.CONDITION_FEATURES[i].key for i in strategy._PREFLOP_EXCLUDED_INDICES}
        for key in ("num_overcards_norm", "top_pair", "third_pair", "overpair", "underpair"):
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


class TestMutateConditionFeaturesEligibility:
    def test_restricted_pool_never_produces_ineligible_indices(self):
        rng = np.random.default_rng(0)
        cf = np.full((strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP]
        result = strategy.mutate_condition_features(cf, 1.0, rng, eligible)
        eligible_set = set(eligible.tolist())
        for x in result.flat:
            assert x == strategy.WILDCARD or int(x) in eligible_set

    def test_default_pool_covers_every_condition_feature(self):
        rng = np.random.default_rng(0)
        cf = np.full((200, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        result = strategy.mutate_condition_features(cf, 1.0, rng)
        assert np.all(result >= strategy.WILDCARD)
        assert np.all(result < strategy.NUM_CONDITION_FEATURES)


class TestActionLabels:
    def test_fold_label_reads_check_fold_give_up(self):
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_FOLD] == "Check / Fold (Give up)"
