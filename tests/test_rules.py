import numpy as np
import pytest

import rules
import strategy
from cards import Card
from features import Situation
from gto import GTO_SPOTS, GTOSpot, SpotMatcher


def make_situation(**overrides) -> Situation:
    defaults = dict(
        hole=[Card.from_str("Ah"), Card.from_str("Kd")],
        board=[],
        street=0,
        pot=10.0,
        call_amount=0.0,
        my_stack=200.0,
        effective_stack=200.0,
        position=0,
        num_seats_this_street=6,
        seat_index=0,
        button_idx=0,
        num_seats_total=6,
        num_active=6,
        num_raises_this_street=0,
        num_preflop_raises=0,
        is_aggressor=False,
        starting_stack=200.0,
        big_blind=2.0,
    )
    defaults.update(overrides)
    return Situation(**defaults)


def make_context(condition_values=None, hole_category=0, situation=None) -> rules.MatchContext:
    if condition_values is None:
        condition_values = np.zeros(strategy.NUM_CONDITION_FEATURES)
    if situation is None:
        situation = make_situation()
    return rules.MatchContext(situation, condition_values, hole_category)


class TestSingleAction:
    def test_resolve_returns_fixed_index_regardless_of_rng(self):
        action = rules.SingleAction(strategy.ACTION_RAISE_75)
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert action.resolve(rng) == strategy.ACTION_RAISE_75
        assert action.resolve(None) == strategy.ACTION_RAISE_75

    def test_describe_returns_the_action_label(self):
        action = rules.SingleAction(strategy.ACTION_CALL)
        assert action.describe() == strategy.ACTION_CATEGORIES[strategy.ACTION_CALL]

    def test_to_indices_returns_no_mix(self):
        action = rules.SingleAction(strategy.ACTION_ALLIN)
        assert action.to_indices() == (strategy.ACTION_ALLIN, strategy.NO_MIX)

    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        action = rules.SingleAction(strategy.ACTION_CALL)
        mutated = action.mutate(rng, 0.0)
        assert mutated == action

    def test_full_rate_can_turn_into_a_mix(self):
        rng = np.random.default_rng(0)
        action = rules.SingleAction(strategy.ACTION_CALL)
        results = {type(action.mutate(rng, 1.0)) for _ in range(50)}
        assert rules.MixAction in results

    def test_mutated_primary_index_stays_in_range(self):
        rng = np.random.default_rng(0)
        action = rules.SingleAction(strategy.ACTION_CALL)
        for _ in range(50):
            mutated = action.mutate(rng, 1.0)
            primary, _ = mutated.to_indices()
            assert 0 <= primary < strategy.NUM_ACTION_CATEGORIES


class TestMixAction:
    def test_resolve_with_rng_produces_both_outcomes(self):
        action = rules.MixAction(strategy.ACTION_CALL, strategy.ACTION_ALLIN)
        rng = np.random.default_rng(0)
        outcomes = {action.resolve(rng) for _ in range(200)}
        assert outcomes == {strategy.ACTION_CALL, strategy.ACTION_ALLIN}

    def test_resolve_without_rng_always_primary(self):
        action = rules.MixAction(strategy.ACTION_CALL, strategy.ACTION_ALLIN)
        for _ in range(20):
            assert action.resolve(None) == strategy.ACTION_CALL

    def test_describe_shows_both_labels(self):
        action = rules.MixAction(strategy.ACTION_CALL, strategy.ACTION_ALLIN)
        label = action.describe()
        assert "50%" in label
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_CALL] in label
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_ALLIN] in label

    def test_to_indices_returns_both(self):
        action = rules.MixAction(strategy.ACTION_CALL, strategy.ACTION_ALLIN)
        assert action.to_indices() == (strategy.ACTION_CALL, strategy.ACTION_ALLIN)

    def test_full_rate_can_collapse_to_single_action(self):
        rng = np.random.default_rng(0)
        action = rules.MixAction(strategy.ACTION_CALL, strategy.ACTION_ALLIN)
        results = {type(action.mutate(rng, 1.0)) for _ in range(50)}
        assert rules.SingleAction in results


class TestRandomAction:
    def test_mix_init_prob_zero_never_produces_mix(self):
        rng = np.random.default_rng(0)
        results = [rules.random_action(rng, mix_init_prob=0.0) for _ in range(30)]
        assert all(isinstance(a, rules.SingleAction) for a in results)

    def test_mix_init_prob_one_always_produces_mix(self):
        rng = np.random.default_rng(0)
        results = [rules.random_action(rng, mix_init_prob=1.0) for _ in range(30)]
        assert all(isinstance(a, rules.MixAction) for a in results)

    def test_primary_index_in_valid_range(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            primary, _ = rules.random_action(rng).to_indices()
            assert 0 <= primary < strategy.NUM_ACTION_CATEGORIES


class TestCondition:
    def _feature(self, key: str) -> int:
        return strategy.feature_index(key)

    def test_matches_boolean_feature_at_threshold_0_5(self):
        fi = self._feature("is_aggressor")
        cond = rules.Condition(fi, 2, (0.5, 0.5), 1)
        true_values = np.zeros(strategy.NUM_CONDITION_FEATURES)
        true_values[fi] = 1.0
        false_values = np.zeros(strategy.NUM_CONDITION_FEATURES)
        false_values[fi] = 0.0
        assert cond.matches(true_values) is True
        assert cond.matches(false_values) is False

    def test_matches_continuous_feature_bucket(self):
        fi = self._feature("hand_category_norm")
        cond = rules.Condition(fi, 2, (0.5, 0.5), 1)
        low = np.zeros(strategy.NUM_CONDITION_FEATURES)
        low[fi] = 0.2
        high = np.zeros(strategy.NUM_CONDITION_FEATURES)
        high[fi] = 0.8
        assert cond.matches(low) is False
        assert cond.matches(high) is True

    def test_effective_bucket_clips_a_stale_bucket_to_the_highest_existing_one(self):
        fi = self._feature("hand_category_norm")
        cond = rules.Condition(fi, 2, (0.5, 0.5), 2)  # bucket 2 doesn't exist when num_buckets == 2
        assert cond.effective_bucket == 1
        high = np.zeros(strategy.NUM_CONDITION_FEATURES)
        high[fi] = 0.9
        assert cond.matches(high) is True  # bucket 1 (clipped from 2) matches a high value

    def test_describe_boolean(self):
        fi = self._feature("is_aggressor")
        spec = strategy.CONDITION_FEATURES[fi]
        true_cond = rules.Condition(fi, 2, (0.5, 0.5), 1)
        false_cond = rules.Condition(fi, 2, (0.5, 0.5), 0)
        assert true_cond.describe() == spec.label
        assert false_cond.describe() == f"Not {spec.label}"

    def test_describe_categorical(self):
        fi = self._feature("hand_category_norm")
        cond = rules.Condition(fi, 2, (0.5, 0.5), 0)
        assert "High Card" in cond.describe()

    def test_random_boolean_feature_always_2_buckets_at_0_5(self):
        rng = np.random.default_rng(0)
        fi = strategy.feature_index("is_aggressor")
        eligible = np.array([fi])
        for _ in range(10):
            cond = rules.Condition.random(rng, eligible)
            assert cond.num_buckets == 2
            assert cond.thresholds == (0.5, 0.5)

    def test_random_stays_within_valid_ranges(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        for _ in range(50):
            cond = rules.Condition.random(rng, eligible)
            assert int(cond.feature_index) in {int(x) for x in eligible}
            assert strategy.MIN_BUCKETS <= cond.num_buckets <= strategy.MAX_BUCKETS
            assert 0 <= cond.bucket < strategy.MAX_BUCKETS

    def test_mutate_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        cond = rules.Condition.random(np.random.default_rng(1), eligible)
        mutated = cond.mutate(rng, 0.0, eligible, 0.05)
        assert mutated == cond

    def test_mutate_full_rate_stays_valid(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        cond = rules.Condition.random(np.random.default_rng(1), eligible)
        for _ in range(30):
            cond = cond.mutate(rng, 1.0, eligible, 0.05)
            assert int(cond.feature_index) in {int(x) for x in eligible}
            assert strategy.MIN_BUCKETS <= cond.num_buckets <= strategy.MAX_BUCKETS
            assert 0.0 <= min(cond.thresholds) and max(cond.thresholds) <= 1.0
            assert list(cond.thresholds) == sorted(cond.thresholds)

    def test_mutate_boolean_feature_keeps_fixed_bucketing(self):
        rng = np.random.default_rng(0)
        fi = strategy.feature_index("is_aggressor")
        eligible = np.array([fi])  # only one eligible feature -- feature identity can't change
        cond = rules.Condition(fi, 2, (0.5, 0.5), 1)
        for _ in range(20):
            cond = cond.mutate(rng, 1.0, eligible, 0.05)
            assert cond.num_buckets == 2
            assert cond.thresholds == (0.5, 0.5)


class TestStandardRuleEvaluate:
    def test_conjunction_all_conditions_must_match(self):
        fi_a = strategy.feature_index("facing_bet")
        fi_b = strategy.feature_index("is_aggressor")
        conditions = (
            rules.Condition(fi_a, 2, (0.5, 0.5), 1),
            rules.Condition(fi_b, 2, (0.5, 0.5), 1),
        )
        rule = rules.StandardRule(conditions, rules.SingleAction(strategy.ACTION_RAISE_75), None, 0.5)

        both_true = np.zeros(strategy.NUM_CONDITION_FEATURES)
        both_true[fi_a] = 1.0
        both_true[fi_b] = 1.0
        only_one = np.zeros(strategy.NUM_CONDITION_FEATURES)
        only_one[fi_a] = 1.0

        situation = make_situation(pot=20.0)
        assert rule.evaluate(make_context(both_true, situation=situation), None) is not None
        assert rule.evaluate(make_context(only_one, situation=situation), None) is None

    def test_hole_category_gate(self):
        fi = strategy.feature_index("facing_bet")
        cond = rules.Condition(fi, 2, (0.0, 0.0), 1)  # always matches
        mask = tuple(i == 3 for i in range(strategy.NUM_HOLE_CATEGORIES))
        rule = rules.StandardRule((cond,), rules.SingleAction(strategy.ACTION_ALLIN), mask, 0.5)

        claimed = rule.evaluate(make_context(hole_category=3), None)
        unclaimed = rule.evaluate(make_context(hole_category=4), None)
        assert claimed is not None
        assert unclaimed is None

    def test_fold_action_produces_fold_decision(self):
        cond = rules.Condition(strategy.feature_index("facing_bet"), 2, (0.0, 0.0), 1)
        rule = rules.StandardRule((cond,), rules.SingleAction(strategy.ACTION_FOLD), None, 0.5)
        decision = rule.evaluate(make_context(), None)
        assert decision == rules.Decision("fold")

    def test_call_action_produces_call_decision(self):
        cond = rules.Condition(strategy.feature_index("facing_bet"), 2, (0.0, 0.0), 1)
        rule = rules.StandardRule((cond,), rules.SingleAction(strategy.ACTION_CALL), None, 0.5)
        decision = rule.evaluate(make_context(), None)
        assert decision == rules.Decision("call")

    def test_raise_action_sizes_at_its_pot_fraction(self):
        cond = rules.Condition(strategy.feature_index("facing_bet"), 2, (0.0, 0.0), 1)
        rule = rules.StandardRule((cond,), rules.SingleAction(strategy.ACTION_RAISE_75), None, 0.5)
        decision = rule.evaluate(make_context(situation=make_situation(pot=40.0)), None)
        assert decision.kind == "raise"
        assert decision.bet_size == pytest.approx(0.75 * 40.0)

    def test_allin_action_shoves_full_stack(self):
        cond = rules.Condition(strategy.feature_index("facing_bet"), 2, (0.0, 0.0), 1)
        rule = rules.StandardRule((cond,), rules.SingleAction(strategy.ACTION_ALLIN), None, 0.5)
        decision = rule.evaluate(make_context(situation=make_situation(my_stack=157.0)), None)
        assert decision == rules.Decision("raise", 157.0)

    def test_mix_action_resolves_through_rng(self):
        cond = rules.Condition(strategy.feature_index("facing_bet"), 2, (0.0, 0.0), 1)
        rule = rules.StandardRule(
            (cond,), rules.MixAction(strategy.ACTION_FOLD, strategy.ACTION_ALLIN), None, 0.5,
        )
        rng = np.random.default_rng(0)
        situation = make_situation(my_stack=100.0)
        outcomes = {rule.evaluate(make_context(situation=situation), rng).kind for _ in range(100)}
        assert outcomes == {"fold", "raise"}

    def test_describe_includes_hole_category_and_action(self):
        fi = strategy.feature_index("hand_category_norm")
        cond = rules.Condition(fi, 2, (0.5, 0.5), 1)
        mask = tuple(label == "Premium Pairs" for label in strategy.HOLE_CATEGORY_LABELS)
        rule = rules.StandardRule((cond,), rules.SingleAction(strategy.ACTION_RAISE_75), mask, 0.5)
        text = rule.describe()
        assert "Hole Category in {Premium Pairs}" in text
        assert "**Raise 75% Pot**" in text


class TestStandardRuleRandomAndMutate:
    def test_random_condition_count_within_bounds(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        for _ in range(30):
            rule = rules.StandardRule.random(rng, eligible)
            assert rules.MIN_CONDITIONS_PER_RULE <= len(rule.conditions) <= rules.MAX_CONDITIONS_PER_RULE

    def test_random_priority_in_0_1(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        for _ in range(30):
            rule = rules.StandardRule.random(rng, eligible)
            assert 0.0 <= rule.priority <= 1.0

    def test_random_hole_category_mask_only_when_requested(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[0]
        without = rules.StandardRule.random(rng, eligible)
        with_mask = rules.StandardRule.random(rng, eligible, hole_category_init_prob=0.5)
        assert without.hole_category_mask is None
        assert with_mask.hole_category_mask is not None
        assert len(with_mask.hole_category_mask) == strategy.NUM_HOLE_CATEGORIES

    def test_mutation_never_leaves_fewer_than_min_conditions(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        rule = rules.StandardRule.random(rng, eligible)
        for _ in range(50):
            rule = rule.mutate(rng, 1.0, eligible, 0.05)
            assert rules.MIN_CONDITIONS_PER_RULE <= len(rule.conditions) <= rules.MAX_CONDITIONS_PER_RULE

    def test_mutation_can_grow_and_shrink_condition_count_over_many_trials(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        rule = rules.StandardRule(
            (rules.Condition.random(rng, eligible),), rules.SingleAction(strategy.ACTION_FOLD), None, 0.5,
        )
        counts = set()
        for _ in range(200):
            rule = rule.mutate(rng, 1.0, eligible, 0.05)
            counts.add(len(rule.conditions))
        assert len(counts) > 1

    def test_mutation_keeps_priority_in_0_1(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        rule = rules.StandardRule.random(rng, eligible)
        for _ in range(50):
            rule = rule.mutate(rng, 1.0, eligible, 0.05)
            assert 0.0 <= rule.priority <= 1.0

    def test_mutation_zero_rate_leaves_rule_unchanged(self):
        rng = np.random.default_rng(0)
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[1]
        rule = rules.StandardRule.random(np.random.default_rng(1), eligible)
        mutated = rule.mutate(rng, 0.0, eligible, 0.05)
        assert mutated == rule


class TestGTOSpotRule:
    def _install_fake_spot(self, action_token: str) -> GTOSpot:
        return GTOSpot(
            key="fake_spot",
            label="Fake Spot",
            matcher=SpotMatcher(),  # matches everything
            action_ranges=((action_token, "AA-22, AKs-32s, AKo-32o"),),
            default_action="fold",
        )

    def test_inactive_never_fires(self):
        spot = self._install_fake_spot("allin")
        rule = rules.GTOSpotRule(spot, active=False)
        assert rule.evaluate(make_context(), None) is None

    def test_active_delegates_to_resolve_spot_action_fold(self):
        spot = self._install_fake_spot("fold")
        rule = rules.GTOSpotRule(spot, active=True)
        decision = rule.evaluate(make_context(), None)
        assert decision == rules.Decision("fold")

    def test_active_delegates_call(self):
        spot = self._install_fake_spot("call")
        rule = rules.GTOSpotRule(spot, active=True)
        decision = rule.evaluate(make_context(), None)
        assert decision == rules.Decision("call")

    def test_active_pot_fraction_raise(self):
        spot = self._install_fake_spot("raise_75")
        rule = rules.GTOSpotRule(spot, active=True)
        decision = rule.evaluate(make_context(situation=make_situation(pot=40.0)), None)
        assert decision.kind == "raise"
        assert decision.bet_size == pytest.approx(0.75 * 40.0)

    def test_active_allin_raise(self):
        spot = self._install_fake_spot("allin")
        rule = rules.GTOSpotRule(spot, active=True)
        decision = rule.evaluate(make_context(situation=make_situation(my_stack=157.0)), None)
        assert decision == rules.Decision("raise", 157.0)

    def test_active_bb_sized_raise(self):
        spot = self._install_fake_spot("raise_12bb")
        rule = rules.GTOSpotRule(spot, active=True)
        situation = make_situation(big_blind=2.0, call_amount=2.0)
        decision = rule.evaluate(make_context(situation=situation), None)
        assert decision == rules.Decision("raise", pytest.approx(12.0 * 2.0 - 2.0))

    def test_non_matching_situation_returns_none(self):
        spot = GTOSpot(
            key="fake_spot", label="Fake Spot",
            matcher=SpotMatcher(street=1),  # only matches flop
            action_ranges=(("allin", "AA-22, AKs-32s, AKo-32o"),),
            default_action="fold",
        )
        rule = rules.GTOSpotRule(spot, active=True)
        preflop_situation = make_situation(street=0)
        assert rule.evaluate(make_context(situation=preflop_situation), None) is None

    def test_describe_includes_label_and_matcher_summary(self):
        spot = GTOSpot(
            key="fake_spot", label="Fake Spot",
            matcher=SpotMatcher(street=0),
            action_ranges=(("fold", "AA"),),
            default_action="fold",
        )
        rule = rules.GTOSpotRule(spot, active=True)
        text = rule.describe()
        assert "Fake Spot" in text
        assert "Preflop" in text

    def test_mutate_zero_rate_never_flips(self):
        rng = np.random.default_rng(0)
        spot = self._install_fake_spot("fold")
        rule = rules.GTOSpotRule(spot, active=True)
        assert rule.mutate(rng, 0.0).active is True

    def test_mutate_full_rate_always_flips(self):
        rng = np.random.default_rng(0)
        spot = self._install_fake_spot("fold")
        rule = rules.GTOSpotRule(spot, active=True)
        assert rule.mutate(rng, 1.0).active is False
        assert rule.mutate(rng, 1.0).active is False  # mutate() returns a new instance -- `rule` itself never changes

    def test_catalog_has_one_rule_per_spot_in_catalog_order(self):
        rng = np.random.default_rng(0)
        catalog = rules.GTOSpotRule.catalog(rng, init_prob=0.0)
        assert [r.spot for r in catalog] == GTO_SPOTS
        assert all(r.active is False for r in catalog)

    def test_catalog_init_prob_one_activates_everything(self):
        rng = np.random.default_rng(0)
        catalog = rules.GTOSpotRule.catalog(rng, init_prob=1.0)
        assert all(r.active for r in catalog)


class TestGtoRulesFromSaved:
    def test_present_keys_use_saved_value_not_rng(self):
        rng = np.random.default_rng(0)
        saved = {spot.key: 1.0 for spot in GTO_SPOTS}
        gto_rules, unknown, missing = rules.gto_rules_from_saved(saved, rng, init_prob=0.0)
        assert all(r.active for r in gto_rules)
        assert unknown == set()
        assert missing == []

    def test_missing_keys_draw_from_rng(self):
        rng = np.random.default_rng(0)
        gto_rules, unknown, missing = rules.gto_rules_from_saved({}, rng, init_prob=1.0)
        assert all(r.active for r in gto_rules)
        assert missing == [spot.key for spot in GTO_SPOTS]

    def test_unknown_keys_are_reported_but_ignored(self):
        rng = np.random.default_rng(0)
        saved = {"not_a_real_spot": 1.0}
        gto_rules, unknown, missing = rules.gto_rules_from_saved(saved, rng, init_prob=0.0)
        assert unknown == {"not_a_real_spot"}
        assert len(gto_rules) == len(GTO_SPOTS)

    def test_only_missing_keys_consume_rng_draws(self):
        # If every key is present, the rng must not be touched at all.
        class _ExplodingRng:
            def random(self):
                raise AssertionError("rng.random() should not be called when every key is present")

        saved = {spot.key: 0.0 for spot in GTO_SPOTS}
        gto_rules, _, _ = rules.gto_rules_from_saved(saved, _ExplodingRng(), init_prob=1.0)
        assert all(not r.active for r in gto_rules)
