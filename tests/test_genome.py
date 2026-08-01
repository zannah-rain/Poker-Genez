import numpy as np
import pytest

import genome as genome_module
import gto as gto_module
import strategy
from cards import Card
from features import Situation
from genome import (
    BET_RAISE, CHECK_CALL, FOLD,
    Genome, load_population, mutate_bool_flags, save_population, uniform_crossover,
)
from gto import GTOSpot, NUM_GTO_SPOTS, SpotMatcher


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


def make_genome(
    num_buckets=None, thresholds=None,
    condition_features=None, condition_buckets=None, rule_actions=None,
    preflop_hole_category_mask=None,
    bucket_noise_std=0.0, gto_flags=None,
) -> Genome:
    """Defaults to an "empty" genome: every rule wildcard-conditioned to
    Fold, 2 buckets (cut at 0.5) for every bucketable feature, every preflop
    rule claiming *every* hole category (so it behaves like a wildcard
    there too, unless a test overrides it), no GTO charts trusted --
    individual tests override just the piece they're exercising."""
    if num_buckets is None:
        num_buckets = np.full(strategy.NUM_BUCKETABLE, 2)
    if thresholds is None:
        thresholds = np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.5)
    condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
    if condition_features is None:
        condition_features = np.full(condition_shape, strategy.WILDCARD, dtype=np.int64)
    if condition_buckets is None:
        condition_buckets = np.zeros(condition_shape, dtype=np.int64)
    if rule_actions is None:
        rule_actions = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
    if preflop_hole_category_mask is None:
        preflop_hole_category_mask = np.ones((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
    if gto_flags is None:
        gto_flags = np.zeros(NUM_GTO_SPOTS)
    return Genome(
        num_buckets=num_buckets, thresholds=thresholds,
        condition_features=condition_features, condition_buckets=condition_buckets,
        rule_actions=rule_actions, preflop_hole_category_mask=preflop_hole_category_mask,
        bucket_noise_std=bucket_noise_std,
        gto_flags=gto_flags,
    )


def genome_with_default_action(action: int, **kwargs) -> Genome:
    """A genome whose rule 0 matches every situation, on every street
    (all-wildcard conditions; for preflop, every hole category claimed too)
    and always plays `action` -- isolates decide()'s action-category ->
    (game action, bet size) mapping from the bucketing/matching/street-
    routing machinery, which is tested separately (see test_strategy.py and
    TestDecideRuleMatching/TestDecideStreetRouting below)."""
    rule_actions = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
    rule_actions[:, 0] = action
    return make_genome(rule_actions=rule_actions, **kwargs)


def genome_with_condition(feature_key: str, bucket: int, action: int, street: int = strategy.PREFLOP, **kwargs) -> Genome:
    """A genome whose rule 0 on `street` (preflop by default, matching
    make_situation's default street=0) requires exactly one (feature,
    bucket) match to play `action`; every other rule is wildcard-Fold, so a
    non-matching situation falls through to the default Fold. Preflop rule
    0 claims every hole category (see make_genome), so this isolates the
    regular condition being tested from the mandatory hole-category axis."""
    condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
    cf = np.full(condition_shape, strategy.WILDCARD, dtype=np.int64)
    cb = np.zeros(condition_shape, dtype=np.int64)
    rule_actions = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
    cf[street, 0, 0] = strategy.feature_index(feature_key)
    cb[street, 0, 0] = bucket
    rule_actions[street, 0] = action
    return make_genome(condition_features=cf, condition_buckets=cb, rule_actions=rule_actions, **kwargs)


class TestMutateBoolFlags:
    def test_zero_rate_never_flips(self):
        rng = np.random.default_rng(0)
        flags = np.array([0.0, 1.0, 0.0, 1.0])
        result = mutate_bool_flags(flags, rate=0.0, rng=rng)
        assert np.array_equal(result, flags)

    def test_full_rate_flips_every_flag(self):
        rng = np.random.default_rng(0)
        flags = np.array([0.0, 1.0, 0.0, 1.0])
        result = mutate_bool_flags(flags, rate=1.0, rng=rng)
        assert np.array_equal(result, 1.0 - flags)


class TestUniformCrossover:
    def test_each_gene_comes_from_one_parent(self):
        rng = np.random.default_rng(0)
        a = np.full(100, 0.0)
        b = np.full(100, 1.0)
        child = uniform_crossover(a, b, rng)
        assert all(v in (0.0, 1.0) for v in child)

    def test_both_parents_contribute_over_a_large_sample(self):
        rng = np.random.default_rng(0)
        a = np.full(200, 0.0)
        b = np.full(200, 1.0)
        child = uniform_crossover(a, b, rng)
        assert np.any(child == 0.0)
        assert np.any(child == 1.0)


class TestGenomeRandom:
    def test_gene_shapes(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert g.num_buckets.shape == (strategy.NUM_BUCKETABLE,)
        assert g.thresholds.shape == (strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1)
        condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        assert g.condition_features.shape == condition_shape
        assert g.condition_buckets.shape == condition_shape
        assert g.rule_actions.shape == (strategy.NUM_STREETS, strategy.RULES_PER_STREET)
        assert g.preflop_hole_category_mask.shape == (strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES)
        assert g.gto_flags.shape == (NUM_GTO_SPOTS,)

    def test_num_buckets_only_takes_min_or_max(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert set(np.unique(g.num_buckets)) <= {strategy.MIN_BUCKETS, strategy.MAX_BUCKETS}

    def test_thresholds_sorted_and_in_range(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert np.all(g.thresholds >= 0.0) and np.all(g.thresholds <= 1.0)
        assert np.all(g.thresholds[:, 0] <= g.thresholds[:, 1])

    def test_condition_features_in_valid_range(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert np.all(g.condition_features >= strategy.WILDCARD)
        assert np.all(g.condition_features < strategy.NUM_CONDITION_FEATURES)

    def test_rule_actions_in_valid_range(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert np.all(g.rule_actions >= 0)
        assert np.all(g.rule_actions < strategy.NUM_ACTION_CATEGORIES)

    def test_preflop_hole_category_mask_is_boolean_valued(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert set(np.unique(g.preflop_hole_category_mask)) <= {0.0, 1.0}

    def test_gto_flags_start_mostly_off(self):
        rng = np.random.default_rng(0)
        flags = np.concatenate([Genome.random(rng).gto_flags for _ in range(20)])
        assert flags.mean() < 0.3

    def test_lower_scale_means_fewer_active_conditions_on_average(self):
        rng = np.random.default_rng(0)
        sparse = np.mean([Genome.random(rng, scale=0.05).nonzero_weight_count() for _ in range(30)])
        dense = np.mean([Genome.random(rng, scale=0.5).nonzero_weight_count() for _ in range(30)])
        assert sparse < dense


class TestGenomeCopy:
    def test_copy_is_independent(self):
        g = make_genome()
        original_first_condition = int(g.condition_features[0, 0, 0])
        original_first_flag = float(g.gto_flags[0])
        original_first_category = float(g.preflop_hole_category_mask[0, 0])
        g2 = g.copy()
        g2.condition_features[0, 0, 0] = original_first_condition + 1
        g2.gto_flags[0] = 1.0 - original_first_flag
        g2.preflop_hole_category_mask[0, 0] = 1.0 - original_first_category
        assert g.condition_features[0, 0, 0] == original_first_condition
        assert g.gto_flags[0] == original_first_flag
        assert g.preflop_hole_category_mask[0, 0] == original_first_category

    def test_copy_preserves_scalars(self):
        g = make_genome(bucket_noise_std=0.07)
        g2 = g.copy()
        assert g2.bucket_noise_std == pytest.approx(0.07)


class TestNonzeroWeightCount:
    def test_counts_non_wildcard_conditions_across_all_streets(self):
        condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        cf = np.full(condition_shape, strategy.WILDCARD, dtype=np.int64)
        cf[strategy.PREFLOP, 0, 0] = 3
        cf[strategy.PREFLOP, 1, 0] = 5
        cf[1, 0, 0] = 1  # flop
        g = make_genome(condition_features=cf)
        assert g.nonzero_weight_count() == 3

    def test_all_wildcard_counts_zero(self):
        g = make_genome()
        assert g.nonzero_weight_count() == 0

    def test_does_not_count_street_or_hole_category_mask(self):
        # Street membership (which pool a rule lives in) and preflop's
        # hole_category_mask are mandatory, not optional conditions -- see
        # genome.py's module docstring -- so a genome that only "spends"
        # genes on those still counts zero.
        g = Genome.random(np.random.default_rng(0))
        g.condition_features[:] = strategy.WILDCARD
        assert g.nonzero_weight_count() == 0


class TestActiveGtoSpots:
    def test_returns_only_flagged_spots_in_catalog_order(self):
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[1] = 1.0
        flags[3] = 1.0
        g = make_genome(gto_flags=flags)
        active = g.active_gto_spots()
        assert [gto_module.GTO_SPOTS.index(s) for s in active] == [1, 3]

    def test_none_active_when_all_flags_off(self):
        g = make_genome()
        assert g.active_gto_spots() == []


class TestDecideActionCategories:
    def test_fold_action_folds_when_legal(self):
        g = genome_with_default_action(strategy.ACTION_FOLD)
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0

    def test_fold_action_checks_when_fold_illegal(self):
        g = genome_with_default_action(strategy.ACTION_FOLD)
        situation = make_situation()
        action, _ = g.decide(situation, legal_actions=[CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL

    def test_call_action_checks_or_calls(self):
        g = genome_with_default_action(strategy.ACTION_CALL)
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_each_raise_category_sizes_at_its_own_fixed_pot_fraction(self):
        for raise_action, fraction in strategy.RAISE_POT_FRACTION.items():
            g = genome_with_default_action(raise_action)
            situation = make_situation(pot=80.0)
            action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
            assert action == BET_RAISE
            assert bet_size == pytest.approx(fraction * 80.0)

    def test_raise_action_falls_back_to_check_call_if_illegal(self):
        g = genome_with_default_action(strategy.ACTION_RAISE_75)
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_allin_action_shoves_full_stack(self):
        g = genome_with_default_action(strategy.ACTION_ALLIN)
        situation = make_situation(my_stack=157.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(157.0)

    def test_allin_action_falls_back_to_check_call_if_illegal(self):
        g = genome_with_default_action(strategy.ACTION_ALLIN)
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL])
        assert action == CHECK_CALL
        assert bet_size == 0.0


class TestDecideRuleMatching:
    def test_no_matching_rule_defaults_to_fold(self):
        g = genome_with_condition("facing_bet", bucket=1, action=strategy.ACTION_RAISE_75)
        situation = make_situation(call_amount=0.0)  # facing_bet bucket 0, condition needs bucket 1
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD

    def test_matching_condition_fires_its_rule(self):
        g = genome_with_condition("facing_bet", bucket=1, action=strategy.ACTION_RAISE_75)
        situation = make_situation(call_amount=5.0, pot=20.0)  # facing_bet bucket 1
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE

    def test_first_matching_rule_in_array_order_wins(self):
        rule_actions = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
        rule_actions[strategy.PREFLOP, 0] = strategy.ACTION_ALLIN
        rule_actions[strategy.PREFLOP, 1] = strategy.ACTION_CALL
        g = make_genome(rule_actions=rule_actions)
        situation = make_situation()
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE  # rule 0 (All-In), not rule 1 (Call)

    def test_multiple_conditions_are_a_conjunction(self):
        condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        cf = np.full(condition_shape, strategy.WILDCARD, dtype=np.int64)
        cb = np.zeros(condition_shape, dtype=np.int64)
        rule_actions = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
        cf[strategy.PREFLOP, 0, 0] = strategy.feature_index("facing_bet")
        cb[strategy.PREFLOP, 0, 0] = 1
        cf[strategy.PREFLOP, 0, 1] = strategy.feature_index("is_aggressor")
        cb[strategy.PREFLOP, 0, 1] = 1
        rule_actions[strategy.PREFLOP, 0] = strategy.ACTION_RAISE_75
        g = make_genome(condition_features=cf, condition_buckets=cb, rule_actions=rule_actions)

        both_true = make_situation(call_amount=5.0, is_aggressor=True)
        only_one_true = make_situation(call_amount=5.0, is_aggressor=False)
        action_both, _ = g.decide(both_true, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        action_one, _ = g.decide(only_one_true, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action_both == BET_RAISE
        assert action_one == FOLD


class TestDecideStreetRouting:
    def test_each_street_only_checks_its_own_pool(self):
        rule_actions = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
        rule_actions[0, 0] = strategy.ACTION_RAISE_25
        rule_actions[1, 0] = strategy.ACTION_RAISE_50
        rule_actions[2, 0] = strategy.ACTION_RAISE_75
        rule_actions[3, 0] = strategy.ACTION_ALLIN
        g = make_genome(rule_actions=rule_actions)
        board_by_street = {
            0: [],
            1: [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("9h")],
            2: [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("9h"), Card.from_str("Kc")],
            3: [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("9h"), Card.from_str("Kc"), Card.from_str("3d")],
        }
        expected = {0: 0.25, 1: 0.50, 2: 0.75, 3: None}  # street 3 -> All-In (checked via bet_size == my_stack)
        for street in range(strategy.NUM_STREETS):
            situation = make_situation(street=street, board=board_by_street[street], pot=100.0, my_stack=500.0)
            action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
            assert action == BET_RAISE
            if street == 3:
                assert bet_size == pytest.approx(500.0)
            else:
                assert bet_size == pytest.approx(expected[street] * 100.0)

    def test_preflop_rule_never_fires_on_other_streets(self):
        # A rule defined only in the preflop pool must not leak into flop/
        # turn/river decisions -- each street's pool is fully independent.
        g = genome_with_condition("facing_bet", bucket=1, action=strategy.ACTION_ALLIN, street=strategy.PREFLOP)
        flop_situation = make_situation(
            street=1, call_amount=5.0, board=[Card.from_str("2c"), Card.from_str("7d"), Card.from_str("9h")],
        )
        action, _ = g.decide(flop_situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD  # flop pool is all-default (wildcard -> Fold), unaffected by the preflop rule


class TestDecidePreflopHoleCategory:
    def _situation_for_category(self, label: str, **overrides):
        hole_by_category = {
            "Premium Pairs": [Card.from_str("Ah"), Card.from_str("Ad")],
            "Junk": [Card.from_str("7h"), Card.from_str("2d")],
            "Axs": [Card.from_str("Ah"), Card.from_str("4h")],
        }
        return make_situation(hole=hole_by_category[label], **overrides)

    def test_rule_only_fires_for_claimed_categories(self):
        mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        mask[0, strategy.HOLE_CATEGORY_LABELS.index("Premium Pairs")] = 1.0
        g = genome_with_default_action(strategy.ACTION_ALLIN, preflop_hole_category_mask=mask)
        # genome_with_default_action sets every street's rule 0 to All-In
        # with wildcard conditions; only preflop additionally requires the
        # hole-category claim, so flop/turn/river are unaffected -- check
        # preflop specifically for both a claimed and an unclaimed hand.
        premium = self._situation_for_category("Premium Pairs")
        junk = self._situation_for_category("Junk")
        action_premium, _ = g.decide(premium, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        action_junk, _ = g.decide(junk, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action_premium == BET_RAISE
        assert action_junk == FOLD  # falls through to the (all-default) Fold

    def test_multiple_claimed_categories_all_match(self):
        mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        mask[0, strategy.HOLE_CATEGORY_LABELS.index("Premium Pairs")] = 1.0
        mask[0, strategy.HOLE_CATEGORY_LABELS.index("Axs")] = 1.0
        g = genome_with_default_action(strategy.ACTION_ALLIN, preflop_hole_category_mask=mask)
        for label in ("Premium Pairs", "Axs"):
            situation = self._situation_for_category(label)
            action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
            assert action == BET_RAISE
        junk_action, _ = g.decide(self._situation_for_category("Junk"), legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert junk_action == FOLD

    def test_empty_mask_never_matches_any_hand(self):
        mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        g = genome_with_default_action(strategy.ACTION_ALLIN, preflop_hole_category_mask=mask)
        for label in ("Premium Pairs", "Junk", "Axs"):
            situation = self._situation_for_category(label)
            action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
            assert action == FOLD

    def test_hole_category_check_is_exact_not_bucketed_by_noise(self):
        # bucket_noise_std jitters ordinary conditions but must never blur
        # which exact hole category a hand belongs to.
        mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        mask[0, strategy.HOLE_CATEGORY_LABELS.index("Premium Pairs")] = 1.0
        g = genome_with_default_action(strategy.ACTION_ALLIN, preflop_hole_category_mask=mask, bucket_noise_std=5.0)
        rng = np.random.default_rng(0)
        situation = self._situation_for_category("Junk")
        outcomes = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)[0] for _ in range(100)}
        assert outcomes == {FOLD}


class TestDecideBucketNoise:
    def test_zero_noise_is_deterministic(self):
        g = genome_with_default_action(strategy.ACTION_RAISE_75, bucket_noise_std=0.0)
        situation = make_situation()
        rng = np.random.default_rng(0)
        a1 = g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)
        a2 = g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)
        assert a1 == a2

    def test_deterministic_without_rng_even_with_nonzero_noise(self):
        g = genome_with_default_action(strategy.ACTION_RAISE_75, bucket_noise_std=0.5)
        situation = make_situation()
        results = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE]) for _ in range(5)}
        assert len(results) == 1

    def test_large_noise_can_flip_a_borderline_bucket(self):
        # A condition requiring hole_high_card_norm's bucket 1 (threshold at
        # 0.5) with a hand landing (Ace-King -> 1.0, comfortably bucket 1
        # without noise) should sometimes flip to bucket 0 under large jitter.
        g = genome_with_condition("hole_high_card_norm", bucket=1, action=strategy.ACTION_ALLIN, bucket_noise_std=0.8)
        situation = make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")])
        rng = np.random.default_rng(0)
        outcomes = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)[0] for _ in range(200)}
        assert BET_RAISE in outcomes
        assert FOLD in outcomes


class TestDecideGtoOverride:
    def _bb_vs_utg_index(self):
        return [s.key for s in gto_module.GTO_SPOTS].index("bb_vs_utg_open_100bb")

    def _matching_situation(self, hole):
        return make_situation(
            hole=hole,
            street=0,
            call_amount=2.0,
            effective_stack=200.0,
            big_blind=2.0,
            seat_index=2,
            button_idx=0,
            num_seats_total=6,
            is_aggressor=False,
            num_preflop_raises=1,
            raised_positions=frozenset({"UTG"}),
        )

    def test_chart_raise_bypasses_rule_list(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        # Rule list would otherwise always fold everything.
        g = genome_with_default_action(strategy.ACTION_FOLD, gto_flags=flags)
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(12.0 * 2.0 - 2.0)

    def test_chart_call(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        g = genome_with_default_action(strategy.ACTION_FOLD, gto_flags=flags)
        situation = self._matching_situation([Card.from_str("8c"), Card.from_str("8d")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_chart_default_fold(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        # Rule list would otherwise always raise -- chart's default should still win.
        g = genome_with_default_action(strategy.ACTION_ALLIN, gto_flags=flags)
        situation = self._matching_situation([Card.from_str("7c"), Card.from_str("2d")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0

    def test_inactive_flag_falls_through_to_rule_list(self):
        g = genome_with_default_action(strategy.ACTION_ALLIN)  # all gto_flags off
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE

    def test_chart_raise_falls_back_to_check_call_if_bet_raise_illegal(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        g = make_genome(gto_flags=flags)
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, bet_size = g.decide(situation, legal_actions=[CHECK_CALL])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_non_matching_situation_ignores_chart(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        g = genome_with_default_action(strategy.ACTION_ALLIN, gto_flags=flags)
        # street=1 (flop) doesn't match this spot's street=0 requirement.
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        situation.street = 1
        situation.board = [Card.from_str("2c"), Card.from_str("5d"), Card.from_str("9h")]
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE  # falls through to the rule list, which also raises here


class TestDecideGtoSizeSpecs(object):
    """Exercises the pot-fraction and all-in raise-size branches of
    _decide_from_gto_charts using a synthetic spot, since the real GTO_SPOTS
    catalog only currently uses bb-sized opens."""

    def _install_fake_spot(self, monkeypatch, action_token):
        spot = GTOSpot(
            key="fake_spot",
            label="Fake Spot",
            matcher=SpotMatcher(),  # matches everything
            action_ranges=((action_token, "AA-22, AKs-32s, AKo-32o"),),
            default_action="fold",
        )
        monkeypatch.setattr(genome_module, "GTO_SPOTS", [spot])

    def test_pot_fraction_raise_size(self, monkeypatch):
        self._install_fake_spot(monkeypatch, "raise_75")
        g = make_genome(gto_flags=np.array([1.0]))
        situation = make_situation(pot=40.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(0.75 * 40.0)

    def test_allin_raise_shoves_full_stack(self, monkeypatch):
        self._install_fake_spot(monkeypatch, "allin")
        g = make_genome(gto_flags=np.array([1.0]))
        situation = make_situation(my_stack=157.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(157.0)

    def test_fold_action(self, monkeypatch):
        self._install_fake_spot(monkeypatch, "fold")
        g = make_genome(gto_flags=np.array([1.0]))
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0


class TestGenomeMutate:
    def test_mutate_returns_new_object_and_keeps_noise_nonnegative(self):
        rng = np.random.default_rng(1)
        g = make_genome(bucket_noise_std=0.01)
        mutated = g.mutate(rng, rate=1.0, continuous_scale=0.3)
        assert mutated is not g
        assert mutated.bucket_noise_std >= 0

    def test_zero_rate_leaves_genes_unchanged(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(2))
        mutated = g.mutate(rng, rate=0.0, continuous_scale=0.3)
        assert np.array_equal(mutated.num_buckets, g.num_buckets)
        assert np.array_equal(mutated.thresholds, g.thresholds)
        assert np.array_equal(mutated.condition_features, g.condition_features)
        assert np.array_equal(mutated.condition_buckets, g.condition_buckets)
        assert np.array_equal(mutated.rule_actions, g.rule_actions)
        assert np.array_equal(mutated.preflop_hole_category_mask, g.preflop_hole_category_mask)
        assert mutated.bucket_noise_std == g.bucket_noise_std

    def test_full_rate_flips_hole_category_mask(self):
        rng = np.random.default_rng(1)
        g = make_genome(preflop_hole_category_mask=np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES)))
        mutated = g.mutate(rng, rate=1.0, continuous_scale=0.3)
        assert np.array_equal(mutated.preflop_hole_category_mask, np.ones_like(g.preflop_hole_category_mask))

    def test_original_genome_is_unmodified(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(3))
        before = g.condition_features.copy()
        g.mutate(rng, rate=1.0, continuous_scale=0.3)
        assert np.array_equal(g.condition_features, before)

    def test_thresholds_stay_sorted_and_in_range_after_mutation(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(4))
        mutated = g.mutate(rng, rate=1.0, continuous_scale=0.3)
        assert np.all(mutated.thresholds >= 0.0) and np.all(mutated.thresholds <= 1.0)
        assert np.all(mutated.thresholds[:, 0] <= mutated.thresholds[:, 1])


class TestGenomeCrossover:
    def test_crossover_returns_new_object(self):
        rng = np.random.default_rng(2)
        a = Genome.random(np.random.default_rng(5))
        b = Genome.random(np.random.default_rng(6))
        child = a.crossover(b, rng)
        assert child is not a and child is not b

    def test_num_buckets_and_thresholds_stay_coupled_per_feature(self):
        # Every feature's num_buckets and thresholds must come from the same
        # parent -- otherwise a bucket count could pair with cut points
        # sized for a different count.
        rng = np.random.default_rng(2)
        a = make_genome(
            num_buckets=np.full(strategy.NUM_BUCKETABLE, 2),
            thresholds=np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.1),
        )
        b = make_genome(
            num_buckets=np.full(strategy.NUM_BUCKETABLE, 3),
            thresholds=np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.9),
        )
        for _ in range(20):
            child = a.crossover(b, rng)
            from_a = child.num_buckets == 2
            from_b = child.num_buckets == 3
            assert np.all(child.thresholds[from_a] == 0.1)
            assert np.all(child.thresholds[from_b] == 0.9)

    def test_rule_conditions_and_action_stay_coupled_per_rule(self):
        rng = np.random.default_rng(2)
        condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        cf_a = np.full(condition_shape, 1)
        cf_b = np.full(condition_shape, 2)
        actions_a = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_FOLD)
        actions_b = np.full((strategy.NUM_STREETS, strategy.RULES_PER_STREET), strategy.ACTION_ALLIN)
        a = make_genome(condition_features=cf_a, rule_actions=actions_a)
        b = make_genome(condition_features=cf_b, rule_actions=actions_b)
        for _ in range(20):
            child = a.crossover(b, rng)
            from_a_rows = np.all(child.condition_features == 1, axis=-1)
            from_b_rows = np.all(child.condition_features == 2, axis=-1)
            assert np.all(child.rule_actions[from_a_rows] == strategy.ACTION_FOLD)
            assert np.all(child.rule_actions[from_b_rows] == strategy.ACTION_ALLIN)

    def test_preflop_hole_category_mask_stays_coupled_with_its_rule(self):
        rng = np.random.default_rng(2)
        condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        cf_a = np.full(condition_shape, 1)
        cf_b = np.full(condition_shape, 2)
        mask_a = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        mask_b = np.ones((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES))
        a = make_genome(condition_features=cf_a, preflop_hole_category_mask=mask_a)
        b = make_genome(condition_features=cf_b, preflop_hole_category_mask=mask_b)
        for _ in range(20):
            child = a.crossover(b, rng)
            from_a_rows = np.all(child.condition_features[strategy.PREFLOP] == 1, axis=-1)
            from_b_rows = np.all(child.condition_features[strategy.PREFLOP] == 2, axis=-1)
            assert np.all(child.preflop_hole_category_mask[from_a_rows] == 0.0)
            assert np.all(child.preflop_hole_category_mask[from_b_rows] == 1.0)

    def test_blended_bucket_noise_is_between_parents(self):
        rng = np.random.default_rng(2)
        a = make_genome(bucket_noise_std=0.1)
        b = make_genome(bucket_noise_std=0.5)
        for _ in range(20):
            child = a.crossover(b, rng)
            assert 0.1 <= child.bucket_noise_std <= 0.5


class TestSerialization:
    def test_to_dict_from_dict_round_trip_preserves_values(self):
        rng = np.random.default_rng(3)
        g = Genome.random(rng)
        data = g.to_dict()
        restored = Genome.from_dict(data, rng)
        assert np.array_equal(g.num_buckets, restored.num_buckets)
        assert np.array_equal(g.thresholds, restored.thresholds)
        assert np.array_equal(g.condition_features, restored.condition_features)
        assert np.array_equal(g.condition_buckets, restored.condition_buckets)
        assert np.array_equal(g.rule_actions, restored.rule_actions)
        assert np.array_equal(g.preflop_hole_category_mask, restored.preflop_hole_category_mask)
        assert np.array_equal(g.gto_flags, restored.gto_flags)
        assert g.bucket_noise_std == pytest.approx(restored.bucket_noise_std)

    def test_to_dict_splits_rules_by_street_label(self):
        g = Genome.random(np.random.default_rng(3))
        data = g.to_dict()
        assert set(data["rules"].keys()) == set(strategy.STREET_LABELS)
        for label in strategy.STREET_LABELS:
            assert len(data["rules"][label]) == strategy.RULES_PER_STREET

    def test_preflop_rules_carry_hole_categories_others_dont(self):
        g = Genome.random(np.random.default_rng(3))
        data = g.to_dict()
        for rule in data["rules"]["Preflop"]:
            assert "hole_categories" in rule
        for label in ("Flop", "Turn", "River"):
            for rule in data["rules"][label]:
                assert "hole_categories" not in rule

    def test_from_dict_drops_unknown_feature_and_fills_missing(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        first_feature = next(iter(data["feature_buckets"]))
        del data["feature_buckets"][first_feature]
        data["feature_buckets"]["not_a_real_feature"] = {"num_buckets": 2, "thresholds": [0.5, 0.5]}
        restored = Genome.from_dict(data, rng)
        assert restored.num_buckets.shape == (strategy.NUM_BUCKETABLE,)
        captured = capsys.readouterr()
        assert "unknown" in captured.out.lower()

    def test_from_dict_drops_unknown_rule_feature(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"]["Flop"][0]["conditions"][0]["feature"] = "not_a_real_feature"
        restored = Genome.from_dict(data, rng)
        assert restored.condition_features[1, 0, 0] == strategy.WILDCARD
        captured = capsys.readouterr()
        assert "unknown" in captured.out.lower()

    def test_from_dict_handles_missing_rules_for_a_street(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"]["River"] = data["rules"]["River"][:-1]
        restored = Genome.from_dict(data, rng)
        assert restored.rule_actions.shape == (strategy.NUM_STREETS, strategy.RULES_PER_STREET)
        captured = capsys.readouterr()
        assert "river" in captured.out.lower()

    def test_from_dict_handles_missing_street_entirely(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        del data["rules"]["Turn"]
        restored = Genome.from_dict(data, rng)
        assert restored.condition_features.shape == (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        captured = capsys.readouterr()
        assert "turn" in captured.out.lower()

    def test_from_dict_treats_old_flat_rule_list_as_no_saved_rules(self, capsys):
        # A save from before rules were split per street used a flat list --
        # there's no sane way to map those onto the new per-street pools.
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"] = [{"conditions": [], "action": "Fold"}]
        restored = Genome.from_dict(data, rng)
        assert restored.condition_features.shape == (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)

    def test_from_dict_drops_unknown_hole_category_label(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"]["Preflop"][0]["hole_categories"] = ["Not A Real Category"]
        restored = Genome.from_dict(data, rng)
        assert restored.preflop_hole_category_mask[0].sum() == 0
        captured = capsys.readouterr()
        assert "hole hand category" in captured.out.lower()

    def test_from_dict_handles_preflop_rule_missing_hole_categories(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        del data["rules"]["Preflop"][0]["hole_categories"]
        restored = Genome.from_dict(data, rng)
        assert restored.preflop_hole_category_mask.shape == (strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES)
        captured = capsys.readouterr()
        assert "hole_categories" in captured.out

    def test_from_dict_handles_missing_bucket_noise(self, capsys):
        rng = np.random.default_rng(5)
        g = Genome.random(rng)
        data = g.to_dict()
        del data["bucket_noise_std"]
        restored = Genome.from_dict(data, rng)
        assert restored.bucket_noise_std >= 0
        captured = capsys.readouterr()
        assert "bucket_noise_std" in captured.out

    def test_from_dict_handles_missing_and_unknown_gto_flags(self, capsys):
        rng = np.random.default_rng(6)
        g = Genome.random(rng)
        data = g.to_dict()
        first_key = next(iter(data["gto_flags"]))
        del data["gto_flags"][first_key]
        data["gto_flags"]["not_a_real_spot"] = 1.0
        restored = Genome.from_dict(data, rng)
        assert restored.gto_flags.shape == (NUM_GTO_SPOTS,)

    def test_save_and_load_round_trip(self, tmp_path):
        rng = np.random.default_rng(7)
        g = Genome.random(rng)
        path = tmp_path / "genome.json"
        g.save(str(path))
        loaded = Genome.load(str(path), rng)
        assert np.array_equal(g.condition_features, loaded.condition_features)
        assert np.array_equal(g.rule_actions, loaded.rule_actions)
        assert np.array_equal(g.preflop_hole_category_mask, loaded.preflop_hole_category_mask)

    def test_save_population_and_load_population_round_trip(self, tmp_path):
        rng = np.random.default_rng(8)
        genomes = [Genome.random(rng) for _ in range(3)]
        path = tmp_path / "population.json"
        save_population(genomes, str(path))
        loaded = load_population(str(path), rng)
        assert len(loaded) == 3
        for original, restored in zip(genomes, loaded):
            assert np.array_equal(original.condition_features, restored.condition_features)
