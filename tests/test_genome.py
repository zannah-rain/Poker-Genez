import numpy as np
import pytest

import gto as gto_module
import rules
import strategy
from cards import Card
from features import Situation
from genome import BET_RAISE, CHECK_CALL, FOLD, Genome, load_population, save_population
from gto import GTOSpot, NUM_GTO_SPOTS, SpotMatcher
from rule_helpers import condition, genome_with_condition, genome_with_default_action, make_genome, make_rule


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


class TestGenomeRandom:
    def test_gene_shapes(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert len(g.rules_by_street) == strategy.NUM_STREETS
        assert len(g.gto_rules) == NUM_GTO_SPOTS

    def test_rule_counts_within_bounds(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        for street_rules in g.rules_by_street:
            assert rules.MIN_RULES_PER_STREET <= len(street_rules) <= rules.MAX_RULES_PER_STREET

    def test_condition_counts_within_bounds(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        for street_rules in g.rules_by_street:
            for rule in street_rules:
                assert rules.MIN_CONDITIONS_PER_RULE <= len(rule.conditions) <= rules.MAX_CONDITIONS_PER_RULE

    def test_priorities_in_0_1_and_sorted_descending(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        for street_rules in g.rules_by_street:
            priorities = [rule.priority for rule in street_rules]
            assert all(0.0 <= p <= 1.0 for p in priorities)
            assert priorities == sorted(priorities, reverse=True)

    def test_preflop_never_references_ineligible_features(self):
        g = Genome.random(np.random.default_rng(0))
        eligible = set(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP].tolist())
        for rule in g.rules_by_street[strategy.PREFLOP]:
            for cond in rule.conditions:
                assert cond.feature_index in eligible

    def test_preflop_hole_category_mask_only_on_preflop(self):
        g = Genome.random(np.random.default_rng(0))
        for rule in g.rules_by_street[strategy.PREFLOP]:
            assert rule.hole_category_mask is not None
            assert len(rule.hole_category_mask) == strategy.NUM_HOLE_CATEGORIES
        for street in (1, 2, 3):
            for rule in g.rules_by_street[street]:
                assert rule.hole_category_mask is None

    def test_rule_actions_in_valid_range(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        for street_rules in g.rules_by_street:
            for rule in street_rules:
                primary, mix = rule.action.to_indices()
                assert 0 <= primary < strategy.NUM_ACTION_CATEGORIES
                assert mix == strategy.NO_MIX or 0 <= mix < strategy.NUM_ACTION_CATEGORIES

    def test_gto_rules_start_mostly_inactive(self):
        rng = np.random.default_rng(0)
        flags = [r.active for _ in range(20) for r in Genome.random(rng).gto_rules]
        assert np.mean(flags) < 0.3

    def test_lower_scale_means_fewer_active_conditions_on_average(self):
        rng = np.random.default_rng(0)
        sparse = np.mean([Genome.random(rng, scale=0.05).nonzero_weight_count() for _ in range(30)])
        dense = np.mean([Genome.random(rng, scale=0.5).nonzero_weight_count() for _ in range(30)])
        assert sparse < dense


class TestGenomeCopy:
    def test_copy_is_a_new_instance_sharing_immutable_state(self):
        g = make_genome()
        g2 = g.copy()
        assert g2 is not g
        assert g2.rules_by_street is g.rules_by_street
        assert g2.gto_rules is g.gto_rules


class TestNonzeroWeightCount:
    def test_counts_conditions_plus_rules(self):
        rule_a = make_rule(conditions=[condition("facing_bet", 1)], street=strategy.PREFLOP)
        rule_b = make_rule(
            conditions=[condition("facing_bet", 1), condition("is_aggressor", 1)], priority=0.1, street=strategy.PREFLOP,
        )
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule_a, rule_b)})
        # Preflop: 2 rules, 1+2=3 conditions. Flop/Turn/River default to 1 rule, 1 condition each -> 3 more rules, 3 conditions.
        assert g.nonzero_weight_count() == (2 + 3) + (3 + 3)

    def test_minimal_genome_counts_one_rule_one_condition_per_street(self):
        g = make_genome()
        assert g.nonzero_weight_count() == strategy.NUM_STREETS * (1 + 1)

    def test_does_not_count_gto_rules(self):
        g = make_genome(gto_active={gto_module.GTO_SPOTS[0].key: True})
        without_gto = make_genome()
        assert g.nonzero_weight_count() == without_gto.nonzero_weight_count()


class TestActiveGtoSpots:
    def test_returns_only_flagged_spots_in_catalog_order(self):
        active_keys = {gto_module.GTO_SPOTS[1].key: True, gto_module.GTO_SPOTS[3].key: True}
        g = make_genome(gto_active=active_keys)
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

    def test_higher_priority_rule_wins_regardless_of_construction_order(self):
        high = make_rule(strategy.ACTION_ALLIN, priority=0.9)
        low = make_rule(strategy.ACTION_CALL, priority=0.1)
        # Passed low-then-high -- make_genome must sort by priority itself,
        # decide() doesn't re-sort at decision time.
        g = make_genome(rules_by_street={strategy.PREFLOP: (low, high)})
        situation = make_situation()
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE  # ALLIN (high priority), not CALL (low priority)

    def test_non_matching_higher_priority_rule_falls_through_to_lower_priority_match(self):
        high = make_rule(
            strategy.ACTION_ALLIN, conditions=[condition("facing_bet", 1)], priority=0.9,
        )
        low = make_rule(strategy.ACTION_CALL, priority=0.1)  # tautological -- always matches
        g = make_genome(rules_by_street={strategy.PREFLOP: (high, low)})
        situation = make_situation(call_amount=0.0)  # facing_bet bucket 0 -- high priority rule doesn't match
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL  # falls through to the lower-priority always-matching rule

    def test_multiple_conditions_are_a_conjunction(self):
        rule = make_rule(
            strategy.ACTION_RAISE_75,
            conditions=[condition("facing_bet", 1), condition("is_aggressor", 1)],
        )
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})

        both_true = make_situation(call_amount=5.0, is_aggressor=True)
        only_one_true = make_situation(call_amount=5.0, is_aggressor=False)
        action_both, _ = g.decide(both_true, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        action_one, _ = g.decide(only_one_true, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action_both == BET_RAISE
        assert action_one == FOLD


class TestDecideStreetRouting:
    def test_each_street_only_checks_its_own_pool(self):
        rules_by_street = {
            0: (make_rule(strategy.ACTION_RAISE_25, street=0),),
            1: (make_rule(strategy.ACTION_RAISE_50, street=1),),
            2: (make_rule(strategy.ACTION_RAISE_75, street=2),),
            3: (make_rule(strategy.ACTION_ALLIN, street=3),),
        }
        g = make_genome(rules_by_street=rules_by_street)
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
        assert action == FOLD  # flop pool defaults to Fold, unaffected by the preflop rule


class TestDecidePreflopHoleCategory:
    def _situation_for_category(self, label: str, **overrides):
        hole_by_category = {
            "Premium Pairs": [Card.from_str("Ah"), Card.from_str("Ad")],
            "Junk": [Card.from_str("7h"), Card.from_str("2d")],
            "Axs": [Card.from_str("Ah"), Card.from_str("4h")],
        }
        return make_situation(hole=hole_by_category[label], **overrides)

    def test_rule_only_fires_for_claimed_categories(self):
        rule = make_rule(strategy.ACTION_ALLIN, hole_categories=["Premium Pairs"])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        premium = self._situation_for_category("Premium Pairs")
        junk = self._situation_for_category("Junk")
        action_premium, _ = g.decide(premium, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        action_junk, _ = g.decide(junk, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action_premium == BET_RAISE
        assert action_junk == FOLD  # falls through to decide()'s own no-match default

    def test_multiple_claimed_categories_all_match(self):
        rule = make_rule(strategy.ACTION_ALLIN, hole_categories=["Premium Pairs", "Axs"])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        for label in ("Premium Pairs", "Axs"):
            situation = self._situation_for_category(label)
            action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
            assert action == BET_RAISE
        junk_action, _ = g.decide(self._situation_for_category("Junk"), legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert junk_action == FOLD

    def test_empty_claimed_set_never_matches_any_hand(self):
        rule = make_rule(strategy.ACTION_ALLIN, hole_categories=[])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        for label in ("Premium Pairs", "Junk", "Axs"):
            situation = self._situation_for_category(label)
            action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
            assert action == FOLD

    def test_hole_category_check_is_exact_not_bucketed(self):
        rule = make_rule(strategy.ACTION_ALLIN, hole_categories=["Premium Pairs"])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        rng = np.random.default_rng(0)
        situation = self._situation_for_category("Junk")
        outcomes = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)[0] for _ in range(100)}
        assert outcomes == {FOLD}


class TestDecideMixActions:
    def test_no_mix_is_deterministic(self):
        g = genome_with_default_action(strategy.ACTION_RAISE_75)
        situation = make_situation()
        rng = np.random.default_rng(0)
        a1 = g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)
        a2 = g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)
        assert a1 == a2

    def test_deterministic_without_rng_even_when_mix(self):
        rule = make_rule(strategy.ACTION_RAISE_75, mix_index=strategy.ACTION_FOLD)
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        situation = make_situation()
        results = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE]) for _ in range(5)}
        assert len(results) == 1

    def test_mix_with_rng_plays_both_actions_over_many_decisions(self):
        rule = make_rule(strategy.ACTION_ALLIN, mix_index=strategy.ACTION_FOLD)
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        situation = make_situation()
        rng = np.random.default_rng(0)
        outcomes = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)[0] for _ in range(200)}
        assert BET_RAISE in outcomes
        assert FOLD in outcomes

    def test_mix_does_not_affect_the_mandatory_hole_category_check(self):
        # A rule that doesn't claim a hand's hole category must still never
        # fire, regardless of whether it's a Mix.
        rule = make_rule(strategy.ACTION_ALLIN, mix_index=strategy.ACTION_CALL, hole_categories=["Premium Pairs"])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        rng = np.random.default_rng(0)
        junk = make_situation(hole=[Card.from_str("7h"), Card.from_str("2d")])
        outcomes = {g.decide(junk, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)[0] for _ in range(50)}
        assert outcomes == {FOLD}


class TestDecideGtoOverride:
    _SPOT_KEY = "bb_vs_utg_open_100bb"

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
        # Rule list would otherwise always fold everything.
        g = genome_with_default_action(strategy.ACTION_FOLD, gto_active={self._SPOT_KEY: True})
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(12.0 * 2.0 - 2.0)

    def test_chart_call(self):
        g = genome_with_default_action(strategy.ACTION_FOLD, gto_active={self._SPOT_KEY: True})
        situation = self._matching_situation([Card.from_str("8c"), Card.from_str("8d")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_chart_default_fold(self):
        # Rule list would otherwise always raise -- chart's default should still win.
        g = genome_with_default_action(strategy.ACTION_ALLIN, gto_active={self._SPOT_KEY: True})
        situation = self._matching_situation([Card.from_str("7c"), Card.from_str("2d")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0

    def test_inactive_flag_falls_through_to_rule_list(self):
        g = genome_with_default_action(strategy.ACTION_ALLIN)  # all gto_rules inactive
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE

    def test_chart_raise_falls_back_to_check_call_if_bet_raise_illegal(self):
        g = make_genome(gto_active={self._SPOT_KEY: True})
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, bet_size = g.decide(situation, legal_actions=[CHECK_CALL])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_non_matching_situation_ignores_chart(self):
        g = genome_with_default_action(strategy.ACTION_ALLIN, gto_active={self._SPOT_KEY: True})
        # street=1 (flop) doesn't match this spot's street=0 requirement.
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        situation.street = 1
        situation.board = [Card.from_str("2c"), Card.from_str("5d"), Card.from_str("9h")]
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE  # falls through to the rule list, which also raises here


class TestDecideGtoSizeSpecs:
    """Exercises the pot-fraction and all-in raise-size branches via a
    synthetic spot constructed directly (not from the real GTO_SPOTS
    catalog, which only currently uses bb-sized opens) -- Genome no longer
    imports gto.GTO_SPOTS directly at decide() time, so this passes an
    explicit gto_rules tuple instead of monkeypatching a module-level list."""

    def _fake_spot(self, action_token: str) -> GTOSpot:
        return GTOSpot(
            key="fake_spot",
            label="Fake Spot",
            matcher=SpotMatcher(),  # matches everything
            action_ranges=((action_token, "AA-22, AKs-32s, AKo-32o"),),
            default_action="fold",
        )

    def test_pot_fraction_raise_size(self):
        g = make_genome(gto_rules=(rules.GTOSpotRule(self._fake_spot("raise_75"), True),))
        situation = make_situation(pot=40.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(0.75 * 40.0)

    def test_allin_raise_shoves_full_stack(self):
        g = make_genome(gto_rules=(rules.GTOSpotRule(self._fake_spot("allin"), True),))
        situation = make_situation(my_stack=157.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(157.0)

    def test_fold_action(self):
        g = make_genome(gto_rules=(rules.GTOSpotRule(self._fake_spot("fold"), True),))
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0


class TestGenomeMutate:
    def test_mutate_returns_new_object(self):
        rng = np.random.default_rng(1)
        g = make_genome()
        mutated = g.mutate(rng, rate=1.0, continuous_scale=0.3)
        assert mutated is not g

    def test_zero_rate_leaves_genes_unchanged(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(2))
        mutated = g.mutate(rng, rate=0.0, continuous_scale=0.3)
        assert mutated.rules_by_street == g.rules_by_street
        assert mutated.gto_rules == g.gto_rules

    def test_original_genome_is_unmodified(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(3))
        before = g.rules_by_street
        g.mutate(rng, rate=1.0, continuous_scale=0.3)
        assert g.rules_by_street is before

    def test_full_rate_mutation_preserves_per_street_eligibility(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(4))
        for _ in range(20):
            g = g.mutate(rng, rate=1.0, continuous_scale=0.3)
        eligible = set(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[strategy.PREFLOP].tolist())
        for rule in g.rules_by_street[strategy.PREFLOP]:
            for cond in rule.conditions:
                assert cond.feature_index in eligible

    def test_full_rate_mutation_keeps_rule_and_condition_counts_within_bounds(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(4))
        for _ in range(20):
            g = g.mutate(rng, rate=1.0, continuous_scale=0.3)
            for street_rules in g.rules_by_street:
                assert rules.MIN_RULES_PER_STREET <= len(street_rules) <= rules.MAX_RULES_PER_STREET
                for rule in street_rules:
                    assert rules.MIN_CONDITIONS_PER_RULE <= len(rule.conditions) <= rules.MAX_CONDITIONS_PER_RULE

    def test_repeated_mutation_can_grow_and_shrink_rule_counts(self):
        rng = np.random.default_rng(2)
        g = make_genome()  # starts at exactly 1 rule per street (the MIN)
        counts_seen = set()
        for _ in range(60):
            g = g.mutate(rng, rate=1.0, continuous_scale=0.3)
            counts_seen.add(len(g.rules_by_street[strategy.PREFLOP]))
        assert len(counts_seen) > 1

    def test_mutated_rules_stay_sorted_by_priority(self):
        rng = np.random.default_rng(1)
        g = Genome.random(np.random.default_rng(4))
        mutated = g.mutate(rng, rate=1.0, continuous_scale=0.3)
        for street_rules in mutated.rules_by_street:
            priorities = [rule.priority for rule in street_rules]
            assert priorities == sorted(priorities, reverse=True)


class TestGenomeCrossover:
    def test_crossover_returns_new_object(self):
        rng = np.random.default_rng(2)
        a = Genome.random(np.random.default_rng(5))
        b = Genome.random(np.random.default_rng(6))
        child = a.crossover(b, rng)
        assert child is not a and child is not b

    def test_child_rules_come_from_one_parent_or_the_other(self):
        rng = np.random.default_rng(2)
        a = Genome.random(np.random.default_rng(5))
        b = Genome.random(np.random.default_rng(6))
        child = a.crossover(b, rng)
        a_rules = set(a.rules_by_street[0]) | set(b.rules_by_street[0])
        for rule in child.rules_by_street[0]:
            assert rule in a_rules

    def test_child_rule_counts_stay_within_bounds(self):
        rng = np.random.default_rng(2)
        a = Genome.random(np.random.default_rng(5))
        b = Genome.random(np.random.default_rng(6))
        for _ in range(20):
            child = a.crossover(b, rng)
            for street_rules in child.rules_by_street:
                assert rules.MIN_RULES_PER_STREET <= len(street_rules) <= rules.MAX_RULES_PER_STREET

    def test_child_rules_stay_sorted_by_priority(self):
        rng = np.random.default_rng(2)
        a = Genome.random(np.random.default_rng(5))
        b = Genome.random(np.random.default_rng(6))
        child = a.crossover(b, rng)
        for street_rules in child.rules_by_street:
            priorities = [rule.priority for rule in street_rules]
            assert priorities == sorted(priorities, reverse=True)

    def test_gto_rules_each_independently_come_from_one_parent(self):
        rng = np.random.default_rng(2)
        a_rules = tuple(rules.GTOSpotRule(spot, True) for spot in gto_module.GTO_SPOTS)
        b_rules = tuple(rules.GTOSpotRule(spot, False) for spot in gto_module.GTO_SPOTS)
        a = make_genome(gto_rules=a_rules)
        b = make_genome(gto_rules=b_rules)
        for _ in range(20):
            child = a.crossover(b, rng)
            assert all(r.active in (True, False) for r in child.gto_rules)
        # Over many trials, both parents' activations should show up.
        outcomes = {child.gto_rules[0].active for child in (a.crossover(b, rng) for _ in range(30))}
        assert outcomes == {True, False}


class TestSerialization:
    def test_to_dict_from_dict_round_trip_preserves_values(self):
        rng = np.random.default_rng(3)
        g = Genome.random(rng)
        data = g.to_dict()
        restored = Genome.from_dict(data, rng)
        assert restored.rules_by_street == g.rules_by_street
        assert restored.gto_rules == g.gto_rules

    def test_to_dict_splits_rules_by_street_label(self):
        g = Genome.random(np.random.default_rng(3))
        data = g.to_dict()
        assert set(data["rules"].keys()) == set(strategy.STREET_LABELS)
        for street, label in enumerate(strategy.STREET_LABELS):
            assert len(data["rules"][label]) == len(g.rules_by_street[street])

    def test_preflop_rules_carry_hole_categories_others_dont(self):
        g = Genome.random(np.random.default_rng(3))
        data = g.to_dict()
        for rule in data["rules"]["Preflop"]:
            assert "hole_categories" in rule
        for label in ("Flop", "Turn", "River"):
            for rule in data["rules"][label]:
                assert "hole_categories" not in rule

    def test_to_dict_includes_mix_action_field(self):
        rule = make_rule(strategy.ACTION_FOLD, mix_index=strategy.ACTION_ALLIN)
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        data = g.to_dict()
        assert data["rules"]["Preflop"][0]["mix_action"] == strategy.ACTION_CATEGORIES[strategy.ACTION_ALLIN]

    def test_to_dict_includes_condition_bucketing_inline(self):
        rule = make_rule(strategy.ACTION_FOLD, conditions=[condition("facing_bet", 1)])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        data = g.to_dict()
        cond = data["rules"]["Preflop"][0]["conditions"][0]
        assert cond["feature"] == "facing_bet"
        assert cond["bucket"] == 1
        assert cond["num_buckets"] == 2

    def test_from_dict_drops_unknown_rule_feature(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"]["Preflop"][0]["conditions"][0]["feature"] = "not_a_real_feature"
        restored = Genome.from_dict(data, rng)
        captured = capsys.readouterr()
        assert "unknown" in captured.out.lower()
        # The dropped condition gets a fresh random replacement rather than
        # leaving the rule with fewer than MIN_CONDITIONS_PER_RULE.
        assert len(restored.rules_by_street[strategy.PREFLOP][0].conditions) >= rules.MIN_CONDITIONS_PER_RULE

    def test_from_dict_drops_ineligible_feature_referenced_by_preflop_rule(self, capsys):
        rng = np.random.default_rng(4)
        ineligible_feature = strategy.CONDITION_FEATURES[next(iter(strategy._PREFLOP_EXCLUDED_INDICES))].key
        rule = make_rule(strategy.ACTION_FOLD, conditions=[condition("facing_bet", 1)])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        data = g.to_dict()
        data["rules"]["Preflop"][0]["conditions"][0]["feature"] = ineligible_feature
        restored = Genome.from_dict(data, rng)
        captured = capsys.readouterr()
        assert "aren't allowed there" in captured.out.lower()
        assert len(restored.rules_by_street[strategy.PREFLOP][0].conditions) >= rules.MIN_CONDITIONS_PER_RULE

    def test_from_dict_repairs_a_rule_with_no_saved_conditions(self, capsys):
        rng = np.random.default_rng(4)
        rule = make_rule(strategy.ACTION_FOLD, conditions=[condition("facing_bet", 1)])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        data = g.to_dict()
        data["rules"]["Preflop"][0]["conditions"] = []
        restored = Genome.from_dict(data, rng)
        assert len(restored.rules_by_street[strategy.PREFLOP][0].conditions) >= rules.MIN_CONDITIONS_PER_RULE

    def test_from_dict_clamps_rule_count_above_max(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        one_rule = data["rules"]["Flop"][0]
        data["rules"]["Flop"] = [one_rule] * (rules.MAX_RULES_PER_STREET + 5)
        restored = Genome.from_dict(data, rng)
        assert len(restored.rules_by_street[1]) == rules.MAX_RULES_PER_STREET
        captured = capsys.readouterr()
        assert "flop" in captured.out.lower()

    def test_from_dict_pads_rule_count_below_min(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"]["River"] = []
        restored = Genome.from_dict(data, rng)
        assert len(restored.rules_by_street[3]) >= rules.MIN_RULES_PER_STREET
        captured = capsys.readouterr()
        assert "river" in captured.out.lower()

    def test_from_dict_handles_missing_street_entirely(self):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        del data["rules"]["Turn"]
        restored = Genome.from_dict(data, rng)
        assert len(restored.rules_by_street) == strategy.NUM_STREETS
        assert len(restored.rules_by_street[2]) >= rules.MIN_RULES_PER_STREET

    def test_from_dict_treats_old_flat_rule_list_as_no_saved_rules(self):
        # A save from before rules were split per street used a flat list --
        # there's no sane way to map those onto the new per-street pools.
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        data["rules"] = [{"conditions": [], "action": "Fold"}]
        restored = Genome.from_dict(data, rng)
        assert len(restored.rules_by_street) == strategy.NUM_STREETS

    def test_from_dict_drops_unknown_hole_category_label(self, capsys):
        rng = np.random.default_rng(4)
        rule = make_rule(strategy.ACTION_FOLD, hole_categories=["Premium Pairs"])
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        data = g.to_dict()
        data["rules"]["Preflop"][0]["hole_categories"] = ["Not A Real Category"]
        restored = Genome.from_dict(data, rng)
        assert restored.rules_by_street[strategy.PREFLOP][0].hole_category_mask == (False,) * strategy.NUM_HOLE_CATEGORIES
        captured = capsys.readouterr()
        assert "hole hand category" in captured.out.lower()

    def test_from_dict_handles_preflop_rule_missing_hole_categories(self, capsys):
        rng = np.random.default_rng(4)
        rule = make_rule(strategy.ACTION_FOLD)
        g = make_genome(rules_by_street={strategy.PREFLOP: (rule,)})
        data = g.to_dict()
        del data["rules"]["Preflop"][0]["hole_categories"]
        restored = Genome.from_dict(data, rng)
        assert restored.rules_by_street[strategy.PREFLOP][0].hole_category_mask is not None
        captured = capsys.readouterr()
        assert "hole_categories" in captured.out

    def test_from_dict_handles_missing_and_unknown_gto_flags(self):
        rng = np.random.default_rng(6)
        g = Genome.random(rng)
        data = g.to_dict()
        first_key = next(iter(data["gto_flags"]))
        del data["gto_flags"][first_key]
        data["gto_flags"]["not_a_real_spot"] = 1.0
        restored = Genome.from_dict(data, rng)
        assert len(restored.gto_rules) == NUM_GTO_SPOTS

    def test_save_and_load_round_trip(self, tmp_path):
        rng = np.random.default_rng(7)
        g = Genome.random(rng)
        path = tmp_path / "genome.json"
        g.save(str(path))
        loaded = Genome.load(str(path), rng)
        assert loaded.rules_by_street == g.rules_by_street
        assert loaded.gto_rules == g.gto_rules

    def test_save_population_and_load_population_round_trip(self, tmp_path):
        rng = np.random.default_rng(8)
        genomes = [Genome.random(rng) for _ in range(3)]
        path = tmp_path / "population.json"
        save_population(genomes, str(path))
        loaded = load_population(str(path), rng)
        assert len(loaded) == 3
        for original, restored in zip(genomes, loaded):
            assert restored.rules_by_street == original.rules_by_street
