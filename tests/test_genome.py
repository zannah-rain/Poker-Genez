import numpy as np
import pytest

import genome as genome_module
import gto as gto_module
from cards import Card
from features import NUM_FEATURES, Situation
from genome import (
    BET_RAISE, CHECK_CALL, FOLD, WEIGHT_ALPHABET,
    Genome, crossover_weights, load_population, mutate_bool_flags,
    mutate_weights, quantize, save_population,
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
    weights_v=None, weights_l=None, bias_v=50.0, bias_l=50.0,
    theta_value=70.0, theta_bluff=70.0, theta_call=40.0,
    kappa=0.5, noise_std=0.0, gto_flags=None,
) -> Genome:
    if weights_v is None:
        weights_v = np.zeros(NUM_FEATURES)
    if weights_l is None:
        weights_l = np.zeros(NUM_FEATURES)
    if gto_flags is None:
        gto_flags = np.zeros(NUM_GTO_SPOTS)
    return Genome(
        weights_v=weights_v, weights_l=weights_l, bias_v=bias_v, bias_l=bias_l,
        theta_value=theta_value, theta_bluff=theta_bluff, theta_call=theta_call,
        kappa=kappa, noise_std=noise_std, gto_flags=gto_flags,
    )


class TestQuantize:
    def test_snaps_to_nearest_alphabet_value(self):
        result = quantize(np.array([0.1, -0.1, 24.0, -24.0]))
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 20.0
        assert result[3] == -20.0

    def test_extreme_values_clamp_to_alphabet_bounds(self):
        result = quantize(np.array([1000.0, -1000.0]))
        assert result[0] == WEIGHT_ALPHABET.max()
        assert result[1] == WEIGHT_ALPHABET.min()

    def test_result_values_always_in_alphabet(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 50, size=200)
        result = quantize(values)
        assert all(v in WEIGHT_ALPHABET for v in result)


class TestMutateWeights:
    def test_zero_rate_never_mutates(self):
        rng = np.random.default_rng(0)
        weights = quantize(np.array([0.0, 10.0, -20.0, 30.0]))
        result = mutate_weights(weights, rate=0.0, rng=rng)
        assert np.array_equal(result, weights)

    def test_full_rate_keeps_values_in_alphabet(self):
        rng = np.random.default_rng(0)
        weights = quantize(np.zeros(50))
        result = mutate_weights(weights, rate=1.0, rng=rng)
        assert all(v in WEIGHT_ALPHABET for v in result)

    def test_full_rate_changes_most_values_over_a_large_sample(self):
        rng = np.random.default_rng(0)
        weights = quantize(np.zeros(500))
        result = mutate_weights(weights, rate=1.0, rng=rng)
        assert np.mean(result != weights) > 0.5


class TestCrossoverWeights:
    def test_each_gene_comes_from_one_parent(self):
        rng = np.random.default_rng(0)
        a = np.full(100, -10.0)
        b = np.full(100, 20.0)
        child = crossover_weights(a, b, rng)
        assert all(v in (-10.0, 20.0) for v in child)

    def test_both_parents_contribute_over_a_large_sample(self):
        rng = np.random.default_rng(0)
        a = np.full(200, -10.0)
        b = np.full(200, 20.0)
        child = crossover_weights(a, b, rng)
        assert np.any(child == -10.0)
        assert np.any(child == 20.0)


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


class TestGenomeRandom:
    def test_weight_shapes_and_alphabet_membership(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert g.weights_v.shape == (NUM_FEATURES,)
        assert g.weights_l.shape == (NUM_FEATURES,)
        assert all(v in WEIGHT_ALPHABET for v in g.weights_v)
        assert all(v in WEIGHT_ALPHABET for v in g.weights_l)

    def test_gto_flags_shape_and_values(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        assert g.gto_flags.shape == (NUM_GTO_SPOTS,)
        assert set(np.unique(g.gto_flags)) <= {0.0, 1.0}

    def test_gto_flags_start_mostly_off(self):
        rng = np.random.default_rng(0)
        # Aggregate over several genomes for a stable average.
        flags = np.concatenate([Genome.random(rng).gto_flags for _ in range(20)])
        assert flags.mean() < 0.3


class TestGenomeCopy:
    def test_copy_is_independent(self):
        g = make_genome(weights_v=quantize(np.zeros(NUM_FEATURES)))
        g2 = g.copy()
        g2.weights_v[0] = 30.0
        g2.gto_flags[0] = 1.0
        assert g.weights_v[0] != 30.0
        assert g.gto_flags[0] != 1.0

    def test_copy_preserves_scalars(self):
        g = make_genome(bias_v=12.0, theta_value=33.0)
        g2 = g.copy()
        assert g2.bias_v == 12.0
        assert g2.theta_value == 33.0


class TestNonzeroWeightCount:
    def test_counts_across_both_axes(self):
        wv = quantize(np.zeros(NUM_FEATURES))
        wl = quantize(np.zeros(NUM_FEATURES))
        wv[0] = 10.0
        wv[1] = -10.0
        wl[2] = 20.0
        g = make_genome(weights_v=wv, weights_l=wl)
        assert g.nonzero_weight_count() == 3

    def test_all_zero_counts_zero(self):
        g = make_genome()
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


class TestComputeVL:
    def test_linear_combination_before_clamp(self):
        weights_v = np.zeros(NUM_FEATURES)
        weights_v[0] = 2.0
        g = make_genome(weights_v=weights_v, bias_v=10.0, bias_l=0.0)
        features = np.zeros(NUM_FEATURES)
        features[0] = 5.0
        v, l = g.compute_v_l(features)
        assert v == pytest.approx(20.0)  # 2.0*5 + 10
        assert l == pytest.approx(0.0)

    def test_clamped_to_zero_and_hundred(self):
        g = make_genome(bias_v=-500.0, bias_l=500.0)
        v, l = g.compute_v_l(np.zeros(NUM_FEATURES))
        assert v == 0.0
        assert l == 100.0


class TestDecideLinearRule:
    def test_bets_when_value_term_clears_theta_value(self):
        g = make_genome(bias_v=80.0, bias_l=0.0, theta_value=70.0, theta_bluff=70.0, kappa=0.5)
        situation = make_situation(pot=100.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx((80.0 - 70.0) / 100.0 * 100.0)

    def test_falls_back_to_check_call_if_bet_raise_illegal(self):
        g = make_genome(bias_v=80.0, bias_l=0.0, theta_value=70.0)
        situation = make_situation()
        action, _ = g.decide(situation, legal_actions=[CHECK_CALL])
        assert action == CHECK_CALL

    def test_calls_when_action_score_negative_but_value_above_theta_call(self):
        g = make_genome(bias_v=50.0, bias_l=0.0, theta_value=70.0, theta_bluff=70.0, theta_call=40.0, kappa=0.5)
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_folds_when_value_below_theta_call(self):
        g = make_genome(bias_v=10.0, bias_l=0.0, theta_value=70.0, theta_bluff=70.0, theta_call=40.0, kappa=0.5)
        situation = make_situation()
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0

    def test_checks_instead_of_folding_when_fold_illegal(self):
        g = make_genome(bias_v=10.0, bias_l=0.0, theta_call=40.0)
        situation = make_situation()
        action, _ = g.decide(situation, legal_actions=[CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL

    def test_bluff_term_can_trigger_bet_raise_with_low_value(self):
        g = make_genome(bias_v=0.0, bias_l=90.0, theta_value=70.0, theta_bluff=70.0, kappa=0.0)
        situation = make_situation(pot=100.0)
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx((90.0 - 70.0) / 100.0 * 100.0)

    def test_kappa_suppresses_bluff_term_as_value_rises(self):
        # Same L, but a higher V should suppress the bluff term via kappa,
        # potentially preventing a bet that would otherwise fire.
        low_v = make_genome(bias_v=0.0, bias_l=90.0, theta_value=99.0, theta_bluff=70.0, kappa=2.0)
        high_v = make_genome(bias_v=50.0, bias_l=90.0, theta_value=99.0, theta_bluff=70.0, kappa=2.0)
        situation = make_situation()
        low_action, _ = low_v.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        high_action, _ = high_v.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        # low_v: A = max(0-99, 90-70-2*0) = 20 > 0 -> bet
        assert low_action == BET_RAISE
        # high_v: A = max(50-99, 90-70-2*50) = max(-49,-80) = -49 -> not bet; V=50>theta_call(40) -> call
        assert high_action == CHECK_CALL

    def test_deterministic_without_rng(self):
        g = make_genome(bias_v=80.0, theta_value=70.0)
        situation = make_situation()
        results = {g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE]) for _ in range(5)}
        assert len(results) == 1

    def test_zero_noise_std_deterministic_even_with_rng(self):
        g = make_genome(bias_v=80.0, theta_value=70.0, noise_std=0.0)
        situation = make_situation()
        rng = np.random.default_rng(0)
        a1 = g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)
        a2 = g.decide(situation, [FOLD, CHECK_CALL, BET_RAISE], rng=rng)
        assert a1 == a2


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

    def test_chart_raise_bypasses_linear_weights(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        # weights would otherwise fold everything (very negative bias)
        g = make_genome(bias_v=-1000.0, bias_l=-1000.0, gto_flags=flags)
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE
        assert bet_size == pytest.approx(12.0 * 2.0 - 2.0)

    def test_chart_call(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        g = make_genome(bias_v=-1000.0, bias_l=-1000.0, gto_flags=flags)
        situation = self._matching_situation([Card.from_str("8c"), Card.from_str("8d")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_chart_default_fold(self):
        idx = self._bb_vs_utg_index()
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[idx] = 1.0
        g = make_genome(bias_v=1000.0, bias_l=1000.0, gto_flags=flags)  # would bet if not for chart
        situation = self._matching_situation([Card.from_str("7c"), Card.from_str("2d")])
        action, bet_size = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == FOLD
        assert bet_size == 0.0

    def test_inactive_flag_falls_through_to_linear_rule(self):
        g = make_genome(bias_v=80.0, theta_value=70.0)  # all gto_flags off
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        # Should use the linear rule (bets for value), not the (unused) chart.
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
        g = make_genome(bias_v=80.0, theta_value=70.0, gto_flags=flags)
        # street=1 (flop) doesn't match this spot's street=0 requirement.
        situation = self._matching_situation([Card.from_str("Kc"), Card.from_str("Kd")])
        situation.street = 1
        situation.board = [Card.from_str("2c"), Card.from_str("5d"), Card.from_str("9h")]
        action, _ = g.decide(situation, legal_actions=[FOLD, CHECK_CALL, BET_RAISE])
        assert action == BET_RAISE  # falls through to the linear rule, which also bets here


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
    def test_mutate_returns_new_object_and_keeps_kappa_noise_nonnegative(self):
        rng = np.random.default_rng(1)
        g = make_genome(kappa=0.01, noise_std=0.01)
        mutated = g.mutate(rng, rate=1.0, continuous_scale=50.0)
        assert mutated is not g
        assert mutated.kappa >= 0
        assert mutated.noise_std >= 0

    def test_zero_rate_leaves_scalars_unchanged(self):
        rng = np.random.default_rng(1)
        g = make_genome(bias_v=11.0, theta_value=22.0)
        mutated = g.mutate(rng, rate=0.0, continuous_scale=10.0)
        assert mutated.bias_v == 11.0
        assert mutated.theta_value == 22.0

    def test_original_genome_is_unmodified(self):
        rng = np.random.default_rng(1)
        weights_v = quantize(np.zeros(NUM_FEATURES))
        g = make_genome(weights_v=weights_v.copy())
        g.mutate(rng, rate=1.0, continuous_scale=10.0)
        assert np.array_equal(g.weights_v, weights_v)


class TestGenomeCrossover:
    def test_crossover_returns_new_object(self):
        rng = np.random.default_rng(2)
        a = make_genome(bias_v=10.0)
        b = make_genome(bias_v=90.0)
        child = a.crossover(b, rng)
        assert child is not a and child is not b

    def test_blended_scalar_is_between_parents(self):
        rng = np.random.default_rng(2)
        a = make_genome(bias_v=10.0)
        b = make_genome(bias_v=90.0)
        for _ in range(20):
            child = a.crossover(b, rng)
            assert 10.0 <= child.bias_v <= 90.0

    def test_kappa_and_noise_stay_nonnegative(self):
        rng = np.random.default_rng(2)
        a = make_genome(kappa=-0.0001, noise_std=0.0001)
        b = make_genome(kappa=0.0001, noise_std=-0.0001)
        child = a.crossover(b, rng)
        assert child.kappa >= 0
        assert child.noise_std >= 0


class TestSerialization:
    def test_to_dict_from_dict_round_trip_preserves_values(self):
        rng = np.random.default_rng(3)
        g = Genome.random(rng)
        data = g.to_dict()
        restored = Genome.from_dict(data, rng)
        assert np.array_equal(g.weights_v, restored.weights_v)
        assert np.array_equal(g.weights_l, restored.weights_l)
        assert np.array_equal(g.gto_flags, restored.gto_flags)
        assert g.bias_v == restored.bias_v
        assert g.theta_value == restored.theta_value

    def test_from_dict_drops_unknown_feature_and_fills_missing(self, capsys):
        rng = np.random.default_rng(4)
        g = Genome.random(rng)
        data = g.to_dict()
        first_feature = next(iter(data["weights_v"]))
        del data["weights_v"][first_feature]
        data["weights_v"]["not_a_real_feature"] = 5.0
        restored = Genome.from_dict(data, rng)
        assert restored.weights_v.shape == (NUM_FEATURES,)
        captured = capsys.readouterr()
        assert "unknown feature" in captured.out.lower() or "unknown" in captured.out.lower()

    def test_from_dict_handles_missing_scalar(self, capsys):
        rng = np.random.default_rng(5)
        g = Genome.random(rng)
        data = g.to_dict()
        del data["scalars"]["kappa"]
        restored = Genome.from_dict(data, rng)
        assert restored.kappa >= 0
        captured = capsys.readouterr()
        assert "kappa" in captured.out

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
        assert np.array_equal(g.weights_v, loaded.weights_v)
        assert g.theta_call == loaded.theta_call

    def test_save_population_and_load_population_round_trip(self, tmp_path):
        rng = np.random.default_rng(8)
        genomes = [Genome.random(rng) for _ in range(3)]
        path = tmp_path / "population.json"
        save_population(genomes, str(path))
        loaded = load_population(str(path), rng)
        assert len(loaded) == 3
        for original, restored in zip(genomes, loaded):
            assert np.array_equal(original.weights_v, restored.weights_v)
