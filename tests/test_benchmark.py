import math

import numpy as np
import pytest

from benchmark import (
    SEATS_PER_SIDE, BenchmarkOutcome, _inverse_normal_cdf, confidence_interval,
    run_benchmark_until_resolved,
)
from game import GameConfig
from genome import BET_RAISE, CHECK_CALL, FOLD, Genome
from player import Player


class FixedGenome:
    def __init__(self, action, bet_size=0.0):
        self.action = action
        self.bet_size = bet_size

    def decide(self, situation, legal_actions, rng=None):
        action = self.action if self.action in legal_actions else CHECK_CALL
        return action, self.bet_size


def make_random_players(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Player(player_id=i, genome=Genome.random(rng)) for i in range(n)]


def make_fixed_players(n, action, id_offset=0, bet_size=0.0):
    return [Player(player_id=id_offset + i, genome=FixedGenome(action, bet_size)) for i in range(n)]


class TestInverseNormalCdf:
    def test_median_is_zero(self):
        assert _inverse_normal_cdf(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_matches_well_known_two_sided_95_percent_z_score(self):
        assert _inverse_normal_cdf(0.975) == pytest.approx(1.959964, abs=1e-4)

    def test_matches_well_known_two_sided_99_percent_z_score(self):
        assert _inverse_normal_cdf(0.995) == pytest.approx(2.575829, abs=1e-4)

    def test_symmetric_around_median(self):
        assert _inverse_normal_cdf(0.9) == pytest.approx(-_inverse_normal_cdf(0.1), abs=1e-6)

    def test_rejects_out_of_range_probabilities(self):
        with pytest.raises(ValueError):
            _inverse_normal_cdf(0.0)
        with pytest.raises(ValueError):
            _inverse_normal_cdf(1.0)
        with pytest.raises(ValueError):
            _inverse_normal_cdf(1.5)


class TestConfidenceInterval:
    def test_matches_manual_normal_approximation(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        se = std / math.sqrt(len(samples))
        z = _inverse_normal_cdf(0.975)
        expected = (mean - z * se, mean + z * se)
        assert confidence_interval(samples, p_value=0.05) == pytest.approx(expected)

    def test_narrower_p_value_gives_wider_interval(self):
        samples = list(range(1, 21))
        low_conf_lo, low_conf_hi = confidence_interval(samples, p_value=0.20)
        high_conf_lo, high_conf_hi = confidence_interval(samples, p_value=0.01)
        assert (high_conf_hi - high_conf_lo) > (low_conf_hi - low_conf_lo)

    def test_more_samples_narrows_the_interval_for_constant_spread(self):
        rng = np.random.default_rng(0)
        small = rng.normal(0, 1, size=10).tolist()
        large = small + rng.normal(0, 1, size=990).tolist()
        lo_small, hi_small = confidence_interval(small, p_value=0.05)
        lo_large, hi_large = confidence_interval(large, p_value=0.05)
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_requires_at_least_two_samples(self):
        with pytest.raises(ValueError):
            confidence_interval([1.0], p_value=0.05)


class TestRunBenchmarkUntilResolved:
    def test_rejects_too_small_min_tables(self):
        with pytest.raises(ValueError):
            run_benchmark_until_resolved(
                make_random_players(3), make_random_players(3), GameConfig(), np.random.default_rng(0),
                min_tables=1,
            )

    def test_rejects_nonpositive_table_batch(self):
        with pytest.raises(ValueError):
            run_benchmark_until_resolved(
                make_random_players(3), make_random_players(3), GameConfig(), np.random.default_rng(0),
                table_batch=0,
            )

    def test_resolves_as_improved_with_a_clear_structural_edge(self):
        # Current side always calls; checkpoint side folds to every real bet,
        # so checkpoint donates its blinds most hands -- a large, consistent
        # edge that should resolve well before the table cap.
        current = make_fixed_players(SEATS_PER_SIDE, CHECK_CALL, id_offset=0)
        checkpoint = make_fixed_players(SEATS_PER_SIDE, FOLD, id_offset=100)
        config = GameConfig(max_hands_per_session=20, starting_stack=200.0)
        outcome = run_benchmark_until_resolved(
            current, checkpoint, config, np.random.default_rng(0),
            min_tables=20, max_tables=200, table_batch=20, p_value=0.05,
        )
        assert isinstance(outcome, BenchmarkOutcome)
        assert outcome.resolved is True
        assert outcome.improved is True
        assert outcome.ci_low > 0.0
        assert outcome.hit_table_cap is False

    def test_resolves_as_regressed_when_roles_are_reversed(self):
        current = make_fixed_players(SEATS_PER_SIDE, FOLD, id_offset=0)
        checkpoint = make_fixed_players(SEATS_PER_SIDE, CHECK_CALL, id_offset=100)
        config = GameConfig(max_hands_per_session=20, starting_stack=200.0)
        outcome = run_benchmark_until_resolved(
            current, checkpoint, config, np.random.default_rng(1),
            min_tables=20, max_tables=200, table_batch=20, p_value=0.05,
        )
        assert outcome.resolved is True
        assert outcome.improved is False
        assert outcome.ci_high < 0.0

    def test_hits_table_cap_and_reports_not_improved_when_sides_are_identical(self):
        # Both sides play an identical (trivial) strategy -- the true edge is
        # exactly zero, so with a tight table cap the CI should still
        # straddle 0 when the cap is reached.
        current = make_fixed_players(SEATS_PER_SIDE, CHECK_CALL, id_offset=0)
        checkpoint = make_fixed_players(SEATS_PER_SIDE, CHECK_CALL, id_offset=100)
        config = GameConfig(max_hands_per_session=5, starting_stack=200.0)
        outcome = run_benchmark_until_resolved(
            current, checkpoint, config, np.random.default_rng(2),
            min_tables=5, max_tables=15, table_batch=5, p_value=0.05,
        )
        assert outcome.tables_played == 15
        assert outcome.hit_table_cap is True
        assert outcome.resolved is False
        assert outcome.improved is False

    def test_never_plays_fewer_than_min_tables(self):
        current = make_random_players(SEATS_PER_SIDE, seed=0)
        checkpoint = make_random_players(SEATS_PER_SIDE, seed=1)
        config = GameConfig(max_hands_per_session=3)
        outcome = run_benchmark_until_resolved(
            current, checkpoint, config, np.random.default_rng(3),
            min_tables=30, max_tables=30, table_batch=10,
        )
        assert outcome.tables_played >= 30

    def test_never_plays_more_than_max_tables(self):
        current = make_fixed_players(SEATS_PER_SIDE, CHECK_CALL, id_offset=0)
        checkpoint = make_fixed_players(SEATS_PER_SIDE, CHECK_CALL, id_offset=100)
        config = GameConfig(max_hands_per_session=5)
        outcome = run_benchmark_until_resolved(
            current, checkpoint, config, np.random.default_rng(4),
            min_tables=10, max_tables=10, table_batch=5,
        )
        assert outcome.tables_played == 10
