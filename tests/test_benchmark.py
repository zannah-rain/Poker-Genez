import numpy as np
import pytest

from benchmark import SEATS_PER_SIDE, BenchmarkResult, run_benchmark
from game import GameConfig
from genome import Genome
from player import Player


def make_random_players(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Player(player_id=i, genome=Genome.random(rng)) for i in range(n)]


class TestBenchmarkResult:
    def test_defaults_are_zero(self):
        result = BenchmarkResult()
        assert result.current_net_total == 0.0
        assert result.checkpoint_net_total == 0.0

    def test_bb_per_100_zero_hands_is_zero(self):
        result = BenchmarkResult()
        assert result.bb_per_100("current", 2.0) == 0.0
        assert result.bb_per_100("checkpoint", 2.0) == 0.0

    def test_bb_per_100_formula_for_current_side(self):
        result = BenchmarkResult(current_net_total=100.0, current_hands_total=50)
        assert result.bb_per_100("current", 2.0) == pytest.approx((100.0 / 2.0) / 50 * 100.0)

    def test_bb_per_100_formula_for_checkpoint_side(self):
        result = BenchmarkResult(checkpoint_net_total=-40.0, checkpoint_hands_total=20)
        assert result.bb_per_100("checkpoint", 2.0) == pytest.approx((-40.0 / 2.0) / 20 * 100.0)


class TestRunBenchmark:
    def test_hands_are_tracked_per_side(self):
        current_pool = make_random_players(4, seed=0)
        checkpoint_pool = make_random_players(4, seed=1)
        config = GameConfig(max_hands_per_session=5)
        result = run_benchmark(current_pool, checkpoint_pool, config, np.random.default_rng(0), num_tables=2)
        expected_hands = 2 * config.max_hands_per_session * SEATS_PER_SIDE
        assert result.current_hands_total == expected_hands
        assert result.checkpoint_hands_total == expected_hands

    def test_chips_are_zero_sum_across_all_tables(self):
        current_pool = make_random_players(4, seed=0)
        checkpoint_pool = make_random_players(4, seed=1)
        config = GameConfig(max_hands_per_session=5)
        result = run_benchmark(current_pool, checkpoint_pool, config, np.random.default_rng(1), num_tables=3)
        assert result.current_net_total + result.checkpoint_net_total == pytest.approx(0.0, abs=1e-6)

    def test_requires_at_least_seats_per_side_players_in_each_pool(self):
        current_pool = make_random_players(SEATS_PER_SIDE, seed=0)
        checkpoint_pool = make_random_players(SEATS_PER_SIDE, seed=1)
        config = GameConfig(max_hands_per_session=3)
        # Should run without error when pools are exactly SEATS_PER_SIDE large.
        result = run_benchmark(current_pool, checkpoint_pool, config, np.random.default_rng(2), num_tables=1)
        assert result.current_hands_total > 0

    def test_num_tables_zero_gives_empty_result(self):
        current_pool = make_random_players(4, seed=0)
        checkpoint_pool = make_random_players(4, seed=1)
        config = GameConfig(max_hands_per_session=3)
        result = run_benchmark(current_pool, checkpoint_pool, config, np.random.default_rng(3), num_tables=0)
        assert result.current_net_total == 0.0
        assert result.current_hands_total == 0
