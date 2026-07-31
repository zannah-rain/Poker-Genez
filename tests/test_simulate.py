from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from game import GameConfig, HandStats
from genome import BET_RAISE, CHECK_CALL, Genome
from player import Player
from simulate import (
    GenerationStats, SimConfig, combine_generation_stats, run_generation,
    run_island_generation, run_session,
)


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


class TestRunSession:
    def test_result_has_expected_keys(self):
        players = make_random_players(3)
        config = GameConfig(max_hands_per_session=5)
        result = run_session(players, config, np.random.default_rng(0))
        assert set(result.keys()) == {"net", "hands_survived", "busted", "winner_id"}

    def test_net_covers_at_least_every_original_seat(self):
        players = make_random_players(3)
        config = GameConfig(max_hands_per_session=5)
        result = run_session(players, config, np.random.default_rng(0))
        assert set(p.player_id for p in players) <= set(result["net"].keys())

    def test_net_is_zero_sum_across_all_seats_and_stints(self):
        players = make_random_players(4)
        config = GameConfig(max_hands_per_session=10)
        result = run_session(players, config, np.random.default_rng(1))
        assert sum(result["net"].values()) == pytest.approx(0.0, abs=1e-6)

    def test_hands_survived_capped_at_session_hand_limit(self):
        players = make_random_players(3)
        config = GameConfig(max_hands_per_session=8)
        result = run_session(players, config, np.random.default_rng(2))
        for hands in result["hands_survived"].values():
            assert 0 <= hands <= 8

    def test_no_one_busts_when_stacks_are_deep_and_short_session(self):
        players = make_random_players(2)
        config = GameConfig(max_hands_per_session=1, starting_stack=10000.0)
        result = run_session(players, config, np.random.default_rng(3))
        assert all(not busted for busted in result["busted"].values())
        assert result["hands_survived"][players[0].player_id] == 1
        assert result["hands_survived"][players[1].player_id] == 1

    def test_winner_id_is_a_seat_that_played(self):
        players = make_random_players(3)
        config = GameConfig(max_hands_per_session=5)
        result = run_session(players, config, np.random.default_rng(4))
        assert result["winner_id"] in result["net"]

    def test_busted_seat_gets_refilled_instead_of_ending_session(self):
        # Seat 0 shoves every hand, seat 1 always calls -- with a short
        # stack for seat 0, a bust is essentially guaranteed within a few
        # hands, and the session should keep going (not stop early).
        shover = Player(player_id=0, genome=FixedGenome(BET_RAISE, bet_size=100000.0))
        caller = Player(player_id=1, genome=FixedGenome(CHECK_CALL))
        config = GameConfig(max_hands_per_session=15, starting_stack=20.0, small_blind=1.0, big_blind=2.0)
        result = run_session([shover, caller], config, np.random.default_rng(5))
        assert any(result["busted"].values())
        # The session should have played every hand regardless of busts.
        assert sum(result["hands_survived"].values()) >= 15

    def test_backfill_pool_defaults_to_table_players(self):
        players = make_random_players(2)
        config = GameConfig(max_hands_per_session=5)
        # No backfill_pool given -- should not raise, and only ever draw
        # replacements from table_players itself.
        result = run_session(players, config, np.random.default_rng(6))
        assert set(result["net"].keys()) <= {p.player_id for p in players}

    def test_explicit_backfill_pool_can_introduce_new_player_ids(self):
        table = make_random_players(2, seed=0)
        pool = table + make_random_players(2, seed=1)
        # Force busts quickly so a refill from the wider pool is likely.
        shover = Player(player_id=100, genome=FixedGenome(BET_RAISE, bet_size=100000.0))
        caller = Player(player_id=101, genome=FixedGenome(CHECK_CALL))
        pool = [shover, caller] + make_random_players(3, seed=2)
        config = GameConfig(max_hands_per_session=15, starting_stack=20.0)
        result = run_session([shover, caller], config, np.random.default_rng(7), backfill_pool=pool)
        assert set(result["net"].keys()) <= {p.player_id for p in pool}

    def test_hand_stats_are_accumulated_when_provided(self):
        players = make_random_players(3)
        config = GameConfig(max_hands_per_session=5)
        stats = HandStats()
        run_session(players, config, np.random.default_rng(8), stats=stats)
        assert sum(stats.action_counts.values()) > 0
        assert len(stats.raises_per_street) > 0


class TestGenerationStats:
    def test_defaults_are_zero(self):
        stats = GenerationStats()
        assert stats.mean_hands_survived == 0.0
        assert stats.bust_rate == 0.0
        assert stats.fold_rate_facing_bet == 0.0
        assert stats.mean_raises_per_street == 0.0

    def test_mean_hands_survived_computed_correctly(self):
        stats = GenerationStats(total_hands_survived=50, total_session_participations=10)
        assert stats.mean_hands_survived == 5.0

    def test_bust_rate_computed_correctly(self):
        stats = GenerationStats(total_busts=3, total_session_participations=12)
        assert stats.bust_rate == 0.25

    def test_fold_rate_facing_bet_uses_hand_stats(self):
        hand_stats = HandStats(facing_bet_decisions=20, facing_bet_folds=5)
        stats = GenerationStats(hand_stats=hand_stats)
        assert stats.fold_rate_facing_bet == 0.25

    def test_mean_raises_per_street(self):
        hand_stats = HandStats(raises_per_street=[0, 2, 4])
        stats = GenerationStats(hand_stats=hand_stats)
        assert stats.mean_raises_per_street == pytest.approx(2.0)


class TestRunGeneration:
    def test_fitness_dict_covers_every_player(self):
        players = make_random_players(6)
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        fitness, gen_stats = run_generation(players, config, sim_config, np.random.default_rng(0))
        assert set(fitness.keys()) == {p.player_id for p in players}

    def test_fitness_is_zero_sum_within_a_full_table_round(self):
        # A single round where the whole population fits at one table:
        # chips only move between these players, so total fitness nets to 0.
        players = make_random_players(6)
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        fitness, _ = run_generation(players, config, sim_config, np.random.default_rng(0))
        assert sum(fitness.values()) == pytest.approx(0.0, abs=1e-6)

    def test_generation_stats_are_populated(self):
        players = make_random_players(6)
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=2, table_size=6)
        _, gen_stats = run_generation(players, config, sim_config, np.random.default_rng(0))
        assert gen_stats.total_hands_survived > 0
        assert gen_stats.total_session_participations > 0
        assert sum(gen_stats.hand_stats.action_counts.values()) > 0

    def test_leftover_players_smaller_than_table_size_are_skipped(self):
        # 7 players, table_size 6 -> one full table + a leftover single
        # player who can't be seated (skipped, per "if len(table) < 2").
        players = make_random_players(7)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        fitness, _ = run_generation(players, config, sim_config, np.random.default_rng(0))
        # The leftover player still gets a fitness entry (initialized to 0),
        # just never actually plays this round.
        assert set(fitness.keys()) == {p.player_id for p in players}

    def test_progress_bar_is_written_to_stderr_by_default(self, capsys):
        players = make_random_players(6)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        run_generation(players, config, sim_config, np.random.default_rng(0))
        assert "generation eval" in capsys.readouterr().err

    def test_show_progress_false_suppresses_the_bar(self, capsys):
        players = make_random_players(6)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        run_generation(players, config, sim_config, np.random.default_rng(0), show_progress=False)
        assert capsys.readouterr().err == ""

    def test_custom_progress_desc_is_used(self, capsys):
        players = make_random_players(6)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        run_generation(
            players, config, sim_config, np.random.default_rng(0),
            progress_desc="island 2/3 eval",
        )
        assert "island 2/3 eval" in capsys.readouterr().err


class TestRunGenerationParallel:
    """num_workers > 1 runs tables across a real ProcessPoolExecutor -- these
    exercise the actual multiprocessing path (module-level worker function,
    pickled Player/GameConfig/rng args, merged per-table HandStats)."""

    def test_matches_sequential_shape_and_invariants(self):
        players = make_random_players(12)
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=2, table_size=6)
        fitness, gen_stats = run_generation(
            players, config, sim_config, np.random.default_rng(0),
            show_progress=False, num_workers=2,
        )
        assert set(fitness.keys()) == {p.player_id for p in players}
        assert sum(fitness.values()) == pytest.approx(0.0, abs=1e-6)
        assert gen_stats.total_hands_survived > 0
        assert gen_stats.total_session_participations > 0
        assert sum(gen_stats.hand_stats.action_counts.values()) > 0
        assert len(gen_stats.hand_stats.raises_per_street) > 0

    def test_leftover_players_are_skipped_in_parallel_mode_too(self):
        players = make_random_players(7)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        fitness, _ = run_generation(
            players, config, sim_config, np.random.default_rng(0),
            show_progress=False, num_workers=2,
        )
        assert set(fitness.keys()) == {p.player_id for p in players}

    def test_reuses_a_provided_executor_without_closing_it(self):
        players = make_random_players(12)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        with ProcessPoolExecutor(max_workers=2) as pool:
            run_generation(
                players, config, sim_config, np.random.default_rng(0),
                show_progress=False, num_workers=2, executor=pool,
            )
            # Still usable afterward -- run_generation must not shut down an
            # executor it didn't create itself.
            fitness2, _ = run_generation(
                players, config, sim_config, np.random.default_rng(1),
                show_progress=False, num_workers=2, executor=pool,
            )
            assert set(fitness2.keys()) == {p.player_id for p in players}


class TestCombineGenerationStats:
    def test_sums_scalar_totals(self):
        a = GenerationStats(total_hands_survived=10, total_session_participations=4, total_busts=1)
        b = GenerationStats(total_hands_survived=20, total_session_participations=6, total_busts=2)
        combined = combine_generation_stats([a, b])
        assert combined.total_hands_survived == 30
        assert combined.total_session_participations == 10
        assert combined.total_busts == 3

    def test_merges_hand_stats_action_counts(self):
        a = GenerationStats(hand_stats=HandStats(action_counts={0: 1, 1: 2, 2: 3}))
        b = GenerationStats(hand_stats=HandStats(action_counts={0: 4, 1: 0, 2: 1}))
        combined = combine_generation_stats([a, b])
        assert combined.hand_stats.action_counts == {0: 5, 1: 2, 2: 4}

    def test_merges_facing_bet_and_raises_per_street(self):
        a = GenerationStats(hand_stats=HandStats(facing_bet_decisions=5, facing_bet_folds=2, raises_per_street=[1, 2]))
        b = GenerationStats(hand_stats=HandStats(facing_bet_decisions=3, facing_bet_folds=1, raises_per_street=[3]))
        combined = combine_generation_stats([a, b])
        assert combined.hand_stats.facing_bet_decisions == 8
        assert combined.hand_stats.facing_bet_folds == 3
        assert combined.hand_stats.raises_per_street == [1, 2, 3]

    def test_empty_list_gives_zeroed_stats(self):
        combined = combine_generation_stats([])
        assert combined.total_hands_survived == 0
        assert combined.mean_hands_survived == 0.0


class TestRunIslandGeneration:
    @staticmethod
    def _islands(sizes, seed=0):
        islands = []
        for i, n in enumerate(sizes):
            offset = i * 1_000_000  # mirrors ga.py's ISLAND_ID_SPACING, avoids id collisions
            rng = np.random.default_rng(seed + i)
            islands.append([Player(player_id=offset + j, genome=Genome.random(rng)) for j in range(n)])
        return islands

    def test_self_mode_never_produces_cross_island_stats(self):
        islands = self._islands([6, 6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=2, table_size=6)
        _, gen_stats_by_island, cross_stats = run_island_generation(
            islands, config, sim_config, np.random.default_rng(0),
            interaction="self", show_progress=False,
        )
        assert cross_stats.total_hands_survived == 0
        for stats in gen_stats_by_island:
            assert stats.total_hands_survived > 0

    def test_inter_mode_leaves_per_island_stats_empty(self):
        islands = self._islands([6, 6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=2, table_size=6)
        _, gen_stats_by_island, cross_stats = run_island_generation(
            islands, config, sim_config, np.random.default_rng(0),
            interaction="inter", show_progress=False,
        )
        assert cross_stats.total_hands_survived > 0
        for stats in gen_stats_by_island:
            assert stats.total_hands_survived == 0

    def test_alternate_splits_rounds_between_both_modes(self):
        islands = self._islands([6, 6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=4, table_size=6)
        _, gen_stats_by_island, cross_stats = run_island_generation(
            islands, config, sim_config, np.random.default_rng(0),
            interaction="alternate", show_progress=False,
        )
        assert cross_stats.total_hands_survived > 0
        for stats in gen_stats_by_island:
            assert stats.total_hands_survived > 0

    def test_alternate_with_a_single_round_stays_self_play_only(self):
        # ceil(1 / 2) == 1 self round, 0 inter rounds -- alternate degrades
        # to self-play rather than silently running an inter round instead.
        islands = self._islands([6, 6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        _, gen_stats_by_island, cross_stats = run_island_generation(
            islands, config, sim_config, np.random.default_rng(0),
            interaction="alternate", show_progress=False,
        )
        assert cross_stats.total_hands_survived == 0
        for stats in gen_stats_by_island:
            assert stats.total_hands_survived > 0

    def test_single_island_ignores_interaction_setting(self):
        islands = self._islands([6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=3, table_size=6)
        _, gen_stats_by_island, cross_stats = run_island_generation(
            islands, config, sim_config, np.random.default_rng(0),
            interaction="inter", show_progress=False,
        )
        assert cross_stats.total_hands_survived == 0
        assert gen_stats_by_island[0].total_hands_survived > 0

    def test_fitness_split_by_home_island_and_zero_sum_overall(self):
        islands = self._islands([6, 6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=2, table_size=6)
        fitness_by_island, _, _ = run_island_generation(
            islands, config, sim_config, np.random.default_rng(0),
            interaction="alternate", show_progress=False,
        )
        for island_players, fitness in zip(islands, fitness_by_island):
            assert set(fitness.keys()) == {p.player_id for p in island_players}
        total = sum(v for fitness in fitness_by_island for v in fitness.values())
        assert total == pytest.approx(0.0, abs=1e-6)

    def test_unknown_interaction_mode_raises(self):
        islands = self._islands([6, 6])
        config = GameConfig(max_hands_per_session=5)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        with pytest.raises(ValueError):
            run_island_generation(
                islands, config, sim_config, np.random.default_rng(0),
                interaction="bogus", show_progress=False,
            )
