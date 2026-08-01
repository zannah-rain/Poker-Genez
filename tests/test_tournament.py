import os

import numpy as np
import pytest

import strategy
from game import GameConfig
from genome import Genome
from gto import NUM_GTO_SPOTS
from player import Player
from simulate import SimConfig
from tournament import (
    PlayerStats, describe_genome, export_top_n, rank_players,
    run_final_tournament,
)


def make_genome(condition_features=None, condition_buckets=None, rule_actions=None, gto_flags=None):
    """Defaults to an "empty" genome: every rule wildcard-conditioned to
    Fold, 2 buckets (cut at 0.5) for every bucketable feature."""
    if condition_features is None:
        condition_features = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
    if condition_buckets is None:
        condition_buckets = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
    if rule_actions is None:
        rule_actions = np.full(strategy.NUM_RULES, strategy.ACTION_FOLD)
    if gto_flags is None:
        gto_flags = np.zeros(NUM_GTO_SPOTS)
    return Genome(
        num_buckets=np.full(strategy.NUM_BUCKETABLE, 2),
        thresholds=np.full((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), 0.5),
        condition_features=condition_features, condition_buckets=condition_buckets,
        rule_actions=rule_actions, raise_size_idx=0, bucket_noise_std=1.0, gto_flags=gto_flags,
    )


def make_random_players(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Player(player_id=i, genome=Genome.random(rng), label=f"P{i}") for i in range(n)]


class TestPlayerStats:
    def test_defaults_avoid_division_by_zero(self):
        stats = PlayerStats(player_id=1)
        assert stats.mean_net_chips == 0.0
        assert stats.bust_rate == 0.0
        assert stats.win_rate == 0.0
        assert stats.mean_hands_survived == 0.0
        assert stats.std_net_chips() == 0.0
        assert stats.bb_per_100(2.0) == 0.0

    def test_mean_net_chips(self):
        stats = PlayerStats(player_id=1, sessions_played=4, total_net_chips=200.0)
        assert stats.mean_net_chips == 50.0

    def test_bust_rate(self):
        stats = PlayerStats(player_id=1, sessions_played=10, busts=3)
        assert stats.bust_rate == 0.3

    def test_win_rate(self):
        stats = PlayerStats(player_id=1, sessions_played=8, sessions_won=2)
        assert stats.win_rate == 0.25

    def test_mean_hands_survived(self):
        stats = PlayerStats(player_id=1, sessions_played=5, total_hands_survived=100)
        assert stats.mean_hands_survived == 20.0

    def test_std_net_chips_matches_numpy(self):
        results = [10.0, -5.0, 20.0, 0.0]
        stats = PlayerStats(player_id=1, session_results=results)
        assert stats.std_net_chips() == pytest.approx(float(np.std(results)))

    def test_bb_per_100_formula(self):
        stats = PlayerStats(player_id=1, total_net_chips=100.0, total_hands_survived=50)
        assert stats.bb_per_100(2.0) == pytest.approx((100.0 / 2.0) / 50 * 100.0)


class TestRunFinalTournament:
    def test_every_player_gets_stats(self):
        players = make_random_players(6)
        stats = run_final_tournament(
            players, GameConfig(max_hands_per_session=5),
            SimConfig(rounds_per_generation=1, table_size=6),
            np.random.default_rng(0),
        )
        assert set(stats.keys()) == {p.player_id for p in players}

    def test_sessions_played_accumulates_across_rounds(self):
        players = make_random_players(6)
        stats = run_final_tournament(
            players, GameConfig(max_hands_per_session=5),
            SimConfig(rounds_per_generation=3, table_size=6),
            np.random.default_rng(0),
        )
        for s in stats.values():
            assert s.sessions_played == 3

    def test_net_chips_are_tracked_per_session(self):
        players = make_random_players(6)
        stats = run_final_tournament(
            players, GameConfig(max_hands_per_session=5),
            SimConfig(rounds_per_generation=2, table_size=6),
            np.random.default_rng(0),
        )
        for s in stats.values():
            assert len(s.session_results) == 2

    def test_progress_bar_is_written_to_stderr_by_default(self, capsys):
        players = make_random_players(6)
        run_final_tournament(
            players, GameConfig(max_hands_per_session=3),
            SimConfig(rounds_per_generation=1, table_size=6),
            np.random.default_rng(0),
        )
        assert "final tournament" in capsys.readouterr().err

    def test_show_progress_false_suppresses_the_bar(self, capsys):
        players = make_random_players(6)
        run_final_tournament(
            players, GameConfig(max_hands_per_session=3),
            SimConfig(rounds_per_generation=1, table_size=6),
            np.random.default_rng(0), show_progress=False,
        )
        assert capsys.readouterr().err == ""

    def test_num_workers_greater_than_one_produces_equivalent_results(self):
        players = make_random_players(12)
        stats = run_final_tournament(
            players, GameConfig(max_hands_per_session=5),
            SimConfig(rounds_per_generation=2, table_size=6),
            np.random.default_rng(0), show_progress=False, num_workers=2,
        )
        assert set(stats.keys()) == {p.player_id for p in players}
        # Every player sits at their own table at least once per round; they
        # may additionally get drawn as a cross-table backfill replacement
        # (backfill_pool is the whole population, not just their table), so
        # this is a floor, not an exact count.
        for s in stats.values():
            assert s.sessions_played >= 2

    def test_num_workers_reuses_a_provided_executor(self):
        from concurrent.futures import ProcessPoolExecutor
        players = make_random_players(12)
        config = GameConfig(max_hands_per_session=3)
        sim_config = SimConfig(rounds_per_generation=1, table_size=6)
        with ProcessPoolExecutor(max_workers=2) as pool:
            run_final_tournament(
                players, config, sim_config, np.random.default_rng(0),
                show_progress=False, num_workers=2, executor=pool,
            )
            stats2 = run_final_tournament(
                players, config, sim_config, np.random.default_rng(1),
                show_progress=False, num_workers=2, executor=pool,
            )
            assert set(stats2.keys()) == {p.player_id for p in players}


class TestRankPlayers:
    def test_ranks_by_mean_net_chips_descending(self):
        players = make_random_players(3)
        stats = {
            players[0].player_id: PlayerStats(player_id=players[0].player_id, sessions_played=1, total_net_chips=10.0),
            players[1].player_id: PlayerStats(player_id=players[1].player_id, sessions_played=1, total_net_chips=50.0),
            players[2].player_id: PlayerStats(player_id=players[2].player_id, sessions_played=1, total_net_chips=-5.0),
        }
        ranked = rank_players(players, stats, sparsity_penalty=0.0)
        assert [p.player_id for p in ranked] == [players[1].player_id, players[0].player_id, players[2].player_id]

    def test_sparsity_penalty_can_change_the_ranking(self):
        cf_sparse = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cf_dense = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cf_dense[:7, :] = 0  # 20 (of 24*3=72) active conditions -- 7 full rules minus one wildcard slot
        cf_dense[6, 2] = strategy.WILDCARD
        sparse_player = Player(player_id=0, genome=make_genome(condition_features=cf_sparse))
        dense_player = Player(player_id=1, genome=make_genome(condition_features=cf_dense))
        stats = {
            0: PlayerStats(player_id=0, sessions_played=1, total_net_chips=90.0),
            1: PlayerStats(player_id=1, sessions_played=1, total_net_chips=100.0),
        }
        # Without a penalty, the slightly better-performing dense player wins.
        ranked_no_penalty = rank_players([sparse_player, dense_player], stats, sparsity_penalty=0.0)
        assert ranked_no_penalty[0].player_id == 1
        # A big enough penalty per nonzero weight flips the ranking.
        ranked_with_penalty = rank_players([sparse_player, dense_player], stats, sparsity_penalty=1.0)
        assert ranked_with_penalty[0].player_id == 0


class TestDescribeGenome:
    def test_includes_player_name_and_rank(self):
        player = Player(player_id=1, genome=make_genome(), label="Champion")
        stats = PlayerStats(player_id=1, sessions_played=10, total_net_chips=123.0)
        report = describe_genome(player, stats, GameConfig(), rank=1)
        assert "Champion" in report
        assert "#1" in report

    def test_no_active_gto_spots_says_so(self):
        player = Player(player_id=1, genome=make_genome())
        stats = PlayerStats(player_id=1)
        report = describe_genome(player, stats, GameConfig(), rank=1)
        assert "doesn't trust any memorized chart" in report

    def test_active_gto_spot_is_listed(self):
        flags = np.zeros(NUM_GTO_SPOTS)
        flags[0] = 1.0
        player = Player(player_id=1, genome=make_genome(gto_flags=flags))
        stats = PlayerStats(player_id=1)
        report = describe_genome(player, stats, GameConfig(), rank=1)
        assert "GTO Chart Overrides" in report
        assert "When this applies" in report

    def test_empty_genome_reports_no_referenced_features(self):
        player = Player(player_id=1, genome=make_genome())
        stats = PlayerStats(player_id=1)
        report = describe_genome(player, stats, GameConfig(), rank=1)
        assert "No rule references a feature" in report

    def test_report_lists_referenced_features_and_rule(self):
        cf = np.full((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), strategy.WILDCARD)
        cb = np.zeros((strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE), dtype=np.int64)
        actions = np.full(strategy.NUM_RULES, strategy.ACTION_FOLD)
        cf[0, 0] = strategy.feature_index("hand_category_norm")
        cb[0, 0] = 1
        actions[0] = strategy.ACTION_RAISE
        player = Player(player_id=1, genome=make_genome(condition_features=cf, condition_buckets=cb, rule_actions=actions))
        stats = PlayerStats(player_id=1)
        report = describe_genome(player, stats, GameConfig(), rank=1)
        assert "Hand Strength Tier" in report
        assert "[rule 0]" in report
        assert "**Raise**" in report


class TestExportTopN:
    def test_writes_leaderboard_and_per_player_files(self, tmp_path):
        players = make_random_players(3)
        stats = {
            p.player_id: PlayerStats(player_id=p.player_id, label=p.label, sessions_played=2, total_net_chips=float(i))
            for i, p in enumerate(players)
        }
        ranked = sorted(players, key=lambda p: stats[p.player_id].total_net_chips, reverse=True)
        out_dir = str(tmp_path / "final")
        export_top_n(ranked, stats, GameConfig(), n=2, out_dir=out_dir)

        assert os.path.exists(os.path.join(out_dir, "leaderboard.md"))
        with open(os.path.join(out_dir, "leaderboard.md"), encoding="utf-8") as f:
            leaderboard = f.read()
        assert "rank" in leaderboard.lower()

        for rank in (1, 2):
            base = f"rank{rank:02d}_player{ranked[rank - 1].player_id}"
            assert os.path.exists(os.path.join(out_dir, f"{base}_strategy.md"))
            assert os.path.exists(os.path.join(out_dir, f"{base}_genome.json"))

        # n=2 -- the 3rd-ranked player shouldn't get exported files.
        base3 = f"rank03_player{ranked[2].player_id}"
        assert not os.path.exists(os.path.join(out_dir, f"{base3}_strategy.md"))

    def test_creates_out_dir_if_missing(self, tmp_path):
        players = make_random_players(2)
        stats = {p.player_id: PlayerStats(player_id=p.player_id) for p in players}
        out_dir = str(tmp_path / "does" / "not" / "exist")
        export_top_n(players, stats, GameConfig(), n=1, out_dir=out_dir)
        assert os.path.isdir(out_dir)
