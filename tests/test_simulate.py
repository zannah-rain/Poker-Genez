import numpy as np
import pytest

from game import GameConfig, HandStats
from genome import BET_RAISE, CHECK_CALL
from player import Player
from simulate import run_session


class FixedGenome:
    def __init__(self, action, bet_size=0.0):
        self.action = action
        self.bet_size = bet_size

    def decide(self, situation, legal_actions, rng=None):
        action = self.action if self.action in legal_actions else CHECK_CALL
        return action, self.bet_size


class RandomPolicy:
    """Picks a uniformly random legal action each decision (raising, when
    legal, to a random fraction of the pot) -- a generic stand-in for
    "some real strategy" in tests that don't care which one, only that
    hands actually play out with realistic variety."""

    def decide(self, situation, legal_actions, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        action = legal_actions[int(rng.integers(0, len(legal_actions)))]
        bet_size = float(rng.uniform(0.25, 1.5)) * max(situation.pot, 1.0) if action == BET_RAISE else 0.0
        return action, bet_size


def make_random_players(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Player(player_id=i, genome=RandomPolicy()) for i in range(n)]


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
        # Deep stacks alone don't guarantee no one busts -- an all-in
        # confrontation risks a full stack regardless of its size -- so this
        # uses deterministic check/call players (see FixedGenome) rather
        # than random genomes, which can (correctly) shove and bust even
        # with a huge starting stack.
        players = [Player(player_id=i, genome=FixedGenome(CHECK_CALL)) for i in range(2)]
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

    def test_refill_never_duplicates_a_player_already_seated_when_avoidable(self):
        # Regression test: with no dedicated backfill_pool (pool ==
        # table_players), a refill used to be able to redraw a player_id
        # already seated at another live seat, letting that identity occupy
        # two seats at once -- double-counted every hand in the
        # hands_survived loop above, so its total could exceed
        # max_hands_per_session. With only 2 distinct pool members and 2
        # seats, avoiding a duplicate is always possible, so this should
        # never happen now.
        shover = Player(player_id=0, genome=FixedGenome(BET_RAISE, bet_size=100000.0))
        caller = Player(player_id=1, genome=FixedGenome(CHECK_CALL))
        config = GameConfig(max_hands_per_session=15, starting_stack=20.0, small_blind=1.0, big_blind=2.0)
        for seed in range(20):
            result = run_session([shover, caller], config, np.random.default_rng(seed))
            for hands in result["hands_survived"].values():
                assert hands <= 15

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
