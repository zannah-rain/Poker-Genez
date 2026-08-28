import numpy as np
import pytest

from cards import Card
from evaluator import evaluate_best
from game import (
    FLOP, PREFLOP, TURN, GameConfig, HandStats, SeatState, _effective_stack,
    _street_order, betting_round, compute_payouts, play_hand,
)
from genome import BET_RAISE, CHECK_CALL, FOLD
from opponent_model import OpponentModel
from player import Player


class FixedGenome:
    """Test double for Genome: always attempts the same action, falling back
    to CHECK_CALL if that action isn't currently legal -- mirrors how a real
    genome degrades (see Genome.decide's own fallbacks)."""

    def __init__(self, action, bet_size=0.0):
        self.action = action
        self.bet_size = bet_size
        self.calls = []

    def decide(self, situation, legal_actions, rng=None):
        self.calls.append((situation, legal_actions))
        action = self.action if self.action in legal_actions else CHECK_CALL
        return action, self.bet_size


def make_player(pid, action, bet_size=0.0):
    return Player(player_id=pid, genome=FixedGenome(action, bet_size))


def make_seats(actions_and_stacks):
    """actions_and_stacks: list of (action, stack) or (action, stack, bet_size)."""
    seats = []
    for i, spec in enumerate(actions_and_stacks):
        action, stack = spec[0], spec[1]
        bet_size = spec[2] if len(spec) > 2 else 0.0
        seats.append(SeatState(player=make_player(i, action, bet_size), stack=stack))
    return seats


class TestSeatState:
    def test_reset_for_hand_clears_hand_state(self):
        seat = SeatState(player=make_player(0, CHECK_CALL), stack=100.0)
        seat.hole = [Card.from_str("Ah")]
        seat.folded = True
        seat.street_committed = 5.0
        seat.total_committed = 5.0
        seat.reset_for_hand()
        assert seat.hole == []
        assert seat.folded is False
        assert seat.street_committed == 0.0
        assert seat.total_committed == 0.0

    def test_reset_marks_busted_seat_as_all_in(self):
        seat = SeatState(player=make_player(0, CHECK_CALL), stack=0.0)
        seat.reset_for_hand()
        assert seat.all_in is True

    def test_reset_leaves_seat_with_chips_not_all_in(self):
        seat = SeatState(player=make_player(0, CHECK_CALL), stack=50.0)
        seat.reset_for_hand()
        assert seat.all_in is False


class TestStreetOrder:
    def test_preflop_matches_seating_preflop_order(self):
        from seating import preflop_order
        assert _street_order(0, 6, PREFLOP) == preflop_order(0, 6)

    def test_postflop_starts_at_small_blind_for_full_ring(self):
        order = _street_order(0, 6, FLOP)
        # button_idx=0, n=6 -> sb=1
        assert order[0] == 1

    def test_postflop_starts_at_big_blind_heads_up(self):
        order = _street_order(0, 2, FLOP)
        # heads-up: button==SB, so postflop starts with the BB (seat 1).
        assert order[0] == 1


class TestEffectiveStack:
    def test_uses_max_of_other_active_stacks(self):
        seats = [
            SeatState(player=make_player(0, CHECK_CALL), stack=100.0),
            SeatState(player=make_player(1, CHECK_CALL), stack=50.0),
            SeatState(player=make_player(2, CHECK_CALL), stack=200.0),
        ]
        assert _effective_stack(seats, 0, [0, 1, 2]) == 100.0

    def test_capped_by_my_own_stack(self):
        seats = [
            SeatState(player=make_player(0, CHECK_CALL), stack=30.0),
            SeatState(player=make_player(1, CHECK_CALL), stack=200.0),
        ]
        assert _effective_stack(seats, 0, [0, 1]) == 30.0

    def test_no_others_returns_own_stack(self):
        seats = [SeatState(player=make_player(0, CHECK_CALL), stack=77.0)]
        assert _effective_stack(seats, 0, [0]) == 77.0


class TestBettingRoundBasic:
    def test_all_check_leaves_pot_and_stacks_unchanged(self):
        seats = make_seats([(CHECK_CALL, 200.0)] * 3)
        pot, num_raises, _ = betting_round(
            seats, order=[0, 1, 2], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        assert pot == 6.0
        assert num_raises == 0
        assert all(s.stack == 200.0 for s in seats)

    def test_raise_then_calls_updates_pot_and_stacks(self):
        seats = make_seats([
            (BET_RAISE, 200.0, 10.0), (CHECK_CALL, 200.0), (CHECK_CALL, 200.0),
        ])
        pot, num_raises, _ = betting_round(
            seats, order=[0, 1, 2], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        assert num_raises == 1
        assert seats[0].stack == 190.0
        assert seats[1].stack == 190.0
        assert seats[2].stack == 190.0
        assert pot == 36.0  # 6 + 10 (raise) + 10 (call) + 10 (call)

    def test_fold_removes_player_from_hand(self):
        seats = make_seats([(FOLD, 200.0), (CHECK_CALL, 200.0)])
        betting_round(
            seats, order=[0, 1], pot=6.0, min_bet=2.0, starting_bet=10.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        assert seats[0].folded is True

    def test_betting_stops_once_only_one_player_remains(self):
        seats = make_seats([(FOLD, 200.0), (FOLD, 200.0), (CHECK_CALL, 200.0)])
        betting_round(
            seats, order=[0, 1, 2], pot=6.0, min_bet=2.0, starting_bet=10.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        # Seat 2 should never have been asked to act (its stack untouched).
        assert seats[2].stack == 200.0
        assert seats[2].street_committed == 0.0

    def test_raise_cannot_exceed_raisers_stack(self):
        seats = make_seats([
            (BET_RAISE, 15.0, 1000.0), (CHECK_CALL, 200.0),
        ])
        pot, _, _ = betting_round(
            seats, order=[0, 1], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        assert seats[0].stack == 0.0
        assert seats[0].all_in is True

    def test_min_raise_floor_uses_pot_fraction(self):
        # A tiny requested raise should be floored up to
        # min_raise_fraction_of_pot * pot on top of the current bet.
        seats = make_seats([
            (BET_RAISE, 200.0, 0.01), (CHECK_CALL, 200.0),
        ])
        pot, _, _ = betting_round(
            seats, order=[0, 1], pot=100.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        # min raise increment = max(2, 2, 0.25*100) = 25 -> raiser commits 25.
        assert seats[0].stack == 175.0

    def test_max_raises_per_street_caps_further_raises(self):
        seats = make_seats([
            (BET_RAISE, 200.0, 10.0), (BET_RAISE, 200.0, 10.0),
        ])
        pot, num_raises, _ = betting_round(
            seats, order=[0, 1], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=1,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        # Seat 1 wanted to re-raise but BET_RAISE was no longer legal
        # (num_raises already hit the cap), so it fell back to a call.
        assert num_raises == 1

    def test_hand_stats_accumulate_action_counts(self):
        stats = HandStats()
        seats = make_seats([(BET_RAISE, 200.0, 10.0), (FOLD, 200.0)])
        betting_round(
            seats, order=[0, 1], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
            stats=stats,
        )
        assert stats.action_counts[BET_RAISE] == 1
        assert stats.action_counts[FOLD] == 1
        assert stats.facing_bet_decisions == 1  # seat 1 faced seat 0's raise
        assert stats.facing_bet_folds == 1
        assert stats.raises_per_street == [1]

    def test_opponent_model_is_updated_during_betting(self):
        model = OpponentModel()
        seats = make_seats([(BET_RAISE, 200.0, 10.0), (CHECK_CALL, 200.0)])
        betting_round(
            seats, order=[0, 1], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
            opp_model=model,
        )
        assert model.get(0).postflop_bets_raises == 1
        assert model.get(1).postflop_calls == 1

    def test_situation_seen_by_genome_has_expected_fields(self):
        seats = make_seats([(CHECK_CALL, 200.0), (CHECK_CALL, 200.0)])
        betting_round(
            seats, order=[0, 1], pot=6.0, min_bet=2.0, starting_bet=0.0,
            board=[], street=FLOP, starting_stack=200.0, button_idx=0,
            completed_street_raises=(), completed_street_aggressors=(), max_raises_per_street=4,
            min_raise_fraction_of_pot=0.25, rng=np.random.default_rng(0),
        )
        situation, legal_actions = seats[0].player.genome.calls[0]
        assert situation.pot == 6.0
        assert situation.street == FLOP
        assert CHECK_CALL in legal_actions
        assert FOLD not in legal_actions  # nothing to call yet


class _RaisePreflopThenCheckCallGenome:
    """Raises preflop whenever legal, checks/calls every other decision --
    for building an is_aggressor scenario with a known preflop raiser."""

    def __init__(self):
        self.calls = []

    def decide(self, situation, legal_actions, rng=None):
        self.calls.append((situation, legal_actions))
        if situation.street == PREFLOP and BET_RAISE in legal_actions:
            return BET_RAISE, 10.0
        return CHECK_CALL, 0.0


class _AlwaysCheckCallGenome:
    def __init__(self):
        self.calls = []

    def decide(self, situation, legal_actions, rng=None):
        self.calls.append((situation, legal_actions))
        return CHECK_CALL, 0.0


class TestIsAggressorReflectsThePreviousStreet:
    def test_preflop_raiser_reads_as_aggressor_on_the_flop(self):
        raiser = _RaisePreflopThenCheckCallGenome()
        caller = _AlwaysCheckCallGenome()
        seats = [
            SeatState(player=Player(player_id=0, genome=raiser), stack=200.0),
            SeatState(player=Player(player_id=1, genome=caller), stack=200.0),
        ]
        play_hand(seats, button_idx=0, config=GameConfig(), rng=np.random.default_rng(0))

        raiser_flop_situations = [s for s, _ in raiser.calls if s.street == FLOP]
        caller_flop_situations = [s for s, _ in caller.calls if s.street == FLOP]
        assert raiser_flop_situations
        assert caller_flop_situations
        assert all(s.is_aggressor_previous_street for s in raiser_flop_situations)
        assert all(not s.is_aggressor_previous_street for s in caller_flop_situations)

    def test_nobody_is_ever_the_aggressor_at_their_own_decision_this_street(self):
        # The player who just raised is skipped in the to-act order until
        # either the street ends or someone else re-raises (which replaces
        # them as the street's own aggressor) -- so at the moment any
        # decision is made, "I raised this same street" must always read
        # False, preflop included (where it's also always False, since
        # there's no street before it).
        raiser = _RaisePreflopThenCheckCallGenome()
        caller = _AlwaysCheckCallGenome()
        seats = [
            SeatState(player=Player(player_id=0, genome=raiser), stack=200.0),
            SeatState(player=Player(player_id=1, genome=caller), stack=200.0),
        ]
        play_hand(seats, button_idx=0, config=GameConfig(), rng=np.random.default_rng(0))

        preflop_situations = [s for s, _ in raiser.calls if s.street == PREFLOP]
        preflop_situations += [s for s, _ in caller.calls if s.street == PREFLOP]
        assert preflop_situations
        assert all(not s.is_aggressor_previous_street for s in preflop_situations)

    def test_no_aggressor_carried_forward_when_previous_street_checked_through(self):
        seats = [
            SeatState(player=Player(player_id=0, genome=_AlwaysCheckCallGenome()), stack=200.0),
            SeatState(player=Player(player_id=1, genome=_AlwaysCheckCallGenome()), stack=200.0),
        ]
        play_hand(seats, button_idx=0, config=GameConfig(), rng=np.random.default_rng(0))

        flop_situations = [
            s for seat in seats for s, _ in seat.player.genome.calls if s.street == FLOP
        ]
        assert flop_situations
        assert all(not s.is_aggressor_previous_street for s in flop_situations)


class _CallPreflopThenRaiseFlopGenome:
    """Calls preflop, raises the flop whenever legal, checks/calls every
    other decision -- the flop counterpart to
    _RaisePreflopThenCheckCallGenome, for a scenario where the preflop and
    flop aggressors are two different players."""

    def __init__(self):
        self.calls = []

    def decide(self, situation, legal_actions, rng=None):
        self.calls.append((situation, legal_actions))
        if situation.street == FLOP and BET_RAISE in legal_actions:
            return BET_RAISE, 10.0
        return CHECK_CALL, 0.0


class TestFrozenPerStreetRaisesAndAggressorFamily:
    """num_raises_preflop/flop/turn and is_aggressor_preflop/flop/turn are
    each pinned to one specific calendar street (unlike num_raises_previous_
    street/is_aggressor_previous_street, which are always relative to
    whatever street is current) -- 0/False until that specific street has
    actually finished, then frozen at its final reading for the rest of the
    hand, however many streets later a decision happens to be."""

    def test_preflop_and_flop_aggressor_readings_stay_independent_on_the_turn(self):
        # Seat 0 opens preflop (its only raise); seat 1 calls preflop, then
        # raises the flop (its only raise) -- so by the turn, the preflop
        # and flop "who raised last" readings genuinely disagree, unlike
        # is_aggressor_previous_street, which can only ever point at one of
        # them (whichever street is immediately before the current one).
        preflop_raiser = _RaisePreflopThenCheckCallGenome()
        flop_raiser = _CallPreflopThenRaiseFlopGenome()
        seats = [
            SeatState(player=Player(player_id=0, genome=preflop_raiser), stack=200.0),
            SeatState(player=Player(player_id=1, genome=flop_raiser), stack=200.0),
        ]
        play_hand(seats, button_idx=0, config=GameConfig(), rng=np.random.default_rng(0))

        preflop_raiser_turn = [s for s, _ in preflop_raiser.calls if s.street == TURN]
        flop_raiser_turn = [s for s, _ in flop_raiser.calls if s.street == TURN]
        assert preflop_raiser_turn
        assert flop_raiser_turn

        for s in preflop_raiser_turn:
            assert s.is_aggressor_preflop is True
            assert s.is_aggressor_flop is False
            assert s.is_aggressor_previous_street is False  # previous (flop) was seat 1, not seat 0
            assert s.num_raises_preflop == 1
            assert s.num_raises_flop == 1
        for s in flop_raiser_turn:
            assert s.is_aggressor_preflop is False
            assert s.is_aggressor_flop is True
            assert s.is_aggressor_previous_street is True
            assert s.num_raises_preflop == 1
            assert s.num_raises_flop == 1

    def test_readings_are_zero_before_their_own_street_has_finished(self):
        preflop_raiser = _RaisePreflopThenCheckCallGenome()
        flop_raiser = _CallPreflopThenRaiseFlopGenome()
        seats = [
            SeatState(player=Player(player_id=0, genome=preflop_raiser), stack=200.0),
            SeatState(player=Player(player_id=1, genome=flop_raiser), stack=200.0),
        ]
        play_hand(seats, button_idx=0, config=GameConfig(), rng=np.random.default_rng(0))

        preflop_situations = [s for s, _ in preflop_raiser.calls if s.street == PREFLOP]
        preflop_situations += [s for s, _ in flop_raiser.calls if s.street == PREFLOP]
        assert preflop_situations
        assert all(s.num_raises_preflop == 0 for s in preflop_situations)
        assert all(s.is_aggressor_preflop is False for s in preflop_situations)

        flop_situations = [s for s, _ in preflop_raiser.calls if s.street == FLOP]
        flop_situations += [s for s, _ in flop_raiser.calls if s.street == FLOP]
        assert flop_situations
        assert all(s.num_raises_flop == 0 for s in flop_situations)
        assert all(s.is_aggressor_flop is False for s in flop_situations)


class TestComputePayouts:
    def _seat_with_hand(self, hole_str, total_committed):
        hole = [Card.from_str(tok) for tok in hole_str.split()]
        seat = SeatState(player=make_player(0, CHECK_CALL), stack=0.0)
        seat.hole = hole
        seat.total_committed = total_committed
        return seat

    def test_single_pot_split_between_two_equal_stacks(self):
        board = [Card.from_str(t) for t in "2c 5d 9h Jc 3s".split()]
        seats = [
            self._seat_with_hand("Ah Kh", 50.0),  # ace high
            self._seat_with_hand("7h 7d", 50.0),  # pair of 7s -- wins
        ]
        payouts = compute_payouts(seats, [0, 1], folded=set(), board=board)
        assert payouts[1] > payouts[0]
        assert payouts[0] == 0.0
        assert payouts[1] == 100.0

    def test_split_pot_on_tie(self):
        board = [Card.from_str(t) for t in "2c 5d 9h Jc 3s".split()]
        seats = [
            self._seat_with_hand("Ah Kh", 50.0),
            self._seat_with_hand("Ad Kd", 50.0),
        ]
        payouts = compute_payouts(seats, [0, 1], folded=set(), board=board)
        assert payouts[0] == payouts[1] == 50.0

    def test_side_pot_with_short_all_in_stack(self):
        board = [Card.from_str(t) for t in "3c 5d 9h Jd Kc".split()]
        seats = [
            self._seat_with_hand("2h 2d", 20.0),   # short stack, worst hand
            self._seat_with_hand("8h 8d", 100.0),  # best hand
            self._seat_with_hand("7h 7d", 100.0),  # middle hand
        ]
        payouts = compute_payouts(seats, [0, 1, 2], folded=set(), board=board)
        # Main pot (3 x 20 = 60) is contested by all three -> seat 1 (best hand) wins it.
        # Side pot (2 x 80 = 160) is contested only by seats 1 and 2 -> seat 1 wins that too.
        assert payouts[1] == 220.0
        assert payouts[0] == 0.0
        assert payouts[2] == 0.0
        assert sum(payouts.values()) == 60.0 + 160.0

    def test_folded_players_are_not_eligible_but_still_contribute(self):
        board = [Card.from_str(t) for t in "2c 5d 9h Jc 3s".split()]
        seats = [
            self._seat_with_hand("Ah Ad", 50.0),  # best hand, but folded
            self._seat_with_hand("2h 3d", 50.0),
        ]
        payouts = compute_payouts(seats, [0, 1], folded={0}, board=board)
        assert payouts[0] == 0.0
        assert payouts[1] == 100.0

    def test_total_payouts_conserve_chips(self):
        board = [Card.from_str(t) for t in "2c 5d 9h Jc 3s".split()]
        seats = [
            self._seat_with_hand("Ah Ad", 30.0),
            self._seat_with_hand("7h 7d", 70.0),
            self._seat_with_hand("2h 3d", 45.0),
        ]
        payouts = compute_payouts(seats, [0, 1, 2], folded=set(), board=board)
        assert sum(payouts.values()) == pytest.approx(30.0 + 70.0 + 45.0)


class TestPlayHandIntegration:
    def _config(self, **overrides):
        defaults = dict(small_blind=1.0, big_blind=2.0, starting_stack=200.0, max_hands_per_session=1)
        defaults.update(overrides)
        return GameConfig(**defaults)

    def test_uncontested_pot_goes_to_last_player_standing(self):
        # 3-handed: BTN and SB fold preflop, BB (never faced with a real
        # decision -- call_amount is 0 for the BB's own option) wins the
        # blinds uncontested, ending the hand before the flop.
        seats = make_seats([(FOLD, 200.0), (FOLD, 200.0), (CHECK_CALL, 200.0)])
        result = play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(0))
        assert result.payouts[2] == 3.0  # wins the sb(1) + bb(2)
        assert result.payouts[0] == 0.0
        assert result.payouts[1] == 0.0
        assert seats[2].stack == 201.0  # +1 net
        assert seats[1].stack == 199.0  # -1 net (posted sb, folded)
        assert seats[0].stack == 200.0  # never put chips in

    def test_chip_conservation_across_a_full_showdown(self):
        seats = make_seats([(CHECK_CALL, 200.0)] * 4)
        total_before = sum(s.stack for s in seats)
        result = play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(1))
        total_after = sum(s.stack for s in seats)
        assert total_after == pytest.approx(total_before)
        assert sum(result.payouts.values()) == pytest.approx(sum(s.total_committed for s in seats))

    def test_full_showdown_deals_a_5_card_board(self):
        seats = make_seats([(CHECK_CALL, 200.0)] * 3)
        result = play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(2))
        assert len(result.board) == 5
        assert len(set(result.board)) == 5

    def test_winner_matches_independent_hand_evaluation(self):
        seats = make_seats([(CHECK_CALL, 200.0)] * 3)
        result = play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(3))
        hands = {i: evaluate_best(s.hole + result.board) for i, s in enumerate(seats)}
        best = max(hands.values())
        expected_winners = [i for i, h in hands.items() if h == best]
        actual_winners = [i for i, net in result.payouts.items() if net > 0]
        assert sorted(actual_winners) == sorted(expected_winners)

    def test_all_in_preflop_runs_the_board_out(self):
        seats = make_seats([(BET_RAISE, 50.0, 1000.0), (CHECK_CALL, 200.0)])
        result = play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(4))
        assert len(result.board) == 5
        assert seats[0].stack == 0.0 or result.payouts[0] > 0

    def test_opponent_model_hands_counter_advances(self):
        seats = make_seats([(CHECK_CALL, 200.0)] * 3)
        model = OpponentModel()
        play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(5), opp_model=model)
        for s in seats:
            assert model.get(s.player.player_id).hands == 1

    def test_reset_for_hand_is_applied_before_dealing(self):
        seats = make_seats([(CHECK_CALL, 200.0)] * 2)
        seats[0].folded = True  # stale state from a hypothetical earlier hand
        play_hand(seats, button_idx=0, config=self._config(), rng=np.random.default_rng(6))
        # If reset_for_hand ran, seat 0 should have been able to participate
        # (not still marked folded from before the hand even started).
        assert len(seats[0].hole) == 2
