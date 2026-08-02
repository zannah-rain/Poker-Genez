"""Dedicated correctness tests for cfr_tree's equity-averaged terminal
showdown (_equity_payouts / _unseen_cards / _sample_completions) -- see
cfr_tree.py's module docstring for why an early all-in's terminal value is
now an average over possible board completions instead of one dealt-out
hand. These are independent of (and don't rely on) test_cfr_tree.py's
whole-tree cross-validation, which now only checks *conservation* through
this code path rather than exact values (a value could be wrong in a way
that still conserves chips -- e.g. two players' equities swapped -- so
these hand-verifiable scenarios are what actually pins down correctness).
"""

import math

import numpy as np
import pytest

import cfr_tree
from cards import Card
from game import GameConfig, SeatState
from player import Player


def _seat(hole_str, total_committed):
    seat = SeatState(player=Player(player_id=0, genome=None), stack=0.0)
    seat.hole = [Card.from_str(tok) for tok in hole_str.split()]
    seat.total_committed = total_committed
    return seat


class TestUnseenCards:
    def test_excludes_board_and_every_seats_hole_cards_folded_or_not(self):
        board = [Card.from_str(t) for t in "2h 7h 9c".split()]
        seats = [_seat("Ac Ad", 10.0), _seat("Kc Kd", 10.0), _seat("2c 3c", 10.0)]
        seats[2].folded = True
        remaining = cfr_tree._unseen_cards(seats, [0, 1, 2], board)
        known = set(board) | set(seats[0].hole) | set(seats[1].hole) | set(seats[2].hole)
        assert len(remaining) == 52 - len(known)
        assert not (set(remaining) & known)

    def test_returns_full_deck_minus_ten_when_nothing_shared(self):
        seats = [_seat("Ac Ad", 10.0), _seat("Kc Kd", 10.0)]
        remaining = cfr_tree._unseen_cards(seats, [0, 1], board=[])
        assert len(remaining) == 48


class TestSampleCompletions:
    def test_small_space_is_exhaustively_enumerated(self):
        seats = [_seat("Ac Ad", 10.0), _seat("Kc Kd", 10.0)]
        remaining = cfr_tree._unseen_cards(seats, [0, 1], board=[])  # 48 cards
        completions = cfr_tree._sample_completions(remaining, missing=1, num_rollouts=50, rng=np.random.default_rng(0))
        assert len(completions) == 48  # <= num_rollouts -> exact enumeration, not sampling
        assert len(set(completions)) == 48  # every completion distinct

    def test_large_space_is_capped_at_num_rollouts(self):
        seats = [_seat("Ac Ad", 10.0), _seat("Kc Kd", 10.0)]
        remaining = cfr_tree._unseen_cards(seats, [0, 1], board=[])  # 48 cards
        completions = cfr_tree._sample_completions(remaining, missing=5, num_rollouts=100, rng=np.random.default_rng(0))
        assert math.comb(48, 5) > 100  # sanity: the full space really is too big to enumerate
        assert len(completions) == 100
        assert all(len(set(c)) == 5 for c in completions)  # no card repeated within one completion


class TestEquityPayouts:
    def test_reduces_to_compute_payouts_on_a_complete_board(self):
        from game import compute_payouts

        board = [Card.from_str(t) for t in "2c 5d 9h Jc 3s".split()]
        seats = [_seat("Ah Kh", 50.0), _seat("7h 7d", 50.0)]
        exact = compute_payouts(seats, [0, 1], folded=set(), board=board)
        equity = cfr_tree._equity_payouts(seats, [0, 1], folded=set(), board=board, num_rollouts=50, rng=np.random.default_rng(0))
        assert equity == exact

    def test_river_only_flush_draw_matches_hand_counted_outs(self):
        # Turn board; B needs one more heart to complete a flush (2h/7h on
        # board + 4h/5h in hand = 4 hearts already). Of the 44 unseen cards,
        # exactly 9 are hearts (Ah 3h 6h 8h 9h Th Jh Qh Kh) and every one of
        # them gives B a flush, which beats A's pair (or trips, if the
        # river happens to be the case-Ace -- flush still outranks trips).
        # Every other card leaves A's pair of aces best (a paired board
        # gives A two pair, still ahead of B's no-pair/no-flush hand; a
        # paired 4/5 only gives B a pair, still behind A's pair of aces).
        # So this is an exact, hand-verifiable 9/44 vs 35/44 split.
        board = [Card.from_str(t) for t in "2h 7h 9c Kd".split()]
        seats = [_seat("Ac Ad", 100.0), _seat("4h 5h", 100.0)]
        payouts = cfr_tree._equity_payouts(
            seats, [0, 1], folded=set(), board=board, num_rollouts=50, rng=np.random.default_rng(0),
        )
        pot = 200.0
        assert payouts[0] == pytest.approx(pot * 35 / 44, abs=1e-6)
        assert payouts[1] == pytest.approx(pot * 9 / 44, abs=1e-6)
        assert payouts[0] + payouts[1] == pytest.approx(pot)

    def test_preflop_allin_approximates_known_aa_vs_kk_equity(self):
        # A very well-known reference number: AA vs KK heads-up preflop is
        # ~81.9% / ~18.1% (and can never split -- two distinct pairs can't
        # tie). With num_rollouts=3000 the Monte Carlo standard error here
        # is under 1 percentage point, so an abs=0.05 tolerance is both
        # tight enough to catch a real bug (e.g. equities swapped, or stuck
        # near 50/50) and loose enough not to flake.
        seats = [_seat("Ac Ad", 100.0), _seat("Kc Kd", 100.0)]
        payouts = cfr_tree._equity_payouts(
            seats, [0, 1], folded=set(), board=[], num_rollouts=3000, rng=np.random.default_rng(0),
        )
        pot = 200.0
        assert payouts[0] / pot == pytest.approx(0.819, abs=0.05)
        assert payouts[1] / pot == pytest.approx(0.181, abs=0.05)
        assert payouts[0] + payouts[1] == pytest.approx(pot)

    def test_conserves_total_pot_with_a_folded_side_pot_contributor(self):
        board = [Card.from_str(t) for t in "2h 7h 9c".split()]
        seats = [_seat("Ac Ad", 60.0), _seat("Kc Kd", 60.0), _seat("2c 3d", 20.0)]
        payouts = cfr_tree._equity_payouts(
            seats, [0, 1, 2], folded={2}, board=board, num_rollouts=50, rng=np.random.default_rng(0),
        )
        assert payouts[2] == 0.0  # folded, never eligible
        assert sum(payouts.values()) == pytest.approx(140.0)
