import random
from collections import Counter

import pytest

from cards import RANKS, SUITS, Card, make_deck
from evaluator import (
    FLUSH, FULL_HOUSE, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH,
    TRIPS, TWO_PAIR,
    best_hand_from_available, category_name, count_straight_draw_outs,
    evaluate_5, evaluate_best, evaluate_preflop_pair, flush_draw_suit,
    has_flush_draw, has_straight_draw,
)


def cards(s):
    """Parses a space-separated card string like 'Ah Kh Qh Jh Th' into Cards."""
    return [Card.from_str(tok) for tok in s.split()]


class TestEvaluate5:
    def test_straight_flush(self):
        result = evaluate_5(cards("9h 8h 7h 6h 5h"))
        assert result[0] == STRAIGHT_FLUSH
        assert result[1] == 9

    def test_wheel_straight_flush(self):
        result = evaluate_5(cards("Ah 2h 3h 4h 5h"))
        assert result[0] == STRAIGHT_FLUSH
        assert result[1] == 5  # ace plays low

    def test_quads(self):
        result = evaluate_5(cards("9h 9d 9c 9s 2c"))
        assert result[0] == QUADS
        assert result[1:] == (9, 2)

    def test_full_house(self):
        result = evaluate_5(cards("9h 9d 9c 2s 2c"))
        assert result[0] == FULL_HOUSE
        assert result[1:] == (9, 2)

    def test_flush(self):
        result = evaluate_5(cards("Ah 9h 7h 4h 2h"))
        assert result[0] == FLUSH
        assert result[1:] == (14, 9, 7, 4, 2)

    def test_straight(self):
        result = evaluate_5(cards("9h 8d 7c 6s 5h"))
        assert result[0] == STRAIGHT
        assert result[1] == 9

    def test_wheel_straight_non_flush(self):
        result = evaluate_5(cards("Ah 2d 3c 4s 5h"))
        assert result[0] == STRAIGHT
        assert result[1] == 5

    def test_trips(self):
        result = evaluate_5(cards("9h 9d 9c 4s 2h"))
        assert result[0] == TRIPS
        assert result[1:] == (9, 4, 2)

    def test_two_pair(self):
        result = evaluate_5(cards("9h 9d 4c 4s 2h"))
        assert result[0] == TWO_PAIR
        assert result[1:] == (9, 4, 2)

    def test_one_pair(self):
        result = evaluate_5(cards("9h 9d 6c 4s 2h"))
        assert result[0] == PAIR
        assert result[1:] == (9, 6, 4, 2)

    def test_high_card(self):
        result = evaluate_5(cards("9h 7d 6c 4s 2h"))
        assert result[0] == HIGH_CARD
        assert result[1:] == (9, 7, 6, 4, 2)

    def test_requires_exactly_5_cards(self):
        with pytest.raises(AssertionError):
            evaluate_5(cards("9h 7d 6c 4s"))

    def test_flush_beats_straight(self):
        flush_hand = evaluate_5(cards("Ah 9h 7h 4h 2h"))
        straight_hand = evaluate_5(cards("9h 8d 7c 6s 5h"))
        assert flush_hand > straight_hand

    def test_full_house_beats_flush(self):
        fh = evaluate_5(cards("9h 9d 9c 2s 2c"))
        flush = evaluate_5(cards("Ah 9h 7h 4h 2h"))
        assert fh > flush

    def test_higher_pair_beats_lower_pair(self):
        high = evaluate_5(cards("9h 9d 6c 4s 2h"))
        low = evaluate_5(cards("8h 8d 6c 4s 2h"))
        assert high > low

    def test_kicker_breaks_tie_between_equal_pairs(self):
        better_kicker = evaluate_5(cards("9h 9d Kc 4s 2h"))
        worse_kicker = evaluate_5(cards("9h 9d 6c 4s 2h"))
        assert better_kicker > worse_kicker


class TestEvaluateBest:
    def test_evaluates_exactly_5(self):
        assert evaluate_best(cards("9h 9d 6c 4s 2h")) == evaluate_5(cards("9h 9d 6c 4s 2h"))

    def test_picks_best_5_of_7(self):
        # Hole 9h9d, board makes a flush available using only 4 board hearts + the 9h.
        seven = cards("9h 9d Ah Kh Qh Jh 2c")
        result = evaluate_best(seven)
        assert result[0] == FLUSH

    def test_picks_best_5_of_6(self):
        six = cards("9h 9d 9c 9s 2c 3d")
        result = evaluate_best(six)
        assert result[0] == QUADS

    def test_requires_at_least_5_cards(self):
        with pytest.raises(ValueError):
            evaluate_best(cards("9h 9d 6c 4s"))

    def test_best_of_7_is_at_least_as_good_as_any_single_5_subset(self):
        import itertools
        seven = cards("Ah Kd 9c 4s 2h Qh Jh")
        best = evaluate_best(seven)
        for combo in itertools.combinations(seven, 5):
            assert best >= evaluate_5(list(combo))


class TestEvaluatePreflopPair:
    def test_pocket_pair_is_true(self):
        assert evaluate_preflop_pair(cards("9h 9d")) is True

    def test_non_pair_is_false(self):
        assert evaluate_preflop_pair(cards("9h 8d")) is False


class TestCategoryName:
    def test_all_categories_named(self):
        names = [category_name(i) for i in range(9)]
        assert names == [
            "High Card", "Pair", "Two Pair", "Trips", "Straight",
            "Flush", "Full House", "Quads", "Straight Flush",
        ]


class TestFlushDraw:
    def test_four_of_a_suit_is_a_flush_draw(self):
        suit = flush_draw_suit(cards("Ah 9h 7h 4h 2c"))
        assert suit == Card.from_str("Ah").suit
        assert has_flush_draw(cards("Ah 9h 7h 4h 2c")) is True

    def test_three_of_a_suit_is_not_a_flush_draw(self):
        assert flush_draw_suit(cards("Ah 9h 7h 4c 2c")) is None
        assert has_flush_draw(cards("Ah 9h 7h 4c 2c")) is False

    def test_five_of_a_suit_is_not_reported_as_a_draw(self):
        # already made -- flush_draw_suit only reports an exact 4-count.
        assert flush_draw_suit(cards("Ah 9h 7h 4h 2h")) is None


class TestStraightDraw:
    def test_open_ended_draw_has_two_outs(self):
        # 6-7-8-9 needs a 5 or a T.
        outs = count_straight_draw_outs(cards("6h 7d 8c 9s 2h"))
        assert outs == 2
        assert has_straight_draw(cards("6h 7d 8c 9s 2h")) is True

    def test_gutshot_has_one_out(self):
        # 6-7-8-T needs exactly a 9.
        outs = count_straight_draw_outs(cards("6h 7d 8c Ts 2h"))
        assert outs == 1

    def test_made_straight_has_zero_draw_outs(self):
        outs = count_straight_draw_outs(cards("6h 7d 8c 9s Th"))
        assert outs == 0
        assert has_straight_draw(cards("6h 7d 8c 9s Th")) is False

    def test_no_connection_has_zero_outs(self):
        outs = count_straight_draw_outs(cards("2h 5d 9c Ks Ah"))
        # 2,5,9,K,A: check no accidental straight completion
        assert isinstance(outs, int)
        assert outs == 0

    def test_wheel_draw_via_ace(self):
        # A-2-3-4 needs a 5 for the wheel; the ace shouldn't double count as
        # both high and low.
        outs = count_straight_draw_outs(cards("Ah 2d 3c 4s Kh"))
        assert outs == 1


class TestBestHandFromAvailable:
    def test_preflop_pocket_pair(self):
        result = best_hand_from_available(cards("9h 9d"), [])
        assert result["category"] == PAIR
        assert result["tiebreak"] == (9,)
        assert result["high_card"] == 9
        assert result["flush_draw"] is False
        assert result["straight_draw"] is False

    def test_preflop_non_pair(self):
        result = best_hand_from_available(cards("Ah 9d"), [])
        assert result["category"] == HIGH_CARD
        assert result["tiebreak"] == (14, 9)
        assert result["high_card"] == 14

    def test_postflop_uses_all_available_cards(self):
        hole = cards("9h 9d")
        board = cards("9c 2s 4h")
        result = best_hand_from_available(hole, board)
        assert result["category"] == TRIPS

    def test_postflop_flush_draw_detected(self):
        hole = cards("Ah 9h")
        board = cards("7h 4h 2c")
        result = best_hand_from_available(hole, board)
        assert result["flush_draw"] is True
        assert result["flush_draw_suit"] == Card.from_str("Ah").suit

    def test_postflop_straight_draw_detected(self):
        hole = cards("6h 7d")
        board = cards("8c 9s 2h")
        result = best_hand_from_available(hole, board)
        assert result["straight_draw"] is True
        assert result["straight_draw_outs"] == 2


def _reference_category(cards_list: list) -> int:
    """Deliberately-independent, from-scratch category classifier (not the
    same code path as evaluate_5 -- no Counter-based count_pattern matching,
    no shared straight-detection helper) used purely to fuzz-check evaluate_5
    against random hands. This guards against a regression being introduced
    by a future performance refactor of evaluate_5 -- if the two
    implementations ever disagree, one of them has a bug."""
    ranks = [c.rank for c in cards_list]
    suits = [c.suit for c in cards_list]
    counts_sorted = sorted(Counter(ranks).values(), reverse=True)
    is_flush = len(set(suits)) == 1

    distinct_ranks = sorted(set(ranks))
    is_straight = False
    if len(distinct_ranks) == 5:
        if distinct_ranks[-1] - distinct_ranks[0] == 4:
            is_straight = True
        elif distinct_ranks == [2, 3, 4, 5, 14]:  # wheel: A-2-3-4-5
            is_straight = True

    if is_straight and is_flush:
        return STRAIGHT_FLUSH
    if counts_sorted == [4, 1]:
        return QUADS
    if counts_sorted == [3, 2]:
        return FULL_HOUSE
    if is_flush:
        return FLUSH
    if is_straight:
        return STRAIGHT
    if counts_sorted == [3, 1, 1]:
        return TRIPS
    if counts_sorted == [2, 2, 1]:
        return TWO_PAIR
    if counts_sorted == [2, 1, 1, 1]:
        return PAIR
    return HIGH_CARD


class TestEvaluate5AgainstIndependentReference:
    """A large random fuzz check against a deliberately-separate reference
    implementation -- see _reference_category's docstring for why this
    exists independently of the more targeted hand-by-hand tests above."""

    def test_matches_reference_classifier_on_many_random_hands(self):
        rng = random.Random(12345)
        deck = make_deck()
        for _ in range(20000):
            hand = rng.sample(deck, 5)
            assert evaluate_5(hand)[0] == _reference_category(hand)

    def test_matches_reference_on_hands_biased_toward_pairs_and_flushes(self):
        # Uniform random hands rarely land on rarer categories (quads,
        # full houses, straight flushes) -- deal from a shrunk deck (fewer
        # distinct ranks/suits) to make those categories common too.
        rng = random.Random(999)
        small_ranks = [RANKS[i] for i in range(6)]  # '2'..'7'
        small_suits = SUITS[:3]
        shrunk_deck = [Card.from_str(f"{r}{s}") for r in small_ranks for s in small_suits]
        for _ in range(20000):
            hand = rng.sample(shrunk_deck, 5)
            assert evaluate_5(hand)[0] == _reference_category(hand)
