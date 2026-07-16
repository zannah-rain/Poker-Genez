"""Hand evaluation: ranks made hands and detects draws.

Category scale (higher is better), matching standard poker hand rankings:
    0 High Card
    1 Pair
    2 Two Pair
    3 Trips
    4 Straight
    5 Flush
    6 Full House
    7 Quads
    8 Straight Flush
"""

from __future__ import annotations

import itertools
from collections import Counter

from cards import Card

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

NUM_CATEGORIES = 9


def _straight_high(distinct_ranks_desc: list[int]) -> int | None:
    """Given distinct ranks sorted descending, return the high card of the best
    straight found, or None. Handles the wheel (A-2-3-4-5) as high card 5."""
    ranks = set(distinct_ranks_desc)
    if 14 in ranks:
        ranks.add(1)  # ace can play low for the wheel
    ordered = sorted(ranks, reverse=True)
    run = 1
    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1] - 1:
            run += 1
            if run >= 5:
                return ordered[i - 4] if ordered[i - 4] != 1 else 5
        else:
            run = 1
    return None


def evaluate_5(cards: list[Card]) -> tuple:
    """Evaluate exactly 5 cards. Returns a comparable tuple (category, tiebreakers...)."""
    assert len(cards) == 5
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    counts = Counter(ranks)
    # (count, rank) sorted by count desc then rank desc -> gives kicker ordering
    by_count = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    count_pattern = [c for _, c in by_count]
    ordered_ranks = [r for r, _ in by_count]

    if is_flush and straight_high:
        return (STRAIGHT_FLUSH, straight_high)
    if count_pattern == [4, 1]:
        return (QUADS, *ordered_ranks)
    if count_pattern == [3, 2]:
        return (FULL_HOUSE, *ordered_ranks)
    if is_flush:
        return (FLUSH, *ranks)
    if straight_high:
        return (STRAIGHT, straight_high)
    if count_pattern == [3, 1, 1]:
        return (TRIPS, *ordered_ranks)
    if count_pattern == [2, 2, 1]:
        return (TWO_PAIR, *ordered_ranks)
    if count_pattern == [2, 1, 1, 1]:
        return (PAIR, *ordered_ranks)
    return (HIGH_CARD, *ranks)


def evaluate_best(cards: list[Card]) -> tuple:
    """Evaluate the best 5-card hand out of 5, 6, or 7 cards."""
    if len(cards) < 5:
        raise ValueError("evaluate_best requires at least 5 cards")
    if len(cards) == 5:
        return evaluate_5(cards)
    return max(evaluate_5(list(combo)) for combo in itertools.combinations(cards, 5))


def evaluate_preflop_pair(hole: list[Card]) -> bool:
    return hole[0].rank == hole[1].rank


def category_name(cat: int) -> str:
    names = [
        "High Card", "Pair", "Two Pair", "Trips", "Straight",
        "Flush", "Full House", "Quads", "Straight Flush",
    ]
    return names[cat]


def has_flush_draw(cards: list[Card]) -> bool:
    """True if there are exactly 4 cards of one suit (a live flush draw)."""
    suits = Counter(c.suit for c in cards)
    return any(count == 4 for count in suits.values())


def has_straight_draw(cards: list[Card]) -> bool:
    """True if adding one more rank could complete a straight
    (open-ended or gutshot) given the current distinct ranks."""
    ranks = set(c.rank for c in cards)
    if 14 in ranks:
        ranks.add(1)
    for candidate in range(1, 15):
        if candidate in ranks:
            continue
        trial = ranks | {candidate}
        if _straight_high(sorted(trial, reverse=True)):
            return True
    return False


def best_hand_from_available(hole: list[Card], board: list[Card]) -> dict:
    """Compute a feature-friendly summary of the best available hand given
    whatever cards are known so far (preflop: hole only; postflop: hole+board).
    """
    all_cards = hole + board
    if len(all_cards) < 5:
        # Preflop: classify using the 2 hole cards only.
        if evaluate_preflop_pair(hole):
            category = PAIR
            tiebreak = (hole[0].rank,)
        else:
            category = HIGH_CARD
            tiebreak = tuple(sorted((c.rank for c in hole), reverse=True))
        high_card = max(c.rank for c in hole)
        return {
            "category": category,
            "tiebreak": tiebreak,
            "high_card": high_card,
            "flush_draw": False,
            "straight_draw": False,
        }

    category, *tiebreak = evaluate_best(all_cards)
    high_card = max(c.rank for c in all_cards)
    return {
        "category": category,
        "tiebreak": tuple(tiebreak),
        "high_card": high_card,
        "flush_draw": has_flush_draw(all_cards),
        "straight_draw": has_straight_draw(all_cards),
    }
