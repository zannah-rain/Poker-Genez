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


def straight_high(distinct_ranks_desc: list[int]) -> int | None:
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
    """Evaluate exactly 5 cards. Returns a comparable tuple (category, tiebreakers...).
    Called for every one of the 21 5-card combinations checked per 7-card
    showdown hand (see evaluate_best), so this is one of the hottest
    functions in the whole simulation -- kept allocation-light accordingly."""
    assert len(cards) == 5
    ranks = sorted((c.rank for c in cards), reverse=True)
    is_flush = len({c.suit for c in cards}) == 1
    made_straight_high = straight_high(ranks)

    counts = Counter(ranks)
    # (count, rank) sorted by count desc then rank desc -> gives kicker
    # ordering. Sorting ascending by the *un*-negated (count, rank) key and
    # then reversing the whole comparison (reverse=True) is equivalent to
    # sorting by (-count, -rank) directly (both are total orders over
    # distinct ranks, so there are no same-key ties to break differently),
    # but skips the negation arithmetic per item.
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    count_pattern = [c for _, c in by_count]
    ordered_ranks = [r for r, _ in by_count]

    if is_flush and made_straight_high:
        return (STRAIGHT_FLUSH, made_straight_high)
    if count_pattern == [4, 1]:
        return (QUADS, *ordered_ranks)
    if count_pattern == [3, 2]:
        return (FULL_HOUSE, *ordered_ranks)
    if is_flush:
        return (FLUSH, *ranks)
    if made_straight_high:
        return (STRAIGHT, made_straight_high)
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


def flush_draw_suit(cards: list[Card]) -> int | None:
    """Returns the suit with exactly 4 cards among `cards` (a live one-card
    flush draw), or None if there isn't one."""
    suits = Counter(c.suit for c in cards)
    for suit, count in suits.items():
        if count == 4:
            return suit
    return None


def has_flush_draw(cards: list[Card]) -> bool:
    """True if there are exactly 4 cards of one suit (a live flush draw)."""
    return flush_draw_suit(cards) is not None


def has_backdoor_flush_draw(cards: list[Card]) -> bool:
    """True if there are exactly 3 cards of one suit among `cards` -- a
    "backdoor" flush draw, needing both the turn and river (in either
    order) to complete a flush rather than just one more card."""
    suits = Counter(c.suit for c in cards)
    return any(count == 3 for count in suits.values())


def count_straight_draw_outs(cards: list[Card]) -> int:
    """Number of distinct ranks that would complete a straight if added to
    these cards. Returns 0 if a straight is already made (that's not a draw).
    Only real ranks (2-14) are tried as candidates -- 14 (Ace) already covers
    the wheel via straight_high's own low-ace handling, so trying a
    standalone "1" as well would double-count the same missing card."""
    ranks = {c.rank for c in cards}
    if straight_high(sorted(ranks, reverse=True)):
        return 0
    outs = 0
    for candidate in range(2, 15):
        if candidate in ranks:
            continue
        if straight_high(sorted(ranks | {candidate}, reverse=True)):
            outs += 1
    return outs


def has_straight_draw(cards: list[Card]) -> bool:
    """True if adding one more rank could complete a straight
    (open-ended or gutshot) given the current distinct ranks."""
    return count_straight_draw_outs(cards) > 0


def has_backdoor_straight_draw(cards: list[Card]) -> bool:
    """True if the current ranks have no live 1-card straight draw and no
    straight is already made (count_straight_draw_outs would read 0 either
    way), but some pair of additional ranks -- added in either order --
    would complete one: a "backdoor" straight draw needing two specific
    ranks across the remaining streets, not just one. Only meaningful where
    the caller restricts it to the flop (2 more streets actually left to
    fill both ranks) -- by the turn there's only one card left to come, so
    "needs 2 more ranks" no longer describes a real draw."""
    ranks = {c.rank for c in cards}
    if straight_high(sorted(ranks, reverse=True)) is not None:
        return False  # already made -- not a draw at all
    if count_straight_draw_outs(cards) > 0:
        return False  # already a live 1-card draw -- not "backdoor"
    for first in range(2, 15):
        if first in ranks:
            continue
        expanded = ranks | {first}
        for second in range(2, 15):
            if second in expanded:
                continue
            if straight_high(sorted(expanded | {second}, reverse=True)) is not None:
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
            "flush_draw_suit": None,
            "straight_draw": False,
            "straight_draw_outs": 0,
        }

    category, *tiebreak = evaluate_best(all_cards)
    high_card = max(c.rank for c in all_cards)
    draw_suit = flush_draw_suit(all_cards)
    straight_outs = count_straight_draw_outs(all_cards)
    return {
        "category": category,
        "tiebreak": tuple(tiebreak),
        "high_card": high_card,
        "flush_draw": draw_suit is not None,
        "flush_draw_suit": draw_suit,
        "straight_draw": straight_outs > 0,
        "straight_draw_outs": straight_outs,
    }
