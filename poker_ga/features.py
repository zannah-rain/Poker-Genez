"""Turns a poker decision point into a fixed-length feature vector.

This is the bridge between the game engine and a genome: every feature here
is a "basic characteristic of the current situation" (hand strength, board
texture, betting context, position, stack depth...) that the genome's
weights turn into action scores.

Each "multi-value" characteristic (e.g. hand strength, high card rank) is
represented twice: once as a single generalized 0-1 feature (so the genome
can learn a linear trend across its values), and once as a set of exact
per-value indicator features (so the genome can also learn a non-linear,
value-specific adjustment). Indicator features declare which generalized
feature and which specific value they belong to via `linked_to` /
`linked_value_index`, purely so exported reports can fold them back into a
single per-value row instead of listing near-duplicate entries.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from cards import Card
from evaluator import (
    FULL_HOUSE, FLUSH, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH,
    TRIPS, TWO_PAIR, best_hand_from_available, count_straight_draw_outs, straight_high,
)
from seating import SEAT_ROLES, seat_role


@dataclass(frozen=True)
class FeatureSpec:
    key: str  # internal identifier; used to look up this feature's value, not its position
    label: str  # short, human-readable name used in exported reports
    description: str  # precise definition of how the value is computed
    kind: str = "boolean"  # "boolean" (0/1) | "categorical" (fixed set of values) | "continuous" (a range)
    # For categorical/continuous features: every (or, for continuous, a representative
    # sample of) normalized value this feature can take, paired with a human-readable
    # label for that value. None for boolean features (they need no value table).
    value_table: tuple | None = None
    # Set on a boolean feature that is an exact indicator for one specific value of
    # another (categorical/continuous) feature -- e.g. "has_pair" is linked to
    # ("hand_category_norm", 1). None for features that stand on their own.
    linked_to: str | None = None
    linked_value_index: int | None = None
    # High-level category for organizing exported reports (e.g. "Made Hand
    # Features"). Only set on "top-level" specs -- parents of a value-table
    # family, and standalone booleans; linked children inherit their
    # parent's group instead of repeating it (see group_of() below).
    group: str | None = None


# Fixed, human-chosen reading order for report sections -- not derived from
# influence, since the point of grouping is a consistent, thematic structure
# a reader can learn once, not a ranking that shuffles per genome.
FEATURE_GROUPS = [
    "Hole Card Characteristics",
    "Board / Flop Characteristics",
    "Made Hand Features",
    "Draw Features",
    "Betting Behaviour Features",
    "Stack & Pot Features",
    "Table & Game State Features",
    "Opponent Tendency Features",
]


def _rank_label(rank: int) -> str:
    return {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}.get(rank, str(rank))


def _rank_key(rank: int) -> str:
    return {11: "jack", 12: "queen", 13: "king", 14: "ace"}.get(rank, str(rank))


# Gaps of 6+ are all equally "not going to make a straight together" (the
# widest gap that can still share a straight is 5, e.g. A-6 via A-2-3-4-5...6
# doesn't fit that one, but 2-7 can't either -- 5 apart is the practical
# ceiling), so anything wider is lumped into one catch-all bucket.
CONNECTIVITY_GAP_CAP = 6


def _rank_gap(r1: int, r2: int) -> int:
    """Rank distance for straight-connectivity purposes, treating the Ace as
    playable high (A-K = 1 apart) or low (A-2 = 1 apart) -- whichever gives
    the smaller gap -- then capped at CONNECTIVITY_GAP_CAP."""
    gap = abs(r1 - r2)
    if r1 == 14:
        gap = min(gap, abs(1 - r2))
    if r2 == 14:
        gap = min(gap, abs(r1 - 1))
    return min(gap, CONNECTIVITY_GAP_CAP)


def _ace_aware_span(ranks: list[int]) -> int:
    """max-min rank span, also trying the Ace as low (1) if present and
    taking whichever span is smaller -- a board like A-2-3 is tightly
    connected for low straights even though 14-2=12 looks wide apart under a
    naive high-Ace span."""
    span = max(ranks) - min(ranks)
    if 14 in ranks:
        low_ranks = [1 if r == 14 else r for r in ranks]
        span = min(span, max(low_ranks) - min(low_ranks))
    return span


_HOLE_CATEGORY_LABELS = (
    "Junk", "Unsuited Connectors", "Suited 2 Gappers", "Suited 1 Gappers", "Suited Connectors",
    "Qxs", "Kxs", "Axo", "Axs", "Average Pairs", "High Pairs", "Premium Pairs",
)  # index order = strength ascending; must match _hole_hand_category's return values


def _hole_hand_category(hole: list[Card]) -> int:
    """Classifies the two hole cards into exactly one of 12 mutually
    exclusive starting-hand buckets (indexes into _HOLE_CATEGORY_LABELS),
    checked strongest-to-weakest so a higher-priority bucket (e.g. a pair)
    always claims a hand before a looser one (e.g. a suited Ace) gets the
    chance to -- e.g. AKs is Axs, not Kxs; KQo is an Unsuited Connector, not
    Junk."""
    r1, r2 = hole[0].rank, hole[1].rank
    suited = hole[0].suit == hole[1].suit
    hi, lo = max(r1, r2), min(r1, r2)
    gap = hi - lo  # plain (non-Ace-wrapping) gap: only reached once hi < 14, so no low-Ace case applies

    if hi == lo:
        if hi >= 12:
            return 11  # Premium Pairs: AA/KK/QQ
        if hi >= 9:
            return 10  # High Pairs: JJ-99
        return 9  # Average Pairs: 22-88
    if hi == 14:
        return 8 if suited else 7  # Axs / Axo
    if hi == 13 and suited:
        return 6  # Kxs (Ace already excluded above)
    if hi == 12 and suited:
        return 5  # Qxs (Ace/King already excluded above)
    if suited and gap == 1:
        return 4  # Suited Connectors
    if suited and gap == 2:
        return 3  # Suited 1 Gappers
    if suited and gap == 3:
        return 2  # Suited 2 Gappers
    if not suited and gap == 1:
        return 1  # Unsuited Connectors
    return 0  # Junk


def _connectivity_label(gap: int) -> str:
    if gap == 0:
        return "Same rank (pocket pair)"
    if gap == 1:
        return "1 apart (connectors, e.g. 9-10)"
    if gap == 2:
        return "2 apart (one-gappers, e.g. 9-J)"
    if gap == 3:
        return "3 apart (two-gappers, e.g. 9-Q)"
    if gap < CONNECTIVITY_GAP_CAP:
        return f"{gap} apart"
    return f"{CONNECTIVITY_GAP_CAP} or more apart"


def _linked_bool(key: str, label: str, description: str, linked_to: str, index: int) -> FeatureSpec:
    return FeatureSpec(key, label, description, kind="boolean", linked_to=linked_to, linked_value_index=index)


# Representative sample points used for every genuinely continuous feature, both for
# its value table and for the "nearest point" bucket indicators computed at runtime.
_BUCKET_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)


_NUM_BUCKETS = len(_BUCKET_POINTS)  # 5


def _nearest_bucket_index(value: float, num_buckets: int = _NUM_BUCKETS) -> int:
    """Index into an evenly-spaced `num_buckets`-point grid over 0-1 (by
    default _BUCKET_POINTS, the shared 5-point grid most continuous
    features use) closest to `value`. call_amount_norm and num_raises_norm
    each use their own different bucket count (see _CALL_SIZE_VALUES /
    _RAISES_VALUES), hence the parameter instead of always assuming 5.
    Since the grid is evenly spaced, the nearest point can be found with
    direct arithmetic instead of scanning+comparing every candidate --
    this is called from extract_features' hottest inner loop (once per
    continuous feature per decision), so the constant-factor savings add up.
    `ceil(x - 0.5)` (rather than the more common `round`) is deliberate: it
    breaks an exact tie between two adjacent points toward the *lower*
    index, matching what the original `min(range(...), key=...)` scan did
    (the first-encountered minimum wins ties, and range() visits indices
    ascending) -- plain `round()` would instead break ties to the nearest
    *even* index (banker's rounding) and silently disagree at the exact
    midpoints."""
    last = num_buckets - 1
    idx = math.ceil(value * last - 0.5)
    return 0 if idx < 0 else (last if idx > last else idx)


def _continuous_children(parent_key: str, value_table: tuple) -> list[FeatureSpec]:
    return [
        _linked_bool(
            f"{parent_key}_bucket_{i}", label,
            f"1 if {parent_key}'s value falls closest to {point:.2f} on a 0-1 scale "
            f"(\"{label}\"), else 0. One of these {len(value_table)} is always 1.",
            parent_key, i,
        )
        for i, (point, label) in enumerate(value_table)
    ]


# Every made-hand category, in strength order -- but Pair, Three of a Kind,
# Straight, and Flush are each split into several sub-buckets (still ordered
# weakest-to-strongest within that category) instead of one flat value, so
# distinctions that used to live in a dozen-plus separate standalone
# booleans (top_pair, overpair, ..., "is this a set/nuts straight/ace-high
# flush") now read as *where on the made-hand strength spectrum* a hand
# sits, which a NN can generalize across the way it can't across unrelated
# flags. See _hand_category_bucket for exactly how each index is chosen.
#
# Pair (indices 1-9), weakest to strongest:
#   Low Pair    -- pocket pair below the whole board
#   Underpair   -- pocket pair below the board's top card, but not Low Pair
#   Pair        -- catch-all: a preflop pocket pair (no board yet), or a
#                  postflop pair matching a board rank outside the top 3
#   Third Pair, Second Pair, Top Pair -- a non-pocket hole card matching
#                  the board's 3rd/2nd/1st highest distinct rank
#   Top Pair + Good Kicker  -- Top Pair, kicker Jack/Queen/King
#   Top Pair + Top Kicker   -- Top Pair, kicker Ace
#   Overpair    -- pocket pair above the whole board (the strongest
#                  single-pair hand, hence placed right before Two Pair)
#
# Three of a Kind (indices 11-13): Bottom Set < Set < Top Set, by whether
# the tripped rank is the board's lowest/other/highest distinct rank (a
# non-pocket-pair "trips" via a paired board is treated as the plain
# middle "Set" bucket).
#
# Straight (indices 14-17): Bottom Straight < Straight < Top Straight <
# Nuts Straight, by comparing this straight's high card against every
# straight_high any 2 hole cards could make on this board -- Nuts requires
# also being the *top* straight AND no flush being possible for anyone
# (fewer than 3 board cards sharing a suit), since a possible flush would
# mean a straight -- even the best one -- isn't provably the best hand.
#
# Flush (indices 18-20): Flush < King High Flush < Ace High Flush, by the
# flush's own high card (not this player's overall hole+board high card,
# which can differ if the highest card isn't of the flush suit).
_HAND_CATEGORY_VALUES = (
    (0 / 23, "High Card"),
    (1 / 23, "Low Pair"), (2 / 23, "Underpair"), (3 / 23, "Pair"),
    (4 / 23, "Third Pair"), (5 / 23, "Second Pair"), (6 / 23, "Top Pair"),
    (7 / 23, "Top Pair + Good Kicker"), (8 / 23, "Top Pair + Top Kicker"), (9 / 23, "Overpair"),
    (10 / 23, "Two Pair"),
    (11 / 23, "Bottom Set"), (12 / 23, "Set"), (13 / 23, "Top Set"),
    (14 / 23, "Bottom Straight"), (15 / 23, "Straight"), (16 / 23, "Top Straight"), (17 / 23, "Nuts Straight"),
    (18 / 23, "Flush"), (19 / 23, "King High Flush"), (20 / 23, "Ace High Flush"),
    (21 / 23, "Full House"), (22 / 23, "Four of a Kind"), (23 / 23, "Straight Flush"),
)
_HOLE_CATEGORY_VALUES = tuple((i / 11, label) for i, label in enumerate(_HOLE_CATEGORY_LABELS))
_HOLE_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_SHARED_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_CONNECTIVITY_VALUES = tuple(
    (1.0 - gap / CONNECTIVITY_GAP_CAP, _connectivity_label(gap)) for gap in range(0, CONNECTIVITY_GAP_CAP + 1)
)
_STREET_VALUES = ((0.0, "Preflop"), (1 / 3, "Flop"), (2 / 3, "Turn"), (1.0, "River"))
# 6 evenly-spaced points (0, 0.2, ..., 1.0) map to call/pot ratios 0, 0.25,
# 0.5, 0.75, 1.0, 1.25 once divided by _CALL_SIZE_CEILING below -- chosen so
# the first 5 points land exactly on the classic pot-fraction labels, with
# room left over for a genuine Overbet bucket instead of lumping "exactly
# pot-sized" and "way more than pot" into one clipped top bucket.
_CALL_SIZE_CEILING = 1.25
_CALL_SIZE_VALUES = (
    (0.0, "No bet to call"),
    (0.2, "Call is 1/4 pot"),
    (0.4, "Call is 1/2 pot"),
    (0.6, "Call is 3/4 pot"),
    (0.8, "Call is a full pot-sized bet"),
    (1.0, "Overbet (more than a full pot, clipped)"),
)
_SPR_VALUES = (
    (0.0, "SPR ≈ 0 (effectively all-in already)"),
    (0.25, "SPR ≈ 5 (shallow — about 5 pot-sized bets behind)"),
    (0.5, "SPR ≈ 10 (medium stack depth)"),
    (0.75, "SPR ≈ 15 (deep)"),
    (1.0, "SPR ≈ 20+ (very deep, clipped)"),
)
_POSITION_VALUES = (
    (0.0, "Acts first this street (out of position)"),
    (0.25, "Acts early"),
    (0.5, "Acts in the middle of the order"),
    (0.75, "Acts late"),
    (1.0, "Acts last this street (in position / on the button)"),
)
_ACTIVE_PLAYERS_VALUES = tuple((n / 6.0, f"{n} players still in the hand") for n in range(2, 7))
_OVERCARDS_VALUES = tuple(
    (
        n / 5.0,
        "No board cards rank higher than my high card" if n == 0
        else f"{n} board card{'s' if n != 1 else ''} rank{'s' if n == 1 else ''} higher than my high card",
    )
    for n in range(6)
)
_RAISES_VALUES = (
    (0.0, "No raises yet this street"),
    (1 / 3, "1 raise so far"),
    (2 / 3, "2 raises so far"),
    (1.0, "3 or more raises so far (clipped) -- by this point it's mostly a shove-or-not decision"),
)
_POT_TYPE_VALUES = (
    (0.0, "Unraised Pot"),
    (1 / 3, "Single Raised Pot"),
    (2 / 3, "3-Bet Pot"),
    (3 / 3, "4-Bet+ Pot"),
)
_STACK_DEPTH_VALUES = (
    (0.0, "Busted / no chips left"),
    (0.25, "Half of the starting stack remaining"),
    (0.5, "At the starting stack (100%)"),
    (0.75, "1.5x the starting stack"),
    (1.0, "2x the starting stack or more (clipped)"),
)
_OPP_VPIP_VALUES = (
    (0.0, "Active opponents never voluntarily play a hand (0% VPIP)"),
    (0.25, "Active opponents voluntarily play about 1 in 4 hands (25% VPIP)"),
    (0.5, "Active opponents voluntarily play about half their hands (50% VPIP), or no reads yet this session"),
    (0.75, "Active opponents voluntarily play about 3 in 4 hands (75% VPIP)"),
    (1.0, "Active opponents voluntarily play (almost) every hand (100% VPIP)"),
)
_OPP_PFR_VALUES = (
    (0.0, "Active opponents never raise preflop (0% PFR)"),
    (0.25, "Active opponents raise preflop about 1 in 4 hands (25% PFR)"),
    (0.5, "Active opponents raise preflop about half their hands (50% PFR), or no reads yet this session"),
    (0.75, "Active opponents raise preflop about 3 in 4 hands (75% PFR)"),
    (1.0, "Active opponents raise preflop (almost) every hand they play (100% PFR)"),
)
_OPP_THREE_BET_VALUES = (
    (0.0, "Active opponents never 3-bet when given the chance (0% 3-bet)"),
    (0.25, "Active opponents 3-bet about 1 in 4 chances (25% 3-bet)"),
    (0.5, "Active opponents 3-bet about half their chances (50% 3-bet), or no reads yet this session"),
    (0.75, "Active opponents 3-bet about 3 in 4 chances (75% 3-bet)"),
    (1.0, "Active opponents 3-bet (almost) every chance they get (100% 3-bet)"),
)
_OPP_FOLD_TO_THREE_BET_VALUES = (
    (0.0, "Active opponents never fold to a 3-bet (0% fold-to-3-bet)"),
    (0.25, "Active opponents fold to a 3-bet about 1 in 4 times (25% fold-to-3-bet)"),
    (0.5, "Active opponents fold to a 3-bet about half the time (50% fold-to-3-bet), or no reads yet this session"),
    (0.75, "Active opponents fold to a 3-bet about 3 in 4 times (75% fold-to-3-bet)"),
    (1.0, "Active opponents (almost) always fold to a 3-bet (100% fold-to-3-bet)"),
)
_OPP_AGGRESSION_FREQ_VALUES = (
    (0.0, "Active opponents never bet/raise postflop when they act (0% aggression, all calls)"),
    (0.25, "Active opponents bet/raise postflop about 1 in 4 times they act (25% aggression)"),
    (0.5, "Active opponents bet/raise postflop about half the time (50% aggression), or no reads yet this session"),
    (0.75, "Active opponents bet/raise postflop about 3 in 4 times (75% aggression)"),
    (1.0, "Active opponents (almost) always bet/raise rather than call postflop (100% aggression)"),
)
_OPP_FOLD_VS_BET_VALUES = (
    (0.0, "Active opponents never fold to a postflop bet (0% fold, very sticky)"),
    (0.25, "Active opponents fold to a postflop bet about 1 in 4 times (25%)"),
    (0.5, "Active opponents fold to a postflop bet about half the time (50%), or no reads yet this session"),
    (0.75, "Active opponents fold to a postflop bet about 3 in 4 times (75%)"),
    (1.0, "Active opponents (almost) always fold to a postflop bet (100%)"),
)
_OPP_VPIP_CHILDREN = _continuous_children("opp_vpip_norm", _OPP_VPIP_VALUES)
_OPP_PFR_CHILDREN = _continuous_children("opp_pfr_norm", _OPP_PFR_VALUES)
_OPP_THREE_BET_CHILDREN = _continuous_children("opp_three_bet_norm", _OPP_THREE_BET_VALUES)
_OPP_FOLD_TO_THREE_BET_CHILDREN = _continuous_children(
    "opp_fold_to_three_bet_norm", _OPP_FOLD_TO_THREE_BET_VALUES
)
_OPP_AGGRESSION_FREQ_CHILDREN = _continuous_children("opp_aggression_freq_norm", _OPP_AGGRESSION_FREQ_VALUES)
_OPP_FOLD_VS_BET_CHILDREN = _continuous_children("opp_fold_vs_bet_norm", _OPP_FOLD_VS_BET_VALUES)

_SEAT_ROLE_LABELS = {
    "UTG": "Under The Gun (UTG)", "HJ": "Hijack (HJ)", "CO": "Cutoff (CO)",
    "BTN": "Button (BTN)", "SB": "Small Blind (SB)", "BB": "Big Blind (BB)",
}
_SEAT_ROLE_VALUES = tuple((i / (len(SEAT_ROLES) - 1), _SEAT_ROLE_LABELS[role]) for i, role in enumerate(SEAT_ROLES))

_HAND_CATEGORY_CHILDREN = [
    _linked_bool(
        "has_high_card", "High Card",
        "1 if the best available hand's category is exactly High Card (no pair or better), else 0.",
        "hand_category_norm", 0,
    ),
    _linked_bool(
        "has_low_pair", "Low Pair",
        "1 if the hole cards are a pocket pair ranked lower than every board card, else 0.",
        "hand_category_norm", 1,
    ),
    _linked_bool(
        "has_underpair", "Underpair",
        "1 if the hole cards are a pocket pair ranked lower than the board's highest card, "
        "but not Low Pair (it still beats at least one board card), else 0.",
        "hand_category_norm", 2,
    ),
    _linked_bool(
        "has_pair", "Pair",
        "1 if the best available hand is exactly a pair that doesn't fit any of the more "
        "specific pair buckets around it -- a preflop pocket pair with no board yet to "
        "compare against, or a postflop pair matching a board rank below the top 3 "
        "distinct ranks -- else 0.",
        "hand_category_norm", 3,
    ),
    _linked_bool(
        "has_third_pair", "Third Pair",
        "1 if exactly one hole card rank-matches the third-highest distinct board rank "
        "(and the best overall hand is exactly a pair), else 0. Needs a board with at "
        "least 3 distinct ranks.",
        "hand_category_norm", 4,
    ),
    _linked_bool(
        "has_second_pair", "Second Pair",
        "1 if exactly one hole card rank-matches the second-highest distinct board rank "
        "(and the best overall hand is exactly a pair), else 0. Needs a board with at "
        "least 2 distinct ranks.",
        "hand_category_norm", 5,
    ),
    _linked_bool(
        "has_top_pair", "Top Pair",
        "1 if exactly one hole card rank-matches the single highest board card, the best "
        "overall hand is exactly a pair, and the other hole card (the kicker) is a Ten or "
        "below -- a Jack, Queen, King, or Ace kicker gets its own Good/Top Kicker bucket "
        "instead -- else 0.",
        "hand_category_norm", 6,
    ),
    _linked_bool(
        "has_top_pair_good_kicker", "Top Pair + Good Kicker",
        "1 if this is Top Pair (see has_top_pair) and the kicker is a Jack, Queen, or "
        "King, else 0.",
        "hand_category_norm", 7,
    ),
    _linked_bool(
        "has_top_pair_top_kicker", "Top Pair + Top Kicker",
        "1 if this is Top Pair (see has_top_pair) and the kicker is an Ace, else 0.",
        "hand_category_norm", 8,
    ),
    _linked_bool(
        "has_overpair", "Overpair",
        "1 if the hole cards are a pocket pair ranked higher than every board card, else 0.",
        "hand_category_norm", 9,
    ),
    _linked_bool(
        "has_two_pair", "Two Pair", "1 if the best available hand is exactly two pair, else 0.",
        "hand_category_norm", 10,
    ),
    _linked_bool(
        "has_bottom_set", "Bottom Set",
        "1 if the hole cards are a pocket pair matching the board's single lowest distinct "
        "rank (the worst possible three of a kind on this board), else 0.",
        "hand_category_norm", 11,
    ),
    _linked_bool(
        "has_set", "Set",
        "1 if the best available hand is exactly three of a kind and it's neither Bottom "
        "Set nor Top Set -- either a middle set (a pocket pair matching a board rank "
        "that's neither the board's highest nor lowest) or the classic board-paired "
        "'trips' case (a non-pocket hole card matching a pair already on the board) -- "
        "else 0.",
        "hand_category_norm", 12,
    ),
    _linked_bool(
        "has_top_set", "Top Set",
        "1 if the hole cards are a pocket pair matching the board's single highest "
        "distinct rank (the best possible three of a kind on this board), else 0.",
        "hand_category_norm", 13,
    ),
    _linked_bool(
        "has_bottom_straight", "Bottom Straight",
        "1 if the best available hand is exactly a straight, and its high card is the "
        "lowest straight_high any 2 hole cards could make on this board, else 0.",
        "hand_category_norm", 14,
    ),
    _linked_bool(
        "has_straight", "Straight",
        "1 if the best available hand is exactly a straight that's neither the highest "
        "nor lowest one possible on this board, else 0.",
        "hand_category_norm", 15,
    ),
    _linked_bool(
        "has_top_straight", "Top Straight",
        "1 if the best available hand is exactly a straight, its high card is the highest "
        "straight_high any 2 hole cards could make on this board, and a flush is still "
        "possible for someone (3+ board cards share a suit) -- so it's the best straight "
        "here but not provably the best hand overall, else 0.",
        "hand_category_norm", 16,
    ),
    _linked_bool(
        "has_nuts_straight", "Nuts Straight",
        "1 if the best available hand is exactly a straight, its high card is the highest "
        "straight_high any 2 hole cards could make on this board, and no flush is "
        "possible for anyone (fewer than 3 board cards share any one suit) -- so no "
        "better hand is possible given the shared cards, else 0.",
        "hand_category_norm", 17,
    ),
    _linked_bool(
        "has_flush", "Flush",
        "1 if the best available hand is exactly a flush and its own highest card isn't "
        "an Ace or King, else 0.",
        "hand_category_norm", 18,
    ),
    _linked_bool(
        "has_king_high_flush", "King High Flush",
        "1 if the best available hand is exactly a flush and its own highest card is a "
        "King, else 0.",
        "hand_category_norm", 19,
    ),
    _linked_bool(
        "has_ace_high_flush", "Ace High Flush",
        "1 if the best available hand is exactly a flush and its own highest card is an "
        "Ace (the best possible flush of that suit), else 0.",
        "hand_category_norm", 20,
    ),
    _linked_bool(
        "has_full_house", "Full House", "1 if the best available hand is exactly a full house, else 0.",
        "hand_category_norm", 21,
    ),
    _linked_bool(
        "has_quads", "Four of a Kind", "1 if the best available hand is exactly four of a kind, else 0.",
        "hand_category_norm", 22,
    ),
    _linked_bool(
        "has_straight_flush", "Straight Flush", "1 if the best available hand is a straight flush, else 0.",
        "hand_category_norm", 23,
    ),
]

_HOLE_HIGH_CARD_CHILDREN = [
    _linked_bool(
        f"hole_high_card_is_{_rank_key(r)}", _rank_label(r),
        f"1 if the higher of this player's two hole cards has rank {_rank_label(r)}, else 0.",
        "hole_high_card_norm", r - 2,
    )
    for r in range(2, 15)
]

_SHARED_HIGH_CARD_CHILDREN = [
    _linked_bool(
        f"shared_high_card_is_{_rank_key(r)}", _rank_label(r),
        f"1 if the highest board (shared/community) card has rank {_rank_label(r)}, else 0. "
        "Never true preflop, since there are no shared cards yet.",
        "shared_high_card_norm", r - 2,
    )
    for r in range(2, 15)
]

_CONNECTIVITY_CHILDREN = [
    _linked_bool(
        "hole_paired", "Same rank (pocket pair)", "1 if both hole cards share a rank, else 0.",
        "hole_connectivity", 0,
    ),
] + [
    _linked_bool(
        f"connectivity_gap_{gap}", _connectivity_label(gap),
        f"1 if the two hole card ranks are exactly {gap} apart (treating the Ace as high or "
        "low, whichever is closer), else 0.",
        "hole_connectivity", gap,
    )
    for gap in range(1, CONNECTIVITY_GAP_CAP)
] + [
    _linked_bool(
        f"connectivity_gap_{CONNECTIVITY_GAP_CAP}plus", _connectivity_label(CONNECTIVITY_GAP_CAP),
        f"1 if the two hole card ranks are {CONNECTIVITY_GAP_CAP} or more apart (treating the "
        "Ace as high or low, whichever is closer), else 0.",
        "hole_connectivity", CONNECTIVITY_GAP_CAP,
    )
]

_HOLE_CATEGORY_KEYS_BY_INDEX = (
    "hole_category_junk", "hole_category_unsuited_connector", "hole_category_suited_two_gapper",
    "hole_category_suited_one_gapper", "hole_category_suited_connector", "hole_category_qxs",
    "hole_category_kxs", "hole_category_axo", "hole_category_axs", "hole_category_average_pair",
    "hole_category_high_pair", "hole_category_premium_pair",
)
_HOLE_CATEGORY_DESCRIPTIONS = (
    "the hand doesn't fit any other bucket (e.g. K9o, J4s, 73o)",
    "the two hole cards are offsuit and 1 apart in rank (e.g. KQo, T9o)",
    "the two hole cards are suited and 3 apart in rank (e.g. J8s, T7s)",
    "the two hole cards are suited and 2 apart in rank (e.g. J9s, T8s)",
    "the two hole cards are suited and 1 apart in rank (e.g. JTs, T9s)",
    "one hole card is a Queen, the other is a suited non-Ace, non-King kicker (e.g. QJs, Q4s)",
    "one hole card is a King, the other is a suited non-Ace kicker (e.g. KQs, K4s)",
    "one hole card is an Ace, the other is an offsuit kicker (e.g. AKo, A4o)",
    "one hole card is an Ace, the other is a suited kicker (e.g. AKs, A4s)",
    "the hole cards are a pocket pair from 22 to 88",
    "the hole cards are a pocket pair from 99 to JJ",
    "the hole cards are a pocket pair of Aces, Kings, or Queens",
)
_HOLE_CATEGORY_CHILDREN = [
    _linked_bool(
        _HOLE_CATEGORY_KEYS_BY_INDEX[i], _HOLE_CATEGORY_LABELS[i],
        f"1 if {_HOLE_CATEGORY_DESCRIPTIONS[i]}, else 0.",
        "hole_hand_category_norm", i,
    )
    for i in range(len(_HOLE_CATEGORY_LABELS))
]

_STREET_CHILDREN = [
    _linked_bool(key, label, f"1 if the current street is {label}, else 0.", "street_norm", i)
    for i, (key, label) in enumerate(
        [("is_preflop", "Preflop"), ("is_flop", "Flop"), ("is_turn", "Turn"), ("is_river", "River")]
    )
]

_CALL_SIZE_CHILDREN = _continuous_children("call_amount_norm", _CALL_SIZE_VALUES)
_SPR_CHILDREN = _continuous_children("spr_norm", _SPR_VALUES)
_POSITION_CHILDREN = _continuous_children("position_norm", _POSITION_VALUES)
_STACK_DEPTH_CHILDREN = _continuous_children("stack_depth_norm", _STACK_DEPTH_VALUES)

_ACTIVE_PLAYERS_CHILDREN = [
    _linked_bool(
        f"active_players_is_{n}", f"{n} players",
        f"1 if exactly {n} players are still in the hand, else 0.",
        "num_active_norm", n - 2,
    )
    for n in range(2, 7)
]

_OVERCARDS_CHILDREN = [
    _linked_bool(
        f"num_overcards_is_{n}", ("0 overcards" if n == 0 else f"{n} overcard{'s' if n != 1 else ''}"),
        f"1 if exactly {n} board cards rank higher than the higher of this player's two hole "
        "cards, else 0.",
        "num_overcards_norm", n,
    )
    for n in range(6)
]

_RAISES_CHILDREN = [
    _linked_bool(
        f"raises_is_{r}", ("No raises" if r == 0 else f"{r} raise{'s' if r != 1 else ''}"),
        f"1 if exactly {r} raises have occurred so far this street, else 0.",
        "num_raises_norm", r,
    )
    for r in range(0, 3)
] + [
    _linked_bool(
        "raises_is_3plus", "3+ raises",
        "1 if 3 or more raises have occurred so far this street, else 0.",
        "num_raises_norm", 3,
    )
]

_POT_TYPE_CHILDREN = [
    _linked_bool(
        "is_unraised_pot", "Unraised Pot",
        "1 if there have been no preflop raises (everyone limped/checked/folded to the big "
        "blind), else 0.",
        "pot_type_norm", 0,
    ),
    _linked_bool(
        "is_single_raised_pot", "Single Raised Pot",
        "1 if there has been exactly 1 preflop raise (the standard 'open'), else 0.",
        "pot_type_norm", 1,
    ),
    _linked_bool(
        "is_3bet_pot", "3-Bet Pot",
        "1 if there have been exactly 2 preflop raises (someone re-raised the opener), else 0.",
        "pot_type_norm", 2,
    ),
    _linked_bool(
        "is_4bet_plus_pot", "4-Bet+ Pot",
        "1 if there have been 3 or more preflop raises, else 0.",
        "pot_type_norm", 3,
    ),
]

_SEAT_ROLE_KEYS = {
    "UTG": "is_utg", "HJ": "is_hijack", "CO": "is_cutoff",
    "BTN": "is_button", "SB": "is_small_blind", "BB": "is_big_blind",
}
_SEAT_ROLE_CHILDREN = [
    _linked_bool(
        _SEAT_ROLE_KEYS[role], _SEAT_ROLE_LABELS[role],
        f"1 if this player's starting seat this hand was {_SEAT_ROLE_LABELS[role]}, else 0.",
        "starting_position_norm", i,
    )
    for i, role in enumerate(SEAT_ROLES)
]

# Flop-texture features describe the flop (board[:3]) and, for the
# connectivity family below, how this player's hole cards personally
# connect to it -- still frozen once the flop is dealt, since neither the
# flop cards nor the hole cards change on the turn/river. All default to
# their lowest-value category before the flop (0 for every child, since
# nothing has happened yet). flop_suit_texture_norm itself is purely about
# the flop's own 3 cards (no hole-card awareness at all, unlike the
# connectivity family) -- see suit_connection_index below for the
# hole-card-aware, non-frozen version of "how connected am I to these suits."
_FLOP_SUIT_TEXTURE_VALUES = ((0.0, "Rainbow"), (0.5, "Flush Draw Flop"), (1.0, "Monotone"))
_FLOP_PAIRING_VALUES = ((0.0, "Unpaired"), (0.5, "Paired"), (1.0, "Tripled"))
_FLOP_CONNECTIVITY_VALUES = (
    (0 / 2, "Disconnected"), (1 / 2, "Connected, No Straight Draw"),
    (2 / 2, "Connected, Straight Draw (4+/8+ Outs)"),
)
_FLOP_WETNESS_VALUES = ((0.0, "Dry"), (1.0, "Wet"))
_FLOP_DYNAMISM_VALUES = ((0.0, "Static"), (1.0, "Dynamic"))

# Unlike the flop-texture family above, suit_connection_index is never frozen
# -- it's recomputed from hole + the *current* board every time
# extract_features runs, so it keeps updating on the turn and river as more
# board cards (and so, potentially, more of this player's suit) show up.
# Pigeonhole guarantees at least 2 once the flop is out (5 cards split across
# 4 suits), rising to the 5-card cap as a flush comes together; preflop
# (2 hole cards only) it's just whether they're suited -- 1 if not, 2 if so.
_SUIT_CONNECTION_VALUES = (
    (1 / 5, "1 Card (unsuited hole cards, no board yet)"), (2 / 5, "2 Cards (suited, or the flop's pigeonhole floor)"),
    (3 / 5, "3 Cards"), (4 / 5, "4 Cards (a flush draw)"), (5 / 5, "5+ Cards (a flush made)"),
)

# flop_suit_texture_norm's own one-hot family: exact indicators for its 3
# suit-family values, no hole-card awareness (see suit_connection_index for
# that dimension, which is entirely separate now rather than folded in).
_FLOP_SUIT_TEXTURE_CHILDREN = [
    _linked_bool(
        "rainbow_flop", "Rainbow",
        "1 if the flop's 3 cards are all different suits, else 0.",
        "flop_suit_texture_norm", 0,
    ),
    _linked_bool(
        "flush_draw_flop", "Flush Draw Flop",
        "1 if exactly 2 of the flop's 3 cards share a suit (a two-tone flop), else 0.",
        "flop_suit_texture_norm", 1,
    ),
    _linked_bool(
        "monotone_flop", "Monotone",
        "1 if all 3 flop cards share the same suit, else 0.",
        "flop_suit_texture_norm", 2,
    ),
]
_SUIT_CONNECTION_CHILDREN = [
    _linked_bool(
        f"suit_connection_is_{n}", f"{n} Card{'s' if n != 1 else ''} Same Suit" if n < 5 else "5+ Cards Same Suit",
        f"1 if the most cards of any single suit among this player's hole cards plus "
        f"the current board is exactly {n}{' or more' if n == 5 else ''}, else 0.",
        "suit_connection_index", n - 1,
    )
    for n in range(1, 6)
]
_FLOP_PAIRING_CHILDREN = [
    _linked_bool(
        "unpaired_flop", "Unpaired",
        "1 if the flop's 3 cards all have different ranks, else 0.",
        "flop_pairing_texture_norm", 0,
    ),
    _linked_bool(
        "paired_flop", "Paired",
        "1 if exactly 2 of the flop's 3 cards share a rank, else 0.",
        "flop_pairing_texture_norm", 1,
    ),
    _linked_bool(
        "tripled_flop", "Tripled",
        "1 if all 3 flop cards share the same rank, else 0.",
        "flop_pairing_texture_norm", 2,
    ),
]
# "connected_flop" is a family-level aggregate (indices 1+2 of the 3-value
# family below), so -- like the suit family booleans above -- it's declared
# standalone rather than as a `linked_to` child.
_FLOP_CONNECTED_FAMILY_SPEC = FeatureSpec(
    "connected_flop", "Connected",
    "1 if the flop's ranks (Ace counted as high or low, whichever is closer) span 4 "
    "or less, regardless of whether this player actually has a straight draw there "
    "(see flop_connectivity_norm for the hole-card-aware breakdown), else 0.",
    group="Board / Flop Characteristics",
)
_FLOP_CONNECTIVITY_CHILDREN = [
    _linked_bool(
        "disconnected_flop", "Disconnected",
        "1 if the flop's ranks (Ace counted as high or low, whichever is closer) span more "
        "than 4, too spread out to easily combine into straights, else 0.",
        "flop_connectivity_norm", 0,
    ),
    _linked_bool(
        "connected_no_straight_draw_flop", "Connected, No Straight Draw",
        "1 if the flop is Connected (see flop_connectivity_norm) but this player's "
        "hole+flop cards have no rank that would complete a straight, else 0.",
        "flop_connectivity_norm", 1,
    ),
    _linked_bool(
        "connected_straight_draw_flop", "Connected, Straight Draw (4+/8+ Outs)",
        "1 if the flop is Connected and this player has a live straight draw there -- "
        "a gutshot (1 rank, ~4 real outs) or open-ended/double-gutshot (2 ranks, ~8 "
        "real outs) -- else 0.",
        "flop_connectivity_norm", 2,
    ),
]
_FLOP_WETNESS_CHILDREN = [
    _linked_bool(
        "dry_flop", "Dry",
        "1 if the flop is Dry (see flop_wetness_norm for the underlying score), else 0.",
        "flop_wetness_norm", 0,
    ),
    _linked_bool(
        "wet_flop", "Wet",
        "1 if the flop is Wet (see flop_wetness_norm for the underlying score), else 0.",
        "flop_wetness_norm", 1,
    ),
]
_FLOP_DYNAMISM_CHILDREN = [
    _linked_bool(
        "static_flop", "Static",
        "1 if the flop is Static (see flop_dynamism_norm), else 0.",
        "flop_dynamism_norm", 0,
    ),
    _linked_bool(
        "dynamic_flop", "Dynamic",
        "1 if the flop is Dynamic (see flop_dynamism_norm), else 0.",
        "flop_dynamism_norm", 1,
    ),
]


# Grouped as parent-feature-then-its-children for readability; array order is
# otherwise arbitrary since extract_features looks values up by key, not position.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "hand_category_norm", "Hand Strength Tier",
        "Made-hand category of the best hand available from hole cards plus the current "
        "board, normalized to 0-1 as bucket_index / 23. Standard High Card < Pair < Two "
        "Pair < Three of a Kind < Straight < Flush < Full House < Four of a Kind < "
        "Straight Flush ordering, but Pair/Three of a Kind/Straight/Flush are each split "
        "into several ordered sub-buckets instead of one flat value -- e.g. Pair spans "
        "Low Pair, Underpair, Pair, Third Pair, Second Pair, Top Pair, Top Pair + Good "
        "Kicker, Top Pair + Top Kicker, and Overpair, weakest to strongest -- folding in "
        "what used to be a dozen-plus separate standalone booleans (top_pair, overpair, "
        "...) directly into this ordinal scale. See _HAND_CATEGORY_VALUES for every "
        "bucket and _hand_category_bucket for exactly how each one is computed.",
        kind="categorical", value_table=_HAND_CATEGORY_VALUES, group="Made Hand Features",
    ),
    *_HAND_CATEGORY_CHILDREN,

    FeatureSpec(
        "hole_high_card_norm", "Hole High Card Rank",
        "Rank of the higher of this player's two hole cards, normalized to 0-1 via "
        "(rank - 2) / 12, so 0.0 = deuce and 1.0 = ace. Always defined, from preflop on.",
        kind="categorical", value_table=_HOLE_HIGH_CARD_VALUES, group="Hole Card Characteristics",
    ),
    *_HOLE_HIGH_CARD_CHILDREN,

    FeatureSpec(
        "shared_high_card_norm", "Shared Cards High Card Rank",
        "Rank of the highest board (shared/community) card, normalized to 0-1 via "
        "(rank - 2) / 12. Defaults to 0.0 (the deuce value) preflop, before any shared "
        "cards are dealt -- pair with the street/is_preflop features to tell 'no board "
        "yet' apart from 'the board's high card really is a deuce'.",
        kind="categorical", value_table=_SHARED_HIGH_CARD_VALUES, group="Board / Flop Characteristics",
    ),
    *_SHARED_HIGH_CARD_CHILDREN,

    FeatureSpec(
        "hole_connectivity", "Hole Card Connectivity",
        "How close together the two hole card ranks are for straight-making purposes: "
        "1 - gap / 6, where gap is the rank distance (Ace counted as high or low, "
        "whichever gives the smaller gap -- so A-K and A-2 are both gap 1) capped at 6, "
        "since anything 6+ apart is equally unable to share a straight. 1.0 means "
        "consecutive ranks (e.g. 9-T); 0.0 means 6 or more apart.",
        kind="categorical", value_table=_CONNECTIVITY_VALUES, group="Hole Card Characteristics",
    ),
    *_CONNECTIVITY_CHILDREN,

    FeatureSpec(
        "hole_hand_category_norm", "Hole Hand Category",
        "Which of 12 mutually exclusive starting-hand buckets the two hole cards fall "
        "into, checked strongest-to-weakest so exactly one applies -- a pair always beats "
        "a suited Ace to its bucket, a suited Ace always beats Kxs, and so on. Priority "
        "order (strongest first): Premium Pairs (AA/KK/QQ) > High Pairs (JJ-99) > Average "
        "Pairs (22-88) > Axs (Ace + suited kicker) > Axo (Ace + offsuit kicker) > Kxs "
        "(King + suited kicker, no Ace) > Qxs (Queen + suited kicker, no Ace/King) > "
        "Suited Connectors (1 apart) > Suited 1 Gappers (2 apart) > Suited 2 Gappers "
        "(3 apart) > Unsuited Connectors (1 apart) > Junk (everything else). Normalized "
        "as bucket_index / 11, where bucket_index counts up from 0 = Junk to 11 = "
        "Premium Pairs, so higher is a stronger starting hand.",
        kind="categorical", value_table=_HOLE_CATEGORY_VALUES, group="Hole Card Characteristics",
    ),
    *_HOLE_CATEGORY_CHILDREN,

    FeatureSpec(
        "street_norm", "Betting Street",
        "Current street normalized to 0-1: 0.0 = preflop, 0.33 = flop, 0.67 = turn, 1.0 = river.",
        kind="categorical", value_table=_STREET_VALUES, group="Table & Game State Features",
    ),
    *_STREET_CHILDREN,

    FeatureSpec(
        "call_amount_norm", "Call Size Vs Pot",
        "Amount required to call, as a fraction of the current pot (call_amount / pot), "
        "divided by 1.25 and clipped to 0-1 -- so a call up to a full pot-sized bet maps "
        "onto the first 5 evenly-spaced points (0, 1/4, 1/2, 3/4, full pot) and anything "
        "noticeably bigger than that (an overbet) clips into its own top bucket instead "
        "of being lumped in with an exactly-pot-sized bet.",
        kind="continuous", value_table=_CALL_SIZE_VALUES, group="Betting Behaviour Features",
    ),
    *_CALL_SIZE_CHILDREN,

    FeatureSpec(
        "spr_norm", "Stack-To-Pot Ratio",
        "Effective stack (the smaller of this player's stack and the largest "
        "active opponent's stack) divided by the current pot, indicating how many "
        "pot-sized bets remain; divided by 20 and clipped to 0-1.",
        kind="continuous", value_table=_SPR_VALUES, group="Stack & Pot Features",
    ),
    *_SPR_CHILDREN,

    FeatureSpec(
        "position_norm", "Table Position",
        "How late this player acts in the current street's action order, from "
        "0.0 (acts first) to 1.0 (acts last).",
        kind="continuous", value_table=_POSITION_VALUES, group="Table & Game State Features",
    ),
    *_POSITION_CHILDREN,

    FeatureSpec(
        "starting_position_norm", "Starting Seat Position",
        "Which of the 6 standard preflop seat roles this player started the hand in, "
        "ordered from first-to-act preflop through the blinds -- UTG, HJ, CO, BTN, SB, BB "
        "-- normalized as role_index / 5. At smaller table sizes the earliest non-blind "
        "roles collapse (e.g. 4-handed there's only UTG and BTN; heads-up the button and "
        "small blind are the same seat, labeled SB).",
        kind="categorical", value_table=_SEAT_ROLE_VALUES, group="Table & Game State Features",
    ),
    *_SEAT_ROLE_CHILDREN,

    FeatureSpec(
        "num_active_norm", "Players Still In Hand",
        "Number of players who have not folded yet this hand, divided by 6.",
        kind="categorical", value_table=_ACTIVE_PLAYERS_VALUES, group="Table & Game State Features",
    ),
    *_ACTIVE_PLAYERS_CHILDREN,

    FeatureSpec(
        "num_raises_norm", "Raises This Street",
        "Number of bets/raises made so far on the current street, divided by 3 and clipped "
        "to 0-1 -- capped at 3 rather than higher, since by the time a street has seen 3 "
        "raises the remaining decision is essentially just shove-or-not regardless of "
        "exactly how many more re-raises it's technically been.",
        kind="categorical", value_table=_RAISES_VALUES, group="Betting Behaviour Features",
    ),
    *_RAISES_CHILDREN,

    FeatureSpec(
        "pot_type_norm", "Pot Type",
        "How many preflop raises shaped this pot, normalized as raise_count / 3 (capped at "
        "3): Unraised (0) < Single Raised (1) < 3-Bet (2) < 4-Bet+ (3 or more). Unlike "
        "Raises This Street, this doesn't reset postflop -- it's frozen at the final preflop "
        "count once the flop is dealt, since 'we're in a 3-bet pot' describes the whole hand, "
        "not just the current street.",
        kind="categorical", value_table=_POT_TYPE_VALUES, group="Betting Behaviour Features",
    ),
    *_POT_TYPE_CHILDREN,

    FeatureSpec(
        "stack_depth_norm", "Stack Depth",
        "This player's remaining stack as a fraction of the session's starting "
        "stack, divided by 2 and clipped to 0-1 (1.0 = double the starting stack or more).",
        kind="continuous", value_table=_STACK_DEPTH_VALUES, group="Stack & Pot Features",
    ),
    *_STACK_DEPTH_CHILDREN,

    # Standalone booleans: not tied to a specific value of any other feature.
    FeatureSpec(
        "flush_draw", "Flush Draw",
        "1 if exactly 4 of the hole+board cards share one suit (a live one-card "
        "flush draw), else 0. Always 0 preflop, since a flush needs 5+ cards.",
        group="Draw Features",
    ),
    FeatureSpec(
        "straight_draw", "Straight Draw",
        "1 if there exists a single rank which, if added to the hole+board cards, "
        "would complete a straight (open-ended or gutshot), else 0. Always 0 preflop.",
        group="Draw Features",
    ),
    FeatureSpec(
        "hole_suited", "Suited Hole Cards", "1 if both hole cards share a suit, else 0.",
        group="Hole Card Characteristics",
    ),
    FeatureSpec(
        "facing_bet", "Facing A Bet",
        "1 if a nonzero amount is required to call (someone has already bet or "
        "raised this street), else 0 (checking is free).",
        group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "is_aggressor", "Last Aggressor",
        "1 if this player made the most recent bet/raise on the *previous* street, else 0 "
        "(always 0 preflop, since there's no previous street). Not the current street: a "
        "player who just raised is skipped over until either the street ends or someone "
        "else re-raises, at which point they're no longer that street's own aggressor -- "
        "so at the moment any decision is actually made, 'I raised this street' is always "
        "false, making the previous street the only version of this that's ever useful.",
        group="Betting Behaviour Features",
    ),

    FeatureSpec(
        "flop_suit_texture_norm", "Flop Suit Texture",
        "How suit-coordinated the flop is, purely from its own 3 cards -- Rainbow "
        "(all 3 different suits) < Flush Draw Flop (exactly 2 share a suit) < "
        "Monotone (all 3 share a suit), normalized as family_index / 2. Doesn't "
        "reflect this player's own hole cards at all -- see suit_connection_index "
        "for how many cards of a single suit this player personally has going "
        "(hole + board combined, and not frozen -- it keeps updating on the turn "
        "and river). Frozen once the flop is dealt (unaffected by the turn/river, "
        "since the flop's own cards don't change); defaults to 0.0 (Rainbow's "
        "value) before the flop.",
        kind="categorical", value_table=_FLOP_SUIT_TEXTURE_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_SUIT_TEXTURE_CHILDREN,

    FeatureSpec(
        "suit_connection_index", "Suit Connection",
        "The most cards of any single suit among this player's hole cards plus "
        "the current board (however many board cards are out so far), capped at "
        "5 and normalized as count / 5. Unlike flop_suit_texture_norm above, this "
        "is never frozen -- it's recomputed from the *current* board every time, "
        "so it keeps updating on the turn and river as more of this player's suit "
        "potentially shows up. Preflop (2 hole cards only) it's just whether "
        "they're suited: 1 card if not, 2 if so. Once the flop is out, pigeonhole "
        "guarantees at least 2 (5 cards split across 4 suits), rising toward the "
        "5-card cap as a flush comes together.",
        kind="categorical", value_table=_SUIT_CONNECTION_VALUES, group="Board / Flop Characteristics",
    ),
    *_SUIT_CONNECTION_CHILDREN,

    FeatureSpec(
        "flop_pairing_texture_norm", "Flop Pairing Texture",
        "How rank-coordinated the flop is: Unpaired (3 different ranks) < Paired (2 "
        "share a rank) < Tripled (all 3 share a rank), normalized as index / 2. Frozen "
        "once the flop is dealt; defaults to 0.0 (Unpaired's value) before the flop.",
        kind="categorical", value_table=_FLOP_PAIRING_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_PAIRING_CHILDREN,

    FeatureSpec(
        "flop_connectivity_norm", "Flop Connectivity",
        "Whether the flop's 3 ranks (Ace counted as high or low, whichever is closer) "
        "are close enough together to make straights possible, and if so, whether this "
        "player's hole cards actually give them a straight draw there: Disconnected if "
        "the ranks span more than 4 (too spread out for any hole cards to complete a "
        "straight); otherwise Connected, split into 'No Straight Draw' (this player's "
        "hole+flop cards have no rank that would complete a straight) and 'Straight "
        "Draw (4+/8+ Outs)' (a gutshot or open-ended/double-gutshot is live). "
        "Normalized as bucket_index / 2. Frozen once the flop is dealt; defaults to "
        "0.0 (Disconnected's value) before the flop.",
        kind="categorical", value_table=_FLOP_CONNECTIVITY_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_CONNECTIVITY_CHILDREN,
    _FLOP_CONNECTED_FAMILY_SPEC,

    FeatureSpec(
        "oesd_possible_flop", "Open Ended Straight Draw Possible Flop",
        "1 if the flop's 3 ranks are all different and span 3 or less (Ace counted as "
        "high or low, whichever is closer) -- tight enough that an open-ended straight "
        "draw is possible for some hole cards -- else 0. A stricter subset of Connected "
        "Flop (not one of its Connected/Disconnected partition values), so it isn't "
        "folded into that table. 0 before the flop.",
        group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "flop_wetness_norm", "Flop Wet vs Dry",
        "How draw-heavy the flop is, purely from its own suit and rank coordination "
        "(not this player's hole cards -- see suit_connection_index and "
        "flop_connectivity_norm for the hole-card-aware versions), combined into one "
        "0-4 score: suit family index (0 Rainbow / 1 Flush Draw Flop / 2 Monotone) + 1 "
        "if the flop's ranks are Connected (span <=4) + 1 if an open-ended straight "
        "draw is possible for some hole cards (see oesd_possible_flop) -- then "
        "thresholded at >=2: Wet (1.0) at or above the threshold, Dry (0.0) below it. "
        "A monotone flop is always Wet on suits alone; a rainbow, disconnected flop is "
        "always Dry. Frozen once the flop is dealt; defaults to 0.0 (Dry's value) "
        "before the flop.",
        kind="categorical", value_table=_FLOP_WETNESS_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_WETNESS_CHILDREN,

    FeatureSpec(
        "flop_dynamism_norm", "Flop Static vs Dynamic",
        "Whether the flop's made-hand hierarchy is likely to get shuffled by the turn "
        "and river: Dynamic (1.0) if the flop is unpaired and its wetness score (see "
        "flop_wetness_norm) plus a high-card adjustment is >=2, else Static (0.0). The "
        "adjustment is -1 when the flop's highest card is a Queen, King, or Ace (fewer "
        "hole cards can credibly out-flop the pair/overpair it hands out, so fewer "
        "runouts change who's ahead), +1 when the flop's highest card is 8 or below "
        "(more hole-card combos have live equity against a low top card), and 0 "
        "otherwise. A paired or tripled flop is always treated as Static regardless of "
        "wetness or card height, since the hierarchy it sets up (trips/boat beats two "
        "pair beats top pair) rarely gets overturned by later streets. Frozen once the "
        "flop is dealt; defaults to 0.0 (Static's value) before the flop.",
        kind="categorical", value_table=_FLOP_DYNAMISM_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_DYNAMISM_CHILDREN,

    FeatureSpec(
        "num_overcards_norm", "Board Overcards",
        "Number of board (community/shared) cards that rank higher than the higher of "
        "this player's two hole cards, normalized to 0-1 via count / 5 (the maximum "
        "possible number of shared cards). 0.0 preflop, since there's no board yet. "
        "Distinct from Shared Cards High Card Rank, which reports the board's single "
        "highest rank regardless of this player's own hole cards.",
        kind="categorical", value_table=_OVERCARDS_VALUES, group="Made Hand Features",
    ),
    *_OVERCARDS_CHILDREN,

    # Hand-vs-board heuristics: not mutually exclusive as a group, so unlike
    # the categorical features above these aren't organized as a one-hot
    # family with a linear parent. (Pair-strength heuristics like Top Pair/
    # Overpair used to live here too, but are now sub-buckets of
    # hand_category_norm instead -- see _HAND_CATEGORY_VALUES.)
    FeatureSpec(
        "ace_high_no_pair", "Ace High (No Made Hand)",
        "1 if the best overall hand is exactly High Card (no pair or better) and the "
        "highest card among hole+board is an Ace, else 0. Different from Hole/Shared "
        "High Card Rank, which report the highest rank regardless of made-hand status.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "king_high_no_pair", "King High (No Made Hand)",
        "1 if the best overall hand is exactly High Card (no pair or better) and the "
        "highest card among hole+board is a King, else 0.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "combo_draw", "Combo Draw",
        "1 if this player has both a flush draw and a straight draw (open-ended or "
        "gutshot) at the same time, else 0.",
        group="Draw Features",
    ),
    FeatureSpec(
        "nuts_flush_draw", "Nuts Flush Draw",
        "1 if this player has a flush draw and holds the Ace of the drawing suit in "
        "their hole cards, else 0.",
        group="Draw Features",
    ),
    FeatureSpec(
        "open_ended_straight_draw", "Open Ended Straight Draw",
        "1 if 2 or more distinct ranks would complete a straight (approximates a true "
        "open-ended draw; also covers double-gutshots, which have similar equity), "
        "else 0.",
        group="Draw Features",
    ),
    FeatureSpec(
        "gutshot", "Gutshot",
        "1 if exactly 1 distinct rank would complete a straight (an inside/gutshot "
        "draw), else 0.",
        group="Draw Features",
    ),
    FeatureSpec(
        "backdoor_flush_draw_2", "Backdoor Flush Draw (2 Cards)",
        "1 if, on the flop only, this player has exactly 3 cards of one suit (needing "
        "both the turn and river to complete a flush) and 2 of those 3 are hole cards, "
        "else 0.",
        group="Draw Features",
    ),
    FeatureSpec(
        "backdoor_flush_draw_1", "Backdoor Flush Draw (1 Card)",
        "1 if, on the flop only, this player has exactly 3 cards of one suit (needing "
        "both the turn and river to complete a flush) and 1 of those 3 is a hole card, "
        "else 0.",
        group="Draw Features",
    ),

    # Opponent-tendency features: observed HUD-style stats accumulated over
    # the current session's hands (see opponent_model.py), not derivable
    # from this hand's cards/board alone -- this is what lets a genome
    # condition on *how opponents have actually played* rather than only on
    # the current situation, the missing ingredient for active exploitation.
    # All default to a neutral 0.5 ("assume average") when nobody's been
    # observed yet, and shrink toward that neutral value on small samples.
    FeatureSpec(
        "opp_vpip_norm", "Opponent VPIP (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of voluntarily putting chips in preflop (calling or raising rather than folding "
        "or checking for free) over hands seen this session. Shrunk toward a neutral 0.5 "
        "on small samples.",
        kind="continuous", value_table=_OPP_VPIP_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_VPIP_CHILDREN,

    FeatureSpec(
        "opp_pfr_norm", "Opponent PFR (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of raising preflop (rather than just calling or folding) over hands seen this "
        "session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_PFR_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_PFR_CHILDREN,

    FeatureSpec(
        "opp_three_bet_norm", "Opponent 3-Bet % (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of re-raising when facing exactly one preflop raise, among their chances to do "
        "so this session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_THREE_BET_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_THREE_BET_CHILDREN,

    FeatureSpec(
        "opp_fold_to_three_bet_norm", "Opponent Fold to 3-Bet % (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of folding when facing a preflop 3-bet, among their chances to do so this "
        "session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_FOLD_TO_THREE_BET_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_FOLD_TO_THREE_BET_CHILDREN,

    FeatureSpec(
        "opp_aggression_freq_norm", "Opponent Postflop Aggression Frequency (Table Average)",
        "Average, across every opponent still active in the hand, of their observed "
        "postflop bet/raise rate -- (bets+raises) / (bets+raises+calls) -- over postflop "
        "actions taken this session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_AGGRESSION_FREQ_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_AGGRESSION_FREQ_CHILDREN,

    FeatureSpec(
        "opp_fold_vs_bet_norm", "Opponent Fold vs Bet, Postflop (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of folding when facing a postflop bet, among their chances to do so this "
        "session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_FOLD_VS_BET_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_FOLD_VS_BET_CHILDREN,
]

FEATURE_NAMES = [spec.key for spec in FEATURE_SPECS]
NUM_FEATURES = len(FEATURE_SPECS)

_SPECS_BY_KEY = {spec.key: spec for spec in FEATURE_SPECS}


def group_of(spec: FeatureSpec) -> str:
    """This spec's report-section group. Top-level specs (value-table
    parents and standalone booleans) set `group` directly; linked children
    inherit their parent's group instead of repeating it."""
    if spec.group:
        return spec.group
    if spec.linked_to:
        return group_of(_SPECS_BY_KEY[spec.linked_to])
    return "Other"


@dataclass
class Situation:
    """Everything a player's genome may condition its decision on."""

    hole: list[Card]
    board: list[Card]
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river
    pot: float
    call_amount: float  # chips needed to call (0 if checking is an option)
    my_stack: float
    effective_stack: float  # min(my_stack, largest active opponent stack)
    position: int  # 0 = first to act this street ... n-1 = last to act
    num_seats_this_street: int
    seat_index: int  # this player's fixed seat this hand
    button_idx: int  # the button's fixed seat this hand
    num_seats_total: int  # seats dealt into this hand (constant all hand, unlike num_active)
    num_active: int  # players still in the hand (not folded)
    num_raises_this_street: int
    num_preflop_raises: int  # frozen once preflop ends, unlike num_raises_this_street
    # Did I make the last bet/raise on the *previous* street? (always False
    # preflop -- there's no previous street to have raised on). Deliberately
    # not "this street": whoever's still deciding on the current street can
    # never themselves be this street's own last aggressor -- raising takes
    # you out of the to-act order until either the street ends or someone
    # else re-raises (see game.betting_round), and a re-raise immediately
    # replaces you as the current street's aggressor -- so that reading
    # would be a structurally-always-False feature. The previous street's
    # aggressor has no such conflict, so it's the one worth conditioning on
    # (e.g. "am I continuation betting").
    is_aggressor: bool
    starting_stack: float
    big_blind: float = 2.0  # lets stack depth be expressed in actual BB (see gto.py's SpotMatcher)
    # Starting positions (seating.SEAT_ROLES) of every *other* seat that has
    # bet/raised this street so far, e.g. frozenset({"UTG"}) for "folds to me
    # in the BB after UTG opens" -- lets gto.py's SpotMatcher express spots
    # like "bb_vs_utg_open" that need to know *who* raised, not just whether
    # someone did.
    raised_positions: frozenset[str] = field(default_factory=frozenset)

    # Opponent-tendency reads (see opponent_model.py), already 0-1 rates --
    # all default to a neutral 0.5 so any caller that doesn't opt into
    # opponent modeling still gets a valid Situation.
    opp_vpip: float = 0.5
    opp_pfr: float = 0.5
    opp_three_bet: float = 0.5
    opp_fold_to_three_bet: float = 0.5
    opp_aggression_freq: float = 0.5
    opp_fold_vs_bet: float = 0.5


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hand_vs_board_heuristics(hole: list[Card], board: list[Card], cat: int, hand: dict) -> dict:
    """High-card / draw-shape heuristics relative to the board that don't
    fit hand_category_norm's ordinal scale (pair/set/straight/flush
    strength relative to the board is folded in there instead -- see
    _hand_category_bucket). Not mutually exclusive as a group, so these
    are standalone booleans rather than a one-hot family."""
    overall_high_card = hand["high_card"]
    ace_high_no_pair = cat == HIGH_CARD and overall_high_card == 14
    king_high_no_pair = cat == HIGH_CARD and overall_high_card == 13

    flush_draw = hand["flush_draw"]
    draw_suit = hand["flush_draw_suit"]
    nuts_flush_draw = flush_draw and any(c.suit == draw_suit and c.rank == 14 for c in hole)

    outs = hand["straight_draw_outs"]
    open_ended_straight_draw = outs >= 2
    gutshot = outs == 1
    combo_draw = flush_draw and outs >= 1

    backdoor_flush_draw_2 = False
    backdoor_flush_draw_1 = False
    if len(board) == 3:
        suit_counts = Counter(c.suit for c in hole + board)
        backdoor_suit = next((s for s, n in suit_counts.items() if n == 3), None)
        if backdoor_suit is not None:
            hole_suited_count = sum(1 for c in hole if c.suit == backdoor_suit)
            backdoor_flush_draw_2 = hole_suited_count == 2
            backdoor_flush_draw_1 = hole_suited_count == 1

    return {
        "ace_high_no_pair": float(ace_high_no_pair),
        "king_high_no_pair": float(king_high_no_pair),
        "nuts_flush_draw": float(nuts_flush_draw),
        "open_ended_straight_draw": float(open_ended_straight_draw),
        "gutshot": float(gutshot),
        "combo_draw": float(combo_draw),
        "backdoor_flush_draw_2": float(backdoor_flush_draw_2),
        "backdoor_flush_draw_1": float(backdoor_flush_draw_1),
    }


# Base index (within hand_category_norm's 24-value table -- see
# _HAND_CATEGORY_VALUES) where each made-hand category's sub-buckets begin.
_HIGH_CARD_INDEX = 0
_PAIR_BASE_INDEX = 1
_TWO_PAIR_INDEX = 10
_SET_BASE_INDEX = 11
_STRAIGHT_BASE_INDEX = 14
_FLUSH_BASE_INDEX = 18
_FULL_HOUSE_INDEX = 21
_QUADS_INDEX = 22
_STRAIGHT_FLUSH_INDEX = 23

# A "good" (but not the very best -- see the Top Kicker bucket) Top Pair
# kicker: Jack, Queen, or King. Ace is handled separately (Top Kicker).
_GOOD_KICKER_RANKS = frozenset({11, 12, 13})


def _pair_bucket_offset(hole: list[Card], board: list[Card]) -> int:
    """0-8 offset within hand_category_norm's Pair range (see
    _HAND_CATEGORY_VALUES, added to _PAIR_BASE_INDEX by the caller) --
    assumes cat == PAIR."""
    hole_ranks = (hole[0].rank, hole[1].rank)
    is_pocket_pair = hole_ranks[0] == hole_ranks[1]
    board_ranks = [c.rank for c in board]

    if is_pocket_pair:
        if not board_ranks:
            return 2  # preflop pocket pair -- no board yet to compare against
        if hole_ranks[0] > max(board_ranks):
            return 8  # Overpair
        if hole_ranks[0] < min(board_ranks):
            return 0  # Low Pair
        return 1  # Underpair (below the top card, but beats at least one board card)

    # Not a pocket pair, so this pair is exactly one hole card rank-matching
    # one board rank (cat == PAIR already guarantees this match exists and
    # is unique -- the same invariant _hand_vs_board_heuristics used to
    # rely on for its own top/second/third_pair checks).
    distinct_board_desc = sorted(set(board_ranks), reverse=True)
    matched_rank = next(r for r in hole_ranks if r in board_ranks)
    if matched_rank == distinct_board_desc[0]:
        kicker = hole_ranks[1] if hole_ranks[0] == matched_rank else hole_ranks[0]
        if kicker == 14:
            return 7  # Top Pair + Top Kicker
        if kicker in _GOOD_KICKER_RANKS:
            return 6  # Top Pair + Good Kicker
        return 5  # Top Pair
    if len(distinct_board_desc) >= 2 and matched_rank == distinct_board_desc[1]:
        return 4  # Second Pair
    if len(distinct_board_desc) >= 3 and matched_rank == distinct_board_desc[2]:
        return 3  # Third Pair
    return 2  # generic Pair (4th+ pair on a wide board)


def _set_bucket_offset(hole: list[Card], board: list[Card]) -> int:
    """0-2 offset within hand_category_norm's Three of a Kind range (see
    _HAND_CATEGORY_VALUES) -- assumes cat == TRIPS."""
    hole_ranks = (hole[0].rank, hole[1].rank)
    if hole_ranks[0] != hole_ranks[1]:
        return 1  # classic "trips" via a paired board -- treated as the plain middle bucket
    distinct_board_desc = sorted({c.rank for c in board}, reverse=True)
    tripped_rank = hole_ranks[0]
    if tripped_rank == distinct_board_desc[0]:
        return 2  # Top Set
    if tripped_rank == distinct_board_desc[-1]:
        return 0  # Bottom Set
    return 1  # Set (middle)


def _achievable_straight_highs(board: list[Card]) -> set[int]:
    """Every straight_high value obtainable by adding up to 2 arbitrary
    ranks to the board's own ranks -- i.e. every straight some hypothetical
    2 hole cards could complete on this board, not just the ones this
    specific player holds. Used to classify a made straight as the
    Bottom/Top one available here (see hand_category_norm's Straight
    sub-buckets). O(13*13) straight checks -- only run when this player's
    own hand is actually a straight, not on every decision, so the cost
    doesn't matter the way it would in extract_features' main path."""
    board_ranks = {c.rank for c in board}
    highs = set()
    for r1 in range(2, 15):
        for r2 in range(2, 15):
            sh = straight_high(sorted(board_ranks | {r1, r2}, reverse=True))
            if sh is not None:
                highs.add(sh)
    return highs


def _board_flush_possible(board: list[Card]) -> bool:
    """True if the board alone already has 3+ cards of one suit -- enough
    that some opponent's 2 hole cards could complete a flush, independent
    of what this player holds. Used for the Nuts Straight bucket: a
    possible flush means even the best straight here isn't provably the
    best hand."""
    if not board:
        return False
    return max(Counter(c.suit for c in board).values()) >= 3


def _straight_bucket_offset(board: list[Card], straight_high_value: int) -> int:
    """0-3 offset within hand_category_norm's Straight range (see
    _HAND_CATEGORY_VALUES) -- assumes cat == STRAIGHT. `straight_high_value`
    must be a member of _achievable_straight_highs(board) -- guaranteed
    since this player's own hole cards are themselves one way to reach it."""
    achievable = _achievable_straight_highs(board)
    if straight_high_value == max(achievable):
        return 3 if not _board_flush_possible(board) else 2  # Nuts vs Top
    if straight_high_value == min(achievable):
        return 0  # Bottom Straight
    return 1  # Straight (middle)


def _flush_bucket_offset(hand: dict) -> int:
    """0-2 offset within hand_category_norm's Flush range (see
    _HAND_CATEGORY_VALUES) -- assumes cat == FLUSH. hand["tiebreak"][0] is
    the flush's own high card, not hand["high_card"] (the highest card
    among *all* hole+board cards regardless of suit) -- the two differ
    whenever the overall highest card isn't of the flush suit."""
    flush_high_card = hand["tiebreak"][0]
    if flush_high_card == 14:
        return 2  # Ace High Flush
    if flush_high_card == 13:
        return 1  # King High Flush
    return 0  # Flush


def _hand_category_bucket(hole: list[Card], board: list[Card], cat: int, hand: dict) -> int:
    """Index into hand_category_norm's 24-value table (see
    _HAND_CATEGORY_VALUES) -- folds what used to be a dozen-plus standalone
    booleans (top_pair, overpair, low_pair, ...) directly into the
    made-hand-strength ordinal scale, plus new Set/Straight/Flush/kicker
    splits, so a NN can read them as *where on the strength spectrum* a
    hand sits rather than as unrelated flags."""
    if cat == HIGH_CARD:
        return _HIGH_CARD_INDEX
    if cat == PAIR:
        return _PAIR_BASE_INDEX + _pair_bucket_offset(hole, board)
    if cat == TWO_PAIR:
        return _TWO_PAIR_INDEX
    if cat == TRIPS:
        return _SET_BASE_INDEX + _set_bucket_offset(hole, board)
    if cat == STRAIGHT:
        return _STRAIGHT_BASE_INDEX + _straight_bucket_offset(board, hand["tiebreak"][0])
    if cat == FLUSH:
        return _FLUSH_BASE_INDEX + _flush_bucket_offset(hand)
    if cat == FULL_HOUSE:
        return _FULL_HOUSE_INDEX
    if cat == QUADS:
        return _QUADS_INDEX
    assert cat == STRAIGHT_FLUSH
    return _STRAIGHT_FLUSH_INDEX


_NO_FLOP_TEXTURE = {
    "flop_suit_texture_norm": 0.0, "rainbow_flop": 0.0, "flush_draw_flop": 0.0, "monotone_flop": 0.0,
    "flop_pairing_texture_norm": 0.0, "unpaired_flop": 0.0, "paired_flop": 0.0, "tripled_flop": 0.0,
    "flop_connectivity_norm": 0.0, "disconnected_flop": 0.0, "connected_flop": 0.0,
    "connected_no_straight_draw_flop": 0.0, "connected_straight_draw_flop": 0.0,
    "oesd_possible_flop": 0.0,
    "flop_wetness_norm": 0.0, "dry_flop": 0.0, "wet_flop": 0.0,
    "flop_dynamism_norm": 0.0, "static_flop": 0.0, "dynamic_flop": 0.0,
}


def _flop_texture(board: list[Card], hole: list[Card]) -> dict:
    """Board-texture facts about the flop specifically (board[:3]) -- plus,
    for the connectivity family, how `hole` connects to it -- frozen once
    the flop is dealt -- the turn/river don't change them, since neither
    the flop cards nor the hole cards do. Defaults all to 0 (their
    lowest-value category) before the flop is dealt. The suit family here
    is purely the flop's own shape (no hole-card awareness) -- see
    _suit_connection_features for that, computed separately since it isn't
    frozen at the flop."""
    if len(board) < 3:
        return dict(_NO_FLOP_TEXTURE)
    flop = board[:3]

    suit_counts = Counter(c.suit for c in flop)
    max_suit_count = max(suit_counts.values())
    monotone, two_tone, rainbow = max_suit_count == 3, max_suit_count == 2, max_suit_count == 1
    suit_index = 2 if monotone else (1 if two_tone else 0)

    rank_counts = Counter(c.rank for c in flop)
    max_rank_count = max(rank_counts.values())
    tripled, paired, unpaired = max_rank_count == 3, max_rank_count == 2, max_rank_count == 1
    pairing_index = 2 if tripled else (1 if paired else 0)

    span = _ace_aware_span([c.rank for c in flop])
    connected = span <= 4
    oesd_possible = unpaired and span <= 3

    if not connected:
        connectivity_index = 0
    elif count_straight_draw_outs(hole + flop) == 0:
        connectivity_index = 1
    else:
        connectivity_index = 2

    # Wetness: how many draws are live, from suit + rank coordination alone
    # (the flop's own shape, not this player's hole cards -- connectivity_index
    # above is the hole-card-aware version of the rank side of this).
    wet_score = suit_index + int(connected) + int(oesd_possible)
    wet = wet_score >= 2

    # Dynamism: whether that draw-heaviness can actually reshuffle who's
    # ahead by the river. Two things pin a board to Static regardless of its
    # wetness score: pairing (made-hand rank -- trips/boat > two pair > top
    # pair -- rarely flips once the board has paired) and very high cards
    # (an A/K/Q on board caps how many hole cards can credibly out-flop the
    # made pair/overpair it hands out, so fewer runouts actually change who's
    # ahead). Low top cards cut the other way -- more hole-card combos have
    # live equity against them -- so they get a matching bonus.
    high_rank = max(c.rank for c in flop)
    high_card_adjustment = -1 if high_rank >= 12 else (1 if high_rank <= 8 else 0)
    dynamic = unpaired and (wet_score + high_card_adjustment) >= 2

    return {
        "flop_suit_texture_norm": suit_index / 2.0,
        "rainbow_flop": float(rainbow),
        "flush_draw_flop": float(two_tone),
        "monotone_flop": float(monotone),
        "flop_pairing_texture_norm": pairing_index / 2.0,
        "unpaired_flop": float(unpaired),
        "paired_flop": float(paired),
        "tripled_flop": float(tripled),
        "flop_connectivity_norm": connectivity_index / 2.0,
        "disconnected_flop": float(connectivity_index == 0),
        "connected_flop": float(connectivity_index != 0),
        "connected_no_straight_draw_flop": float(connectivity_index == 1),
        "connected_straight_draw_flop": float(connectivity_index == 2),
        "oesd_possible_flop": float(oesd_possible),
        "flop_wetness_norm": float(wet),
        "dry_flop": float(not wet),
        "wet_flop": float(wet),
        "flop_dynamism_norm": float(dynamic),
        "static_flop": float(not dynamic),
        "dynamic_flop": float(dynamic),
    }


_SUIT_CONNECTION_KEYS = tuple(spec.key for spec in _SUIT_CONNECTION_CHILDREN)  # n=1..5


def _suit_connection_features(hole: list[Card], board: list[Card]) -> dict:
    """suit_connection_index and its one-hot children (see that FeatureSpec):
    the most cards of any single suit among hole + the *current* board,
    capped at 5 -- unlike _flop_texture's suit family, this isn't frozen at
    the flop, so it's computed fresh from whatever board extract_features
    was called with (3, 4, or 5 cards -- or 0, preflop)."""
    count = min(max(Counter(c.suit for c in hole + board).values()), 5)
    values = {"suit_connection_index": count / 5.0}
    for i, key in enumerate(_SUIT_CONNECTION_KEYS):
        values[key] = float(count == i + 1)
    return values


# Precomputed, call-invariant key names for extract_features' per-rank and
# per-bucket one-hot loops below. These are exactly the same FeatureSpec keys
# already built above (e.g. _HOLE_HIGH_CARD_CHILDREN) -- just indexed by
# position here instead of being rebuilt from an f-string (and, for the rank
# ones, a `_rank_key` dict lookup) on every single extract_features call.
# extract_features runs once per in-game decision, so this recomputation was
# previously one of the hottest code paths in the whole simulation.
_HAND_CATEGORY_KEYS = tuple(spec.key for spec in _HAND_CATEGORY_CHILDREN)  # index 0..23, see _hand_category_bucket
_HOLE_HIGH_CARD_KEYS = tuple(spec.key for spec in _HOLE_HIGH_CARD_CHILDREN)  # rank 2..14
_SHARED_HIGH_CARD_KEYS = tuple(spec.key for spec in _SHARED_HIGH_CARD_CHILDREN)  # rank 2..14
_CONNECTIVITY_GAP_KEYS = tuple(spec.key for spec in _CONNECTIVITY_CHILDREN[1:-1])  # gap 1..CAP-1
_CONNECTIVITY_GAP_PLUS_KEY = _CONNECTIVITY_CHILDREN[-1].key
_HOLE_CATEGORY_KEYS = tuple(spec.key for spec in _HOLE_CATEGORY_CHILDREN)  # index 0..11, see _hole_hand_category
_ACTIVE_PLAYERS_KEYS = tuple(spec.key for spec in _ACTIVE_PLAYERS_CHILDREN)  # n=2..6
_RAISES_KEYS = tuple(spec.key for spec in _RAISES_CHILDREN[:3])  # r=0..2 ("raises_is_3plus" is already a literal)
_OVERCARDS_KEYS = tuple(spec.key for spec in _OVERCARDS_CHILDREN)  # n=0..5

_BUCKET_KEY_FAMILIES = {
    "call_amount_norm": _CALL_SIZE_CHILDREN,
    "spr_norm": _SPR_CHILDREN,
    "position_norm": _POSITION_CHILDREN,
    "stack_depth_norm": _STACK_DEPTH_CHILDREN,
    "opp_vpip_norm": _OPP_VPIP_CHILDREN,
    "opp_pfr_norm": _OPP_PFR_CHILDREN,
    "opp_three_bet_norm": _OPP_THREE_BET_CHILDREN,
    "opp_fold_to_three_bet_norm": _OPP_FOLD_TO_THREE_BET_CHILDREN,
    "opp_aggression_freq_norm": _OPP_AGGRESSION_FREQ_CHILDREN,
    "opp_fold_vs_bet_norm": _OPP_FOLD_VS_BET_CHILDREN,
}
_BUCKET_KEYS_BY_FEATURE = {
    name: tuple(spec.key for spec in children) for name, children in _BUCKET_KEY_FAMILIES.items()
}


def extract_features(sit: Situation) -> np.ndarray:
    hand = best_hand_from_available(sit.hole, sit.board)
    cat = hand["category"]
    hand_category_bucket = _hand_category_bucket(sit.hole, sit.board, cat, hand)

    hole_suited = float(sit.hole[0].suit == sit.hole[1].suit)
    gap = _rank_gap(sit.hole[0].rank, sit.hole[1].rank)
    hole_connectivity = _clip01(1.0 - gap / CONNECTIVITY_GAP_CAP)
    hole_category = _hole_hand_category(sit.hole)

    call_amount_norm = _clip01(sit.call_amount / max(sit.pot, 1.0) / _CALL_SIZE_CEILING)
    spr = sit.effective_stack / max(sit.pot, 1.0)
    spr_norm = _clip01(spr / 20.0)
    position_norm = 0.0
    if sit.num_seats_this_street > 1:
        position_norm = sit.position / (sit.num_seats_this_street - 1)
    num_active_norm = _clip01(sit.num_active / 6.0)
    num_raises_norm = _clip01(sit.num_raises_this_street / 3.0)
    pot_type_raises = min(sit.num_preflop_raises, 3)
    stack_depth_norm = _clip01(sit.my_stack / max(sit.starting_stack, 1.0) / 2.0)
    hole_high_card_rank = max(sit.hole[0].rank, sit.hole[1].rank)
    shared_high_card_rank = max((c.rank for c in sit.board), default=None)
    num_overcards = sum(1 for c in sit.board if c.rank > hole_high_card_rank)
    role = seat_role(sit.seat_index, sit.button_idx, sit.num_seats_total)
    role_index = SEAT_ROLES.index(role)

    values = {
        "hand_category_norm": hand_category_bucket / (len(_HAND_CATEGORY_VALUES) - 1),
        "hole_high_card_norm": (hole_high_card_rank - 2) / 12.0,
        "shared_high_card_norm": ((shared_high_card_rank - 2) / 12.0) if shared_high_card_rank else 0.0,
        "num_overcards_norm": num_overcards / 5.0,
        "flush_draw": float(hand["flush_draw"]),
        "straight_draw": float(hand["straight_draw"]),
        "hole_suited": hole_suited,
        "hole_connectivity": hole_connectivity,
        "hole_hand_category_norm": hole_category / 11.0,
        "street_norm": sit.street / 3.0,
        "facing_bet": float(sit.call_amount > 0),
        "is_aggressor": float(sit.is_aggressor),
        "call_amount_norm": call_amount_norm,
        "spr_norm": spr_norm,
        "position_norm": position_norm,
        "starting_position_norm": role_index / (len(SEAT_ROLES) - 1),
        "num_active_norm": num_active_norm,
        "num_raises_norm": num_raises_norm,
        "pot_type_norm": pot_type_raises / 3.0,
        "stack_depth_norm": stack_depth_norm,
        "opp_vpip_norm": _clip01(sit.opp_vpip),
        "opp_pfr_norm": _clip01(sit.opp_pfr),
        "opp_three_bet_norm": _clip01(sit.opp_three_bet),
        "opp_fold_to_three_bet_norm": _clip01(sit.opp_fold_to_three_bet),
        "opp_aggression_freq_norm": _clip01(sit.opp_aggression_freq),
        "opp_fold_vs_bet_norm": _clip01(sit.opp_fold_vs_bet),
    }
    for i, seat_role_name in enumerate(SEAT_ROLES):
        values[_SEAT_ROLE_KEYS[seat_role_name]] = float(role_index == i)

    # Exact one-hot indicators for hand_category_norm's 24 sub-buckets.
    for i, key in enumerate(_HAND_CATEGORY_KEYS):
        values[key] = float(hand_category_bucket == i)

    for i, r in enumerate(range(2, 15)):
        values[_HOLE_HIGH_CARD_KEYS[i]] = float(hole_high_card_rank == r)
        values[_SHARED_HIGH_CARD_KEYS[i]] = float(shared_high_card_rank == r)

    values["hole_paired"] = float(gap == 0)
    for i, g in enumerate(range(1, CONNECTIVITY_GAP_CAP)):
        values[_CONNECTIVITY_GAP_KEYS[i]] = float(gap == g)
    values[_CONNECTIVITY_GAP_PLUS_KEY] = float(gap == CONNECTIVITY_GAP_CAP)

    for i in range(len(_HOLE_CATEGORY_KEYS)):
        values[_HOLE_CATEGORY_KEYS[i]] = float(hole_category == i)

    for i, key in enumerate(("is_preflop", "is_flop", "is_turn", "is_river")):
        values[key] = float(sit.street == i)

    for i, n in enumerate(range(2, 7)):
        values[_ACTIVE_PLAYERS_KEYS[i]] = float(sit.num_active == n)

    for i, n in enumerate(range(6)):
        values[_OVERCARDS_KEYS[i]] = float(num_overcards == n)

    for i, r in enumerate(range(0, 3)):
        values[_RAISES_KEYS[i]] = float(sit.num_raises_this_street == r)
    values["raises_is_3plus"] = float(sit.num_raises_this_street >= 3)

    for i, key in enumerate(("is_unraised_pot", "is_single_raised_pot", "is_3bet_pot", "is_4bet_plus_pot")):
        values[key] = float(pot_type_raises == i)

    # Nearest-representative-point bucket indicators for genuinely continuous features.
    for feature_key, raw_value in (
        ("call_amount_norm", call_amount_norm), ("spr_norm", spr_norm),
        ("position_norm", position_norm), ("stack_depth_norm", stack_depth_norm),
        ("opp_vpip_norm", values["opp_vpip_norm"]), ("opp_pfr_norm", values["opp_pfr_norm"]),
        ("opp_three_bet_norm", values["opp_three_bet_norm"]),
        ("opp_fold_to_three_bet_norm", values["opp_fold_to_three_bet_norm"]),
        ("opp_aggression_freq_norm", values["opp_aggression_freq_norm"]),
        ("opp_fold_vs_bet_norm", values["opp_fold_vs_bet_norm"]),
    ):
        bucket_keys = _BUCKET_KEYS_BY_FEATURE[feature_key]
        nearest = _nearest_bucket_index(raw_value, len(bucket_keys))
        for i, key in enumerate(bucket_keys):
            values[key] = float(i == nearest)

    values.update(_hand_vs_board_heuristics(sit.hole, sit.board, cat, hand))
    values.update(_flop_texture(sit.board, sit.hole))
    values.update(_suit_connection_features(sit.hole, sit.board))

    return np.array([values[spec.key] for spec in FEATURE_SPECS], dtype=np.float64)
