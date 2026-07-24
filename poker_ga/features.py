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
    TRIPS, TWO_PAIR, best_hand_from_available,
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
_LAST_BUCKET_INDEX = _NUM_BUCKETS - 1


def _nearest_bucket_index(value: float) -> int:
    """Index into _BUCKET_POINTS closest to `value`. Since _BUCKET_POINTS is
    evenly spaced (0, 0.25, 0.5, ..., 1.0), the nearest point can be found
    with direct arithmetic instead of scanning+comparing all 5 candidates --
    this is called from extract_features' hottest inner loop (once per
    continuous feature per decision), so the constant-factor savings add up.
    `ceil(x - 0.5)` (rather than the more common `round`) is deliberate: it
    breaks an exact tie between two adjacent points toward the *lower*
    index, matching what the original `min(range(...), key=...)` scan did
    (the first-encountered minimum wins ties, and range() visits indices
    ascending) -- plain `round()` would instead break ties to the nearest
    *even* index (banker's rounding) and silently disagree at the exact
    midpoints (0.125, 0.375, 0.625, 0.875)."""
    idx = math.ceil(value * _LAST_BUCKET_INDEX - 0.5)
    return 0 if idx < 0 else (_LAST_BUCKET_INDEX if idx > _LAST_BUCKET_INDEX else idx)


def _continuous_children(parent_key: str, value_table: tuple) -> list[FeatureSpec]:
    return [
        _linked_bool(
            f"{parent_key}_bucket_{i}", label,
            f"1 if {parent_key}'s value falls closest to {point:.2f} on a 0-1 scale "
            f"(\"{label}\"), else 0. One of these 5 is always 1.",
            parent_key, i,
        )
        for i, (point, label) in enumerate(value_table)
    ]


_HAND_CATEGORY_VALUES = (
    (0 / 8, "High Card"), (1 / 8, "Pair"), (2 / 8, "Two Pair"), (3 / 8, "Three of a Kind"),
    (4 / 8, "Straight"), (5 / 8, "Flush"), (6 / 8, "Full House"), (7 / 8, "Four of a Kind"),
    (8 / 8, "Straight Flush"),
)
_HOLE_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_SHARED_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_CONNECTIVITY_VALUES = tuple(
    (1.0 - gap / CONNECTIVITY_GAP_CAP, _connectivity_label(gap)) for gap in range(0, CONNECTIVITY_GAP_CAP + 1)
)
_STREET_VALUES = ((0.0, "Preflop"), (1 / 3, "Flop"), (2 / 3, "Turn"), (1.0, "River"))
_CALL_SIZE_VALUES = (
    (0.0, "No bet to call"),
    (0.25, "Call is 1/4 pot"),
    (0.5, "Call is 1/2 pot"),
    (0.75, "Call is 3/4 pot"),
    (1.0, "Call is a full pot-sized bet or larger (clipped)"),
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
_RAISES_VALUES = (
    (0.0, "No raises yet this street"),
    (0.25, "1 raise so far"),
    (0.5, "2 raises so far"),
    (0.75, "3 raises so far"),
    (1.0, "4 or more raises so far (clipped)"),
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
    (0.0, "Active opponents almost never voluntarily play a hand (very tight)"),
    (0.25, "Active opponents play somewhat tighter than average"),
    (0.5, "Average looseness (or no reads yet this session)"),
    (0.75, "Active opponents play somewhat looser than average"),
    (1.0, "Active opponents voluntarily play almost every hand (very loose)"),
)
_OPP_PFR_VALUES = (
    (0.0, "Active opponents almost never raise preflop (passive)"),
    (0.25, "Somewhat less preflop raising than average"),
    (0.5, "Average preflop aggression (or no reads yet this session)"),
    (0.75, "Somewhat more preflop raising than average"),
    (1.0, "Active opponents raise preflop almost every time they play (very aggressive)"),
)
_OPP_THREE_BET_VALUES = (
    (0.0, "Active opponents almost never 3-bet when given the chance"),
    (0.25, "Somewhat less 3-betting than average"),
    (0.5, "Average 3-bet frequency (or no reads yet this session)"),
    (0.75, "Somewhat more 3-betting than average"),
    (1.0, "Active opponents 3-bet almost every chance they get"),
)
_OPP_FOLD_TO_THREE_BET_VALUES = (
    (0.0, "Active opponents almost never fold to a 3-bet"),
    (0.25, "Somewhat less likely than average to fold to a 3-bet"),
    (0.5, "Average fold-to-3-bet rate (or no reads yet this session)"),
    (0.75, "Somewhat more likely than average to fold to a 3-bet"),
    (1.0, "Active opponents almost always fold to a 3-bet"),
)
_OPP_AGGRESSION_FREQ_VALUES = (
    (0.0, "Active opponents almost never bet/raise postflop when they act (all calls)"),
    (0.25, "Somewhat more call-heavy than average postflop"),
    (0.5, "Average postflop aggression (or no reads yet this session)"),
    (0.75, "Somewhat more bet/raise-heavy than average postflop"),
    (1.0, "Active opponents almost always bet/raise rather than call postflop"),
)
_OPP_FOLD_VS_BET_VALUES = (
    (0.0, "Active opponents almost never fold to a postflop bet (very sticky)"),
    (0.25, "Somewhat less likely than average to fold postflop"),
    (0.5, "Average postflop fold rate (or no reads yet this session)"),
    (0.75, "Somewhat more likely than average to fold postflop"),
    (1.0, "Active opponents almost always fold to a postflop bet"),
)
_OPP_SAMPLE_VALUES = (
    (0.0, "No hands observed yet this session -- every other opponent read is a neutral default"),
    (0.25, "A few hands observed -- reads are still mostly the neutral default"),
    (0.5, "A moderate sample -- reads are meaningfully shrunk toward neutral"),
    (0.75, "A solid sample -- reads are close to the raw observed rate"),
    (1.0, "30+ hands observed -- reads are trustworthy"),
)
_VILLAIN_THREE_BET_VALUES = _OPP_THREE_BET_VALUES
_VILLAIN_FOLD_VS_BET_VALUES = _OPP_FOLD_VS_BET_VALUES
_VILLAIN_AGGRESSION_FREQ_VALUES = _OPP_AGGRESSION_FREQ_VALUES

_OPP_VPIP_CHILDREN = _continuous_children("opp_vpip_norm", _OPP_VPIP_VALUES)
_OPP_PFR_CHILDREN = _continuous_children("opp_pfr_norm", _OPP_PFR_VALUES)
_OPP_THREE_BET_CHILDREN = _continuous_children("opp_three_bet_norm", _OPP_THREE_BET_VALUES)
_OPP_FOLD_TO_THREE_BET_CHILDREN = _continuous_children(
    "opp_fold_to_three_bet_norm", _OPP_FOLD_TO_THREE_BET_VALUES
)
_OPP_AGGRESSION_FREQ_CHILDREN = _continuous_children("opp_aggression_freq_norm", _OPP_AGGRESSION_FREQ_VALUES)
_OPP_FOLD_VS_BET_CHILDREN = _continuous_children("opp_fold_vs_bet_norm", _OPP_FOLD_VS_BET_VALUES)
_OPP_SAMPLE_CHILDREN = _continuous_children("opp_sample_norm", _OPP_SAMPLE_VALUES)
_VILLAIN_THREE_BET_CHILDREN = _continuous_children("villain_three_bet_norm", _VILLAIN_THREE_BET_VALUES)
_VILLAIN_FOLD_VS_BET_CHILDREN = _continuous_children("villain_fold_vs_bet_norm", _VILLAIN_FOLD_VS_BET_VALUES)
_VILLAIN_AGGRESSION_FREQ_CHILDREN = _continuous_children(
    "villain_aggression_freq_norm", _VILLAIN_AGGRESSION_FREQ_VALUES
)

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
    _linked_bool("has_pair", "Pair", "1 if the best available hand is exactly a pair, else 0.", "hand_category_norm", 1),
    _linked_bool(
        "has_two_pair", "Two Pair", "1 if the best available hand is exactly two pair, else 0.",
        "hand_category_norm", 2,
    ),
    _linked_bool(
        "has_trips", "Three of a Kind", "1 if the best available hand is exactly three of a kind, else 0.",
        "hand_category_norm", 3,
    ),
    _linked_bool(
        "has_straight", "Straight", "1 if the best available hand is exactly a straight, else 0.",
        "hand_category_norm", 4,
    ),
    _linked_bool(
        "has_flush", "Flush", "1 if the best available hand is exactly a flush, else 0.",
        "hand_category_norm", 5,
    ),
    _linked_bool(
        "has_full_house", "Full House", "1 if the best available hand is exactly a full house, else 0.",
        "hand_category_norm", 6,
    ),
    _linked_bool(
        "has_quads", "Four of a Kind", "1 if the best available hand is exactly four of a kind, else 0.",
        "hand_category_norm", 7,
    ),
    _linked_bool(
        "has_straight_flush", "Straight Flush", "1 if the best available hand is a straight flush, else 0.",
        "hand_category_norm", 8,
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

_RAISES_CHILDREN = [
    _linked_bool(
        f"raises_is_{r}", ("No raises" if r == 0 else f"{r} raise{'s' if r != 1 else ''}"),
        f"1 if exactly {r} raises have occurred so far this street, else 0.",
        "num_raises_norm", r,
    )
    for r in range(0, 4)
] + [
    _linked_bool(
        "raises_is_4plus", "4+ raises",
        "1 if 4 or more raises have occurred so far this street, else 0.",
        "num_raises_norm", 4,
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

# Flop-texture features describe the flop itself (board[:3]), not this
# player's hand, and are frozen once the flop is dealt -- the turn and river
# don't change them. All default to their lowest-value category before the
# flop (0 for every child, since nothing has happened yet).
_FLOP_SUIT_VALUES = ((0.0, "Rainbow"), (0.5, "Flush Draw Flop"), (1.0, "Monotone"))
_FLOP_PAIRING_VALUES = ((0.0, "Unpaired"), (0.5, "Paired"), (1.0, "Tripled"))
_FLOP_CONNECTIVITY_VALUES = ((0.0, "Disconnected"), (1.0, "Connected"))

_FLOP_SUIT_CHILDREN = [
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
_FLOP_CONNECTIVITY_CHILDREN = [
    _linked_bool(
        "disconnected_flop", "Disconnected",
        "1 if the flop's ranks (Ace counted as high or low, whichever is closer) span more "
        "than 4, too spread out to easily combine into straights, else 0.",
        "flop_connectivity_norm", 0,
    ),
    _linked_bool(
        "connected_flop", "Connected",
        "1 if the flop's ranks (Ace counted as high or low, whichever is closer) span 4 or "
        "less, else 0.",
        "flop_connectivity_norm", 1,
    ),
]


# Grouped as parent-feature-then-its-children for readability; array order is
# otherwise arbitrary since extract_features looks values up by key, not position.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "hand_category_norm", "Hand Strength Tier",
        "Made-hand category of the best hand available from hole cards plus the "
        "current board, normalized to 0-1 as category_index / 8 (0.0 = high card, "
        "0.125 = pair, 0.25 = two pair, 0.375 = trips, 0.5 = straight, 0.625 = flush, "
        "0.75 = full house, 0.875 = quads, 1.0 = straight flush).",
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
        "street_norm", "Betting Street",
        "Current street normalized to 0-1: 0.0 = preflop, 0.33 = flop, 0.67 = turn, 1.0 = river.",
        kind="categorical", value_table=_STREET_VALUES, group="Table & Game State Features",
    ),
    *_STREET_CHILDREN,

    FeatureSpec(
        "call_amount_norm", "Call Size Vs Pot",
        "Amount required to call, as a fraction of the current pot "
        "(call_amount / pot), clipped to 0-1.",
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
        "Number of bets/raises made so far on the current street, divided by 4 and clipped to 0-1.",
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
        "1 if this player made the most recent bet/raise on the current street, else 0.",
        group="Betting Behaviour Features",
    ),

    FeatureSpec(
        "flop_suit_texture_norm", "Flop Suit Texture",
        "How suit-coordinated the flop is, from driest to wettest: Rainbow (all 3 "
        "different suits) < Flush Draw Flop (2 share a suit) < Monotone (all 3 share a "
        "suit), normalized as index / 2. Frozen once the flop is dealt (unaffected by "
        "the turn/river); defaults to 0.0 (Rainbow's value) before the flop.",
        kind="categorical", value_table=_FLOP_SUIT_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_SUIT_CHILDREN,

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
        "are close enough together to easily make straights: Connected if their span "
        "(highest - lowest) is 4 or less, else Disconnected. Frozen once the flop is "
        "dealt; defaults to 0.0 (Disconnected's value) before the flop.",
        kind="categorical", value_table=_FLOP_CONNECTIVITY_VALUES, group="Board / Flop Characteristics",
    ),
    *_FLOP_CONNECTIVITY_CHILDREN,

    FeatureSpec(
        "oesd_possible_flop", "Open Ended Straight Draw Possible Flop",
        "1 if the flop's 3 ranks are all different and span 3 or less (Ace counted as "
        "high or low, whichever is closer) -- tight enough that an open-ended straight "
        "draw is possible for some hole cards -- else 0. A stricter subset of Connected "
        "Flop (not one of its Connected/Disconnected partition values), so it isn't "
        "folded into that table. 0 before the flop.",
        group="Board / Flop Characteristics",
    ),

    # Hand-vs-board heuristics: not mutually exclusive as a group (e.g. Low
    # Pair is a subset of Underpair), so unlike the categorical features
    # above these aren't organized as a one-hot family with a linear parent.
    FeatureSpec(
        "top_pair", "Top Pair",
        "1 if exactly one hole card ranks-matches the single highest board card (and "
        "the best overall hand is exactly a pair, i.e. this isn't also two pair/trips), "
        "else 0.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "second_pair", "Second Pair",
        "1 if exactly one hole card rank-matches the second-highest distinct board "
        "rank (and the best overall hand is exactly a pair), else 0. Needs a board "
        "with at least 2 distinct ranks.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "third_pair", "Third Pair",
        "1 if exactly one hole card rank-matches the third-highest distinct board rank "
        "(and the best overall hand is exactly a pair), else 0. Needs a board with at "
        "least 3 distinct ranks.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "overpair", "Overpair",
        "1 if the hole cards are a pocket pair ranked higher than every board card, "
        "else 0.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "underpair", "Underpair",
        "1 if the hole cards are a pocket pair ranked lower than the highest board "
        "card, else 0.",
        group="Made Hand Features",
    ),
    FeatureSpec(
        "low_pair", "Low Pair",
        "1 if the hole cards are a pocket pair ranked lower than every board card (a "
        "stricter version of Underpair -- below the whole board, not just the top "
        "card), else 0.",
        group="Made Hand Features",
    ),
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
    # observed yet, and shrink toward that neutral value on small samples --
    # pair with Opponent Sample Size to tell "genuinely average" apart from
    # "no read yet."
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

    FeatureSpec(
        "opp_sample_norm", "Opponent Sample Size (Table Average)",
        "Average, across every opponent still active in the hand, of how many hands "
        "they've been observed for this session, divided by 30 and clipped to 0-1 -- how "
        "much the six Opponent Tendency reads above should be trusted versus treated as "
        "the neutral default.",
        kind="continuous", value_table=_OPP_SAMPLE_VALUES, group="Opponent Tendency Features",
    ),
    *_OPP_SAMPLE_CHILDREN,

    FeatureSpec(
        "villain_three_bet_norm", "Current Aggressor's 3-Bet %",
        "The observed 3-bet rate (see Opponent 3-Bet %) of specifically whichever player "
        "made the last bet/raise this street -- the single opponent actually applying "
        "pressure right now, not a table average. Neutral 0.5 if nobody has bet/raised "
        "yet this street.",
        kind="continuous", value_table=_VILLAIN_THREE_BET_VALUES, group="Opponent Tendency Features",
    ),
    *_VILLAIN_THREE_BET_CHILDREN,

    FeatureSpec(
        "villain_fold_vs_bet_norm", "Current Aggressor's Fold vs Bet, Postflop",
        "The observed postflop fold-vs-bet rate (see Opponent Fold vs Bet, Postflop) of "
        "specifically whichever player made the last bet/raise this street. Neutral 0.5 "
        "if nobody has bet/raised yet this street.",
        kind="continuous", value_table=_VILLAIN_FOLD_VS_BET_VALUES, group="Opponent Tendency Features",
    ),
    *_VILLAIN_FOLD_VS_BET_CHILDREN,

    FeatureSpec(
        "villain_aggression_freq_norm", "Current Aggressor's Postflop Aggression Frequency",
        "The observed postflop aggression frequency (see Opponent Postflop Aggression "
        "Frequency) of specifically whichever player made the last bet/raise this "
        "street. Neutral 0.5 if nobody has bet/raised yet this street.",
        kind="continuous", value_table=_VILLAIN_AGGRESSION_FREQ_VALUES, group="Opponent Tendency Features",
    ),
    *_VILLAIN_AGGRESSION_FREQ_CHILDREN,
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
    is_aggressor: bool  # did I make the last bet/raise this street?
    starting_stack: float
    big_blind: float = 2.0  # lets stack depth be expressed in actual BB (see gto.py's SpotMatcher)
    # Starting positions (seating.SEAT_ROLES) of every *other* seat that has
    # bet/raised this street so far, e.g. frozenset({"UTG"}) for "folds to me
    # in the BB after UTG opens" -- lets gto.py's SpotMatcher express spots
    # like "bb_vs_utg_open" that need to know *who* raised, not just whether
    # someone did.
    raised_positions: frozenset[str] = field(default_factory=frozenset)

    # Opponent-tendency reads (see opponent_model.py), already 0-1 rates --
    # all default to a neutral 0.5 (0.0 for sample size) so any caller that
    # doesn't opt into opponent modeling still gets a valid Situation.
    opp_vpip: float = 0.5
    opp_pfr: float = 0.5
    opp_three_bet: float = 0.5
    opp_fold_to_three_bet: float = 0.5
    opp_aggression_freq: float = 0.5
    opp_fold_vs_bet: float = 0.5
    opp_sample: float = 0.0
    villain_three_bet: float = 0.5
    villain_fold_vs_bet: float = 0.5
    villain_aggression_freq: float = 0.5


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hand_vs_board_heuristics(hole: list[Card], board: list[Card], cat: int, hand: dict) -> dict:
    """Pair-strength / high-card / draw-shape heuristics relative to the
    board. Not mutually exclusive as a group (e.g. Low Pair is a subset of
    Underpair), so these are standalone booleans rather than a one-hot
    family like the categorical features in FEATURE_SPECS."""
    hole_ranks = (hole[0].rank, hole[1].rank)
    is_pocket_pair = hole_ranks[0] == hole_ranks[1]
    board_ranks = [c.rank for c in board]
    distinct_board_desc = sorted(set(board_ranks), reverse=True)

    matched_board_rank = None
    if cat == PAIR and not is_pocket_pair:
        matches = [r for r in hole_ranks if r in board_ranks]
        if len(matches) == 1:
            matched_board_rank = matches[0]

    top_pair = matched_board_rank is not None and matched_board_rank == distinct_board_desc[0]
    second_pair = (
        matched_board_rank is not None
        and len(distinct_board_desc) >= 2
        and matched_board_rank == distinct_board_desc[1]
    )
    third_pair = (
        matched_board_rank is not None
        and len(distinct_board_desc) >= 3
        and matched_board_rank == distinct_board_desc[2]
    )

    overpair = is_pocket_pair and bool(board_ranks) and hole_ranks[0] > max(board_ranks)
    underpair = is_pocket_pair and bool(board_ranks) and hole_ranks[0] < max(board_ranks)
    low_pair = is_pocket_pair and bool(board_ranks) and hole_ranks[0] < min(board_ranks)

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
        "top_pair": float(top_pair),
        "second_pair": float(second_pair),
        "third_pair": float(third_pair),
        "overpair": float(overpair),
        "underpair": float(underpair),
        "low_pair": float(low_pair),
        "ace_high_no_pair": float(ace_high_no_pair),
        "king_high_no_pair": float(king_high_no_pair),
        "nuts_flush_draw": float(nuts_flush_draw),
        "open_ended_straight_draw": float(open_ended_straight_draw),
        "gutshot": float(gutshot),
        "combo_draw": float(combo_draw),
        "backdoor_flush_draw_2": float(backdoor_flush_draw_2),
        "backdoor_flush_draw_1": float(backdoor_flush_draw_1),
    }


_NO_FLOP_TEXTURE = {
    "flop_suit_texture_norm": 0.0, "rainbow_flop": 0.0, "flush_draw_flop": 0.0, "monotone_flop": 0.0,
    "flop_pairing_texture_norm": 0.0, "unpaired_flop": 0.0, "paired_flop": 0.0, "tripled_flop": 0.0,
    "flop_connectivity_norm": 0.0, "disconnected_flop": 0.0, "connected_flop": 0.0,
    "oesd_possible_flop": 0.0,
}


def _flop_texture(board: list[Card]) -> dict:
    """Board-texture facts about the flop specifically (board[:3]), frozen
    once dealt -- the turn/river don't change them. Defaults all to 0 (their
    lowest-value category) before the flop is dealt."""
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

    return {
        "flop_suit_texture_norm": suit_index / 2.0,
        "rainbow_flop": float(rainbow),
        "flush_draw_flop": float(two_tone),
        "monotone_flop": float(monotone),
        "flop_pairing_texture_norm": pairing_index / 2.0,
        "unpaired_flop": float(unpaired),
        "paired_flop": float(paired),
        "tripled_flop": float(tripled),
        "flop_connectivity_norm": float(connected),
        "disconnected_flop": float(not connected),
        "connected_flop": float(connected),
        "oesd_possible_flop": float(oesd_possible),
    }


# Precomputed, call-invariant key names for extract_features' per-rank and
# per-bucket one-hot loops below. These are exactly the same FeatureSpec keys
# already built above (e.g. _HOLE_HIGH_CARD_CHILDREN) -- just indexed by
# position here instead of being rebuilt from an f-string (and, for the rank
# ones, a `_rank_key` dict lookup) on every single extract_features call.
# extract_features runs once per in-game decision, so this recomputation was
# previously one of the hottest code paths in the whole simulation.
_HOLE_HIGH_CARD_KEYS = tuple(spec.key for spec in _HOLE_HIGH_CARD_CHILDREN)  # rank 2..14
_SHARED_HIGH_CARD_KEYS = tuple(spec.key for spec in _SHARED_HIGH_CARD_CHILDREN)  # rank 2..14
_CONNECTIVITY_GAP_KEYS = tuple(spec.key for spec in _CONNECTIVITY_CHILDREN[1:-1])  # gap 1..CAP-1
_CONNECTIVITY_GAP_PLUS_KEY = _CONNECTIVITY_CHILDREN[-1].key
_ACTIVE_PLAYERS_KEYS = tuple(spec.key for spec in _ACTIVE_PLAYERS_CHILDREN)  # n=2..6
_RAISES_KEYS = tuple(spec.key for spec in _RAISES_CHILDREN[:4])  # r=0..3 ("raises_is_4plus" is already a literal)

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
    "opp_sample_norm": _OPP_SAMPLE_CHILDREN,
    "villain_three_bet_norm": _VILLAIN_THREE_BET_CHILDREN,
    "villain_fold_vs_bet_norm": _VILLAIN_FOLD_VS_BET_CHILDREN,
    "villain_aggression_freq_norm": _VILLAIN_AGGRESSION_FREQ_CHILDREN,
}
_BUCKET_KEYS_BY_FEATURE = {
    name: tuple(spec.key for spec in children) for name, children in _BUCKET_KEY_FAMILIES.items()
}


def extract_features(sit: Situation) -> np.ndarray:
    hand = best_hand_from_available(sit.hole, sit.board)
    cat = hand["category"]

    hole_suited = float(sit.hole[0].suit == sit.hole[1].suit)
    gap = _rank_gap(sit.hole[0].rank, sit.hole[1].rank)
    hole_connectivity = _clip01(1.0 - gap / CONNECTIVITY_GAP_CAP)

    call_amount_norm = _clip01(sit.call_amount / max(sit.pot, 1.0))
    spr = sit.effective_stack / max(sit.pot, 1.0)
    spr_norm = _clip01(spr / 20.0)
    position_norm = 0.0
    if sit.num_seats_this_street > 1:
        position_norm = sit.position / (sit.num_seats_this_street - 1)
    num_active_norm = _clip01(sit.num_active / 6.0)
    num_raises_norm = _clip01(sit.num_raises_this_street / 4.0)
    pot_type_raises = min(sit.num_preflop_raises, 3)
    stack_depth_norm = _clip01(sit.my_stack / max(sit.starting_stack, 1.0) / 2.0)
    hole_high_card_rank = max(sit.hole[0].rank, sit.hole[1].rank)
    shared_high_card_rank = max((c.rank for c in sit.board), default=None)
    role = seat_role(sit.seat_index, sit.button_idx, sit.num_seats_total)
    role_index = SEAT_ROLES.index(role)

    values = {
        "hand_category_norm": cat / 8.0,
        "hole_high_card_norm": (hole_high_card_rank - 2) / 12.0,
        "shared_high_card_norm": ((shared_high_card_rank - 2) / 12.0) if shared_high_card_rank else 0.0,
        "flush_draw": float(hand["flush_draw"]),
        "straight_draw": float(hand["straight_draw"]),
        "hole_suited": hole_suited,
        "hole_connectivity": hole_connectivity,
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
        "opp_sample_norm": _clip01(sit.opp_sample),
        "villain_three_bet_norm": _clip01(sit.villain_three_bet),
        "villain_fold_vs_bet_norm": _clip01(sit.villain_fold_vs_bet),
        "villain_aggression_freq_norm": _clip01(sit.villain_aggression_freq),
    }
    for i, seat_role_name in enumerate(SEAT_ROLES):
        values[_SEAT_ROLE_KEYS[seat_role_name]] = float(role_index == i)

    # Exact one-hot indicators for enumerable categorical features.
    for category_value, key in (
        (HIGH_CARD, "has_high_card"), (PAIR, "has_pair"), (TWO_PAIR, "has_two_pair"),
        (TRIPS, "has_trips"), (STRAIGHT, "has_straight"), (FLUSH, "has_flush"),
        (FULL_HOUSE, "has_full_house"), (QUADS, "has_quads"), (STRAIGHT_FLUSH, "has_straight_flush"),
    ):
        values[key] = float(cat == category_value)

    for i, r in enumerate(range(2, 15)):
        values[_HOLE_HIGH_CARD_KEYS[i]] = float(hole_high_card_rank == r)
        values[_SHARED_HIGH_CARD_KEYS[i]] = float(shared_high_card_rank == r)

    values["hole_paired"] = float(gap == 0)
    for i, g in enumerate(range(1, CONNECTIVITY_GAP_CAP)):
        values[_CONNECTIVITY_GAP_KEYS[i]] = float(gap == g)
    values[_CONNECTIVITY_GAP_PLUS_KEY] = float(gap == CONNECTIVITY_GAP_CAP)

    for i, key in enumerate(("is_preflop", "is_flop", "is_turn", "is_river")):
        values[key] = float(sit.street == i)

    for i, n in enumerate(range(2, 7)):
        values[_ACTIVE_PLAYERS_KEYS[i]] = float(sit.num_active == n)

    for i, r in enumerate(range(0, 4)):
        values[_RAISES_KEYS[i]] = float(sit.num_raises_this_street == r)
    values["raises_is_4plus"] = float(sit.num_raises_this_street >= 4)

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
        ("opp_sample_norm", values["opp_sample_norm"]),
        ("villain_three_bet_norm", values["villain_three_bet_norm"]),
        ("villain_fold_vs_bet_norm", values["villain_fold_vs_bet_norm"]),
        ("villain_aggression_freq_norm", values["villain_aggression_freq_norm"]),
    ):
        nearest = _nearest_bucket_index(raw_value)
        bucket_keys = _BUCKET_KEYS_BY_FEATURE[feature_key]
        for i in range(_NUM_BUCKETS):
            values[bucket_keys[i]] = float(i == nearest)

    values.update(_hand_vs_board_heuristics(sit.hole, sit.board, cat, hand))
    values.update(_flop_texture(sit.board))

    return np.array([values[spec.key] for spec in FEATURE_SPECS], dtype=np.float64)
