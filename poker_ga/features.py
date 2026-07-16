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


def _rank_label(rank: int) -> str:
    return {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}.get(rank, str(rank))


def _rank_key(rank: int) -> str:
    return {11: "jack", 12: "queen", 13: "king", 14: "ace"}.get(rank, str(rank))


def _connectivity_label(gap: int) -> str:
    if gap == 0:
        return "Same rank (pocket pair)"
    if gap == 1:
        return "1 apart (connectors, e.g. 9-10)"
    if gap == 2:
        return "2 apart (one-gappers, e.g. 9-J)"
    if gap == 3:
        return "3 apart (two-gappers, e.g. 9-Q)"
    return f"{gap} apart"


def _linked_bool(key: str, label: str, description: str, linked_to: str, index: int) -> FeatureSpec:
    return FeatureSpec(key, label, description, kind="boolean", linked_to=linked_to, linked_value_index=index)


# Representative sample points used for every genuinely continuous feature, both for
# its value table and for the "nearest point" bucket indicators computed at runtime.
_BUCKET_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _nearest_bucket_index(value: float) -> int:
    return min(range(len(_BUCKET_POINTS)), key=lambda i: abs(value - _BUCKET_POINTS[i]))


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
_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_CONNECTIVITY_VALUES = tuple((1.0 - gap / 12.0, _connectivity_label(gap)) for gap in range(0, 13))
_STREET_VALUES = ((0.0, "Preflop"), (1 / 3, "Flop"), (2 / 3, "Turn"), (1.0, "River"))
_POT_ODDS_VALUES = (
    (0.0, "Nothing to call"),
    (0.25, "3:1 pot odds (risk 1 to win 3) — e.g. facing a small bet"),
    (0.5, "1:1 pot odds (risk 1 to win 1) — facing a pot-sized bet"),
    (0.75, "1:3 pot odds (risk 3 to win 1) — facing a large overbet"),
    (1.0, "Facing an enormous overbet (call far exceeds the pot)"),
)
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
_STACK_DEPTH_VALUES = (
    (0.0, "Busted / no chips left"),
    (0.25, "Half of the starting stack remaining"),
    (0.5, "At the starting stack (100%)"),
    (0.75, "1.5x the starting stack"),
    (1.0, "2x the starting stack or more (clipped)"),
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

_HIGH_CARD_CHILDREN = [
    _linked_bool(
        f"high_card_is_{_rank_key(r)}", _rank_label(r),
        f"1 if the highest card among hole+board cards has rank {_rank_label(r)}, else 0.",
        "high_card_norm", r - 2,
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
        f"1 if the two hole card ranks are exactly {gap} apart, else 0.",
        "hole_connectivity", gap,
    )
    for gap in range(1, 13)
]

_STREET_CHILDREN = [
    _linked_bool(key, label, f"1 if the current street is {label}, else 0.", "street_norm", i)
    for i, (key, label) in enumerate(
        [("is_preflop", "Preflop"), ("is_flop", "Flop"), ("is_turn", "Turn"), ("is_river", "River")]
    )
]

_POT_ODDS_CHILDREN = _continuous_children("pot_odds", _POT_ODDS_VALUES)
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


# Grouped as parent-feature-then-its-children for readability; array order is
# otherwise arbitrary since extract_features looks values up by key, not position.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "hand_category_norm", "Hand Strength Tier",
        "Made-hand category of the best hand available from hole cards plus the "
        "current board, normalized to 0-1 as category_index / 8 (0.0 = high card, "
        "0.125 = pair, 0.25 = two pair, 0.375 = trips, 0.5 = straight, 0.625 = flush, "
        "0.75 = full house, 0.875 = quads, 1.0 = straight flush).",
        kind="categorical", value_table=_HAND_CATEGORY_VALUES,
    ),
    *_HAND_CATEGORY_CHILDREN,

    FeatureSpec(
        "high_card_norm", "High Card Rank",
        "Rank of the single highest card among hole+board cards, normalized to 0-1 "
        "via (rank - 2) / 12, so 0.0 = deuce and 1.0 = ace.",
        kind="categorical", value_table=_HIGH_CARD_VALUES,
    ),
    *_HIGH_CARD_CHILDREN,

    FeatureSpec(
        "hole_connectivity", "Hole Card Connectivity",
        "How close together the two hole card ranks are: 1 - |rank1 - rank2| / 12. "
        "1.0 means consecutive ranks (e.g. 9-T); 0.0 means the widest possible gap (2-A).",
        kind="categorical", value_table=_CONNECTIVITY_VALUES,
    ),
    *_CONNECTIVITY_CHILDREN,

    FeatureSpec(
        "street_norm", "Betting Street",
        "Current street normalized to 0-1: 0.0 = preflop, 0.33 = flop, 0.67 = turn, 1.0 = river.",
        kind="categorical", value_table=_STREET_VALUES,
    ),
    *_STREET_CHILDREN,

    FeatureSpec(
        "pot_odds", "Pot Odds",
        "Fraction of the resulting pot that calling would represent: "
        "call_amount / (pot + call_amount). 0 if there is nothing to call.",
        kind="continuous", value_table=_POT_ODDS_VALUES,
    ),
    *_POT_ODDS_CHILDREN,

    FeatureSpec(
        "call_amount_norm", "Call Size Vs Pot",
        "Amount required to call, as a fraction of the current pot "
        "(call_amount / pot), clipped to 0-1.",
        kind="continuous", value_table=_CALL_SIZE_VALUES,
    ),
    *_CALL_SIZE_CHILDREN,

    FeatureSpec(
        "spr_norm", "Stack-To-Pot Ratio",
        "Effective stack (the smaller of this player's stack and the largest "
        "active opponent's stack) divided by the current pot, indicating how many "
        "pot-sized bets remain; divided by 20 and clipped to 0-1.",
        kind="continuous", value_table=_SPR_VALUES,
    ),
    *_SPR_CHILDREN,

    FeatureSpec(
        "position_norm", "Table Position",
        "How late this player acts in the current street's action order, from "
        "0.0 (acts first) to 1.0 (acts last).",
        kind="continuous", value_table=_POSITION_VALUES,
    ),
    *_POSITION_CHILDREN,

    FeatureSpec(
        "starting_position_norm", "Starting Seat Position",
        "Which of the 6 standard preflop seat roles this player started the hand in, "
        "ordered from first-to-act preflop through the blinds -- UTG, HJ, CO, BTN, SB, BB "
        "-- normalized as role_index / 5. At smaller table sizes the earliest non-blind "
        "roles collapse (e.g. 4-handed there's only UTG and BTN; heads-up the button and "
        "small blind are the same seat, labeled SB).",
        kind="categorical", value_table=_SEAT_ROLE_VALUES,
    ),
    *_SEAT_ROLE_CHILDREN,

    FeatureSpec(
        "num_active_norm", "Players Still In Hand",
        "Number of players who have not folded yet this hand, divided by 6.",
        kind="categorical", value_table=_ACTIVE_PLAYERS_VALUES,
    ),
    *_ACTIVE_PLAYERS_CHILDREN,

    FeatureSpec(
        "num_raises_norm", "Raises This Street",
        "Number of bets/raises made so far on the current street, divided by 4 and clipped to 0-1.",
        kind="categorical", value_table=_RAISES_VALUES,
    ),
    *_RAISES_CHILDREN,

    FeatureSpec(
        "stack_depth_norm", "Stack Depth",
        "This player's remaining stack as a fraction of the session's starting "
        "stack, divided by 2 and clipped to 0-1 (1.0 = double the starting stack or more).",
        kind="continuous", value_table=_STACK_DEPTH_VALUES,
    ),
    *_STACK_DEPTH_CHILDREN,

    # Standalone booleans: not tied to a specific value of any other feature.
    FeatureSpec(
        "flush_draw", "Flush Draw",
        "1 if exactly 4 of the hole+board cards share one suit (a live one-card "
        "flush draw), else 0. Always 0 preflop, since a flush needs 5+ cards.",
    ),
    FeatureSpec(
        "straight_draw", "Straight Draw",
        "1 if there exists a single rank which, if added to the hole+board cards, "
        "would complete a straight (open-ended or gutshot), else 0. Always 0 preflop.",
    ),
    FeatureSpec("hole_suited", "Suited Hole Cards", "1 if both hole cards share a suit, else 0."),
    FeatureSpec(
        "facing_bet", "Facing A Bet",
        "1 if a nonzero amount is required to call (someone has already bet or "
        "raised this street), else 0 (checking is free).",
    ),
    FeatureSpec(
        "is_aggressor", "Last Aggressor",
        "1 if this player made the most recent bet/raise on the current street, else 0.",
    ),
]

FEATURE_NAMES = [spec.key for spec in FEATURE_SPECS]
NUM_FEATURES = len(FEATURE_SPECS)


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
    is_aggressor: bool  # did I make the last bet/raise this street?
    starting_stack: float


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_features(sit: Situation) -> np.ndarray:
    hand = best_hand_from_available(sit.hole, sit.board)
    cat = hand["category"]

    hole_suited = float(sit.hole[0].suit == sit.hole[1].suit)
    gap = abs(sit.hole[0].rank - sit.hole[1].rank)
    hole_connectivity = _clip01(1.0 - gap / 12.0)

    pot_odds = 0.0
    if sit.call_amount > 0:
        pot_odds = sit.call_amount / max(sit.pot + sit.call_amount, 1e-6)

    call_amount_norm = _clip01(sit.call_amount / max(sit.pot, 1.0))
    spr = sit.effective_stack / max(sit.pot, 1.0)
    spr_norm = _clip01(spr / 20.0)
    position_norm = 0.0
    if sit.num_seats_this_street > 1:
        position_norm = sit.position / (sit.num_seats_this_street - 1)
    num_active_norm = _clip01(sit.num_active / 6.0)
    num_raises_norm = _clip01(sit.num_raises_this_street / 4.0)
    stack_depth_norm = _clip01(sit.my_stack / max(sit.starting_stack, 1.0) / 2.0)
    high_card_norm = (hand["high_card"] - 2) / 12.0
    role = seat_role(sit.seat_index, sit.button_idx, sit.num_seats_total)
    role_index = SEAT_ROLES.index(role)

    values = {
        "hand_category_norm": cat / 8.0,
        "high_card_norm": high_card_norm,
        "flush_draw": float(hand["flush_draw"]),
        "straight_draw": float(hand["straight_draw"]),
        "hole_suited": hole_suited,
        "hole_connectivity": hole_connectivity,
        "street_norm": sit.street / 3.0,
        "facing_bet": float(sit.call_amount > 0),
        "is_aggressor": float(sit.is_aggressor),
        "pot_odds": pot_odds,
        "call_amount_norm": call_amount_norm,
        "spr_norm": spr_norm,
        "position_norm": position_norm,
        "starting_position_norm": role_index / (len(SEAT_ROLES) - 1),
        "num_active_norm": num_active_norm,
        "num_raises_norm": num_raises_norm,
        "stack_depth_norm": stack_depth_norm,
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

    for r in range(2, 15):
        values[f"high_card_is_{_rank_key(r)}"] = float(hand["high_card"] == r)

    values["hole_paired"] = float(gap == 0)
    for g in range(1, 13):
        values[f"connectivity_gap_{g}"] = float(gap == g)

    for i, key in enumerate(("is_preflop", "is_flop", "is_turn", "is_river")):
        values[key] = float(sit.street == i)

    for n in range(2, 7):
        values[f"active_players_is_{n}"] = float(sit.num_active == n)

    for r in range(0, 4):
        values[f"raises_is_{r}"] = float(sit.num_raises_this_street == r)
    values["raises_is_4plus"] = float(sit.num_raises_this_street >= 4)

    # Nearest-representative-point bucket indicators for genuinely continuous features.
    for feature_key, raw_value in (
        ("pot_odds", pot_odds), ("call_amount_norm", call_amount_norm), ("spr_norm", spr_norm),
        ("position_norm", position_norm), ("stack_depth_norm", stack_depth_norm),
    ):
        nearest = _nearest_bucket_index(raw_value)
        for i in range(len(_BUCKET_POINTS)):
            values[f"{feature_key}_bucket_{i}"] = float(i == nearest)

    return np.array([values[spec.key] for spec in FEATURE_SPECS], dtype=np.float64)
