"""Turns a poker decision point into a fixed-length feature vector.

This is the bridge between the game engine and a genome: every feature here
is a "basic characteristic of the current situation" (hand strength, board
texture, betting context, position, stack depth...) that the genome's
weights turn into action scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cards import Card
from evaluator import (
    FULL_HOUSE, FLUSH, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH,
    TRIPS, TWO_PAIR, best_hand_from_available,
)

@dataclass(frozen=True)
class FeatureSpec:
    key: str  # internal identifier; array order is defined by FEATURE_SPECS
    label: str  # short, human-readable name used in exported reports
    description: str  # precise definition of how the value is computed
    kind: str = "boolean"  # "boolean" (0/1) | "categorical" (fixed set of values) | "continuous" (a range)
    # For categorical/continuous features: every (or, for continuous, a representative
    # sample of) normalized value this feature can take, paired with a human-readable
    # label for that value. None for boolean features (they need no value table).
    value_table: tuple | None = None


def _rank_label(rank: int) -> str:
    return {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}.get(rank, str(rank))


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
    (0.2, "4:1 pot odds (risk 1 to win 4) — e.g. facing a small bet"),
    (1 / 3, "2:1 pot odds (risk 1 to win 2) — e.g. facing roughly a half-pot bet"),
    (0.5, "1:1 pot odds (risk 1 to win 1) — facing a pot-sized bet"),
    (2 / 3, "1:2 pot odds (risk 2 to win 1) — facing an overbet"),
    (0.8, "1:4 pot odds (risk 4 to win 1) — facing a large overbet"),
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


# Order here fixes the feature vector layout (and therefore genome weight
# shape) everywhere else in the codebase -- append, don't reorder.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "hand_category_norm", "Hand Strength Tier",
        "Made-hand category of the best hand available from hole cards plus the "
        "current board, normalized to 0-1 as category_index / 8 (0.0 = high card, "
        "0.125 = pair, 0.25 = two pair, 0.375 = trips, 0.5 = straight, 0.625 = flush, "
        "0.75 = full house, 0.875 = quads, 1.0 = straight flush).",
        kind="categorical", value_table=_HAND_CATEGORY_VALUES,
    ),
    FeatureSpec("has_pair", "Has Pair", "1 if the best available hand is exactly a pair, else 0."),
    FeatureSpec("has_two_pair", "Has Two Pair", "1 if the best available hand is exactly two pair, else 0."),
    FeatureSpec(
        "has_trips", "Has Three Of A Kind",
        "1 if the best available hand is exactly three of a kind, else 0.",
    ),
    FeatureSpec("has_straight", "Has Straight", "1 if the best available hand is exactly a straight, else 0."),
    FeatureSpec("has_flush", "Has Flush", "1 if the best available hand is exactly a flush, else 0."),
    FeatureSpec(
        "has_full_house", "Has Full House",
        "1 if the best available hand is exactly a full house, else 0.",
    ),
    FeatureSpec(
        "has_quads", "Has Four Of A Kind",
        "1 if the best available hand is exactly four of a kind, else 0.",
    ),
    FeatureSpec(
        "has_straight_flush", "Has Straight Flush",
        "1 if the best available hand is a straight flush, else 0.",
    ),
    FeatureSpec(
        "high_card_norm", "High Card Rank",
        "Rank of the single highest card among hole+board cards, normalized to 0-1 "
        "via (rank - 2) / 12, so 0.0 = deuce and 1.0 = ace.",
        kind="categorical", value_table=_HIGH_CARD_VALUES,
    ),
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
    FeatureSpec("hole_paired", "Pocket Pair", "1 if both hole cards share a rank, else 0."),
    FeatureSpec(
        "hole_connectivity", "Hole Card Connectivity",
        "How close together the two hole card ranks are: 1 - |rank1 - rank2| / 12. "
        "1.0 means consecutive ranks (e.g. 9-T); 0.0 means the widest possible gap (2-A).",
        kind="categorical", value_table=_CONNECTIVITY_VALUES,
    ),
    FeatureSpec(
        "street_norm", "Betting Street",
        "Current street normalized to 0-1: 0.0 = preflop, 0.33 = flop, 0.67 = turn, 1.0 = river.",
        kind="categorical", value_table=_STREET_VALUES,
    ),
    FeatureSpec(
        "facing_bet", "Facing A Bet",
        "1 if a nonzero amount is required to call (someone has already bet or "
        "raised this street), else 0 (checking is free).",
    ),
    FeatureSpec(
        "is_aggressor", "Last Aggressor",
        "1 if this player made the most recent bet/raise on the current street, else 0.",
    ),
    FeatureSpec(
        "pot_odds", "Pot Odds",
        "Fraction of the resulting pot that calling would represent: "
        "call_amount / (pot + call_amount). 0 if there is nothing to call.",
        kind="continuous", value_table=_POT_ODDS_VALUES,
    ),
    FeatureSpec(
        "call_amount_norm", "Call Size Vs Pot",
        "Amount required to call, as a fraction of the current pot "
        "(call_amount / pot), clipped to 0-1.",
        kind="continuous", value_table=_CALL_SIZE_VALUES,
    ),
    FeatureSpec(
        "spr_norm", "Stack-To-Pot Ratio",
        "Effective stack (the smaller of this player's stack and the largest "
        "active opponent's stack) divided by the current pot, indicating how many "
        "pot-sized bets remain; divided by 20 and clipped to 0-1.",
        kind="continuous", value_table=_SPR_VALUES,
    ),
    FeatureSpec(
        "position_norm", "Table Position",
        "How late this player acts in the current street's action order, from "
        "0.0 (acts first) to 1.0 (acts last).",
        kind="continuous", value_table=_POSITION_VALUES,
    ),
    FeatureSpec("is_button", "On The Button", "1 if this player holds the dealer/button seat this hand, else 0."),
    FeatureSpec(
        "num_active_norm", "Players Still In Hand",
        "Number of players who have not folded yet this hand, divided by 6.",
        kind="categorical", value_table=_ACTIVE_PLAYERS_VALUES,
    ),
    FeatureSpec(
        "num_raises_norm", "Raises This Street",
        "Number of bets/raises made so far on the current street, divided by 4 and clipped to 0-1.",
        kind="categorical", value_table=_RAISES_VALUES,
    ),
    FeatureSpec(
        "stack_depth_norm", "Stack Depth",
        "This player's remaining stack as a fraction of the session's starting "
        "stack, divided by 2 and clipped to 0-1 (1.0 = double the starting stack or more).",
        kind="continuous", value_table=_STACK_DEPTH_VALUES,
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
    is_button: bool
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
    hole_paired = float(sit.hole[0].rank == sit.hole[1].rank)
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

    vec = [
        cat / 8.0,
        float(cat == PAIR),
        float(cat == TWO_PAIR),
        float(cat == TRIPS),
        float(cat == STRAIGHT),
        float(cat == FLUSH),
        float(cat == FULL_HOUSE),
        float(cat == QUADS),
        float(cat == STRAIGHT_FLUSH),
        (hand["high_card"] - 2) / 12.0,
        float(hand["flush_draw"]),
        float(hand["straight_draw"]),
        hole_suited,
        hole_paired,
        hole_connectivity,
        sit.street / 3.0,
        float(sit.call_amount > 0),
        float(sit.is_aggressor),
        pot_odds,
        call_amount_norm,
        spr_norm,
        position_norm,
        float(sit.is_button),
        num_active_norm,
        num_raises_norm,
        stack_depth_norm,
    ]
    return np.array(vec, dtype=np.float64)
