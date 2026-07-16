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


# Order here fixes the feature vector layout (and therefore genome weight
# shape) everywhere else in the codebase -- append, don't reorder.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "hand_category_norm", "Hand Strength Tier",
        "Made-hand category of the best hand available from hole cards plus the "
        "current board, normalized to 0-1 as category_index / 8 (0.0 = high card, "
        "0.125 = pair, 0.25 = two pair, 0.375 = trips, 0.5 = straight, 0.625 = flush, "
        "0.75 = full house, 0.875 = quads, 1.0 = straight flush).",
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
    ),
    FeatureSpec(
        "street_norm", "Betting Street",
        "Current street normalized to 0-1: 0.0 = preflop, 0.33 = flop, 0.67 = turn, 1.0 = river.",
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
    ),
    FeatureSpec(
        "call_amount_norm", "Call Size Vs Pot",
        "Amount required to call, as a fraction of the current pot "
        "(call_amount / pot), clipped to 0-1.",
    ),
    FeatureSpec(
        "spr_norm", "Stack-To-Pot Ratio",
        "Effective stack (the smaller of this player's stack and the largest "
        "active opponent's stack) divided by the current pot, indicating how many "
        "pot-sized bets remain; divided by 20 and clipped to 0-1.",
    ),
    FeatureSpec(
        "position_norm", "Table Position",
        "How late this player acts in the current street's action order, from "
        "0.0 (acts first) to 1.0 (acts last).",
    ),
    FeatureSpec("is_button", "On The Button", "1 if this player holds the dealer/button seat this hand, else 0."),
    FeatureSpec(
        "num_active_norm", "Players Still In Hand",
        "Number of players who have not folded yet this hand, divided by 6.",
    ),
    FeatureSpec(
        "num_raises_norm", "Raises This Street",
        "Number of bets/raises made so far on the current street, divided by 4 and clipped to 0-1.",
    ),
    FeatureSpec(
        "stack_depth_norm", "Stack Depth",
        "This player's remaining stack as a fraction of the session's starting "
        "stack, divided by 2 and clipped to 0-1 (1.0 = double the starting stack or more).",
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
