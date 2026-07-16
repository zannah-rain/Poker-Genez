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

FEATURE_NAMES = [
    "hand_category_norm",
    "has_pair",
    "has_two_pair",
    "has_trips",
    "has_straight",
    "has_flush",
    "has_full_house",
    "has_quads",
    "has_straight_flush",
    "high_card_norm",
    "flush_draw",
    "straight_draw",
    "hole_suited",
    "hole_paired",
    "hole_connectivity",
    "street_norm",
    "facing_bet",
    "is_aggressor",
    "pot_odds",
    "call_amount_norm",
    "spr_norm",
    "position_norm",
    "is_button",
    "num_active_norm",
    "num_raises_norm",
    "stack_depth_norm",
]

NUM_FEATURES = len(FEATURE_NAMES)


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
