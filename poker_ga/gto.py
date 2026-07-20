"""Spots and charts: the pieces that let a genome memorize an exact,
solved-looking strategy for a specific, well-defined situation, the way a
human would memorize a range chart for "UTG open, 100BB" and just apply it
verbatim rather than reasoning about V/L every time.

A GTOSpot combines:
  - a SpotMatcher: a readable, declarative definition of which situations
    this spot applies to (street, pot type, position, stack depth, etc.)
  - action_ranges: which range (see ranges.py) takes which action in that
    spot, checked in order -- the first range a hand falls in wins.

This is deliberately a *fixed, code-defined catalog* (GTO_SPOTS below), the
same pattern as features.py's FEATURE_SPECS -- extend it by adding entries,
not by making it runtime-configurable. The example spots included are
illustrative (reasonable, hand-picked ranges), not verified solver output;
swap in real solved charts for genuine accuracy.

Whether a genome actually *uses* a given spot's chart is a separate,
evolvable per-genome gene (Genome.gto_flags in genome.py) -- a spot existing
in this catalog doesn't mean every genome plays it that way, only that a
genome *can* learn to trust it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from features import Situation
from ranges import hand_label, parse_range
from seating import SEAT_ROLES, seat_role

POT_TYPE_LABELS = ("Unraised", "Single Raised", "3-Bet", "4-Bet+")


@dataclass(frozen=True)
class SpotMatcher:
    """Declarative criteria checked against a Situation to decide whether a
    GTOSpot applies. Every field is optional (None = "don't care"); a
    situation matches only if every specified field agrees.

    Note: `is_aggressor` only distinguishes "I made the last raise" from "I
    didn't" -- Situation doesn't track *which* seat/position made a raise,
    so a matcher can express "BTN facing a 3-bet" but not "BTN facing a
    3-bet specifically from the BB" (there's no opponent-position data to
    match against)."""

    street: int | None = None  # 0=preflop, 1=flop, 2=turn, 3=river
    pot_type: int | None = None  # 0=unraised, 1=single raised, 2=3-bet, 3=4-bet+ (num_preflop_raises, capped at 3)
    position: str | None = None  # one of seating.SEAT_ROLES, or None for "any position"
    is_aggressor: bool | None = None  # did I make the last bet/raise this street?
    facing_bet: bool | None = None  # is there a nonzero amount required to call?
    min_effective_bb: float | None = None  # inclusive, in actual big blinds
    max_effective_bb: float | None = None  # inclusive

    def matches(self, situation: Situation) -> bool:
        if self.street is not None and situation.street != self.street:
            return False
        if self.pot_type is not None and min(situation.num_preflop_raises, 3) != self.pot_type:
            return False
        if self.position is not None:
            role = seat_role(situation.seat_index, situation.button_idx, situation.num_seats_total)
            if role != self.position:
                return False
        if self.is_aggressor is not None and situation.is_aggressor != self.is_aggressor:
            return False
        if self.facing_bet is not None and (situation.call_amount > 1e-9) != self.facing_bet:
            return False
        if self.min_effective_bb is not None or self.max_effective_bb is not None:
            effective_bb = situation.effective_stack / max(situation.big_blind, 1e-9)
            if self.min_effective_bb is not None and effective_bb < self.min_effective_bb:
                return False
            if self.max_effective_bb is not None and effective_bb > self.max_effective_bb:
                return False
        return True

    def describe(self) -> str:
        parts = []
        if self.street is not None:
            parts.append(("Preflop", "Flop", "Turn", "River")[self.street])
        if self.position is not None:
            parts.append(self.position)
        if self.pot_type is not None:
            parts.append(POT_TYPE_LABELS[self.pot_type])
        if self.facing_bet is True:
            parts.append("facing a bet")
        elif self.facing_bet is False:
            parts.append("not facing a bet")
        if self.is_aggressor is True:
            parts.append("I'm the last aggressor")
        elif self.is_aggressor is False:
            parts.append("I'm not the last aggressor")
        if self.min_effective_bb is not None and self.max_effective_bb is not None:
            parts.append(f"{self.min_effective_bb:.0f}-{self.max_effective_bb:.0f}BB effective")
        elif self.min_effective_bb is not None:
            parts.append(f">={self.min_effective_bb:.0f}BB effective")
        elif self.max_effective_bb is not None:
            parts.append(f"<={self.max_effective_bb:.0f}BB effective")
        return ", ".join(parts) if parts else "any situation"


@dataclass(frozen=True)
class GTOSpot:
    key: str  # short, stable identifier -- this spot's gene name in reports
    label: str  # human-readable description, e.g. "UTG Open -- 100BB stack"
    matcher: SpotMatcher
    action_ranges: tuple[tuple[str, str], ...]  # ((action_token, range_str), ...), checked in order
    default_action: str = "fold"  # applied if the hand isn't in any listed range, once the spot matches
    parsed_ranges: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        parsed = tuple((action, parse_range(range_str)) for action, range_str in self.action_ranges)
        object.__setattr__(self, "parsed_ranges", parsed)


def parse_action_token(token: str) -> tuple[str, float | None]:
    """Returns (kind, pot_fraction): kind is "fold"/"call"/"raise";
    pot_fraction is None for fold/call, a fraction of pot for "raise_NN"
    (e.g. "raise_75" -> 0.75), or None for "allin" (a sentinel meaning
    "shove the full stack", handled by the caller)."""
    token = token.strip().lower()
    if token == "fold":
        return "fold", None
    if token == "call":
        return "call", None
    if token == "allin":
        return "raise", None
    m = re.fullmatch(r"raise_(\d+(?:\.\d+)?)", token)
    if m:
        return "raise", float(m.group(1)) / 100.0
    raise ValueError(f"Unknown GTO action token: {token!r}")


def resolve_spot_action(spot: GTOSpot, situation: Situation) -> tuple[str, float | None] | None:
    """If `situation` matches this spot, returns this hand's action (falling
    back to spot.default_action if the hand isn't in any listed range).
    Returns None if the spot doesn't apply to this situation at all."""
    if not spot.matcher.matches(situation):
        return None
    label = hand_label(situation.hole[0], situation.hole[1])
    for action_token, range_set in spot.parsed_ranges:
        if label in range_set:
            return parse_action_token(action_token)
    return parse_action_token(spot.default_action)


# A small, illustrative starter catalog -- hand-picked, reasonable ranges,
# not verified solver output. Extend by adding more GTOSpot entries; a
# genome only actually plays one of these if its corresponding gto_flags
# gene is on (see genome.py).
GTO_SPOTS: list[GTOSpot] = [
    GTOSpot(
        key="utg_open_100bb",
        label="UTG Open -- 100BB stack",
        matcher=SpotMatcher(
            street=0, pot_type=0, position="UTG", facing_bet=False,
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_250", "77+, ATs+, KTs+, QTs+, JTs, T9s, 98s, AJo+, KQo"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="btn_open_100bb",
        label="BTN Open -- 100BB stack",
        matcher=SpotMatcher(
            street=0, pot_type=0, position="BTN", facing_bet=False,
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            (
                "raise_250",
                "22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 96s+, 86s+, 75s+, 64s+, 53s+, "
                "A2o+, K8o+, Q9o+, J9o+, T9o",
            ),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="bb_vs_single_raise_100bb",
        label="BB Facing A Single Raise -- 100BB stack",
        matcher=SpotMatcher(
            street=0, pot_type=1, position="BB", facing_bet=True, is_aggressor=False,
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_300", "TT+, AQs+, AKo"),
            ("call", "22-99, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, ATo+, KJo+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="btn_vs_3bet_20bb",
        label="BTN Facing A 3-Bet -- 20BB stack",
        matcher=SpotMatcher(
            street=0, pot_type=2, position="BTN", facing_bet=True, is_aggressor=False,
            min_effective_bb=15, max_effective_bb=25,
        ),
        action_ranges=(
            ("allin", "88+, AJs+, KQs, AQo+"),
        ),
        default_action="fold",
    ),
]

NUM_GTO_SPOTS = len(GTO_SPOTS)
