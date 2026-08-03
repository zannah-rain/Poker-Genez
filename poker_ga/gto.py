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

    `raised_positions` lets a matcher pin down *who* has raised so far this
    street (not just whether someone has, like `facing_bet`/`is_aggressor`
    do), e.g. `raised_positions=("UTG",)` for "bb_vs_utg_open" -- BB facing
    exactly an UTG open, nobody else having entered the pot yet. It's an
    exact match against Situation.raised_positions (the *other* seats that
    raised, not me): every position listed must have raised, and no
    position outside the list may have raised either."""

    street: int | None = None  # 0=preflop, 1=flop, 2=turn, 3=river
    pot_type: int | None = None  # 0=unraised, 1=single raised, 2=3-bet, 3=4-bet+ (num_preflop_raises, capped at 3)
    position: str | None = None  # one of seating.SEAT_ROLES, or None for "any position"
    is_aggressor: bool | None = None  # did I make the last bet/raise on the *previous* street? (e.g. continuation betting)
    facing_bet: bool | None = None  # is there a nonzero amount required to call?
    raised_positions: frozenset[str] | None = None  # exact set of other seats' positions that have raised this street
    min_effective_bb: float | None = None  # inclusive, in actual big blinds
    max_effective_bb: float | None = None  # inclusive
    min_call_bb: float | None = None  # inclusive; size of the amount I'd need to call, in actual big blinds
    max_call_bb: float | None = None  # inclusive

    def __post_init__(self):
        if self.raised_positions is not None:
            object.__setattr__(self, "raised_positions", frozenset(self.raised_positions))

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
        if self.raised_positions is not None and situation.raised_positions != self.raised_positions:
            return False
        if self.min_effective_bb is not None or self.max_effective_bb is not None:
            effective_bb = situation.effective_stack / max(situation.big_blind, 1e-9)
            if self.min_effective_bb is not None and effective_bb < self.min_effective_bb:
                return False
            if self.max_effective_bb is not None and effective_bb > self.max_effective_bb:
                return False
        if self.min_call_bb is not None or self.max_call_bb is not None:
            call_bb = situation.call_amount / max(situation.big_blind, 1e-9)
            if self.min_call_bb is not None and call_bb < self.min_call_bb:
                return False
            if self.max_call_bb is not None and call_bb > self.max_call_bb:
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
            parts.append("I raised last street")
        elif self.is_aggressor is False:
            parts.append("I didn't raise last street")
        if self.raised_positions is not None:
            if self.raised_positions:
                parts.append(f"vs {'+'.join(sorted(self.raised_positions))} raise")
            else:
                parts.append("vs no raises")
        if self.min_effective_bb is not None and self.max_effective_bb is not None:
            parts.append(f"{self.min_effective_bb:.0f}-{self.max_effective_bb:.0f}BB effective")
        elif self.min_effective_bb is not None:
            parts.append(f">={self.min_effective_bb:.0f}BB effective")
        elif self.max_effective_bb is not None:
            parts.append(f"<={self.max_effective_bb:.0f}BB effective")
        if self.min_call_bb is not None and self.max_call_bb is not None:
            parts.append(f"{self.min_call_bb:.1f}-{self.max_call_bb:.1f}BB to call")
        elif self.min_call_bb is not None:
            parts.append(f">={self.min_call_bb:.1f}BB to call")
        elif self.max_call_bb is not None:
            parts.append(f"<={self.max_call_bb:.1f}BB to call")
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


def parse_action_token(token: str) -> tuple[str, tuple[str, float] | None]:
    """Returns (kind, size_spec). kind is "fold"/"call"/"raise". size_spec is:
      - None                -- fold/call (unused); or "allin" for raise
                                (sentinel meaning "shove the full stack")
      - ("pot", fraction)   -- raise sized at `fraction` x pot, e.g.
                                "raise_75" -> ("pot", 0.75)
      - ("bb", n)           -- raise sized so this player's TOTAL commitment
                                this street reaches `n` big blinds, e.g.
                                "raise_2.5bb" -> ("bb", 2.5) -- the natural
                                way to specify an open size ("raise to 2.5bb")
    size_spec is resolved into actual chips by the caller (genome.py), which
    has access to the current pot/call_amount/big_blind."""
    token = token.strip().lower()
    if token == "fold":
        return "fold", None
    if token == "call":
        return "call", None
    if token == "allin":
        return "raise", None
    m = re.fullmatch(r"raise_(\d+(?:\.\d+)?)bb", token)
    if m:
        return "raise", ("bb", float(m.group(1)))
    m = re.fullmatch(r"raise_(\d+(?:\.\d+)?)", token)
    if m:
        return "raise", ("pot", float(m.group(1)) / 100.0)
    raise ValueError(f"Unknown GTO action token: {token!r}")


def resolve_spot_action(spot: GTOSpot, situation: Situation) -> tuple[str, tuple[str, float] | None] | None:
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
            # Not `facing_bet=False`: preflop, everyone except the BB has a
            # nonzero call_amount even in an unraised pot (they still owe
            # the BB's forced post to continue), so `facing_bet` is True for
            # basically every preflop decision that isn't the BB's option.
            # `pot_type=0` (no raises yet) is what actually means "open."
            street=0, pot_type=0, position="UTG",
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_2bb", "ATo+, KJo+, QJo+, 55+, T9s+, J9s+, Q8s+, K5s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="hj_open_100bb",
        label="HJ Open -- 100BB stack",
        matcher=SpotMatcher(
            # Not `facing_bet=False`: preflop, everyone except the BB has a
            # nonzero call_amount even in an unraised pot (they still owe
            # the BB's forced post to continue), so `facing_bet` is True for
            # basically every preflop decision that isn't the BB's option.
            # `pot_type=0` (no raises yet) is what actually means "open."
            street=0, pot_type=0, position="HJ",
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_2bb", "A9o+, KTo+, QTo+, JTo+, 44+, T8s+, J8s+, Q6s+, K4s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="co_open_100bb",
        label="CO Open -- 100BB stack",
        matcher=SpotMatcher(
            # Not `facing_bet=False`: preflop, everyone except the BB has a
            # nonzero call_amount even in an unraised pot (they still owe
            # the BB's forced post to continue), so `facing_bet` is True for
            # basically every preflop decision that isn't the BB's option.
            # `pot_type=0` (no raises yet) is what actually means "open."
            street=0, pot_type=0, position="CO",
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_2.3bb", "A5o, A8o+, KTo+, QTo+, JTo+, 44+, 98s, T8s+, J7s+, Q5s+, K3s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="btn_open_100bb",
        label="BTN Open -- 100BB stack",
        matcher=SpotMatcher(
            street=0, pot_type=0, position="BTN",
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            (
                "raise_2.5bb", "A3o+, K8o+, Q9o+, J9o+, T8o+, 33+, AKs-54s, AQs-75s, 96s, T6s+, J4s+, Q2s+, K2s+, A2s+"
            ),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="sb_open_100bb",
        label="SB Open -- 100BB stack",
        matcher=SpotMatcher(
            # Not `facing_bet=False`: preflop, everyone except the BB has a
            # nonzero call_amount even in an unraised pot (they still owe
            # the BB's forced post to continue), so `facing_bet` is True for
            # basically every preflop decision that isn't the BB's option.
            # `pot_type=0` (no raises yet) is what actually means "open."
            street=0, pot_type=0, position="SB",
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_3bb", "A4o+, K8o+, Q9o+, J9o+, T9o+, 22+, AKs-54s, AQs-64s, AJs-85s, T6s+, J4s+, Q2s+, K2s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="HJ_vs_utg_open_100bb",
        label="HJ vs UTG Open -- 100BB stack",
        matcher=SpotMatcher(
            # `raised_positions=("UTG",)` pins this to *exactly* an UTG open
            # reaching the BB unopposed -- pot_type=1 alone would also match
            # "everyone else folded to a HJ/CO/BTN/SB open," which needs a
            # different (wider) defense.
            street=0, pot_type=1, position="HJ", facing_bet=True, is_aggressor=False,
            raised_positions=("UTG",),
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_6.5bb", "TT+, ATs+, A8s, A5s, A4s, A3s, KTs+, K5s, AQo+, KQo"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="CO_vs_utg_open_100bb",
        label="CO vs UTG Open -- 100BB stack",
        matcher=SpotMatcher(
            # `raised_positions=("UTG",)` pins this to *exactly* an UTG open
            # reaching the BB unopposed -- pot_type=1 alone would also match
            # "everyone else folded to a HJ/CO/BTN/SB open," which needs a
            # different (wider) defense.
            street=0, pot_type=1, position="CO", facing_bet=True, is_aggressor=False,
            raised_positions=("UTG",),
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_6.5bb", "99+, AQs+, A8s, A7s, A5s, A4s, A3s, KJs, KTs, K5s, AQo+, KQo"),
            ("call", "88, ATs-AQs, KQs, QTs, JTs, 54s"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="BTN_vs_utg_open_100bb",
        label="BTN vs UTG Open -- 100BB stack",
        matcher=SpotMatcher(
            # `raised_positions=("UTG",)` pins this to *exactly* an UTG open
            # reaching the BB unopposed -- pot_type=1 alone would also match
            # "everyone else folded to a HJ/CO/BTN/SB open," which needs a
            # different (wider) defense.
            street=0, pot_type=1, position="BTN", facing_bet=True, is_aggressor=False,
            raised_positions=("UTG",),
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_7.5bb", "QQ+, AKs+, A8s, A7s, A5s, A4s, A3s, A2s, K5s, AKo, KQo"),
            ("call", "66-QQ, A5s, A9s-AQs, K9s-KQs, QTs-QJs, JTs, T9s, 65s, 54s, AQo"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="sb_vs_utg_open_100bb",
        label="SB vs UTG Open -- 100BB stack",
        matcher=SpotMatcher(
            # `raised_positions=("UTG",)` pins this to *exactly* an UTG open
            # reaching the BB unopposed -- pot_type=1 alone would also match
            # "everyone else folded to a HJ/CO/BTN/SB open," which needs a
            # different (wider) defense.
            street=0, pot_type=1, position="SB", facing_bet=True, is_aggressor=False,
            raised_positions=("UTG",),
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_10bb", "QQ+, AQs+, A8s, A4s, A3s, AKo"),
            ("call", "66-JJ, A5s, A7s-AJs, KTs-KQs, QTs-QJs, JTs, T9s"),
        ),
        default_action="fold",
        ),
    GTOSpot(
        key="bb_vs_utg_open_100bb",
        label="BB vs UTG Open -- 100BB stack",
        matcher=SpotMatcher(
            # `raised_positions=("UTG",)` pins this to *exactly* an UTG open
            # reaching the BB unopposed -- pot_type=1 alone would also match
            # "everyone else folded to a HJ/CO/BTN/SB open," which needs a
            # different (wider) defense.
            street=0, pot_type=1, position="BB", facing_bet=True, is_aggressor=False,
            raised_positions=("UTG",),
            min_effective_bb=80, max_effective_bb=120,
        ),
        action_ranges=(
            ("raise_12bb", "KK+, AKs+, AKo"),
            ("call", "22-QQ, A2s+, K2s+, Q2s+, J5s+, T6s+, 95s+, 84s+, 74s+, 63s+, 52s+, 42s+, 32s+, A5o, A8o+, KTo+, QTo+, JTo+, T9o+, 98o"),
        ),
        default_action="fold",
    ),
]

NUM_GTO_SPOTS = len(GTO_SPOTS)
