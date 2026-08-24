"""Spots: the pieces that let a CFR traversal (and, at inference time,
cfr_policy.DeepCFRPolicy) memorize an exact, solved-looking strategy for a
specific, well-defined situation, the way a human would memorize a range
chart for "UTG open, 100BB" and just apply it verbatim rather than trusting
the net's own guess there.

A GTOSpot combines:
  - a SpotMatcher: a readable, declarative definition of which situations
    this spot applies to (street, pot type, position, stack depth, etc.)
  - action_ranges: which strategy.ACTION_CATEGORIES token takes which range
    (see ranges.py) in that spot, checked in order -- the first range a
    hand falls in wins; default_action otherwise.

Unlike a genome's evolvable weights, a GTOSpot's chart is never learned --
it's a hard override: whenever a decision's Situation matches an active
spot, that decision is played exactly as the chart says (see
cfr_tree._decision_node/cfr_policy.DeepCFRPolicy.decide), for *every* seat,
not just whichever one the net would otherwise be training. The net still
learns the optimal response everywhere else, including its own best reply
to those fixed actions -- since every other seat plays them too, "everyone
opens 100BB UTG with this exact range" becomes part of the environment the
rest of the strategy is trained against, not something left to chance.

This is deliberately a *fixed, code-defined catalog* (GTO_SPOTS below), the
same pattern as features.py's FEATURE_SPECS -- extend it by adding entries,
not by making it runtime-configurable. The example spots included are
illustrative (reasonable, hand-picked ranges), not verified solver output;
swap in real solved charts for genuine accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import strategy
from features import Situation
from ranges import hand_label, parse_range
from seating import seat_role

POT_TYPE_LABELS = ("Unraised", "Single Raised", "3-Bet", "4-Bet+")

# Short tokens a GTOSpot's action_ranges/default_action can use, mapped onto
# strategy.ACTION_CATEGORIES -- the same 9-way vocabulary the net itself
# predicts over, so a fixed spot's chart and a learned decision are always
# directly comparable/interchangeable.
_ACTION_TOKENS: dict[str, int] = {
    "fold": strategy.ACTION_FOLD,
    "call": strategy.ACTION_CALL,
    "raise_25": strategy.ACTION_RAISE_25,
    "raise_50": strategy.ACTION_RAISE_50,
    "raise_75": strategy.ACTION_RAISE_75,
    "raise_100": strategy.ACTION_RAISE_100,
    "raise_125": strategy.ACTION_RAISE_125,
    "raise_150": strategy.ACTION_RAISE_150,
    "allin": strategy.ACTION_ALLIN,
}


def parse_action_token(token: str) -> int:
    """A GTOSpot chart token (e.g. "raise_150") -> its
    strategy.ACTION_CATEGORIES index. Raises ValueError (listing every
    valid token) for anything else -- a typo here should fail loudly at
    catalog-definition time, not silently resolve to the wrong action."""
    token = token.strip().lower()
    if token not in _ACTION_TOKENS:
        raise ValueError(f"Unknown GTO action token {token!r}; expected one of {sorted(_ACTION_TOKENS)}")
    return _ACTION_TOKENS[token]


@dataclass(frozen=True)
class SpotMatcher:
    """Declarative criteria checked against a Situation to decide whether a
    GTOSpot applies. Every field is optional (None = "don't care"); a
    situation matches only if every specified field agrees.

    Note: `is_aggressor` only distinguishes "I made the last raise" from "I
    didn't" -- Situation doesn't track *which* seat made a raise, so a
    matcher can express "BTN facing a 3-bet" but not "BTN facing a 3-bet
    specifically from the BB"."""

    street: int | None = None  # 0=preflop, 1=flop, 2=turn, 3=river
    pot_type: int | None = None  # 0=unraised, 1=single raised, 2=3-bet, 3=4-bet+ (num_preflop_raises, capped at 3)
    position: str | None = None  # one of seating.SEAT_ROLES, or None for "any position"
    is_aggressor: bool | None = None  # did I make the last bet/raise on the *previous* street?
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
    key: str  # short, stable identifier
    label: str  # human-readable description, e.g. "UTG Open -- 100BB stack"
    matcher: SpotMatcher
    action_ranges: tuple[tuple[str, str], ...]  # ((action_token, range_str), ...), checked in order
    default_action: str = "fold"  # applied if the hand isn't in any listed range, once the spot matches
    resolved_ranges: tuple = field(init=False, repr=False, compare=False)
    default_action_index: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        resolved = tuple((parse_action_token(action), parse_range(range_str)) for action, range_str in self.action_ranges)
        object.__setattr__(self, "resolved_ranges", resolved)
        object.__setattr__(self, "default_action_index", parse_action_token(self.default_action))


def resolve_spot_action(spot: GTOSpot, situation: Situation) -> int | None:
    """This decision's fixed strategy.ACTION_CATEGORIES index if `situation`
    matches `spot` (falling back to spot.default_action_index if the hand
    isn't in any of its listed ranges), else None if the spot doesn't apply
    to this situation at all."""
    if not spot.matcher.matches(situation):
        return None
    label = hand_label(situation.hole[0], situation.hole[1])
    for action_index, range_set in spot.resolved_ranges:
        if label in range_set:
            return action_index
    return spot.default_action_index


def first_matching_action(spots: tuple[GTOSpot, ...], situation: Situation) -> int | None:
    """The first spot (in catalog order) whose matcher applies to
    `situation`, resolved to its fixed action -- None if no spot in `spots`
    applies at all, in which case the caller should fall through to its
    normal (learned) decision-making instead."""
    for spot in spots:
        action = resolve_spot_action(spot, situation)
        if action is not None:
            return action
    return None


# A small, illustrative starter catalog -- hand-picked, reasonable ranges,
# not verified solver output. Extend by adding more GTOSpot entries; pass
# `gto.GTO_SPOTS` (or a subset) to cfr_tree.traverse_hand/cfr_train.
# DeepCFRConfig/cfr_policy.DeepCFRPolicy to actually make use of it -- an
# empty tuple (the default everywhere) leaves every decision fully learned,
# same as before this module existed.
GTO_SPOTS: tuple[GTOSpot, ...] = (
    GTOSpot(
        key="utg_open_100bb",
        label="UTG Open -- 100BB stack",
        matcher=SpotMatcher(street=0, pot_type=0, position="UTG", facing_bet=False, min_effective_bb=80, max_effective_bb=120),
        action_ranges=(
            ("raise_150", "77+, ATs+, KTs+, QTs+, JTs, T9s, 98s, AJo+, KQo"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="btn_open_100bb",
        label="BTN Open -- 100BB stack",
        matcher=SpotMatcher(street=0, pot_type=0, position="BTN", facing_bet=False, min_effective_bb=80, max_effective_bb=120),
        action_ranges=(
            (
                "raise_150",
                "22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 96s+, 86s+, 75s+, 64s+, 53s+, "
                "A2o+, K8o+, Q9o+, J9o+, T9o",
            ),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="bb_vs_single_raise_100bb",
        label="BB Facing A Single Raise -- 100BB stack",
        matcher=SpotMatcher(street=0, pot_type=1, position="BB", facing_bet=True, is_aggressor=False, min_effective_bb=80, max_effective_bb=120),
        action_ranges=(
            ("raise_150", "TT+, AQs+, AKo"),
            ("call", "22-99, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, ATo+, KJo+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="btn_vs_3bet_20bb",
        label="BTN Facing A 3-Bet -- 20BB stack",
        matcher=SpotMatcher(street=0, pot_type=2, position="BTN", facing_bet=True, is_aggressor=False, min_effective_bb=15, max_effective_bb=25),
        action_ranges=(
            ("allin", "88+, AJs+, KQs, AQo+"),
        ),
        default_action="fold",
    ),
)
