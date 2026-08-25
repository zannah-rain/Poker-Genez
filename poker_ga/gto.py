"""Spots: the pieces that let a CFR traversal (and, at inference time,
cfr_policy.DeepCFRPolicy) memorize an exact, solved-looking strategy for a
specific, well-defined situation, the way a human would memorize a range
chart for "UTG open, 100BB" and just apply it verbatim rather than trusting
the net's own guess there.

A GTOSpot combines:
  - a SpotMatcher: a readable, declarative definition of which situations
    this spot applies to (street, pot type, position, stack depth, etc.)
  - action_ranges: which range (see ranges.py) takes which action in that
    spot, checked in order -- the first range a hand falls in wins;
    default_action otherwise. An action token is either one of
    strategy.ACTION_CATEGORIES' pot-relative sizes ("fold", "call",
    "raise_25".."raise_150", "allin" -- the same 9-way vocabulary the net
    itself predicts over) or a fixed "raise_<N>bb" (e.g. "raise_1.5bb"): a
    raise sized to exactly N big blinds regardless of pot size, for spots
    (e.g. small, fixed-size continuation bets) better expressed that way
    than as a pot fraction.

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
from rules import Decision, _standard_decision
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

# "raise_<N>bb" (e.g. "raise_1.5bb", "raise_3bb") -- a fixed number of big
# blinds, unlike _ACTION_TOKENS' pot-fraction categories, whose chip size
# grows/shrinks with the pot. Matched against the already-lowercased token,
# so the "BB" suffix in a chart like "raise_1.5BB" is case-insensitive.
_BB_RAISE_TOKEN_RE = re.compile(r"raise_(\d+(?:\.\d+)?)bb")


@dataclass(frozen=True)
class _ActionSpec:
    """An action_ranges/default_action token, resolved down to either a
    strategy.ACTION_CATEGORIES index (a pot-relative action, or fold/call)
    or a fixed big-blind raise size -- exactly one of the two is set.
    Deferred one step further than a plain Decision since a category's own
    chip size still depends on the Situation it's actually played in (its
    pot), whereas a fixed-BB raise's chip size only depends on that
    Situation's big_blind (the same every hand) -- either way, `decision`
    is what finally turns this into game-mechanics-ready chips."""

    category: int | None = None
    fixed_raise_bb: float | None = None

    def decision(self, situation: Situation) -> Decision:
        # bet_size is a chip amount on top of whatever's already required to
        # call this street -- the same "raise BY this much" convention
        # rules._standard_decision's own pot-fraction categories use (see
        # cfr_tree._apply_raise), just sized in a fixed number of big blinds
        # instead of a fraction of the (possibly very different, hand to
        # hand) pot.
        if self.fixed_raise_bb is not None:
            return Decision("raise", self.fixed_raise_bb * situation.big_blind)
        return _standard_decision(self.category, situation)


def parse_action_token(token: str) -> _ActionSpec:
    """A GTOSpot chart token -- either one of _ACTION_TOKENS (e.g.
    "raise_150", a pot-relative category) or "raise_<N>bb" (e.g.
    "raise_1.5bb", a fixed big-blind raise size regardless of pot) -- to
    the _ActionSpec that resolves it. Raises ValueError for anything else
    -- a typo here should fail loudly at catalog-definition time, not
    silently resolve to the wrong action."""
    token = token.strip().lower()
    if token in _ACTION_TOKENS:
        return _ActionSpec(category=_ACTION_TOKENS[token])
    match = _BB_RAISE_TOKEN_RE.fullmatch(token)
    if match:
        return _ActionSpec(fixed_raise_bb=float(match.group(1)))
    raise ValueError(
        f"Unknown GTO action token {token!r}; expected one of {sorted(_ACTION_TOKENS)} or \"raise_<N>bb\" (e.g. \"raise_1.5bb\")"
    )


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
    default_action_spec: _ActionSpec = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        resolved = tuple((parse_action_token(action), parse_range(range_str)) for action, range_str in self.action_ranges)
        object.__setattr__(self, "resolved_ranges", resolved)
        object.__setattr__(self, "default_action_spec", parse_action_token(self.default_action))


def resolve_spot_action(spot: GTOSpot, situation: Situation) -> Decision | None:
    """This decision's fixed rules.Decision if `situation` matches `spot`
    (falling back to spot.default_action_spec if the hand isn't in any of
    its listed ranges), else None if the spot doesn't apply to this
    situation at all."""
    if not spot.matcher.matches(situation):
        return None
    label = hand_label(situation.hole[0], situation.hole[1])
    for action_spec, range_set in spot.resolved_ranges:
        if label in range_set:
            return action_spec.decision(situation)
    return spot.default_action_spec.decision(situation)


def first_matching_action(spots: tuple[GTOSpot, ...], situation: Situation) -> Decision | None:
    """The first spot (in catalog order) whose matcher applies to
    `situation`, resolved to its fixed Decision -- None if no spot in
    `spots` applies at all, in which case the caller should fall through to
    its normal (learned) decision-making instead."""
    for spot in spots:
        decision = resolve_spot_action(spot, situation)
        if decision is not None:
            return decision
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
            ("raise_2BB", "ATo+, KJo+, 55+, T9s+, J9s+, Q9s+, K5s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="hj_open_100bb",
        label="HJ Open -- 100BB stack",
        matcher=SpotMatcher(street=0, pot_type=0, position="HJ", facing_bet=False, min_effective_bb=80, max_effective_bb=120),
        action_ranges=(
            ("raise_2BB", "ATo+, KTo+, QJo+, 44+, 65s, 87s+, 97s+, T8s+, J8s+, Q8s+, K4s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
        key="co_open_100bb",
        label="CO Open -- 100BB stack",
        matcher=SpotMatcher(street=0, pot_type=0, position="CO", facing_bet=False, min_effective_bb=80, max_effective_bb=120),
        action_ranges=(
            ("raise_2.3BB", "A5o, A8o+, KTo+, QTo+, JTo+, 44+, 65s+, 97s+, T7s+, J7s+, Q5s+, K2s+, A2s+"),
        ),
        default_action="fold",
    ),
    GTOSpot(
            key="btn_open_100bb",
            label="BTN Open -- 100BB stack",
            matcher=SpotMatcher(street=0, pot_type=0, position="BTN", facing_bet=False, min_effective_bb=80, max_effective_bb=120),
            action_ranges=(
                ("raise_2.5BB", "A3o+, K8o+, Q9o+, J9o+, T9o+, 22+, 54s+, 75s+, 96s+, T5s+, J4s+, Q2s+, K2s+, A2s+"),
            ),
            default_action="fold",
    ),
    GTOSpot(
        key="sb_open_100bb",
        label="SB Open -- 100BB stack",
        matcher=SpotMatcher(street=0, pot_type=0, position="SB", facing_bet=False, min_effective_bb=80, max_effective_bb=120),
        action_ranges=(
            ("call", "74s, 95s, T5s, T4s, J3s, J2s, A3s, Qts, Q9s"),
            ("raise_3BB", "A3o+, K8o+, Q9o+, J9o+, T8o+, 98o+, 22+, 54s+, 53s+, 85s+, 96s+, T6s+, J4s+, Q2s+, K2s+, A4s+, A2s"),
        ),
        default_action="fold",
    ),
    # GTOSpot(
    #     key="bb_vs_single_raise_100bb",
    #     label="BB Facing A Single Raise -- 100BB stack",
    #     matcher=SpotMatcher(street=0, pot_type=1, position="BB", facing_bet=True, is_aggressor=False, min_effective_bb=80, max_effective_bb=120),
    #     action_ranges=(
    #         ("raise_150", "TT+, AQs+, AKo"),
    #         ("call", "22-99, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, ATo+, KJo+"),
    #     ),
    #     default_action="fold",
    # ),
    # GTOSpot(
    #     key="btn_vs_3bet_20bb",
    #     label="BTN Facing A 3-Bet -- 20BB stack",
    #     matcher=SpotMatcher(street=0, pot_type=2, position="BTN", facing_bet=True, is_aggressor=False, min_effective_bb=15, max_effective_bb=25),
    #     action_ranges=(
    #         ("allin", "88+, AJs+, KQs, AQo+"),
    #     ),
    #     default_action="fold",
    # ),
)
