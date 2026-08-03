"""What a decision-maker wants to do, independent of game mechanics
(legal_actions, chip clamping) -- shared between Single Deep CFR's
action-category space (cfr_actions.py) and genome.py's legal-action
fallback glue (_apply_decision)."""

from __future__ import annotations

from dataclasses import dataclass

import strategy
from features import Situation


@dataclass(frozen=True)
class Decision:
    """kind: "fold" | "call" | "raise". `bet_size` (chips) is only
    meaningful when kind == "raise"."""

    kind: str
    bet_size: float = 0.0


def _standard_decision(action_index: int, situation: Situation) -> Decision:
    if action_index == strategy.ACTION_FOLD:
        return Decision("fold")
    if action_index == strategy.ACTION_CALL:
        return Decision("call")
    if action_index == strategy.ACTION_ALLIN:
        return Decision("raise", situation.my_stack)
    fraction = strategy.RAISE_POT_FRACTION[action_index]
    return Decision("raise", fraction * max(situation.pot, 1.0))
