"""The game engine's 3-way action space (Fold / Check-or-Call / Bet-or-Raise)
and the shared glue that turns a rules.Decision into one of those, applying
the legal-action fallbacks every decision-maker needs (a rule-based genome's
decide(), Single Deep CFR's cfr_actions.category_to_game_action, ...).
"""

from __future__ import annotations

import rules

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]


def _apply_decision(decision: rules.Decision, legal_actions: list[int]) -> tuple[int, float]:
    """Converts a rule's Decision into a (game_action, bet_size) pair,
    applying the shared legal-action fallbacks (Fold -> Check/Call if
    folding isn't legal, Raise -> Check/Call if raising isn't legal)."""
    if decision.kind == "fold":
        action = FOLD if FOLD in legal_actions else CHECK_CALL
        return action, 0.0
    if decision.kind == "call":
        return CHECK_CALL, 0.0
    # "raise"
    if BET_RAISE not in legal_actions:
        return CHECK_CALL, 0.0
    return BET_RAISE, decision.bet_size
