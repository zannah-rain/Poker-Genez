"""The discrete action-category vocabulary Single Deep CFR's advantage
network predicts over (ACTION_CATEGORIES/NUM_ACTION_CATEGORIES/...), shared
by cfr_actions.py (translating to/from the game engine's 3-way legal-action
space), rules.py's _standard_decision, and cfr_networks.py/cfr_explorer.py
(labeling a net's per-category output)."""

from __future__ import annotations

# Named ACTION_* (not FOLD/CALL/...) to stay visually distinct from
# genome.py's own FOLD/CHECK_CALL/BET_RAISE game-action constants -- these
# index ACTION_CATEGORIES (the net's chosen strategy category), those index
# legal_actions (what the game engine will actually accept).
#
# Raise is 6 separate fixed-size categories rather than one "Raise" category
# plus a continuous size output, so the net picks its size directly (e.g.
# "Raise 75% Pot") off a small, human-memorizable menu. Ordered from least
# to most aggressive alongside Fold/Call/All-In so the whole list is one
# continuum.
(
    ACTION_FOLD, ACTION_CALL,
    ACTION_RAISE_25, ACTION_RAISE_50, ACTION_RAISE_75,
    ACTION_RAISE_100, ACTION_RAISE_125, ACTION_RAISE_150,
    ACTION_ALLIN,
) = range(9)
ACTION_CATEGORIES = [
    "Check / Fold (Give up)", "Call",
    "Raise 25% Pot", "Raise 50% Pot", "Raise 75% Pot",
    "Raise 100% Pot", "Raise 125% Pot", "Raise 150% Pot",
    "All-In",
]
NUM_ACTION_CATEGORIES = len(ACTION_CATEGORIES)

# action index -> pot-fraction raise size, for every Raise category (added
# on top of whatever's already bet this street).
RAISE_ACTIONS = (
    ACTION_RAISE_25, ACTION_RAISE_50, ACTION_RAISE_75,
    ACTION_RAISE_100, ACTION_RAISE_125, ACTION_RAISE_150,
)
RAISE_POT_FRACTION = {
    ACTION_RAISE_25: 0.25, ACTION_RAISE_50: 0.5, ACTION_RAISE_75: 0.75,
    ACTION_RAISE_100: 1.0, ACTION_RAISE_125: 1.25, ACTION_RAISE_150: 1.5,
}
