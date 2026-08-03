"""A player is just an identity + a decision-maker. Chip stacks live on the
table/session, not here, so the same player object can be reused across many
sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Player:
    player_id: int
    # Duck-typed: anything with .decide(situation, legal_actions, rng) ->
    # (action, bet_size) -- e.g. cfr_policy.DeepCFRPolicy, or a test double
    # (see tests/test_game.py's FixedGenome).
    genome: Any
    generation: int = 0
    label: str = ""

    def __repr__(self) -> str:
        name = self.label or f"P{self.player_id}"
        return f"<{name} gen={self.generation}>"
