"""A player is just an identity + a genome. Chip stacks live on the table/
session, not here, so the same player object can be reused across many
sessions when accumulating fitness."""

from __future__ import annotations

from dataclasses import dataclass

from genome import Genome


@dataclass
class Player:
    player_id: int
    genome: Genome
    generation: int = 0
    label: str = ""

    def __repr__(self) -> str:
        name = self.label or f"P{self.player_id}"
        return f"<{name} gen={self.generation}>"
