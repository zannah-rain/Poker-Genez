"""Card and deck primitives for the poker GA framework."""

from __future__ import annotations

import random
from dataclasses import dataclass

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

RANK_VALUES = {r: i + 2 for i, r in enumerate(RANKS)}  # '2' -> 2 ... 'A' -> 14


@dataclass(frozen=True, order=True)
class Card:
    rank: int  # 2..14
    suit: int  # 0..3

    def __repr__(self) -> str:
        return f"{RANKS[self.rank - 2]}{SUITS[self.suit]}"

    @staticmethod
    def from_str(s: str) -> "Card":
        rank = RANK_VALUES[s[0].upper()]
        suit = SUITS.index(s[1].lower())
        return Card(rank, suit)


def make_deck() -> list[Card]:
    return [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]


class Deck:
    """A shuffled 52-card deck that deals without replacement."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self._cards = make_deck()
        self._rng.shuffle(self._cards)
        self._pos = 0

    def deal(self, n: int = 1) -> list[Card]:
        if self._pos + n > len(self._cards):
            raise ValueError("Not enough cards left in deck")
        dealt = self._cards[self._pos : self._pos + n]
        self._pos += n
        return dealt
