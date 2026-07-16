"""Seat-position helpers shared between the game engine (game.py) and feature
extraction (features.py). Kept dependency-free (no poker_ga imports) so both
can use it without risking a circular import.
"""

from __future__ import annotations


def blind_indices(button_idx: int, n: int) -> tuple[int, int]:
    """Returns (small_blind_seat, big_blind_seat). Heads-up, the button posts
    the small blind."""
    if n == 2:
        return button_idx, (button_idx + 1) % n
    return (button_idx + 1) % n, (button_idx + 2) % n


def preflop_order(button_idx: int, n: int) -> list[int]:
    """Preflop action order: first-to-act ... button, small blind, big blind
    (the blinds act last preflop since they already have chips in)."""
    _, bb_idx = blind_indices(button_idx, n)
    start = (bb_idx + 1) % n
    return [(start + k) % n for k in range(n)]


# Labels for the non-blind seats, UTG-first ... BTN-last, keyed by how many
# non-blind seats exist (n - 2). Matches standard naming at every table size
# the framework supports (2-6 handed) rather than mechanically re-using the
# full 6-max names when seats don't exist -- e.g. at a 4-handed table the
# lone seat before the button is UTG, not "CO", since there's no separate
# hijack/cutoff distinction to be made with only 2 non-blind seats.
_NON_BLIND_LABEL_SCHEMES: dict[int, list[str]] = {
    0: [],
    1: ["BTN"],
    2: ["UTG", "BTN"],
    3: ["UTG", "CO", "BTN"],
    4: ["UTG", "HJ", "CO", "BTN"],
}

SEAT_ROLES = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]


def seat_role(seat_idx: int, button_idx: int, n: int) -> str:
    """Returns this seat's starting position for the hand: one of
    'UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB'. Heads-up, the button/small-blind
    seat is labeled 'SB' (they're the same seat)."""
    sb_idx, bb_idx = blind_indices(button_idx, n)
    if seat_idx == sb_idx:
        return "SB"
    if seat_idx == bb_idx:
        return "BB"
    non_blind = preflop_order(button_idx, n)[:-2] if n > 2 else []
    labels = _NON_BLIND_LABEL_SCHEMES[len(non_blind)]
    return labels[non_blind.index(seat_idx)]
