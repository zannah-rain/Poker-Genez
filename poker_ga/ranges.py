"""Human-readable starting-hand ranges, e.g. "AA-77, AJs+, AQo+, KQs",
parsed into the standard 169-hand canonical abstraction: 13 pocket pairs +
78 suited + 78 offsuit two-card combos, ignoring exact suit -- preflop, all
4 suits of a suited combo play identically by symmetry, and likewise for
offsuit combos, so this is the same abstraction every poker range tool
(PokerStove, GTO+, etc.) and range chart uses. This is what lets gto.py's
GTOSpot charts be written the same way a human would write them down.

Supported token syntax (comma-separated, whitespace-insensitive):
  - exact pair:            "77"
  - pair range:             "AA-77"   (77, 88, ..., AA)
  - pair "+":                "77+"     (77 up to AA)
  - exact suited/offsuit:   "AKs" / "AKo"
  - suited/offsuit "+":     "AJs+"    (AJs, AQs, AKs -- top card fixed, second card up to one below it)
  - fixed-top-card range:   "AJs-A5s" (AJs, ATs, A9s, A8s, A7s, A6s, A5s)
  - matching-gap connector range: "T9s-54s" (T9s, 98s, 87s, 76s, 65s, 54s)
"""

from __future__ import annotations

import re

from cards import RANKS, Card  # RANKS = "23456789TJQKA" -- index 0 = '2' ... 12 = 'A'

RANK_TO_INDEX = {r: i for i, r in enumerate(RANKS)}


def _canonical_label(rank1_idx: int, rank2_idx: int, suited: bool) -> str:
    """rank*_idx are 0(='2')..12(='A'). Returns e.g. "AKs", "AKo", "77"."""
    if rank1_idx == rank2_idx:
        return RANKS[rank1_idx] * 2
    hi, lo = max(rank1_idx, rank2_idx), min(rank1_idx, rank2_idx)
    return f"{RANKS[hi]}{RANKS[lo]}{'s' if suited else 'o'}"


def hand_label(card1: Card, card2: Card) -> str:
    """The canonical range-chart label for two hole cards, e.g. "AKs"."""
    r1, r2 = card1.rank - 2, card2.rank - 2
    suited = card1.suit == card2.suit
    return _canonical_label(r1, r2, suited)


ALL_HAND_LABELS = frozenset(
    [RANKS[i] * 2 for i in range(13)]
    + [_canonical_label(hi, lo, True) for hi in range(13) for lo in range(hi)]
    + [_canonical_label(hi, lo, False) for hi in range(13) for lo in range(hi)]
)  # all 169 canonical starting hands


def _normalize(token: str) -> str:
    """Case-insensitive: uppercase the rank characters, lowercase a trailing
    suited/offsuit marker if present."""
    token = token.strip()
    if len(token) >= 3 and token[-1] in "soSO":
        return token[:-1].upper() + token[-1].lower()
    return token.upper()


def _parse_pair_token(token: str) -> int | None:
    """Returns the rank index if `token` is a bare pair like "77", else None."""
    token = _normalize(token)
    if len(token) == 2 and token[0] == token[1] and token[0] in RANK_TO_INDEX:
        return RANK_TO_INDEX[token[0]]
    return None


def _parse_nonpair_token(token: str) -> tuple[int, int, str] | None:
    """Returns (hi_rank_idx, lo_rank_idx, "s"|"o") for tokens like "AKs"/"AKo", else None."""
    token = _normalize(token)
    if len(token) == 3 and token[2] in ("s", "o"):
        r1, r2 = token[0], token[1]
        if r1 in RANK_TO_INDEX and r2 in RANK_TO_INDEX and r1 != r2:
            i1, i2 = RANK_TO_INDEX[r1], RANK_TO_INDEX[r2]
            return max(i1, i2), min(i1, i2), token[2]
    return None


def _expand_range(start_token: str, end_token: str) -> set[str]:
    start_pair, end_pair = _parse_pair_token(start_token), _parse_pair_token(end_token)
    if start_pair is not None and end_pair is not None:
        lo, hi = sorted([start_pair, end_pair])
        return {RANKS[i] * 2 for i in range(lo, hi + 1)}

    start_np, end_np = _parse_nonpair_token(start_token), _parse_nonpair_token(end_token)
    if start_np is not None and end_np is not None:
        s_hi, s_lo, s_suit = start_np
        e_hi, e_lo, e_suit = end_np
        if s_suit != e_suit:
            raise ValueError(f"Range endpoints must both be suited or both offsuit: {start_token}-{end_token}")
        if s_hi == e_hi:
            # Fixed top card, second card ranges, e.g. "AJs-A5s".
            lo, hi = sorted([s_lo, e_lo])
            return {_canonical_label(s_hi, i, s_suit == "s") for i in range(lo, hi + 1)}
        # Matching-gap connector range, e.g. "T9s-54s": both ranks shift down
        # together, one gap apart the whole way.
        s_gap, e_gap = s_hi - s_lo, e_hi - e_lo
        if s_gap != e_gap:
            raise ValueError(f"Unsupported range (mismatched gaps): {start_token}-{end_token}")
        lo_hi, hi_hi = sorted([s_hi, e_hi])
        return {
            _canonical_label(top, top - s_gap, s_suit == "s")
            for top in range(lo_hi, hi_hi + 1)
            if 0 <= top - s_gap < 13
        }

    raise ValueError(f"Could not parse range endpoints: {start_token!r}-{end_token!r}")


def _expand_token(token: str) -> set[str]:
    token = token.strip()
    if not token:
        return set()

    plus = token.endswith("+")
    if not plus and "-" in token:
        start_token, _, end_token = token.partition("-")
        return _expand_range(start_token, end_token)

    base = token[:-1] if plus else token
    pair_rank = _parse_pair_token(base)
    if pair_rank is not None:
        if plus:
            return {RANKS[i] * 2 for i in range(pair_rank, 13)}
        return {RANKS[pair_rank] * 2}

    nonpair = _parse_nonpair_token(base)
    if nonpair is not None:
        hi, lo, suit = nonpair
        if plus:
            return {_canonical_label(hi, i, suit == "s") for i in range(lo, hi)}
        return {_canonical_label(hi, lo, suit == "s")}

    raise ValueError(f"Could not parse range token: {token!r}")


def parse_range(range_str: str) -> frozenset[str]:
    """Parses a comma-separated range string into the set of canonical hand
    labels it covers -- see module docstring for supported syntax."""
    labels: set[str] = set()
    for token in range_str.split(","):
        token = token.strip()
        if token:
            labels |= _expand_token(token)
    return frozenset(labels)


def in_range(card1: Card, card2: Card, range_set: frozenset[str]) -> bool:
    return hand_label(card1, card2) in range_set
