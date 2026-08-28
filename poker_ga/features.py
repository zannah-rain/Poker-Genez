"""Turns a poker decision point into a fixed-length feature vector.

This is the bridge between the game engine and a decision-maker: every
feature here is a "basic characteristic of the current situation" (hand
strength, board texture, betting context, position, stack depth...). A
Single Deep CFR advantage network (see cfr_features.py) reads a configured
subset of these directly.

Each "multi-value" characteristic (e.g. hand strength, high card rank) is
represented as a single ordinal/categorical 0-1 feature, since a neural
network can read that value directly and learn any non-linear response to
it. A feature can still declare a `linked_to` / `linked_value_index`
relationship to another feature -- e.g. the Exact Hole Hand grid's second
axis -- purely so exported reports can fold related rows together.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from cards import RANKS, Card
from evaluator import (
    FULL_HOUSE, FLUSH, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH,
    TRIPS, TWO_PAIR, best_hand_from_available, count_straight_draw_outs,
    has_backdoor_flush_draw, has_backdoor_straight_draw, straight_high,
)
from seating import SEAT_ROLES, seat_role


@dataclass(frozen=True)
class FeatureSpec:
    key: str  # internal identifier; used to look up this feature's value, not its position
    label: str  # short, human-readable name used in exported reports
    description: str  # precise definition of how the value is computed
    kind: str = "boolean"  # "boolean" (0/1) | "categorical" (fixed set of values) | "continuous" (a range)
    # For categorical/continuous features: every (or, for continuous, a representative
    # sample of) normalized value this feature can take, paired with a human-readable
    # label for that value. None for boolean features (they need no value table).
    value_table: tuple | None = None
    # Set on a boolean feature that is an exact indicator for one specific value of
    # another (categorical/continuous) feature -- e.g. "has_pair" is linked to
    # ("hand_category_norm", 1). None for features that stand on their own.
    linked_to: str | None = None
    linked_value_index: int | None = None
    # High-level category for organizing exported reports (e.g. "Made Hand
    # Features"). Only set on "top-level" specs -- parents of a value-table
    # family, and standalone booleans; linked children inherit their
    # parent's group instead of repeating it (see group_of() below).
    group: str | None = None
    # True for a feature that reads as MASKED in some situations (e.g. a
    # draw-shape feature at the river, once no more cards are coming to
    # complete or miss a draw) instead of a real 0-1 reading -- see MASKED's
    # own docstring. Lets cfr_features.py's generic bucket_label/
    # bucket_categories/bucket_labels recognize and label that reading
    # instead of nearest-point-matching it into whichever real bucket
    # happens to sit closest to a negative number.
    maskable: bool = False
    # True for a categorical/continuous feature whose value_table's own
    # 0.0 point is a genuinely distinct, discrete state (e.g. "no bet to
    # call at all") rather than just the low end of a continuous scale.
    # Plain nearest-point matching would otherwise silently absorb small
    # *nonzero* readings into that same bucket purely because they land
    # closer to 0.0 than to the next real point -- see call_amount_norm,
    # where a call of up to 1/8 pot would otherwise read as "No bet to
    # call" alongside a literal check. With this set, cfr_features.py's
    # bucket_label only ever returns the 0.0 point's own label for an
    # exact 0.0 reading; every other value matches among the *remaining*
    # points instead, never that one.
    zero_bucket_is_exact: bool = False


# Sentinel for a `maskable` FeatureSpec's reading whenever the situation
# makes that feature inapplicable (e.g. a draw-shape feature at the river,
# once no more cards are coming, or the Exact Hole Hand grid postflop, once
# there's a board to read instead of just the two hole cards) -- deliberately
# outside the 0-1 range every real feature value lives in, rather than
# reusing 0.0 (a real, if extreme, value for most features), so a masked
# reading can never be mistaken for a real one.
MASKED = -1.0


# Fixed, human-chosen reading order for report sections -- not derived from
# influence, since the point of grouping is a consistent, thematic structure
# a reader can learn once, not a ranking that shuffles per genome.
FEATURE_GROUPS = [
    "Hole Card Characteristics",
    "Board / Flop Characteristics",
    "Made Hand Features",
    "Draw Features",
    "Betting Behaviour Features",
    "Stack & Pot Features",
    "Table & Game State Features",
    "Opponent Tendency Features",
]


def _rank_label(rank: int) -> str:
    return {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}.get(rank, str(rank))


def _rank_key(rank: int) -> str:
    return {11: "jack", 12: "queen", 13: "king", 14: "ace"}.get(rank, str(rank))


# Gaps of 6+ are all equally "not going to make a straight together" (the
# widest gap that can still share a straight is 5, e.g. A-6 via A-2-3-4-5...6
# doesn't fit that one, but 2-7 can't either -- 5 apart is the practical
# ceiling), so anything wider is lumped into one catch-all bucket.
CONNECTIVITY_GAP_CAP = 6


def _rank_gap(r1: int, r2: int) -> int:
    """Rank distance for straight-connectivity purposes, treating the Ace as
    playable high (A-K = 1 apart) or low (A-2 = 1 apart) -- whichever gives
    the smaller gap -- then capped at CONNECTIVITY_GAP_CAP."""
    gap = abs(r1 - r2)
    if r1 == 14:
        gap = min(gap, abs(1 - r2))
    if r2 == 14:
        gap = min(gap, abs(r1 - 1))
    return min(gap, CONNECTIVITY_GAP_CAP)


def _ace_aware_span(ranks: list[int]) -> int:
    """max-min rank span, also trying the Ace as low (1) if present and
    taking whichever span is smaller -- a board like A-2-3 is tightly
    connected for low straights even though 14-2=12 looks wide apart under a
    naive high-Ace span."""
    span = max(ranks) - min(ranks)
    if 14 in ranks:
        low_ranks = [1 if r == 14 else r for r in ranks]
        span = min(span, max(low_ranks) - min(low_ranks))
    return span


_HOLE_CATEGORY_LABELS = (
    "Junk", "Unsuited Connectors", "Suited 2 Gappers", "Suited 1 Gappers", "Suited Connectors",
    "Qxs", "Kxs", "Axo", "Axs", "Average Pairs", "High Pairs", "Premium Pairs",
)  # index order = strength ascending; must match _hole_hand_category's return values


# The classic 13x13 preflop starting-hand grid every range-chart tool uses:
# both axes run Ace..deuce (index 0 = Ace), pocket pairs sit on the main
# diagonal, suited combos above it, offsuit combos below it -- see
# _hole_hand_grid_indices/hole_hand_grid_label.
HOLE_HAND_GRID_RANKS = tuple(range(14, 1, -1))  # 14 (Ace), 13 (King), ..., 2
HOLE_HAND_GRID_SIZE = len(HOLE_HAND_GRID_RANKS)
HOLE_HAND_GRID_RANK_LABELS = tuple(RANKS[r - 2] for r in HOLE_HAND_GRID_RANKS)  # "A", "K", ..., "2"


def _hole_hand_grid_indices(hole: list[Card]) -> tuple[int, int]:
    """(row, col) in [0, HOLE_HAND_GRID_SIZE) x [0, HOLE_HAND_GRID_SIZE) for
    this exact starting hand. A pocket pair always lands on the diagonal
    (row == col). Otherwise the higher-ranked card sets the row and the
    lower-ranked one sets the column for a suited combo (row < col, above
    the diagonal), and the reverse for offsuit (row > col, below the
    diagonal) -- see hole_hand_grid_label for the inverse mapping."""
    r0, r1 = hole[0].rank, hole[1].rank
    hi, lo = max(r0, r1), min(r0, r1)
    hi_idx, lo_idx = 14 - hi, 14 - lo
    if hi == lo:
        return hi_idx, hi_idx
    suited = hole[0].suit == hole[1].suit
    return (hi_idx, lo_idx) if suited else (lo_idx, hi_idx)


def hole_hand_grid_label(row: int, col: int) -> str:
    """Human-readable combo code for one (row, col) grid cell -- e.g. "AA",
    "AKs", "AKo" -- the inverse of _hole_hand_grid_indices. Used by
    cfr_explorer.py to label every cell of its Exact Hole Hand heatmaps."""
    hi_label = HOLE_HAND_GRID_RANK_LABELS[min(row, col)]
    lo_label = HOLE_HAND_GRID_RANK_LABELS[max(row, col)]
    if row == col:
        return hi_label + lo_label
    return hi_label + lo_label + ("s" if row < col else "o")


def _hole_hand_category(hole: list[Card]) -> int:
    """Classifies the two hole cards into exactly one of 12 mutually
    exclusive starting-hand buckets (indexes into _HOLE_CATEGORY_LABELS),
    checked strongest-to-weakest so a higher-priority bucket (e.g. a pair)
    always claims a hand before a looser one (e.g. a suited Ace) gets the
    chance to -- e.g. AKs is Axs, not Kxs; KQo is an Unsuited Connector, not
    Junk."""
    r1, r2 = hole[0].rank, hole[1].rank
    suited = hole[0].suit == hole[1].suit
    hi, lo = max(r1, r2), min(r1, r2)
    gap = hi - lo  # plain (non-Ace-wrapping) gap: only reached once hi < 14, so no low-Ace case applies

    if hi == lo:
        if hi >= 12:
            return 11  # Premium Pairs: AA/KK/QQ
        if hi >= 9:
            return 10  # High Pairs: JJ-99
        return 9  # Average Pairs: 22-88
    if hi == 14:
        return 8 if suited else 7  # Axs / Axo
    if hi == 13 and suited:
        return 6  # Kxs (Ace already excluded above)
    if hi == 12 and suited:
        return 5  # Qxs (Ace/King already excluded above)
    if suited and gap == 1:
        return 4  # Suited Connectors
    if suited and gap == 2:
        return 3  # Suited 1 Gappers
    if suited and gap == 3:
        return 2  # Suited 2 Gappers
    if not suited and gap == 1:
        return 1  # Unsuited Connectors
    return 0  # Junk


def _connectivity_label(gap: int) -> str:
    if gap == 0:
        return "Same rank (pocket pair)"
    if gap == 1:
        return "1 apart (connectors, e.g. 9-10)"
    if gap == 2:
        return "2 apart (one-gappers, e.g. 9-J)"
    if gap == 3:
        return "3 apart (two-gappers, e.g. 9-Q)"
    if gap < CONNECTIVITY_GAP_CAP:
        return f"{gap} apart"
    return f"{CONNECTIVITY_GAP_CAP} or more apart"


# Every made-hand category, in strength order -- but High Card, Pair, Three
# of a Kind, Straight, and Flush are each split into several sub-buckets
# (still ordered weakest-to-strongest within that category) instead of one
# flat value, so distinctions that used to live in a dozen-plus separate
# standalone booleans (top_pair, overpair, ace_high_no_pair, ..., "is this a
# set/nuts straight/ace-high flush") now read as *where on the made-hand
# strength spectrum* a hand sits, which a NN can generalize across the way
# it can't across unrelated flags. See _hand_category_bucket for exactly how
# each index is chosen.
#
# High Card (indices 0-2): High Card < King High < Ace High, by this
# player's overall (hole+board) high card -- folds what used to be the
# standalone ace_high_no_pair/king_high_no_pair booleans in here instead.
#
# Pair (indices 3-12), weakest to strongest:
#   Underpair   -- pocket pair strictly below every board card (e.g. 2-2 on
#                  K-9-4 -- beats none of it)
#   Ninja Pair  -- pocket pair below the board's top card, but beats at
#                  least one board card (e.g. 6-6 on K-9-4 -- beats the 4,
#                  loses to the K and 9)
#   Bottom Pair -- a non-pocket hole card matching the board's *lowest*
#                  distinct rank -- only distinct from Third Pair on a
#                  board with 4+ distinct ranks (a 3-distinct-rank board's
#                  lowest rank already *is* its 3rd highest, so that case
#                  reads as Third Pair instead -- see _pair_bucket_offset)
#   Pair        -- catch-all: a preflop pocket pair (no board yet), the
#                  board already paired on its own (contributing nothing
#                  from either hole card), or a postflop pair matching a
#                  board rank that's none of top/second/third/bottom (the
#                  4th-highest of 5 distinct ranks)
#   Third Pair, Second Pair, Top Pair -- a non-pocket hole card matching
#                  the board's 3rd/2nd/1st highest distinct rank
#   Top Pair + Good Kicker  -- Top Pair, kicker Jack/Queen/King
#   Top Pair + Top Kicker   -- Top Pair, kicker Ace
#   Overpair    -- pocket pair above the whole board (the strongest
#                  single-pair hand, hence placed right before Two Pair)
#
# Three of a Kind (indices 14-16): Bottom Set < Set < Top Set, by whether
# the tripped rank is the board's lowest/other/highest distinct rank (a
# non-pocket-pair "trips" via a paired board is treated as the plain
# middle "Set" bucket).
#
# Straight (indices 17-20): Bottom Straight < Straight < Top Straight <
# Nuts Straight, by comparing this straight's high card against every
# straight_high any 2 hole cards could make on this board -- Nuts requires
# also being the *top* straight AND no flush being possible for anyone
# (fewer than 3 board cards sharing a suit), since a possible flush would
# mean a straight -- even the best one -- isn't provably the best hand.
#
# Flush (indices 21-23): Flush < King High Flush < Ace High Flush, by the
# flush's own high card (not this player's overall hole+board high card,
# which can differ if the highest card isn't of the flush suit).
_HAND_CATEGORY_VALUES = (
    (0 / 26, "High Card"), (1 / 26, "King High"), (2 / 26, "Ace High"),
    (3 / 26, "Underpair"), (4 / 26, "Ninja Pair"), (5 / 26, "Bottom Pair"), (6 / 26, "Pair"),
    (7 / 26, "Third Pair"), (8 / 26, "Second Pair"), (9 / 26, "Top Pair"),
    (10 / 26, "Top Pair + Good Kicker"), (11 / 26, "Top Pair + Top Kicker"), (12 / 26, "Overpair"),
    (13 / 26, "Two Pair"),
    (14 / 26, "Bottom Set"), (15 / 26, "Set"), (16 / 26, "Top Set"),
    (17 / 26, "Bottom Straight"), (18 / 26, "Straight"), (19 / 26, "Top Straight"), (20 / 26, "Nuts Straight"),
    (21 / 26, "Flush"), (22 / 26, "King High Flush"), (23 / 26, "Ace High Flush"),
    (24 / 26, "Full House"), (25 / 26, "Four of a Kind"), (26 / 26, "Straight Flush"),
)
_HOLE_CATEGORY_VALUES = tuple((i / 11, label) for i, label in enumerate(_HOLE_CATEGORY_LABELS))
_HOLE_HAND_GRID_VALUES = tuple(
    (i / (HOLE_HAND_GRID_SIZE - 1), HOLE_HAND_GRID_RANK_LABELS[i]) for i in range(HOLE_HAND_GRID_SIZE)
)
_HOLE_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_SHARED_HIGH_CARD_VALUES = tuple(((r - 2) / 12, _rank_label(r)) for r in range(2, 15))
_CONNECTIVITY_VALUES = tuple(
    (1.0 - gap / CONNECTIVITY_GAP_CAP, _connectivity_label(gap)) for gap in range(0, CONNECTIVITY_GAP_CAP + 1)
)
_STREET_VALUES = ((0.0, "Preflop"), (1 / 3, "Flop"), (2 / 3, "Turn"), (1.0, "River"))
# 6 evenly-spaced points (0, 0.2, ..., 1.0) map to call/pot ratios 0, 0.25,
# 0.5, 0.75, 1.0, 1.25 once divided by _CALL_SIZE_CEILING below -- chosen so
# the first 5 points land exactly on the classic pot-fraction labels, with
# room left over for a genuine Overbet bucket instead of lumping "exactly
# pot-sized" and "way more than pot" into one clipped top bucket.
_CALL_SIZE_CEILING = 1.25
_CALL_SIZE_VALUES = (
    (0.0, "No bet to call"),
    (0.2, "Call is 1/4 pot"),
    (0.4, "Call is 1/2 pot"),
    (0.6, "Call is 3/4 pot"),
    (0.8, "Call is a full pot-sized bet"),
    (1.0, "Overbet (more than a full pot, clipped)"),
)
_SPR_VALUES = (
    (0.0, "SPR ≈ 0 (effectively all-in already)"),
    (0.25, "SPR ≈ 5 (shallow — about 5 pot-sized bets behind)"),
    (0.5, "SPR ≈ 10 (medium stack depth)"),
    (0.75, "SPR ≈ 15 (deep)"),
    (1.0, "SPR ≈ 20+ (very deep, clipped)"),
)
_POSITION_VALUES = (
    (0.0, "Acts first this street (out of position)"),
    (0.25, "Acts early"),
    (0.5, "Acts in the middle of the order"),
    (0.75, "Acts late"),
    (1.0, "Acts last this street (in position / on the button)"),
)
_ACTIVE_PLAYERS_VALUES = tuple((n / 6.0, f"{n} players still in the hand") for n in range(2, 7))
_OVERCARDS_VALUES = tuple(
    (
        n / 5.0,
        "No board cards rank higher than my high card" if n == 0
        else f"{n} board card{'s' if n != 1 else ''} rank{'s' if n == 1 else ''} higher than my high card",
    )
    for n in range(6)
)
# The single "Draw" family feature (draw_norm) -- see its own FeatureSpec
# below for why this collapses what used to be several separate draw
# signals into one, ordered roughly by equity (index 0 lowest, matching
# every other ordinal family in this file). Combo Draw and (Nuts) Flush
# Draw are the same underlying signal the old combo_draw/nuts_flush_draw
# booleans used; BDFD/BDSD are new (see has_backdoor_flush_draw/
# has_backdoor_straight_draw in evaluator.py) -- a plain (non-backdoor)
# flush draw was already computed (hand["flush_draw"]) but never exposed
# as its own top-level feature before, only ever folded into
# nuts_flush_draw/combo_draw.
_DRAW_VALUES = (
    (0 / 7, "No Draw"),
    (1 / 7, "Backdoor Straight Draw (BDSD, Flop Only)"),
    (2 / 7, "Backdoor Flush Draw (BDFD, Flop Only)"),
    (3 / 7, "Gutshot (4 Outs)"),
    (4 / 7, "Open Ended Straight Draw (OESD, 8 Outs)"),
    (5 / 7, "Flush Draw (9 Outs)"),
    (6 / 7, "Nuts Flush Draw (9 Outs)"),
    (7 / 7, "Combo Draw (Flush + Straight)"),
)
# Shared by the whole "how many raises" family (num_raises_norm and every
# num_raises_previous_street/preflop/flop/turn sibling below) -- deliberately
# worded generically ("No raises", not "No raises yet this street") since
# the same table backs both the live current-street reading and every
# frozen-once-that-street-ended one, and a person reading e.g. "Raises
# Preflop: 3 or more raises (clipped)" on the river shouldn't have to
# mentally substitute in whichever street that particular feature is
# actually about.
_RAISES_VALUES = (
    (0.0, "No raises"),
    (1 / 3, "1 raise"),
    (2 / 3, "2 raises"),
    (1.0, "3 or more raises (clipped) -- by this point it's mostly a shove-or-not decision"),
)
_STACK_DEPTH_VALUES = (
    (0.0, "Very short stack (~0bb)"),
    (0.25, "Short stack (~50bb)"),
    (0.5, "Standard stack (~100bb)"),
    (0.75, "Deep stack (~150bb)"),
    (1.0, "Very deep stack (~200bb or more, clipped)"),
)
_OPP_VPIP_VALUES = (
    (0.0, "Active opponents never voluntarily play a hand (0% VPIP)"),
    (0.25, "Active opponents voluntarily play about 1 in 4 hands (25% VPIP)"),
    (0.5, "Active opponents voluntarily play about half their hands (50% VPIP), or no reads yet this session"),
    (0.75, "Active opponents voluntarily play about 3 in 4 hands (75% VPIP)"),
    (1.0, "Active opponents voluntarily play (almost) every hand (100% VPIP)"),
)
_OPP_PFR_VALUES = (
    (0.0, "Active opponents never raise preflop (0% PFR)"),
    (0.25, "Active opponents raise preflop about 1 in 4 hands (25% PFR)"),
    (0.5, "Active opponents raise preflop about half their hands (50% PFR), or no reads yet this session"),
    (0.75, "Active opponents raise preflop about 3 in 4 hands (75% PFR)"),
    (1.0, "Active opponents raise preflop (almost) every hand they play (100% PFR)"),
)
_OPP_THREE_BET_VALUES = (
    (0.0, "Active opponents never 3-bet when given the chance (0% 3-bet)"),
    (0.25, "Active opponents 3-bet about 1 in 4 chances (25% 3-bet)"),
    (0.5, "Active opponents 3-bet about half their chances (50% 3-bet), or no reads yet this session"),
    (0.75, "Active opponents 3-bet about 3 in 4 chances (75% 3-bet)"),
    (1.0, "Active opponents 3-bet (almost) every chance they get (100% 3-bet)"),
)
_OPP_FOLD_TO_THREE_BET_VALUES = (
    (0.0, "Active opponents never fold to a 3-bet (0% fold-to-3-bet)"),
    (0.25, "Active opponents fold to a 3-bet about 1 in 4 times (25% fold-to-3-bet)"),
    (0.5, "Active opponents fold to a 3-bet about half the time (50% fold-to-3-bet), or no reads yet this session"),
    (0.75, "Active opponents fold to a 3-bet about 3 in 4 times (75% fold-to-3-bet)"),
    (1.0, "Active opponents (almost) always fold to a 3-bet (100% fold-to-3-bet)"),
)
_OPP_AGGRESSION_FREQ_VALUES = (
    (0.0, "Active opponents never bet/raise postflop when they act (0% aggression, all calls)"),
    (0.25, "Active opponents bet/raise postflop about 1 in 4 times they act (25% aggression)"),
    (0.5, "Active opponents bet/raise postflop about half the time (50% aggression), or no reads yet this session"),
    (0.75, "Active opponents bet/raise postflop about 3 in 4 times (75% aggression)"),
    (1.0, "Active opponents (almost) always bet/raise rather than call postflop (100% aggression)"),
)
_OPP_FOLD_VS_BET_VALUES = (
    (0.0, "Active opponents never fold to a postflop bet (0% fold, very sticky)"),
    (0.25, "Active opponents fold to a postflop bet about 1 in 4 times (25%)"),
    (0.5, "Active opponents fold to a postflop bet about half the time (50%), or no reads yet this session"),
    (0.75, "Active opponents fold to a postflop bet about 3 in 4 times (75%)"),
    (1.0, "Active opponents (almost) always fold to a postflop bet (100%)"),
)
_SEAT_ROLE_LABELS = {
    "UTG": "Under The Gun (UTG)", "HJ": "Hijack (HJ)", "CO": "Cutoff (CO)",
    "BTN": "Button (BTN)", "SB": "Small Blind (SB)", "BB": "Big Blind (BB)",
}
_SEAT_ROLE_VALUES = tuple((i / (len(SEAT_ROLES) - 1), _SEAT_ROLE_LABELS[role]) for i, role in enumerate(SEAT_ROLES))

# Flop-texture features describe the flop (board[:3]) and, for the
# connectivity family below, how this player's hole cards personally
# connect to it -- still frozen once the flop is dealt, since neither the
# flop cards nor the hole cards change on the turn/river. All default to
# their lowest-value category before the flop (0 for every child, since
# nothing has happened yet). flop_suit_texture_norm itself is purely about
# the flop's own 3 cards (no hole-card awareness at all, unlike the
# connectivity family) -- see suit_connection_index below for the
# hole-card-aware, non-frozen version of "how connected am I to these suits."
_FLOP_SUIT_TEXTURE_VALUES = ((0.0, "Rainbow"), (0.5, "Flush Draw Flop"), (1.0, "Monotone"))
_FLOP_PAIRING_VALUES = ((0.0, "Unpaired"), (0.5, "Paired"), (1.0, "Tripled"))
_FLOP_CONNECTIVITY_VALUES = (
    (0 / 2, "Disconnected"), (1 / 2, "Connected, No Straight Draw"),
    (2 / 2, "Connected, Straight Draw (4+/8+ Outs)"),
)
_FLOP_WETNESS_VALUES = ((0.0, "Dry"), (1.0, "Wet"))
_FLOP_DYNAMISM_VALUES = ((0.0, "Static"), (1.0, "Dynamic"))

# Unlike the flop-texture family above, suit_connection_index is never frozen
# -- it's recomputed from hole + the *current* board every time
# extract_features runs, so it keeps updating on the turn and river as more
# board cards (and so, potentially, more of this player's suit) show up.
# Pigeonhole guarantees at least 2 once the flop is out (5 cards split across
# 4 suits), rising to the 5-card cap as a flush comes together; preflop
# (2 hole cards only) it's just whether they're suited -- 1 if not, 2 if so.
_SUIT_CONNECTION_VALUES = (
    (1 / 5, "1 Card (unsuited hole cards, no board yet)"), (2 / 5, "2 Cards (suited, or the flop's pigeonhole floor)"),
    (3 / 5, "3 Cards"), (4 / 5, "4 Cards (a flush draw)"), (5 / 5, "5+ Cards (a flush made)"),
)

# "connected_flop" is a family-level aggregate (indices 1+2 of the 3-value
# family below), so it's declared standalone rather than as a `linked_to`
# child (which can only ever point at one single index).
_FLOP_CONNECTED_FAMILY_SPEC = FeatureSpec(
    "connected_flop", "Connected",
    "1 if the flop's ranks (Ace counted as high or low, whichever is closer) span 4 "
    "or less, regardless of whether this player actually has a straight draw there "
    "(see flop_connectivity_norm for the hole-card-aware breakdown), else 0.",
    group="Board / Flop Characteristics",
)

# Grouped as parent-feature-then-its-children for readability; array order is
# otherwise arbitrary since extract_features looks values up by key, not position.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "hand_category_norm", "Hand Strength Tier",
        "Made-hand category of the best hand available from hole cards plus the current "
        "board, normalized to 0-1 as bucket_index / 26. Standard High Card < Pair < Two "
        "Pair < Three of a Kind < Straight < Flush < Full House < Four of a Kind < "
        "Straight Flush ordering, but High Card/Pair/Three of a Kind/Straight/Flush are "
        "each split into several ordered sub-buckets instead of one flat value -- e.g. "
        "Pair spans Underpair, Ninja Pair, Bottom Pair, Pair, Third Pair, Second Pair, "
        "Top Pair, Top Pair + Good Kicker, Top Pair + Top Kicker, and Overpair, weakest "
        "to strongest, and High Card spans High Card, King High, and Ace High -- folding "
        "in what used to be a dozen-plus separate standalone booleans (top_pair, overpair, "
        "ace_high_no_pair, ...) directly into this ordinal scale. See "
        "_HAND_CATEGORY_VALUES for every bucket and _hand_category_bucket for exactly "
        "how each one is computed.",
        kind="categorical", value_table=_HAND_CATEGORY_VALUES, group="Made Hand Features",
    ),

    FeatureSpec(
        "hole_high_card_norm", "Hole High Card Rank",
        "Rank of the higher of this player's two hole cards, normalized to 0-1 via "
        "(rank - 2) / 12, so 0.0 = deuce and 1.0 = ace. Always defined, from preflop on.",
        kind="categorical", value_table=_HOLE_HIGH_CARD_VALUES, group="Hole Card Characteristics",
    ),

    FeatureSpec(
        "shared_high_card_norm", "Shared Cards High Card Rank",
        "Rank of the highest board (shared/community) card, normalized to 0-1 via "
        "(rank - 2) / 12. Defaults to 0.0 (the deuce value) preflop, before any shared "
        "cards are dealt -- pair with the street/is_preflop features to tell 'no board "
        "yet' apart from 'the board's high card really is a deuce'.",
        kind="categorical", value_table=_SHARED_HIGH_CARD_VALUES, group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "hole_connectivity", "Hole Card Connectivity",
        "How close together the two hole card ranks are for straight-making purposes: "
        "1 - gap / 6, where gap is the rank distance (Ace counted as high or low, "
        "whichever gives the smaller gap -- so A-K and A-2 are both gap 1) capped at 6, "
        "since anything 6+ apart is equally unable to share a straight. 1.0 means "
        "consecutive ranks (e.g. 9-T); 0.0 means 6 or more apart.",
        kind="categorical", value_table=_CONNECTIVITY_VALUES, group="Hole Card Characteristics",
    ),

    FeatureSpec(
        "hole_hand_category_norm", "Hole Hand Category",
        "Which of 12 mutually exclusive starting-hand buckets the two hole cards fall "
        "into, checked strongest-to-weakest so exactly one applies -- a pair always beats "
        "a suited Ace to its bucket, a suited Ace always beats Kxs, and so on. Priority "
        "order (strongest first): Premium Pairs (AA/KK/QQ) > High Pairs (JJ-99) > Average "
        "Pairs (22-88) > Axs (Ace + suited kicker) > Axo (Ace + offsuit kicker) > Kxs "
        "(King + suited kicker, no Ace) > Qxs (Queen + suited kicker, no Ace/King) > "
        "Suited Connectors (1 apart) > Suited 1 Gappers (2 apart) > Suited 2 Gappers "
        "(3 apart) > Unsuited Connectors (1 apart) > Junk (everything else). Normalized "
        "as bucket_index / 11, where bucket_index counts up from 0 = Junk to 11 = "
        "Premium Pairs, so higher is a stronger starting hand.",
        kind="categorical", value_table=_HOLE_CATEGORY_VALUES, group="Hole Card Characteristics",
    ),

    FeatureSpec(
        "hole_hand_grid_x_norm", "Exact Hole Hand",
        "Column position of this player's exact starting hand (both hole card ranks, "
        "plus suited/offsuit) in the classic 13x13 preflop starting-hand grid, "
        "normalized as rank_index / 12 (0.0 = Ace, 1.0 = deuce). Paired with "
        "hole_hand_grid_y_norm (the row position -- linked here purely so the two "
        "collapse into one entry everywhere features are listed, the same way a "
        "one-hot indicator collapses into its parent elsewhere in this catalog; it "
        "isn't a value-indicator of hole_hand_grid_x_norm the way a real linked child "
        "is), the (row, column) cell puts pocket pairs on the diagonal, suited combos "
        "above it, and offsuit combos below it -- exactly where a human would find "
        "this hand on a range chart (see features.hole_hand_grid_label). Realistic to "
        "memorize a chart against all 169 exact starting hands the way real players "
        "do preflop, not against the far larger space of postflop "
        "hole-cards-vs-board combinations, so this is masked to -1.0 "
        "(features.MASKED) outside preflop.",
        kind="categorical", value_table=_HOLE_HAND_GRID_VALUES, group="Hole Card Characteristics",
        maskable=True,
    ),
    FeatureSpec(
        "hole_hand_grid_y_norm", "Exact Hole Hand (Row)",
        "Row position of this player's exact starting hand in the same 13x13 grid -- "
        "see hole_hand_grid_x_norm for the full explanation, layout, and preflop-only "
        "masking.",
        kind="categorical", value_table=_HOLE_HAND_GRID_VALUES, group="Hole Card Characteristics",
        linked_to="hole_hand_grid_x_norm", maskable=True,
    ),

    FeatureSpec(
        "street_norm", "Betting Street",
        "Current street normalized to 0-1: 0.0 = preflop, 0.33 = flop, 0.67 = turn, 1.0 = river.",
        kind="categorical", value_table=_STREET_VALUES, group="Table & Game State Features",
    ),

    FeatureSpec(
        "call_amount_norm", "Call Size Vs Pot",
        "Amount required to call, as a fraction of the current pot (call_amount / pot), "
        "divided by 1.25 and clipped to 0-1 -- so a call up to a full pot-sized bet maps "
        "onto the first 5 evenly-spaced points (0, 1/4, 1/2, 3/4, full pot) and anything "
        "noticeably bigger than that (an overbet) clips into its own top bucket instead "
        "of being lumped in with an exactly-pot-sized bet.",
        kind="continuous", value_table=_CALL_SIZE_VALUES, group="Betting Behaviour Features",
        zero_bucket_is_exact=True,
    ),

    FeatureSpec(
        "spr_norm", "Stack-To-Pot Ratio",
        "Effective stack (the smaller of this player's stack and the largest "
        "active opponent's stack) divided by the current pot, indicating how many "
        "pot-sized bets remain; divided by 20 and clipped to 0-1.",
        kind="continuous", value_table=_SPR_VALUES, group="Stack & Pot Features",
    ),

    FeatureSpec(
        "position_norm", "In/Out of Position",
        "How late this player acts in the current street's action order, from "
        "0.0 (acts first) to 1.0 (acts last) -- relative only to players who "
        "haven't folded yet, not this player's fixed seat at the table (see "
        "Starting Seat Position for that): a player who was dealt into a late "
        "seat but finds everyone ahead of them has folded reads as acting "
        "early here, since that's what actually determines their own "
        "information/leverage at the moment of this decision.",
        kind="continuous", value_table=_POSITION_VALUES, group="Table & Game State Features",
    ),

    FeatureSpec(
        "starting_position_norm", "Starting Seat Position",
        "Which of the 6 standard preflop seat roles this player started the hand in, "
        "ordered from first-to-act preflop through the blinds -- UTG, HJ, CO, BTN, SB, BB "
        "-- normalized as role_index / 5. At smaller table sizes the earliest non-blind "
        "roles collapse (e.g. 4-handed there's only UTG and BTN; heads-up the button and "
        "small blind are the same seat, labeled SB).",
        kind="categorical", value_table=_SEAT_ROLE_VALUES, group="Table & Game State Features",
    ),

    FeatureSpec(
        "num_active_norm", "Players Still In Hand",
        "Number of players who have not folded yet this hand, divided by 6.",
        kind="categorical", value_table=_ACTIVE_PLAYERS_VALUES, group="Table & Game State Features",
    ),

    FeatureSpec(
        "num_raises_norm", "Raises This Street",
        "Number of bets/raises made so far on the current street, divided by 3 and clipped "
        "to 0-1 -- capped at 3 rather than higher, since by the time a street has seen 3 "
        "raises the remaining decision is essentially just shove-or-not regardless of "
        "exactly how many more re-raises it's technically been. Live -- resets to 0 at the "
        "start of every street, unlike the frozen, one-street-late Raises Last Street below "
        "(or Raises Preflop/Flop/Turn, each pinned to one specific calendar street -- see "
        "their own descriptions).",
        kind="categorical", value_table=_RAISES_VALUES, group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "raises_last_street_norm", "Raises Last Street",
        "Raises This Street's own reading, carried forward one street late: how many "
        "bets/raises the *previous* street ended with (0 preflop, since there's no previous "
        "street), normalized the same way. Frozen for the rest of the current street once "
        "read -- it only advances again when the next street starts. Pairs with Last "
        "Aggressor - Previous Street below the same way Raises This Street pairs with "
        "wanting to know *who* made those raises.",
        kind="categorical", value_table=_RAISES_VALUES, group="Betting Behaviour Features",
    ),

    FeatureSpec(
        "raises_preflop_norm", "Raises Preflop",
        "How many raises preflop ended with, normalized the same way as Raises This Street -- "
        "but pinned to preflop specifically rather than whichever street is current: 0 until "
        "preflop has actually finished (never the live in-progress preflop count -- see "
        "Raises This Street for that), then frozen at that final count for the rest of the "
        "hand, however many streets later a decision happens to be. The generalized, "
        "every-street version of what used to be a preflop-only 'Pot Type' feature.",
        kind="categorical", value_table=_RAISES_VALUES, group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "raises_flop_norm", "Raises Flop",
        "Raises Preflop's own explanation, one street later: 0 until the flop has finished, "
        "then frozen at its final raise count for the rest of the hand (so only ever "
        "meaningful from the turn onward).",
        kind="categorical", value_table=_RAISES_VALUES, group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "raises_turn_norm", "Raises Turn",
        "Raises Preflop's own explanation, two streets later: 0 until the turn has finished, "
        "then frozen at its final raise count (so only ever meaningful on the river -- "
        "there's no Raises River, since nothing left in the hand would ever get to see it).",
        kind="categorical", value_table=_RAISES_VALUES, group="Betting Behaviour Features",
    ),

    FeatureSpec(
        "stack_depth_norm", "Stack Depth",
        "This player's starting stack for the hand (fixed for its whole duration -- not "
        "the player's current, mid-hand stack, see Stack-To-Pot Ratio for that), in big "
        "blinds, divided by 200 and clipped to 0-1 (1.0 = 200bb or more).",
        kind="continuous", value_table=_STACK_DEPTH_VALUES, group="Stack & Pot Features",
    ),

    # Standalone booleans: not tied to a specific value of any other feature.
    FeatureSpec(
        "hole_suited", "Suited Hole Cards", "1 if both hole cards share a suit, else 0.",
        group="Hole Card Characteristics",
    ),
    FeatureSpec(
        "is_aggressor_previous_street", "Last Aggressor - Previous Street",
        "1 if this player made the most recent bet/raise on the *previous* street, else 0 "
        "(always 0 preflop, since there's no previous street). Not the current street: a "
        "player who just raised is skipped over until either the street ends or someone "
        "else re-raises, at which point they're no longer that street's own aggressor -- "
        "so at the moment any decision is actually made, 'I raised this street' is always "
        "false, making the previous street the only *relative* version of this that's ever "
        "useful (see Last Aggressor Preflop/Flop/Turn below for the *specific-calendar-"
        "street* versions, e.g. 'did I open this pot', that stay meaningful for the rest of "
        "the hand rather than just one street).",
        group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "is_aggressor_preflop", "Last Aggressor Preflop",
        "1 if this player made the last bet/raise preflop, else 0 -- 0 until preflop has "
        "actually finished, then frozen at that reading for the rest of the hand. Unlike "
        "Last Aggressor - Previous Street, this doesn't reset as later streets pass: it keeps "
        "answering 'did I open this pot' all the way to the river.",
        group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "is_aggressor_flop", "Last Aggressor Flop",
        "Last Aggressor Preflop's own explanation, one street later: 0 until the flop has "
        "finished, then frozen at whether this player made the flop's last raise (so only "
        "ever meaningful from the turn onward).",
        group="Betting Behaviour Features",
    ),
    FeatureSpec(
        "is_aggressor_turn", "Last Aggressor Turn",
        "Last Aggressor Preflop's own explanation, two streets later: 0 until the turn has "
        "finished, then frozen at whether this player made the turn's last raise (so only "
        "ever meaningful on the river -- there's no Last Aggressor River, since nothing left "
        "in the hand would ever get to see it).",
        group="Betting Behaviour Features",
    ),

    FeatureSpec(
        "flop_suit_texture_norm", "Flop Suit Texture",
        "How suit-coordinated the flop is, purely from its own 3 cards -- Rainbow "
        "(all 3 different suits) < Flush Draw Flop (exactly 2 share a suit) < "
        "Monotone (all 3 share a suit), normalized as family_index / 2. Doesn't "
        "reflect this player's own hole cards at all -- see suit_connection_index "
        "for how many cards of a single suit this player personally has going "
        "(hole + board combined, and not frozen -- it keeps updating on the turn "
        "and river). Frozen once the flop is dealt (unaffected by the turn/river, "
        "since the flop's own cards don't change); defaults to 0.0 (Rainbow's "
        "value) before the flop.",
        kind="categorical", value_table=_FLOP_SUIT_TEXTURE_VALUES, group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "suit_connection_index", "Suit Connection",
        "The most cards of any single suit among this player's hole cards plus "
        "the current board (however many board cards are out so far), capped at "
        "5 and normalized as count / 5. Unlike flop_suit_texture_norm above, this "
        "is never frozen -- it's recomputed from the *current* board every time, "
        "so it keeps updating on the turn as more of this player's suit "
        "potentially shows up. Preflop (2 hole cards only) it's just whether "
        "they're suited: 1 card if not, 2 if so. Once the flop is out, pigeonhole "
        "guarantees at least 2 (5 cards split across 4 suits), rising toward the "
        "5-card cap as a flush comes together. Masked to -1.0 (features.MASKED) "
        "on the river: by then a made flush is already fully captured by "
        "hand_category_norm, and 'how many cards of one suit' stops being a "
        "*draw* signal once there's no next card left to complete one.",
        kind="categorical", value_table=_SUIT_CONNECTION_VALUES, group="Board / Flop Characteristics",
        maskable=True,
    ),

    FeatureSpec(
        "flop_pairing_texture_norm", "Flop Pairing Texture",
        "How rank-coordinated the flop is: Unpaired (3 different ranks) < Paired (2 "
        "share a rank) < Tripled (all 3 share a rank), normalized as index / 2. Frozen "
        "once the flop is dealt; defaults to 0.0 (Unpaired's value) before the flop.",
        kind="categorical", value_table=_FLOP_PAIRING_VALUES, group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "flop_connectivity_norm", "Flop Connectivity",
        "Whether the flop's 3 ranks (Ace counted as high or low, whichever is closer) "
        "are close enough together to make straights possible, and if so, whether this "
        "player's hole cards actually give them a straight draw there: Disconnected if "
        "the ranks span more than 4 (too spread out for any hole cards to complete a "
        "straight); otherwise Connected, split into 'No Straight Draw' (this player's "
        "hole+flop cards have no rank that would complete a straight) and 'Straight "
        "Draw (4+/8+ Outs)' (a gutshot or open-ended/double-gutshot is live). "
        "Normalized as bucket_index / 2. Frozen once the flop is dealt; defaults to "
        "0.0 (Disconnected's value) before the flop.",
        kind="categorical", value_table=_FLOP_CONNECTIVITY_VALUES, group="Board / Flop Characteristics",
    ),
    _FLOP_CONNECTED_FAMILY_SPEC,

    FeatureSpec(
        "oesd_possible_flop", "Open Ended Straight Draw Possible Flop",
        "1 if the flop's 3 ranks are all different and span 3 or less (Ace counted as "
        "high or low, whichever is closer) -- tight enough that an open-ended straight "
        "draw is possible for some hole cards -- else 0. A stricter subset of Connected "
        "Flop (not one of its Connected/Disconnected partition values), so it isn't "
        "folded into that table. 0 before the flop.",
        group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "flop_wetness_norm", "Flop Wet vs Dry",
        "How draw-heavy the flop is, purely from its own suit and rank coordination "
        "(not this player's hole cards -- see suit_connection_index and "
        "flop_connectivity_norm for the hole-card-aware versions), combined into one "
        "0-4 score: suit family index (0 Rainbow / 1 Flush Draw Flop / 2 Monotone) + 1 "
        "if the flop's ranks are Connected (span <=4) + 1 if an open-ended straight "
        "draw is possible for some hole cards (see oesd_possible_flop) -- then "
        "thresholded at >=2: Wet (1.0) at or above the threshold, Dry (0.0) below it. "
        "A monotone flop is always Wet on suits alone; a rainbow, disconnected flop is "
        "always Dry. Frozen once the flop is dealt; defaults to 0.0 (Dry's value) "
        "before the flop.",
        kind="categorical", value_table=_FLOP_WETNESS_VALUES, group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "flop_dynamism_norm", "Flop Static vs Dynamic",
        "Whether the flop's made-hand hierarchy is likely to get shuffled by the turn "
        "and river: Dynamic (1.0) if the flop is unpaired and its wetness score (see "
        "flop_wetness_norm) plus a high-card adjustment is >=2, else Static (0.0). The "
        "adjustment is -1 when the flop's highest card is a Queen, King, or Ace (fewer "
        "hole cards can credibly out-flop the pair/overpair it hands out, so fewer "
        "runouts change who's ahead), +1 when the flop's highest card is 8 or below "
        "(more hole-card combos have live equity against a low top card), and 0 "
        "otherwise. A paired or tripled flop is always treated as Static regardless of "
        "wetness or card height, since the hierarchy it sets up (trips/boat beats two "
        "pair beats top pair) rarely gets overturned by later streets. Frozen once the "
        "flop is dealt; defaults to 0.0 (Static's value) before the flop.",
        kind="categorical", value_table=_FLOP_DYNAMISM_VALUES, group="Board / Flop Characteristics",
    ),

    FeatureSpec(
        "num_overcards_norm", "Board Overcards",
        "Number of board (community/shared) cards that rank higher than the higher of "
        "this player's two hole cards, normalized to 0-1 via count / 5 (the maximum "
        "possible number of shared cards). 0.0 preflop, since there's no board yet. "
        "Distinct from Shared Cards High Card Rank, which reports the board's single "
        "highest rank regardless of this player's own hole cards.",
        kind="categorical", value_table=_OVERCARDS_VALUES, group="Made Hand Features",
    ),

    # Hand-vs-board heuristics: not mutually exclusive as a group, so unlike
    # the categorical features above these aren't organized as a one-hot
    # family with a linear parent. (Pair-strength heuristics like Top Pair/
    # Overpair, and the Ace High/King High no-made-hand heuristics, used to
    # live here too, but are now sub-buckets of hand_category_norm instead
    # -- see _HAND_CATEGORY_VALUES.)
    FeatureSpec(
        "draw_norm", "Draw",
        "This player's single strongest live draw, ordered roughly by equity: Combo "
        "Draw (a flush draw and a straight draw at once) > Nuts Flush Draw > Flush "
        "Draw (9 outs either way) > Open Ended Straight Draw (8 outs) > Gutshot (4 "
        "outs) > Backdoor Flush Draw > Backdoor Straight Draw (BDFD/BDSD: flop-only, "
        "needing both the turn and river) > No Draw. Where a hand qualifies for more "
        "than one of these at once, the highest-equity one wins -- see _DRAW_VALUES "
        "for the exact order -- collapsing what used to be several separate draw "
        "signals into one feature without losing the information a player actually "
        "cares about (the strongest draw in play). Always No Draw preflop; masked to "
        "-1.0 (features.MASKED) on the river, once no next card is left to complete "
        "or miss any draw.",
        kind="categorical", value_table=_DRAW_VALUES, group="Draw Features",
        maskable=True,
    ),

    # Opponent-tendency features: observed HUD-style stats accumulated over
    # the current session's hands (see opponent_model.py), not derivable
    # from this hand's cards/board alone -- this is what lets a genome
    # condition on *how opponents have actually played* rather than only on
    # the current situation, the missing ingredient for active exploitation.
    # All default to a neutral 0.5 ("assume average") when nobody's been
    # observed yet, and shrink toward that neutral value on small samples.
    FeatureSpec(
        "opp_vpip_norm", "Opponent VPIP (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of voluntarily putting chips in preflop (calling or raising rather than folding "
        "or checking for free) over hands seen this session. Shrunk toward a neutral 0.5 "
        "on small samples.",
        kind="continuous", value_table=_OPP_VPIP_VALUES, group="Opponent Tendency Features",
    ),

    FeatureSpec(
        "opp_pfr_norm", "Opponent PFR (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of raising preflop (rather than just calling or folding) over hands seen this "
        "session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_PFR_VALUES, group="Opponent Tendency Features",
    ),

    FeatureSpec(
        "opp_three_bet_norm", "Opponent 3-Bet % (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of re-raising when facing exactly one preflop raise, among their chances to do "
        "so this session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_THREE_BET_VALUES, group="Opponent Tendency Features",
    ),

    FeatureSpec(
        "opp_fold_to_three_bet_norm", "Opponent Fold to 3-Bet % (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of folding when facing a preflop 3-bet, among their chances to do so this "
        "session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_FOLD_TO_THREE_BET_VALUES, group="Opponent Tendency Features",
    ),

    FeatureSpec(
        "opp_aggression_freq_norm", "Opponent Postflop Aggression Frequency (Table Average)",
        "Average, across every opponent still active in the hand, of their observed "
        "postflop bet/raise rate -- (bets+raises) / (bets+raises+calls) -- over postflop "
        "actions taken this session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_AGGRESSION_FREQ_VALUES, group="Opponent Tendency Features",
    ),

    FeatureSpec(
        "opp_fold_vs_bet_norm", "Opponent Fold vs Bet, Postflop (Table Average)",
        "Average, across every opponent still active in the hand, of their observed rate "
        "of folding when facing a postflop bet, among their chances to do so this "
        "session. Shrunk toward a neutral 0.5 on small samples.",
        kind="continuous", value_table=_OPP_FOLD_VS_BET_VALUES, group="Opponent Tendency Features",
    ),
]

FEATURE_NAMES = [spec.key for spec in FEATURE_SPECS]
NUM_FEATURES = len(FEATURE_SPECS)

_SPECS_BY_KEY = {spec.key: spec for spec in FEATURE_SPECS}


def group_of(spec: FeatureSpec) -> str:
    """This spec's report-section group. Top-level specs (value-table
    parents and standalone booleans) set `group` directly; linked children
    inherit their parent's group instead of repeating it."""
    if spec.group:
        return spec.group
    if spec.linked_to:
        return group_of(_SPECS_BY_KEY[spec.linked_to])
    return "Other"


@dataclass
class Situation:
    """Everything a player's genome may condition its decision on."""

    hole: list[Card]
    board: list[Card]
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river
    pot: float
    call_amount: float  # chips needed to call (0 if checking is an option)
    my_stack: float
    effective_stack: float  # min(my_stack, largest active opponent stack)
    position: int  # 0 = first to act this street ... n-1 = last to act
    num_seats_this_street: int
    seat_index: int  # this player's fixed seat this hand
    button_idx: int  # the button's fixed seat this hand
    num_seats_total: int  # seats dealt into this hand (constant all hand, unlike num_active)
    num_active: int  # players still in the hand (not folded)
    num_raises_this_street: int
    # num_raises_previous_street/num_raises_preflop/num_raises_flop/
    # num_raises_turn are all "frozen once that specific street ends" reads,
    # unlike num_raises_this_street's own live, current-street count: each
    # is 0 until its own street has actually finished, then keeps that final
    # count for every later street (including a skipped one -- see
    # game.betting_round's own docstring). num_raises_previous_street is
    # *relative* to whatever street is current (the street just before it,
    # whichever that is); the other three are each pinned to one specific
    # calendar street regardless of which street is current, so e.g.
    # num_raises_preflop stays meaningful all the way to the river, not just
    # on the flop. There's no num_raises_river: nothing left in the hand
    # would ever get to observe it.
    num_raises_previous_street: int
    num_raises_preflop: int
    num_raises_flop: int
    num_raises_turn: int
    # is_aggressor_previous_street/_preflop/_flop/_turn mirror the
    # num_raises_* family above exactly, one street late (frozen the same
    # way, same "previous vs one specific calendar street" split) -- did I
    # make the last bet/raise on that (now-finished) street? Deliberately
    # never "this [current] street": whoever's still deciding on the
    # current street can never themselves be its own last aggressor --
    # raising takes you out of the to-act order until either the street
    # ends or someone else re-raises (see game.betting_round), and a
    # re-raise immediately replaces you as the current street's aggressor --
    # so that reading would be a structurally-always-False feature. Every
    # street that's actually finished has no such conflict, so each is a
    # real, useful read (e.g. is_aggressor_previous_street for "am I
    # continuation betting"; is_aggressor_preflop for "did I open this pot,
    # no matter how many streets ago").
    is_aggressor_previous_street: bool
    is_aggressor_preflop: bool
    is_aggressor_flop: bool
    is_aggressor_turn: bool
    starting_stack: float
    big_blind: float = 2.0  # lets stack depth be expressed in actual BB
    # Starting positions (seating.SEAT_ROLES) of every *other* seat that has
    # bet/raised this street so far, e.g. frozenset({"UTG"}) for "folds to me
    # in the BB after UTG opens".
    raised_positions: frozenset[str] = field(default_factory=frozenset)

    # Opponent-tendency reads (see opponent_model.py), already 0-1 rates --
    # all default to a neutral 0.5 so any caller that doesn't opt into
    # opponent modeling still gets a valid Situation.
    opp_vpip: float = 0.5
    opp_pfr: float = 0.5
    opp_three_bet: float = 0.5
    opp_fold_to_three_bet: float = 0.5
    opp_aggression_freq: float = 0.5
    opp_fold_vs_bet: float = 0.5


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hand_vs_board_heuristics(hole: list[Card], board: list[Card], hand: dict, street: int) -> dict:
    """draw_norm: this player's single strongest live draw (see its own
    FeatureSpec for the full precedence order and why it collapses what
    used to be several separate draw signals into one). Doesn't fit
    hand_category_norm's ordinal scale (pair/set/straight/flush strength
    relative to the board, and now also no-made-hand high-card strength,
    are folded in there instead -- see _hand_category_bucket). Masked to
    MASKED on the river (street == 3): a draw is by definition unresolved
    (neither made nor whiffed yet), which is no longer a meaningful state
    once there's no next card left to complete it."""
    if street == 3:
        return {"draw_norm": MASKED}

    flush_draw = hand["flush_draw"]
    draw_suit = hand["flush_draw_suit"]
    nuts_flush_draw = flush_draw and any(c.suit == draw_suit and c.rank == 14 for c in hole)
    outs = hand["straight_draw_outs"]
    combo_draw = flush_draw and outs >= 1

    # Highest-equity match wins -- see _DRAW_VALUES for the exact order.
    # Backdoor draws (BDFD/BDSD) only mean anything on the flop specifically
    # (street == 1): by the turn there's only one card left to come, so
    # "needs 2 more running cards" no longer describes a real 2-streets-left
    # draw, even though has_backdoor_flush_draw/has_backdoor_straight_draw's
    # own checks could occasionally still coincidentally hold later.
    if combo_draw:
        bucket = 7
    elif nuts_flush_draw:
        bucket = 6
    elif flush_draw:
        bucket = 5
    elif outs >= 2:
        bucket = 4
    elif outs == 1:
        bucket = 3
    elif street == 1 and has_backdoor_flush_draw(hole + board):
        bucket = 2
    elif street == 1 and has_backdoor_straight_draw(hole + board):
        bucket = 1
    else:
        bucket = 0

    return {"draw_norm": bucket / 7.0}


# Base index (within hand_category_norm's 27-value table -- see
# _HAND_CATEGORY_VALUES) where each made-hand category's sub-buckets begin.
_HIGH_CARD_BASE_INDEX = 0
_PAIR_BASE_INDEX = 3
_TWO_PAIR_INDEX = 13
_SET_BASE_INDEX = 14
_STRAIGHT_BASE_INDEX = 17
_FLUSH_BASE_INDEX = 21
_FULL_HOUSE_INDEX = 24
_QUADS_INDEX = 25
_STRAIGHT_FLUSH_INDEX = 26

def _high_card_bucket_offset(hand: dict) -> int:
    """0-2 offset within hand_category_norm's High Card range (see
    _HAND_CATEGORY_VALUES, added to _HIGH_CARD_BASE_INDEX by the caller) --
    assumes cat == HIGH_CARD. Folds what used to be the standalone
    ace_high_no_pair/king_high_no_pair booleans directly into this scale."""
    overall_high_card = hand["high_card"]
    if overall_high_card == 14:
        return 2  # Ace High
    if overall_high_card == 13:
        return 1  # King High
    return 0  # High Card (anything below a King)


# A "good" (but not the very best -- see the Top Kicker bucket) Top Pair
# kicker: Jack, Queen, or King. Ace is handled separately (Top Kicker).
_GOOD_KICKER_RANKS = frozenset({11, 12, 13})


def _pair_bucket_offset(hole: list[Card], board: list[Card]) -> int:
    """0-9 offset within hand_category_norm's Pair range (see
    _HAND_CATEGORY_VALUES, added to _PAIR_BASE_INDEX by the caller) --
    assumes cat == PAIR."""
    hole_ranks = (hole[0].rank, hole[1].rank)
    is_pocket_pair = hole_ranks[0] == hole_ranks[1]
    board_ranks = [c.rank for c in board]

    if is_pocket_pair:
        if not board_ranks:
            return 3  # preflop pocket pair -- no board yet to compare against
        if hole_ranks[0] > max(board_ranks):
            return 9  # Overpair
        if hole_ranks[0] < min(board_ranks):
            return 0  # Underpair (strictly below every board card)
        return 1  # Ninja Pair (below the top card, but beats at least one board card)

    # Not a pocket pair. Usually this pair is exactly one hole card
    # rank-matching one board rank -- but the board can also already be
    # paired on its own (e.g. board 5-5-K, hole A-2), in which case neither
    # hole card matches and the pair is entirely the board's; that's not a
    # top/second/third/bottom pair relative to either hole card, so it
    # falls through to the generic Pair bucket below.
    distinct_board_desc = sorted(set(board_ranks), reverse=True)
    matches = [r for r in hole_ranks if r in board_ranks]
    if not matches:
        return 3  # board itself is paired; neither hole card contributes
    matched_rank = matches[0]
    if matched_rank == distinct_board_desc[0]:
        kicker = hole_ranks[1] if hole_ranks[0] == matched_rank else hole_ranks[0]
        if kicker == 14:
            return 8  # Top Pair + Top Kicker
        if kicker in _GOOD_KICKER_RANKS:
            return 7  # Top Pair + Good Kicker
        return 6  # Top Pair
    if len(distinct_board_desc) >= 2 and matched_rank == distinct_board_desc[1]:
        return 5  # Second Pair
    if len(distinct_board_desc) >= 3 and matched_rank == distinct_board_desc[2]:
        return 4  # Third Pair
    # Bottom Pair -- the board's own lowest distinct rank -- only reachable
    # (and only distinct from Third Pair) once the board has 4+ distinct
    # ranks; on a 3-distinct-rank board the lowest rank *is* the 3rd
    # highest, already returned above as Third Pair.
    if len(distinct_board_desc) >= 4 and matched_rank == distinct_board_desc[-1]:
        return 2  # Bottom Pair
    return 3  # generic Pair (a rank that's none of top/second/third/bottom -- only
    # possible as the 4th-highest of exactly 5 distinct board ranks)


def _set_bucket_offset(hole: list[Card], board: list[Card]) -> int:
    """0-2 offset within hand_category_norm's Three of a Kind range (see
    _HAND_CATEGORY_VALUES) -- assumes cat == TRIPS."""
    hole_ranks = (hole[0].rank, hole[1].rank)
    if hole_ranks[0] != hole_ranks[1]:
        return 1  # classic "trips" via a paired board -- treated as the plain middle bucket
    distinct_board_desc = sorted({c.rank for c in board}, reverse=True)
    tripped_rank = hole_ranks[0]
    if tripped_rank == distinct_board_desc[0]:
        return 2  # Top Set
    if tripped_rank == distinct_board_desc[-1]:
        return 0  # Bottom Set
    return 1  # Set (middle)


def _achievable_straight_highs(board: list[Card]) -> set[int]:
    """Every straight_high value obtainable by adding up to 2 arbitrary
    ranks to the board's own ranks -- i.e. every straight some hypothetical
    2 hole cards could complete on this board, not just the ones this
    specific player holds. Used to classify a made straight as the
    Bottom/Top one available here (see hand_category_norm's Straight
    sub-buckets). O(13*13) straight checks -- only run when this player's
    own hand is actually a straight, not on every decision, so the cost
    doesn't matter the way it would in extract_features' main path."""
    board_ranks = {c.rank for c in board}
    highs = set()
    for r1 in range(2, 15):
        for r2 in range(2, 15):
            sh = straight_high(sorted(board_ranks | {r1, r2}, reverse=True))
            if sh is not None:
                highs.add(sh)
    return highs


def _board_flush_possible(board: list[Card]) -> bool:
    """True if the board alone already has 3+ cards of one suit -- enough
    that some opponent's 2 hole cards could complete a flush, independent
    of what this player holds. Used for the Nuts Straight bucket: a
    possible flush means even the best straight here isn't provably the
    best hand."""
    if not board:
        return False
    return max(Counter(c.suit for c in board).values()) >= 3


def _straight_bucket_offset(board: list[Card], straight_high_value: int) -> int:
    """0-3 offset within hand_category_norm's Straight range (see
    _HAND_CATEGORY_VALUES) -- assumes cat == STRAIGHT. `straight_high_value`
    must be a member of _achievable_straight_highs(board) -- guaranteed
    since this player's own hole cards are themselves one way to reach it."""
    achievable = _achievable_straight_highs(board)
    if straight_high_value == max(achievable):
        return 3 if not _board_flush_possible(board) else 2  # Nuts vs Top
    if straight_high_value == min(achievable):
        return 0  # Bottom Straight
    return 1  # Straight (middle)


def _flush_bucket_offset(hand: dict) -> int:
    """0-2 offset within hand_category_norm's Flush range (see
    _HAND_CATEGORY_VALUES) -- assumes cat == FLUSH. hand["tiebreak"][0] is
    the flush's own high card, not hand["high_card"] (the highest card
    among *all* hole+board cards regardless of suit) -- the two differ
    whenever the overall highest card isn't of the flush suit."""
    flush_high_card = hand["tiebreak"][0]
    if flush_high_card == 14:
        return 2  # Ace High Flush
    if flush_high_card == 13:
        return 1  # King High Flush
    return 0  # Flush


def _hand_category_bucket(hole: list[Card], board: list[Card], cat: int, hand: dict) -> int:
    """Index into hand_category_norm's 27-value table (see
    _HAND_CATEGORY_VALUES) -- folds what used to be a dozen-plus standalone
    booleans (top_pair, overpair, low_pair, ace_high_no_pair, ...) directly
    into the made-hand-strength ordinal scale, plus new High Card/Set/
    Straight/Flush/kicker splits, so a NN can read them as *where on the
    strength spectrum* a hand sits rather than as unrelated flags."""
    if cat == HIGH_CARD:
        return _HIGH_CARD_BASE_INDEX + _high_card_bucket_offset(hand)
    if cat == PAIR:
        return _PAIR_BASE_INDEX + _pair_bucket_offset(hole, board)
    if cat == TWO_PAIR:
        return _TWO_PAIR_INDEX
    if cat == TRIPS:
        return _SET_BASE_INDEX + _set_bucket_offset(hole, board)
    if cat == STRAIGHT:
        return _STRAIGHT_BASE_INDEX + _straight_bucket_offset(board, hand["tiebreak"][0])
    if cat == FLUSH:
        return _FLUSH_BASE_INDEX + _flush_bucket_offset(hand)
    if cat == FULL_HOUSE:
        return _FULL_HOUSE_INDEX
    if cat == QUADS:
        return _QUADS_INDEX
    assert cat == STRAIGHT_FLUSH
    return _STRAIGHT_FLUSH_INDEX


def _hole_hand_grid_features(hole: list[Card], street: int) -> dict:
    """hole_hand_grid_x_norm/y_norm: masked to MASKED outside preflop
    (street != 0) -- the hole cards themselves don't change, but a
    human can realistically memorize a chart against all 169 exact
    preflop combos, not against the far larger space of postflop
    hole-cards-vs-board combinations, so the network shouldn't be handed
    this exact an identity once there's a board to read instead."""
    if street != 0:
        return {"hole_hand_grid_x_norm": MASKED, "hole_hand_grid_y_norm": MASKED}
    row, col = _hole_hand_grid_indices(hole)
    denom = HOLE_HAND_GRID_SIZE - 1
    return {"hole_hand_grid_x_norm": col / denom, "hole_hand_grid_y_norm": row / denom}


_NO_FLOP_TEXTURE = {
    "flop_suit_texture_norm": 0.0,
    "flop_pairing_texture_norm": 0.0,
    "flop_connectivity_norm": 0.0, "connected_flop": 0.0,
    "oesd_possible_flop": 0.0,
    "flop_wetness_norm": 0.0,
    "flop_dynamism_norm": 0.0,
}


def _flop_texture(board: list[Card], hole: list[Card]) -> dict:
    """Board-texture facts about the flop specifically (board[:3]) -- plus,
    for the connectivity family, how `hole` connects to it -- frozen once
    the flop is dealt -- the turn/river don't change them, since neither
    the flop cards nor the hole cards do. Defaults all to 0 (their
    lowest-value category) before the flop is dealt. The suit family here
    is purely the flop's own shape (no hole-card awareness) -- see
    _suit_connection_features for that, computed separately since it isn't
    frozen at the flop."""
    if len(board) < 3:
        return dict(_NO_FLOP_TEXTURE)
    flop = board[:3]

    suit_counts = Counter(c.suit for c in flop)
    max_suit_count = max(suit_counts.values())
    monotone, two_tone = max_suit_count == 3, max_suit_count == 2
    suit_index = 2 if monotone else (1 if two_tone else 0)

    rank_counts = Counter(c.rank for c in flop)
    max_rank_count = max(rank_counts.values())
    tripled, paired, unpaired = max_rank_count == 3, max_rank_count == 2, max_rank_count == 1
    pairing_index = 2 if tripled else (1 if paired else 0)

    span = _ace_aware_span([c.rank for c in flop])
    connected = span <= 4
    oesd_possible = unpaired and span <= 3

    if not connected:
        connectivity_index = 0
    elif count_straight_draw_outs(hole + flop) == 0:
        connectivity_index = 1
    else:
        connectivity_index = 2

    # Wetness: how many draws are live, from suit + rank coordination alone
    # (the flop's own shape, not this player's hole cards -- connectivity_index
    # above is the hole-card-aware version of the rank side of this).
    wet_score = suit_index + int(connected) + int(oesd_possible)
    wet = wet_score >= 2

    # Dynamism: whether that draw-heaviness can actually reshuffle who's
    # ahead by the river. Two things pin a board to Static regardless of its
    # wetness score: pairing (made-hand rank -- trips/boat > two pair > top
    # pair -- rarely flips once the board has paired) and very high cards
    # (an A/K/Q on board caps how many hole cards can credibly out-flop the
    # made pair/overpair it hands out, so fewer runouts actually change who's
    # ahead). Low top cards cut the other way -- more hole-card combos have
    # live equity against them -- so they get a matching bonus.
    high_rank = max(c.rank for c in flop)
    high_card_adjustment = -1 if high_rank >= 12 else (1 if high_rank <= 8 else 0)
    dynamic = unpaired and (wet_score + high_card_adjustment) >= 2

    return {
        "flop_suit_texture_norm": suit_index / 2.0,
        "flop_pairing_texture_norm": pairing_index / 2.0,
        "flop_connectivity_norm": connectivity_index / 2.0,
        "connected_flop": float(connectivity_index != 0),
        "oesd_possible_flop": float(oesd_possible),
        "flop_wetness_norm": float(wet),
        "flop_dynamism_norm": float(dynamic),
    }


def _suit_connection_features(hole: list[Card], board: list[Card], street: int) -> dict:
    """suit_connection_index: the most cards of any single suit among hole +
    the *current* board, capped at 5 -- unlike _flop_texture's suit family,
    this isn't frozen at the flop, so it's computed fresh from whatever board
    extract_features was called with (3, 4, or 5 cards -- or 0, preflop).
    Masked to MASKED on the river (street == 3) -- see this feature's
    FeatureSpec description."""
    if street == 3:
        return {"suit_connection_index": MASKED}
    count = min(max(Counter(c.suit for c in hole + board).values()), 5)
    return {"suit_connection_index": count / 5.0}


def extract_features(sit: Situation) -> np.ndarray:
    hand = best_hand_from_available(sit.hole, sit.board)
    cat = hand["category"]
    hand_category_bucket = _hand_category_bucket(sit.hole, sit.board, cat, hand)

    hole_suited = float(sit.hole[0].suit == sit.hole[1].suit)
    gap = _rank_gap(sit.hole[0].rank, sit.hole[1].rank)
    hole_connectivity = _clip01(1.0 - gap / CONNECTIVITY_GAP_CAP)
    hole_category = _hole_hand_category(sit.hole)

    call_amount_norm = _clip01(sit.call_amount / max(sit.pot, 1.0) / _CALL_SIZE_CEILING)
    spr = sit.effective_stack / max(sit.pot, 1.0)
    spr_norm = _clip01(spr / 20.0)
    position_norm = 0.0
    if sit.num_seats_this_street > 1:
        position_norm = sit.position / (sit.num_seats_this_street - 1)
    num_active_norm = _clip01(sit.num_active / 6.0)
    num_raises_norm = _clip01(sit.num_raises_this_street / 3.0)
    raises_last_street_norm = _clip01(sit.num_raises_previous_street / 3.0)
    raises_preflop_norm = min(sit.num_raises_preflop, 3) / 3.0
    raises_flop_norm = min(sit.num_raises_flop, 3) / 3.0
    raises_turn_norm = min(sit.num_raises_turn, 3) / 3.0
    # Fixed for the whole hand (this player's stack *before* any chips went
    # in), not the live, street-to-street-changing current stack -- that's
    # what Stack-To-Pot Ratio (spr_norm, above) already captures.
    stack_depth_norm = _clip01(sit.starting_stack / max(sit.big_blind, 1.0) / 200.0)
    hole_high_card_rank = max(sit.hole[0].rank, sit.hole[1].rank)
    shared_high_card_rank = max((c.rank for c in sit.board), default=None)
    num_overcards = sum(1 for c in sit.board if c.rank > hole_high_card_rank)
    role = seat_role(sit.seat_index, sit.button_idx, sit.num_seats_total)
    role_index = SEAT_ROLES.index(role)

    values = {
        "hand_category_norm": hand_category_bucket / (len(_HAND_CATEGORY_VALUES) - 1),
        "hole_high_card_norm": (hole_high_card_rank - 2) / 12.0,
        "shared_high_card_norm": ((shared_high_card_rank - 2) / 12.0) if shared_high_card_rank else 0.0,
        "num_overcards_norm": num_overcards / 5.0,
        "hole_suited": hole_suited,
        "hole_connectivity": hole_connectivity,
        "hole_hand_category_norm": hole_category / 11.0,
        "street_norm": sit.street / 3.0,
        "is_aggressor_previous_street": float(sit.is_aggressor_previous_street),
        "is_aggressor_preflop": float(sit.is_aggressor_preflop),
        "is_aggressor_flop": float(sit.is_aggressor_flop),
        "is_aggressor_turn": float(sit.is_aggressor_turn),
        "call_amount_norm": call_amount_norm,
        "spr_norm": spr_norm,
        "position_norm": position_norm,
        "starting_position_norm": role_index / (len(SEAT_ROLES) - 1),
        "num_active_norm": num_active_norm,
        "num_raises_norm": num_raises_norm,
        "raises_last_street_norm": raises_last_street_norm,
        "raises_preflop_norm": raises_preflop_norm,
        "raises_flop_norm": raises_flop_norm,
        "raises_turn_norm": raises_turn_norm,
        "stack_depth_norm": stack_depth_norm,
        "opp_vpip_norm": _clip01(sit.opp_vpip),
        "opp_pfr_norm": _clip01(sit.opp_pfr),
        "opp_three_bet_norm": _clip01(sit.opp_three_bet),
        "opp_fold_to_three_bet_norm": _clip01(sit.opp_fold_to_three_bet),
        "opp_aggression_freq_norm": _clip01(sit.opp_aggression_freq),
        "opp_fold_vs_bet_norm": _clip01(sit.opp_fold_vs_bet),
    }
    values.update(_hand_vs_board_heuristics(sit.hole, sit.board, hand, sit.street))
    values.update(_hole_hand_grid_features(sit.hole, sit.street))
    values.update(_flop_texture(sit.board, sit.hole))
    values.update(_suit_connection_features(sit.hole, sit.board, sit.street))

    return np.array([values[spec.key] for spec in FEATURE_SPECS], dtype=np.float64)
