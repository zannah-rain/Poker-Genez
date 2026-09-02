"""A 6-max No-Limit Hold'em engine: blinds, betting rounds, all-ins with
correct side pots, and showdown. Player decisions are delegated entirely to
each seat's Player.genome (any duck-typed .decide(situation, legal_actions,
rng) -- e.g. cfr_policy.DeepCFRPolicy) via features.Situation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cards import Card, Deck
from evaluator import evaluate_best
from features import Situation
from genome import BET_RAISE, CHECK_CALL, FOLD
from opponent_model import OpponentModel, compute_opponent_features
from player import Player
from seating import blind_indices, preflop_order, seat_role

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
STREET_DEAL_COUNTS = {FLOP: 3, TURN: 1, RIVER: 1}


@dataclass
class SeatState:
    player: Player
    stack: float
    hole: list[Card] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    street_committed: float = 0.0
    total_committed: float = 0.0

    def reset_for_hand(self) -> None:
        self.hole = []
        self.folded = False
        self.all_in = self.stack <= 0
        self.street_committed = 0.0
        self.total_committed = 0.0


@dataclass
class GameConfig:
    small_blind: float = 1.0
    big_blind: float = 2.0
    starting_stack: float = 200.0  # in big blinds' worth of chips (== 100bb if bb=2)
    max_hands_per_session: int = 500
    # Caps re-raising within a single street, matching the standard "N-bet
    # cap" used in many structured games. Without this, nothing stops two
    # genomes that both read "villain raised" as a reason to raise again
    # (a locally-learnable but collectively pathological correlation) from
    # re-raising each other dozens of times in one street, inflating the
    # pot to an all-in almost every hand regardless of actual hand strength.
    max_raises_per_street: int = 4
    # Floors every raise-to at (current bet + this fraction of the pot), on
    # top of the standard min-bet/previous-raise-size floor. Without this, a
    # genome whose action score A is only barely positive can raise for
    # next to nothing (bet_size scales with A), which is nearly free for the
    # aggressor but still forces whoever's next to act to make a decision --
    # a cheap way to grind opponents toward folding or committing chips a
    # sliver at a time, regardless of actual hand strength.
    min_raise_fraction_of_pot: float = 0.25


@dataclass
class HandStats:
    """Mutable, optional accumulator for sense-check metrics -- threaded
    through betting_round/play_hand so simulate.py can report generation-
    level trends (action mix, fold rate when facing a bet, raises per
    street) without every caller needing to opt in or pay for it."""
    action_counts: dict = field(default_factory=lambda: {FOLD: 0, CHECK_CALL: 0, BET_RAISE: 0})
    # Facing-a-bet decisions specifically (i.e. FOLD was a legal action) --
    # the fold rate that actually matters for "is this genome ever folding,"
    # since most CHECK_CALL actions are free checks, not a genuine choice to
    # continue over folding.
    facing_bet_decisions: int = 0
    facing_bet_folds: int = 0
    raises_per_street: list = field(default_factory=list)


@dataclass
class HandResult:
    payouts: dict  # seat_index -> chips won
    board: list


def _street_order(button_idx: int, n: int, street: int) -> list[int]:
    if street == PREFLOP:
        return preflop_order(button_idx, n)
    sb_idx, bb_idx = blind_indices(button_idx, n)
    start = bb_idx if n == 2 else sb_idx
    return [(start + k) % n for k in range(n)]


def _effective_stack(seats: list[SeatState], me: int, non_folded: list[int]) -> float:
    others = [seats[j].stack for j in non_folded if j != me]
    if not others:
        return seats[me].stack
    return min(seats[me].stack, max(others))


def betting_round(
    seats: list[SeatState],
    order: list[int],
    pot: float,
    min_bet: float,
    starting_bet: float,
    board: list[Card],
    street: int,
    starting_stack: float,
    button_idx: int,
    completed_street_raises: tuple[int, ...],
    completed_street_aggressors: tuple[int | None, ...],
    max_raises_per_street: int,
    min_raise_fraction_of_pot: float,
    rng: np.random.Generator,
    stats: HandStats | None = None,
    log: list | None = None,
    opp_model: OpponentModel | None = None,
) -> tuple[float, int, int | None]:
    """Runs one betting round in-place on `seats`. Returns (updated pot,
    number of raises made this street, this street's own last aggressor --
    None if nobody raised -- for the caller to append onto
    `completed_street_raises`/`completed_street_aggressors` once this
    street is over).

    `completed_street_raises`/`completed_street_aggressors` hold one entry
    per street *already* completed before this one (index 0 = preflop, 1 =
    flop, 2 = turn -- always exactly `street` entries long, since this
    street itself isn't in there yet), feeding every Situation this round
    builds with the frozen per-calendar-street num_raises_preflop/flop/turn
    fields (see features.Situation's own field docs) -- each stays 0
    until its own street has actually finished, then keeps that final
    reading for every later street, including a skipped one (see
    play_hand, which appends a (0, None) placeholder for a street where
    betting never even happens because too few players still have chips --
    once that's true for one street it's true for every later one too, so
    no Situation ever actually observes the placeholder). Deliberately not
    this street's own live-tracked aggressor for the "previous street"
    reading specifically: whoever raised most recently *this* street is
    skipped over in the to-act order until either the street ends or
    someone else re-raises (immediately replacing them as this street's
    aggressor), so a player can never be making a decision while also
    being this street's own last aggressor -- see
    features.Situation.is_aggressor_previous_street. `opp_model`, if given,
    both supplies each decision's opponent-tendency features and gets
    updated with the outcome of that decision -- see opponent_model.py."""
    current_bet = starting_bet
    last_raise_increment = min_bet
    last_aggressor: int | None = None  # this street's own (live-updating) aggressor
    num_raises = 0
    raiser_seats: set[int] = set()

    num_raises_previous_street = completed_street_raises[-1] if completed_street_raises else 0
    num_raises_preflop = completed_street_raises[0] if len(completed_street_raises) > 0 else 0
    num_raises_flop = completed_street_raises[1] if len(completed_street_raises) > 1 else 0
    num_raises_turn = completed_street_raises[2] if len(completed_street_raises) > 2 else 0

    to_act = deque(i for i in order if not seats[i].all_in and not seats[i].folded)

    while to_act:
        i = to_act.popleft()
        if seats[i].folded or seats[i].all_in:
            continue
        non_folded = [s for s in order if not seats[s].folded]
        if len(non_folded) <= 1:
            break

        call_amount = current_bet - seats[i].street_committed
        legal_actions = [CHECK_CALL]
        if call_amount > 1e-9:
            legal_actions.append(FOLD)
        if seats[i].stack > call_amount + 1e-9 and num_raises < max_raises_per_street:
            legal_actions.append(BET_RAISE)

        position = non_folded.index(i)
        opponent_ids = [seats[j].player.player_id for j in non_folded if j != i]
        opp_features = compute_opponent_features(opp_model, opponent_ids)
        situation = Situation(
            hole=seats[i].hole,
            board=board,
            street=street,
            pot=pot,
            call_amount=max(call_amount, 0.0),
            my_stack=seats[i].stack,
            effective_stack=_effective_stack(seats, i, non_folded),
            position=position,
            num_seats_this_street=len(non_folded),
            seat_index=i,
            button_idx=button_idx,
            num_seats_total=len(seats),
            num_active=len(non_folded),
            num_raises_this_street=num_raises,
            num_raises_previous_street=num_raises_previous_street,
            num_raises_preflop=num_raises_preflop,
            num_raises_flop=num_raises_flop,
            num_raises_turn=num_raises_turn,
            is_aggressor_previous_street=(completed_street_aggressors[-1] == i if completed_street_aggressors else False),
            starting_stack=starting_stack,
            big_blind=min_bet,
            raised_positions=frozenset(
                seat_role(j, button_idx, len(seats)) for j in raiser_seats if j != i
            ),
            last_raiser_position=(
                seat_role(last_aggressor, button_idx, len(seats)) if last_aggressor is not None else None
            ),
            active_positions=frozenset(seat_role(j, button_idx, len(seats)) for j in non_folded if j != i),
            opp_vpip=opp_features.opp_vpip,
            opp_pfr=opp_features.opp_pfr,
            opp_three_bet=opp_features.opp_three_bet,
            opp_fold_to_three_bet=opp_features.opp_fold_to_three_bet,
            opp_aggression_freq=opp_features.opp_aggression_freq,
            opp_fold_vs_bet=opp_features.opp_fold_vs_bet,
        )
        action, raw_bet_size = seats[i].player.genome.decide(situation, legal_actions, rng)

        if opp_model is not None:
            opp_model.record_action(
                seats[i].player.player_id, street == PREFLOP, action,
                max(call_amount, 0.0), num_raises, legal_actions,
            )

        if stats is not None:
            stats.action_counts[action] += 1
            if FOLD in legal_actions:
                stats.facing_bet_decisions += 1
                if action == FOLD:
                    stats.facing_bet_folds += 1

        if action == FOLD:
            seats[i].folded = True
            if log is not None:
                log.append((i, "fold"))
        elif action == BET_RAISE:
            max_raise_to = seats[i].street_committed + seats[i].stack
            min_raise_increment = max(last_raise_increment, min_bet, min_raise_fraction_of_pot * pot)
            min_raise_to = min(current_bet + min_raise_increment, max_raise_to)
            desired_total = (current_bet if current_bet > 0 else 0.0) + raw_bet_size
            new_bet = min(max(desired_total, min_raise_to), max_raise_to)
            add = new_bet - seats[i].street_committed
            add = min(add, seats[i].stack)
            seats[i].stack -= add
            seats[i].street_committed += add
            seats[i].total_committed += add
            pot += add
            if seats[i].stack <= 1e-9:
                seats[i].all_in = True
            last_raise_increment = max(new_bet - current_bet, min_bet)
            current_bet = new_bet
            last_aggressor = i
            raiser_seats.add(i)
            num_raises += 1
            if log is not None:
                log.append((i, "raise", new_bet))
            to_act = deque(
                s for s in order if s != i and not seats[s].folded and not seats[s].all_in
            )
        else:  # CHECK_CALL
            add = min(max(call_amount, 0.0), seats[i].stack)
            seats[i].stack -= add
            seats[i].street_committed += add
            seats[i].total_committed += add
            pot += add
            if seats[i].stack <= 1e-9:
                seats[i].all_in = True
            if log is not None:
                log.append((i, "call" if add > 1e-9 else "check"))

    if stats is not None:
        stats.raises_per_street.append(num_raises)

    return pot, num_raises, last_aggressor


def compute_payouts(
    seats: list[SeatState],
    contributor_indices: list[int],
    folded: set[int],
    board: list[Card],
) -> dict[int, float]:
    """Standard side-pot algorithm: repeatedly peel off the smallest
    remaining contribution as a pot layer, split among non-folded
    contributors to that layer by best hand."""
    remaining = {i: seats[i].total_committed for i in contributor_indices}
    payouts = {i: 0.0 for i in contributor_indices}
    hand_cache: dict[int, tuple] = {}

    def hand_of(i: int) -> tuple:
        if i not in hand_cache:
            hand_cache[i] = evaluate_best(seats[i].hole + board)
        return hand_cache[i]

    while any(v > 1e-9 for v in remaining.values()):
        nonzero = {i: v for i, v in remaining.items() if v > 1e-9}
        layer_amount = min(nonzero.values())
        pot_amount = 0.0
        eligible = []
        for i, v in nonzero.items():
            take = min(v, layer_amount)
            pot_amount += take
            remaining[i] -= take
            if i not in folded:
                eligible.append(i)
        if not eligible:
            # Everyone contributing to this layer already folded; return
            # unclaimed chips proportionally to whoever is still eligible
            # overall (rare edge case with simultaneous side actions).
            still_in = [i for i in contributor_indices if i not in folded]
            eligible = still_in
        if eligible:
            best = max(hand_of(i) for i in eligible)
            winners = [i for i in eligible if hand_of(i) == best]
            share = pot_amount / len(winners)
            for w in winners:
                payouts[w] += share
    return payouts


def play_hand(
    seats: list[SeatState],
    button_idx: int,
    config: GameConfig,
    rng: np.random.Generator,
    stats: HandStats | None = None,
    opp_model: OpponentModel | None = None,
) -> HandResult:
    """Plays a single hand in-place on `seats` (a list of currently-alive
    players at the table). Returns the payouts and final board. `opp_model`,
    if given, has its per-hand bookkeeping reset here (see
    OpponentModel.start_hand) and is threaded into every betting round."""
    n = len(seats)
    for s in seats:
        s.reset_for_hand()
    if opp_model is not None:
        opp_model.start_hand([s.player.player_id for s in seats])

    deck = Deck(rng=_np_rng_to_random(rng))
    for i in range(n):
        seats[i].hole = deck.deal(2)

    sb_idx, bb_idx = blind_indices(button_idx, n)
    sb_amt = min(config.small_blind, seats[sb_idx].stack)
    bb_amt = min(config.big_blind, seats[bb_idx].stack)
    for idx, amt in ((sb_idx, sb_amt), (bb_idx, bb_amt)):
        seats[idx].stack -= amt
        seats[idx].street_committed = amt
        seats[idx].total_committed = amt
        if seats[idx].stack <= 1e-9:
            seats[idx].all_in = True
    pot = sb_amt + bb_amt

    board: list[Card] = []
    contributor_indices = list(range(n))
    # completed_street_raises[s]/completed_street_aggressors[s] = street s's
    # own final raise count / last aggressor seat (None if nobody raised),
    # appended once street s is over (see betting_round's own docstring for
    # why a street where betting never even happens still gets a (0, None)
    # placeholder appended here, and why that's never actually observed).
    completed_street_raises: tuple[int, ...] = ()
    completed_street_aggressors: tuple[int | None, ...] = ()

    for street in range(4):
        if street > 0:
            board += deck.deal(STREET_DEAL_COUNTS[street])
            for s in seats:
                s.street_committed = 0.0

        order = [i for i in _street_order(button_idx, n, street) if not seats[i].folded]
        non_folded = order
        if len(non_folded) <= 1:
            break

        not_all_in = [i for i in non_folded if not seats[i].all_in]
        if len(not_all_in) >= 2:
            starting_bet = max((seats[i].street_committed for i in non_folded), default=0.0)
            pot, street_num_raises, street_aggressor = betting_round(
                seats, order, pot, config.big_blind, starting_bet, board, street,
                config.starting_stack, button_idx, completed_street_raises, completed_street_aggressors,
                config.max_raises_per_street, config.min_raise_fraction_of_pot, rng,
                stats=stats, opp_model=opp_model,
            )
        else:
            street_num_raises, street_aggressor = 0, None
        completed_street_raises += (street_num_raises,)
        completed_street_aggressors += (street_aggressor,)
        non_folded = [i for i in order if not seats[i].folded]
        if len(non_folded) <= 1:
            break

    non_folded = [i for i in contributor_indices if not seats[i].folded]
    folded = {i for i in contributor_indices if seats[i].folded}

    if len(non_folded) == 1:
        winner = non_folded[0]
        payouts = {i: 0.0 for i in contributor_indices}
        payouts[winner] = sum(seats[i].total_committed for i in contributor_indices)
    else:
        # deal remaining board cards if the hand ended in all-ins early
        while len(board) < 5:
            board += deck.deal(1)
        payouts = compute_payouts(seats, contributor_indices, folded, board)

    for i in contributor_indices:
        seats[i].stack += payouts[i]

    return HandResult(payouts=payouts, board=board)


def _np_rng_to_random(rng: np.random.Generator):
    """Seeds a python random.Random (Deck's shuffle interface) from a numpy
    Generator so the whole framework can be driven by one RNG seed."""
    import random

    return random.Random(int(rng.integers(0, 2**32 - 1)))
