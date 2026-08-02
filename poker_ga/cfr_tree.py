"""External-sampling MCCFR tree traversal for Single Deep CFR, over the
*real* multiway NLHE betting mechanics (game.py) but with each decision
conditioned on an abstracted feature vector (cfr_features.py) instead of a
true information set -- see this project's Deep CFR plan for the rationale.

This intentionally mirrors game.betting_round's/game.play_hand's control
flow step-for-step (same to_act-deque-with-reopen-on-raise loop, same
street-advance/showdown structure) but recursively instead of iteratively,
so it can *branch* at the traversing player's own decision points (trying
every legal action to compute counterfactual regret) instead of only ever
following one line of play. Every other seat -- "opponents" from the
traverser's perspective, even though it's genuinely self-play, one shared
net for every seat -- samples a single action from its current strategy
(external sampling: no importance-weight correction needed, since sampling
opponents according to their own strategy already reproduces the correct
reach-probability-weighted expectation).

Raise-size clamping (_apply_raise below) is re-derived from the same
formula game.betting_round uses (game.py's BET_RAISE branch) rather than
imported, since it's inline in an imperative loop there -- cross-validated
against the real engine by tests/test_cfr_tree.py rather than risking a
refactor of the production betting engine.

Terminal showdown values are an *equity* estimate (see _equity_payouts),
not one single dealt-out board -- when a hand ends all-in before the river,
naively dealing one random runout is an unbiased but very high-variance
sample of the true counterfactual value (e.g. a 60/40 all-in becomes a
binary win-the-whole-pot/lose-everything target instead of something near
"60% of the pot"), and Single Deep CFR's regression target is exactly that
counterfactual value -- feeding the network high-variance labels makes
training much noisier than it needs to be for a quantity we can just
compute (near-)exactly instead of sampling once.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

import cfr_actions
import cfr_features
from cards import Card, Deck, make_deck
from features import Situation
from game import (
    PREFLOP, RIVER, STREET_DEAL_COUNTS, GameConfig, SeatState,
    _effective_stack, _np_rng_to_random, _street_order, compute_payouts,
)
from genome import BET_RAISE, CHECK_CALL, FOLD
from player import Player
from seating import blind_indices, seat_role

NUM_ACTIONS = cfr_actions.NUM_ACTIONS

# Default cap on how many board completions a single terminal-showdown call
# will evaluate (see _sample_completions): a river showdown is always exact
# and free (0 missing cards); a turn all-in is exact too since there are
# only ~44-46 possible river cards, comfortably under this cap; a flop or
# preflop all-in falls back to this many random rollouts, averaged. Bigger
# = lower-variance regression targets but more compute per terminal node --
# tune via cfr_train.DeepCFRConfig/cfr_main's --num-equity-rollouts.
DEFAULT_NUM_EQUITY_ROLLOUTS = 50


@dataclass
class _HandState:
    """The mutable state of one Monte Carlo-sampled hand. Only ever copied
    at the traversing player's own branch points (see _decision_node) --
    every other node mutates in place, since exactly one path forward is
    ever taken from there."""

    seats: list[SeatState]
    board: list[Card]
    deck: Deck
    button_idx: int
    config: GameConfig

    def copy(self) -> "_HandState":
        return _HandState(
            seats=[replace(s) for s in self.seats],
            board=list(self.board),
            deck=copy.copy(self.deck),  # shallow: Deck's _cards list is read-only after shuffle
            button_idx=self.button_idx,
            config=self.config,
        )


@dataclass(frozen=True)
class _TraversalContext:
    """Everything about *how this traversal is learning* -- constant for
    the whole recursive walk, unlike _HandState (which gets copied at the
    traverser's own branch points). Bundled into one object purely so the
    recursive functions below don't each need a dozen positional
    parameters just to pass constants through untouched.

    `net` must expose `.predict(features: np.ndarray) -> np.ndarray` (shape
    (NUM_ACTIONS,), raw regret estimates) -- see cfr_networks.AdvantageNet.
    `reservoir` must expose `.add(features, regrets, legal_mask, t)` -- see
    cfr_reservoir.ReservoirBuffer.
    """

    traverser: int
    net: object
    reservoir: object
    rng: np.random.Generator
    t: float
    feature_indices: np.ndarray
    num_equity_rollouts: int


def _apply_fold(state: _HandState, i: int) -> None:
    state.seats[i].folded = True


def _apply_call(state: _HandState, i: int, current_bet: float, pot: float) -> float:
    seat = state.seats[i]
    call_amount = current_bet - seat.street_committed
    add = min(max(call_amount, 0.0), seat.stack)
    seat.stack -= add
    seat.street_committed += add
    seat.total_committed += add
    pot += add
    if seat.stack <= 1e-9:
        seat.all_in = True
    return pot


def _apply_raise(
    state: _HandState, i: int, raw_bet_size: float, current_bet: float,
    last_raise_increment: float, pot: float, min_bet: float, min_raise_fraction_of_pot: float,
) -> tuple[float, float, float]:
    """Mirrors game.betting_round's BET_RAISE branch exactly. Returns
    (new_pot, new_current_bet, new_last_raise_increment)."""
    seat = state.seats[i]
    max_raise_to = seat.street_committed + seat.stack
    min_raise_increment = max(last_raise_increment, min_bet, min_raise_fraction_of_pot * pot)
    min_raise_to = min(current_bet + min_raise_increment, max_raise_to)
    desired_total = (current_bet if current_bet > 0 else 0.0) + raw_bet_size
    new_bet = min(max(desired_total, min_raise_to), max_raise_to)
    add = new_bet - seat.street_committed
    add = min(add, seat.stack)
    seat.stack -= add
    seat.street_committed += add
    seat.total_committed += add
    pot += add
    if seat.stack <= 1e-9:
        seat.all_in = True
    new_last_raise_increment = max(new_bet - current_bet, min_bet)
    return pot, new_bet, new_last_raise_increment


def _terminal_fold_win(state: _HandState, pot: float, non_folded: list[int], ctx: _TraversalContext) -> float:
    winner = non_folded[0]
    if winner == ctx.traverser:
        return pot - state.seats[ctx.traverser].total_committed
    return -state.seats[ctx.traverser].total_committed


def _unseen_cards(seats: list[SeatState], contributor_indices: list[int], board: list[Card]) -> list[Card]:
    """Every card not already dealt to *some* seat's hole cards (folded or
    not -- every seat was dealt hole cards up front for this Monte Carlo
    hand, see traverse_hand) or already on the board -- the pool a
    still-missing board completion is drawn from."""
    known = set(board)
    for i in contributor_indices:
        known.update(seats[i].hole)
    return [c for c in make_deck() if c not in known]


def _sample_completions(remaining: list[Card], missing: int, num_rollouts: int, rng: np.random.Generator) -> list[tuple]:
    """Every way to complete the board (an exact equity calculation) if
    there are few enough of them; otherwise `num_rollouts` independently
    sampled completions (a Monte Carlo equity estimate). Either way this is
    capped at `num_rollouts` evaluations, so a terminal node's cost never
    depends on how many cards happen to still be missing -- a river-only
    completion (~45 possibilities) ends up exact "for free" once
    num_rollouts is at least that large, while a full 5-card preflop
    all-in (millions of possibilities) always falls back to sampling."""
    total_combos = math.comb(len(remaining), missing)
    if total_combos <= num_rollouts:
        return list(itertools.combinations(remaining, missing))
    completions = []
    for _ in range(num_rollouts):
        idx = rng.choice(len(remaining), size=missing, replace=False)
        completions.append(tuple(remaining[i] for i in idx))
    return completions


def _equity_payouts(
    seats: list[SeatState], contributor_indices: list[int], folded: set[int], board: list[Card],
    num_rollouts: int, rng: np.random.Generator,
) -> dict[int, float]:
    """Expected showdown payout for every contributor, averaged over every
    (or `num_rollouts` sampled) way the board could still be completed --
    generalizes game.compute_payouts (a single fully-dealt board) to a
    possibly-incomplete one. Reduces to exactly compute_payouts, with zero
    extra cost, once the board already has all 5 cards."""
    missing = 5 - len(board)
    if missing <= 0:
        return compute_payouts(seats, contributor_indices, folded, board)

    remaining = _unseen_cards(seats, contributor_indices, board)
    completions = _sample_completions(remaining, missing, num_rollouts, rng)

    totals = {i: 0.0 for i in contributor_indices}
    for completion in completions:
        for i, value in compute_payouts(seats, contributor_indices, folded, board + list(completion)).items():
            totals[i] += value
    n = len(completions)
    return {i: value / n for i, value in totals.items()}


def _terminal_showdown(state: _HandState, ctx: _TraversalContext) -> float:
    contributor_indices = list(range(len(state.seats)))
    folded = {i for i in contributor_indices if state.seats[i].folded}
    payouts = _equity_payouts(state.seats, contributor_indices, folded, state.board, ctx.num_equity_rollouts, ctx.rng)
    return payouts.get(ctx.traverser, 0.0) - state.seats[ctx.traverser].total_committed


def _build_situation(
    state: _HandState, street: int, order: list[int], i: int, pot: float, call_amount: float,
    num_raises: int, preflop_raise_count: int, last_aggressor: int | None, raiser_seats: frozenset,
) -> Situation:
    seats = state.seats
    non_folded = [s for s in order if not seats[s].folded]
    return Situation(
        hole=seats[i].hole,
        board=state.board,
        street=street,
        pot=pot,
        call_amount=max(call_amount, 0.0),
        my_stack=seats[i].stack,
        effective_stack=_effective_stack(seats, i, non_folded),
        position=order.index(i),
        num_seats_this_street=len(order),
        seat_index=i,
        button_idx=state.button_idx,
        num_seats_total=len(seats),
        num_active=len(non_folded),
        num_raises_this_street=num_raises,
        num_preflop_raises=(num_raises if street == PREFLOP else preflop_raise_count),
        is_aggressor=(last_aggressor == i),
        starting_stack=state.config.starting_stack,
        big_blind=state.config.big_blind,
        raised_positions=frozenset(seat_role(j, state.button_idx, len(seats)) for j in raiser_seats if j != i),
    )


def _decision_node(
    state: _HandState, street: int, to_act: list[int], order: list[int], pot: float,
    current_bet: float, last_raise_increment: float, num_raises: int, last_aggressor: int | None,
    raiser_seats: frozenset, preflop_raise_count: int, ctx: _TraversalContext,
) -> float:
    seats = state.seats
    non_folded = [s for s in order if not seats[s].folded]
    if len(non_folded) <= 1:
        return _terminal_fold_win(state, pot, non_folded, ctx)

    if not to_act:
        new_preflop_raise_count = num_raises if street == PREFLOP else preflop_raise_count
        return _start_street(state, street + 1, pot, new_preflop_raise_count, ctx)

    i, rest = to_act[0], to_act[1:]
    if seats[i].folded or seats[i].all_in:
        return _decision_node(
            state, street, rest, order, pot, current_bet, last_raise_increment, num_raises,
            last_aggressor, raiser_seats, preflop_raise_count, ctx,
        )

    call_amount = current_bet - seats[i].street_committed
    legal_actions = [CHECK_CALL]
    if call_amount > 1e-9:
        legal_actions.append(FOLD)
    if seats[i].stack > call_amount + 1e-9 and num_raises < state.config.max_raises_per_street:
        legal_actions.append(BET_RAISE)

    situation = _build_situation(
        state, street, order, i, pot, call_amount, num_raises, preflop_raise_count,
        last_aggressor, raiser_seats,
    )
    legal_mask = cfr_actions.legal_action_categories(legal_actions)
    feats = cfr_features.extract_subset(situation, ctx.feature_indices)
    regrets = ctx.net.predict(feats)
    sigma = cfr_actions.regret_matching(regrets, legal_mask)
    legal_idx = np.flatnonzero(legal_mask)

    def _apply_and_recurse(child_state: _HandState, action_index: int) -> float:
        game_action, raw_bet_size = cfr_actions.category_to_game_action(action_index, situation, legal_actions)
        if game_action == FOLD:
            _apply_fold(child_state, i)
            return _decision_node(
                child_state, street, rest, order, pot, current_bet, last_raise_increment, num_raises,
                last_aggressor, raiser_seats, preflop_raise_count, ctx,
            )
        if game_action == CHECK_CALL:
            new_pot = _apply_call(child_state, i, current_bet, pot)
            return _decision_node(
                child_state, street, rest, order, new_pot, current_bet, last_raise_increment, num_raises,
                last_aggressor, raiser_seats, preflop_raise_count, ctx,
            )
        # BET_RAISE
        new_pot, new_current_bet, new_last_raise_increment = _apply_raise(
            child_state, i, raw_bet_size, current_bet, last_raise_increment, pot,
            state.config.big_blind, state.config.min_raise_fraction_of_pot,
        )
        new_to_act = [s for s in order if s != i and not child_state.seats[s].folded and not child_state.seats[s].all_in]
        new_raiser_seats = raiser_seats | {i}
        return _decision_node(
            child_state, street, new_to_act, order, new_pot, new_current_bet, new_last_raise_increment,
            num_raises + 1, i, new_raiser_seats, preflop_raise_count, ctx,
        )

    if i == ctx.traverser:
        # Explore every legal action (that's the whole point of being the
        # traverser) and weight them by the *current* strategy to get this
        # node's value -- then each action's regret is just how much better
        # or worse it did than that weighted average.
        values = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal_idx:
            values[a] = _apply_and_recurse(state.copy(), int(a))
        v_node = float(np.dot(sigma[legal_idx], values[legal_idx]))
        regret_vec = np.zeros(NUM_ACTIONS, dtype=np.float64)
        regret_vec[legal_idx] = values[legal_idx] - v_node
        ctx.reservoir.add(feats, regret_vec.astype(np.float32), legal_mask, float(ctx.t))
        return v_node

    p = sigma[legal_idx]
    p = p / p.sum()  # guard against float roundoff so np.random.Generator.choice's sum-to-1 check never flakes
    chosen = int(ctx.rng.choice(legal_idx, p=p))
    return _apply_and_recurse(state, chosen)


def _start_street(state: _HandState, street: int, pot: float, preflop_raise_count: int, ctx: _TraversalContext) -> float:
    if street > RIVER:
        return _terminal_showdown(state, ctx)

    seats = state.seats
    if street > PREFLOP:
        state.board += state.deck.deal(STREET_DEAL_COUNTS[street])
        for s in seats:
            s.street_committed = 0.0

    n = len(seats)
    order = [i for i in _street_order(state.button_idx, n, street) if not seats[i].folded]
    non_folded = order
    if len(non_folded) <= 1:
        return _terminal_fold_win(state, pot, non_folded, ctx)

    not_all_in = [i for i in non_folded if not seats[i].all_in]
    if len(not_all_in) < 2:
        return _start_street(state, street + 1, pot, preflop_raise_count, ctx)

    to_act = [i for i in order if not seats[i].all_in]
    starting_bet = max((seats[i].street_committed for i in non_folded), default=0.0)
    return _decision_node(
        state, street, to_act, order, pot, starting_bet, state.config.big_blind, 0, None,
        frozenset(), preflop_raise_count, ctx,
    )


def traverse_hand(
    net, reservoir, table_size: int, config: GameConfig, rng: np.random.Generator, t: float,
    feature_indices: np.ndarray, num_equity_rollouts: int = DEFAULT_NUM_EQUITY_ROLLOUTS,
) -> float:
    """Runs one Monte Carlo external-sampling traversal of a full hand from
    a fresh deal, seating `table_size` players around a randomized button --
    picks a uniformly random traverser seat, whose exact counterfactual
    regret samples get pushed into `reservoir` along the way. Returns the
    traverser's net chip result for this sampled hand (mainly useful for
    logging/sanity checks -- the training signal is the reservoir samples,
    not this return value).
    """
    seats = [
        SeatState(player=Player(player_id=i, genome=None), stack=config.starting_stack)
        for i in range(table_size)
    ]
    for s in seats:
        s.reset_for_hand()

    button_idx = int(rng.integers(0, table_size))
    deck = Deck(rng=_np_rng_to_random(rng))
    for i in range(table_size):
        seats[i].hole = deck.deal(2)

    sb_idx, bb_idx = blind_indices(button_idx, table_size)
    sb_amt = min(config.small_blind, seats[sb_idx].stack)
    bb_amt = min(config.big_blind, seats[bb_idx].stack)
    for idx, amt in ((sb_idx, sb_amt), (bb_idx, bb_amt)):
        seats[idx].stack -= amt
        seats[idx].street_committed = amt
        seats[idx].total_committed = amt
        if seats[idx].stack <= 1e-9:
            seats[idx].all_in = True
    pot = sb_amt + bb_amt

    traverser = int(rng.integers(0, table_size))
    state = _HandState(seats=seats, board=[], deck=deck, button_idx=button_idx, config=config)
    ctx = _TraversalContext(
        traverser=traverser, net=net, reservoir=reservoir, rng=rng, t=t,
        feature_indices=feature_indices, num_equity_rollouts=num_equity_rollouts,
    )
    return _start_street(state, PREFLOP, pot, 0, ctx)
