"""Cross-validates cfr_tree's recursive betting/showdown machinery -- the
one place raise-clamping logic is intentionally duplicated from
game.betting_round rather than imported (see cfr_tree.py's module
docstring) -- against the real game.play_hand engine.

Strategy: script every seat to always resolve its action via
cfr_actions.category_to_game_action(target_category, ...) -- both as a
genome.decide() implementation fed to the real engine (ScriptedGenome) and
as a "dominant regret" fake net fed to cfr_tree (DominantRegretNet, whose
predict() puts all regret-matching probability mass on target_category).
Both systems then resolve every betting decision identically, for any
traverser seat.

Two different checks follow from that, depending on whether the scenario
reaches a *complete* board (all 5 community cards) before it's resolved:

- Full-board scenarios (everyone calls to a river showdown, or everyone
  folds preflop) are still checked for an *exact* match against
  game.play_hand's single dealt-out hand -- cfr_tree._terminal_showdown
  reduces to plain game.compute_payouts with zero equity-averaging
  whenever the board's already complete (see cfr_tree._equity_payouts), so
  there's no reason the two should ever disagree.
- Early-all-in scenarios (a side-pot showdown before the river) are
  *intentionally* no longer a single dealt-out hand on the cfr_tree side --
  see cfr_tree.py's module docstring for why terminal values there are now
  an *equity* estimate instead, which does not correspond to any one
  concrete game.play_hand outcome. Those are checked instead for the one
  invariant that must hold exactly regardless of how many board completions
  get averaged: summing every seat's net payoff across all of them must
  come out to zero, since the whole pot is always fully redistributed (see
  _assert_conserves_chips). Equity-value *correctness* itself (not just
  conservation) is covered separately by test_cfr_equity.py's
  hand-verifiable scenarios.
"""

import numpy as np
import pytest

import cfr_actions
import cfr_features
import cfr_tree
import gto
import strategy
from cards import Card, Deck
from game import PREFLOP, GameConfig, SeatState, _np_rng_to_random, play_hand
from player import Player
from seating import blind_indices

_FEATURE_INDICES = cfr_features.feature_indices(cfr_features.DEFAULT_FEATURE_KEYS)


class _ScriptedGenome:
    """Always resolves via the exact same function cfr_tree uses to turn an
    action category into a game action -- so a whole-table script of these
    is byte-for-byte comparable to cfr_tree's own dominant-regret walk."""

    def __init__(self, category):
        self.category = category

    def decide(self, situation, legal_actions, rng=None):
        return cfr_actions.category_to_game_action(self.category, situation, legal_actions)


class _DominantRegretNet:
    """Fake advantage net: regret_matching on this output puts probability
    1.0 on `category` whenever it's legal. When it *isn't* (e.g. a raise
    category once the street's raise cap is hit, or Fold when checking is
    free), Call is the second-priority action -- Call is always legal, so
    this always resolves the same way _apply_decision's own fallback does
    (Fold/Raise -> Check/Call when illegal), not a uniform draw over
    whatever else happens to be legal."""

    def __init__(self, category, num_actions=cfr_actions.NUM_ACTIONS):
        self.category = category
        self.num_actions = num_actions

    def predict(self, features):
        regrets = np.full(self.num_actions, -1e9, dtype=np.float64)
        regrets[strategy.ACTION_CALL] = 1.0
        regrets[self.category] = 1e9
        return regrets


class _DiscardingReservoir:
    def add(self, features, regrets, legal_mask, t):
        pass


def _run_reference(stacks, button_idx, config, seed, target_category):
    seats = [
        SeatState(player=Player(player_id=i, genome=_ScriptedGenome(target_category)), stack=stacks[i])
        for i in range(len(stacks))
    ]
    result = play_hand(seats, button_idx=button_idx, config=config, rng=np.random.default_rng(seed))
    return result, seats


def _run_cfr(stacks, button_idx, config, seed, target_category, traverser, num_equity_rollouts=cfr_tree.DEFAULT_NUM_EQUITY_ROLLOUTS):
    rng = np.random.default_rng(seed)
    deck = Deck(rng=_np_rng_to_random(rng))
    seats = [
        SeatState(player=Player(player_id=i, genome=None), stack=stacks[i])
        for i in range(len(stacks))
    ]
    for s in seats:
        s.reset_for_hand()
    for i in range(len(seats)):
        seats[i].hole = deck.deal(2)

    sb_idx, bb_idx = blind_indices(button_idx, len(seats))
    sb_amt = min(config.small_blind, seats[sb_idx].stack)
    bb_amt = min(config.big_blind, seats[bb_idx].stack)
    for idx, amt in ((sb_idx, sb_amt), (bb_idx, bb_amt)):
        seats[idx].stack -= amt
        seats[idx].street_committed = amt
        seats[idx].total_committed = amt
        if seats[idx].stack <= 1e-9:
            seats[idx].all_in = True
    pot = sb_amt + bb_amt

    state = cfr_tree._HandState(
        seats=seats, board=[], deck=deck, button_idx=button_idx, config=config,
        starting_stacks=list(stacks),
    )
    ctx = cfr_tree._TraversalContext(
        traverser=traverser, net=_DominantRegretNet(target_category), reservoir=_DiscardingReservoir(),
        rng=rng, t=1.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=num_equity_rollouts,
    )
    return cfr_tree._start_street(state, PREFLOP, pot, 0, None, ctx, 1.0)


def _assert_matches_reference(stacks, button_idx, config, seed, target_category):
    """For scenarios that always reach a complete board before resolving
    (see module docstring) -- an exact match is the right invariant here."""
    reference, reference_seats = _run_reference(stacks, button_idx, config, seed, target_category)
    for traverser in range(len(stacks)):
        cfr_value = _run_cfr(stacks, button_idx, config, seed, target_category, traverser)
        expected = reference.payouts[traverser] - reference_seats[traverser].total_committed
        # abs tolerance: regret_matching's near-1.0 (not exactly 1.0, due to
        # the dominant-vs-fallback regret ratio) sigma can leak a ~1e-9
        # fraction of weight onto a branch whose value differs by O(pot) --
        # negligible in absolute chip terms, but bigger than pytest.approx's
        # default abs=1e-12 once expected happens to be exactly 0.
        assert cfr_value == pytest.approx(expected, abs=1e-6), f"traverser={traverser}"


def _assert_conserves_chips(stacks, button_idx, config, seed, target_category):
    """For scenarios that can reach an early all-in (an incomplete board at
    showdown, see module docstring): cfr_tree's terminal value there is now
    an equity estimate, not a single concrete outcome, so it won't match
    game.play_hand's one dealt-out hand -- but summing every seat's net
    payoff (each already relative to that seat's own total_committed) must
    still come out to exactly zero, however the board got completed,
    because the whole pot always gets fully redistributed (see
    cfr_tree._equity_payouts's docstring: averaging preserves that sum-to-
    the-pot invariant exactly, for any set of sampled completions)."""
    values = [_run_cfr(stacks, button_idx, config, seed, target_category, traverser) for traverser in range(len(stacks))]
    assert sum(values) == pytest.approx(0.0, abs=1e-6)


class TestCfrTreeMatchesGameEngine:
    def test_everyone_always_calls_reaches_a_full_showdown(self):
        config = GameConfig(small_blind=1.0, big_blind=2.0, starting_stack=200.0)
        for seed in range(2):
            _assert_matches_reference([200.0] * 4, button_idx=0, config=config, seed=seed, target_category=strategy.ACTION_CALL)

    def test_everyone_always_folds_ends_uncontested_preflop(self):
        config = GameConfig(small_blind=1.0, big_blind=2.0, starting_stack=200.0)
        for seed in range(3):
            _assert_matches_reference([200.0] * 3, button_idx=0, config=config, seed=seed, target_category=strategy.ACTION_FOLD)

    def test_different_button_positions(self):
        config = GameConfig(small_blind=1.0, big_blind=2.0, starting_stack=200.0)
        for button_idx in range(2):
            _assert_matches_reference([200.0] * 4, button_idx=button_idx, config=config, seed=42, target_category=strategy.ACTION_CALL)


class TestCfrTreeConservesChipsThroughEquityAveraging:
    def test_everyone_always_shoves_produces_side_pots(self):
        config = GameConfig(small_blind=1.0, big_blind=2.0, starting_stack=200.0)
        for seed in range(3):
            _assert_conserves_chips([50.0, 100.0, 150.0], button_idx=0, config=config, seed=seed, target_category=strategy.ACTION_ALLIN)

    def test_raising_to_the_street_cap_then_calling(self):
        config = GameConfig(small_blind=1.0, big_blind=2.0, starting_stack=200.0, max_raises_per_street=3)
        for seed in range(2):
            _assert_conserves_chips([200.0, 200.0], button_idx=0, config=config, seed=seed, target_category=strategy.ACTION_RAISE_100)


class TestIsAggressorReflectsThePreviousStreet:
    def _make_state(self):
        seats = [SeatState(player=Player(player_id=i, genome=None), stack=200.0) for i in range(2)]
        for s in seats:
            s.hole = [Card.from_str("Ah"), Card.from_str("Kh")]
        deck = Deck(rng=_np_rng_to_random(np.random.default_rng(0)))
        return cfr_tree._HandState(
            seats=seats, board=[], deck=deck, button_idx=0, config=GameConfig(),
            starting_stacks=[200.0, 200.0],
        )

    def test_build_situation_uses_previous_street_aggressor_not_this_streets(self):
        state = self._make_state()
        aggressors_situation = cfr_tree._build_situation(
            state, street=1, order=[0, 1], i=0, pot=10.0, call_amount=0.0,
            num_raises=0, preflop_raise_count=1, previous_street_aggressor=0, raiser_seats=frozenset(),
        )
        other_seats_situation = cfr_tree._build_situation(
            state, street=1, order=[0, 1], i=1, pot=10.0, call_amount=0.0,
            num_raises=0, preflop_raise_count=1, previous_street_aggressor=0, raiser_seats=frozenset(),
        )
        assert aggressors_situation.is_aggressor is True
        assert other_seats_situation.is_aggressor is False

    def test_decision_node_hands_off_its_own_street_aggressor_as_the_next_streets_previous(self, monkeypatch):
        # Seat 1 raised at some point this (fictional) street and nobody
        # re-raised since (street_aggressor=1) -- once to_act empties out,
        # _decision_node should hand that off as the *next* street's
        # previous_street_aggressor, not whatever was true before this one
        # started (previous_street_aggressor=None here, deliberately
        # different from street_aggressor, so the test would fail if the
        # two ever got mixed up).
        captured = {}

        def fake_start_street(state, street, pot, preflop_raise_count, previous_street_aggressor, ctx, path_weight):
            captured["previous_street_aggressor"] = previous_street_aggressor
            captured["street"] = street
            return 0.0

        monkeypatch.setattr(cfr_tree, "_start_street", fake_start_street)
        state = self._make_state()
        ctx = cfr_tree._TraversalContext(
            traverser=0, net=_DominantRegretNet(strategy.ACTION_CALL), reservoir=_DiscardingReservoir(),
            rng=np.random.default_rng(0), t=1.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
        )

        cfr_tree._decision_node(
            state, street=0, to_act=[], order=[0, 1], pot=10.0, current_bet=0.0, last_raise_increment=2.0,
            num_raises=1, street_aggressor=1, previous_street_aggressor=None, raiser_seats=frozenset({1}),
            preflop_raise_count=0, ctx=ctx, path_weight=1.0,
        )

        assert captured["previous_street_aggressor"] == 1
        assert captured["street"] == 1


class TestPositionRelativeToNonFoldedPlayers:
    """features.Situation.position/num_seats_this_street (and so
    position_norm) should reflect only players still in the hand at the
    moment of this decision -- not `order`'s fixed street-start lineup --
    since a player who was dealt a late seat but finds everyone ahead of
    them has already folded is, in every way that matters to the decision
    in front of them, acting early, not late."""

    def _make_state(self, num_seats=4):
        seats = [SeatState(player=Player(player_id=i, genome=None), stack=200.0) for i in range(num_seats)]
        for s in seats:
            s.hole = [Card.from_str("Ah"), Card.from_str("Kh")]
        deck = Deck(rng=_np_rng_to_random(np.random.default_rng(0)))
        return cfr_tree._HandState(
            seats=seats, board=[], deck=deck, button_idx=0, config=GameConfig(),
            starting_stacks=[200.0] * num_seats,
        )

    def test_folded_seats_are_excluded_from_position_and_count(self):
        # order=[0, 1, 2, 3] at street start, but seats 0 and 2 have since
        # folded -- seat 1 should read as acting first (position 0.0) of 2
        # remaining, not second (position 1/3) of the original 4.
        state = self._make_state(num_seats=4)
        state.seats[0].folded = True
        state.seats[2].folded = True

        seat1_situation = cfr_tree._build_situation(
            state, street=1, order=[0, 1, 2, 3], i=1, pot=10.0, call_amount=0.0,
            num_raises=0, preflop_raise_count=0, previous_street_aggressor=None, raiser_seats=frozenset(),
        )
        seat3_situation = cfr_tree._build_situation(
            state, street=1, order=[0, 1, 2, 3], i=3, pot=10.0, call_amount=0.0,
            num_raises=0, preflop_raise_count=0, previous_street_aggressor=None, raiser_seats=frozenset(),
        )

        assert seat1_situation.num_seats_this_street == 2
        assert seat1_situation.position == 0
        assert seat3_situation.num_seats_this_street == 2
        assert seat3_situation.position == 1

    def test_no_folds_matches_the_full_street_start_order(self):
        state = self._make_state(num_seats=4)
        situation = cfr_tree._build_situation(
            state, street=1, order=[0, 1, 2, 3], i=2, pot=10.0, call_amount=0.0,
            num_raises=0, preflop_raise_count=0, previous_street_aggressor=None, raiser_seats=frozenset(),
        )
        assert situation.num_seats_this_street == 4
        assert situation.position == 2


class _ExplodingNet:
    """Fails the test loudly if _decision_node ever consults the net for a
    decision a gto_spots entry was supposed to fix outright."""

    def predict(self, features):
        raise AssertionError("net.predict() called for a decision a matching gto_spots entry should have fixed")


class _ExplodingReservoir:
    """Fails the test loudly if _decision_node ever adds a training sample
    for a decision a gto_spots entry was supposed to fix outright -- a
    fixed decision isn't learned, so it should never produce a regret
    target."""

    def add(self, features, regrets, legal_mask, t):
        raise AssertionError("reservoir.add() called for a decision a matching gto_spots entry should have fixed")


class TestGtoSpotsOverrideDecisions:
    """cfr_tree._decision_node should play a matching gto.GTOSpot's fixed
    action verbatim -- for every seat, traverser included -- without ever
    touching the net or reservoir for that decision (see gto.py's module
    docstring)."""

    def _make_state(self, num_seats=2, stack=200.0):
        seats = [SeatState(player=Player(player_id=i, genome=None), stack=stack) for i in range(num_seats)]
        for s in seats:
            s.hole = [Card.from_str("7c"), Card.from_str("2d")]  # a hand no default_action="fold" range would cover
        deck = Deck(rng=_np_rng_to_random(np.random.default_rng(0)))
        return cfr_tree._HandState(
            seats=seats, board=[], deck=deck, button_idx=0, config=GameConfig(),
            starting_stacks=[stack] * num_seats,
        )

    def _fold_everywhere_spot(self):
        return gto.GTOSpot(
            key="always_fold", label="Always Fold Test Spot",
            matcher=gto.SpotMatcher(), action_ranges=(), default_action="fold",
        )

    def test_matching_spot_forces_fold_without_touching_net_or_reservoir(self):
        state = self._make_state()
        ctx = cfr_tree._TraversalContext(
            traverser=1, net=_ExplodingNet(), reservoir=_ExplodingReservoir(),
            rng=np.random.default_rng(0), t=1.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
            gto_spots=(self._fold_everywhere_spot(),),
        )
        # Seat 0 facing a bet (call_amount > 0) with an active gto_spots
        # catalog that always resolves to fold: this should fold seat 0
        # immediately, ending the hand uncontested, and never reach
        # net.predict()/reservoir.add() (both of which would raise).
        value = cfr_tree._decision_node(
            state, street=0, to_act=[0], order=[0, 1], pot=10.0, current_bet=6.0, last_raise_increment=2.0,
            num_raises=1, street_aggressor=1, previous_street_aggressor=None, raiser_seats=frozenset({1}),
            preflop_raise_count=1, ctx=ctx, path_weight=1.0,
        )
        # Traverser (seat 1) already put in `pot`'s worth this street; once
        # seat 0 folds uncontested, the traverser wins pot minus their own
        # total contribution.
        assert value == pytest.approx(10.0 - state.seats[1].total_committed)

    def test_empty_gto_spots_falls_through_to_the_net(self):
        state = self._make_state()
        ctx = cfr_tree._TraversalContext(
            traverser=1, net=_DominantRegretNet(strategy.ACTION_CALL), reservoir=_DiscardingReservoir(),
            rng=np.random.default_rng(0), t=1.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
            gto_spots=(),
        )
        # No fixed spot in play -- should reach the (dominant-regret) net
        # instead of raising, i.e. this must not itself raise.
        cfr_tree._decision_node(
            state, street=0, to_act=[0], order=[0, 1], pot=10.0, current_bet=6.0, last_raise_increment=2.0,
            num_raises=1, street_aggressor=1, previous_street_aggressor=None, raiser_seats=frozenset({1}),
            preflop_raise_count=1, ctx=ctx, path_weight=1.0,
        )

    def test_non_matching_spot_falls_through_to_the_net(self):
        state = self._make_state()
        river_only_spot = gto.GTOSpot(
            key="river_only", label="River Only", matcher=gto.SpotMatcher(street=3),
            action_ranges=(), default_action="fold",
        )
        ctx = cfr_tree._TraversalContext(
            traverser=1, net=_DominantRegretNet(strategy.ACTION_CALL), reservoir=_DiscardingReservoir(),
            rng=np.random.default_rng(0), t=1.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
            gto_spots=(river_only_spot,),
        )
        # This decision is preflop -- the river-only spot shouldn't match,
        # so this must fall through to the net rather than force a fold.
        cfr_tree._decision_node(
            state, street=0, to_act=[0], order=[0, 1], pot=10.0, current_bet=6.0, last_raise_increment=2.0,
            num_raises=1, street_aggressor=1, previous_street_aggressor=None, raiser_seats=frozenset({1}),
            preflop_raise_count=1, ctx=ctx, path_weight=1.0,
        )

    def test_fixed_bb_raise_spot_sizes_by_big_blind_not_pot(self, monkeypatch):
        # A fixed "raise_1.5bb" spot should commit exactly current_bet +
        # 1.5 * big_blind chips, regardless of how large `pot` is --
        # unlike a pot-fraction category (see gto.py's _ActionSpec.decision).
        # Intercepts the *next* decision (seat 1's, once seat 0's forced
        # raise reopens the action) the same way
        # test_decision_node_hands_off_its_own_street_aggressor... does for
        # _start_street, since otherwise the street/hand would keep playing
        # out (this same universal spot would also fix seat 1's response)
        # until street_committed gets reset for the next street anyway.
        real_decision_node = cfr_tree._decision_node
        captured = {}

        def fake_decision_node(state, *args, **kwargs):
            captured["seat0_street_committed"] = state.seats[0].street_committed
            return 0.0

        monkeypatch.setattr(cfr_tree, "_decision_node", fake_decision_node)

        config = GameConfig(small_blind=1.0, big_blind=2.0)
        state = cfr_tree._HandState(
            seats=[SeatState(player=Player(player_id=i, genome=None), stack=200.0) for i in range(2)],
            board=[], deck=Deck(rng=_np_rng_to_random(np.random.default_rng(0))), button_idx=0, config=config,
            starting_stacks=[200.0, 200.0],
        )
        for s in state.seats:
            s.hole = [Card.from_str("7c"), Card.from_str("2d")]
        state.seats[0].street_committed = 4.0
        state.seats[0].stack -= 4.0
        fixed_bb_spot = gto.GTOSpot(
            key="fixed_bb", label="Fixed BB Raise Test Spot",
            matcher=gto.SpotMatcher(), action_ranges=(), default_action="raise_1.5bb",
        )
        ctx = cfr_tree._TraversalContext(
            traverser=1, net=_ExplodingNet(), reservoir=_ExplodingReservoir(),
            rng=np.random.default_rng(0), t=1.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
            gto_spots=(fixed_bb_spot,),
        )
        real_decision_node(
            state, street=0, to_act=[0], order=[0, 1], pot=6.0, current_bet=6.0, last_raise_increment=2.0,
            num_raises=1, street_aggressor=1, previous_street_aggressor=None, raiser_seats=frozenset({1}),
            preflop_raise_count=1, ctx=ctx, path_weight=1.0,
        )
        # Was facing a call of 2.0 (current_bet 6.0 - already committed 4.0)
        # -- the fixed spot's raise adds 1.5 * big_blind (3.0) on top of
        # current_bet, for a total of 9.0 (see gto.py's _ActionSpec.decision).
        assert captured["seat0_street_committed"] == pytest.approx(9.0)


class _RecordingReservoir:
    """Records every sample's own stored weight (t), in the order
    _decision_node adds them. Exploration is post-order -- a node's own
    sample is only added once every one of its own children has already
    been fully explored (v_node isn't known until then) -- so the very
    first traverser decision reached in a hand always ends up *last* in
    this list, not first."""

    def __init__(self):
        self.weights = []

    def add(self, features, regrets, legal_mask, t):
        self.weights.append(t)


class TestPathWeightCorrectsOverRepresentation:
    """See cfr_tree.py's own module docstring: full exploration of the
    traverser's own actions means a hand that happens to reach many, or
    branchy, decision points would otherwise push vastly more reservoir
    samples -- and so vastly more training weight, once the reservoir's
    contents get regressed on -- than one that resolves quickly, purely
    from tree structure, not real importance. _decision_node/
    _apply_and_recurse correct for this via `path_weight`, folded into
    each sample's own stored `t`."""

    def test_dividing_by_branch_count_at_a_single_traverser_node(self, monkeypatch):
        # Direct, exact check of the core mechanism, bypassing traverse_hand's
        # own randomness: call_amount is 0 here (current_bet ==
        # street_committed), so the only *game* actions are check/call and
        # bet/raise -- but bet/raise itself splits into 7 size-specific
        # action categories (see strategy.ACTION_CATEGORIES), for 8 legal
        # categories total (cfr_actions.legal_action_categories) -- each
        # should get exactly 1/8 of the incoming path_weight.
        seats = [SeatState(player=Player(player_id=i, genome=None), stack=200.0) for i in range(2)]
        for s in seats:
            s.hole = [Card.from_str("7c"), Card.from_str("2d")]
        deck = Deck(rng=_np_rng_to_random(np.random.default_rng(0)))
        state = cfr_tree._HandState(
            seats=seats, board=[], deck=deck, button_idx=0, config=GameConfig(),
            starting_stacks=[200.0, 200.0],
        )
        ctx = cfr_tree._TraversalContext(
            traverser=0, net=_DominantRegretNet(strategy.ACTION_RAISE_75), reservoir=_DiscardingReservoir(),
            rng=np.random.default_rng(0), t=100.0, feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
        )
        captured_weights = []

        def fake_decision_node(state, *args, **kwargs):
            captured_weights.append(args[-1])  # path_weight is always the last positional arg
            return 0.0

        real_decision_node = cfr_tree._decision_node
        monkeypatch.setattr(cfr_tree, "_decision_node", fake_decision_node)
        real_decision_node(
            state, street=0, to_act=[0], order=[0, 1], pot=10.0, current_bet=0.0, last_raise_increment=2.0,
            num_raises=0, street_aggressor=None, previous_street_aggressor=None, raiser_seats=frozenset(),
            preflop_raise_count=0, ctx=ctx, path_weight=1.0,
        )
        assert captured_weights == pytest.approx([0.125] * 8)

    def test_first_traverser_decision_in_a_hand_keeps_the_full_weight(self):
        reservoir = _RecordingReservoir()
        cfr_tree.traverse_hand(
            net=_DominantRegretNet(strategy.ACTION_RAISE_75), reservoir=reservoir, table_size=2,
            config=GameConfig(), rng=np.random.default_rng(0), t=100.0, feature_indices=_FEATURE_INDICES,
            num_equity_rollouts=1,
        )
        assert reservoir.weights[-1] == pytest.approx(100.0)

    def test_deeper_decisions_are_discounted_including_within_the_same_street(self):
        # A raise-dominant net for *every* seat (not just the traverser)
        # reliably produces a same-street raise war: the traverser's own
        # BET_RAISE branch leads to the opponent's single-sampled response
        # (also dominant-raise, since the net is shared -- see cfr_tree.py's
        # own module docstring on this being genuinely self-play), which
        # re-raises back for another traverser decision on the very same
        # street, not a different one -- proving the correction isn't just
        # a per-street thing.
        reservoir = _RecordingReservoir()
        cfr_tree.traverse_hand(
            net=_DominantRegretNet(strategy.ACTION_RAISE_75), reservoir=reservoir, table_size=2,
            config=GameConfig(max_raises_per_street=4), rng=np.random.default_rng(0), t=100.0,
            feature_indices=_FEATURE_INDICES, num_equity_rollouts=1,
        )
        assert len(reservoir.weights) > 1
        assert all(w <= 100.0 for w in reservoir.weights)
        assert min(reservoir.weights) < 100.0 / 4  # meaningfully discounted, not just float noise
