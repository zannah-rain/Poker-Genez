import numpy as np
import pytest

import cfr_actions
import cfr_features
import strategy
from cfr_networks import AdvantageNet
from cfr_policy import DeepCFRPolicy
from cards import Card
from features import Situation
from game import GameConfig, SeatState, play_hand
from genome import BET_RAISE, CHECK_CALL, FOLD
from player import Player

_FEATURE_KEYS = cfr_features.DEFAULT_FEATURE_KEYS


class _FakeNet:
    def __init__(self, regrets):
        self.regrets = np.asarray(regrets, dtype=np.float64)

    def predict(self, features):
        return self.regrets


def _make_situation(**overrides):
    defaults = dict(
        hole=[Card.from_str("Ah"), Card.from_str("Kh")],
        board=[Card.from_str(t) for t in "2c 5d 9h".split()],
        street=1,
        pot=10.0,
        call_amount=2.0,
        my_stack=190.0,
        effective_stack=190.0,
        position=0,
        num_seats_this_street=3,
        seat_index=0,
        button_idx=0,
        num_seats_total=3,
        num_active=3,
        num_raises_this_street=1,
        num_raises_previous_street=0,
        num_raises_preflop=1,
        num_raises_flop=0,
        num_raises_turn=0,
        is_aggressor_previous_street=False,
        is_aggressor_preflop=False,
        is_aggressor_flop=False,
        is_aggressor_turn=False,
        starting_stack=200.0,
    )
    defaults.update(overrides)
    return Situation(**defaults)


class TestDecideOnlyReturnsLegalActions:
    def test_prefers_a_legal_raise_when_regrets_favor_it(self):
        regrets = np.full(strategy.NUM_ACTION_CATEGORIES, -1.0)
        regrets[strategy.ACTION_RAISE_75] = 100.0
        policy = DeepCFRPolicy(_FakeNet(regrets), _FEATURE_KEYS, deterministic=True)
        situation = _make_situation()
        action, bet_size = policy.decide(situation, [CHECK_CALL, FOLD, BET_RAISE], rng=np.random.default_rng(0))
        assert action == BET_RAISE
        assert bet_size > 0

    def test_ignores_an_illegal_favorite_and_picks_the_best_legal_alternative(self):
        regrets = np.full(strategy.NUM_ACTION_CATEGORIES, -1.0)
        regrets[strategy.ACTION_ALLIN] = 100.0  # requires BET_RAISE, which won't be legal here
        regrets[strategy.ACTION_CALL] = 5.0  # the best action among what's actually legal
        policy = DeepCFRPolicy(_FakeNet(regrets), _FEATURE_KEYS, deterministic=True)
        situation = _make_situation()
        action, _ = policy.decide(situation, [CHECK_CALL, FOLD], rng=np.random.default_rng(0))
        assert action == CHECK_CALL

    def test_never_returns_fold_when_fold_is_illegal(self):
        regrets = np.full(strategy.NUM_ACTION_CATEGORIES, -1.0)
        regrets[strategy.ACTION_FOLD] = 100.0
        policy = DeepCFRPolicy(_FakeNet(regrets), _FEATURE_KEYS, deterministic=True)
        situation = _make_situation(call_amount=0.0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            action, _ = policy.decide(situation, [CHECK_CALL], rng=rng)
            assert action == CHECK_CALL

    def test_stochastic_mode_only_samples_legal_categories(self):
        regrets = np.ones(strategy.NUM_ACTION_CATEGORIES)  # uniform regret matching over everything legal
        policy = DeepCFRPolicy(_FakeNet(regrets), _FEATURE_KEYS, deterministic=False)
        situation = _make_situation()
        rng = np.random.default_rng(0)
        for _ in range(50):
            action, _ = policy.decide(situation, [CHECK_CALL, FOLD], rng=rng)
            assert action in (CHECK_CALL, FOLD)


class _FixedGenome:
    def __init__(self, action, bet_size=0.0):
        self.action = action
        self.bet_size = bet_size

    def decide(self, situation, legal_actions, rng=None):
        action = self.action if self.action in legal_actions else CHECK_CALL
        return action, self.bet_size


class TestIntegrationWithRealGameEngine:
    def test_plays_several_hands_without_crashing_and_conserves_chips(self):
        net = AdvantageNet(input_dim=len(cfr_features.feature_indices(_FEATURE_KEYS)), hidden_sizes=(16, 16))
        policy = DeepCFRPolicy(net, _FEATURE_KEYS, deterministic=False)
        config = GameConfig(small_blind=1.0, big_blind=2.0, starting_stack=200.0)

        seats = [
            SeatState(player=Player(player_id=0, genome=policy), stack=200.0),
            SeatState(player=Player(player_id=1, genome=_FixedGenome(CHECK_CALL)), stack=200.0),
            SeatState(player=Player(player_id=2, genome=_FixedGenome(BET_RAISE, 10.0)), stack=200.0),
        ]
        rng = np.random.default_rng(0)
        for hand_idx in range(10):
            total_before = sum(s.stack for s in seats)
            result = play_hand(seats, button_idx=hand_idx % 3, config=config, rng=rng)
            total_after = sum(s.stack for s in seats)
            assert total_after == pytest.approx(total_before)
            assert sum(result.payouts.values()) == pytest.approx(sum(s.total_committed for s in seats))
