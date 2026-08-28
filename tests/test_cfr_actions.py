import numpy as np
import pytest

import cfr_actions
import strategy
from cards import Card
from features import Situation
from genome import BET_RAISE, CHECK_CALL, FOLD, _apply_decision
from rules import _standard_decision


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


class TestLegalActionCategories:
    def test_call_always_legal(self):
        mask = cfr_actions.legal_action_categories([CHECK_CALL])
        assert mask[strategy.ACTION_CALL]

    def test_fold_legal_only_when_fold_is_legal(self):
        assert cfr_actions.legal_action_categories([CHECK_CALL, FOLD])[strategy.ACTION_FOLD]
        assert not cfr_actions.legal_action_categories([CHECK_CALL])[strategy.ACTION_FOLD]

    def test_raises_and_allin_legal_only_when_bet_raise_is_legal(self):
        with_raise = cfr_actions.legal_action_categories([CHECK_CALL, FOLD, BET_RAISE])
        without_raise = cfr_actions.legal_action_categories([CHECK_CALL, FOLD])
        for a in strategy.RAISE_ACTIONS:
            assert with_raise[a]
            assert not without_raise[a]
        assert with_raise[strategy.ACTION_ALLIN]
        assert not without_raise[strategy.ACTION_ALLIN]

    def test_mask_shape(self):
        mask = cfr_actions.legal_action_categories([CHECK_CALL])
        assert mask.shape == (strategy.NUM_ACTION_CATEGORIES,)


class TestCategoryToGameAction:
    def test_matches_standard_decision_and_apply_decision_directly(self):
        situation = _make_situation()
        legal_actions = [CHECK_CALL, FOLD, BET_RAISE]
        for action_index in range(strategy.NUM_ACTION_CATEGORIES):
            expected = _apply_decision(_standard_decision(action_index, situation), legal_actions)
            actual = cfr_actions.category_to_game_action(action_index, situation, legal_actions)
            assert actual == expected

    def test_fold_category_falls_back_to_call_when_fold_illegal(self):
        situation = _make_situation(call_amount=0.0)
        legal_actions = [CHECK_CALL]  # no FOLD, no BET_RAISE
        action, bet_size = cfr_actions.category_to_game_action(strategy.ACTION_FOLD, situation, legal_actions)
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_raise_category_falls_back_to_call_when_raise_illegal(self):
        situation = _make_situation()
        legal_actions = [CHECK_CALL, FOLD]  # no BET_RAISE
        action, bet_size = cfr_actions.category_to_game_action(strategy.ACTION_RAISE_75, situation, legal_actions)
        assert action == CHECK_CALL
        assert bet_size == 0.0

    def test_allin_category_uses_full_stack(self):
        situation = _make_situation(my_stack=123.0)
        legal_actions = [CHECK_CALL, FOLD, BET_RAISE]
        action, bet_size = cfr_actions.category_to_game_action(strategy.ACTION_ALLIN, situation, legal_actions)
        assert action == BET_RAISE
        assert bet_size == 123.0


class TestRegretMatching:
    def test_normalizes_positive_regrets_over_legal_actions(self):
        regrets = np.array([1.0, 3.0, 0.0], dtype=np.float64)
        mask = np.array([True, True, True])
        sigma = cfr_actions.regret_matching(regrets, mask)
        assert sigma.sum() == pytest.approx(1.0)
        assert sigma[0] == pytest.approx(0.25)
        assert sigma[1] == pytest.approx(0.75)
        assert sigma[2] == pytest.approx(0.0)

    def test_zero_mass_on_illegal_actions(self):
        regrets = np.array([5.0, 5.0, 5.0])
        mask = np.array([True, False, True])
        sigma = cfr_actions.regret_matching(regrets, mask)
        assert sigma[1] == 0.0
        assert sigma.sum() == pytest.approx(1.0)

    def test_uniform_fallback_when_no_positive_regret(self):
        regrets = np.array([-1.0, -2.0, 0.0])
        mask = np.array([True, True, True])
        sigma = cfr_actions.regret_matching(regrets, mask)
        assert np.allclose(sigma, [1 / 3, 1 / 3, 1 / 3])

    def test_uniform_fallback_respects_legal_mask(self):
        regrets = np.array([-1.0, -2.0, 0.0])
        mask = np.array([True, False, True])
        sigma = cfr_actions.regret_matching(regrets, mask)
        assert sigma[1] == 0.0
        assert sigma[0] == pytest.approx(0.5)
        assert sigma[2] == pytest.approx(0.5)


class TestRegretMatchingBatch:
    def test_agrees_with_regret_matching_row_by_row(self):
        rng = np.random.default_rng(0)
        regrets = rng.normal(size=(20, strategy.NUM_ACTION_CATEGORIES))
        legal_masks = rng.random((20, strategy.NUM_ACTION_CATEGORIES)) > 0.4
        legal_masks[:, strategy.ACTION_CALL] = True  # Call is always legal, matching legal_action_categories

        batch = cfr_actions.regret_matching_batch(regrets, legal_masks)

        for i in range(20):
            expected = cfr_actions.regret_matching(regrets[i], legal_masks[i])
            assert np.allclose(batch[i], expected), f"row {i}"

    def test_output_shape(self):
        regrets = np.zeros((7, strategy.NUM_ACTION_CATEGORIES))
        legal_masks = np.ones((7, strategy.NUM_ACTION_CATEGORIES), dtype=bool)
        batch = cfr_actions.regret_matching_batch(regrets, legal_masks)
        assert batch.shape == (7, strategy.NUM_ACTION_CATEGORIES)

    def test_every_row_sums_to_one(self):
        rng = np.random.default_rng(1)
        regrets = rng.normal(size=(50, strategy.NUM_ACTION_CATEGORIES))
        legal_masks = rng.random((50, strategy.NUM_ACTION_CATEGORIES)) > 0.3
        legal_masks[:, strategy.ACTION_CALL] = True
        batch = cfr_actions.regret_matching_batch(regrets, legal_masks)
        assert np.allclose(batch.sum(axis=1), 1.0)
