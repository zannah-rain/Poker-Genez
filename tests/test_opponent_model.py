import pytest

from genome import BET_RAISE, CHECK_CALL, FOLD
from opponent_model import (
    CONFIDENCE_HANDS, NEUTRAL, OpponentModel, OpponentStats,
    compute_opponent_features,
)


class TestOpponentStatsDefaults:
    def test_all_rates_default_to_neutral_with_no_observations(self):
        stats = OpponentStats()
        assert stats.vpip_rate() == NEUTRAL
        assert stats.pfr_rate() == NEUTRAL
        assert stats.three_bet_rate() == NEUTRAL
        assert stats.fold_to_three_bet_rate() == NEUTRAL
        assert stats.aggression_freq() == NEUTRAL
        assert stats.fold_vs_bet_rate() == NEUTRAL


class TestRateShrinkage:
    def test_small_sample_shrinks_toward_neutral(self):
        # 1 hit out of 1 chance shouldn't read as a deterministic 100%.
        stats = OpponentStats(vpip_hands=1, hands=1)
        rate = stats.vpip_rate()
        assert NEUTRAL < rate < 1.0

    def test_large_sample_converges_to_observed_rate(self):
        hands = int(CONFIDENCE_HANDS * 100)
        stats = OpponentStats(vpip_hands=hands, hands=hands)  # always VPIP'd
        assert stats.vpip_rate() == pytest.approx(1.0, abs=0.01)

    def test_zero_chances_is_exactly_neutral(self):
        stats = OpponentStats(three_bet_hits=0, three_bet_chances=0)
        assert stats.three_bet_rate() == NEUTRAL


class TestOpponentModelGet:
    def test_get_creates_stats_on_first_access(self):
        model = OpponentModel()
        stats = model.get(1)
        assert isinstance(stats, OpponentStats)

    def test_get_returns_same_object_for_same_player(self):
        model = OpponentModel()
        first = model.get(1)
        first.hands = 5
        second = model.get(1)
        assert second.hands == 5
        assert second is first


class TestStartHand:
    def test_increments_hands_for_every_player(self):
        model = OpponentModel()
        model.start_hand([1, 2, 3])
        assert model.get(1).hands == 1
        assert model.get(2).hands == 1
        assert model.get(3).hands == 1
        model.start_hand([1, 2, 3])
        assert model.get(1).hands == 2

    def test_resets_per_hand_dedup_flags(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, is_preflop=True, action=CHECK_CALL, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).vpip_hands == 1
        # Same hand, another decision that would also count as VPIP -- should
        # NOT double count within the same hand.
        model.record_action(1, is_preflop=True, action=CHECK_CALL, call_amount=4.0, num_raises_before=1, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).vpip_hands == 1
        # New hand -- dedup resets, so this hand's VPIP can be counted again.
        model.start_hand([1])
        model.record_action(1, is_preflop=True, action=CHECK_CALL, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).vpip_hands == 2


class TestRecordActionPreflop:
    def test_call_facing_a_bet_counts_as_vpip(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, CHECK_CALL, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD])
        assert model.get(1).vpip_hands == 1
        assert model.get(1).pfr_hands == 0

    def test_free_check_does_not_count_as_vpip(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, CHECK_CALL, call_amount=0.0, num_raises_before=0, legal_actions=[CHECK_CALL, BET_RAISE])
        assert model.get(1).vpip_hands == 0

    def test_raise_counts_as_both_vpip_and_pfr(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, BET_RAISE, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).vpip_hands == 1
        assert model.get(1).pfr_hands == 1

    def test_fold_does_not_count_as_vpip_or_pfr(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, FOLD, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD])
        assert model.get(1).vpip_hands == 0
        assert model.get(1).pfr_hands == 0

    def test_three_bet_chance_and_hit(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, BET_RAISE, call_amount=4.0, num_raises_before=1, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).three_bet_chances == 1
        assert model.get(1).three_bet_hits == 1

    def test_three_bet_chance_without_hit(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, CHECK_CALL, call_amount=4.0, num_raises_before=1, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).three_bet_chances == 1
        assert model.get(1).three_bet_hits == 0

    def test_three_bet_chance_requires_raise_being_legal(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, CHECK_CALL, call_amount=4.0, num_raises_before=1, legal_actions=[CHECK_CALL])
        assert model.get(1).three_bet_chances == 0

    def test_fold_to_three_bet_chance_and_hit(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, FOLD, call_amount=8.0, num_raises_before=2, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).fold_to_three_bet_chances == 1
        assert model.get(1).fold_to_three_bet_hits == 1

    def test_fold_to_three_bet_chance_requires_fold_being_legal(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, True, CHECK_CALL, call_amount=8.0, num_raises_before=2, legal_actions=[CHECK_CALL])
        assert model.get(1).fold_to_three_bet_chances == 0


class TestRecordActionPostflop:
    def test_bet_raise_counts_toward_aggression(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, False, BET_RAISE, call_amount=0.0, num_raises_before=0, legal_actions=[CHECK_CALL, BET_RAISE])
        assert model.get(1).postflop_bets_raises == 1

    def test_call_of_nonzero_amount_counts_toward_aggression_denominator(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, False, CHECK_CALL, call_amount=5.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).postflop_calls == 1

    def test_free_check_does_not_count_as_a_postflop_call(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, False, CHECK_CALL, call_amount=0.0, num_raises_before=0, legal_actions=[CHECK_CALL, BET_RAISE])
        assert model.get(1).postflop_calls == 0

    def test_fold_vs_bet_chance_and_hit(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, False, FOLD, call_amount=5.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).fold_vs_bet_chances == 1
        assert model.get(1).fold_vs_bet_hits == 1

    def test_fold_vs_bet_chance_requires_facing_a_bet(self):
        model = OpponentModel()
        model.start_hand([1])
        model.record_action(1, False, CHECK_CALL, call_amount=0.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD, BET_RAISE])
        assert model.get(1).fold_vs_bet_chances == 0


class TestComputeOpponentFeatures:
    def test_no_model_returns_all_neutral_defaults(self):
        features = compute_opponent_features(None, [2, 3], villain_id=2)
        assert features.opp_vpip == NEUTRAL
        assert features.villain_three_bet == NEUTRAL

    def test_no_active_opponents_returns_defaults(self):
        model = OpponentModel()
        features = compute_opponent_features(model, [], villain_id=None)
        assert features.opp_vpip == NEUTRAL

    def test_averages_across_active_opponents(self):
        model = OpponentModel()
        # Player 2: always VPIPs. Player 3: never does (over a big sample so
        # shrinkage doesn't wash out the difference).
        for _ in range(500):
            model.start_hand([2, 3])
            model.record_action(2, True, CHECK_CALL, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD])
            model.record_action(3, True, FOLD, call_amount=2.0, num_raises_before=0, legal_actions=[CHECK_CALL, FOLD])
        features = compute_opponent_features(model, [2, 3], villain_id=None)
        assert features.opp_vpip == pytest.approx(0.5, abs=0.05)

    def test_villain_defaults_to_neutral_when_nobody_has_acted(self):
        model = OpponentModel()
        model.start_hand([2])
        features = compute_opponent_features(model, [2], villain_id=None)
        assert features.villain_three_bet == NEUTRAL
        assert features.villain_fold_vs_bet == NEUTRAL
        assert features.villain_aggression_freq == NEUTRAL

    def test_villain_reads_specific_players_stats(self):
        model = OpponentModel()
        for _ in range(500):
            model.start_hand([2, 3])
            model.record_action(2, False, BET_RAISE, call_amount=0.0, num_raises_before=0, legal_actions=[CHECK_CALL, BET_RAISE])
        features = compute_opponent_features(model, [2, 3], villain_id=2)
        assert features.villain_aggression_freq == pytest.approx(1.0, abs=0.05)
