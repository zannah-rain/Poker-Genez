import numpy as np
import pytest

from cards import Card
from evaluator import PAIR, TRIPS
from features import (
    FEATURE_GROUPS, FEATURE_NAMES, FEATURE_SPECS, NUM_FEATURES, Situation,
    _ace_aware_span, _connectivity_label, _nearest_bucket_index, _rank_gap,
    extract_features, group_of,
)
from seating import SEAT_ROLES


def make_situation(**overrides) -> Situation:
    defaults = dict(
        hole=[Card.from_str("Ah"), Card.from_str("Kd")],
        board=[],
        street=0,
        pot=10.0,
        call_amount=0.0,
        my_stack=200.0,
        effective_stack=200.0,
        position=0,
        num_seats_this_street=6,
        seat_index=0,
        button_idx=0,
        num_seats_total=6,
        num_active=6,
        num_raises_this_street=0,
        num_preflop_raises=0,
        is_aggressor=False,
        starting_stack=200.0,
        big_blind=2.0,
    )
    defaults.update(overrides)
    return Situation(**defaults)


def values_by_key(situation: Situation) -> dict:
    features = extract_features(situation)
    return dict(zip(FEATURE_NAMES, features))


class TestFeatureCatalogIntegrity:
    def test_names_and_specs_line_up(self):
        assert NUM_FEATURES == len(FEATURE_SPECS) == len(FEATURE_NAMES)
        assert FEATURE_NAMES == [spec.key for spec in FEATURE_SPECS]

    def test_keys_are_unique(self):
        assert len(set(FEATURE_NAMES)) == NUM_FEATURES

    def test_every_spec_group_is_a_known_group_or_inherited(self):
        for spec in FEATURE_SPECS:
            assert group_of(spec) in FEATURE_GROUPS

    def test_linked_children_inherit_parent_group(self):
        parents_by_key = {s.key: s for s in FEATURE_SPECS}
        for spec in FEATURE_SPECS:
            if spec.linked_to is not None:
                parent = parents_by_key[spec.linked_to]
                assert group_of(spec) == group_of(parent)

    def test_categorical_and_continuous_specs_have_value_tables(self):
        for spec in FEATURE_SPECS:
            if spec.kind in ("categorical", "continuous"):
                assert spec.value_table is not None
                assert len(spec.value_table) > 0


class TestExtractFeaturesShape:
    def test_returns_correct_length_float_array(self):
        features = extract_features(make_situation())
        assert isinstance(features, np.ndarray)
        assert features.shape == (NUM_FEATURES,)
        assert features.dtype == np.float64

    def test_all_values_are_finite(self):
        features = extract_features(make_situation())
        assert np.all(np.isfinite(features))


class TestHandCategoryFeatures:
    def test_trips_hand_sets_category_and_indicator(self):
        situation = make_situation(
            hole=[Card.from_str("9h"), Card.from_str("9d")],
            board=[Card.from_str("9c"), Card.from_str("2s"), Card.from_str("4h")],
            street=1,
        )
        values = values_by_key(situation)
        assert values["hand_category_norm"] == pytest.approx(TRIPS / 8.0)
        assert values["has_trips"] == 1.0
        for key in ["has_high_card", "has_pair", "has_two_pair", "has_straight", "has_flush",
                    "has_full_house", "has_quads", "has_straight_flush"]:
            assert values[key] == 0.0

    def test_preflop_pocket_pair_is_a_pair(self):
        situation = make_situation(hole=[Card.from_str("9h"), Card.from_str("9d")], board=[])
        values = values_by_key(situation)
        assert values["hand_category_norm"] == pytest.approx(PAIR / 8.0)
        assert values["has_pair"] == 1.0


class TestHoleCardFeatures:
    def test_suited_flag(self):
        suited = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kh")]))
        offsuit = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")]))
        assert suited["hole_suited"] == 1.0
        assert offsuit["hole_suited"] == 0.0

    def test_pocket_pair_has_max_connectivity(self):
        values = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("9d")]))
        assert values["hole_connectivity"] == 1.0
        assert values["hole_paired"] == 1.0

    def test_connectors_have_high_but_not_max_connectivity(self):
        values = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("8d")]))
        assert values["hole_connectivity"] == pytest.approx(1 - 1 / 6)
        assert values["hole_paired"] == 0.0
        assert values["connectivity_gap_1"] == 1.0

    def test_only_pocket_pair_reaches_max_connectivity(self):
        paired = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("9d")]))
        connectors = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("8d")]))
        assert paired["hole_connectivity"] == 1.0
        assert connectors["hole_connectivity"] < 1.0

    def test_wide_gap_hands_have_zero_connectivity(self):
        values = values_by_key(make_situation(hole=[Card.from_str("2h"), Card.from_str("Kd")]))
        assert values["hole_connectivity"] == 0.0
        assert values["connectivity_gap_6plus"] == 1.0

    def test_ace_king_treated_as_gap_one(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")]))
        assert values["connectivity_gap_1"] == 1.0

    def test_ace_deuce_treated_as_gap_one(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("2d")]))
        assert values["connectivity_gap_1"] == 1.0

    def test_hole_high_card_indicator(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("2d")]))
        assert values["hole_high_card_norm"] == pytest.approx((14 - 2) / 12.0)
        assert values["hole_high_card_is_ace"] == 1.0
        assert values["hole_high_card_is_king"] == 0.0


class TestPositionAndStreet:
    def test_position_norm_zero_when_only_one_seat(self):
        values = values_by_key(make_situation(num_seats_this_street=1, position=0))
        assert values["position_norm"] == 0.0

    def test_position_norm_scales_between_0_and_1(self):
        first = values_by_key(make_situation(num_seats_this_street=4, position=0))
        last = values_by_key(make_situation(num_seats_this_street=4, position=3))
        mid = values_by_key(make_situation(num_seats_this_street=4, position=1))
        assert first["position_norm"] == 0.0
        assert last["position_norm"] == 1.0
        assert mid["position_norm"] == pytest.approx(1 / 3)

    def test_street_one_hot(self):
        for street, key in enumerate(["is_preflop", "is_flop", "is_turn", "is_river"]):
            values = values_by_key(make_situation(street=street))
            assert values[key] == 1.0
            assert values["street_norm"] == pytest.approx(street / 3.0)
            others = {"is_preflop", "is_flop", "is_turn", "is_river"} - {key}
            for other in others:
                assert values[other] == 0.0


class TestBettingAndPotFeatures:
    def test_facing_bet_flag(self):
        assert values_by_key(make_situation(call_amount=0.0))["facing_bet"] == 0.0
        assert values_by_key(make_situation(call_amount=5.0))["facing_bet"] == 1.0

    def test_call_amount_norm_clips_above_pot(self):
        values = values_by_key(make_situation(pot=100.0, call_amount=500.0))
        assert values["call_amount_norm"] == 1.0

    def test_call_amount_norm_fraction_of_pot(self):
        values = values_by_key(make_situation(pot=100.0, call_amount=25.0))
        assert values["call_amount_norm"] == pytest.approx(0.25)

    def test_spr_norm_clips_at_20x_pot(self):
        values = values_by_key(make_situation(pot=10.0, effective_stack=1000.0))
        assert values["spr_norm"] == 1.0

    def test_num_raises_norm_clips_at_4(self):
        values = values_by_key(make_situation(num_raises_this_street=10))
        assert values["num_raises_norm"] == 1.0
        assert values["raises_is_4plus"] == 1.0

    def test_pot_type_norm_freezes_preflop_raise_count(self):
        values = values_by_key(make_situation(num_preflop_raises=2, num_raises_this_street=0, street=1))
        assert values["pot_type_norm"] == pytest.approx(2 / 3)
        assert values["is_3bet_pot"] == 1.0

    def test_pot_type_norm_clips_at_3_raises(self):
        values = values_by_key(make_situation(num_preflop_raises=10))
        assert values["pot_type_norm"] == 1.0
        assert values["is_4bet_plus_pot"] == 1.0

    def test_is_aggressor_flag(self):
        assert values_by_key(make_situation(is_aggressor=True))["is_aggressor"] == 1.0
        assert values_by_key(make_situation(is_aggressor=False))["is_aggressor"] == 0.0


class TestStackAndSeatFeatures:
    def test_stack_depth_norm_at_starting_stack_is_half(self):
        values = values_by_key(make_situation(my_stack=200.0, starting_stack=200.0))
        assert values["stack_depth_norm"] == pytest.approx(0.5)

    def test_stack_depth_norm_clips_at_double_stack(self):
        values = values_by_key(make_situation(my_stack=1000.0, starting_stack=200.0))
        assert values["stack_depth_norm"] == 1.0

    def test_stack_depth_norm_zero_when_busted(self):
        values = values_by_key(make_situation(my_stack=0.0, starting_stack=200.0))
        assert values["stack_depth_norm"] == 0.0

    def test_num_active_norm(self):
        values = values_by_key(make_situation(num_active=3))
        assert values["num_active_norm"] == pytest.approx(0.5)
        assert values["active_players_is_3"] == 1.0

    def test_starting_seat_role_one_hot_matches_seat_role(self):
        # button_idx=0, n=6: seat 2 is BB (see seating.blind_indices).
        values = values_by_key(make_situation(seat_index=2, button_idx=0, num_seats_total=6))
        assert values["is_big_blind"] == 1.0
        for role in SEAT_ROLES:
            if role != "BB":
                assert values[{"UTG": "is_utg", "HJ": "is_hijack", "CO": "is_cutoff",
                               "BTN": "is_button", "SB": "is_small_blind", "BB": "is_big_blind"}[role]] == 0.0


class TestFlopTexture:
    def test_preflop_defaults_are_all_zero(self):
        values = values_by_key(make_situation(board=[]))
        for key in ["flop_suit_texture_norm", "rainbow_flop", "flush_draw_flop", "monotone_flop",
                    "flop_pairing_texture_norm", "unpaired_flop", "paired_flop", "tripled_flop",
                    "flop_connectivity_norm", "disconnected_flop", "connected_flop", "oesd_possible_flop"]:
            assert values[key] == 0.0

    def test_rainbow_unpaired_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["rainbow_flop"] == 1.0
        assert values["unpaired_flop"] == 1.0

    def test_monotone_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["monotone_flop"] == 1.0

    def test_two_tone_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flush_draw_flop"] == 1.0

    def test_paired_flop(self):
        board = [Card.from_str("2c"), Card.from_str("2d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["paired_flop"] == 1.0

    def test_tripled_flop(self):
        board = [Card.from_str("2c"), Card.from_str("2d"), Card.from_str("2h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["tripled_flop"] == 1.0

    def test_connected_flop_within_span_4(self):
        board = [Card.from_str("5c"), Card.from_str("7d"), Card.from_str("9h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["connected_flop"] == 1.0

    def test_disconnected_flop_wide_span(self):
        board = [Card.from_str("2c"), Card.from_str("8d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["disconnected_flop"] == 1.0

    def test_oesd_possible_needs_unpaired_and_tight_span(self):
        board = [Card.from_str("5c"), Card.from_str("6d"), Card.from_str("7h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["oesd_possible_flop"] == 1.0

    def test_flop_texture_frozen_on_turn_and_river(self):
        board3 = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]  # monotone
        flop_values = values_by_key(make_situation(board=board3, street=1))
        board4 = board3 + [Card.from_str("9d")]  # turn card breaks the monotone look
        turn_values = values_by_key(make_situation(board=board4, street=2))
        assert turn_values["monotone_flop"] == flop_values["monotone_flop"] == 1.0


class TestHandVsBoardHeuristics:
    def test_top_pair(self):
        hole = [Card.from_str("Kh"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["top_pair"] == 1.0
        assert values["second_pair"] == 0.0

    def test_second_pair(self):
        hole = [Card.from_str("7h"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["second_pair"] == 1.0
        assert values["top_pair"] == 0.0

    def test_third_pair(self):
        hole = [Card.from_str("3h"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["third_pair"] == 1.0

    def test_overpair(self):
        hole = [Card.from_str("Kh"), Card.from_str("Kd")]
        board = [Card.from_str("7c"), Card.from_str("4d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["overpair"] == 1.0
        assert values["underpair"] == 0.0

    def test_underpair_and_low_pair(self):
        hole = [Card.from_str("4h"), Card.from_str("4d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("5h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["underpair"] == 1.0
        assert values["low_pair"] == 1.0  # 4 is below every board card

    def test_underpair_but_not_low_pair(self):
        # pair (7) is below the top card (K) but above the bottom card (3).
        hole = [Card.from_str("6h"), Card.from_str("6d")]
        board = [Card.from_str("Kc"), Card.from_str("5d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["underpair"] == 1.0
        assert values["low_pair"] == 0.0

    def test_ace_high_no_pair(self):
        hole = [Card.from_str("Ah"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["ace_high_no_pair"] == 1.0

    def test_king_high_no_pair(self):
        hole = [Card.from_str("Kh"), Card.from_str("2d")]
        board = [Card.from_str("Qc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["king_high_no_pair"] == 1.0

    def test_nuts_flush_draw(self):
        hole = [Card.from_str("Ah"), Card.from_str("9h")]
        board = [Card.from_str("7h"), Card.from_str("4h"), Card.from_str("2c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["flush_draw"] == 1.0
        assert values["nuts_flush_draw"] == 1.0

    def test_flush_draw_without_the_nut_card(self):
        hole = [Card.from_str("Qh"), Card.from_str("9h")]
        board = [Card.from_str("7h"), Card.from_str("4h"), Card.from_str("2c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["flush_draw"] == 1.0
        assert values["nuts_flush_draw"] == 0.0

    def test_open_ended_straight_draw_and_gutshot(self):
        oesd_hole = [Card.from_str("6h"), Card.from_str("7d")]
        oesd_board = [Card.from_str("8c"), Card.from_str("9s"), Card.from_str("2h")]
        values = values_by_key(make_situation(hole=oesd_hole, board=oesd_board, street=1))
        assert values["open_ended_straight_draw"] == 1.0
        assert values["gutshot"] == 0.0

        gutshot_hole = [Card.from_str("6h"), Card.from_str("7d")]
        gutshot_board = [Card.from_str("8c"), Card.from_str("Ts"), Card.from_str("2h")]
        values2 = values_by_key(make_situation(hole=gutshot_hole, board=gutshot_board, street=1))
        assert values2["gutshot"] == 1.0
        assert values2["open_ended_straight_draw"] == 0.0

    def test_combo_draw(self):
        hole = [Card.from_str("6h"), Card.from_str("7h")]
        board = [Card.from_str("8h"), Card.from_str("9h"), Card.from_str("2c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["flush_draw"] == 1.0
        assert values["straight_draw"] == 1.0
        assert values["combo_draw"] == 1.0

    def test_backdoor_flush_draw_two_hole_cards(self):
        hole = [Card.from_str("Ah"), Card.from_str("9h")]
        board = [Card.from_str("7h"), Card.from_str("4c"), Card.from_str("2d")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["backdoor_flush_draw_2"] == 1.0
        assert values["backdoor_flush_draw_1"] == 0.0

    def test_backdoor_flush_draw_one_hole_card(self):
        hole = [Card.from_str("Ah"), Card.from_str("9c")]
        board = [Card.from_str("7h"), Card.from_str("4h"), Card.from_str("2d")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["backdoor_flush_draw_1"] == 1.0
        assert values["backdoor_flush_draw_2"] == 0.0

    def test_backdoor_draws_only_apply_on_the_flop(self):
        hole = [Card.from_str("Ah"), Card.from_str("9h")]
        board = [Card.from_str("7h"), Card.from_str("4c"), Card.from_str("2d"), Card.from_str("3s")]
        values = values_by_key(make_situation(hole=hole, board=board, street=2))
        assert values["backdoor_flush_draw_2"] == 0.0
        assert values["backdoor_flush_draw_1"] == 0.0


class TestOpponentTendencyFeatures:
    def test_defaults_are_neutral(self):
        values = values_by_key(make_situation())
        assert values["opp_vpip_norm"] == 0.5

    def test_passes_through_situation_values(self):
        values = values_by_key(make_situation(
            opp_vpip=0.8, opp_pfr=0.3, villain_three_bet=0.9,
        ))
        assert values["opp_vpip_norm"] == pytest.approx(0.8)
        assert values["opp_pfr_norm"] == pytest.approx(0.3)
        assert values["villain_three_bet_norm"] == pytest.approx(0.9)

    def test_opponent_sample_size_feature_was_removed(self):
        # "doesn't interact with any other features" -- dropped as a genome
        # feature entirely, not just left unused.
        assert "opp_sample_norm" not in FEATURE_NAMES

    def test_clips_out_of_range_values(self):
        values = values_by_key(make_situation(opp_vpip=1.5))
        assert values["opp_vpip_norm"] == 1.0


class TestBucketIndicators:
    def test_exactly_one_bucket_active_per_continuous_family(self):
        values = values_by_key(make_situation(pot=100.0, call_amount=25.0))
        bucket_keys = [f"call_amount_norm_bucket_{i}" for i in range(5)]
        active = [values[k] for k in bucket_keys]
        assert sum(active) == 1.0

    def test_nearest_bucket_helper_picks_closest_point(self):
        assert _nearest_bucket_index(0.0) == 0
        assert _nearest_bucket_index(1.0) == 4
        assert _nearest_bucket_index(0.26) == 1
        assert _nearest_bucket_index(0.6) == 2

    def test_nearest_bucket_helper_breaks_exact_ties_toward_the_lower_index(self):
        # 0.125/0.375/0.625/0.875 sit exactly halfway between two bucket
        # points -- the original min-over-range implementation resolves
        # such ties to the first (lower) index it encounters, so any
        # faster reimplementation must match that, not round-half-to-even.
        assert _nearest_bucket_index(0.125) == 0
        assert _nearest_bucket_index(0.375) == 1
        assert _nearest_bucket_index(0.625) == 2
        assert _nearest_bucket_index(0.875) == 3

    def test_nearest_bucket_helper_clamps_out_of_range_values(self):
        assert _nearest_bucket_index(-0.1) == 0
        assert _nearest_bucket_index(1.3) == 4


class TestPrivateHelpers:
    def test_rank_gap_treats_ace_as_high_or_low(self):
        assert _rank_gap(14, 13) == 1  # A-K
        assert _rank_gap(14, 2) == 1  # A-2
        assert _rank_gap(2, 3) == 1
        assert _rank_gap(2, 9) == 6  # capped

    def test_ace_aware_span_prefers_low_straight_reading(self):
        assert _ace_aware_span([14, 2, 3]) == 2  # A-2-3 read as 1-2-3
        assert _ace_aware_span([14, 13, 12]) == 2  # A-K-Q read high

    def test_connectivity_label_examples(self):
        assert _connectivity_label(0) == "Same rank (pocket pair)"
        assert _connectivity_label(1) == "1 apart (connectors, e.g. 9-10)"
        assert "or more apart" in _connectivity_label(6)
