import numpy as np
import pytest

from cards import Card
from features import (
    FEATURE_GROUPS, FEATURE_NAMES, FEATURE_SPECS, HOLE_HAND_GRID_MASKED, NUM_FEATURES, Situation,
    _ace_aware_span, _connectivity_label, _rank_gap,
    extract_features, group_of, hole_hand_grid_label,
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
    def test_trips_hand_sets_category(self):
        # Pocket 9s matching the board's single highest card (9c) -- the
        # best possible three of a kind on this board -- is Top Set (15/25).
        situation = make_situation(
            hole=[Card.from_str("9h"), Card.from_str("9d")],
            board=[Card.from_str("9c"), Card.from_str("2s"), Card.from_str("4h")],
            street=1,
        )
        values = values_by_key(situation)
        assert values["hand_category_norm"] == pytest.approx(15 / 25)

    def test_preflop_pocket_pair_is_a_pair(self):
        # No board yet to classify the pocket pair as over/under/low
        # against -- lands in the generic "Pair" catch-all bucket.
        situation = make_situation(hole=[Card.from_str("9h"), Card.from_str("9d")], board=[])
        values = values_by_key(situation)
        assert values["hand_category_norm"] == pytest.approx(5 / 25)

    def test_ace_high_no_pair_is_the_ace_high_bucket(self):
        hole = [Card.from_str("Ah"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(2 / 25)

    def test_king_high_no_pair_is_the_king_high_bucket(self):
        hole = [Card.from_str("Kh"), Card.from_str("2d")]
        board = [Card.from_str("Qc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(1 / 25)

    def test_queen_high_no_pair_is_the_generic_high_card_bucket(self):
        hole = [Card.from_str("Qh"), Card.from_str("2d")]
        board = [Card.from_str("Jc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(0 / 25)


class TestHoleCardFeatures:
    def test_suited_flag(self):
        suited = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kh")]))
        offsuit = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")]))
        assert suited["hole_suited"] == 1.0
        assert offsuit["hole_suited"] == 0.0

    def test_pocket_pair_has_max_connectivity(self):
        values = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("9d")]))
        assert values["hole_connectivity"] == 1.0

    def test_connectors_have_high_but_not_max_connectivity(self):
        values = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("8d")]))
        assert values["hole_connectivity"] == pytest.approx(1 - 1 / 6)

    def test_only_pocket_pair_reaches_max_connectivity(self):
        paired = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("9d")]))
        connectors = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("8d")]))
        assert paired["hole_connectivity"] == 1.0
        assert connectors["hole_connectivity"] < 1.0

    def test_wide_gap_hands_have_zero_connectivity(self):
        values = values_by_key(make_situation(hole=[Card.from_str("2h"), Card.from_str("Kd")]))
        assert values["hole_connectivity"] == 0.0

    def test_ace_king_and_ace_deuce_both_treated_as_gap_one(self):
        # Ace plays both high and low for connectivity purposes, so A-K and
        # A-2 read the same 1-apart gap as a genuine 9-8 (see
        # test_connectors_have_high_but_not_max_connectivity).
        ak = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")]))
        a2 = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("2d")]))
        connectors = values_by_key(make_situation(hole=[Card.from_str("9h"), Card.from_str("8d")]))
        assert ak["hole_connectivity"] == a2["hole_connectivity"] == connectors["hole_connectivity"]

    def test_hole_high_card_indicator(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("2d")]))
        assert values["hole_high_card_norm"] == pytest.approx((14 - 2) / 12.0)


class TestHoleHandCategoryFeature:
    def test_premium_high_and_average_pairs(self):
        assert values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Ad")]))[
            "hole_hand_category_norm"] == pytest.approx(11 / 11)  # Premium Pairs
        assert values_by_key(make_situation(hole=[Card.from_str("Th"), Card.from_str("Td")]))[
            "hole_hand_category_norm"] == pytest.approx(10 / 11)  # High Pairs
        assert values_by_key(make_situation(hole=[Card.from_str("5h"), Card.from_str("5d")]))[
            "hole_hand_category_norm"] == pytest.approx(9 / 11)  # Average Pairs

    def test_ace_suited_vs_offsuit(self):
        assert values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("4h")]))[
            "hole_hand_category_norm"] == pytest.approx(8 / 11)  # Axs
        assert values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("4d")]))[
            "hole_hand_category_norm"] == pytest.approx(7 / 11)  # Axo

    def test_ace_king_suited_is_axs_not_kxs(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kh")]))
        assert values["hole_hand_category_norm"] == pytest.approx(8 / 11)  # Axs, not Kxs (6/11)

    def test_king_suited_and_queen_suited_kickers(self):
        assert values_by_key(make_situation(hole=[Card.from_str("Kh"), Card.from_str("4h")]))[
            "hole_hand_category_norm"] == pytest.approx(6 / 11)  # Kxs
        assert values_by_key(make_situation(hole=[Card.from_str("Qh"), Card.from_str("4h")]))[
            "hole_hand_category_norm"] == pytest.approx(5 / 11)  # Qxs

    def test_king_queen_suited_is_kxs_not_qxs(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Kh"), Card.from_str("Qh")]))
        assert values["hole_hand_category_norm"] == pytest.approx(6 / 11)  # Kxs, not Qxs (5/11)

    def test_suited_connectors_gappers(self):
        assert values_by_key(make_situation(hole=[Card.from_str("Jh"), Card.from_str("Th")]))[
            "hole_hand_category_norm"] == pytest.approx(4 / 11)  # Suited Connectors
        assert values_by_key(make_situation(hole=[Card.from_str("Jh"), Card.from_str("9h")]))[
            "hole_hand_category_norm"] == pytest.approx(3 / 11)  # Suited 1 Gappers
        assert values_by_key(make_situation(hole=[Card.from_str("Jh"), Card.from_str("8h")]))[
            "hole_hand_category_norm"] == pytest.approx(2 / 11)  # Suited 2 Gappers

    def test_unsuited_connectors(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Jh"), Card.from_str("Td")]))
        assert values["hole_hand_category_norm"] == pytest.approx(1 / 11)

    def test_junk_hand(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Jh"), Card.from_str("5d")]))
        assert values["hole_hand_category_norm"] == pytest.approx(0 / 11)


class TestHoleHandGridFeature:
    def test_pocket_pair_is_on_the_diagonal(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Ad")]))
        assert values["hole_hand_grid_x_norm"] == pytest.approx(0.0)
        assert values["hole_hand_grid_y_norm"] == pytest.approx(0.0)

        values = values_by_key(make_situation(hole=[Card.from_str("2h"), Card.from_str("2d")]))
        assert values["hole_hand_grid_x_norm"] == pytest.approx(1.0)
        assert values["hole_hand_grid_y_norm"] == pytest.approx(1.0)

    def test_suited_combo_is_above_the_diagonal(self):
        # AKs: x (col) = K's index (1/12), y (row) = A's index (0) -- row < col.
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kh")]))
        assert values["hole_hand_grid_y_norm"] < values["hole_hand_grid_x_norm"]
        assert values["hole_hand_grid_x_norm"] == pytest.approx(1 / 12)
        assert values["hole_hand_grid_y_norm"] == pytest.approx(0.0)

    def test_offsuit_combo_is_below_the_diagonal(self):
        # AKo: x (col) = A's index (0), y (row) = K's index (1/12) -- row > col.
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")]))
        assert values["hole_hand_grid_y_norm"] > values["hole_hand_grid_x_norm"]
        assert values["hole_hand_grid_x_norm"] == pytest.approx(0.0)
        assert values["hole_hand_grid_y_norm"] == pytest.approx(1 / 12)

    def test_masked_outside_preflop(self):
        board = [Card.from_str("2c"), Card.from_str("5d"), Card.from_str("9h")]
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kh")], board=board, street=1))
        assert values["hole_hand_grid_x_norm"] == -1.0
        assert values["hole_hand_grid_y_norm"] == -1.0

    def test_masked_value_is_outside_the_normal_0_1_range(self):
        # A deliberately out-of-range sentinel -- 0.0 would collide with AA's
        # own real (preflop) grid position and be silently misread as it.
        assert not (0.0 <= HOLE_HAND_GRID_MASKED <= 1.0)


class TestHoleHandGridLabel:
    def test_pairs(self):
        assert hole_hand_grid_label(0, 0) == "AA"
        assert hole_hand_grid_label(12, 12) == "22"

    def test_suited_above_diagonal(self):
        assert hole_hand_grid_label(0, 1) == "AKs"
        assert hole_hand_grid_label(0, 12) == "A2s"

    def test_offsuit_below_diagonal(self):
        assert hole_hand_grid_label(1, 0) == "AKo"
        assert hole_hand_grid_label(12, 0) == "A2o"

    def test_label_only_depends_on_the_unordered_row_col_pair(self):
        # (row, col) and (col, row) name the same two ranks -- only whether
        # row < col (suited) or row > col (offsuit) should differ.
        assert hole_hand_grid_label(2, 5)[:2] == hole_hand_grid_label(5, 2)[:2]
        assert hole_hand_grid_label(2, 5).endswith("s")
        assert hole_hand_grid_label(5, 2).endswith("o")


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

    def test_street_norm_by_street(self):
        for street, expected in enumerate([0.0, 1 / 3, 2 / 3, 1.0]):
            values = values_by_key(make_situation(street=street))
            assert values["street_norm"] == pytest.approx(expected)


class TestBettingAndPotFeatures:
    def test_call_amount_norm_clips_above_pot(self):
        # Comfortably past even the Overbet ceiling (1.25x pot) -> clipped.
        values = values_by_key(make_situation(pot=100.0, call_amount=500.0))
        assert values["call_amount_norm"] == 1.0

    def test_call_amount_norm_fraction_of_pot(self):
        # 0.25x pot / 1.25 (the Overbet ceiling) == 0.2.
        values = values_by_key(make_situation(pot=100.0, call_amount=25.0))
        assert values["call_amount_norm"] == pytest.approx(0.2)

    def test_call_amount_norm_full_pot_bet(self):
        values = values_by_key(make_situation(pot=100.0, call_amount=100.0))
        assert values["call_amount_norm"] == pytest.approx(0.8)

    def test_call_amount_norm_overbet(self):
        values = values_by_key(make_situation(pot=100.0, call_amount=150.0))
        assert values["call_amount_norm"] == 1.0

    def test_spr_norm_clips_at_20x_pot(self):
        values = values_by_key(make_situation(pot=10.0, effective_stack=1000.0))
        assert values["spr_norm"] == 1.0

    def test_num_raises_norm_clips_at_3(self):
        values = values_by_key(make_situation(num_raises_this_street=10))
        assert values["num_raises_norm"] == 1.0

    def test_pot_type_norm_freezes_preflop_raise_count(self):
        values = values_by_key(make_situation(num_preflop_raises=2, num_raises_this_street=0, street=1))
        assert values["pot_type_norm"] == pytest.approx(2 / 3)

    def test_pot_type_norm_clips_at_3_raises(self):
        values = values_by_key(make_situation(num_preflop_raises=10))
        assert values["pot_type_norm"] == 1.0

    def test_is_aggressor_flag(self):
        assert values_by_key(make_situation(is_aggressor=True))["is_aggressor"] == 1.0
        assert values_by_key(make_situation(is_aggressor=False))["is_aggressor"] == 0.0


class TestStackAndSeatFeatures:
    def test_stack_depth_norm_at_100bb_is_half(self):
        values = values_by_key(make_situation(starting_stack=200.0, big_blind=2.0))
        assert values["stack_depth_norm"] == pytest.approx(0.5)

    def test_stack_depth_norm_clips_at_200bb(self):
        values = values_by_key(make_situation(starting_stack=1000.0, big_blind=2.0))
        assert values["stack_depth_norm"] == 1.0

    def test_stack_depth_norm_zero_at_zero_starting_stack(self):
        values = values_by_key(make_situation(starting_stack=0.0, big_blind=2.0))
        assert values["stack_depth_norm"] == 0.0

    def test_stack_depth_norm_unaffected_by_mid_hand_stack_changes(self):
        # Fixed for the whole hand -- a player who's already committed chips
        # this hand (my_stack < starting_stack) still reads the same Stack
        # Depth as at the start of the hand.
        at_start = values_by_key(make_situation(my_stack=200.0, starting_stack=200.0, big_blind=2.0))
        mid_hand = values_by_key(make_situation(my_stack=50.0, starting_stack=200.0, big_blind=2.0))
        assert at_start["stack_depth_norm"] == mid_hand["stack_depth_norm"] == pytest.approx(0.5)

    def test_num_active_norm(self):
        values = values_by_key(make_situation(num_active=3))
        assert values["num_active_norm"] == pytest.approx(0.5)

    def test_starting_position_norm_matches_seat_role(self):
        # button_idx=0, n=6: seat 2 is BB (see seating.blind_indices) --
        # the last role in SEAT_ROLES, so its normalized index is 1.0.
        bb = values_by_key(make_situation(seat_index=2, button_idx=0, num_seats_total=6))
        assert bb["starting_position_norm"] == pytest.approx(1.0)

        # seat 3 is UTG -- the first role, normalized index 0.0.
        utg = values_by_key(make_situation(seat_index=3, button_idx=0, num_seats_total=6))
        assert utg["starting_position_norm"] == pytest.approx(0.0)
        assert len(SEAT_ROLES) == 6


class TestFlopTexture:
    def test_preflop_defaults_are_all_zero(self):
        values = values_by_key(make_situation(board=[]))
        for key in ["flop_suit_texture_norm", "flop_pairing_texture_norm",
                    "flop_connectivity_norm", "connected_flop", "oesd_possible_flop",
                    "flop_wetness_norm", "flop_dynamism_norm"]:
            assert values[key] == 0.0

    def test_rainbow_unpaired_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_suit_texture_norm"] == 0.0
        assert values["flop_pairing_texture_norm"] == 0.0

    def test_monotone_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_suit_texture_norm"] == 1.0

    def test_two_tone_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_suit_texture_norm"] == pytest.approx(0.5)

    def test_paired_flop(self):
        board = [Card.from_str("2c"), Card.from_str("2d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_pairing_texture_norm"] == pytest.approx(0.5)

    def test_tripled_flop(self):
        board = [Card.from_str("2c"), Card.from_str("2d"), Card.from_str("2h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_pairing_texture_norm"] == 1.0

    def test_connected_flop_within_span_4(self):
        board = [Card.from_str("5c"), Card.from_str("7d"), Card.from_str("9h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["connected_flop"] == 1.0

    def test_disconnected_flop_wide_span(self):
        board = [Card.from_str("2c"), Card.from_str("8d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["connected_flop"] == 0.0
        assert values["flop_connectivity_norm"] == 0.0

    def test_oesd_possible_needs_unpaired_and_tight_span(self):
        board = [Card.from_str("5c"), Card.from_str("6d"), Card.from_str("7h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["oesd_possible_flop"] == 1.0

    def test_suit_family_true_regardless_of_hole_card_connection(self):
        # flop_suit_texture_norm is now purely the flop's own 3-card suit
        # shape -- no hole-card awareness at all (see TestSuitConnectionIndex
        # for that dimension).
        board = [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("Kh")]
        no_match = values_by_key(make_situation(hole=[Card.from_str("9s"), Card.from_str("2s")], board=board))
        two_match = values_by_key(make_situation(hole=[Card.from_str("9c"), Card.from_str("9d")], board=board))
        assert no_match["flop_suit_texture_norm"] == two_match["flop_suit_texture_norm"] == 0.0

    def test_monotone_hole_card_hit_still_reads_a_made_flush_elsewhere(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        one = values_by_key(make_situation(hole=[Card.from_str("9c"), Card.from_str("2s")], board=board))
        two = values_by_key(make_situation(hole=[Card.from_str("9c"), Card.from_str("4c")], board=board))
        assert one["suit_connection_index"] == pytest.approx(4 / 5)  # flush draw: 4 cards of one suit
        # Both hole cards + all 3 board cards = a made flush -- King high
        # (the board's own Kc), since that's the flush's own highest card.
        assert two["hand_category_norm"] == pytest.approx(21 / 25)  # King High Flush

    def test_connected_flop_without_a_straight_draw(self):
        board = [Card.from_str("5c"), Card.from_str("7d"), Card.from_str("9h")]
        values = values_by_key(make_situation(hole=[Card.from_str("2s"), Card.from_str("3s")], board=board))
        assert values["flop_connectivity_norm"] == pytest.approx(0.5)
        assert values["connected_flop"] == 1.0  # family aggregate still true

    def test_connected_flop_with_a_straight_draw(self):
        board = [Card.from_str("5c"), Card.from_str("7d"), Card.from_str("9h")]
        values = values_by_key(make_situation(hole=[Card.from_str("6s"), Card.from_str("2d")], board=board))
        assert values["flop_connectivity_norm"] == 1.0
        assert values["straight_draw_norm"] > 0.0

    def test_dry_static_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("Kh")]  # rainbow, disconnected
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_wetness_norm"] == 0.0
        assert values["flop_dynamism_norm"] == 0.0

    def test_two_tone_connected_flop_is_wet_and_dynamic(self):
        board = [Card.from_str("9h"), Card.from_str("8h"), Card.from_str("7c")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_wetness_norm"] == 1.0
        assert values["flop_dynamism_norm"] == 1.0

    def test_monotone_flop_is_wet_even_when_disconnected(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_wetness_norm"] == 1.0
        # King on board (>=Q) pulls a borderline wetness score below the
        # dynamism threshold -- see test_high_card_pins_borderline_wet_flop_to_static.
        assert values["flop_dynamism_norm"] == 0.0

    def test_high_card_pins_borderline_wet_flop_to_static(self):
        # Rainbow, connected, straight-draw-possible -> wetness score 2 (Wet),
        # but the King on board applies a -1 dynamism adjustment, dropping it
        # below the >=2 dynamism threshold.
        board = [Card.from_str("Kc"), Card.from_str("Qd"), Card.from_str("Jh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_wetness_norm"] == 1.0
        assert values["flop_dynamism_norm"] == 0.0

    def test_strongly_wet_high_board_stays_dynamic(self):
        # Wetness score 3 is high enough to survive the -1 high-card penalty.
        board = [Card.from_str("Qc"), Card.from_str("Jc"), Card.from_str("Th")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_dynamism_norm"] == 1.0

    def test_low_card_board_promotes_borderline_dry_flop_to_dynamic(self):
        # Rainbow, connected (span 4), no OESD -> wetness score 1 (Dry), but
        # every card is 8 or below, applying a +1 dynamism adjustment that
        # reaches the >=2 dynamism threshold despite being Dry.
        board = [Card.from_str("2c"), Card.from_str("4d"), Card.from_str("6h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_wetness_norm"] == 0.0
        assert values["flop_dynamism_norm"] == 1.0

    def test_low_card_bonus_does_not_rescue_a_fully_dry_flop(self):
        board = [Card.from_str("2c"), Card.from_str("7d"), Card.from_str("8h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_wetness_norm"] == 0.0
        assert values["flop_dynamism_norm"] == 0.0

    def test_ace_high_board_gets_the_high_card_penalty_not_the_low_card_bonus(self):
        board = [Card.from_str("Ac"), Card.from_str("4d"), Card.from_str("6h")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_dynamism_norm"] == 0.0

    def test_paired_flop_is_static_regardless_of_wetness(self):
        board = [Card.from_str("2c"), Card.from_str("2d"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_dynamism_norm"] == 0.0

    def test_tripled_flop_is_static(self):
        board = [Card.from_str("Kc"), Card.from_str("Kd"), Card.from_str("Kh")]
        values = values_by_key(make_situation(board=board, street=1))
        assert values["flop_dynamism_norm"] == 0.0

    def test_wetness_and_dynamism_frozen_on_turn_and_river(self):
        board3 = [Card.from_str("9h"), Card.from_str("8h"), Card.from_str("7c")]
        flop_values = values_by_key(make_situation(board=board3, street=1))
        board4 = board3 + [Card.from_str("2d")]
        turn_values = values_by_key(make_situation(board=board4, street=2))
        assert turn_values["flop_wetness_norm"] == flop_values["flop_wetness_norm"] == 1.0
        assert turn_values["flop_dynamism_norm"] == flop_values["flop_dynamism_norm"] == 1.0

    def test_flop_texture_frozen_on_turn_and_river(self):
        board3 = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]  # monotone
        flop_values = values_by_key(make_situation(board=board3, street=1))
        board4 = board3 + [Card.from_str("9d")]  # turn card breaks the monotone look
        turn_values = values_by_key(make_situation(board=board4, street=2))
        assert turn_values["flop_suit_texture_norm"] == flop_values["flop_suit_texture_norm"] == 1.0


class TestSuitConnectionIndex:
    def test_preflop_unsuited_hole_cards(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kd")], board=[]))
        assert values["suit_connection_index"] == pytest.approx(1 / 5)

    def test_preflop_suited_hole_cards(self):
        values = values_by_key(make_situation(hole=[Card.from_str("Ah"), Card.from_str("Kh")], board=[]))
        assert values["suit_connection_index"] == pytest.approx(2 / 5)

    def test_monotone_flop_alone_reaches_3_even_with_unrelated_hole_cards(self):
        # Pigeonhole: 5 cards (2 hole + 3 board) split across 4 suits always
        # gives some suit a count of at least 2 -- here the board's own 3
        # matching clubs already put it at 3, before the hole cards even
        # enter into it.
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        values = values_by_key(make_situation(hole=[Card.from_str("9d"), Card.from_str("2s")], board=board))
        assert values["suit_connection_index"] == pytest.approx(3 / 5)

    def test_one_hole_card_matching_the_flops_suit_reaches_4(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        values = values_by_key(make_situation(hole=[Card.from_str("9c"), Card.from_str("2s")], board=board))
        assert values["suit_connection_index"] == pytest.approx(4 / 5)

    def test_both_hole_cards_matching_a_monotone_flop_is_capped_at_5(self):
        board = [Card.from_str("2c"), Card.from_str("7c"), Card.from_str("Kc")]
        values = values_by_key(make_situation(hole=[Card.from_str("9c"), Card.from_str("4c")], board=board))
        assert values["suit_connection_index"] == 1.0

    def test_varies_through_streets_unlike_flop_suit_texture_norm(self):
        hole = [Card.from_str("Ah"), Card.from_str("Kd")]
        board3 = [Card.from_str("2h"), Card.from_str("7h"), Card.from_str("Qc")]  # two-tone: 2 hearts + Ah = 3
        flop_values = values_by_key(make_situation(hole=hole, board=board3, street=1))
        board4 = board3 + [Card.from_str("9h")]  # a 3rd board heart pushes the count to 4
        turn_values = values_by_key(make_situation(hole=hole, board=board4, street=2))

        assert flop_values["suit_connection_index"] == pytest.approx(3 / 5)
        assert turn_values["suit_connection_index"] == pytest.approx(4 / 5)
        # flop_suit_texture_norm stays frozen at the flop's own shape
        # (two-tone), unaffected by the turn card adding a 3rd heart.
        assert flop_values["flop_suit_texture_norm"] == turn_values["flop_suit_texture_norm"] == 0.5


class TestHandCategoryPairBuckets:
    """Pair-strength-vs-board classification is a set of mutually exclusive
    hand_category_norm sub-buckets (see _HAND_CATEGORY_VALUES) -- e.g. Low
    Pair and Underpair are two distinct buckets, not overlapping booleans."""

    def test_top_pair_weak_kicker(self):
        hole = [Card.from_str("Kh"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(8 / 25)  # Top Pair

    def test_top_pair_good_kicker(self):
        hole = [Card.from_str("Kh"), Card.from_str("Jd")]  # Jack kicker
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(9 / 25)  # Top Pair + Good Kicker

    def test_top_pair_top_kicker(self):
        hole = [Card.from_str("Kh"), Card.from_str("Ad")]  # Ace kicker
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(10 / 25)  # Top Pair + Top Kicker

    def test_second_pair(self):
        hole = [Card.from_str("7h"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(7 / 25)  # Second Pair

    def test_third_pair(self):
        hole = [Card.from_str("3h"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(6 / 25)  # Third Pair

    def test_overpair(self):
        hole = [Card.from_str("Kh"), Card.from_str("Kd")]
        board = [Card.from_str("7c"), Card.from_str("4d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(11 / 25)  # Overpair

    def test_low_pair(self):
        # 4 is below every board card -- Low Pair, not Underpair.
        hole = [Card.from_str("4h"), Card.from_str("4d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("5h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(3 / 25)  # Low Pair

    def test_underpair_not_low_pair(self):
        # pair (6) is below the top card (K) but above the bottom card (3).
        hole = [Card.from_str("6h"), Card.from_str("6d")]
        board = [Card.from_str("Kc"), Card.from_str("5d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["hand_category_norm"] == pytest.approx(4 / 25)  # Underpair

    def test_preflop_pocket_pair_is_the_generic_pair_bucket(self):
        values = values_by_key(make_situation(hole=[Card.from_str("6h"), Card.from_str("6d")], board=[]))
        assert values["hand_category_norm"] == pytest.approx(5 / 25)  # Pair


class TestHandVsBoardHeuristics:
    def test_nuts_flush_draw(self):
        hole = [Card.from_str("Ah"), Card.from_str("9h")]
        board = [Card.from_str("7h"), Card.from_str("4h"), Card.from_str("2c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["suit_connection_index"] == pytest.approx(4 / 5)  # flush draw: 4 cards of one suit
        assert values["nuts_flush_draw"] == 1.0

    def test_flush_draw_without_the_nut_card(self):
        hole = [Card.from_str("Qh"), Card.from_str("9h")]
        board = [Card.from_str("7h"), Card.from_str("4h"), Card.from_str("2c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["suit_connection_index"] == pytest.approx(4 / 5)  # flush draw: 4 cards of one suit
        assert values["nuts_flush_draw"] == 0.0

    def test_open_ended_straight_draw_is_the_top_bucket(self):
        hole = [Card.from_str("6h"), Card.from_str("7d")]
        board = [Card.from_str("8c"), Card.from_str("9s"), Card.from_str("2h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["straight_draw_norm"] == pytest.approx(1.0)

    def test_gutshot_is_the_middle_bucket(self):
        hole = [Card.from_str("6h"), Card.from_str("7d")]
        board = [Card.from_str("8c"), Card.from_str("Ts"), Card.from_str("2h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["straight_draw_norm"] == pytest.approx(0.5)

    def test_no_straight_draw_is_the_bottom_bucket(self):
        hole = [Card.from_str("2h"), Card.from_str("2d")]
        board = [Card.from_str("9c"), Card.from_str("5s"), Card.from_str("Kh")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["straight_draw_norm"] == 0.0

    def test_combo_draw(self):
        hole = [Card.from_str("6h"), Card.from_str("7h")]
        board = [Card.from_str("8h"), Card.from_str("9h"), Card.from_str("2c")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["suit_connection_index"] == pytest.approx(4 / 5)  # flush draw: 4 cards of one suit
        assert values["straight_draw_norm"] > 0.0
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


class TestOvercardsFeature:
    def test_zero_overcards_preflop(self):
        # No board yet, so nothing can rank higher than the hole cards.
        values = values_by_key(make_situation(
            hole=[Card.from_str("Kh"), Card.from_str("2d")], board=[],
        ))
        assert values["num_overcards_norm"] == 0.0

    def test_counts_board_cards_ranked_above_the_hole_high_card(self):
        # Hole high card is 9; board has two cards above it (K, T) and one below (4).
        hole = [Card.from_str("9h"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("Ts"), Card.from_str("4h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["num_overcards_norm"] == pytest.approx(2 / 5.0)

    def test_pocket_pair_counts_overcards_above_the_pair(self):
        # An underpair's overcard count should match the board cards above it.
        hole = [Card.from_str("6h"), Card.from_str("6d")]
        board = [Card.from_str("Kc"), Card.from_str("5d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["num_overcards_norm"] == pytest.approx(1 / 5.0)

    def test_no_overcards_when_hole_high_card_beats_the_whole_board(self):
        hole = [Card.from_str("Ah"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("Qd"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["num_overcards_norm"] == 0.0

    def test_river_can_reach_max_of_five_overcards(self):
        hole = [Card.from_str("2h"), Card.from_str("3d")]
        board = [
            Card.from_str("Kc"), Card.from_str("Qd"), Card.from_str("Jh"),
            Card.from_str("Ts"), Card.from_str("9c"),
        ]
        values = values_by_key(make_situation(hole=hole, board=board, street=3))
        assert values["num_overcards_norm"] == 1.0

    def test_equal_rank_board_card_does_not_count_as_an_overcard(self):
        # Board card matching the hole high card exactly is a pair, not an overcard.
        hole = [Card.from_str("Kh"), Card.from_str("2d")]
        board = [Card.from_str("Kc"), Card.from_str("7d"), Card.from_str("3h")]
        values = values_by_key(make_situation(hole=hole, board=board, street=1))
        assert values["num_overcards_norm"] == 0.0


class TestOpponentTendencyFeatures:
    def test_defaults_are_neutral(self):
        values = values_by_key(make_situation())
        assert values["opp_vpip_norm"] == 0.5

    def test_passes_through_situation_values(self):
        values = values_by_key(make_situation(
            opp_vpip=0.8, opp_pfr=0.3,
        ))
        assert values["opp_vpip_norm"] == pytest.approx(0.8)
        assert values["opp_pfr_norm"] == pytest.approx(0.3)

    def test_opponent_sample_size_feature_was_removed(self):
        # "doesn't interact with any other features" -- dropped as a genome
        # feature entirely, not just left unused.
        assert "opp_sample_norm" not in FEATURE_NAMES

    def test_villain_features_were_removed(self):
        # "Current Aggressor" reads were redundant with the table-average
        # versions (and, for 3-bet %, incoherent -- the current aggressor
        # can't have 3-bet before the player evaluating this feature) --
        # dropped entirely, not just left unused.
        assert "villain_three_bet_norm" not in FEATURE_NAMES
        assert "villain_fold_vs_bet_norm" not in FEATURE_NAMES
        assert "villain_aggression_freq_norm" not in FEATURE_NAMES

    def test_clips_out_of_range_values(self):
        values = values_by_key(make_situation(opp_vpip=1.5))
        assert values["opp_vpip_norm"] == 1.0


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
