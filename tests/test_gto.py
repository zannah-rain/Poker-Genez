"""Covers ranges.py's chart-string parsing and gto.py's SpotMatcher/
GTOSpot resolution -- see cfr_tree.py/cfr_policy.py's own integration
points for how a matched spot's fixed action actually short-circuits a
real decision (cross-validated there, not here)."""

import pytest

import gto
import rules
import strategy
from cards import Card
from features import Situation
from ranges import hand_label, parse_range


def test_parse_range_pair_plus():
    assert parse_range("77+") == {"77", "88", "99", "TT", "JJ", "QQ", "KK", "AA"}


def test_parse_range_pair_range():
    assert parse_range("AA-QQ") == {"AA", "KK", "QQ"}


def test_parse_range_suited_plus():
    assert parse_range("AJs+") == {"AJs", "AQs", "AKs"}


def test_parse_range_fixed_top_card_range():
    assert parse_range("AJs-A9s") == {"AJs", "ATs", "A9s"}


def test_parse_range_connector_range():
    assert parse_range("T9s-87s") == {"T9s", "98s", "87s"}


def test_parse_range_comma_list():
    assert parse_range("AA, KK, AKo") == {"AA", "KK", "AKo"}


def test_hand_label_ignores_exact_suit():
    assert hand_label(Card.from_str("Ah"), Card.from_str("Kh")) == "AKs"
    assert hand_label(Card.from_str("Ah"), Card.from_str("Kd")) == "AKo"
    assert hand_label(Card.from_str("7c"), Card.from_str("7d")) == "77"


def test_parse_action_token_valid_and_invalid():
    assert gto.parse_action_token("fold").category == strategy.ACTION_FOLD
    assert gto.parse_action_token("raise_150").category == strategy.ACTION_RAISE_150
    assert gto.parse_action_token("ALLIN").category == strategy.ACTION_ALLIN
    with pytest.raises(ValueError):
        gto.parse_action_token("raise_999")


def test_parse_action_token_fixed_bb_raise():
    assert gto.parse_action_token("raise_1.5bb").fixed_raise_bb == pytest.approx(1.5)
    assert gto.parse_action_token("raise_3BB").fixed_raise_bb == pytest.approx(3.0)
    with pytest.raises(ValueError):
        gto.parse_action_token("raise_bb")


def _situation(**overrides) -> Situation:
    defaults = dict(
        hole=[Card.from_str("Ah"), Card.from_str("Kh")],
        board=[],
        street=0,
        pot=3.0,
        call_amount=0.0,
        my_stack=200.0,
        effective_stack=200.0,
        position=0,
        num_seats_this_street=6,
        seat_index=0,
        button_idx=3,
        num_seats_total=6,
        num_active=6,
        num_raises_this_street=0,
        num_raises_previous_street=0,
        num_raises_preflop=0,
        num_raises_flop=0,
        num_raises_turn=0,
        is_aggressor_previous_street=False,
        is_aggressor_preflop=False,
        is_aggressor_flop=False,
        is_aggressor_turn=False,
        starting_stack=200.0,
        big_blind=2.0,
    )
    defaults.update(overrides)
    return Situation(**defaults)


def test_spot_matcher_matches_every_declared_field():
    matcher = gto.SpotMatcher(
        street=0, pot_type=0, position="UTG", facing_bet=False, min_effective_bb=80, max_effective_bb=120,
    )
    # seat_index=0, button_idx=3, num_seats_total=6 -> UTG (see seating.seat_role).
    situation = _situation(seat_index=0, button_idx=3, num_seats_total=6, effective_stack=200.0, big_blind=2.0)
    assert matcher.matches(situation)


def test_spot_matcher_rejects_wrong_position():
    matcher = gto.SpotMatcher(position="BTN")
    situation = _situation(seat_index=0, button_idx=3, num_seats_total=6)  # UTG, not BTN
    assert not matcher.matches(situation)


def test_spot_matcher_rejects_out_of_range_stack_depth():
    matcher = gto.SpotMatcher(min_effective_bb=80, max_effective_bb=120)
    situation = _situation(effective_stack=40.0, big_blind=2.0)  # 20BB effective
    assert not matcher.matches(situation)


def test_resolve_spot_action_uses_range_then_default():
    spot = gto.GTOSpot(
        key="test_spot", label="Test", matcher=gto.SpotMatcher(street=0),
        action_ranges=(("raise_150", "AA, KK"),), default_action="fold",
    )
    aa_situation = _situation(hole=[Card.from_str("Ah"), Card.from_str("As")])
    aa_decision = gto.resolve_spot_action(spot, aa_situation)
    assert aa_decision == rules.Decision("raise", strategy.RAISE_POT_FRACTION[strategy.ACTION_RAISE_150] * max(aa_situation.pot, 1.0))

    trash_situation = _situation(hole=[Card.from_str("7c"), Card.from_str("2d")])
    assert gto.resolve_spot_action(spot, trash_situation) == rules.Decision("fold")


def test_resolve_spot_action_fixed_bb_raise_ignores_pot_size():
    spot = gto.GTOSpot(
        key="test_spot", label="Test", matcher=gto.SpotMatcher(street=0),
        action_ranges=(), default_action="raise_1.5bb",
    )
    small_pot = gto.resolve_spot_action(spot, _situation(pot=3.0, big_blind=2.0))
    huge_pot = gto.resolve_spot_action(spot, _situation(pot=300.0, big_blind=2.0))
    assert small_pot == huge_pot == rules.Decision("raise", 3.0)  # 1.5 * 2.0 big blind, regardless of pot


def test_resolve_spot_action_none_when_matcher_rejects():
    spot = gto.GTOSpot(
        key="test_spot", label="Test", matcher=gto.SpotMatcher(street=3),  # river only
        action_ranges=(("call", "AA+"),),
    )
    situation = _situation(street=0)  # preflop
    assert gto.resolve_spot_action(spot, situation) is None


def test_first_matching_action_stops_at_first_applicable_spot():
    never_matches = gto.GTOSpot(
        key="never", label="Never", matcher=gto.SpotMatcher(street=3), action_ranges=(), default_action="call",
    )
    always_matches = gto.GTOSpot(
        key="always", label="Always", matcher=gto.SpotMatcher(street=0), action_ranges=(), default_action="fold",
    )
    situation = _situation(street=0)
    assert gto.first_matching_action((never_matches, always_matches), situation) == rules.Decision("fold")


def test_first_matching_action_none_when_nothing_applies():
    spot = gto.GTOSpot(key="river_only", label="River", matcher=gto.SpotMatcher(street=3), action_ranges=(), default_action="call")
    situation = _situation(street=0)
    assert gto.first_matching_action((spot,), situation) is None
