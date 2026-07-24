import pytest

from cards import Card
from features import Situation
from gto import (
    GTO_SPOTS, NUM_GTO_SPOTS, GTOSpot, SpotMatcher, parse_action_token,
    resolve_spot_action,
)
from ranges import ALL_HAND_LABELS


def make_situation(**overrides) -> Situation:
    defaults = dict(
        hole=[Card.from_str("Ah"), Card.from_str("Kd")],
        board=[],
        street=0,
        pot=10.0,
        call_amount=2.0,
        my_stack=200.0,
        effective_stack=200.0,
        position=0,
        num_seats_this_street=6,
        seat_index=3,  # UTG when button_idx=0, n=6
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


class TestSpotMatcherMatches:
    def test_no_criteria_matches_anything(self):
        assert SpotMatcher().matches(make_situation()) is True

    def test_street_must_match(self):
        matcher = SpotMatcher(street=0)
        assert matcher.matches(make_situation(street=0)) is True
        assert matcher.matches(make_situation(street=1)) is False

    def test_pot_type_uses_clipped_preflop_raise_count(self):
        matcher = SpotMatcher(pot_type=3)
        assert matcher.matches(make_situation(num_preflop_raises=3)) is True
        assert matcher.matches(make_situation(num_preflop_raises=10)) is True  # clipped to 3
        assert matcher.matches(make_situation(num_preflop_raises=2)) is False

    def test_position_checks_seat_role(self):
        matcher = SpotMatcher(position="UTG")
        assert matcher.matches(make_situation(seat_index=3, button_idx=0, num_seats_total=6)) is True
        assert matcher.matches(make_situation(seat_index=2, button_idx=0, num_seats_total=6)) is False

    def test_is_aggressor_flag(self):
        matcher = SpotMatcher(is_aggressor=True)
        assert matcher.matches(make_situation(is_aggressor=True)) is True
        assert matcher.matches(make_situation(is_aggressor=False)) is False

    def test_facing_bet_flag(self):
        matcher = SpotMatcher(facing_bet=True)
        assert matcher.matches(make_situation(call_amount=5.0)) is True
        assert matcher.matches(make_situation(call_amount=0.0)) is False

    def test_raised_positions_must_match_exactly(self):
        matcher = SpotMatcher(raised_positions=("UTG",))
        assert matcher.matches(make_situation(raised_positions=frozenset({"UTG"}))) is True
        # Extra raiser not covered by the matcher -> no match.
        assert matcher.matches(make_situation(raised_positions=frozenset({"UTG", "CO"}))) is False
        # Missing the required raiser -> no match.
        assert matcher.matches(make_situation(raised_positions=frozenset())) is False

    def test_effective_bb_range(self):
        matcher = SpotMatcher(min_effective_bb=80, max_effective_bb=120)
        assert matcher.matches(make_situation(effective_stack=200.0, big_blind=2.0)) is True  # 100bb
        assert matcher.matches(make_situation(effective_stack=100.0, big_blind=2.0)) is False  # 50bb
        assert matcher.matches(make_situation(effective_stack=300.0, big_blind=2.0)) is False  # 150bb

    def test_call_bb_range(self):
        matcher = SpotMatcher(min_call_bb=1.0, max_call_bb=3.0)
        assert matcher.matches(make_situation(call_amount=4.0, big_blind=2.0)) is True  # 2bb
        assert matcher.matches(make_situation(call_amount=10.0, big_blind=2.0)) is False  # 5bb

    def test_all_criteria_must_agree(self):
        matcher = SpotMatcher(street=0, position="UTG", pot_type=0)
        good = make_situation(street=0, seat_index=3, button_idx=0, num_seats_total=6, num_preflop_raises=0)
        bad = make_situation(street=0, seat_index=3, button_idx=0, num_seats_total=6, num_preflop_raises=1)
        assert matcher.matches(good) is True
        assert matcher.matches(bad) is False


class TestSpotMatcherDescribe:
    def test_no_criteria_describes_as_any_situation(self):
        assert SpotMatcher().describe() == "any situation"

    def test_describes_street_and_position(self):
        text = SpotMatcher(street=0, position="UTG").describe()
        assert "Preflop" in text
        assert "UTG" in text

    def test_describes_effective_stack_range(self):
        text = SpotMatcher(min_effective_bb=80, max_effective_bb=120).describe()
        assert "80" in text and "120" in text


class TestParseActionToken:
    def test_fold(self):
        assert parse_action_token("fold") == ("fold", None)

    def test_call(self):
        assert parse_action_token("call") == ("call", None)

    def test_allin(self):
        assert parse_action_token("allin") == ("raise", None)

    def test_raise_bb_sizing(self):
        assert parse_action_token("raise_2.5bb") == ("raise", ("bb", 2.5))
        assert parse_action_token("raise_10bb") == ("raise", ("bb", 10.0))

    def test_raise_pot_fraction_sizing(self):
        assert parse_action_token("raise_75") == ("raise", ("pot", 0.75))
        assert parse_action_token("raise_100") == ("raise", ("pot", 1.0))

    def test_is_case_and_whitespace_insensitive(self):
        assert parse_action_token("  FOLD  ") == ("fold", None)
        assert parse_action_token("RAISE_2BB") == ("raise", ("bb", 2.0))

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError):
            parse_action_token("bogus")


class TestResolveSpotAction:
    def _spot(self):
        return GTOSpot(
            key="test_spot",
            label="Test Spot",
            matcher=SpotMatcher(street=0, position="UTG"),
            action_ranges=(("raise_2bb", "AA, KK, AKs"), ("call", "QQ, JJ")),
            default_action="fold",
        )

    def test_returns_none_when_matcher_does_not_match(self):
        spot = self._spot()
        situation = make_situation(street=1)  # wrong street
        assert resolve_spot_action(spot, situation) is None

    def test_hand_in_first_range_wins(self):
        spot = self._spot()
        situation = make_situation(hole=[Card.from_str("Ah"), Card.from_str("Ad")])
        assert resolve_spot_action(spot, situation) == ("raise", ("bb", 2.0))

    def test_hand_in_second_range(self):
        spot = self._spot()
        situation = make_situation(hole=[Card.from_str("Qh"), Card.from_str("Qd")])
        assert resolve_spot_action(spot, situation) == ("call", None)

    def test_hand_not_in_any_range_uses_default(self):
        spot = self._spot()
        situation = make_situation(hole=[Card.from_str("7h"), Card.from_str("2d")])
        assert resolve_spot_action(spot, situation) == ("fold", None)

    def test_first_matching_range_wins_when_overlapping(self):
        spot = GTOSpot(
            key="overlap",
            label="Overlap",
            matcher=SpotMatcher(),
            action_ranges=(("call", "AA"), ("raise_2bb", "AA")),
        )
        situation = make_situation(hole=[Card.from_str("Ah"), Card.from_str("Ad")])
        assert resolve_spot_action(spot, situation) == ("call", None)


class TestGtoSpotsCatalog:
    def test_num_gto_spots_matches_catalog_length(self):
        assert NUM_GTO_SPOTS == len(GTO_SPOTS)

    def test_keys_are_unique(self):
        keys = [spot.key for spot in GTO_SPOTS]
        assert len(keys) == len(set(keys))

    def test_every_spot_has_nonempty_parsed_ranges(self):
        for spot in GTO_SPOTS:
            assert len(spot.parsed_ranges) > 0

    def test_every_range_is_a_subset_of_canonical_hands(self):
        for spot in GTO_SPOTS:
            for _action, range_set in spot.parsed_ranges:
                assert range_set <= ALL_HAND_LABELS

    def test_every_action_token_parses(self):
        for spot in GTO_SPOTS:
            for action_token, _range_str in spot.action_ranges:
                parse_action_token(action_token)  # should not raise
            parse_action_token(spot.default_action)

    def test_every_spot_matcher_can_evaluate_a_situation(self):
        situation = make_situation()
        for spot in GTO_SPOTS:
            # Just verify calling matches() doesn't blow up on a real situation.
            assert isinstance(spot.matcher.matches(situation), bool)
