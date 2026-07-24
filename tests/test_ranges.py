import pytest

from cards import Card
from ranges import ALL_HAND_LABELS, hand_label, in_range, parse_range


class TestHandLabel:
    def test_pocket_pair(self):
        assert hand_label(Card.from_str("9h"), Card.from_str("9d")) == "99"

    def test_suited(self):
        assert hand_label(Card.from_str("Ah"), Card.from_str("Kh")) == "AKs"

    def test_offsuit(self):
        assert hand_label(Card.from_str("Ah"), Card.from_str("Kd")) == "AKo"

    def test_high_card_always_listed_first(self):
        assert hand_label(Card.from_str("Kd"), Card.from_str("Ah")) == "AKo"


class TestAllHandLabels:
    def test_has_169_canonical_hands(self):
        assert len(ALL_HAND_LABELS) == 169

    def test_contains_expected_examples(self):
        for label in ["AA", "72o", "72s", "AKs", "AKo"]:
            assert label in ALL_HAND_LABELS


class TestParseRangeTokens:
    def test_exact_pair(self):
        assert parse_range("77") == {"77"}

    def test_pair_range(self):
        assert parse_range("AA-77") == {"77", "88", "99", "TT", "JJ", "QQ", "KK", "AA"}

    def test_pair_plus(self):
        assert parse_range("QQ+") == {"QQ", "KK", "AA"}

    def test_exact_suited_and_offsuit(self):
        assert parse_range("AKs") == {"AKs"}
        assert parse_range("AKo") == {"AKo"}

    def test_suited_plus(self):
        assert parse_range("AJs+") == {"AJs", "AQs", "AKs"}

    def test_offsuit_plus(self):
        assert parse_range("AJo+") == {"AJo", "AQo", "AKo"}

    def test_fixed_top_card_range(self):
        assert parse_range("AJs-A5s") == {"AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s"}

    def test_matching_gap_connector_range(self):
        assert parse_range("T9s-54s") == {"T9s", "98s", "87s", "76s", "65s", "54s"}

    def test_comma_separated_tokens_combine(self):
        assert parse_range("AA, KK, AKs") == {"AA", "KK", "AKs"}

    def test_whitespace_insensitive(self):
        assert parse_range("  AA ,KK ,   AKs") == {"AA", "KK", "AKs"}

    def test_case_insensitive(self):
        assert parse_range("aks") == {"AKs"}
        assert parse_range("aa") == {"AA"}

    def test_empty_string_gives_empty_range(self):
        assert parse_range("") == frozenset()

    def test_trailing_comma_ignored(self):
        assert parse_range("AA,") == {"AA"}

    def test_mismatched_suited_offsuit_range_raises(self):
        with pytest.raises(ValueError):
            parse_range("AJs-A5o")

    def test_mismatched_gap_connector_range_raises(self):
        with pytest.raises(ValueError):
            parse_range("T9s-72s")

    def test_garbage_token_raises(self):
        with pytest.raises(ValueError):
            parse_range("XYZ")

    def test_result_is_subset_of_all_hand_labels(self):
        result = parse_range("AA-22, AKs-32s, AKo-32o")
        assert result <= ALL_HAND_LABELS


class TestInRange:
    def test_hand_in_range(self):
        range_set = parse_range("AA, KK, AKs")
        assert in_range(Card.from_str("Ah"), Card.from_str("Ad"), range_set) is True

    def test_hand_not_in_range(self):
        range_set = parse_range("AA, KK, AKs")
        assert in_range(Card.from_str("2h"), Card.from_str("7d"), range_set) is False

    def test_suitedness_matters(self):
        range_set = parse_range("AKs")
        assert in_range(Card.from_str("Ah"), Card.from_str("Kh"), range_set) is True
        assert in_range(Card.from_str("Ah"), Card.from_str("Kd"), range_set) is False
