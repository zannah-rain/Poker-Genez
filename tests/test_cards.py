import random

import pytest

from cards import RANKS, SUITS, Card, Deck, make_deck


class TestCard:
    def test_from_str_parses_rank_and_suit(self):
        card = Card.from_str("Ah")
        assert card.rank == 14
        assert card.suit == SUITS.index("h")

    def test_from_str_is_case_insensitive(self):
        assert Card.from_str("ah") == Card.from_str("AH")

    def test_repr_round_trips_through_from_str(self):
        for s in ["2c", "Td", "Jh", "Qs", "Kc", "As"]:
            assert repr(Card.from_str(s)) == s

    def test_low_and_high_rank_values(self):
        assert Card.from_str("2c").rank == 2
        assert Card.from_str("Ac").rank == 14

    def test_cards_are_ordered_by_rank_then_suit(self):
        assert Card(5, 0) < Card(6, 0)
        assert Card(5, 0) < Card(5, 1)
        assert not (Card(5, 1) < Card(5, 0))

    def test_equal_cards_compare_equal(self):
        assert Card.from_str("Kh") == Card.from_str("Kh")
        assert Card.from_str("Kh") != Card.from_str("Ks")

    def test_card_is_hashable(self):
        # frozen dataclass -> usable in sets/dict keys
        assert len({Card.from_str("Kh"), Card.from_str("Kh"), Card.from_str("Ks")}) == 2


class TestMakeDeck:
    def test_has_52_unique_cards(self):
        deck = make_deck()
        assert len(deck) == 52
        assert len(set(deck)) == 52

    def test_covers_every_rank_and_suit_combination(self):
        deck = make_deck()
        ranks = {c.rank for c in deck}
        suits = {c.suit for c in deck}
        assert ranks == set(range(2, 15))
        assert suits == set(range(4))
        assert len(RANKS) == 13


class TestDeck:
    def test_deals_52_unique_cards_with_no_repeats(self):
        deck = Deck(rng=random.Random(0))
        dealt = []
        for _ in range(52):
            dealt.extend(deck.deal(1))
        assert len(dealt) == 52
        assert len(set(dealt)) == 52

    def test_deal_multiple_at_once_matches_dealing_one_at_a_time(self):
        deck_a = Deck(rng=random.Random(123))
        deck_b = Deck(rng=random.Random(123))
        combined = deck_a.deal(5)
        individually = [deck_b.deal(1)[0] for _ in range(5)]
        assert combined == individually

    def test_raises_when_not_enough_cards_left(self):
        deck = Deck(rng=random.Random(1))
        deck.deal(50)
        with pytest.raises(ValueError):
            deck.deal(3)

    def test_deals_exactly_remaining_cards_without_error(self):
        deck = Deck(rng=random.Random(1))
        deck.deal(50)
        assert len(deck.deal(2)) == 2

    def test_same_seed_produces_same_shuffle(self):
        deck_a = Deck(rng=random.Random(42))
        deck_b = Deck(rng=random.Random(42))
        assert deck_a.deal(52) == deck_b.deal(52)

    def test_different_seeds_produce_different_order_generally(self):
        deck_a = Deck(rng=random.Random(1))
        deck_b = Deck(rng=random.Random(2))
        assert deck_a.deal(52) != deck_b.deal(52)
