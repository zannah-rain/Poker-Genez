import pytest

from seating import SEAT_ROLES, blind_indices, preflop_order, seat_role


class TestBlindIndices:
    def test_heads_up_button_is_small_blind(self):
        assert blind_indices(0, 2) == (0, 1)
        assert blind_indices(1, 2) == (1, 0)

    def test_full_ring_blinds_are_seats_after_button(self):
        assert blind_indices(0, 6) == (1, 2)
        assert blind_indices(5, 6) == (0, 1)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_blind_seats_are_within_table(self, n):
        sb, bb = blind_indices(0, n)
        assert 0 <= sb < n
        assert 0 <= bb < n
        assert sb != bb


class TestPreflopOrder:
    def test_order_starts_after_big_blind_and_ends_on_blinds(self):
        order = preflop_order(0, 6)
        sb, bb = blind_indices(0, 6)
        assert order[-2:] == [sb, bb]
        assert order[0] == (bb + 1) % 6

    def test_order_is_a_permutation_of_all_seats(self):
        for n in [2, 3, 4, 5, 6]:
            order = preflop_order(0, n)
            assert sorted(order) == list(range(n))

    def test_heads_up_button_acts_first_preflop(self):
        # Heads-up, button==SB acts first preflop, BB last.
        order = preflop_order(0, 2)
        assert order == [0, 1]


class TestSeatRole:
    def test_heads_up_labels_button_as_sb(self):
        assert seat_role(0, 0, 2) == "SB"
        assert seat_role(1, 0, 2) == "BB"

    def test_six_max_labels_every_seat_uniquely(self):
        roles = [seat_role(i, 0, 6) for i in range(6)]
        assert sorted(roles) == sorted(SEAT_ROLES)

    def test_button_and_blinds_labeled_correctly(self):
        # button_idx=0 in 6-max: SB=1, BB=2, non-blind seats 3,4,5,0 -> UTG,HJ,CO,BTN
        assert seat_role(1, 0, 6) == "SB"
        assert seat_role(2, 0, 6) == "BB"
        assert seat_role(0, 0, 6) == "BTN"

    def test_four_handed_has_no_hijack_or_cutoff(self):
        roles = {seat_role(i, 0, 4) for i in range(4)}
        assert roles == {"UTG", "BTN", "SB", "BB"}

    def test_three_handed_roles(self):
        # 3-handed: 1 non-blind seat, labeled BTN (see _NON_BLIND_LABEL_SCHEMES).
        roles = [seat_role(i, 0, 3) for i in range(3)]
        assert sorted(roles) == sorted(["BTN", "SB", "BB"])

    def test_role_is_invariant_to_which_seat_is_button(self):
        # Rotating the button should just rotate which physical seat gets
        # which role, not change the *set* of roles dealt out.
        roles_btn0 = {seat_role(i, 0, 6) for i in range(6)}
        roles_btn3 = {seat_role(i, 3, 6) for i in range(6)}
        assert roles_btn0 == roles_btn3 == set(SEAT_ROLES)
