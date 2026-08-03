import strategy


class TestRaiseActionCategories:
    def test_every_raise_action_has_a_pot_fraction(self):
        for action in strategy.RAISE_ACTIONS:
            assert action in strategy.RAISE_POT_FRACTION

    def test_fold_call_allin_are_not_raise_actions(self):
        for action in (strategy.ACTION_FOLD, strategy.ACTION_CALL, strategy.ACTION_ALLIN):
            assert action not in strategy.RAISE_ACTIONS
            assert action not in strategy.RAISE_POT_FRACTION

    def test_pot_fractions_ascend_with_action_index(self):
        fractions = [strategy.RAISE_POT_FRACTION[a] for a in strategy.RAISE_ACTIONS]
        assert fractions == sorted(fractions)

    def test_action_categories_length_matches_num_action_categories(self):
        assert len(strategy.ACTION_CATEGORIES) == strategy.NUM_ACTION_CATEGORIES
        assert strategy.NUM_ACTION_CATEGORIES == 2 + len(strategy.RAISE_ACTIONS) + 1


class TestActionLabels:
    def test_fold_label_reads_check_fold_give_up(self):
        assert strategy.ACTION_CATEGORIES[strategy.ACTION_FOLD] == "Check / Fold (Give up)"
