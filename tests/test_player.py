from player import Player


class _StubPolicy:
    """A minimal duck-typed decision-maker -- Player.genome only ever needs
    .decide(situation, legal_actions, rng), never a concrete type (see
    player.py, or tests/test_game.py's FixedGenome for a fuller example)."""

    def decide(self, situation, legal_actions, rng=None):
        return legal_actions[0], 0.0


def make_genome():
    return _StubPolicy()


class TestPlayer:
    def test_repr_uses_label_when_set(self):
        p = Player(player_id=1, genome=make_genome(), generation=3, label="Alice")
        assert repr(p) == "<Alice gen=3>"

    def test_repr_falls_back_to_player_id(self):
        p = Player(player_id=7, genome=make_genome(), generation=2)
        assert repr(p) == "<P7 gen=2>"

    def test_defaults(self):
        p = Player(player_id=1, genome=make_genome())
        assert p.generation == 0
        assert p.label == ""
