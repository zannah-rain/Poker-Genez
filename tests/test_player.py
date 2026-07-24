import numpy as np

from genome import Genome
from player import Player


def make_genome():
    rng = np.random.default_rng(0)
    return Genome.random(rng)


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
