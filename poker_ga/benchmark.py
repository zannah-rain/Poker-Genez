"""Cross-generation benchmark: plays the current population head-to-head
against a saved checkpoint from N generations ago, in 3-vs-3 tables, to give
a tangible, apples-to-apples measure of whether evolution is actually
improving play. The per-generation fitness number isn't useful for this --
it only ranks genomes against that generation's own random opponents, so a
"fitness" of 500 at generation 10 and 500 at generation 50 aren't
comparable. A direct match against a fixed past checkpoint is."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game import GameConfig, SeatState, play_hand
from opponent_model import OpponentModel
from player import Player

SEATS_PER_SIDE = 3


@dataclass
class BenchmarkResult:
    current_net_total: float = 0.0
    checkpoint_net_total: float = 0.0
    current_hands_total: int = 0
    checkpoint_hands_total: int = 0

    def bb_per_100(self, side: str, big_blind: float) -> float:
        net = self.current_net_total if side == "current" else self.checkpoint_net_total
        hands = self.current_hands_total if side == "current" else self.checkpoint_hands_total
        if hands == 0:
            return 0.0
        return (net / big_blind) / hands * 100.0


def _play_side_match(
    current_pool: list[Player],
    checkpoint_pool: list[Player],
    game_config: GameConfig,
    rng: np.random.Generator,
) -> tuple[float, float, int, int]:
    """Plays one 3-vs-3 table to the hand cap. Any busted seat is refilled
    with a fresh player from its OWN side's pool (current or checkpoint), so
    the match stays a genuine 3v3 for the whole session instead of one side
    slowly being replaced by the other. Returns (current_net,
    checkpoint_net, current_hands, checkpoint_hands)."""
    current_idx = rng.choice(len(current_pool), size=SEATS_PER_SIDE, replace=False)
    checkpoint_idx = rng.choice(len(checkpoint_pool), size=SEATS_PER_SIDE, replace=False)
    seats = [SeatState(player=current_pool[i], stack=game_config.starting_stack) for i in current_idx] + [
        SeatState(player=checkpoint_pool[i], stack=game_config.starting_stack) for i in checkpoint_idx
    ]
    sides = ["current"] * SEATS_PER_SIDE + ["checkpoint"] * SEATS_PER_SIDE
    order = rng.permutation(len(seats))  # so "current" isn't always dealt the button first
    seats = [seats[i] for i in order]
    sides = [sides[i] for i in order]
    opp_model = OpponentModel()

    current_net = checkpoint_net = 0.0
    current_hands = checkpoint_hands = 0

    button_idx = 0
    hands_played = 0
    while hands_played < game_config.max_hands_per_session:
        for side in sides:
            if side == "current":
                current_hands += 1
            else:
                checkpoint_hands += 1

        play_hand(seats, button_idx % len(seats), game_config, rng, opp_model=opp_model)
        hands_played += 1
        button_idx = (button_idx + 1) % len(seats)

        for i, s in enumerate(seats):
            if s.stack <= 1e-9:
                if sides[i] == "current":
                    current_net -= game_config.starting_stack
                    replacement = current_pool[int(rng.integers(0, len(current_pool)))]
                else:
                    checkpoint_net -= game_config.starting_stack
                    replacement = checkpoint_pool[int(rng.integers(0, len(checkpoint_pool)))]
                seats[i] = SeatState(player=replacement, stack=game_config.starting_stack)

    for i, s in enumerate(seats):
        if sides[i] == "current":
            current_net += s.stack - game_config.starting_stack
        else:
            checkpoint_net += s.stack - game_config.starting_stack

    return current_net, checkpoint_net, current_hands, checkpoint_hands


def run_benchmark(
    current_players: list[Player],
    checkpoint_players: list[Player],
    game_config: GameConfig,
    rng: np.random.Generator,
    num_tables: int = 60,
) -> BenchmarkResult:
    """Plays `num_tables` independent 3-vs-3 tables, each seating a random
    sample of 3 from `current_players` against a random sample of 3 from
    `checkpoint_players`, and aggregates net chips + bb/100 for each side."""
    result = BenchmarkResult()
    for _ in range(num_tables):
        cur_net, chk_net, cur_hands, chk_hands = _play_side_match(
            current_players, checkpoint_players, game_config, rng
        )
        result.current_net_total += cur_net
        result.checkpoint_net_total += chk_net
        result.current_hands_total += cur_hands
        result.checkpoint_hands_total += chk_hands
    return result
