"""Orchestrates play: seats players at random 6-max tables, plays sessions
of hands until players bust or a hand cap is hit, and turns chip results
into fitness scores the GA can select on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game import GameConfig, SeatState, play_hand
from player import Player

TABLE_SIZE = 6


@dataclass
class SimConfig:
    rounds_per_generation: int = 3  # independent random re-seatings per generation
    table_size: int = TABLE_SIZE


def run_session(players: list[Player], game_config: GameConfig, rng: np.random.Generator) -> dict[int, float]:
    """Plays one session at one table until only one player has chips left
    or the hand cap is reached. A player is removed from the table the
    instant their stack hits zero (busted). Returns net chip result
    (final - starting) per player_id; busted players score -starting_stack."""
    seats = [SeatState(player=p, stack=game_config.starting_stack) for p in players]
    starting_ids = [p.player_id for p in players]

    button_idx = 0
    hands_played = 0
    while len(seats) > 1 and hands_played < game_config.max_hands_per_session:
        play_hand(seats, button_idx % len(seats), game_config, rng)
        hands_played += 1
        seats = [s for s in seats if s.stack > 1e-9]
        if seats:
            button_idx = (button_idx + 1) % len(seats)

    final_stack = {s.player.player_id: s.stack for s in seats}
    return {pid: final_stack.get(pid, 0.0) - game_config.starting_stack for pid in starting_ids}


def run_generation(
    players: list[Player],
    game_config: GameConfig,
    sim_config: SimConfig,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Runs `rounds_per_generation` rounds of random table re-seating across
    the whole population, accumulating each player's net chip result. This
    is the fitness signal the GA selects on."""
    total_fitness = {p.player_id: 0.0 for p in players}

    for _ in range(sim_config.rounds_per_generation):
        order = rng.permutation(len(players))
        shuffled = [players[i] for i in order]
        for start in range(0, len(shuffled), sim_config.table_size):
            table = shuffled[start : start + sim_config.table_size]
            if len(table) < 2:
                continue
            results = run_session(table, game_config, rng)
            for pid, delta in results.items():
                total_fitness[pid] += delta

    return total_fitness
