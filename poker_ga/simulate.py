"""Orchestrates play: seats players at random 6-max tables, plays sessions
of hands (refilling any seat that busts with a fresh player, rather than
ending the session early) until a hand cap is hit, and turns chip results
into fitness scores the GA can select on."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from game import GameConfig, HandStats, SeatState, play_hand
from player import Player

TABLE_SIZE = 6


@dataclass
class SimConfig:
    rounds_per_generation: int = 3  # independent random re-seatings per generation
    table_size: int = TABLE_SIZE


def run_session(
    table_players: list[Player],
    game_config: GameConfig,
    rng: np.random.Generator,
    backfill_pool: list[Player] | None = None,
    stats: HandStats | None = None,
) -> dict:
    """Plays one table's session to the hand cap. Any seat whose stack hits
    zero is immediately refilled with a fresh player (a full starting stack)
    drawn from `backfill_pool` (or `table_players` itself if no pool is
    given), instead of ending the session early -- models a real cash table,
    where the game keeps going as players bust and new ones sit down, rather
    than the whole table closing. Because seats never shrink, the button
    just rotates seat-by-seat every hand regardless of who currently
    occupies each seat.

    Returns a dict with:
      - "net": {player_id: net chip result} for every player who occupied a
        seat at any point this session (more than len(table_players) if any
        seat busted and was refilled). A player who busted and was later
        drawn as a refill elsewhere accumulates results across both stints.
      - "hands_survived": {player_id: hands participated in}
      - "busted": {player_id: True if that stint ended in a bust}
      - "winner_id": player_id with the largest stack at the final hand.
    """
    pool = backfill_pool if backfill_pool else table_players
    seats = [SeatState(player=p, stack=game_config.starting_stack) for p in table_players]

    net: dict[int, float] = {}
    hands_survived: dict[int, int] = {}
    busted: dict[int, bool] = {}

    button_idx = 0
    hands_played = 0
    while hands_played < game_config.max_hands_per_session:
        for s in seats:
            hands_survived[s.player.player_id] = hands_survived.get(s.player.player_id, 0) + 1

        play_hand(seats, button_idx % len(seats), game_config, rng, stats=stats)
        hands_played += 1
        button_idx = (button_idx + 1) % len(seats)

        for i, s in enumerate(seats):
            if s.stack <= 1e-9:
                pid = s.player.player_id
                net[pid] = net.get(pid, 0.0) - game_config.starting_stack
                busted[pid] = True
                replacement = pool[int(rng.integers(0, len(pool)))]
                seats[i] = SeatState(player=replacement, stack=game_config.starting_stack)

    for s in seats:
        pid = s.player.player_id
        net[pid] = net.get(pid, 0.0) + s.stack - game_config.starting_stack
        busted.setdefault(pid, False)
        # A seat refilled on the session's last hand never got a turn through
        # the hands_survived-counting loop above (that loop only runs before
        # a hand is played, and no more hands are played after this refill).
        hands_survived.setdefault(pid, 0)
    winner_id = max(seats, key=lambda s: s.stack).player.player_id

    return {"net": net, "hands_survived": hands_survived, "busted": busted, "winner_id": winner_id}


@dataclass
class GenerationStats:
    """Aggregate sense-check metrics across a whole generation's fitness
    pass -- not used for selection, just for spotting pathological drift
    (e.g. hands survived collapsing, fold rate collapsing) as it happens."""
    total_hands_survived: int = 0
    total_session_participations: int = 0
    total_busts: int = 0
    hand_stats: HandStats = field(default_factory=HandStats)

    @property
    def mean_hands_survived(self) -> float:
        return self.total_hands_survived / max(self.total_session_participations, 1)

    @property
    def bust_rate(self) -> float:
        return self.total_busts / max(self.total_session_participations, 1)

    @property
    def fold_rate_facing_bet(self) -> float:
        return self.hand_stats.facing_bet_folds / max(self.hand_stats.facing_bet_decisions, 1)

    @property
    def mean_raises_per_street(self) -> float:
        raises = self.hand_stats.raises_per_street
        return float(np.mean(raises)) if raises else 0.0


def run_generation(
    players: list[Player],
    game_config: GameConfig,
    sim_config: SimConfig,
    rng: np.random.Generator,
) -> tuple[dict[int, float], GenerationStats]:
    """Runs `rounds_per_generation` rounds of random table re-seating across
    the whole population, accumulating each player's net chip result (the
    fitness signal the GA selects on) alongside generation-level sense-check
    metrics (mean hands survived, fold rate facing a bet, bust rate,
    raises/street)."""
    total_fitness = {p.player_id: 0.0 for p in players}
    gen_stats = GenerationStats()

    for _ in range(sim_config.rounds_per_generation):
        order = rng.permutation(len(players))
        shuffled = [players[i] for i in order]
        for start in range(0, len(shuffled), sim_config.table_size):
            table = shuffled[start : start + sim_config.table_size]
            if len(table) < 2:
                continue
            result = run_session(table, game_config, rng, backfill_pool=players, stats=gen_stats.hand_stats)
            for pid, delta in result["net"].items():
                total_fitness[pid] += delta
            for pid, hands in result["hands_survived"].items():
                gen_stats.total_hands_survived += hands
                gen_stats.total_session_participations += 1
                if result["busted"].get(pid):
                    gen_stats.total_busts += 1

    return total_fitness, gen_stats
