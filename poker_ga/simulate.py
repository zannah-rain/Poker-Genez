"""Orchestrates play: seats players at random 6-max tables and plays a
session of hands (refilling any seat that busts with a fresh player, rather
than ending the session early) until a hand cap is hit. Used by
benchmark.py's checkpoint matches."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from game import GameConfig, HandStats, SeatState, play_hand
from opponent_model import OpponentModel
from player import Player

TABLE_SIZE = 6


def _executor_scope(executor: Executor | None, num_workers: int):
    """Context manager yielding the Executor to submit table-match tasks to.
    If the caller already has a long-lived executor (cfr_main.py creates one
    once and reuses it for the whole training run, since spinning up worker
    processes has real, repeated-per-call overhead), it's reused as-is and
    left open when the `with` block exits. Otherwise a fresh
    ProcessPoolExecutor is created just for this call and torn down
    afterward -- convenient for standalone use (tests, one-off scripts)
    without requiring the caller to manage a pool."""
    if executor is not None:
        return nullcontext(executor)
    return ProcessPoolExecutor(max_workers=num_workers)


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
    # Fresh per session, not per hand or per Player -- opponent reads should
    # reflect this table's play so far this sitting, the way a live HUD
    # resets rather than a permanent cross-session dossier (see
    # opponent_model.py).
    opp_model = OpponentModel()

    net: dict[int, float] = {}
    hands_survived: dict[int, int] = {}
    busted: dict[int, bool] = {}

    button_idx = 0
    hands_played = 0
    while hands_played < game_config.max_hands_per_session:
        for s in seats:
            hands_survived[s.player.player_id] = hands_survived.get(s.player.player_id, 0) + 1

        play_hand(seats, button_idx % len(seats), game_config, rng, stats=stats, opp_model=opp_model)
        hands_played += 1
        button_idx = (button_idx + 1) % len(seats)

        # Snapshot who's still live *before* any replacement this hand, so a
        # refill can't pick a player_id already seated elsewhere at this
        # table -- otherwise the same identity could occupy two seats at
        # once, double-counting them in the hands_survived loop above on
        # every subsequent hand. Updated as replacements are assigned below
        # so two seats busting in the same hand can't refill with each
        # other's replacement either.
        occupied_ids = {s.player.player_id for s in seats if s.stack > 1e-9}
        for i, s in enumerate(seats):
            if s.stack <= 1e-9:
                pid = s.player.player_id
                net[pid] = net.get(pid, 0.0) - game_config.starting_stack
                busted[pid] = True
                # Falls back to the full pool (allowing a duplicate) only if
                # every pool member is already seated -- unavoidable when
                # the pool is no bigger than the table itself.
                candidates = [p for p in pool if p.player_id not in occupied_ids] or pool
                replacement = candidates[int(rng.integers(0, len(candidates)))]
                seats[i] = SeatState(player=replacement, stack=game_config.starting_stack)
                occupied_ids.add(replacement.player_id)

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

